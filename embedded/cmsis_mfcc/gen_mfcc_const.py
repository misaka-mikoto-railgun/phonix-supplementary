"""gen_mfcc_const.py — F405 MFCC latency용 상수 헤더 생성 + librosa parity 검증.

학습(librosa 0.11.0)과 동일 사양:
  sr=48000 n_fft=2048 win=1200(Hann,2048 zero-pad) hop=480 center=True(constant pad)
  n_mels=128 Slaney(htk=False,fmin=0,fmax=24000,norm='slaney') power=2 log=10log10(max(.,1e-10))
  DCT-II ortho 앞 8 (n_mfcc=8)
  feat10 = [MFCC0..7, log_RMS(frame=1200,hop=480,center), centroid_norm]
  ★ centroid 는 librosa 기본 win_length=n_fft=2048(full Hann) → MFCC와 별도 FFT, magnitude 가중.
32프레임만 계산: idx=linspace(0,400,32).astype(int).  top_db 무시(decision).
디바이스 테스트 클립은 정수 LCG로 host/C 비트동일 생성 → 헤더엔 x_ref(32x10)만 임베드.
"""
import numpy as np, librosa
from pathlib import Path

SR, NFFT, WIN, HOP, NMELS, NMFCC, SEQ = 48000, 2048, 1200, 480, 128, 8, 32
HERE = Path(__file__).resolve().parent

# ── 상수 ─────────────────────────────────────────────────────────────────────
hann1200 = librosa.filters.get_window('hann', WIN, fftbins=True).astype(np.float64)
padc = (NFFT - WIN) // 2
hann_mfcc = np.zeros(NFFT); hann_mfcc[padc:padc + WIN] = hann1200      # MFCC 창(2048, 가운데 1200)
hann_full = librosa.filters.get_window('hann', NFFT, fftbins=True).astype(np.float64)  # centroid 창
mel = librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=NMELS, fmin=0, fmax=24000,
                          htk=False, norm='slaney').astype(np.float64)  # (128,1025)
n = np.arange(NMELS); D = np.zeros((NMFCC, NMELS))
for k in range(NMFCC):
    D[k] = np.cos(np.pi * k * (2 * n + 1) / (2 * NMELS)) * (np.sqrt(1.0 / NMELS) if k == 0 else np.sqrt(2.0 / NMELS))
freqs = np.fft.rfftfreq(NFFT, 1.0 / SR)          # (1025,)
idx = np.linspace(0, 400, SEQ).astype(int)
NBIN = NFFT // 2 + 1

# mel sparse (contiguous range per mel)
mel_start, mel_count, mel_w = [], [], []
for m in range(NMELS):
    nz = np.nonzero(mel[m])[0]
    a, b = int(nz[0]), int(nz[-1])
    mel_start.append(a); mel_count.append(b - a + 1)
    mel_w.extend(mel[m, a:b + 1].tolist())
mel_w = np.array(mel_w); mel_off = np.concatenate([[0], np.cumsum(mel_count)[:-1]]).astype(int)

# ── LCG 테스트 클립 (host/C 비트동일) ───────────────────────────────────────
def lcg_clip(n_samples=192000, seed=12345):
    """정수 LCG (uint32) — C 와 비트동일. y in [-0.9, 0.9)."""
    y = np.empty(n_samples, np.float32); s = seed & 0xFFFFFFFF
    for i in range(n_samples):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        y[i] = np.float32((s >> 8) * (1.0 / 16777216.0) * 1.8 - 0.9)
    return y

# 32 프레임 세그먼트(2048) 추출 → flash 임베드. RMS 는 seg[424:1624] 로 유도(=pad600 창과 동일).
RMS_OFF = (NFFT - WIN) // 2   # 424
def extract_segs(y):
    p = np.pad(y, 1024, mode='constant')
    return np.stack([p[t * HOP:t * HOP + NFFT] for t in idx]).astype(np.float32)  # (32,2048)

# ── C-mirror = 디바이스가 할 연산 (오직 seg 로부터) ─────────────────────────
def cmirror_from_segs(segs):
    cols = []
    for f in range(SEQ):
        seg = segs[f].astype(np.float64)
        spec = np.fft.rfft(seg * hann_mfcc); power = spec.real**2 + spec.imag**2
        m8 = D @ (10 * np.log10(np.maximum(mel @ power, 1e-10)))
        segr = seg[RMS_OFF:RMS_OFF + WIN]; lrms = np.log(np.sqrt(np.mean(segr**2)) + 1e-8)
        mag = np.abs(np.fft.rfft(seg * hann_full)); cn = ((freqs * mag).sum() / (mag.sum() + 1e-12)) / 24000.0
        cols.append(np.concatenate([m8, [lrms, cn]]))
    F = np.stack(cols, 1); F = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-8)
    return F.T.astype(np.float32)

def librosa_ref(y):
    mf = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=NMFCC, n_fft=NFFT, hop_length=HOP, win_length=WIN)
    le = np.log(librosa.feature.rms(y=y, frame_length=WIN, hop_length=HOP) + 1e-8)
    c = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=NFFT, hop_length=HOP) / (SR / 2)
    f = np.concatenate([mf, le, c], 0)[:, idx]
    f = (f - f.mean(1, keepdims=True)) / (f.std(1, keepdims=True) + 1e-8)
    return f.T.astype(np.float32)

# (1) 알고리즘 ↔ librosa 동등성 증명: 실제 4s clip (저장 안 함, flash 절약)
y_real = lcg_clip(192000)
x_mir = cmirror_from_segs(extract_segs(y_real)); x_lib = librosa_ref(y_real)
err = np.abs(x_mir - x_lib)
print(f"[host parity] algo vs librosa (real 4s, z-scored x[32,10]): max abs err={err.max():.2e}")

# (2) 디바이스용 소형 테스트버퍼: 8192 샘플 → 32 overlapping frames (flash 32KB).
#     MFCC 연산량은 데이터 무관이라 latency 동일. float 저장 → 디바이스 parity 정확.
CLIP = lcg_clip(8192)
HOPS = (8192 - NFFT) // (SEQ - 1)            # 198
offs = [f * HOPS for f in range(SEQ)]
segs8 = np.stack([CLIP[o:o + NFFT] for o in offs]).astype(np.float32)
x_ref = cmirror_from_segs(segs8)             # 디바이스가 비교할 reference

# ── C 헤더 emit ──────────────────────────────────────────────────────────────
def _fval(v):
    s = "%.9g" % v
    if not any(c in s for c in ".eEnN"):   # 정수형(예: '0','1') → 유효 float 리터럴로
        s += ".0"
    return s + "f"

def carr(name, a, t="float"):
    a = np.asarray(a).ravel()
    body = ", ".join(("%d" % v) if t == "int" else _fval(v) for v in a)
    return f"const {t} {name}[{a.size}] = {{ {body} }};\n"

H = HERE / "mfcc_const.h"
with open(H, "w", encoding="utf-8") as f:
    f.write("// AUTO-GENERATED by gen_mfcc_const.py (librosa 0.11.0 training spec)\n")
    f.write("#ifndef MFCC_CONST_H\n#define MFCC_CONST_H\n#include <stdint.h>\n")
    f.write(f"#define SR {SR}\n#define NFFT {NFFT}\n#define WINLEN {WIN}\n#define HOP {HOP}\n")
    f.write(f"#define NMELS {NMELS}\n#define NMFCC {NMFCC}\n#define SEQ {SEQ}\n#define NBIN {NBIN}\n")
    f.write(f"#define NMELW {mel_w.size}\n#define NSAMP 192000\n#define PAD_MFCC 1024\n#define PAD_RMS {WIN//2}\n")
    f.write(carr("HANN_MFCC", hann_mfcc))           # 2048 (MFCC 창)
    f.write(carr("HANN_FULL", hann_full))           # 2048 (centroid 창)
    f.write(carr("MEL_W", mel_w))                    # sparse weights
    f.write(carr("MEL_START", mel_start, "int"))
    f.write(carr("MEL_COUNT", mel_count, "int"))
    f.write(carr("MEL_OFF", mel_off, "int"))
    f.write(carr("DCT", D))                          # 8x128 (row-major: k*128+n)
    f.write(carr("CFREQ", freqs))                    # 1025 (centroid 주파수)
    f.write(f"#define RMS_OFF {RMS_OFF}\n")
    f.write(f"#define NCLIP {CLIP.size}\n")
    f.write(carr("CLIP", CLIP))                      # 8192 측정 버퍼 (flash 32KB)
    f.write(carr("FRAME_OFF", offs, "int"))          # 32 프레임 시작 오프셋(CLIP 내)
    f.write(carr("X_REF", x_ref))                    # 32x10 reference
    f.write("#endif\n")
print(f"[emit] {H.name}: HANN x2, MEL_W({mel_w.size}), DCT(8x128), CFREQ(1025), CLIP({CLIP.size}), X_REF(32x10)")
print(f"       header float count ~= {hann_mfcc.size*2 + mel_w.size + D.size + freqs.size + x_ref.size} (~{(hann_mfcc.size*2+mel_w.size+D.size+freqs.size+x_ref.size)*4//1024} KB)")
