"""
export_onnx.py — STM32F405 latency 벤치마크용 ONNX export (fairness 통일)
======================================================================
대상: A0, A2, E3/E4/E5, AC{1,2,3}_Biquad  (의사결정 2026-06-21 확정)

fairness 통일 조건
------------------
  * opset            : 13  (ST Edge AI Core 2.2.0 / X-CUBE-AI 10.2.0; fused
                        LayerNormalization is unsupported, so opset 13 keeps it decomposed)
  * batch            : 1, static shape (dynamic axes OFF)
  * TCN backbone in  : feat (1, 32, 10)  — seq_len=32(=4.0s block), in_dim=10
  * dtype            : float32 only (양자화 없음)

clamp trap 회피
---------------
  * default load_model (gain_max=6) 경로를 우회.
  * A0/A2  : DualObjectiveAdaptivePEQ(gain_max=12, fc_max=16000)  (driver CONFIGS g12_f16k)
  * AC*    : *_Biquad(gain_max=12.0)  (revision arch_biquad 패치본, self._gain_max)
  * export 전 gain>6 관측 게이트로 ±12 bound 가 실제로 살아있는지 양성 확인.

그래프 경계 (neural forward 만, post-proc 제외)
---------------------------------------------
  * A0/A2/AC : (room_corr[128], fc, gain, q)     — 7-band
        제외 post-proc: ①pref_curve(band_gains→128 보간, host에서 trivial)
                        ②DifferentiablePEQResponse 가우시안 재구성(학습 전용)
                        ③closed-form biquad 계수계산(C에서 측정 = 비교표 2번째 컬럼)
  * E3/E4    : (fc, gain, q)                      — 5-band parametric PEQ
        제외 post-proc: 가우시안 재구성 + 5-band biquad 계수계산
  * E5       : (fc, gain, q)  (= 내부 E3 room-corrector; E2 pref는 고정 테이블/비신경망)

searchsorted 회피
-----------------
  pref_curve 의 _interp 는 band_freqs/target_freqs 가 고정 buffer 이므로
  상수 선형사상(W: n_freqs×n_bands). aten::searchsorted(opset17 미지원)를
  수치적으로 동일한 matmul 로 치환 → 그래프에서 제거(parity 보존).
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE       = Path(__file__).resolve().parent          # .../revision_gain_freq/onnx_export
REV_ROOT   = HERE.parent                              # .../revision_gain_freq
ORIG_ROOT  = REV_ROOT.parent                          # .../data
sys.path.insert(0, str(REV_ROOT))                     # revision(패치본) 우선 import

from model import DualObjectiveAdaptivePEQ
from baselines import E3_NercessianMLP, E4_PepeCNN, E5_Sequential
from arch_biquad import AC1_BiLSTM_Biquad, AC2_GRU_Biquad, AC3_Conformer_Biquad
from dataset_generator_v4_tracklevel import PEQDataset

# opset 13: nn.LayerNorm 을 기본연산(ReduceMean/Sub/Pow/Sqrt/Div...)으로 분해.
# (opset>=17 은 fused 'LayerNormalization' 단일 op 생성 → 구버전 X-CUBE-AI 미지원.)
OPSET   = 13
# ORT BASIC 최적화 ON: AC2(GRU)/AC3(Conformer Mod) 가 이게 있어야 stedgeai analyze 통과.
# (검증: OFF 시 AC2/AC3 도 INTERNAL ERROR.)  AC1(BiLSTM)은 seq-first LSTM 으로 별도 회피.
ORT_OPTIMIZE = True
SEQ_LEN = 32
IN_DIM  = 10
N_ROOM  = 128
DATA    = ORIG_ROOT / "data" / "dataset_v3" / "test_synth"
OUT_DIR = HERE
device  = torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────
# searchsorted/ScatterND 제거: pref_curve 를 상수 선형사상(matmul)으로 벡터화
# ──────────────────────────────────────────────────────────────────────────
import types


def _build_W(model):
    """W[:,j] = orig_interp(band_freqs, e_j, target_freqs).  pref_curve = band_gains @ W.T.
    band_freqs/target_freqs 가 고정 buffer 이므로 _interp 는 상수 선형사상."""
    xp = model.band_freqs.detach()
    x  = model.target_freqs.detach()
    nb = xp.shape[0]
    orig = DualObjectiveAdaptivePEQ._interp
    cols = [orig(xp, torch.eye(nb)[j], x) for j in range(nb)]
    W = torch.stack(cols, dim=1).to(torch.float32)   # (n_freqs, n_bands)
    pou = float((W.sum(dim=1) - 1.0).abs().max())     # partition-of-unity 점검(≈0)
    return W, pou


def _a0_forward_vec(self, x, room_response, mode_id, band_gains):
    """A0/A2 forward 의 pref_curve 루프(ScatterND)를 matmul 로 치환한 export 전용 forward.
    그래프 경계(room_corr, fc, gain, q)만 산출 — 가우시안 peq_response 는 미계산(그래프 밖)."""
    room_vec = self.room_encoder(room_response)
    pref_vec = self.pref_encoder(mode_id, band_gains)
    x = self.input_proj(x)
    for tcn in self.tcn_blocks:
        x = tcn(x)
    ctx, _ = self.pool(x)
    room_input = room_vec
    room_mean  = torch.tanh(self.room_mean_head(room_input)) * self.room_mean_scale
    room_shape = self.room_shape_head(room_input) * self.room_shape_scale
    room_shape = room_shape - room_shape.mean(dim=-1, keepdim=True)
    room_corr  = room_mean + room_shape
    pref_curve = torch.matmul(band_gains, self._Winterp.t())   # 벡터화 (no loop/scatter)
    pref_input = torch.cat([ctx, room_vec, pref_vec, room_corr, pref_curve], dim=-1)
    peq_params = self.peq_head(pref_input)
    fc_raw, gain_raw, q_raw = peq_params.chunk(3, dim=-1)
    fc   = self.fc_min + torch.sigmoid(fc_raw) * (self.fc_max - self.fc_min)
    gain = torch.tanh(gain_raw) * self.gain_max
    q    = self.q_min + torch.sigmoid(q_raw) * (self.q_max - self.q_min)
    return {"room_correction_db": room_corr, "fc": fc, "gain": gain, "q": q}


def fold_pad_into_conv(path):
    """standalone (Pad→Conv) 를 Conv.pads 로 흡수.
    X-CUBE-AI 10.2.0 의 standalone Pad codegen 버그(_Pad_output_0_value_data[]={[]}) 회피.
    조건: Pad mode=constant, value==0, N/C 패딩 0(spatial L 만), Pad 출력 소비자 1개(해당 Conv)."""
    import onnx
    from onnx import numpy_helper
    import numpy as _np
    m = onnx.load(path); g = m.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    const = {n.output[0]: n for n in g.node if n.op_type == "Constant"}
    def gett(nm):
        if nm in inits: return inits[nm]
        if nm in const:
            for a in const[nm].attribute:
                if a.name == "value": return numpy_helper.to_array(a.t)
        return None
    from collections import Counter
    consumers = Counter(i for n in g.node for i in n.input)
    out2node = {o: n for n in g.node for o in n.output}
    folded = 0; to_remove = []
    for conv in g.node:
        if conv.op_type != "Conv": continue
        pad = out2node.get(conv.input[0])
        if pad is None or pad.op_type != "Pad": continue
        if consumers[pad.output[0]] != 1: continue
        pads_arr = gett(pad.input[1]) if len(pad.input) >= 2 else None
        if pads_arr is None: continue
        mode = next((a.s.decode() for a in pad.attribute if a.name == "mode"), "constant")
        cval = gett(pad.input[2]) if len(pad.input) >= 3 and pad.input[2] else None
        cval_v = float(_np.asarray(cval).reshape(-1)[0]) if cval is not None else 0.0
        if mode != "constant" or cval_v != 0.0: continue
        rank = len(pads_arr) // 2
        begins = [int(v) for v in pads_arr[:rank]]; ends = [int(v) for v in pads_arr[rank:]]
        # spatial(L=마지막 차원)만 패딩 허용 — N/C(앞쪽) 패딩은 0 이어야.
        if any(begins[:-1]) or any(ends[:-1]): continue
        Lb, Le = begins[-1], ends[-1]
        pa = next((a for a in conv.attribute if a.name == "pads"), None)
        if pa is None:
            conv.attribute.append(onnx.helper.make_attribute("pads", [Lb, Le]))
        else:
            cur = list(pa.ints); new = [cur[0] + Lb, cur[1] + Le]
            pa.ClearField("ints"); pa.ints.extend(new)
        conv.input[0] = pad.input[0]          # Conv 입력을 Pad 입력으로 재연결
        to_remove.append(pad); folded += 1
    for pad in to_remove:
        g.node.remove(pad)
    # 미사용 initializer / Constant 정리
    used = set(i for n in g.node for i in n.input) | {o.name for o in g.output}
    for init in [i for i in g.initializer if i.name not in used]:
        g.initializer.remove(init)
    for cn in [n for n in g.node if n.op_type == "Constant" and n.output[0] not in used]:
        g.node.remove(cn)
    onnx.checker.check_model(m); onnx.save(m, path)
    return folded


def _pref_fwd_onehot(self, mode_oh, band_gains):
    """UserPreferenceEncoder.forward 의 nn.Embedding(Gather)를 one-hot matmul 로 치환.
    mode_oh: (B, n_modes) float one-hot.  emb = mode_oh @ weight  (수치 동일)."""
    mode_emb = torch.matmul(mode_oh, self.mode_embedding.weight)   # (B,4)@(4,16)->(B,16)
    return self.fusion(torch.cat([mode_emb, self.band_proj(band_gains)], dim=-1))


def install_gru_split(model):
    """2-layer batch_first GRU → 1-layer GRU 2개 직렬(가중치 복사).
    ST Edge AI Core 의 stacked-GRU transpose 식별 실패 회피. hn[-1] == 마지막층 마지막스텝 출력."""
    g = model.gru
    H, I = g.hidden_size, g.input_size
    a = nn.GRU(I, H, 1, batch_first=True).eval()
    b = nn.GRU(H, H, 1, batch_first=True).eval()
    with torch.no_grad():
        a.weight_ih_l0.copy_(g.weight_ih_l0); a.weight_hh_l0.copy_(g.weight_hh_l0)
        a.bias_ih_l0.copy_(g.bias_ih_l0);     a.bias_hh_l0.copy_(g.bias_hh_l0)
        b.weight_ih_l0.copy_(g.weight_ih_l1); b.weight_hh_l0.copy_(g.weight_hh_l1)
        b.bias_ih_l0.copy_(g.bias_ih_l1);     b.bias_hh_l0.copy_(g.bias_hh_l1)
    model.gru_a, model.gru_b = a, b
    orig = model.aggregate
    def agg(self, h):
        oa, _ = self.gru_a(h); ob, _ = self.gru_b(oa); return ob[:, -1, :]
    with torch.no_grad():
        h = torch.randn(1, SEQ_LEN, model.hidden_dim)
        err = float((orig(h) - agg(model, h)).abs().max())
    model.aggregate = types.MethodType(agg, model)
    return err


def install_bilstm_split(model):
    """1-layer bidirectional LSTM → 단방향 LSTM 2개(forward + 시간역전 backward, 가중치 복사).
    batch_first=False(seq-first) 로 구성해 torch 가 LSTM 앞뒤에 넣는 batch_first transpose
    자체를 생성하지 않게 함(ST Edge AI Core 가 in-network LSTM transpose 를 remap 못 하는
    문제 회피). time-mean 은 순서 무관 → backward 출력 재역전 불필요."""
    l = model.bilstm
    I, H = l.input_size, l.hidden_size
    f = nn.LSTM(I, H, 1, batch_first=False).eval()
    b = nn.LSTM(I, H, 1, batch_first=False).eval()
    with torch.no_grad():
        f.weight_ih_l0.copy_(l.weight_ih_l0); f.weight_hh_l0.copy_(l.weight_hh_l0)
        f.bias_ih_l0.copy_(l.bias_ih_l0);     f.bias_hh_l0.copy_(l.bias_hh_l0)
        b.weight_ih_l0.copy_(l.weight_ih_l0_reverse); b.weight_hh_l0.copy_(l.weight_hh_l0_reverse)
        b.bias_ih_l0.copy_(l.bias_ih_l0_reverse);     b.bias_hh_l0.copy_(l.bias_hh_l0_reverse)
    model.lstm_f, model.lstm_b = f, b
    orig = model.aggregate
    def agg(self, h):
        # 각 LSTM 이 '자신만의' Transpose 를 바로 앞에 두도록 분리(공유 시 ST 의
        # transpose↔recurrent 1:1 식별 실패). backward 는 flip 후 transpose.
        hsf = h.transpose(0, 1)                          # (B,T,C)→(T,B,C)  → lstm_f
        hsb = torch.flip(h, [1]).transpose(0, 1)         # 시간축 역전 후 seq-first → lstm_b
        of, _ = self.lstm_f(hsf)                         # (T,B,H)
        ob, _ = self.lstm_b(hsb)                         # (T,B,H)
        return torch.cat([of.mean(0), ob.mean(0)], dim=-1)   # time-mean → (B,2H)
    with torch.no_grad():
        h = torch.randn(1, SEQ_LEN, model.hidden_dim)
        err = float((orig(h) - agg(model, h)).abs().max())
    model.aggregate = types.MethodType(agg, model)
    return err


class _UnrolledLSTM(nn.Module):
    """단방향 1-layer LSTM 을 정적 시간스텝(T) 으로 언롤 → Gemm/Sigmoid/Tanh/Mul/Add 만 사용.
    ONNX LSTM op·transpose 없음 → ST Edge AI Core 호환. 입력 (B,T,I), 출력 = time-mean h_t (B,H).
    PyTorch gate 순서 [i,f,g,o], bias = b_ih + b_hh (한 번 합산)."""
    def __init__(self, W_ih, W_hh, b_ih, b_hh, T):
        super().__init__()
        # 입력/은닉 가중치를 합쳐 한 번의 Gemm 으로: gates = [x_t, h] @ [W_ih|W_hh]^T + bias.
        # 유일한 Add 가 상수 1D bias → ST 'multi-dim Gemm bias' 미지원 회피(가변 Add 없음).
        W_comb = torch.cat([W_ih, W_hh], dim=1)                 # (4H, I+H)
        self.register_buffer("W_comb_t", W_comb.t().contiguous())  # (I+H, 4H)
        self.register_buffer("bias", (b_ih + b_hh).contiguous())   # (4H,)
        self.H = W_hh.shape[1]; self.I = W_ih.shape[1]; self.T = T

    def forward(self, x):                       # x: (B,T,I)  (export 시 B=1 static)
        B = x.shape[0]
        h = x.new_zeros(B, self.H); c = x.new_zeros(B, self.H)
        acc = x.new_zeros(B, self.H)
        for t in range(self.T):
            z = torch.cat([x[:, t, :], h], dim=-1)              # (1, I+H)
            g = torch.addmm(self.bias, z, self.W_comb_t)        # (1,4H) Gemm(C=상수bias)
            i, f, gg, o = g.chunk(4, dim=-1)
            i = torch.sigmoid(i); f = torch.sigmoid(f)
            gg = torch.tanh(gg);  o = torch.sigmoid(o)
            c = f * c + i * gg
            h = o * torch.tanh(c)
            acc = acc + h
        return acc / self.T                     # time-mean (B,H)


def install_bilstm_unroll(model):
    """BiLSTM → forward/backward 정적 언롤 2개. bidirectional weight_*_l0(forward) /
    weight_*_l0_reverse(backward) 복사. backward 는 시간축 flip 입력(time-mean 은 순서무관).
    ST Edge AI Core 가 TCN→LSTM transpose 를 remap 못 하는 문제를 LSTM op 제거로 우회."""
    l = model.bilstm
    seq = SEQ_LEN
    uf = _UnrolledLSTM(l.weight_ih_l0, l.weight_hh_l0, l.bias_ih_l0, l.bias_hh_l0, seq).eval()
    ub = _UnrolledLSTM(l.weight_ih_l0_reverse, l.weight_hh_l0_reverse,
                       l.bias_ih_l0_reverse, l.bias_hh_l0_reverse, seq).eval()
    model.ulstm_f, model.ulstm_b = uf, ub
    orig = model.aggregate
    def agg(self, h):
        mf = self.ulstm_f(h)                       # (B,H) forward time-mean
        mb = self.ulstm_b(torch.flip(h, [1]))      # (B,H) backward time-mean
        return torch.cat([mf, mb], dim=-1)         # (B,2H) == out.mean(1)
    with torch.no_grad():
        hh = torch.randn(1, SEQ_LEN, model.hidden_dim)
        err = float((orig(hh) - agg(model, hh)).abs().max())
    model.aggregate = types.MethodType(agg, model)
    return err


def install_onehot_mode(model):
    """mode_id(int64) embedding Gather 제거 → mode_onehot(float) matmul.
    (ST Edge AI Core 가 embedding 테이블 leading dim(4)을 batch 로 오인하는 문제 회피.)"""
    model.pref_encoder.forward = types.MethodType(_pref_fwd_onehot, model.pref_encoder)


def install_interp_matmul(model, is_a0):
    """pref_curve 의 searchsorted/ScatterND 제거.
      A0/A2 : forward 자체를 벡터화 버전으로 교체(루프 inline).
      AC     : _pref_curve(메서드) 만 matmul 로 교체."""
    W, pou = _build_W(model)
    model._Winterp = W
    if is_a0:
        model.forward = types.MethodType(_a0_forward_vec, model)
    else:
        model._pref_curve = lambda bg: torch.matmul(bg, W.t())
    return pou


# ──────────────────────────────────────────────────────────────────────────
# export wrapper (그래프 경계 = native neural output)
# ──────────────────────────────────────────────────────────────────────────
class Wrap7(torch.nn.Module):
    """A0/A2/AC: (room_corr, fc, gain, q)"""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, feat, room, mode, band):
        o = self.m(feat, room, mode, band)
        return o["room_correction_db"], o["fc"], o["gain"], o["q"]


class WrapE(torch.nn.Module):
    """E3/E4: (fc, gain, q), 입력 feat 만"""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, feat):
        o = self.m(feat)
        return o["fc"], o["gain"], o["q"]


class WrapE5(torch.nn.Module):
    """E5: 내부 E3 room-corrector 의 (fc, gain, q)"""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, feat):
        o = self.m.room_corrector(feat)
        return o["fc"], o["gain"], o["q"]


# ──────────────────────────────────────────────────────────────────────────
# 체크포인트 로드 (load_model 우회 — 명시적 인스턴스 + state_dict)
# ──────────────────────────────────────────────────────────────────────────
def load_ckpt(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    return len(missing), len(unexpected)


# ──────────────────────────────────────────────────────────────────────────
# export 사양
# ──────────────────────────────────────────────────────────────────────────
SPECS = [
    # name, onnx, group, builder, ckpt, seed, gain_max, fc_max, conditional, wrap
    ("A0_Proposed", "a0.onnx", "A0",
     lambda: DualObjectiveAdaptivePEQ(gain_max=12.0, fc_max=16000.0),
     REV_ROOT / "checkpoints" / "A0_g12_f16k_s7.pt", 7, 12.0, 16000.0, True, "w7"),
    ("A2_withPrefLoss", "a2.onnx", "A2",
     lambda: DualObjectiveAdaptivePEQ(gain_max=12.0, fc_max=16000.0),
     REV_ROOT / "checkpoints" / "A2_g12_f16k_s7.pt", 7, 12.0, 16000.0, True, "w7"),
    ("E3_Nercessian", "e3_nercessian.onnx", "E",
     lambda: E3_NercessianMLP(),
     ORIG_ROOT / "checkpoints" / "full" / "E3_Nercessian.pt", 42, 12.0, 24000.0, False, "we"),
    ("E4_Pepe", "e4_pepe.onnx", "E",
     lambda: E4_PepeCNN(),
     ORIG_ROOT / "checkpoints" / "full" / "E4_Pepe.pt", 42, 12.0, 24000.0, False, "we"),
    ("E5_Sequential", "e5_sequential.onnx", "E",
     lambda: E5_Sequential(),
     ORIG_ROOT / "checkpoints" / "full" / "E5_Sequential.pt", 42, 12.0, 24000.0, False, "we5"),
    ("AC1_BiLSTM_Biquad", "ac1_bilstm_biquad.onnx", "AC",
     lambda: AC1_BiLSTM_Biquad(gain_max=12.0),
     REV_ROOT / "checkpoints" / "AC1_BiLSTM_Biquad_g12.pt", None, 12.0, 16000.0, True, "w7"),
    ("AC2_GRU_Biquad", "ac2_gru_biquad.onnx", "AC",
     lambda: AC2_GRU_Biquad(gain_max=12.0),
     REV_ROOT / "checkpoints" / "AC2_GRU_Biquad_g12.pt", None, 12.0, 16000.0, True, "w7"),
    ("AC3_Conformer_Biquad", "ac3_conformer_biquad.onnx", "AC",
     lambda: AC3_Conformer_Biquad(gain_max=12.0),
     REV_ROOT / "checkpoints" / "AC3_Conformer_Biquad_g12.pt", None, 12.0, 16000.0, True, "w7"),
]

OUT_FORMAT = {
    "w7":  "room_corr[1,128] (dense room-correction dB) + fc/gain/q [1,7] (7-band biquad params)",
    "we":  "fc/gain/q [1,5] (5-band parametric PEQ params)",
    "we5": "fc/gain/q [1,5] (내부 E3 room-corrector; E2 pref는 고정 테이블/비신경망)",
}
POSTPROC = {
    "w7":  "host/C: ①pref_curve = band_gains→128 선형보간, ②7-band closed-form biquad(RBJ) 계수계산 후 room_corr(dense)와 합성. 가우시안 재구성은 학습전용이라 미사용.",
    "we":  "host/C: 5-band closed-form biquad 계수계산. 가우시안 재구성은 학습전용.",
    "we5": "host/C: 5-band closed-form biquad(E3) + 고정 모드 프로파일(E2) 합산.",
}
BOUNDARY = {
    "w7":  "input → {room_corr, fc, gain, q}.  peq_response(가우시안)·biquad 계수계산·pref_curve 합성은 그래프 밖.",
    "we":  "input → {fc, gain, q}.  response(가우시안)·biquad 계수계산은 그래프 밖.",
    "we5": "input → {fc, gain, q}(E3).  E2 테이블 합산은 그래프 밖.",
}


def main():
    print("=" * 70)
    print(f"ONNX EXPORT  opset={OPSET}  batch=1  static  fp32")
    print("=" * 70)

    # 실제 입력(±12 게이트·parity 용) — 고정 batch=1 로 슬라이스
    ds = PEQDataset(str(DATA), device="cpu")
    full = ds.get_all()
    n_modes = 4
    oh = lambda idx: torch.nn.functional.one_hot(idx.long(), n_modes).float()
    real = {
        "feat": full["features"][:1].float(),
        "room": full["room_response"][:1].float(),
        "mode": oh(full["mode_id"][:1]),       # one-hot (1, 4) float — Gather 제거
        "band": full["band_gains"][:1].float(),
    }
    # gain 관측은 더 큰 표본에서 (±6 binding 입증)
    big = {
        "feat": full["features"][:512].float(),
        "room": full["room_response"][:512].float(),
        "mode": oh(full["mode_id"][:512]),
        "band": full["band_gains"][:512].float(),
    }

    import onnx, onnxruntime as ort
    try:
        from thop import profile as thop_profile
    except Exception:
        thop_profile = None

    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "fairness": {
            "opset": OPSET, "batch": 1, "dynamic_axes": False, "dtype": "float32",
            "backbone_input_shape": [1, SEQ_LEN, IN_DIM],
            "conditional_inputs": {"room_response": [1, N_ROOM],
                                   "mode_onehot": [1, 4], "band_gains": [1, 10]},
            "interp_replaced_with_matmul": True,
            "mode_embedding_replaced_with_onehot_matmul": True,
            "host_preprocessing": "mode_onehot = one_hot(mode_id, 4) — host/C에서 trivial 인코딩(그래프 밖)",
        },
        "variants": [],
    }

    all_ok = True
    for (name, onnx_name, group, build, ckpt, seed, gmax, fmax, cond, wrap) in SPECS:
        print(f"\n── {name}  ({group})  ───────────────────────")
        if not ckpt.exists():
            print(f"  !! ckpt 없음: {ckpt}")
            all_ok = False
            continue
        model = build().eval()
        nm, nu = load_ckpt(model, ckpt)
        print(f"  ckpt: {ckpt.name}  (missing={nm}, unexpected={nu})")

        interp_err = None
        if cond:
            install_onehot_mode(model)           # embedding Gather 제거
            interp_err = install_interp_matmul(model, is_a0=group in ("A0", "A2"))
            print(f"  interp→matmul partition-of-unity err: {interp_err:.2e}")
            if name == "AC2_GRU_Biquad":
                e = install_gru_split(model)
                print(f"  GRU 2-layer→1-layer×2 split, self-check err={e:.2e}")
            elif name == "AC1_BiLSTM_Biquad":
                # 표준 uni-LSTM 분해본은 ST 2.2.0 임포트 불가(TCN→LSTM transpose remap 실패).
                # → LSTM 재귀를 정적 언롤(Gemm/Sigmoid/Tanh)하여 LSTM op 자체 제거.
                e = install_bilstm_unroll(model)
                print(f"  BiLSTM→static-unroll(Gemm), self-check err={e:.2e}")

        # ── wrapper & 입력 ──
        if wrap == "w7":
            wmod = Wrap7(model); args = (real["feat"], real["room"], real["mode"], real["band"])
            in_names = ["feat", "room", "mode_onehot", "band"]
            out_names = ["room_corr", "fc", "gain", "q"]
        elif wrap == "we":
            wmod = WrapE(model); args = (real["feat"],)
            in_names = ["feat"]; out_names = ["fc", "gain", "q"]
        else:  # we5
            wmod = WrapE5(model); args = (real["feat"],)
            in_names = ["feat"]; out_names = ["fc", "gain", "q"]
        wmod.eval()

        # ── gain>6 게이트 (±12 모델만) ──
        gain_max_obs = None; over6 = None
        with torch.no_grad():
            if cond:
                ob = model(big["feat"], big["room"], big["mode"], big["band"])
            elif wrap == "we5":
                ob = model.room_corrector(big["feat"])
            else:
                ob = model(big["feat"])
            g = ob["gain"]
            gain_max_obs = float(g.abs().max())
            over6 = float((g.abs() > 6.0).float().mean())
        # 게이트는 clamp-trap 위험이 있는 gain_max-주입 모델(A0/A2/AC)에만 적용.
        # E 변형은 baselines.py 에 ±12 구조적 하드코딩 → trap 해당없음(n/a).
        if cond:
            gate = "PASS" if gain_max_obs > 6.0 else "FAIL"
        else:
            gate = "n/a (구조적 ±12, trap 없음)"
        print(f"  gain: max|g|={gain_max_obs:.3f}  |g|>6={over6*100:.1f}%  (bound±{gmax:.0f}) gate={gate}")
        if gate == "FAIL":
            all_ok = False

        # ── export ──
        out_path = OUT_DIR / onnx_name
        torch.onnx.export(
            wmod, args, str(out_path),
            input_names=in_names, output_names=out_names,
            opset_version=OPSET, dynamic_axes=None, do_constant_folding=True,
        )
        # ── ORT BASIC graph 최적화(상수폴딩) → 최종 산출물.
        #    표준 ai.onnx 도메인만 사용(com.microsoft 도입 안 함) → Cube.AI 호환 유지.
        #    단, RNN(LSTM/GRU) 모델은 ORT 가 LSTM 주변 Transpose 를 재배치해
        #    ST Edge AI Core 의 transpose remapping 을 깨뜨림 → 최적화 건너뜀.
        #    (stedgeai 자체 onnx optimizer 가 상수/clutter 를 처리.)
        n_before = len(onnx.load(str(out_path)).graph.node)
        do_opt = ORT_OPTIMIZE
        if do_opt:
            tmp_opt = OUT_DIR / (onnx_name + ".opt.tmp")
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            so.optimized_model_filepath = str(tmp_opt)
            _ = ort.InferenceSession(
                str(out_path), so, providers=["CPUExecutionProvider"],
                # MatMulAddFusion 비활성: 언롤 LSTM 에서 matmul+가변텐서 Add 를 Gemm(C=가변)
                # 으로 융합 → ST 'multi-dim Gemm bias' 미지원. nn.Linear 은 이미 Gemm 이라 무영향.
                disabled_optimizers=["LayerNormFusion", "SimplifiedLayerNormFusion",
                                     "SkipLayerNormFusion", "MatMulAddFusion"],
            )
            import os as _os
            _os.replace(str(tmp_opt), str(out_path))
        # standalone (Pad→Conv) → Conv.pads 흡수 (X-CUBE-AI 10.2.0 Pad codegen 버그 회피)
        n_folded = fold_pad_into_conv(str(out_path))
        mod = onnx.load(str(out_path)); onnx.checker.check_model(mod)
        ops = sorted({n.op_type for n in mod.graph.node})
        n_after = len(mod.graph.node)
        domains = sorted({(n.domain or "ai.onnx") for n in mod.graph.node})
        assert "LayerNormalization" not in ops, f"{name}: LayerNormalization 재출현!"
        # conv 앞 standalone Pad 가 남아있지 않은지 확인
        o2n = {o: n for n in mod.graph.node for o in n.output}
        pad_before_conv = [n.name for n in mod.graph.node if n.op_type == "Conv"
                           and (o2n.get(n.input[0]) is not None)
                           and o2n[n.input[0]].op_type == "Pad"]
        assert not pad_before_conv, f"{name}: Conv 앞 Pad 잔존 {pad_before_conv}"
        print(f"  optimize(BASIC={do_opt}): nodes {n_before}->{n_after}  Pad→Conv folded={n_folded}  domains={domains}")

        # 하드 블로커(그래프 임포트 실패 유발 가능): 즉시 제거 대상.
        HARD = {"NonZero", "ScatterND", "ScatterElements", "GatherND",
                "Loop", "If", "TopK", "NonMaxSuppression", "Range", "Mod",
                "LayerNormalization", "GroupNormalization"}
        # 버전 의존(구버전 X-CUBE-AI 에서 막힐 수 있어 확인 필요): 제거는 안 함.
        VERIFY = {"Erf", "Gelu", "Pad", "Softmax", "ReduceSum",
                  "InstanceNormalization", "Einsum", "GRU", "LSTM"}
        flagged = sorted(set(ops) & HARD)            # 0 이어야 정상
        verify_ops = sorted(set(ops) & VERIFY)       # X-CUBE-AI 버전별 확인 권장

        # ── parity (eager vs onnxruntime) ──
        with torch.no_grad():
            ref = wmod(*args)
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        feeds = {in_names[i]: args[i].cpu().numpy() for i in range(len(in_names))}
        orto = sess.run(out_names, feeds)
        max_abs = max(float(np.abs(r.cpu().numpy() - o).max()) for r, o in zip(ref, orto))
        print(f"  parity max|abs err|={max_abs:.2e}")
        print(f"  ops={ops}")
        if flagged:
            print(f"  !! HARD blocker op(제거 필요): {flagged}")
        if verify_ops:
            print(f"  ~  VERIFY op(X-CUBE-AI 버전 확인): {verify_ops}")

        # ── MACC (thop, 근사) ──
        macc = None
        if thop_profile is not None:
            try:
                macc, _ = thop_profile(wmod, inputs=args, verbose=False)
                macc = int(macc)
            except Exception as e:
                macc = f"thop 실패: {e}"

        n_params = int(sum(p.numel() for p in model.parameters()))
        out_shapes = {out_names[i]: list(orto[i].shape) for i in range(len(out_names))}

        manifest["variants"].append({
            "name": name, "group": group, "onnx": onnx_name,
            "checkpoint": str(ckpt.relative_to(ORIG_ROOT)),
            "seed": seed, "gain_max": gmax, "fc_max": fmax,
            "n_params": n_params,
            "input_shape": {in_names[i]: list(args[i].shape) for i in range(len(in_names))},
            "output_shape": out_shapes,
            "output_format": OUT_FORMAT[wrap],
            "graph_boundary": BOUNDARY[wrap],
            "excluded_postprocessing": POSTPROC[wrap],
            "parity_max_abs_err": max_abs,
            "gain_max_observed": gain_max_obs,
            "gain_over6_frac": over6,
            "gain_gate": gate,
            "interp_matmul_pou_err": interp_err,
            "onnx_ops": ops,
            "onnx_domains": domains,
            "onnx_node_count": n_after,
            "cubeai_hard_blocker_ops": flagged,
            "cubeai_verify_ops": verify_ops,
            "macc_thop": macc,
        })

    # ── manifest 저장 ──
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest_md(manifest)
    print("\n" + "=" * 70)
    print(f"DONE — manifest.json / manifest.md 작성.  전체 게이트: {'ALL OK' if all_ok else '주의(위 로그 확인)'}")
    print("=" * 70)
    return 0 if all_ok else 1


def write_manifest_md(man):
    L = []
    L.append("# ONNX Export Manifest — STM32F405 latency benchmark\n")
    L.append(f"- generated: {man['generated']}")
    f = man["fairness"]
    L.append(f"- opset: **{f['opset']}**, batch=1, dynamic_axes={f['dynamic_axes']}, dtype={f['dtype']}")
    L.append(f"- backbone input: `feat {f['backbone_input_shape']}` (seq_len=32=4.0s, in_dim=10)")
    L.append(f"- conditional inputs: room_response[1,128], **mode_onehot[1,4] float**, band_gains[1,10]")
    L.append(f"- pref_curve `_interp` → 상수 matmul 치환(searchsorted 제거, parity 보존)")
    L.append(f"- mode embedding(nn.Embedding Gather) → one-hot matmul 치환"
             f"(ST Edge AI Core 의 embedding-table batch 오인 회피; host에서 one-hot 인코딩)\n")
    L.append(f"- standalone Pad → Conv.pads 흡수(fold_pad_into_conv): X-CUBE-AI 10.2.0 의 "
             f"Pad codegen 버그 회피. causal pads=[(k-1)*dilation, 0]")
    L.append("| variant | group | onnx | ckpt | seed | gain_max | params | parity max|err| | max\\|gain\\| | gate | MACC(thop) | HARD blocker | VERIFY ops |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for v in man["variants"]:
        L.append(f"| {v['name']} | {v['group']} | {v['onnx']} | {Path(v['checkpoint']).name} | "
                 f"{v['seed']} | {v['gain_max']} | {v['n_params']:,} | {v['parity_max_abs_err']:.2e} | "
                 f"{v['gain_max_observed']:.2f} | {v['gain_gate']} | {v['macc_thop']} | "
                 f"{v['cubeai_hard_blocker_ops'] or '-'} | {v['cubeai_verify_ops'] or '-'} |")
    L.append("\n## 그래프 경계 & 제외 post-processing\n")
    for v in man["variants"]:
        L.append(f"### {v['name']} (`{v['onnx']}`)")
        L.append(f"- output format: {v['output_format']}")
        L.append(f"- output shapes: {v['output_shape']}")
        L.append(f"- graph boundary: {v['graph_boundary']}")
        L.append(f"- excluded post-proc (C에서 측정 = 비교표 2nd 컬럼): {v['excluded_postprocessing']}")
        L.append(f"- onnx ops: {v['onnx_ops']}\n")
    (Path(man.get("_md_dir", OUT_DIR)) / "manifest.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
