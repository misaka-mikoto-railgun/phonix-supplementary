"""train set 분포 스캔 (PTQ calibration 설계용) — 추출 전 보고."""
import sys, glob, json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REV  = HERE.parent
ORIG = REV.parent
sys.path.insert(0, str(REV))
from model import DualObjectiveAdaptivePEQ

TRAIN = ORIG / "data" / "dataset_v3" / "train"
CKPT  = REV / "checkpoints" / "A0_g12_f16k_s7.pt"

chunks = sorted(glob.glob(str(TRAIN / "chunk_*.npz")))
acc = {k: [] for k in ["features", "room_response", "mode_id", "band_gains", "room_id", "track_id"]}
for c in chunks:
    d = np.load(c, allow_pickle=False)
    for k in acc:
        acc[k].append(d[k])
D = {k: np.concatenate(v) for k, v in acc.items()}
N = len(D["mode_id"])
print(f"N train = {N}\n")

# ── (3) mode ─────────────────────────────────────────────
print("== (3) MODE ==")
mids, mcnt = np.unique(D["mode_id"], return_counts=True)
for m, ct in zip(mids, mcnt):
    print(f"  mode {int(m)}: {ct}  ({ct/N*100:.1f}%)")

# ── (1) room ─────────────────────────────────────────────
print("\n== (1) ROOM (room_id) ==")
rids, rcnt = np.unique(D["room_id"], return_counts=True)
print(f"  distinct room_id: {len(rids)}   (range {rids.min()}..{rids.max()})")
print(f"  samples/room: min={rcnt.min()} max={rcnt.max()} mean={rcnt.mean():.1f} median={np.median(rcnt):.0f}")
print(f"  distinct track_id: {len(np.unique(D['track_id']))}")
# room_response 자체 변동(같은 room_id 내 mic jitter)
rr = D["room_response"]
print(f"  room_response[128] value range: min={rr.min():.2f} max={rr.max():.2f} "
      f"p1={np.percentile(rr,1):.2f} p99={np.percentile(rr,99):.2f}")

# ── (4) backbone feature x[32,10] ───────────────────────
print("\n== (4) FEATURE x[32,10] ==")
X = D["features"]
print(f"  global: min={X.min():.3f} max={X.max():.3f} mean={X.mean():.3f} std={X.std():.3f}")
print(f"  pct: p0.1={np.percentile(X,0.1):.3f} p1={np.percentile(X,1):.3f} "
      f"p50={np.percentile(X,50):.3f} p99={np.percentile(X,99):.3f} p99.9={np.percentile(X,99.9):.3f}")
print("  per-feature(10) min/max:")
for j in range(X.shape[2]):
    xj = X[:, :, j]
    print(f"    f{j}: [{xj.min():7.3f}, {xj.max():7.3f}]  p1={np.percentile(xj,1):7.3f} p99={np.percentile(xj,99):7.3f}")

# ── (2) gain: input band_gains + learned output gain ────
print("\n== (2) GAIN ==")
bg = D["band_gains"]
print(f"  [input band_gains[10]] min={bg.min():.2f} max={bg.max():.2f} "
      f"p1={np.percentile(bg,1):.2f} p99={np.percentile(bg,99):.2f}")
for thr in (6, 9, 11):
    frac = (np.abs(bg) > thr).mean()
    nsamp = (np.abs(bg).max(axis=1) > thr).sum()
    print(f"    |band_gain|>{thr}: {frac*100:.2f}% of values, {nsamp} samples have >=1 band over")

# 학습된 출력 gain (A0 gain_max=12, s7)
print("  [learned output gain (A0 s7, gain_max=12)] running model...")
model = DualObjectiveAdaptivePEQ(gain_max=12.0, fc_max=16000.0).eval()
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
st = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
model.load_state_dict(st, strict=False)
outg = []
with torch.no_grad():
    for s in range(0, N, 1024):
        xb = torch.from_numpy(X[s:s+1024]).float()
        rb = torch.from_numpy(rr[s:s+1024]).float()
        mb = torch.from_numpy(D["mode_id"][s:s+1024]).long()
        gb = torch.from_numpy(bg[s:s+1024]).float()
        o = model(xb, rb, mb, gb)
        outg.append(o["gain"].numpy())
OG = np.concatenate(outg)   # (N,7)
print(f"    output gain[7] min={OG.min():.2f} max={OG.max():.2f} "
      f"p1={np.percentile(OG,1):.2f} p50={np.percentile(OG,50):.2f} p99={np.percentile(OG,99):.2f}")
for thr in (6, 9, 11):
    frac = (np.abs(OG) > thr).mean()
    nsamp = (np.abs(OG).max(axis=1) > thr).sum()
    print(f"    |out gain|>{thr}: {frac*100:.2f}% of values, {nsamp} samples ({nsamp/N*100:.1f}%) have >=1 filter over")
# 히스토그램 (output gain)
h, edges = np.histogram(OG.ravel(), bins=[-12,-11,-9,-6,-3,0,3,6,9,11,12])
print("    out-gain hist:", {f"[{edges[i]:.0f},{edges[i+1]:.0f})": int(h[i]) for i in range(len(h))})

# 저장 (추출 단계 재사용)
np.savez(HERE / "_scan_cache.npz",
         mode_id=D["mode_id"], room_id=D["room_id"], band_gains=bg,
         out_gain_absmax=np.abs(OG).max(axis=1))
print("\n[cache] _scan_cache.npz saved (mode_id, room_id, band_gains, out_gain_absmax)")
