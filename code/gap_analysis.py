"""
gap_analysis.py — g12_f16k 의 synthetic-to-real gap 통계 점검
=============================================================
제안 ①: g12_f16k 3 seed 각각의 (real LSD - synth LSD) per-seed gap 을 계산하고,
baseline g6_f16k 의 single-seed gap(0.499)을 통계적으로 유의하게 초과하는지 확인.

드라이버 내장 evaluate(올바른 ±12 bound 로 인스턴스화) 재사용 — 평가 경로 일관.
"""
import numpy as np
import torch
from scipy import stats

from run_gain_freq_ablation import (
    CONFIGS, SEEDS_DEFAULT, DEFAULT_DATA, BASELINE_CKPT, HERE,
    build_model, cname, evaluate,
)
from dataset_generator_v4_tracklevel import PEQDataset
import train_full as TF

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
SAVE = HERE / "checkpoints"
CFG = "g12_f16k"
SEEDS = SEEDS_DEFAULT
BASELINE_GAP = 0.499  # 원본/재현 baseline (real-synth)

for s in SEEDS:
    TF._REGISTRY_TARGET[cname(CFG, s)] = "dual"
TF._REGISTRY_TARGET[cname("g6_f16k", 42)] = "dual"

ds_synth = PEQDataset(f"{DEFAULT_DATA}/test_synth", device=str(device))
ds_real  = PEQDataset(f"{DEFAULT_DATA}/test_real",  device=str(device))


def lsd_for(cfg, seed, ckpt, ds):
    m = build_model(cfg)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    m.load_state_dict(state, strict=False)
    return evaluate(cname(cfg, seed), cfg, m, ds, device)["lsd_mean"]


# ── baseline (single seed) ────────────────────────────────────────────────
b_synth = lsd_for("g6_f16k", 42, BASELINE_CKPT, ds_synth)
b_real  = lsd_for("g6_f16k", 42, BASELINE_CKPT, ds_real)
b_gap = b_real - b_synth

# ── g12_f16k per-seed ─────────────────────────────────────────────────────
rows = []
for s in SEEDS:
    cp = SAVE / f"{cname(CFG, s)}.pt"
    syn = lsd_for(CFG, s, cp, ds_synth)
    rea = lsd_for(CFG, s, cp, ds_real)
    rows.append((s, syn, rea, rea - syn))

gaps = np.array([r[3] for r in rows])
syns = np.array([r[1] for r in rows])
reas = np.array([r[2] for r in rows])

print("=" * 72)
print("synthetic-to-real gap 분해  (LSD, test_synth / test_real)")
print("=" * 72)
print(f"{'seed':>6} {'synth LSD':>10} {'real LSD':>10} {'gap(real-synth)':>16}")
print("-" * 46)
print(f"{'g6 base':>6} {b_synth:>10.4f} {b_real:>10.4f} {b_gap:>16.4f}")
for s, syn, rea, g in rows:
    print(f"{('g12 s'+str(s)):>6} {syn:>10.4f} {rea:>10.4f} {g:>16.4f}")
print("-" * 46)
print(f"g12_f16k gap : mean={gaps.mean():.4f}  std={gaps.std(ddof=1):.4f}  "
      f"(n={len(gaps)})  vs baseline gap={b_gap:.4f}")
print(f"g12 synth LSD: mean={syns.mean():.4f} std={syns.std(ddof=1):.4f}  "
      f"(baseline synth={b_synth:.4f}, Δ={syns.mean()-b_synth:+.4f})")
print(f"g12 real  LSD: mean={reas.mean():.4f} std={reas.std(ddof=1):.4f}  "
      f"(baseline real ={b_real:.4f}, Δ={reas.mean()-b_real:+.4f})")

# ── one-sample test: g12 gaps vs baseline 0.499 ───────────────────────────
t, p = stats.ttest_1samp(gaps, BASELINE_GAP)
# 95% CI of mean gap
ci = stats.t.interval(0.95, len(gaps)-1, loc=gaps.mean(),
                      scale=stats.sem(gaps)) if len(gaps) > 1 else (np.nan, np.nan)
print("-" * 46)
print(f"one-sample t-test (g12 gap vs {BASELINE_GAP}):  t={t:.3f}  p={p:.4f}  "
      f"(n={len(gaps)}, df={len(gaps)-1})")
print(f"g12 gap 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]  "
      f"→ {BASELINE_GAP} {'밖(유의)' if not (ci[0] <= BASELINE_GAP <= ci[1]) else '안(불유의)'}")
