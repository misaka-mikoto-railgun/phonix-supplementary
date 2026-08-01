"""
paired_stats.py — Table stats(paired) 재계산, 3-seed 집계 (결정 c)
==================================================================
각 seed s∈{42,123,7} 에서 A0_s vs 비교대상 per-sample paired test:
  - A0 vs A2  : 같은 seed 끼리 (A0_s vs A2_s)  ← 둘 다 3-seed
  - A0 vs {A1,A3,E3,E4,AC1,AC2,AC3} : 비교대상은 원본 single-seed(42) 고정
3-seed effect size(Δ, d_z, Win%)를 mean±std 로 집계.

부호 규약(원본 tab:stats 동일):
  LSD: Δ = mean(A0_lsd - base_lsd)  (음수 = A0 LSD 낮음 = A0 better), Win% = P(A0_lsd < base_lsd)
  DMR: Δ = mean(A0_dmr - base_dmr)  (양수 = A0 better),               Win% = P(A0_dmr > base_dmr)
  d_z = mean(diff)/std(diff)  (paired Cohen's d_z)

A0/A2 는 gain±12(패치된 build_registry), 비교대상은 원본 ckpt(read-only). test_synth.
"""
import json
from pathlib import Path

import numpy as np
import torch

import train_full as TF
from run_gain_freq_ablation import evaluate, cname
import cli_paths
import ckpt_io
from dataset_generator_v4_tracklevel import PEQDataset

_P, _ = cli_paths.parse("paired statistics, 3-seed", require=("data_dir", "ckpt_dir", "rev_ckpt_dir"))
DEFAULT_DATA = _P.data_dir      # --data_dir
FULL = _P.ckpt_dir              # --ckpt_dir       pre-revision checkpoints
SAVE = _P.rev_ckpt_dir          # --rev_ckpt_dir   +/-12 dB revision checkpoints
CKE = _P.eval_ckpt_dir          # --eval_ckpt_dir  evaluation staging
OUT = _P.out_dir                # --out_dir        created if missing

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
SEEDS = [42, 123, 7]

# 비교대상: (canonical name, checkpoint file in checkpoints/full)  — single-seed(42)
COMPARATORS = [
    ("A1_NoRoomInput", "A1_NoRoomInput.pt"),
    ("A3_NoPrefInput", "A3_NoPrefInput.pt"),
    ("E3_Nercessian",  "E3_Nercessian.pt"),
    ("E4_Pepe",        "E4_Pepe.pt"),
    ("AC1_BiLSTM",     "AC1_BiLSTM.pt"),
    ("AC2_GRU",        "AC2_GRU.pt"),
    ("AC3_Conformer",  "AC3_Conformer.pt"),
]

registry = TF.build_registry()   # 패치됨: A0/A2 gain_max=12
for name, _ in COMPARATORS:
    TF._REGISTRY_TARGET[name] = "dual"
TF._REGISTRY_TARGET["A0_Proposed"] = "dual"
TF._REGISTRY_TARGET["A2_withPrefLoss"] = "dual"

ds = PEQDataset(f"{DEFAULT_DATA}/test_synth", device=str(device))

def per_sample(name, model_key, ckpt_path):
    """ckpt 로드 후 per-sample lsd_arr/dmr_arr 반환 (evaluate 재사용, 동일 샘플순서)."""
    model = registry[model_key]["model"] if model_key in registry else None
    if model is None:
        raise KeyError(model_key)
    ckpt_io.load_into(model, ckpt_path, map_location=device, label=model_key)
    r = evaluate(name, None, model, ds, device)
    return r["lsd_arr"], r["dmr_arr"]

def paired(a0_arr, base_arr, metric):
    """metric: 'lsd'(낮을수록 좋음) | 'dmr'(높을수록 좋음)."""
    diff = a0_arr - base_arr                    # A0 - base
    d_mean = float(diff.mean())
    d_z = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
    if metric == "lsd":
        win = float(np.mean(a0_arr < base_arr))
    else:
        win = float(np.mean(a0_arr > base_arr))
    return d_mean, d_z, win

# 비교대상 per-sample (single-seed, seed 무관하게 1회) ─────────────────────────
base_cache = {}
for name, ckf in COMPARATORS:
    base_cache[name] = per_sample(name, name, FULL / ckf)

# A0_s / A2_s per-seed ────────────────────────────────────────────────────────
a0_by_seed, a2_by_seed = {}, {}
for s in SEEDS:
    a0_by_seed[s] = per_sample("A0_Proposed", "A0_Proposed", SAVE / f"{cname('g12_f16k', s, 'A0')}.pt")
    a2_by_seed[s] = per_sample("A2_withPrefLoss", "A2_withPrefLoss", SAVE / f"{cname('g12_f16k', s, 'A2')}.pt")

# ── paired 계산: 비교목록 = A2(같은 seed) + 나머지(single) ────────────────────
targets = ["A2_withPrefLoss"] + [n for n, _ in COMPARATORS]
agg = {}   # agg[target][metric] = list over seeds of (Δ, d_z, win)
for tgt in targets:
    agg[tgt] = {"lsd": [], "dmr": []}
    for s in SEEDS:
        a0_lsd, a0_dmr = a0_by_seed[s]
        if tgt == "A2_withPrefLoss":
            b_lsd, b_dmr = a2_by_seed[s]           # 같은 seed
        else:
            b_lsd, b_dmr = base_cache[tgt]          # single-seed(42)
        agg[tgt]["lsd"].append(paired(a0_lsd, b_lsd, "lsd"))
        agg[tgt]["dmr"].append(paired(a0_dmr, b_dmr, "dmr"))

def ms(rows, i):
    v = np.array([r[i] for r in rows])
    return float(v.mean()), float(v.std(ddof=1))

# 원본 tab:stats 값 (jaes_optimized.tex) — 비교용
ORIG = {
    "A1_NoRoomInput": {"lsd": (-1.455, -1.082, 0.865), "dmr": (+0.096, +0.806, 0.783)},
    "A2_withPrefLoss":{"lsd": (-0.261, -0.486, 0.673), "dmr": (+0.044, +0.404, 0.496)},
    "A3_NoPrefInput": {"lsd": (-3.886, -4.226, 1.000), "dmr": (+0.420, +2.376, 0.995)},
    "E3_Nercessian":  {"lsd": (-4.522, -3.702, 0.997), "dmr": (+0.422, +2.329, 0.991)},
    "E4_Pepe":        {"lsd": (-4.366, -4.033, 1.000), "dmr": (+0.425, +2.431, 0.993)},
    "AC1_BiLSTM":     {"lsd": (+0.589, +1.194, 0.076), "dmr": (-0.019, -0.560, 0.228)},
    "AC2_GRU":        {"lsd": (+0.582, +1.175, 0.083), "dmr": (-0.018, -0.550, 0.234)},
    "AC3_Conformer":  {"lsd": (+0.572, +1.159, 0.087), "dmr": (-0.017, -0.539, 0.242)},
}

print("=" * 104)
print("Paired test (3-seed): A0 vs target  —  test_synth")
print("  A0/A2 = gain±12, 3 seeds(42,123,7);  A1/A3/E3/E4/AC1-3 = original single-seed(42)")
print("  부호: LSD Δ=A0-base(음수=A0 better), DMR Δ=A0-base(양수=A0 better). [orig]=원본 single-seed 값")
print("=" * 104)
hdr = f"{'A0 vs':>16} {'metric':>4} {'Δ(mean±std)':>20} {'d_z(mean±std)':>20} {'Win%(mean±std)':>20}   {'[orig Δ/d_z/Win]':>22}"
print(hdr); print("-" * len(hdr))
summary = {}
for tgt in targets:
    summary[tgt] = {}
    for metric in ("lsd", "dmr"):
        dm, ds_ = ms(agg[tgt][metric], 0)
        zm, zs  = ms(agg[tgt][metric], 1)
        wm, ws  = ms(agg[tgt][metric], 2)
        o = ORIG.get(tgt, {}).get(metric, (None, None, None))
        ostr = f"{o[0]:+.3f}/{o[1]:+.2f}/{o[2]*100:.0f}%" if o[0] is not None else "—"
        print(f"{tgt:>16} {metric:>4} {dm:>+9.3f}±{ds_:<8.3f} {zm:>+9.2f}±{zs:<8.2f} "
              f"{wm*100:>7.1f}±{ws*100:<8.1f}   {ostr:>22}")
        summary[tgt][metric] = dict(delta=(dm, ds_), d_z=(zm, zs), win=(wm, ws), orig=o)

out = OUT / "paired_stats_3seed_test_synth.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"\n저장: {out}")
