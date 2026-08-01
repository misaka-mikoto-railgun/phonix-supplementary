"""
param_dist_gain_freq.py — gain / centre-frequency distribution statistics
========================================================================
Sanity check on the Stage-B parameter distributions: how often the learned
values sit against their bounds.

Two saturation criteria:
  (a) own bound      : |gain| > (gain_max - 0.2)   "still pinned at this setting?"
  (b) fixed ±6 dB    : |gain| > 6.0                "would ±6 dB have pinned it?"  ★key
  and likewise for the centre frequency:
  (a) near own edge  : fc > 0.875 * fc_max,  fc < 100
  (b) fixed 16 kHz   : fc > 16000.0                "would 16 kHz have blocked it?" ★key

Criterion (b) carries the argument: a large fraction beyond the fixed
reference means the narrower bound was suppressing corrections the model
would otherwise have produced.

Output: per-config table of both criteria (seed mean±std) and gain/fc
histograms (png/pdf). Each checkpoint is instantiated with its own bounds, so
no silent clamping occurs.

Usage:
  python param_dist_gain_freq.py --configs all --seeds 42 123 7
  python param_dist_gain_freq.py --configs g12_f16k g12_f20k
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import train_full as TF
from model import DualObjectiveAdaptivePEQ
from dataset_generator_v4_tracklevel import PEQDataset
from run_gain_freq_ablation import (
    CONFIGS, SEEDS_DEFAULT, DEFAULT_DATA, BASELINE_CKPT, HERE,
    build_model, cname, FIXED_GAIN_REF, FIXED_FC_REF,
)


@torch.no_grad()
def collect_gain_fc(name, model, ds, device, batch_size=512):
    """test set 전체에서 per-section |gain|, fc 수집 → (N*K,) 1D 배열."""
    model.eval().to(device)
    gains, fcs = [], []
    for batch in ds.iter_batches(batch_size, shuffle=False):
        out = TF.model_forward(name, model, batch)
        gains.append(out["gain"].cpu().numpy())
        fcs.append(out["fc"].cpu().numpy())
    return np.abs(np.concatenate(gains)).ravel(), np.concatenate(fcs).ravel()


def _ckpt_path(cfg, seed, save_dir):
    return BASELINE_CKPT if cfg == "g6_f16k" else (save_dir / f"{cname(cfg, seed)}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",  default=str(DEFAULT_DATA))
    ap.add_argument("--save_dir",  default=str(HERE / "checkpoints"))
    ap.add_argument("--out_dir",   default=str(HERE / "results"))
    ap.add_argument("--configs",   nargs="*", default=["all"])
    ap.add_argument("--seeds",     type=int, nargs="*", default=SEEDS_DEFAULT)
    ap.add_argument("--test_split", default="test_synth")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--no_cuda",   action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    configs = list(CONFIGS.keys()) if args.configs == ["all"] else args.configs
    save_dir = Path(args.save_dir)
    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    for c in configs:
        for s in args.seeds:
            TF._REGISTRY_TARGET[cname(c, s)] = "dual"

    ds = PEQDataset(f"{data_dir}/{args.test_split}", device=str(device))

    rows = {}          # rows[cfg] = dict of per-seed metric lists
    hist_gain = {}     # hist_gain[cfg] = concatenated |gain| (last seed for plot)
    hist_fc = {}
    for c in configs:
        gmax = CONFIGS[c]["gain_max"]; fmax = CONFIGS[c]["fc_max"]
        sat_self, over6, fc_self_hi, fc_over16k, fc_lo = [], [], [], [], []
        for seed in args.seeds:
            cp = _ckpt_path(c, seed, save_dir)
            if not Path(cp).exists():
                print(f"  [{c} s{seed}] ckpt 없음 — 스킵 ({cp})")
                continue
            model = build_model(c)
            ck = torch.load(cp, map_location=device, weights_only=False)
            state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            model.load_state_dict(state, strict=False)
            g, fc = collect_gain_fc(cname(c, seed), model, ds, device, args.batch_size)
            sat_self.append(float(np.mean(g > (gmax - 0.2))))   # (a)
            over6.append(float(np.mean(g > FIXED_GAIN_REF)))     # (b) ★
            fc_self_hi.append(float(np.mean(fc > 0.875 * fmax))) # (a)
            fc_over16k.append(float(np.mean(fc > FIXED_FC_REF))) # (b) ★
            fc_lo.append(float(np.mean(fc < 100.0)))
            hist_gain[c] = g; hist_fc[c] = fc
            if c == "g6_f16k":
                break  # baseline 동일 ckpt
        if not over6:
            continue

        def ms(v): return (float(np.mean(v)), float(np.std(v)), len(v))
        rows[c] = dict(
            gain_max=gmax, fc_max=fmax,
            gain_sat_self=ms(sat_self), gain_over6=ms(over6),
            fc_hi_self=ms(fc_self_hi), fc_over16k=ms(fc_over16k), fc_lo=ms(fc_lo),
        )

    # ── 표 출력 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*92}")
    print(f"saturation 두 기준 (seed mean±std %)  —  test={args.test_split}")
    print("  (a) self  = 자기 boundary 기준 |g|>gmax-0.2 / fc>0.875·fmax")
    print("  (b) fixed = ±6 / 16k 고정 기준 |g|>6 / fc>16k   ← 핵심 증거")
    print("=" * 92)
    hdr = (f"{'config':>9} {'gain_sat(a)':>12} {'gain>6(b)':>11} "
           f"{'fc_hi(a)':>10} {'fc>16k(b)':>11} {'fc<100':>9}")
    print(hdr); print("-" * len(hdr))
    for c in configs:
        if c not in rows:
            continue
        r = rows[c]
        def cell(t): return f"{t[0]*100:5.1f}±{t[1]*100:4.1f}"
        print(f"{c:>9} {cell(r['gain_sat_self']):>12} {cell(r['gain_over6']):>11} "
              f"{cell(r['fc_hi_self']):>10} {cell(r['fc_over16k']):>11} {cell(r['fc_lo']):>9}")

    print("\n해석: ±12 모델의 gain>6(b) 가 높을수록, ±6 제약이 그만큼 보정을 억제했다는 직접 증거.")

    # ── 히스토그램 ───────────────────────────────────────────────────────────
    if hist_gain:
        fig, (axg, axf) = plt.subplots(1, 2, figsize=(12, 4.2))
        for c in configs:
            if c not in hist_gain:
                continue
            axg.hist(hist_gain[c], bins=60, histtype="step", density=True, label=c)
            axf.hist(hist_fc[c], bins=60, histtype="step", density=True, label=c)
        axg.axvline(FIXED_GAIN_REF, color="k", ls="--", lw=1, label="±6 ref")
        axg.set_xlabel("|gain| (dB)"); axg.set_ylabel("density"); axg.set_title("per-section |gain|"); axg.legend(fontsize=8)
        axf.axvline(FIXED_FC_REF, color="k", ls="--", lw=1, label="16k ref")
        axf.set_xlabel("fc (Hz)"); axf.set_title("center frequency"); axf.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"param_dist_gain_freq_{args.test_split}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"figure 저장: {out_dir / f'param_dist_gain_freq_{args.test_split}.png'}")

    out_path = out_dir / f"param_dist_gain_freq_{args.test_split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
