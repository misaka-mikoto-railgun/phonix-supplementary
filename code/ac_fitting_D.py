"""
AC Fitting  --  Option D: naive seven-point sampling (re-implementation)
=======================================================================
The original generator of results_json/ac_fitting_D_naive7pt.csv was not kept in
the release; this script regenerates it with the same definition and the exact
RBJ biquad rendering used everywhere else in the revised pipeline:

  1. take the AC model's 128-bin dense output (stat_dir/{model}_pred.npy),
  2. sample it at seven centre frequencies (log-spaced over [FC_MIN, FC_MAX]),
  3. use the sampled dB value as each section's gain (plateau gain for a shelf),
     fixed Q (1.41 for peaking sections, 1/sqrt2 = no resonant lip for the shelves),
  4. render the seven exact RBJ sections with the SAME section types as the A0 /
     Option A / Option C head (default LPPPPPH: low shelf + 5 peaking + high shelf;
     pass --section_types PPPPPPP for the all-peaking variant) and compute LSD vs
     dual target.

No model is trained or fitted, so the Stage-B gain bound plays no part.

Outputs (same names the rest of the pipeline expects):
  {out_dir}/tables/ac_fitting_D_naive7pt.csv
  {stat_dir}/{model}_7pt_sampled_lsd.npy          (read by plot_ac_fitting.py)

Usage
-----
  python ac_fitting_D.py --stat_dir ./paper_outputs/stats --out_dir ./paper_outputs --model AC2_GRU
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from exact_peq_np import option_d_naive7pt, TARGET_FREQS

FC_MIN, FC_MAX = 80.0, 16000.0
Q_FIXED = 1.41
SHELF_Q = 0.70710678          # Butterworth shelf slope: no overshoot
SECTION_TYPES = "LPPPPPH"     # must match model.py / ac_fitting_A.py


def lsd(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=-1))


def bootstrap_ci(arr, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(arr, len(arr), replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat_dir", default="./paper_outputs/stats")
    ap.add_argument("--out_dir", default="./paper_outputs")
    ap.add_argument("--model", default="AC2_GRU")
    ap.add_argument("--q", type=float, default=Q_FIXED, help="Q of the peaking sections")
    ap.add_argument("--shelf_q", type=float, default=SHELF_Q, help="Q of the shelf sections (L/H)")
    ap.add_argument("--section_types", default=SECTION_TYPES, help="7 chars of P/L/H")
    ap.add_argument("--n_samples", type=int, default=0)
    args = ap.parse_args()

    stat_dir, out_dir = Path(args.stat_dir), Path(args.out_dir)
    tab_dir = out_dir / "tables"; tab_dir.mkdir(parents=True, exist_ok=True)

    pred = np.load(stat_dir / f"{args.model}_pred.npy").astype(np.float64)
    dual = np.load(stat_dir / "targets_dual.npy").astype(np.float64)
    a0_path = stat_dir / "A0_Proposed_lsd.npy"
    a0_lsd = np.load(a0_path).astype(np.float64) if a0_path.exists() else None
    N = len(pred) if args.n_samples <= 0 else min(args.n_samples, len(pred))
    pred, dual = pred[:N], dual[:N]
    if a0_lsd is not None:
        a0_lsd = a0_lsd[:N]

    fc_points = np.geomspace(FC_MIN, FC_MAX, 7)
    types = args.section_types.upper()
    assert len(types) == 7 and set(types) <= set("PLH"), "--section_types must be 7 chars of P/L/H"
    q_per_section = np.array([args.shelf_q if c in "LH" else args.q for c in types])
    sampled = option_d_naive7pt(pred, fc_points=fc_points, q=q_per_section, freqs=TARGET_FREQS, section_types=types)

    lsd_raw = lsd(pred, dual)
    lsd_d = lsd(sampled, dual)
    np.save(stat_dir / f"{args.model}_7pt_sampled_lsd.npy", lsd_d)

    rows = [
        (f"{args.model} dense (raw)", lsd_raw),
        (f"{args.model} 7-pt sampled ({types}, Q={args.q:g}/{args.shelf_q:.3g})", lsd_d),
        ("Representation penalty", lsd_d - lsd_raw),
    ]
    if a0_lsd is not None:
        rows.append(("A0_Proposed (reference)", a0_lsd))

    print(f"Option D - naive 7-point sampling of {args.model} dense output  (N={N}, exact RBJ rendering, sections={types})")
    print(f"fc points (Hz): {np.round(fc_points).astype(int).tolist()}   Q peaking = {args.q}, Q shelf = {args.shelf_q}")
    print(f"{'Model/Condition':36s} | {'Mean':>8} | {'95% CI':>20}")
    print("-" * 70)
    out = []
    for name, arr in rows:
        lo, hi = bootstrap_ci(arr)
        print(f"{name:36s} | {arr.mean():8.4f} | [{lo:.4f},{hi:.4f}]")
        out.append({"Model/Condition": name, "LSD mean": f"{arr.mean():.4f}", "95% CI": f"[{lo:.4f},{hi:.4f}]"})
    pd.DataFrame(out).to_csv(tab_dir / "ac_fitting_D_naive7pt.csv", index=False)
    print(f"\nsaved: {tab_dir / 'ac_fitting_D_naive7pt.csv'}, {stat_dir / (args.model + '_7pt_sampled_lsd.npy')}")


if __name__ == "__main__":
    main()