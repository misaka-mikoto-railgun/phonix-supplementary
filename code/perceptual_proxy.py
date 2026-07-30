"""
Perceptual Proxy Analysis  --  PHONIX Adaptive EQ
==================================================
Listening test 부재를 보완하는 perceptual proxy metric 계산.

Metrics
-------
1. ERB-weighted LSD   -- Moore & Glasberg (1983) 청각 임계대역 가중
2. 1/3-octave LSD     -- ANSI S1.11 방식 스펙트럼 평활화 후 LSD
3. JND analysis       -- Toole & Olive (1988) JND threshold (0.5 dB) 비교
4. Bootstrap 95% CI   -- 모든 LSD 메트릭에 신뢰구간 추가

Usage
-----
  # CI + JND 즉시 실행 (pred 배열 불필요)
  python perceptual_proxy.py --stat_dir ./paper_outputs/stats --out_dir ./paper_outputs --ci_only

  # 전체 실행 (experiments_fixed_updated.py 재실행 후 pred 배열 생성 시)
  python perceptual_proxy.py --stat_dir ./paper_outputs/stats --out_dir ./paper_outputs

Output
------
  paper_outputs/tables/table_bootstrap_ci.{csv,tex}
  paper_outputs/tables/table_jnd_analysis.{csv,tex}
  paper_outputs/tables/table_perceptual_proxy.{csv,tex}   (pred 배열 있을 때)
  paper_outputs/figures/fig_perceptual_proxy.{pdf,png}
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "lines.linewidth":   1.4,
})

# ──────────────────────────────────────────────────────────
# 주파수 축 및 perceptual 가중치
# ──────────────────────────────────────────────────────────

TARGET_FREQS = np.geomspace(20.0, 24000.0, 128).astype(np.float32)

def _erb(freqs: np.ndarray) -> np.ndarray:
    """Moore & Glasberg (1983): ERB(f) = 24.7*(4.37*f/1000 + 1)"""
    return 24.7 * (4.37 * freqs / 1000.0 + 1.0)

def _erb_weights(freqs: np.ndarray) -> np.ndarray:
    w = 1.0 / _erb(freqs)
    return (w / w.sum()).astype(np.float32)

def _third_oct_matrix(freqs: np.ndarray) -> np.ndarray:
    factor = 2.0 ** (1.0 / 6.0)
    n = len(freqs)
    W = np.zeros((n, n), dtype=np.float32)
    for i, f in enumerate(freqs):
        mask = (freqs >= f / factor) & (freqs <= f * factor)
        cnt = mask.sum()
        if cnt > 0:
            W[i, mask] = 1.0 / cnt
    return W


ERB_W   = _erb_weights(TARGET_FREQS)
THIRD_W = _third_oct_matrix(TARGET_FREQS)

# ──────────────────────────────────────────────────────────
# 메트릭 함수
# ──────────────────────────────────────────────────────────

def lsd_std(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - target) ** 2, axis=-1))

def lsd_erb_fn(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    diff2 = (pred - target) ** 2
    return np.sqrt((diff2 * ERB_W[None, :]).sum(axis=-1))

def lsd_third_oct(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_s   = pred   @ THIRD_W.T
    target_s = target @ THIRD_W.T
    return np.sqrt(np.mean((pred_s - target_s) ** 2, axis=-1))

def bootstrap_ci(arr: np.ndarray, n_boot: int = 2000,
                 ci: float = 95.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(arr, len(arr), replace=True).mean() for _ in range(n_boot)]
    )
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return float(lo), float(hi)

# ──────────────────────────────────────────────────────────
# 모델 목록 (메인 폴더 naming convention)
# ──────────────────────────────────────────────────────────

# A0_Proposed    = proposed model (no preference loss, the proposed model in the main results)
# A2_withPrefLoss = ablation: with preference loss (former A0_Full)
KEY_MODELS = [
    ("A0_Proposed",      "A0 Proposed"),
    ("A2_withPrefLoss",  "A2 (w/ PrefLoss)"),
    ("A1_NoRoomInput",   "A1 (w/o Room)"),
    ("A3_NoPrefInput",   "A3 (w/o PrefInput)"),
    ("AC1_BiLSTM",       "AC1 BiLSTM"),
    ("AC2_GRU",          "AC2 GRU"),
    ("AC3_Conformer",    "AC3 Conformer"),
    ("E3_Nercessian",    "E3 Nercessian"),
    ("E4_Pepe",          "E4 Pepe"),
    ("E6_DSP",           "E6 DSP"),
]

PROPOSED = "A0_Proposed"
JND_DB   = 0.5   # Toole & Olive (1988), 1 kHz EQ peak JND

# ──────────────────────────────────────────────────────────
# 표 생성
# ──────────────────────────────────────────────────────────

def build_bootstrap_ci_table(stat_dir: Path, use_real: bool = False) -> pd.DataFrame:
    suffix = "_real_lsd.npy" if use_real else "_lsd.npy"
    rows = []
    for name, label in KEY_MODELS:
        p = stat_dir / f"{name}{suffix}"
        if not p.exists():
            continue
        arr = np.load(p)
        lo, hi = bootstrap_ci(arr)
        rows.append({
            "Model":    label,
            "Mean LSD": f"{arr.mean():.4f}",
            "95% CI":   f"[{lo:.4f}, {hi:.4f}]",
        })
    return pd.DataFrame(rows)


def build_jnd_table(stat_dir: Path) -> pd.DataFrame:
    p0 = stat_dir / f"{PROPOSED}_lsd.npy"
    if not p0.exists():
        return pd.DataFrame()
    a0 = np.load(p0)
    rows = []
    for name, label in KEY_MODELS:
        if name == PROPOSED:
            continue
        p = stat_dir / f"{name}_lsd.npy"
        if not p.exists():
            continue
        delta = np.abs(a0 - np.load(p))
        pct   = float((delta < JND_DB).mean() * 100)
        rows.append({
            "Comparison":              f"Proposed vs {label}",
            "Mean |delta-LSD| (dB)":  f"{delta.mean():.4f}",
            f"% below {JND_DB} dB JND": f"{pct:.1f}%",
            "Perceptual equiv.":       "Yes" if pct >= 50 else "No",
        })
    return pd.DataFrame(rows)


def build_perceptual_table(stat_dir: Path) -> pd.DataFrame:
    dual_path = stat_dir / "targets_dual.npy"
    if not dual_path.exists():
        return pd.DataFrame()
    dual_target = np.load(dual_path)
    rows = []
    for name, label in KEY_MODELS:
        pred_path = stat_dir / f"{name}_pred.npy"
        if not pred_path.exists():
            continue
        pred = np.load(pred_path)
        lsd_path = stat_dir / f"{name}_lsd.npy"
        std_arr  = np.load(lsd_path) if lsd_path.exists() else lsd_std(pred, dual_target)
        erb_arr  = lsd_erb_fn(pred, dual_target)
        oct_arr  = lsd_third_oct(pred, dual_target)
        lo_s, hi_s = bootstrap_ci(std_arr)
        lo_e, hi_e = bootstrap_ci(erb_arr)
        lo_3, hi_3 = bootstrap_ci(oct_arr)
        rows.append({
            "Model":             label,
            "LSD [95% CI]":      f"{std_arr.mean():.3f} [{lo_s:.3f},{hi_s:.3f}]",
            "ERB-LSD [95% CI]":  f"{erb_arr.mean():.3f} [{lo_e:.3f},{hi_e:.3f}]",
            "1/3-oct LSD [CI]":  f"{oct_arr.mean():.3f} [{lo_3:.3f},{hi_3:.3f}]",
        })
    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────

def fig_perceptual_proxy(stat_dir: Path, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    # (a) ERB weight curve
    ax = axes[0]
    ax.plot(TARGET_FREQS, ERB_W * 1000, color="#2980B9", lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Weight x 1e3 (a.u.)")
    ax.set_title("(a) ERB Auditory Weighting")
    ax.set_xlim(20, 24000)
    ax.axvspan(20, 1000, alpha=0.08, color="#E74C3C", label="Low-freq emphasis")
    ax.legend(fontsize=7)

    # (b) LSD comparison bar chart (if pred arrays exist)
    ax = axes[1]
    dual_path = stat_dir / "targets_dual.npy"
    pred_avail = dual_path.exists() and (stat_dir / f"{PROPOSED}_pred.npy").exists()
    if pred_avail:
        dual_target = np.load(dual_path)
        compare_models = [
            ("A2_withPrefLoss", "Proposed", "#C0392B"),
            ("A0_Proposed",     "A0",       "#E67E22"),
            ("AC2_GRU",         "AC2",      "#2980B9"),
            ("AC3_Conformer",   "AC3",      "#8E44AD"),
        ]
        x = np.arange(3)
        labels = ["LSD", "ERB-LSD", "1/3-oct LSD"]
        w = 0.18
        offsets = np.linspace(-0.27, 0.27, len(compare_models))
        for i, (name, disp, color) in enumerate(compare_models):
            pp = stat_dir / f"{name}_pred.npy"
            if not pp.exists():
                continue
            pred = np.load(pp)
            lp = stat_dir / f"{name}_lsd.npy"
            vals = [
                np.load(lp).mean() if lp.exists() else lsd_std(pred, dual_target).mean(),
                lsd_erb_fn(pred, dual_target).mean(),
                lsd_third_oct(pred, dual_target).mean(),
            ]
            ax.bar(x + offsets[i], vals, width=w, label=disp, color=color, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("LSD (dB)")
        ax.set_title("(b) Perceptual Proxy Metrics")
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5,
                "Re-run experiments_fixed_updated.py\nto generate _pred.npy arrays",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=7.5, color="gray", style="italic")
        ax.set_title("(b) Perceptual Proxy Metrics (pending)")

    # (c) JND histogram
    ax = axes[2]
    p0 = stat_dir / f"{PROPOSED}_lsd.npy"
    if p0.exists():
        a0 = np.load(p0)
        jnd_pairs = [
            ("A0_Proposed",  "A0",  "#E67E22"),
            ("AC2_GRU",      "AC2", "#2980B9"),
            ("AC3_Conformer","AC3", "#8E44AD"),
        ]
        bins = np.linspace(0, 3.0, 40)
        for name, disp, color in jnd_pairs:
            p = stat_dir / f"{name}_lsd.npy"
            if not p.exists():
                continue
            delta = np.abs(a0 - np.load(p))
            pct = (delta < JND_DB).mean() * 100
            ax.hist(delta, bins=bins, alpha=0.45, color=color,
                    label=f"Prop. vs {disp} ({pct:.0f}% < JND)")
        ax.axvline(JND_DB, color="red", ls="--", lw=1.5,
                   label=f"JND = {JND_DB} dB\n(Toole & Olive 1988)")
        ax.set_xlabel("|delta-LSD| per sample (dB)")
        ax.set_ylabel("Count")
        ax.set_title("(c) Per-Sample Difference vs. JND")
        ax.legend(fontsize=6.5)
    else:
        ax.set_title("(c) JND Analysis")

    fig.tight_layout(pad=1.0)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig_perceptual_proxy.{ext}")
    plt.close(fig)
    print(f"  Saved fig_perceptual_proxy.pdf/.png")

# ──────────────────────────────────────────────────────────
# LaTeX helper
# ──────────────────────────────────────────────────────────

def df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    n = len(df.columns)
    col_fmt = "l" + "r" * (n - 1)
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        " & ".join(df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def save_table(df: pd.DataFrame, out_dir: Path, stem: str, caption: str, label: str):
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tab_dir / f"{stem}.csv", index=False)
    (tab_dir / f"{stem}.tex").write_text(
        df_to_latex(df, caption, label), encoding="utf-8"
    )
    print(f"  Saved {stem}.csv / .tex")

# ──────────────────────────────────────────────────────────
# 콘솔 요약
# ──────────────────────────────────────────────────────────

def print_bootstrap_summary(stat_dir: Path):
    print("\n-- Bootstrap 95% CI (LSD, n_boot=2000) --")
    print(f"  {'Model':28s} | {'Mean LSD':>10} | {'95% CI':>22}")
    print("  " + "-" * 66)
    for name, label in KEY_MODELS:
        p = stat_dir / f"{name}_lsd.npy"
        if not p.exists():
            continue
        arr = np.load(p)
        lo, hi = bootstrap_ci(arr)
        print(f"  {label:28s} | {arr.mean():>10.4f} | [{lo:.4f}, {hi:.4f}]")


def print_jnd_summary(stat_dir: Path):
    p0 = stat_dir / f"{PROPOSED}_lsd.npy"
    if not p0.exists():
        print("  [JND] Proposed LSD array not found -- skip")
        return
    a0 = np.load(p0)
    print(f"\n-- JND Analysis (Toole & Olive 1988, threshold = {JND_DB} dB) --")
    for name, label in KEY_MODELS:
        if name == PROPOSED:
            continue
        p = stat_dir / f"{name}_lsd.npy"
        if not p.exists():
            continue
        delta = np.abs(a0 - np.load(p))
        pct   = (delta < JND_DB).mean() * 100
        print(f"  Proposed vs {label:22s}: mean|dLSD|={delta.mean():.3f} dB, "
              f"{pct:.1f}% samples < JND")
    print()
    print("  -> At 1 kHz, EQ peak JND ~= 0.5 dB (Toole & Olive, 1988).")
    print("     Samples below JND threshold represent perceptually equivalent")
    print("     corrections where the auditory system cannot distinguish models.")

# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat_dir", default="./paper_outputs/stats")
    parser.add_argument("--out_dir",  default="./paper_outputs")
    parser.add_argument("--ci_only",  action="store_true",
                        help="Bootstrap CI + JND only (no pred arrays needed)")
    parser.add_argument("--n_boot",   type=int, default=2000)
    args = parser.parse_args()

    stat_dir = Path(args.stat_dir)
    out_dir  = Path(args.out_dir)

    print("=" * 60)
    print("Perceptual Proxy Analysis -- PHONIX Adaptive EQ")
    print("=" * 60)

    # 1. Bootstrap CI
    print_bootstrap_summary(stat_dir)
    ci_df = build_bootstrap_ci_table(stat_dir)
    if not ci_df.empty:
        save_table(
            ci_df, out_dir, "table_bootstrap_ci",
            "95\\% Bootstrap Confidence Intervals for Dual-Target LSD ($N=5000$)",
            "tab:bootstrap_ci",
        )

    # 2. JND analysis
    print_jnd_summary(stat_dir)
    jnd_df = build_jnd_table(stat_dir)
    if not jnd_df.empty:
        save_table(
            jnd_df, out_dir, "table_jnd_analysis",
            f"Per-Sample LSD Difference vs. Perceptual JND Threshold "
            f"(Toole \\& Olive, 1988; $\\delta={JND_DB}$ dB)",
            "tab:jnd_analysis",
        )

    # 3. ERB-LSD + 1/3-oct LSD (pred 배열 필요)
    if not args.ci_only:
        print("\n-- Perceptual Proxy Metrics (ERB-LSD, 1/3-oct LSD) --")
        dual_p = stat_dir / "targets_dual.npy"
        pred_p = stat_dir / f"{PROPOSED}_pred.npy"
        if not dual_p.exists() or not pred_p.exists():
            print("  [SKIP] _pred.npy arrays not found.")
            print("  -> Re-run experiments_fixed_updated.py with checkpoints first:")
            print("     python experiments_fixed_updated.py"
                  " --data_dir ./data/dataset_v3"
                  " --ckpt_dir ./checkpoints/full"
                  " --out_dir ./paper_outputs")
        else:
            perc_df = build_perceptual_table(stat_dir)
            if not perc_df.empty:
                print(perc_df.to_string(index=False))
                save_table(
                    perc_df, out_dir, "table_perceptual_proxy",
                    "Perceptual Proxy Metrics: ERB-Weighted and 1/3-Octave Smoothed LSD "
                    "with 95\\% Bootstrap CI",
                    "tab:perceptual_proxy",
                )

    # 4. Figure
    print("\n-- Generating figure --")
    fig_perceptual_proxy(stat_dir, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
