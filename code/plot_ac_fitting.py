"""
plot_ac_fitting.py — AC Fitting 결과 비교 bar chart
====================================================
Options A/C/D 결과를 한 figure 로 시각화.

Bar groups:
  A0_Proposed (reference)
  AC2_GRU     raw / 7-pt sampled (D) / scipy fitted (A) / biquad retrained (C)
  (AC1/AC3 Biquad 있으면 추가)

Usage
-----
  python plot_ac_fitting.py --stat_dir ./paper_outputs/stats --out_dir ./paper_outputs
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────────────────
# 메트릭
# ──────────────────────────────────────────────────────────

def lsd(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=-1))

def bootstrap_ci(arr, n_boot=2000, seed=42):
    rng   = np.random.default_rng(seed)
    means = np.array([rng.choice(arr, len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_lsd(path: Path) -> np.ndarray | None:
    if path.exists():
        return np.load(path).astype(np.float64)
    return None


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat_dir", default="./paper_outputs/stats")
    parser.add_argument("--out_dir",  default="./paper_outputs")
    parser.add_argument("--ac_model", default="AC2_GRU",
                        help="기준 AC 모델 (AC2_GRU 권장)")
    parser.add_argument("--dpi",      type=int, default=300)
    args = parser.parse_args()

    stat_dir = Path(args.stat_dir)
    fig_dir  = Path(args.out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ac  = args.ac_model                    # e.g. "AC2_GRU"
    dual_path = stat_dir / "targets_dual.npy"

    # ── 데이터 수집 ──────────────────────────────────────────
    entries = []   # (label, lsd_array, color, group)

    # A0 reference
    a0_lsd = load_lsd(stat_dir / "A0_Proposed_lsd.npy")
    if a0_lsd is not None:
        entries.append(("A0\nProposed", a0_lsd, "#2196F3", "A0"))

    # AC raw
    ac_lsd = load_lsd(stat_dir / f"{ac}_lsd.npy")
    if ac_lsd is not None:
        entries.append((f"{ac}\nraw", ac_lsd, "#FF9800", "AC"))

    # Option D: 7-point sampled
    d_lsd = load_lsd(stat_dir / f"{ac}_7pt_sampled_lsd.npy")
    if d_lsd is not None:
        entries.append((f"{ac}\n7-pt (D)", d_lsd, "#FF5722", "AC"))

    # Option A: scipy fitted
    a_lsd = load_lsd(stat_dir / f"{ac}_biquad_fitted_lsd.npy")
    if a_lsd is not None:
        entries.append((f"{ac}\nfitted (A)", a_lsd, "#E91E63", "AC"))

    # Option C: biquad retrained — AC2_GRU_Biquad
    c2_lsd = load_lsd(stat_dir / f"{ac}_Biquad_lsd.npy")
    if c2_lsd is not None:
        entries.append((f"{ac}\nBiquad (C)", c2_lsd, "#9C27B0", "AC_Biquad"))

    # AC1 / AC3 Biquad (있으면)
    if load_lsd(stat_dir / "AC1_BiLSTM_Biquad_lsd.npy") is not None:
        entries.append(("AC1\nBiquad (C)", load_lsd(stat_dir / "AC1_BiLSTM_Biquad_lsd.npy"),
                        "#673AB7", "AC_Biquad"))
    if load_lsd(stat_dir / "AC3_Conformer_Biquad_lsd.npy") is not None:
        entries.append(("AC3\nBiquad (C)", load_lsd(stat_dir / "AC3_Conformer_Biquad_lsd.npy"),
                        "#7B1FA2", "AC_Biquad"))

    if not entries:
        print("[ERROR] 사용 가능한 lsd.npy 파일 없음. experiments_fixed_updated.py 먼저 실행.")
        return

    # 배열 길이 통일 (가장 짧은 것 기준)
    N = min(len(e[1]) for e in entries)
    entries = [(lbl, arr[:N], clr, grp) for lbl, arr, clr, grp in entries]

    # ── 통계 계산 ────────────────────────────────────────────
    means  = [arr.mean() for _, arr, _, _ in entries]
    cis    = [bootstrap_ci(arr) for _, arr, _, _ in entries]
    errors = [(m - lo, hi - m) for m, (lo, hi) in zip(means, cis)]

    labels = [lbl for lbl, _, _, _ in entries]
    colors = [clr for _, _, clr, _ in entries]
    xerr   = np.array([[e[0], e[1]] for e in errors]).T   # (2, N_bars)

    # ── Figure ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(7, len(entries) * 1.2), 4.5))

    x = np.arange(len(entries))
    bars = ax.bar(x, means, color=colors, width=0.6,
                  yerr=xerr, capsize=4, error_kw={"linewidth": 1.2, "ecolor": "#333333"})

    # JND 참조선 (A0 mean 기준 ±0.5)
    if a0_lsd is not None:
        a0_mean = a0_lsd[:N].mean()
        ax.axhline(a0_mean + 0.5, color="#2196F3", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(a0_mean,       color="#2196F3", linestyle="-",  linewidth=1.0, alpha=0.5)
        ax.text(len(entries) - 0.4, a0_mean + 0.52,
                "A0 + JND (0.5 dB)", color="#2196F3", fontsize=7, va="bottom")

    # 값 레이블
    for i, (bar, m) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Dual-target LSD (dB)", fontsize=10)
    ax.set_title("AC Fitting: Representation Penalty Analysis\n"
                 "(error bars = 95% bootstrap CI)", fontsize=10)
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 범례
    legend_patches = [
        mpatches.Patch(color="#2196F3", label="A0 Proposed (reference)"),
        mpatches.Patch(color="#FF9800", label="AC raw (dense 128-bin)"),
        mpatches.Patch(color="#FF5722", label="Option D: 7-pt sampling"),
        mpatches.Patch(color="#E91E63", label="Option A: scipy fitting"),
        mpatches.Patch(color="#9C27B0", label="Option C: biquad retrained"),
    ]
    ax.legend(handles=legend_patches, fontsize=7.5, loc="upper left",
              framealpha=0.8, ncol=2)

    plt.tight_layout()

    stem = f"fig_ac_fitting_{ac}"
    fig.savefig(fig_dir / f"{stem}.pdf", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / stem}.pdf / .png")

    # ── 콘솔 요약 ────────────────────────────────────────────
    print(f"\n{'Condition':30s}  {'Mean LSD':>10}  {'95% CI':>22}")
    print("-" * 66)
    for lbl, arr, _, _ in entries:
        lo, hi = bootstrap_ci(arr)
        print(f"  {lbl.replace(chr(10),' '):28s}  {arr.mean():>10.4f}  [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
