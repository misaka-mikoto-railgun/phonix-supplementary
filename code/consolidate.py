"""
consolidate.py — JAES revision 전체 실험 정리
=============================================
revision_gain_freq/paper_outputs/{tables,figures} 에
  - 모든 실험 결과를 CSV 테이블로
  - 색맹(Okabe–Ito) + 흑백인쇄 대응(마커/선종/해치 병행) figure 로
정리. 라벨 겹침/잘림 방지(constrained_layout + bbox_inches='tight' + 회전/여백).

값 출처: gain_freq_summary_*, paired_stats_3seed, track_stats_3seed,
         ac_biquad_table, param_dist_gain_freq, gap_analysis (revision/results).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE / "paper_outputs"
TAB = OUT / "tables"; FIG = OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

# ── Okabe–Ito 색맹 안전 팔레트 ────────────────────────────────────────────────
OI = dict(black="#000000", orange="#E69F00", skyblue="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", purple="#CC79A7",
          grey="#999999")
plt.rcParams.update({
    "figure.constrained_layout.use": True, "figure.dpi": 200,
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  fig: {name}.png/.pdf")

# ════════════════════════════════════════════════════════════════════════════
# 1. 데이터 (revision/results 에서 확정된 값)
# ════════════════════════════════════════════════════════════════════════════
# Main (synth/real). A0/A2 = gain±12 3-seed mean±std; comparators = single-seed(42).
MAIN = [
    # model, group, synthLSD, synthLSD_std, synthDMR, synthCos, realLSD, realLSD_std, realDMR, seed_note
    ("A0_Proposed(±12)", "Proposed", 1.095, 0.116, 0.929, 0.974, 1.792, 0.182, 0.891, "3-seed"),
    ("A0_Proposed(±6,orig)", "Reference", 1.442, 0.0, 0.928, 0.960, 1.941, 0.0, 0.885, "1-seed"),
    ("A2_withPrefLoss(±12)", "Ablation", 1.329, 0.368, 0.933, 0.958, 1.981, 0.301, 0.889, "3-seed"),
    ("A1_NoRoomInput", "Ablation", 2.897, 0.0, 0.832, np.nan, 4.875, 0.0, 0.740, "1-seed"),
    ("E3_Nercessian", "Baseline", 5.964, 0.0, 0.507, np.nan, 6.792, 0.0, 0.512, "1-seed"),
    ("E4_Pepe", "Baseline", 5.809, 0.0, 0.504, np.nan, 6.623, 0.0, 0.516, "1-seed"),
    ("AC1_BiLSTM", "Arch", 0.853, 0.0, 0.947, 0.982, 1.508, 0.0, 0.910, "1-seed"),
    ("AC2_GRU", "Arch", 0.860, 0.0, 0.946, 0.982, 1.506, 0.0, 0.910, "1-seed"),
    ("AC3_Conformer", "Arch", 0.870, 0.0, 0.945, 0.982, 1.576, 0.0, 0.906, "1-seed"),
]
# Paired 3-seed: target -> (LSDΔ,LSDΔstd,LSDdz,LSDwin, DMRΔ,DMRdz,DMRwin, origLSDΔ,origLSDdz,origWin)
PAIRED = {
    "A2": (-0.234, 0.519, -0.19, 57.9, -0.004, 0.02, 41.7, -0.261, -0.486, 67.3),
    "A1": (-1.618, 0.142, -1.47, 96.2, 0.084, 0.74, 78.3, -1.455, -1.082, 86.5),
    "A3": (-4.233, 0.142, -4.32, 100.0, 0.420, 2.32, 99.0, -3.886, -4.226, 100.0),
    "E3": (-4.869, 0.142, -4.03, 100.0, 0.422, 2.31, 98.6, -4.522, -3.702, 99.7),
    "E4": (-4.714, 0.142, -4.37, 99.9, 0.425, 2.39, 98.7, -4.366, -4.033, 100.0),
    "AC1": (0.242, 0.142, 0.70, 20.5, -0.018, -0.28, 29.8, 0.589, 1.194, 7.6),
    "AC2": (0.235, 0.142, 0.66, 21.8, -0.017, -0.26, 30.4, 0.582, 1.175, 8.3),
    "AC3": (0.225, 0.142, 0.62, 23.5, -0.016, -0.23, 32.6, 0.572, 1.159, 8.7),
}
# Track 3-seed (N=1306): target -> (LSDΔ,LSDΔstd,LSDdz, DMRΔ,DMRdz)
TRACK = {
    "A2": (-0.235, 0.521, -0.23, -0.005, 0.02), "A1": (-1.604, 0.140, -1.83, 0.083, 0.96),
    "A3": (-4.239, 0.140, -5.61, 0.422, 3.01), "E3": (-4.892, 0.140, -5.11, 0.427, 3.05),
    "E4": (-4.730, 0.140, -5.59, 0.428, 3.17), "AC1": (0.243, 0.140, 0.94, -0.018, -0.36),
    "AC2": (0.237, 0.140, 0.92, -0.017, -0.35), "AC3": (0.224, 0.140, 0.84, -0.016, -0.29),
}
# Saturation (A0, param_dist two criteria): config -> (gain_sat, gain_sat_std, gain>6, gain>6_std, fc_hi, fc>16k)
SAT = {
    "±6 / 16k (orig)": (57.8, 0.0, 0.0, 0.0, 17.9, 0.0),
    "±12 / 16k (new)": (16.6, 4.7, 72.8, 3.9, 11.9, 0.0),
    "±12 / 20k": (7.5, 3.4, 74.1, 0.5, 5.8, 9.8),
}
# AC_Biquad: name -> (lsd, ci_lo, ci_hi, vsA0, pct_jnd, penalty, sat_self, gain_over6, gmax)
ACBQ = {
    "AC1 ±6": (1.039, 1.028, 1.051, -0.056, 89.2, 0.187, 9.7, 0.0, 6),
    "AC1 ±12": (1.010, 0.999, 1.021, -0.085, 88.8, 0.157, 0.0, 38.0, 12),
    "AC2 ±6": (1.005, 0.993, 1.016, -0.090, 89.8, 0.145, 2.8, 0.0, 6),
    "AC2 ±12": (1.017, 1.006, 1.029, -0.078, 88.7, 0.157, 0.0, 46.0, 12),
    "AC3 ±6": (1.009, 0.997, 1.020, -0.086, 87.2, 0.138, 3.3, 0.0, 6),
    "AC3 ±12": (1.010, 0.999, 1.021, -0.085, 88.3, 0.140, 0.0, 64.0, 12),
}
OPTION = {"OptA ±6 (SciPy)": 1.026, "OptA ±12 (SciPy)": 0.918,
          "OptC ±12 (retrain, AC2)": 1.017, "OptD (naive 7pt)": 7.755, "dense AC2 (raw)": 0.860}
A0_MEAN = 1.095
# Gap analysis (A0 per-seed)
GAP = {"s42": (0.9994, 1.6421), "s123": (1.2578, 2.0478), "s7": (1.0279, 1.6868)}
GAP_BASE = 0.499

# ════════════════════════════════════════════════════════════════════════════
# 2. CSV 테이블
# ════════════════════════════════════════════════════════════════════════════
def w(df, name):
    df.to_csv(TAB / name, index=False, encoding="utf-8-sig")
    print(f"  csv: {name}")

# T1 main
df1 = pd.DataFrame([{
    "Model": m, "Group": g, "Synth LSD": s, "Synth LSD std": ss, "Synth DMR": sd,
    "Synth CosSim": sc, "Real LSD": r, "Real LSD std": rs, "Real DMR": rd,
    "Domain gap (real-synth)": round(r - s, 3), "Seeds": note,
} for m, g, s, ss, sd, sc, r, rs, rd, note in MAIN])
w(df1, "T1_main_results.csv")

# T2 paired
df2 = pd.DataFrame([{
    "A0 vs": k, "LSD Δ": v[0], "LSD Δ std": v[1], "LSD d_z": v[2], "LSD Win%": v[3],
    "DMR Δ": v[4], "DMR d_z": v[5], "DMR Win%": v[6],
    "orig LSD Δ": v[7], "orig LSD d_z": v[8], "orig Win%": v[9],
} for k, v in PAIRED.items()])
w(df2, "T2_paired_stats_3seed.csv")

# T3 track
df3 = pd.DataFrame([{
    "A0 vs": k, "N_groups": 1306, "LSD Δ": v[0], "LSD Δ std": v[1], "LSD d_z": v[2],
    "DMR Δ": v[3], "DMR d_z": v[4],
} for k, v in TRACK.items()])
w(df3, "T3_tracklevel_3seed.csv")

# T4 saturation
df4 = pd.DataFrame([{
    "Config": k, "gain_sat(self)%": v[0], "gain_sat std": v[1], "gain>6%": v[2],
    "gain>6 std": v[3], "fc_hi(self)%": v[4], "fc>16k%": v[5],
} for k, v in SAT.items()])
w(df4, "T4_saturation.csv")

# T5 ac_fitting
df5 = pd.DataFrame([{
    "Config": k, "LSD": v[0], "CI low": v[1], "CI high": v[2], "vs A0(1.095)": v[3],
    "% < JND": v[4], "repr. penalty": v[5], "gain_sat(self)%": v[6], "gain>6%": v[7], "gain_max": v[8],
} for k, v in ACBQ.items()])
w(df5, "T5_ac_fitting.csv")

# T6 option ceilings
df6 = pd.DataFrame([{"Condition": k, "LSD": v} for k, v in OPTION.items()])
w(df6, "T6_biquad_ceiling.csv")

# T7 gap analysis
gaps = [GAP[s][1] - GAP[s][0] for s in GAP]
df7 = pd.DataFrame([{"seed": s, "synth LSD": GAP[s][0], "real LSD": GAP[s][1],
                     "gap (real-synth)": round(GAP[s][1]-GAP[s][0], 4)} for s in GAP]
                   + [{"seed": "mean±std", "synth LSD": round(np.mean([GAP[s][0] for s in GAP]),4),
                       "real LSD": round(np.mean([GAP[s][1] for s in GAP]),4),
                       "gap (real-synth)": f"{np.mean(gaps):.4f}±{np.std(gaps,ddof=1):.4f}"},
                      {"seed": "baseline ±6", "synth LSD": 1.442, "real LSD": 1.941, "gap (real-synth)": GAP_BASE}])
w(df7, "T7_gap_analysis.csv")

# ════════════════════════════════════════════════════════════════════════════
# 3. FIGURES (색맹 + 흑백 대응: 색 + 마커 + 선종 + 해치 병행)
# ════════════════════════════════════════════════════════════════════════════

# F1: A0 gain/fc saturation — ±6 vs ±12 (headline, fc fixed at 16 kHz)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
cfgs = ["±6 / 16k (orig)", "±12 / 16k (new)"]   # freq 16k 고정 채택 → 20k 그룹 제외
x = np.arange(len(cfgs)); width = 0.26
metrics = [("gain saturated (self-bound)", 0, OI["vermillion"], "//"),
           ("gain > 6 dB (vs ±6 ref)", 2, OI["blue"], ".."),
           ("fc near upper bound", 4, OI["green"], "xx")]
for i, (lab, idx, col, hatch) in enumerate(metrics):
    vals = [SAT[c][idx] for c in cfgs]
    errs = [SAT[c][idx+1] if idx in (0, 2) else 0 for c in cfgs]
    bars = ax.bar(x + (i-1)*width, vals, width, yerr=errs, capsize=3, label=lab,
                  color=col, edgecolor="black", linewidth=0.8, hatch=hatch)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 1), textcoords="offset points")
ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=0)
ax.set_ylabel("Fraction of filters (%)"); ax.set_ylim(0, 85)
ax.set_title("A0 PEQ gain saturation: ±6 dB over-constrained, relieved at ±12 dB\n(center frequency fixed at 16 kHz)")
ax.legend(loc="upper left", framealpha=0.95)
ax.annotate("57.8% → 16.6%\n(±6 over-constrained)", xy=(-0.26, 57.8), xytext=(0.30, 66),
            fontsize=9, ha="left", arrowprops=dict(arrowstyle="->", color="black"))
save(fig, "F1_saturation_A0")

# F2: A0 vs AC_Biquad gap shrink (±6 vs ±12)
fig, ax = plt.subplots(figsize=(6.8, 4.0))
archs = ["AC1", "AC2", "AC3"]; x = np.arange(len(archs)); width = 0.34
g6 = [abs(ACBQ[f"{a} ±6"][3]) for a in archs]   # |vs A0| but A0 differs; use absolute LSD diff vs respective A0
# 실제 격차는 (AC_Biquad LSD − A0 LSD); ±6 A0=1.442, ±12 A0=1.095
g6_gap = [1.442 - ACBQ[f"{a} ±6"][0] for a in archs]   # A0(±6) - ACbq(±6) → A0 higher = positive gap
g12_gap = [A0_MEAN - ACBQ[f"{a} ±12"][0] for a in archs]
b1 = ax.bar(x - width/2, g6_gap, width, label="±6 (A0=1.442)", color=OI["grey"],
            edgecolor="black", hatch="", linewidth=0.8)
b2 = ax.bar(x + width/2, g12_gap, width, label="±12 (A0=1.095)", color=OI["orange"],
            edgecolor="black", hatch="//", linewidth=0.8)
for b in list(b1)+list(b2):
    ax.annotate(f"{b.get_height():.2f}", (b.get_x()+b.get_width()/2, b.get_height()),
                ha="center", va="bottom", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f"{a}_Biquad" for a in archs])
ax.set_ylabel("LSD gap = A0 − AC_Biquad (dB)\n(positive = A0 higher LSD)")
ax.set_title("A0 vs AC_Biquad under identical 7-band ±12 constraint:\ngap shrinks ~0.4 → ~0.08 dB")
ax.legend(loc="upper right", framealpha=0.95)
save(fig, "F2_A0_vs_ACbiquad_gap")

# F3: paired effect sizes forest (LSD d_z, 3-seed)
fig, ax = plt.subplots(figsize=(7.0, 4.6))
order = ["E4", "E3", "A3", "A1", "A2", "AC3", "AC2", "AC1"]
y = np.arange(len(order))
new_dz = [PAIRED[k][2] for k in order]; new_err = [abs(PAIRED[k][1]/1.0) for k in order]
orig_dz = [PAIRED[k][8] for k in order]
ax.axvline(0, color=OI["grey"], lw=1.2, ls="-")
ax.scatter(orig_dz, y+0.12, marker="s", s=55, facecolors="none", edgecolors=OI["grey"],
           linewidths=1.4, label="original (±6, 1-seed)", zorder=3)
ax.errorbar(new_dz, y-0.12, xerr=new_err, fmt="o", color=OI["blue"], ms=7, capsize=3,
            label="revised (±12, 3-seed)", zorder=4)
for i, k in enumerate(order):
    ax.annotate(k, (min(new_dz[i], orig_dz[i]), i), xytext=(-8, 0),
                textcoords="offset points", va="center", ha="right", fontsize=9)
ax.set_yticks([]); ax.set_xlabel("paired Cohen's $d_z$ on LSD  (negative = A0 better)")
ax.set_title("A0 vs each model — paired effect size (sample-level, test_synth)")
ax.legend(loc="upper left", framealpha=0.95)   # 좌상단(빈 공간) — 최하단 E4 가림 방지
ax.set_xlim(-6.0, 2.2); ax.set_ylim(-0.6, len(order)-0.4)
ax.text(0.55, 3.5, "A0 worse\n(vs AC)", fontsize=8, color=OI["vermillion"], ha="left")
save(fig, "F3_paired_effectsize_forest")

# F4: A0 vs A2 stability (per-seed)
fig, ax = plt.subplots(figsize=(6.0, 4.0))
a0_seeds = [0.9994, 1.2578, 1.0279]; a2_seeds = [1.0306, 1.1054, 1.8517]
seed_lab = ["s42", "s123", "s7"]
xa = [0]*3; xb = [1]*3
ax.scatter(xa, a0_seeds, marker="o", s=70, color=OI["blue"], label="A0 (per seed)", zorder=3)
ax.scatter(xb, a2_seeds, marker="^", s=70, color=OI["vermillion"], label="A2 (per seed)", zorder=3)
for i in range(3):
    ax.plot([0, 1], [a0_seeds[i], a2_seeds[i]], color=OI["grey"], ls=":", lw=1, zorder=1)
    ax.annotate(seed_lab[i], (1.02, a2_seeds[i]), fontsize=8, va="center")
ax.errorbar(-0.18, np.mean(a0_seeds), yerr=np.std(a0_seeds, ddof=1), fmt="s", color=OI["blue"],
            ms=9, capsize=4, label="A0 mean±std")
ax.errorbar(1.18, np.mean(a2_seeds), yerr=np.std(a2_seeds, ddof=1), fmt="D", color=OI["vermillion"],
            ms=9, capsize=4, label="A2 mean±std")
ax.set_xticks([0, 1]); ax.set_xticklabels(["A0 (proposed)", "A2 (+Pref Loss)"])
ax.set_xlim(-0.5, 1.6); ax.set_ylabel("Synthetic test LSD (dB)")
ax.set_title("A0 vs A2: A0 lower mean and 3× smaller variance\n(σ: 0.12 vs 0.37)")
ax.legend(loc="upper left", framealpha=0.95, fontsize=8)
save(fig, "F4_A0_vs_A2_stability")

# F5: OOD synth→real gap — AC variants 맥락 추가 ("0.70 = 정상 범위")
# AC raw(dense) gap: 이전 AC 평가(ac_gap_eval) synth/real
AC_OOD = {"AC1": (0.8527, 1.5083), "AC2": (0.8600, 1.5057), "AC3": (0.8702, 1.5760)}
AC_GAP = {k: v[1]-v[0] for k, v in AC_OOD.items()}
AC_LO, AC_HI = min(AC_GAP.values()), max(AC_GAP.values())   # 0.646 ~ 0.706

mods = ["A0\n(±6,orig)", "A0\n(±12)", "A2\n(±12)", "AC1", "AC2", "AC3"]
synth = [1.442, 1.095, 1.329, 0.8527, 0.8600, 0.8702]
real  = [1.941, 1.792, 1.981, 1.5083, 1.5057, 1.5760]
synth_e = [0, 0.116, 0.368, 0, 0, 0]; real_e = [0, 0.182, 0.301, 0, 0, 0]
gaps = [r - s for r, s in zip(real, synth)]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.55, 1]})

# ── (좌) synth/real 막대 — real도 함께 개선 = overfit 아님 ─────────────────
x = np.arange(len(mods)); width = 0.36
axL.bar(x - width/2, synth, width, yerr=synth_e, capsize=3, label="synthetic RIR",
        color=OI["skyblue"], edgecolor="black", hatch="", linewidth=0.8)
axL.bar(x + width/2, real, width, yerr=real_e, capsize=3, label="real RIR",
        color=OI["blue"], edgecolor="black", hatch="\\\\", linewidth=0.8)
for i in range(len(mods)):
    axL.annotate(f"{gaps[i]:.2f}", (x[i], max(synth[i], real[i]) + 0.06), ha="center", fontsize=8)
axL.axvline(2.5, color=OI["grey"], ls=":", lw=1)   # proposed/ablation | AC 구분
axL.set_xticks(x); axL.set_xticklabels(mods)
axL.set_ylabel("Dual-target LSD (dB)"); axL.set_ylim(0, 2.65)
axL.set_title("Synthetic vs real LSD (gap labelled on top)\nreal also improves under ±12 → not overfitting")
axL.legend(loc="upper right", framealpha=0.95, fontsize=9)

# ── (우) gap 비교 + AC 정상범위 band ──────────────────────────────────────
gmods = ["A0 (±6, orig)", "A0 (±12)", "A2 (±12)"]; gvals = [0.499, 0.697, 0.652]
gy = np.arange(len(gmods))
axR.axvspan(AC_LO, AC_HI, color=OI["green"], alpha=0.20, zorder=0)
axR.text((AC_LO+AC_HI)/2, -0.52, f"AC variants\ngap range\n{AC_LO:.2f}–{AC_HI:.2f}",
         color=OI["green"], fontsize=8.5, ha="center", va="bottom", fontweight="bold")
bars = axR.barh(gy, gvals, height=0.46, color=[OI["grey"], OI["orange"], OI["purple"]],
                edgecolor="black", linewidth=0.8, zorder=2)
bars[0].set_hatch("");  bars[1].set_hatch("//"); bars[2].set_hatch("..")
for b, v in zip(bars, gvals):
    inside = AC_LO <= v <= AC_HI
    axR.annotate(f"{v:.2f}" + ("  ✓in range" if inside else ""),
                 (v+0.008, b.get_y()+b.get_height()/2), va="center", fontsize=9,
                 color=(OI["green"] if inside else "black"))
axR.set_yticks(gy); axR.set_yticklabels(gmods)
axR.set_ylim(2.7, -1.1); axR.set_xlim(0, 0.86)   # 반전(상단=A0±6) + 하단 band 라벨 공간
axR.set_xlabel("Domain gap = real − synth LSD (dB)")
axR.set_title("A0(±12) gap 0.70 lies within the AC normal range\n(not an overfit anomaly)")
save(fig, "F5_ood_gap")

# F6: biquad ceiling — OptD(7.755) off-scale 제외, 0.8~1.15 확대 (OptA vs OptC 0.1dB 가시화)
fig, ax = plt.subplots(figsize=(7.4, 4.2))
conds = ["dense AC2\n(raw)", "OptA ±6\n(SciPy)", "OptA ±12\n(SciPy)", "OptC ±12\n(AC1)",
         "OptC ±12\n(AC2)", "OptC ±12\n(AC3)"]
vals = [0.860, 1.026, 0.918, 1.010, 1.017, 1.010]
cols = [OI["grey"], OI["orange"], OI["vermillion"], OI["green"], OI["green"], OI["green"]]
hatches = ["", "//", "\\\\", "..", "..", ".."]
b = ax.bar(range(len(conds)), vals, color=cols, edgecolor="black", linewidth=0.9, width=0.62)
for bar, h in zip(b, hatches): bar.set_hatch(h)
for bar, v in zip(b, vals):
    ax.annotate(f"{v:.3f}", (bar.get_x()+bar.get_width()/2, v), ha="center", va="bottom", fontsize=9)
ax.axhline(A0_MEAN, color=OI["blue"], ls="--", lw=1.8, label=f"A0 (±12) = {A0_MEAN}")
# OptA ±12 vs OptC ±12 0.1 dB 차이: 막대 사이 빈 x=2.5 에 세로 양방향 화살표 + 흰 박스 라벨
ax.axhline(0.918, xmin=0.33, xmax=0.92, color=OI["vermillion"], ls=":", lw=1.1)
ax.annotate("", xy=(2.5, 0.918), xytext=(2.5, 1.012),
            arrowprops=dict(arrowstyle="<->", color=OI["vermillion"], lw=1.6))
ax.text(2.5, 0.965, "Δ ≈ 0.1 dB\n(OptA < OptC)", color=OI["vermillion"], fontsize=8.5, ha="center",
        va="center", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=OI["vermillion"], lw=0.8))
ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, fontsize=8.5)
ax.set_ylabel("Dual-target LSD (dB)"); ax.set_ylim(0.80, 1.16)
ax.set_title("Biquad reduction ceiling (AC2_GRU): OptA(0.918) ≠ OptC(~1.01) at ±12\n"
             "[OptD naive 7-pt sampling = 7.755 dB, off-scale]")
ax.legend(loc="upper left", framealpha=0.95)
save(fig, "F6_biquad_ceiling")

print(f"\n완료 → {OUT}")
print(f"  tables: {len(list(TAB.glob('*.csv')))}개  |  figures: {len(list(FIG.glob('*.png')))}개")
