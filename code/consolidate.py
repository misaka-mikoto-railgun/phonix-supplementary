"""
consolidate.py — consolidated tables and figures
================================================
Writes to paper_outputs/{tables,figures}:
  - 모든 실험 결과를 CSV 테이블로
  - 색맹(Okabe–Ito) + 흑백인쇄 대응(마커/선종/해치 병행) figure 로
정리. 라벨 겹침/잘림 방지(constrained_layout + bbox_inches='tight' + 회전/여백).

값 출처: gain_freq_summary_*, paired_stats_3seed, track_stats_3seed,
         ac_biquad_table, param_dist_gain_freq, gap_analysis (revision/results).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_ap = argparse.ArgumentParser(description="consolidated tables and figures")
_ap.add_argument("--results_dir", default=str(HERE / "results"),
                 help="where the generators wrote their output (read first)")
_ap.add_argument("--out_dir", default=str(HERE / "paper_outputs"),
                 help="destination for tables/ and figures/ (created if missing)")
_args = _ap.parse_args()

RES = Path(_args.results_dir)
OUT = Path(_args.out_dir)
TAB = OUT / "tables"; FIG = OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)


def _inp(name, *subdirs):
    """생성기가 방금 쓴 results/ 를 먼저 보고, 없으면 리포에 커밋된 위치를 쓴다."""
    for d in (RES, *(ROOT / s for s in subdirs)):
        if (d / name).is_file():
            return d / name
    raise FileNotFoundError(f"{name} not found in {RES} or {list(subdirs)}")

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
# Paired 3-seed: target -> (LSDΔ,LSDΔstd,LSDdz,LSDwin, DMRΔ,DMRdz,DMRwin, origLSDΔ,origLSDdz,origWin)
# 부호 규약: paired_stats.py 는 Δ = mean(A0 − base). LSD 는 음수, DMR 은 양수가
# A0 우세이고 Win% 는 각각 P(A0<base), P(A0>base) 이다. 'orig' 3 원소는 개정
# 이전(±6) 단일 seed 값으로, 같은 규약이지만 seed 산포가 없다.
_ps = json.loads(_inp("paired_stats_3seed_test_synth.json", "results_json").read_text(encoding="utf-8"))
PAIRED = {k.split("_")[0]: (round(v["lsd"]["delta"][0], 3), round(v["lsd"]["delta"][1], 3),
                            round(v["lsd"]["d_z"][0], 2), round(v["lsd"]["win"][0] * 100, 1),
                            round(v["dmr"]["delta"][0], 3), round(v["dmr"]["d_z"][0], 2),
                            round(v["dmr"]["win"][0] * 100, 1),
                            round(v["lsd"]["orig"][0], 3), round(v["lsd"]["orig"][1], 3),
                            round(v["lsd"]["orig"][2] * 100, 1))
          for k, v in _ps.items()}
# Track 3-seed: target -> (LSDΔ,LSDΔstd,LSDdz, DMRΔ,DMRdz)
# 부호 규약: track_stats.py 는 mean_diff = A0 − comparison 으로 낸다. LSD 가 음수면
# A0 의 per-track LSD 가 낮다(=A0 우세), DMR 은 양수면 A0 우세. results/track_level/
# 쪽은 반대 규약(comparison − A0)이므로 여기서 섞어 읽지 않는다.
_tr = json.loads(_inp("track_stats_3seed_test_synth.json", "results_json").read_text(encoding="utf-8"))
TRACK = {k.split("_")[0]: (round(v["lsd_delta"][0], 3), round(v["lsd_delta"][1], 3),
                           round(v["lsd_dz"][0], 2), round(v["dmr_delta"][0], 3),
                           round(v["dmr_dz"][0], 2), v["n_groups"])
         for k, v in _tr.items()}
# Saturation (A0, param_dist two criteria): config -> (gain_sat, gain_sat_std, gain>6, gain>6_std, fc_hi, fc>16k)
# saturation: param_dist_gain_freq.py 의 test_synth 산출물에서 읽는다.
# 각 항목은 [seed 평균, seed 표준편차(ddof=1), n] 의 비율이므로 100 을 곱한다.
_pd = json.loads(_inp("param_dist_gain_freq_test_synth.json",
                      "results_json").read_text(encoding="utf-8"))
SAT = {disp: (round(_pd[k]["gain_sat_self"][0] * 100, 1), round(_pd[k]["gain_sat_self"][1] * 100, 1),
              round(_pd[k]["gain_over6"][0] * 100, 1), round(_pd[k]["gain_over6"][1] * 100, 1),
              round(_pd[k]["fc_hi_self"][0] * 100, 1), round(_pd[k]["fc_over16k"][0] * 100, 1))
       for disp, k in [("±6 / 16k (orig)", "g6_f16k"), ("±12 / 16k (new)", "g12_f16k"),
                       ("±12 / 20k", "g12_f20k")]}
# AC_Biquad: name -> (lsd, ci_lo, ci_hi, vsA0, _unused, penalty, sat_self, gain_over6, gmax)
# 인덱스 4 는 예전에 %<JND 를 담았으나 지금은 쓰지 않는다. 그 값은 아래
# ACBQ_JND 가 table7_perceptual.csv 에서 읽는다 — 산출 경로를 하나로 두기 위함.
_bq = json.loads(_inp("ac_biquad_table.json", "results_json").read_text(encoding="utf-8"))
_bqc = _bq["configs"]
ACBQ = {f"AC{i} ±{g}": (round(c["lsd_mean"], 3), round(c["ci"][0], 3), round(c["ci"][1], 3),
                        round(c["vs_a0"], 3), None, round(c["penalty"], 3),
                        round(c["sat_self"], 1), round(c["gain_over6"], 1), int(c["gain_max"]))
        for i, _n in enumerate(["AC1_BiLSTM_Biquad", "AC2_GRU_Biquad", "AC3_Conformer_Biquad"], 1)
        for g, c in [(6, _bqc[f"{_n}_g6"]), (12, _bqc[f"{_n}_g12"])]}
# %<JND 는 |LSD_model − LSD_A0| < 0.5 dB 이고 A0 는 ±12 3-seed 기준선이다.
# ±6 모델을 ±12 기준선에 대는 것은 교차 비교이므로 그 열은 ±12 행에만 정의된다.
# ±6 행 자체는 유지한다 — gain_sat / gain>6 열이 §1.3 의 근거이기 때문이다.
_pj = pd.read_csv(_inp("table7_perceptual.csv", "tables"), encoding="utf-8-sig")
_pj = dict(zip(_pj["Model"], _pj["pct_below_JND"]))
ACBQ_JND = {f"AC{i} ±12": round(float(_pj[f"AC{i}_Biquad"]), 1) for i in (1, 2, 3)}
# A0(±12) 3-seed 기준선. ac_biquad_table.py 가 per-sample LSD 를 seed 축으로 평균한
# 배열에서 낸 값이며, 표·그림에 인쇄되는 자리수(3)로 맞춘다.
A0_MEAN = round(_bq["a0_reference"]["lsd_mean"], 3)

# ±6 기준선(A0, 단일 seed 42)은 3-seed 표에 없으므로 seed 요약에서 읽는다.
_g6s = json.loads(_inp("gain_freq_summary_A0_test_synth.json", "results_json").read_text(encoding="utf-8"))
_g6r = json.loads(_inp("gain_freq_summary_A0_test_real.json", "results_json").read_text(encoding="utf-8"))
A0_G6 = (_g6s["g6_f16k"]["lsd"][0], _g6r["g6_f16k"]["lsd"][0])   # (synth, real)
A0_G6_SYNTH = round(A0_G6[0], 3)

# ════════════════════════════════════════════════════════════════════════════
# 2. CSV 테이블
# ════════════════════════════════════════════════════════════════════════════
def w(df, name):
    df.to_csv(TAB / name, index=False, encoding="utf-8-sig")
    print(f"  csv: {name}")

# T2 paired
df2 = pd.DataFrame([{
    "A0 vs": k, "LSD Δ": v[0], "LSD Δ std": v[1], "LSD d_z": v[2], "LSD Win%": v[3],
    "DMR Δ": v[4], "DMR d_z": v[5], "DMR Win%": v[6],
    "orig LSD Δ": v[7], "orig LSD d_z": v[8], "orig Win%": v[9],
} for k, v in PAIRED.items()])
w(df2, "T2_paired_stats_3seed.csv")

# T3 track
df3 = pd.DataFrame([{
    "A0 vs": k, "N_groups": v[5], "LSD Δ": v[0], "LSD Δ std": v[1], "LSD d_z": v[2],
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
    "Config": k, "LSD": v[0], "CI low": v[1], "CI high": v[2], f"vs A0({A0_MEAN})": v[3],
    "% < JND": ACBQ_JND.get(k, ""), "repr. penalty": v[5],
    "gain_sat(self)%": v[6], "gain>6%": v[7], "gain_max": v[8],
} for k, v in ACBQ.items()])
w(df5, "T5_ac_fitting.csv")

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
ax.annotate(f"{SAT[cfgs[0]][0]}% → {SAT[cfgs[1]][0]}%\n(±6 over-constrained)",
            xy=(-0.26, SAT[cfgs[0]][0]), xytext=(0.30, 66),
            fontsize=9, ha="left", arrowprops=dict(arrowstyle="->", color="black"))
save(fig, "F1_saturation_A0")

# F2: A0 vs AC_Biquad gap shrink (±6 vs ±12)
fig, ax = plt.subplots(figsize=(6.8, 4.0))
archs = ["AC1", "AC2", "AC3"]; x = np.arange(len(archs)); width = 0.34
g6 = [abs(ACBQ[f"{a} ±6"][3]) for a in archs]   # |vs A0| but A0 differs; use absolute LSD diff vs respective A0
# 실제 격차는 (AC_Biquad LSD − A0 LSD). ±6 기준선은 단일 seed 요약에서 읽는다.
g6_gap = [A0_G6_SYNTH - ACBQ[f"{a} ±6"][0] for a in archs]   # A0(±6) - ACbq(±6) → A0 higher = positive gap
g12_gap = [A0_MEAN - ACBQ[f"{a} ±12"][0] for a in archs]
b1 = ax.bar(x - width/2, g6_gap, width, label=f"±6 (A0={A0_G6_SYNTH})", color=OI["grey"],
            edgecolor="black", hatch="", linewidth=0.8)
b2 = ax.bar(x + width/2, g12_gap, width, label=f"±12 (A0={A0_MEAN})", color=OI["orange"],
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
# seed 별 LSD 는 요약 JSON 의 lsd_per_seed 에서 읽는다(seeds 순서와 짝).
_a2s = json.loads(_inp("gain_freq_summary_A2_test_synth.json", "results_json").read_text(encoding="utf-8"))
a0_seeds = _g6s["g12_f16k"]["lsd_per_seed"]; a2_seeds = _a2s["g12_f16k"]["lsd_per_seed"]
seed_lab = [f"s{s}" for s in _g6s["g12_f16k"]["seeds"]]
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
# synth/real/gap 과 seed 표준편차는 모두 table5_ood.csv 에서 온다. gain_freq_summary_*
# 는 seed 별 요약이라 A2 의 경우 폐기된 s7 run-1 값을 아직 담고 있다 — 여기서 쓰지 않는다.
_ood = pd.read_csv(_inp("table5_ood.csv", "tables"), encoding="utf-8-sig").set_index("model")
AC_GAP = {f"AC{i}": float(_ood.loc[n, "gap"])
          for i, n in enumerate(["AC1 BiLSTM", "AC2 GRU", "AC3 Conformer"], 1)}
AC_LO, AC_HI = min(AC_GAP.values()), max(AC_GAP.values())   # 0.646 ~ 0.706

# ±6 기준선은 3-seed 표에 없으므로 단일 seed 요약에서 읽는다(A0 만 해당).

_rows = ["A0 Proposed ±12", "A2 +PrefLoss ±12", "AC1 BiLSTM", "AC2 GRU", "AC3 Conformer"]
mods = ["A0\n(±6,orig)", "A0\n(±12)", "A2\n(±12)", "AC1", "AC2", "AC3"]
synth = [A0_G6[0]] + [float(_ood.loc[r, "synth"]) for r in _rows]
real = [A0_G6[1]] + [float(_ood.loc[r, "real"]) for r in _rows]
synth_e = [0.0] + [float(_ood.loc[r, "synth_sd"]) for r in _rows]
real_e = [0.0] + [float(_ood.loc[r, "real_sd"]) for r in _rows]
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
gmods = ["A0 (±6, orig)", "A0 (±12)", "A2 (±12)"]
gvals = [A0_G6[1] - A0_G6[0], float(_ood.loc["A0 Proposed ±12", "gap"]),
         float(_ood.loc["A2 +PrefLoss ±12", "gap"])]
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

# F6: biquad ceiling — OptD 는 off-scale 이라 제외, 0.8~1.15 확대 (OptA vs OptC 0.1dB 가시화)
fig, ax = plt.subplots(figsize=(7.4, 4.2))
conds = ["dense AC2\n(raw)", "OptA ±6\n(SciPy)", "OptA ±12\n(SciPy)", "OptC ±12\n(AC1)",
         "OptC ±12\n(AC2)", "OptC ±12\n(AC3)"]
_t6 = pd.read_csv(_inp("table6_biquad.csv", "tables"), encoding="utf-8-sig").set_index("Configuration")
vals = [float(_t6.loc[r, "LSD"]) for r in
        ["dense AC2 (raw, reference)", "Option A SciPy fit ±6 (AC2, orig)",
         "Option A SciPy fit ±12 (AC2)", "AC1_BiLSTM Biquad (Option C, ±12)",
         "AC2_GRU Biquad (Option C, ±12)", "AC3_Conformer Biquad (Option C, ±12)"]]
cols = [OI["grey"], OI["orange"], OI["vermillion"], OI["green"], OI["green"], OI["green"]]
hatches = ["", "//", "\\\\", "..", "..", ".."]
b = ax.bar(range(len(conds)), vals, color=cols, edgecolor="black", linewidth=0.9, width=0.62)
for bar, h in zip(b, hatches): bar.set_hatch(h)
for bar, v in zip(b, vals):
    ax.annotate(f"{v:.3f}", (bar.get_x()+bar.get_width()/2, v), ha="center", va="bottom", fontsize=9)
ax.axhline(A0_MEAN, color=OI["blue"], ls="--", lw=1.8, label=f"A0 (±12) = {A0_MEAN}")
# OptA ±12 vs OptC ±12 0.1 dB 차이: 막대 사이 빈 x=2.5 에 세로 양방향 화살표 + 흰 박스 라벨
ax.axhline(vals[2], xmin=0.33, xmax=0.92, color=OI["vermillion"], ls=":", lw=1.1)
_optc = float(np.mean(vals[3:]))
ax.annotate("", xy=(2.5, vals[2]), xytext=(2.5, _optc),
            arrowprops=dict(arrowstyle="<->", color=OI["vermillion"], lw=1.6))
ax.text(2.5, (vals[2] + _optc) / 2, "Δ ≈ 0.1 dB\n(OptA < OptC)", color=OI["vermillion"], fontsize=8.5, ha="center",
        va="center", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=OI["vermillion"], lw=0.8))
ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, fontsize=8.5)
ax.set_ylabel("Dual-target LSD (dB)"); ax.set_ylim(0.80, 1.16)
_optd = float(_t6.loc["Option D naive 7-pt (AC2)", "LSD"])
ax.set_title(f"Biquad reduction ceiling (AC2_GRU): OptA({vals[2]:.3f}) ≠ "
             f"OptC(~{_optc:.2f}) at ±12\n"
             f"[OptD naive 7-pt sampling = {_optd:.3f} dB, off-scale]")
ax.legend(loc="upper left", framealpha=0.95)
save(fig, "F6_biquad_ceiling")

print(f"\n완료 → {OUT}")
print(f"  tables: {len(list(TAB.glob('*.csv')))}개  |  figures: {len(list(FIG.glob('*.png')))}개")
