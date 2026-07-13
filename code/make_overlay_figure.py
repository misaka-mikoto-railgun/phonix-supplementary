"""
make_overlay_figure.py — R2-6 frequency-response overlay (Option A, 4 curves)
=============================================================================
results/fig_overlay_sample.json 을 읽어 publication-grade 벡터 PDF/PNG 생성.
곡선: H_room=-T_room(회색) · R_hat(주황) · T_pref(검정 점선) · H_hat(파랑 굵게, z-top).
±0.5 dB JND 음영(T_pref 둘레). log-x(20-20k), 0 dB 기준선, LSD 코너박스. Okabe-Ito.
guard: H_hat == R_hat - T_room.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 토글 ─────────────────────────────────────────────────────────────────────
SHOW_BANDS = False     # 7밴드 underlay (clutter 방지 기본 off)
SPAN_2COL  = False     # True=양단(7in), False=단일(3.5in)

HERE = Path(__file__).resolve().parent
J = json.load(open(HERE/"results"/"fig_overlay_sample.json", encoding="utf-8"))
f  = np.array(J["freqs_Hz"]); Tr = np.array(J["T_room"]); Tp = np.array(J["T_pref"])
R  = np.array(J["R_hat"]);    H  = np.array(J["H_hat"]);  m = J["meta"]
Hroom = -Tr
assert np.allclose(H, R - Tr, atol=1e-4), "guard 실패: H_hat != R_hat - T_room"

OI = dict(gray="#999999", orange="#E69F00", black="#000000", blue="#0072B2", green="#009E73")
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,   # 편집가능 텍스트
    "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "savefig.bbox": "tight", "figure.constrained_layout.use": True,
})
fig, ax = plt.subplots(figsize=((7.0, 3.4) if SPAN_2COL else (3.6, 3.0)))

# JND 음영 (T_pref ±0.5)
ax.fill_between(f, Tp-0.5, Tp+0.5, color=OI["green"], alpha=0.16, lw=0,
                label=r"$\pm0.5$ dB JND", zorder=1)
ax.axhline(0, color="0.4", lw=0.7, zorder=0)

# 7밴드 underlay (옵션)
if SHOW_BANDS and "bands_band0to6" in J:
    for b in J["bands_band0to6"]:
        ax.plot(f, b, color=OI["gray"], lw=0.5, alpha=0.35, zorder=1)

# 4곡선 (z-order: H_hat 최상위)
ax.plot(f, Hroom, color=OI["gray"],   lw=1.4, ls="-",  label=r"$H_\mathrm{room}=-T_\mathrm{room}$", zorder=2)
ax.plot(f, R,     color=OI["orange"], lw=1.6, ls="-",  label=r"$\hat{R}$ (applied EQ)", zorder=3)
ax.plot(f, Tp,    color=OI["black"],  lw=1.5, ls="--", label=r"$T_\mathrm{pref}$ (target)", zorder=4)
ax.plot(f, H,     color=OI["blue"],   lw=2.4, ls="-",  label=r"$\hat{H}$ (heard)", zorder=5)

ax.set_xscale("log"); ax.set_xlim(20, 20000)
ax.set_xticks([20,50,100,200,500,1000,2000,5000,10000,20000])
ax.set_xticklabels(["20","50","100","200","500","1k","2k","5k","10k","20k"])
ymin = min(Hroom.min(), R.min(), Tp.min(), H.min()); ymax = max(Hroom.max(), R.max(), Tp.max(), H.max())
pad = 0.08*(ymax-ymin); ax.set_ylim(ymin-pad, ymax+pad)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude (dB)")

# LSD 코너 박스
ax.text(0.025, 0.04,
        f"$\\mathrm{{LSD}}_\\mathrm{{dual}}={m['LSD_dual']:.2f}$ dB\n$\\mathrm{{LSD}}_\\mathrm{{pref}}={m['LSD_pref']:.2f}$ dB",
        transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.6, alpha=0.92))

ax.legend(loc="upper right", framealpha=0.92, ncol=1, handlelength=1.7,
          borderpad=0.35, labelspacing=0.3)

out = HERE/"results"/"paper_outputs"/"figures"/"fig_response_overlay"
out.parent.mkdir(parents=True, exist_ok=True)
for ext in ("pdf","png"):
    fig.savefig(f"{out}.{ext}", dpi=300)
plt.close(fig)
print(f"saved: {out}.pdf / .png  (SHOW_BANDS={SHOW_BANDS}, SPAN_2COL={SPAN_2COL})")
print(f"sample={m['sample_idx']} room={m['room_id']} preset={m['preset_id_mode']} "
      f"LSD_dual={m['LSD_dual']:.2f} LSD_pref={m['LSD_pref']:.2f}")
