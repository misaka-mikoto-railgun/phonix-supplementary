"""
ac_biquad_table.py — tab:ac_fitting / tab:perceptual (AC_Biquad) 재산출
=======================================================================
- ±6 baseline: 원본 checkpoints/full/AC{n}_*_Biquad.pt → gain_max=6 인스턴스
- ±12 신규    : revision/checkpoints/AC{n}_*_Biquad_g12.pt → gain_max=12 인스턴스
- dense raw   : AC{n}.pt (representation penalty 용)
- A0 기준     : g12 3-seed per-sample LSD 평균 (vs A0 mean=1.095 결정 반영)

%<JND 정의(원본 동일, ac_fitting_C.py:307 / perceptual_proxy.py:171):
  per-sample |AC_Biquad_LSD - A0_LSD| < 0.5 dB 비율
saturation 두 기준:
  ±6: |gain|>5.8 (자기경계)  /  ±12: |gain|>6.0(±6 고정 기준), |gain|>11.8(자기경계)
"""
import json
from pathlib import Path
import numpy as np
import torch

import train_full as TF
from arch_biquad import BIQUAD_REGISTRY
from run_gain_freq_ablation import ORIG_ROOT, HERE, DEFAULT_DATA, evaluate, cname
from dataset_generator_v4_tracklevel import PEQDataset
from ac_fitting_C import bootstrap_ci, JND_THRESHOLD

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
FULL = ORIG_ROOT / "checkpoints" / "full"
SAVE = HERE / "checkpoints"
A0_MEAN = 1.095   # 결정: vs A0 는 3-seed mean
SEEDS = [42, 123, 7]

ARCH = [  # (biquad model name, ±6 ckpt, dense raw ckpt)
    ("AC1_BiLSTM_Biquad",   "AC1_BiLSTM_Biquad.pt",   "AC1_BiLSTM",    "AC1_BiLSTM.pt"),
    ("AC2_GRU_Biquad",      "AC2_GRU_Biquad.pt",      "AC2_GRU",       "AC2_GRU.pt"),
    ("AC3_Conformer_Biquad","AC3_Conformer_Biquad.pt","AC3_Conformer", "AC3_Conformer.pt"),
]

registry = TF.build_registry()
ds = PEQDataset(f"{DEFAULT_DATA}/test_synth", device=str(device))
dual = torch.cat([b["dual_target"] for b in ds.iter_batches(100000, shuffle=False)]).cpu().numpy()


@torch.no_grad()
def biquad_lsd_gain(name, gain_max, ckpt):
    """AC_Biquad forward → per-sample LSD(vs dual) + |gain| 전체."""
    m = BIQUAD_REGISTRY[name](gain_max=gain_max).to(device).eval()
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    m.load_state_dict(state, strict=False)
    preds, gains = [], []
    for b in ds.iter_batches(512, shuffle=False):
        out = m(b["features"], b["room_response"], b["mode_id"], b["band_gains"])
        preds.append(out["pred_response_db"].cpu().numpy())
        gains.append(out["gain"].abs().cpu().numpy())
    pred = np.concatenate(preds)
    lsd = np.sqrt(((pred - dual) ** 2).mean(axis=-1))
    return lsd, np.concatenate(gains)


@torch.no_grad()
def dense_lsd(reg_key, ckpt):
    m = registry[reg_key]["model"]
    TF._REGISTRY_TARGET[reg_key] = "dual"
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    m.load_state_dict(state, strict=False)
    return evaluate(reg_key, None, m.to(device).eval(), ds, device)["lsd_arr"]


# A0 g12 3-seed per-sample LSD 평균 ────────────────────────────────────────────
TF._REGISTRY_TARGET["A0_Proposed"] = "dual"
a0_arrs = []
for s in SEEDS:
    m = registry["A0_Proposed"]["model"]
    ck = torch.load(SAVE / f"{cname('g12_f16k', s, 'A0')}.pt", map_location=device, weights_only=False)
    m.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
    a0_arrs.append(evaluate("A0_Proposed", None, m.to(device).eval(), ds, device)["lsd_arr"])
a0_lsd = np.mean(a0_arrs, axis=0)   # per-sample 3-seed 평균
print(f"A0 g12 per-sample LSD: 3-seed mean = {a0_lsd.mean():.4f} (= vs-A0 기준)")

rows = {}
print("\n" + "=" * 104)
print("tab:ac_fitting (Option C) 재산출 — A0 기준 = g12 3-seed mean")
print("=" * 104)
hdr = (f"{'config':>24} {'LSD[95%CI]':>22} {'vsA0':>8} {'%<JND':>7} "
       f"{'penalty':>8} {'sat(self)':>10} {'gain>6':>8}")
print(hdr); print("-" * len(hdr))
for bname, b6, dname, draw in ARCH:
    raw = dense_lsd(dname, FULL / draw)
    for tag, gmax, ckpt, satlab in [("±6(orig)", 6.0, FULL / b6, "|g|>5.8"),
                                     ("±12(new)", 12.0, SAVE / f"{bname}_g12.pt", "|g|>11.8")]:
        if not Path(ckpt).exists():
            print(f"  [{bname} {tag}] ckpt 없음 — 스킵 ({ckpt})"); continue
        lsd, g = biquad_lsd_gain(bname, gmax, ckpt)
        lo, hi = bootstrap_ci(lsd)
        vsA0 = lsd.mean() - A0_MEAN
        jnd = float((np.abs(lsd - a0_lsd) < JND_THRESHOLD).mean() * 100)
        penalty = float((lsd - raw).mean())
        sat_self = float((g > (gmax - 0.2)).mean() * 100)
        over6 = float((g > 6.0).mean() * 100)
        key = f"{bname}_{'g6' if gmax==6 else 'g12'}"
        rows[key] = dict(lsd_mean=float(lsd.mean()), ci=[lo, hi], vs_a0=vsA0,
                         pct_jnd=jnd, penalty=penalty, sat_self=sat_self, gain_over6=over6,
                         gain_max=gmax)
        print(f"  {bname+' '+tag:>22} {lsd.mean():>6.3f}[{lo:.3f},{hi:.3f}] {vsA0:>+8.3f} "
              f"{jnd:>6.1f} {penalty:>+8.3f} {sat_self:>9.1f} {over6:>7.1f}")

print(f"\nA0 (g12, 3-seed mean) LSD = {A0_MEAN:.3f}  →  'vsA0' = biquad_mean - {A0_MEAN}")
print("saturation: ±6 자기경계 |g|>5.8 / ±12 자기경계 |g|>11.8, gain>6 = ±6 고정기준 초과(±12에서)")
out = HERE / "results" / "ac_biquad_table.json"
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"\n저장: {out}")
