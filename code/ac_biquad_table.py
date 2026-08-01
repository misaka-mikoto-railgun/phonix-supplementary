"""
ac_biquad_table.py — tab:ac_fitting / tab:perceptual (AC_Biquad) 재산출
=======================================================================
- ±6 baseline: 원본 checkpoints/full/AC{n}_*_Biquad.pt → gain_max=6 인스턴스
- ±12 신규    : revision/checkpoints/AC{n}_*_Biquad_g12.pt → gain_max=12 인스턴스
- dense raw   : AC{n}.pt (representation penalty 용)
- A0 기준     : g12 3-seed per-sample LSD 평균배열 (여기서 산출, 하드코딩 없음)

A0 기준선(a0_reference)은 seed 42/123/7 의 per-sample LSD 를 sample 축으로 정렬한
뒤 seed 축으로 평균한 N=3000 배열이다. 그 배열의 평균이 vs_a0 의 기준이고,
같은 배열의 bootstrap_ci(n_boot=2000, seed=42) 가 tab:ac_fitting 의 A0 행 CI 이다.
seed 를 표본처럼 취급하는 pooled(N=9000) bootstrap 은 구간이 좁아지므로 쓰지 않는다.

%<JND 는 여기서 내지 않는다. 그 값은 table7_perceptual.py 한 곳에서만 산출하며
(seed 별 백분율을 구한 뒤 3-seed 평균), 이 스크립트가 다른 방식으로 같은 이름의
값을 함께 내면 같은 열에 두 값이 존재하게 된다.

saturation 두 기준:
  ±6: |gain|>5.8 (자기경계)  /  ±12: |gain|>6.0(±6 고정 기준), |gain|>11.8(자기경계)
"""
import json
from pathlib import Path
import numpy as np
import torch

import train_full as TF
from arch_biquad import BIQUAD_REGISTRY
from run_gain_freq_ablation import evaluate, cname
import cli_paths
import ckpt_io
from dataset_generator_v4_tracklevel import PEQDataset
from ac_fitting_C import bootstrap_ci

_P, _ = cli_paths.parse("AC_Biquad recomputation (Option C)", require=("data_dir", "ckpt_dir", "rev_ckpt_dir"))
DEFAULT_DATA = _P.data_dir      # --data_dir
FULL = _P.ckpt_dir              # --ckpt_dir       pre-revision checkpoints
SAVE = _P.rev_ckpt_dir          # --rev_ckpt_dir   +/-12 dB revision checkpoints
CKE = _P.eval_ckpt_dir          # --eval_ckpt_dir  evaluation staging
OUT = _P.out_dir                # --out_dir        created if missing

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
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
    ckpt_io.load_into(m, ckpt, map_location=device, label=name)
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
    ckpt_io.load_into(m, ckpt, map_location=device, label=reg_key)
    return evaluate(reg_key, None, m.to(device).eval(), ds, device)["lsd_arr"]

# A0 g12 3-seed per-sample LSD 평균 ────────────────────────────────────────────
TF._REGISTRY_TARGET["A0_Proposed"] = "dual"
a0_arrs = []
for s in SEEDS:
    m = registry["A0_Proposed"]["model"]
    ckpt_io.load_into(m, SAVE / f"{cname('g12_f16k', s, 'A0')}.pt",
                      map_location=device, label=f"A0_Proposed s{s}")
    a0_arrs.append(evaluate("A0_Proposed", None, m.to(device).eval(), ds, device)["lsd_arr"])
a0_lsd = np.mean(a0_arrs, axis=0)   # per-sample 3-seed 평균, N=3000
A0_MEAN = float(a0_lsd.mean())
A0_CI = bootstrap_ci(a0_lsd)        # tab:ac_fitting 의 A0 행 CI
print(f"A0 g12 per-sample LSD: 3-seed mean = {A0_MEAN:.4f} "
      f"CI=[{A0_CI[0]:.4f},{A0_CI[1]:.4f}]  (N={a0_lsd.size}, = vs-A0 기준)")

rows = {}
print("\n" + "=" * 104)
print("tab:ac_fitting (Option C) 재산출 — A0 기준 = g12 3-seed mean")
print("=" * 104)
hdr = (f"{'config':>24} {'LSD[95%CI]':>22} {'vsA0':>8} "
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
        penalty = float((lsd - raw).mean())
        sat_self = float((g > (gmax - 0.2)).mean() * 100)
        over6 = float((g > 6.0).mean() * 100)
        key = f"{bname}_{'g6' if gmax==6 else 'g12'}"
        rows[key] = dict(lsd_mean=float(lsd.mean()), ci=[lo, hi], vs_a0=vsA0,
                         penalty=penalty, sat_self=sat_self, gain_over6=over6,
                         gain_max=gmax)
        print(f"  {bname+' '+tag:>22} {lsd.mean():>6.3f}[{lo:.3f},{hi:.3f}] {vsA0:>+8.3f} "
              f"{penalty:>+8.3f} {sat_self:>9.1f} {over6:>7.1f}")

print(f"\nA0 (g12, 3-seed mean) LSD = {A0_MEAN:.3f}  →  'vsA0' = biquad_mean - {A0_MEAN:.3f}")
print("saturation: ±6 자기경계 |g|>5.8 / ±12 자기경계 |g|>11.8, gain>6 = ±6 고정기준 초과(±12에서)")
print("%<JND 는 여기서 내지 않는다 — table7_perceptual.py 단일 경로.")

payload = {
    "a0_reference": {
        "lsd_mean": A0_MEAN,
        "ci": list(A0_CI),
        "n": int(a0_lsd.size),
        "n_boot": 2000,
        "bootstrap_seed": 42,
        "seeds": SEEDS,
        "definition": ("per-sample LSD of A0 g12_f16k averaged over the three "
                       "training seeds (N=3000), then bootstrapped; not the "
                       "pooled N=9000 resample"),
    },
    "configs": rows,
}
out = OUT / "ac_biquad_table.json"
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"\n저장: {out}")
