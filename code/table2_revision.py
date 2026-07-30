"""
table2_revision.py — Table 2 (Main results, SYNTHETIC test) on ±12 revision basis
==================================================================================
1:1 전사용. 추정 없음. 모든 LSD/DMR/CosSim 은 test_synth 에서 직접 계산.
  - metric 정의: experiments_fixed_updated.py 와 동일 (lsd vs dual_target;
    dmr/cossim 은 heard = pred - room_target 대상).
  - A0/A2: gain±12, 3-seed(42,123,7) mean±std (+ pooled bootstrap CI 참고).
  - 그 외(E1–E6, A1, A3, AC1–AC3): gain 무관 → 원본 ckpt 재평가(= revision 값과 동일).
  - Params/RTF: checkpoints/full/results_final.json (측정값, gain 무관).
출력: results/table2_revision_synth.csv  +  콘솔.
"""
import json
from pathlib import Path
import numpy as np
import torch

import train_full as TF
from run_gain_freq_ablation import ORIG_ROOT, HERE, DEFAULT_DATA, cname
from dataset_generator_v4_tracklevel import PEQDataset
from ac_fitting_C import bootstrap_ci

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
FULL = ORIG_ROOT / "checkpoints" / "full"
CKE  = HERE / "ckpt_eval"
SAVE = HERE / "checkpoints"
SEEDS = [42, 123, 7]

# metric (experiments_fixed_updated.py 와 동일 정의)
def m_lsd(pred, tgt):  return np.sqrt(np.mean((pred - tgt) ** 2, axis=-1))
def m_dmr(h, pref):    return np.mean((np.sign(h) == np.sign(pref)).astype(float), axis=-1)
def m_cos(h, pref):
    num = np.sum(h * pref, axis=-1)
    den = np.linalg.norm(h, axis=-1) * np.linalg.norm(pref, axis=-1) + 1e-8
    return num / den

reg = TF.build_registry()
rf = json.load(open(FULL / "results_final.json", encoding="utf-8"))
PARAMS = {k: rf.get(k, {}).get("params") for k in rf}
RTF    = {k: rf.get(k, {}).get("rtf") for k in rf}

ds = PEQDataset(f"{DEFAULT_DATA}/test_synth", device=str(device))
dual = torch.cat([b["dual_target"] for b in ds.iter_batches(10**9, shuffle=False)]).cpu().numpy()
room = torch.cat([b["room_target"] for b in ds.iter_batches(10**9, shuffle=False)]).cpu().numpy()
pref = torch.cat([b["pref_target"] for b in ds.iter_batches(10**9, shuffle=False)]).cpu().numpy()


@torch.no_grad()
def predict(name, ckpt=None):
    # TF.model_forward = train_full/driver 파이프라인 (revision paired/track 와 동일).
    # build_registry 의 baselines.py 클래스 시그니처에 맞음.
    model = reg[name]["model"]
    TF._REGISTRY_TARGET[name] = reg[name]["target"]
    if ckpt is not None and Path(ckpt).exists():
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck, strict=False)
    model.to(device).eval()
    preds = [TF.model_forward(name, model, b)["pred_response_db"].cpu().numpy()
             for b in ds.iter_batches(512, shuffle=False)]
    return np.concatenate(preds)


def arrs(pred):
    h = pred - room
    return m_lsd(pred, dual), m_dmr(h, pref), m_cos(h, pref)


# (display, registry name, ckpt) — E1/E2/E6 는 ckpt 없음(analytical)
ROWS = [
    ("E1 No Processing",      "E1_NoEQ",          None),
    ("E2 Static Mode EQ",     "E2_StaticEQ",      None),
    ("E3 Nercessian MLP",     "E3_Nercessian",    CKE/"E3_Nercessian.pt"),
    ("E4 Pepe CNN",           "E4_Pepe",          CKE/"E4_Pepe.pt"),
    ("E5 Sequential",         "E5_Sequential",    CKE/"E5_Sequential.pt"),
    ("E6 DSP Analytical",     "E6_DSP",           None),
    ("AC1 TCN+BiLSTM",        "AC1_BiLSTM",       CKE/"AC1_BiLSTM.pt"),
    ("AC2 TCN+GRU",           "AC2_GRU",          CKE/"AC2_GRU.pt"),
    ("AC3 TCN+Conformer",     "AC3_Conformer",    CKE/"AC3_Conformer.pt"),
    ("A1 w/o Room Input",     "A1_NoRoomInput",   CKE/"A1_NoRoomInput.pt"),
    ("A3 w/o Pref Input",     "A3_NoPrefInput",   CKE/"A3_NoPrefInput.pt"),
]

out = []
for disp, name, ckpt in ROWS:
    lsd_a, dmr_a, cos_a = arrs(predict(name, ckpt))
    lo, hi = bootstrap_ci(lsd_a)
    out.append(dict(model=disp, lsd=lsd_a.mean(), ci_lo=lo, ci_hi=hi, lsd_std="",
                    dmr=dmr_a.mean(), cossim=cos_a.mean(),
                    rtf=RTF.get(name), params=PARAMS.get(name), seeds="1(orig)"))

# A0 / A2 — gain±12, 3-seed
for disp, var in [("A0 Proposed (±12)", "A0"), ("A2 with Pref Loss (±12)", "A2")]:
    regname = "A0_Proposed" if var == "A0" else "A2_withPrefLoss"
    lsd_means, dmrs, coss, lsd_pool = [], [], [], []
    for s in SEEDS:
        p = predict(regname, SAVE / f"{cname('g12_f16k', s, var)}.pt")
        la, da, ca = arrs(p)
        lsd_means.append(la.mean()); dmrs.append(da.mean()); coss.append(ca.mean()); lsd_pool.append(la)
    lsd_pool = np.concatenate(lsd_pool)
    lo, hi = bootstrap_ci(lsd_pool)   # pooled 3-seed bootstrap (참고)
    out.append(dict(model=disp, lsd=float(np.mean(lsd_means)), ci_lo=lo, ci_hi=hi,
                    lsd_std=float(np.std(lsd_means, ddof=1)),
                    dmr=float(np.mean(dmrs)), cossim=float(np.mean(coss)),
                    rtf=RTF.get(regname), params=PARAMS.get(regname), seeds="3(42,123,7)"))

import csv
csvp = HERE / "results" / "table2_revision_synth.csv"
with open(csvp, "w", newline="", encoding="utf-8-sig") as f:
    wtr = csv.DictWriter(f, fieldnames=["model", "lsd", "lsd_std", "ci_lo", "ci_hi",
                                        "dmr", "cossim", "rtf", "params", "seeds"])
    wtr.writeheader()
    for r in out:
        wtr.writerow({**r, "lsd": f"{r['lsd']:.3f}", "ci_lo": f"{r['ci_lo']:.3f}",
                      "ci_hi": f"{r['ci_hi']:.3f}", "dmr": f"{r['dmr']:.3f}",
                      "cossim": f"{r['cossim']:.3f}",
                      "lsd_std": (f"{r['lsd_std']:.3f}" if r['lsd_std'] != "" else "")})

print("=" * 104)
print("Table 2 (MAIN, SYNTHETIC test_synth, N=3000) — gain ±12 revision basis")
print("LSD = LSD_dual (predicted full response vs dual target). DMR/CosSim on heard=pred−room.")
print("=" * 104)
hdr = f"{'Model':24} {'LSD_dual':>9} {'95% CI':>16} {'±std':>6} {'DMR':>6} {'CosSim':>7} {'RTF†':>7} {'Params':>9} {'seeds':>12}"
print(hdr); print("-" * len(hdr))
for r in out:
    std = f"±{r['lsd_std']:.3f}" if r['lsd_std'] != "" else "  —  "
    pr = f"{r['params']:,}" if r['params'] else "0"
    print(f"{r['model']:24} {r['lsd']:>9.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}] {std:>6} "
          f"{r['dmr']:>6.3f} {r['cossim']:>7.3f} {str(r['rtf']):>7} {pr:>9} {r['seeds']:>12}")
print(f"\n저장: {csvp}")
print("출처: predictions=this revision eval(test_synth); Params/RTF=checkpoints/full/results_final.json(gain무관 측정값)")
