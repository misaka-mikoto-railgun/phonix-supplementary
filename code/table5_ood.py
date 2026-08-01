"""
table5_ood.py — Table 5 (OOD) + §4.4, ±12·train_full 기준
=========================================================
추정 없음. synth(test_synth, N=3000) 와 real(test_real, BUT+OpenAIR, N=2000) 직접 계산.
metric: lsd vs dual_target; dmr 은 heard=pred-room vs pref. A0/A2=gain±12 3-seed; 나머지 single.
A1 = train_full 경로(features_clean) 기준으로 real 값 산출.
source-wise: room_id → rir_map.json 경로로 BUT / OpenAIR 분할.
출력: 모델별 synth LSD, real LSD, real DMR, gap(real-synth), BUT-LSD, OpenAIR-LSD.
"""
import json
import numpy as np
import torch

import train_full as TF
from run_gain_freq_ablation import cname
import cli_paths
import ckpt_io
from dataset_generator_v4_tracklevel import PEQDataset
from export_track_level_predictions import TrackAwarePEQDataset

_P, _ = cli_paths.parse("Table 3 (out-of-distribution)", require=("data_dir", "ckpt_dir", "rev_ckpt_dir", "eval_ckpt_dir"))
DEFAULT_DATA = _P.data_dir      # --data_dir
FULL = _P.ckpt_dir              # --ckpt_dir       pre-revision checkpoints
SAVE = _P.rev_ckpt_dir          # --rev_ckpt_dir   +/-12 dB revision checkpoints
CKE = _P.eval_ckpt_dir          # --eval_ckpt_dir  evaluation staging
OUT = _P.out_dir                # --out_dir        created if missing

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
SEEDS = [42, 123, 7]
reg = TF.build_registry()

# ── 데이터 ───────────────────────────────────────────────────────────────────
ds_s = PEQDataset(f"{DEFAULT_DATA}/test_synth", device=str(device))
dual_s = torch.cat([b["dual_target"] for b in ds_s.iter_batches(10**9, False)]).cpu().numpy()
ds_r = TrackAwarePEQDataset(f"{DEFAULT_DATA}/test_real", device=str(device))
dual_r = ds_r.data["dual_target"].cpu().numpy(); room_r = ds_r.data["room_target"].cpu().numpy()
pref_r = ds_r.data["pref_target"].cpu().numpy(); rid = ds_r.data["room_id"].cpu().numpy()

# room_id → source (BUT / OpenAIR)
rmap = json.load(open(f"{DEFAULT_DATA}/test_real/rir_map.json", encoding="utf-8"))
id2path = {r["id"]: r["value"].lower() for r in rmap["rirs"]}
def src(i):
    p = id2path.get(int(i), "")
    return "BUT" if "but_reverb" in p or "but_" in p else ("OpenAIR" if "openair" in p or "open_air" in p else "other")
source = np.array([src(i) for i in rid])
is_but = source == "BUT"; is_oa = source == "OpenAIR"
print(f"real N={len(rid)}  BUT={is_but.sum()}  OpenAIR={is_oa.sum()}  other={(~is_but&~is_oa).sum()}")

def lsd(p, t): return np.sqrt(np.mean((p - t) ** 2, -1))
def dmr(h, q): return np.mean((np.sign(h) == np.sign(q)).astype(float), -1)

@torch.no_grad()
def predict(name, ckpt, dataobj):
    m = reg[name]["model"]; TF._REGISTRY_TARGET[name] = reg[name]["target"]
    # E1/E2/E6 are analytical and carry no checkpoint; anything else must load.
    ckpt_io.load_into(m, ckpt, map_location=device, label=name)
    m.to(device).eval()
    return np.concatenate([TF.model_forward(name, m, b)["pred_response_db"].cpu().numpy()
                           for b in dataobj.iter_batches(512, False)])

# (display, regname, ckpt-or-None, multiseed?)
MODELS = [
    ("E3 Nercessian", "E3_Nercessian", CKE/"E3_Nercessian.pt", False),
    ("E4 Pepe",       "E4_Pepe",       CKE/"E4_Pepe.pt",       False),
    ("E6 DSP",        "E6_DSP",        None,                   False),
    ("AC1 BiLSTM",    "AC1_BiLSTM",    CKE/"AC1_BiLSTM.pt",    False),
    ("AC2 GRU",       "AC2_GRU",       CKE/"AC2_GRU.pt",       False),
    ("AC3 Conformer", "AC3_Conformer", CKE/"AC3_Conformer.pt", False),
    ("A1 w/o Room",   "A1_NoRoomInput",CKE/"A1_NoRoomInput.pt",False),
    ("A0 Proposed ±12","A0_Proposed",  None,                   True),
    ("A2 +PrefLoss ±12","A2_withPrefLoss",None,                True),
]

rows = []
for disp, name, ckpt, multi in MODELS:
    if multi:
        var = "A0" if name == "A0_Proposed" else "A2"
        sl, rl, rd, bl, ol = [], [], [], [], []
        for s in SEEDS:
            ps = predict(name, SAVE/f"{cname('g12_f16k',s,var)}.pt", ds_s)
            pr = predict(name, SAVE/f"{cname('g12_f16k',s,var)}.pt", ds_r)
            sl.append(lsd(ps, dual_s).mean()); rl.append(lsd(pr, dual_r).mean())
            rd.append(dmr(pr-room_r, pref_r).mean())
            bl.append(lsd(pr, dual_r)[is_but].mean()); ol.append(lsd(pr, dual_r)[is_oa].mean())
        synth, real = float(np.mean(sl)), float(np.mean(rl))
        rows.append(dict(model=disp, synth=synth, synth_sd=np.std(sl,ddof=1), real=real, real_sd=np.std(rl,ddof=1),
                         rdmr=float(np.mean(rd)), gap=real-synth, but=float(np.mean(bl)), oa=float(np.mean(ol))))
    else:
        ps = predict(name, ckpt, ds_s); pr = predict(name, ckpt, ds_r)
        synth = lsd(ps, dual_s).mean(); real = lsd(pr, dual_r).mean()
        rl_arr = lsd(pr, dual_r)
        rows.append(dict(model=disp, synth=float(synth), synth_sd=0.0, real=float(real), real_sd=0.0,
                         rdmr=float(dmr(pr-room_r, pref_r).mean()), gap=float(real-synth),
                         but=float(rl_arr[is_but].mean()), oa=float(rl_arr[is_oa].mean())))

# ── 출력 ─────────────────────────────────────────────────────────────────────
print("\n"+"="*104)
print("Table 5 (OOD) — synth(N=3000) vs real(BUT+OpenAIR, N=2000), ±12·train_full")
print("LSD = LSD_dual. A0/A2=3-seed mean. gap = real - synth.")
print("="*104)
h=f"{'Model':18}{'synth LSD':>11}{'real LSD':>11}{'real DMR':>9}{'gap':>8}{'BUT-LSD':>9}{'OpenAIR':>9}"
print(h); print("-"*len(h))
for r in rows:
    print(f"{r['model']:18}{r['synth']:>11.3f}{r['real']:>11.3f}{r['rdmr']:>9.3f}{r['gap']:>+8.3f}{r['but']:>9.3f}{r['oa']:>9.3f}")

# AC gap 범위 + A0 비교
ac = [r for r in rows if r["model"].startswith("AC")]
ac_gaps = [r["gap"] for r in ac]
a0 = next(r for r in rows if r["model"].startswith("A0"))
print("-"*len(h))
print(f"AC dense gap 범위: {min(ac_gaps):.3f} ~ {max(ac_gaps):.3f}   |   A0(±12) gap = {a0['gap']:.3f}")
inside = min(ac_gaps) <= a0["gap"] <= max(ac_gaps)
print(f"A0 gap 가장 작은가? {'아니오 — AC 범위 안' if inside else ('예' if a0['gap']<min(ac_gaps) else '아니오 — AC보다 큼')} "
      f"(0.697 vs AC {min(ac_gaps):.3f}~{max(ac_gaps):.3f})")

import csv
with open(OUT / "table5_ood.csv","w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f, fieldnames=["model","synth","synth_sd","real","real_sd","rdmr","gap","but","oa"])
    w.writeheader()
    for r in rows: w.writerow({k:(f"{v:.4f}" if isinstance(v,float) else v) for k,v in r.items()})
print(f"\n저장: {OUT / 'table5_ood.csv'}")
print("출처: predictions=this revision eval(test_synth/test_real); source-split=test_real/rir_map.json(room_id)")
