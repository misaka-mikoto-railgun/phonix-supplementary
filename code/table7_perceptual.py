"""
table7_perceptual.py — Table 7 (tab:perceptual), ±12·train_full 기준
====================================================================
컬럼: LSD | ERB-LSD | 1/3-oct LSD | |Δ| vs A0 | %<JND(0.5dB)
추정 없음. test_synth. 메트릭 = perceptual_proxy.py (lsd_std/lsd_erb_fn/lsd_third_oct).
|Δ|/%<JND 정의 = perceptual_proxy.build_jnd_table: per-sample |LSD_model − LSD_A0| (lsd_std).
A0/A2 = gain±12 3-seed; A1 = train_full(2.713). A0 기준 = 3-seed, seed pairing = Table 4 와 동일.
"""
import numpy as np, torch, csv
import train_full as TF
from run_gain_freq_ablation import ORIG_ROOT, HERE, DEFAULT_DATA, cname
from dataset_generator_v4_tracklevel import PEQDataset
from arch_biquad import BIQUAD_REGISTRY
from perceptual_proxy import lsd_std, lsd_erb_fn, lsd_third_oct, JND_DB

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
FULL=ORIG_ROOT/"checkpoints"/"full"; CKE=HERE/"ckpt_eval"; SAVE=HERE/"checkpoints"; SEEDS=[42,123,7]
reg = TF.build_registry()
ds = PEQDataset(f"{DEFAULT_DATA}/test_synth", device=str(device))
dual = torch.cat([b["dual_target"] for b in ds.iter_batches(10**9, False)]).cpu().numpy()

@torch.no_grad()
def pred_std(name, ckpt):
    m = reg[name]["model"]; TF._REGISTRY_TARGET[name]=reg[name]["target"]
    if ckpt is not None:
        ck=torch.load(ckpt,map_location=device,weights_only=False)
        m.load_state_dict(ck["model"] if isinstance(ck,dict) and "model" in ck else ck, strict=False)
    m.to(device).eval()
    return np.concatenate([TF.model_forward(name,m,b)["pred_response_db"].cpu().numpy() for b in ds.iter_batches(512,False)])

@torch.no_grad()
def pred_biquad(name, ckpt):
    m = BIQUAD_REGISTRY[name](gain_max=12.0).to(device).eval()
    ck=torch.load(ckpt,map_location=device,weights_only=False)
    m.load_state_dict(ck["model"] if isinstance(ck,dict) and "model" in ck else ck, strict=False)
    out=[]
    for b in ds.iter_batches(512,False):
        out.append(m(b["features"],b["room_response"],b["mode_id"],b["band_gains"])["pred_response_db"].cpu().numpy())
    return np.concatenate(out)

def three(p):  # (LSD, ERB, third) per-sample arrays
    return lsd_std(p,dual), lsd_erb_fn(p,dual), lsd_third_oct(p,dual)

# ── 예측 → per-sample 메트릭 ─────────────────────────────────────────────────
A0_lsd = {s: lsd_std(pred_std("A0_Proposed", SAVE/f"{cname('g12_f16k',s,'A0')}.pt"), dual) for s in SEEDS}
A0_all = {s: three(pred_std("A0_Proposed", SAVE/f"{cname('g12_f16k',s,'A0')}.pt")) for s in SEEDS}
A2_all = {s: three(pred_std("A2_withPrefLoss", SAVE/f"{cname('g12_f16k',s,'A2')}.pt")) for s in SEEDS}

def mlevel_multiseed(per_seed):  # (LSD,ERB,oct) means averaged over seeds
    return tuple(float(np.mean([per_seed[s][i].mean() for s in SEEDS])) for i in range(3))

def jnd_vs_a0(model_lsd, a2_per_seed=None):
    """|Δ| & %<JND vs A0(3-seed). single: model_lsd(arr). A2: a2_per_seed{s:lsd}."""
    deltas, pcts = [], []
    for s in SEEDS:
        ml = a2_per_seed[s] if a2_per_seed is not None else model_lsd
        d = np.abs(A0_lsd[s] - ml)
        deltas.append(d.mean()); pcts.append((d < JND_DB).mean()*100)
    return float(np.mean(deltas)), float(np.mean(pcts))

# 모델 등록: (disp, kind, regname/biqname, ckpt)
SINGLE = [
    ("A1 w/o Room","A1_NoRoomInput",CKE/"A1_NoRoomInput.pt"),
    ("A3 w/o Pref","A3_NoPrefInput",CKE/"A3_NoPrefInput.pt"),
    ("AC1 raw","AC1_BiLSTM",CKE/"AC1_BiLSTM.pt"),
    ("AC2 raw","AC2_GRU",CKE/"AC2_GRU.pt"),
    ("AC3 raw","AC3_Conformer",CKE/"AC3_Conformer.pt"),
    ("E3 Nercessian","E3_Nercessian",CKE/"E3_Nercessian.pt"),
    ("E4 Pepe","E4_Pepe",CKE/"E4_Pepe.pt"),
    ("E6 DSP","E6_DSP",None),
]
BIQ = [("AC1_Biquad","AC1_BiLSTM_Biquad",SAVE/"AC1_BiLSTM_Biquad_g12.pt"),
       ("AC2_Biquad","AC2_GRU_Biquad",SAVE/"AC2_GRU_Biquad_g12.pt"),
       ("AC3_Biquad","AC3_Conformer_Biquad",SAVE/"AC3_Conformer_Biquad_g12.pt")]

rows=[]
# A0 (기준)
a0L,a0E,a0O = mlevel_multiseed(A0_all)
rows.append(("A0 Proposed ±12", a0L,a0E,a0O, 0.0, 100.0))
# A2 (3-seed, same-seed pairing)
a2L,a2E,a2O = mlevel_multiseed(A2_all)
d,p = jnd_vs_a0(None, a2_per_seed={s:A2_all[s][0] for s in SEEDS})
rows.append(("A2 +PrefLoss ±12", a2L,a2E,a2O, d,p))
# single-seed 모델
for disp,name,ck in SINGLE:
    L,E,O = three(pred_std(name,ck)); d,p = jnd_vs_a0(L)
    rows.append((disp, L.mean(),E.mean(),O.mean(), d,p))
# biquad
for disp,name,ck in BIQ:
    L,E,O = three(pred_biquad(name,ck)); d,p = jnd_vs_a0(L)
    rows.append((disp, L.mean(),E.mean(),O.mean(), d,p))

print("="*96)
print("Table 7 (Perceptual proxy, test_synth, ±12·train_full). |Δ|/%<JND vs A0(1.095, 3-seed, seed-avg).")
print("="*96)
h=f"{'Model':20}{'LSD':>8}{'ERB-LSD':>9}{'1/3-oct':>9}{'|Δ|vsA0':>9}{'%<JND':>8}"
print(h); print("-"*len(h))
for r in rows:
    print(f"{r[0]:20}{r[1]:>8.3f}{r[2]:>9.3f}{r[3]:>9.3f}{r[4]:>9.3f}{r[5]:>8.1f}")

with open(HERE/"results"/"table7_perceptual.csv","w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f); w.writerow(["Model","LSD","ERB_LSD","third_oct_LSD","abs_delta_vs_A0","pct_below_JND"])
    for r in rows: w.writerow([r[0]]+[f"{x:.4f}" for x in r[1:]])
print(f"\n저장: {HERE/'results'/'table7_perceptual.csv'}")
