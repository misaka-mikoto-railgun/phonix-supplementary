"""
table4_paired.py — Table 4 (Paired Statistical Diagnostics), ±12·train_full 기준
================================================================================
추정 없음. test_synth. A0=gain±12 3-seed; A2=3-seed(같은 seed 짝); 나머지 single-seed.
A1 = train_full 경로(features_clean) = 2.713 (Table 2/3 와 동일).
metric: lsd vs dual_target; dmr 은 heard=pred-room_target vs pref_target.

산출(LSD & DMR 각각):
  sample-level (3-seed mean±std): Δmean(A0-base), d_z, Win%, Wilcoxon p
  track-level  (N_groups=1306):    Δmean, d_z, Win%, Wilcoxon p   (track_level_eval 재사용)
"""
import json
import numpy as np
import torch
from scipy import stats

import train_full as TF
from run_gain_freq_ablation import ORIG_ROOT, HERE, DEFAULT_DATA, cname
from track_level_eval import compare_two_prediction_sets
from export_track_level_predictions import TrackAwarePEQDataset

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
FULL = ORIG_ROOT/"checkpoints"/"full"; CKE = HERE/"ckpt_eval"; SAVE = HERE/"checkpoints"
SEEDS = [42, 123, 7]
reg = TF.build_registry()
ds = TrackAwarePEQDataset(f"{DEFAULT_DATA}/test_synth", device=str(device))
dual = ds.data["dual_target"].cpu().numpy(); room = ds.data["room_target"].cpu().numpy()
pref = ds.data["pref_target"].cpu().numpy(); tid = ds.data["track_id"].cpu().numpy()

def lsd(p,t): return np.sqrt(np.mean((p-t)**2, -1))
def dmr(h,q): return np.mean((np.sign(h)==np.sign(q)).astype(float), -1)

@torch.no_grad()
def predict(name, ckpt):
    m = reg[name]["model"]; TF._REGISTRY_TARGET[name] = reg[name]["target"]
    if ckpt is not None:
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        m.load_state_dict(ck["model"] if isinstance(ck,dict) and "model" in ck else ck, strict=False)
    m.to(device).eval()
    return np.concatenate([TF.model_forward(name, m, b)["pred_response_db"].cpu().numpy()
                           for b in ds.iter_batches(512, False)])

# 예측: A0/A2 per-seed, 비교대상 single
A0p = {s: predict("A0_Proposed", SAVE/f"{cname('g12_f16k',s,'A0')}.pt") for s in SEEDS}
A2p = {s: predict("A2_withPrefLoss", SAVE/f"{cname('g12_f16k',s,'A2')}.pt") for s in SEEDS}
COMP = [("A1_NoRoomInput","A1_NoRoomInput.pt"),("A3_NoPrefInput","A3_NoPrefInput.pt"),
        ("E3_Nercessian","E3_Nercessian.pt"),("E4_Pepe","E4_Pepe.pt"),
        ("AC1_BiLSTM","AC1_BiLSTM.pt"),("AC2_GRU","AC2_GRU.pt"),("AC3_Conformer","AC3_Conformer.pt")]
Cp = {n: predict(n, CKE/f) for n,f in COMP}

def ms(v): return float(np.mean(v)), float(np.std(v, ddof=1))

def sample_paired(a0_pred_by_seed, base_by_seed_or_arr, same_seed):
    """3-seed 평균 sample-level Δ/d_z/Win/p (LSD & DMR)."""
    L = {k: [] for k in ("dl","dz","win","p")}; D = {k: [] for k in ("dl","dz","win","p")}
    for s in SEEDS:
        a0L = lsd(a0_pred_by_seed[s], dual); a0D = dmr(a0_pred_by_seed[s]-room, pref)
        bp = base_by_seed_or_arr[s] if same_seed else base_by_seed_or_arr
        bL = lsd(bp, dual); bD = dmr(bp-room, pref)
        dl = a0L-bL; dd = a0D-bD
        L["dl"].append(dl.mean()); L["dz"].append(dl.mean()/dl.std(ddof=1)); L["win"].append(np.mean(a0L<bL))
        D["dl"].append(dd.mean()); D["dz"].append(dd.mean()/(dd.std(ddof=1)+1e-12)); D["win"].append(np.mean(a0D>bD))
        try: L["p"].append(stats.wilcoxon(dl)[1])
        except Exception: L["p"].append(np.nan)
        try: D["p"].append(stats.wilcoxon(dd)[1])
        except Exception: D["p"].append(np.nan)
    return L, D

def track_paired(a0_pred_by_seed, base_by_seed_or_arr, same_seed):
    """3-seed 평균 track-level(N=1306) Δ/d_z/Win/p (compare_two_prediction_sets)."""
    out = {m: {k: [] for k in ("dl","dz","win","p")} for m in ("lsd","dmr")}
    ng = None
    for s in SEEDS:
        bp = base_by_seed_or_arr[s] if same_seed else base_by_seed_or_arr
        base = {"pred": bp, "target": dual, "pref_target": pref, "room_target": room, "track_id": tid}
        cand = {"pred": a0_pred_by_seed[s], "target": dual, "pref_target": pref, "room_target": room, "track_id": tid}
        r = compare_two_prediction_sets(base, cand, group_key="track_id", n_boot=200, seed=42)
        ng = r.n_groups
        for m in ("lsd","dmr"):
            mm = r.metrics[m]
            out[m]["dl"].append(mm.mean_diff); out[m]["dz"].append(mm.cohens_dz)
            out[m]["win"].append(mm.win_rate); out[m]["p"].append(mm.p_wilcoxon)
    return out, ng

TARGETS = [("A2", A2p, True)] + [(n.split("_")[0], Cp[n], False) for n,_ in COMP]
rows = []
for lab, base, ss in TARGETS:
    sL, sD = sample_paired(A0p, base, ss)
    tr, ng = track_paired(A0p, base, ss)
    rows.append(dict(
        tgt=lab,
        s_lsd_d=ms(sL["dl"]), s_lsd_dz=ms(sL["dz"]), s_lsd_win=ms(sL["win"]), s_lsd_p=np.nanmax(sL["p"]),
        s_dmr_d=ms(sD["dl"]), s_dmr_dz=ms(sD["dz"]), s_dmr_win=ms(sD["win"]),
        t_lsd_d=ms(tr["lsd"]["dl"]), t_lsd_dz=ms(tr["lsd"]["dz"]), t_lsd_win=ms(tr["lsd"]["win"]),
        t_dmr_d=ms(tr["dmr"]["dl"]), t_dmr_dz=ms(tr["dmr"]["dz"]), ng=ng))

# ── 출력 ─────────────────────────────────────────────────────────────────────
print("="*118)
print("Table 4 — A0 vs each model, paired (test_synth, ±12·train_full).  ‡=Wilcoxon p<0.001")
print("  SAMPLE-level (N=3000), 3-seed mean±std.  A1=features_clean(2.713) pipeline.")
print("="*118)
h=f"{'A0 vs':6}{'LSD Δ':>14}{'LSD d_z':>14}{'LSD Win%':>13}{'p':>9}{'  ':2}{'DMR Δ':>12}{'DMR d_z':>12}{'DMR Win%':>12}"
print(h); print("-"*len(h))
for r in rows:
    mark="‡" if (r["s_lsd_p"] is not None and r["s_lsd_p"]<1e-3) else ""
    print(f"{r['tgt']:6}{r['s_lsd_d'][0]:>+8.3f}±{r['s_lsd_d'][1]:<4.2f}{r['s_lsd_dz'][0]:>+8.2f}±{r['s_lsd_dz'][1]:<4.2f}"
          f"{r['s_lsd_win'][0]*100:>8.1f}±{r['s_lsd_win'][1]*100:<3.0f}{r['s_lsd_p']:>8.0e}{mark:>2}"
          f"{r['s_dmr_d'][0]:>+8.3f}{r['s_dmr_dz'][0]:>+12.2f}{r['s_dmr_win'][0]*100:>10.1f}")

print("\n"+"="*100)
print(f"TRACK-level (N_groups={rows[0]['ng']}), 3-seed mean±std")
print("="*100)
h2=f"{'A0 vs':6}{'LSD Δ':>16}{'LSD d_z':>14}{'LSD Win%':>13}{'  ':2}{'DMR Δ':>12}{'DMR d_z':>12}"
print(h2); print("-"*len(h2))
for r in rows:
    print(f"{r['tgt']:6}{r['t_lsd_d'][0]:>+9.3f}±{r['t_lsd_d'][1]:<5.2f}{r['t_lsd_dz'][0]:>+8.2f}±{r['t_lsd_dz'][1]:<4.2f}"
          f"{r['t_lsd_win'][0]*100:>8.1f}±{r['t_lsd_win'][1]*100:<3.0f}{r['t_dmr_d'][0]:>+12.3f}{r['t_dmr_dz'][0]:>+12.2f}")

# CSV
import csv
with open(HERE/"results"/"table4_paired_synth.csv","w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f); w.writerow(["A0_vs","level","metric","delta_mean","delta_std","d_z","d_z_std","win_pct","wilcoxon_p","N"])
    for r in rows:
        w.writerow([r["tgt"],"sample","LSD",f"{r['s_lsd_d'][0]:.4f}",f"{r['s_lsd_d'][1]:.4f}",f"{r['s_lsd_dz'][0]:.3f}",f"{r['s_lsd_dz'][1]:.3f}",f"{r['s_lsd_win'][0]*100:.1f}",f"{r['s_lsd_p']:.2e}",3000])
        w.writerow([r["tgt"],"sample","DMR",f"{r['s_dmr_d'][0]:.4f}",f"{r['s_dmr_d'][1]:.4f}",f"{r['s_dmr_dz'][0]:.3f}",f"{r['s_dmr_dz'][1]:.3f}",f"{r['s_dmr_win'][0]*100:.1f}","",3000])
        w.writerow([r["tgt"],"track","LSD",f"{r['t_lsd_d'][0]:.4f}",f"{r['t_lsd_d'][1]:.4f}",f"{r['t_lsd_dz'][0]:.3f}",f"{r['t_lsd_dz'][1]:.3f}",f"{r['t_lsd_win'][0]*100:.1f}","",r["ng"]])
        w.writerow([r["tgt"],"track","DMR",f"{r['t_dmr_d'][0]:.4f}",f"{r['t_dmr_d'][1]:.4f}",f"{r['t_dmr_dz'][0]:.3f}",f"{r['t_dmr_dz'][1]:.3f}","","",r["ng"]])
print(f"\n저장: {HERE/'results'/'table4_paired_synth.csv'}")
print("주의: A0 vs A2 sample-level은 ‡(p<0.001)이나 effect 미미(d_z≈-0.19, Win 58%) + seed-level n.s.(§4.2). 본문 병기 필수.")
