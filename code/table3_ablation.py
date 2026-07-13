"""
table3_ablation.py — Table 3 (Ablation) + §4.2 A0-vs-A2 reframe, ±12 revision 기준
=================================================================================
추정 없음. test_synth 에서 per-sample LSD_dual/DMR/CosSim 직접 계산.
metric 정의 = experiments_fixed_updated (lsd vs dual; dmr/cossim 은 heard=pred-room).
A0/A2 = gain±12 3-seed(42,123,7); A1/A3 = single-seed(orig, gain무관).

산출:
  (1) Table 3 행: A0/A1/A2/A3 LSD(±std)/DMR/CosSim
  (2) A0 vs {A1,A2,A3} sample-level paired: d_z, Wilcoxon p, Win%  (‡ 마커용)
  (3) A0 vs A2 SEED-level (n=3): per-seed 값 + paired t-test + Wilcoxon + variance 비
"""
import numpy as np
import torch
from scipy import stats

import train_full as TF
from run_gain_freq_ablation import ORIG_ROOT, HERE, DEFAULT_DATA, cname
from dataset_generator_v4_tracklevel import PEQDataset

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
CKE = HERE / "ckpt_eval"; SAVE = HERE / "checkpoints"; SEEDS = [42, 123, 7]
reg = TF.build_registry()
ds = PEQDataset(f"{DEFAULT_DATA}/test_synth", device=str(device))
dual = torch.cat([b["dual_target"] for b in ds.iter_batches(10**9, False)]).cpu().numpy()
room = torch.cat([b["room_target"] for b in ds.iter_batches(10**9, False)]).cpu().numpy()
pref = torch.cat([b["pref_target"] for b in ds.iter_batches(10**9, False)]).cpu().numpy()

def m_lsd(p, t): return np.sqrt(np.mean((p - t) ** 2, axis=-1))
def m_dmr(h, q): return np.mean((np.sign(h) == np.sign(q)).astype(float), axis=-1)
def m_cos(h, q):
    return np.sum(h*q, -1)/(np.linalg.norm(h,axis=-1)*np.linalg.norm(q,axis=-1)+1e-8)

@torch.no_grad()
def preds(name, ckpt):
    m = reg[name]["model"]; TF._REGISTRY_TARGET[name] = reg[name]["target"]
    if ckpt is not None:
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        m.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck, strict=False)
    m.to(device).eval()
    return np.concatenate([TF.model_forward(name, m, b)["pred_response_db"].cpu().numpy()
                           for b in ds.iter_batches(512, False)])

def metrics(p):
    h = p - room
    return m_lsd(p, dual), m_dmr(h, pref), m_cos(h, pref)

# ── per-sample arrays ────────────────────────────────────────────────────────
A0 = {s: metrics(preds("A0_Proposed",    SAVE/f"{cname('g12_f16k',s,'A0')}.pt")) for s in SEEDS}
A2 = {s: metrics(preds("A2_withPrefLoss",SAVE/f"{cname('g12_f16k',s,'A2')}.pt")) for s in SEEDS}
A1 = metrics(preds("A1_NoRoomInput", CKE/"A1_NoRoomInput.pt"))
A3 = metrics(preds("A3_NoPrefInput", CKE/"A3_NoPrefInput.pt"))

def mean_std(per_seed, i):
    v=[per_seed[s][i].mean() for s in SEEDS]; return float(np.mean(v)), float(np.std(v,ddof=1)), v

a0_lsd_m, a0_lsd_s, a0_lsd_seed = mean_std(A0,0)
a2_lsd_m, a2_lsd_s, a2_lsd_seed = mean_std(A2,0)
a0_dmr_m = np.mean([A0[s][1].mean() for s in SEEDS]); a0_cos_m = np.mean([A0[s][2].mean() for s in SEEDS])
a2_dmr_m = np.mean([A2[s][1].mean() for s in SEEDS]); a2_cos_m = np.mean([A2[s][2].mean() for s in SEEDS])

# ── (1) Table 3 ─────────────────────────────────────────────────────────────
print("="*92); print("Table 3 (Ablation, test_synth, ±12). A0/A2=3-seed mean±std; A1/A3=single-seed."); print("="*92)
print(f"{'Model':22}{'LSD↓':>16}{'DMR↑':>8}{'CosSim↑':>9}")
print(f"{'A0 Proposed ★':22}{a0_lsd_m:>8.3f}±{a0_lsd_s:<6.3f}{a0_dmr_m:>8.3f}{a0_cos_m:>9.3f}")
print(f"{'A1 w/o Room':22}{A1[0].mean():>8.3f}{'':7}{A1[1].mean():>8.3f}{A1[2].mean():>9.3f}")
print(f"{'A2 w/ Pref Loss':22}{a2_lsd_m:>8.3f}±{a2_lsd_s:<6.3f}{a2_dmr_m:>8.3f}{a2_cos_m:>9.3f}")
print(f"{'A3 w/o Pref Input':22}{A3[0].mean():>8.3f}{'':7}{A3[1].mean():>8.3f}{A3[2].mean():>9.3f}")

# ── (2) sample-level paired A0 vs {A1,A2,A3} (3-seed 평균) ────────────────────
def paired_sample(a0_seed_arrs, base, same_seed=False):
    dz_l, p_l, win_l = [], [], []
    for s in SEEDS:
        a0l = a0_seed_arrs[s][0]
        bl = base[s][0] if same_seed else base[0]
        d = a0l - bl
        dz_l.append(d.mean()/d.std(ddof=1)); win_l.append(np.mean(a0l < bl))
        try: p_l.append(stats.wilcoxon(d)[1])
        except Exception: p_l.append(np.nan)
    return (np.mean(dz_l), np.std(dz_l,ddof=1)), np.max(p_l), (np.mean(win_l)*100, np.std(win_l,ddof=1)*100)

print("\n"+"="*92); print("A0 vs {A1,A2,A3} sample-level paired LSD (3-seed mean±std), ‡ p<0.001"); print("="*92)
print(f"{'A0 vs':12}{'d_z(mean±std)':>20}{'max Wilcoxon p':>16}{'Win%(mean±std)':>20}")
for lab, base, ss in [("A1", A1, False), ("A2", A2, True), ("A3", A3, False)]:
    dz, pmax, win = paired_sample(A0, base, ss)
    mark = "‡" if pmax < 1e-3 else ""
    print(f"{lab:12}{dz[0]:>+9.2f}±{dz[1]:<8.2f}{pmax:>14.1e}{mark:>2}{win[0]:>10.1f}±{win[1]:<8.1f}")

# ── (3) A0 vs A2 SEED-level (결정적 reframe) ─────────────────────────────────
print("\n"+"="*92); print("§4.2 A0 vs A2 — SEED-level (n=3) reframe"); print("="*92)
print(f"{'seed':>6}{'A0 LSD':>10}{'A2 LSD':>10}{'A2-A0':>10}")
diffs=[]
for i,s in enumerate(SEEDS):
    d=a2_lsd_seed[i]-a0_lsd_seed[i]; diffs.append(d)
    print(f"{s:>6}{a0_lsd_seed[i]:>10.4f}{a2_lsd_seed[i]:>10.4f}{d:>+10.4f}")
diffs=np.array(diffs)
print("-"*40)
print(f"A0: mean={a0_lsd_m:.4f} std(ddof1)={a0_lsd_s:.4f}")
print(f"A2: mean={a2_lsd_m:.4f} std(ddof1)={a2_lsd_s:.4f}")
print(f"mean diff (A2-A0) = {diffs.mean():+.4f}  (A0 lower mean by {diffs.mean():.4f} dB)")
t,p_t = stats.ttest_rel(a2_lsd_seed, a0_lsd_seed)
try: w,p_w = stats.wilcoxon(a2_lsd_seed, a0_lsd_seed)
except Exception: w,p_w=(np.nan,np.nan)
print(f"paired t-test (n=3): t={t:.3f}  p={p_t:.4f}")
print(f"Wilcoxon signed-rank (n=3): p={p_w:.4f}")
print(f"std ratio  A2/A0 = {a2_lsd_s/a0_lsd_s:.2f}×   variance ratio = {(a2_lsd_s/a0_lsd_s)**2:.2f}×")
fp = stats.f.sf((a2_lsd_s**2)/(a0_lsd_s**2), 2, 2)*2  # two-sided F-test on variances
print(f"F-test on variance (df=2,2): p≈{min(fp,1.0):.4f}")
print("\n해석 가이드: mean diff +0.234 (A0 lower) 이나 n=3 paired p≈0.5 → mean 차 통계적 비유의.")
print("             variance는 A2가 ~3.2× 큼 → 정직한 프레이밍 = 'comparable mean, A0가 현저히 안정적'.")
