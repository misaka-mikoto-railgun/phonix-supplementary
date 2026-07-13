"""
track_stats.py — track-level(N=1306) 재계산, 3-seed 집계
========================================================
track_level_eval.compare_two_prediction_sets 재사용.
baseline=비교대상, candidate=A0_s  → mean_diff = A0 - comp
  LSD: 음수 = A0 per-track LSD 낮음 = A0 better
  DMR: 양수 = A0 better
A0/A2 = gain±12 3-seed; 비교대상 = 원본 single-seed(42). test_synth.
A0 vs A2 는 같은 seed, 나머지는 single-seed comp.
"""
import json
from pathlib import Path

import numpy as np
import torch

import train_full as TF
from run_gain_freq_ablation import ORIG_ROOT, HERE, DEFAULT_DATA, cname
from track_level_eval import compare_two_prediction_sets
from export_track_level_predictions import TrackAwarePEQDataset

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
FULL = ORIG_ROOT / "checkpoints" / "full"
SAVE = HERE / "checkpoints"
SEEDS = [42, 123, 7]
N_BOOT = 1000

COMPARATORS = [
    ("A1_NoRoomInput", "A1_NoRoomInput.pt"),
    ("A3_NoPrefInput", "A3_NoPrefInput.pt"),
    ("E3_Nercessian",  "E3_Nercessian.pt"),
    ("E4_Pepe",        "E4_Pepe.pt"),
    ("AC1_BiLSTM",     "AC1_BiLSTM.pt"),
    ("AC2_GRU",        "AC2_GRU.pt"),
    ("AC3_Conformer",  "AC3_Conformer.pt"),
]

registry = TF.build_registry()
for n, _ in COMPARATORS:
    TF._REGISTRY_TARGET[n] = "dual"
TF._REGISTRY_TARGET["A0_Proposed"] = "dual"
TF._REGISTRY_TARGET["A2_withPrefLoss"] = "dual"

ds = TrackAwarePEQDataset(f"{DEFAULT_DATA}/test_synth", device=str(device))
track_id = ds.data["track_id"].cpu().numpy()
target  = ds.data["dual_target"].cpu().numpy()
pref_t  = ds.data["pref_target"].cpu().numpy()
room_t  = ds.data["room_target"].cpu().numpy()
print(f"n_groups (unique track_id) = {len(np.unique(track_id))}")


@torch.no_grad()
def predict(name, model_key, ckpt):
    model = registry[model_key]["model"]
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    preds = []
    for batch in ds.iter_batches(512, shuffle=False):
        preds.append(TF.model_forward(name, model, batch)["pred_response_db"].cpu().numpy())
    return np.concatenate(preds)


def payload(pred):
    return {"pred": pred, "target": target, "pref_target": pref_t,
            "room_target": room_t, "track_id": track_id}


# 비교대상 single-seed 예측 ────────────────────────────────────────────────────
base_pl = {n: payload(predict(n, n, FULL / f)) for n, f in COMPARATORS}
# A0/A2 per-seed 예측 ──────────────────────────────────────────────────────────
a0_pl = {s: payload(predict("A0_Proposed", "A0_Proposed", SAVE / f"{cname('g12_f16k', s, 'A0')}.pt")) for s in SEEDS}
a2_pl = {s: payload(predict("A2_withPrefLoss", "A2_withPrefLoss", SAVE / f"{cname('g12_f16k', s, 'A2')}.pt")) for s in SEEDS}

targets = ["A2_withPrefLoss"] + [n for n, _ in COMPARATORS]

# 원본 track-level 본문 수치(tex): A0 vs A1 LSD -1.254, DMR +0.084; AC -0.574~-0.592 (A0 higher LSD)
ORIG_TXT = {
    "A1_NoRoomInput": "orig: LSD -1.254, DMR +0.084",
    "AC2_GRU":        "orig: AC -0.574~-0.592 lower LSD than A0",
}

print("\n" + "=" * 96)
print("Track-level (N_groups, 3-seed): A0 vs target  —  candidate=A0, baseline=comp → Δ=A0-comp")
print("  LSD Δ 음수=A0 better, DMR Δ 양수=A0 better.  A0/A2=3seed, comp=single(42)")
print("=" * 96)
hdr = f"{'A0 vs':>16} {'LSD Δ(mean±std)':>20} {'LSD d_z':>14} {'DMR Δ(mean±std)':>20} {'DMR d_z':>14} {'n_grp':>6}"
print(hdr); print("-" * len(hdr))
summary = {}
for tgt in targets:
    lsd_d, lsd_z, dmr_d, dmr_z, ng = [], [], [], [], None
    for s in SEEDS:
        base = a2_pl[s] if tgt == "A2_withPrefLoss" else base_pl[tgt]
        res = compare_two_prediction_sets(base, a0_pl[s], group_key="track_id", n_boot=N_BOOT, seed=42)
        ng = res.n_groups
        lsd_d.append(res.metrics["lsd"].mean_diff); lsd_z.append(res.metrics["lsd"].cohens_dz)
        dmr_d.append(res.metrics["dmr"].mean_diff); dmr_z.append(res.metrics["dmr"].cohens_dz)
    def ms(v): return float(np.mean(v)), float(np.std(v, ddof=1))
    ld = ms(lsd_d); lz = ms(lsd_z); dd = ms(dmr_d); dz = ms(dmr_z)
    print(f"{tgt:>16} {ld[0]:>+9.3f}±{ld[1]:<8.3f} {lz[0]:>+8.2f} "
          f"{dd[0]:>+9.3f}±{dd[1]:<8.3f} {dz[0]:>+8.2f} {ng:>6}")
    summary[tgt] = dict(n_groups=ng, lsd_delta=ld, lsd_dz=lz, dmr_delta=dd, dmr_dz=dz)

out = HERE / "results" / "track_stats_3seed_test_synth.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"\n저장: {out}")
