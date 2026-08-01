"""
extract_overlay_sample.py — frequency-response overlay, single sample
=====================================================================
Extracts the test_synth sample sitting at the median LSD_dual of A0
(median-rank seed 7), so the figure shows a typical case rather than a best
one. The model is instantiated with gain_max=12.0 and the bound is asserted
before the forward pass, so the ±6 clamp cannot silently apply.

Output: results/fig_overlay_sample.json — freqs / T_room / T_pref / T_dual /
R_hat / H_hat, the seven individual band responses, and metadata.

Note: T_dual = clip(smooth(T_room + T_pref), ±12) is not a plain sum; see
compute_dual_target in dataset_generator_v4_tracklevel.py. The stored
dual_target is used as-is, since that is the target R_hat is trained against.
"""
import json
import numpy as np
import torch

import train_full as TF
from run_gain_freq_ablation import HERE, DEFAULT_DATA, cname
from export_track_level_predictions import TrackAwarePEQDataset

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
SAVE = HERE / "checkpoints"
SEED = 7   # 대표 seed (median seed, 다른 figure와 일관)

ds = TrackAwarePEQDataset(f"{DEFAULT_DATA}/test_synth", device=str(device))
dual = ds.data["dual_target"].cpu().numpy()
roomT = ds.data["room_target"].cpu().numpy()
prefT = ds.data["pref_target"].cpu().numpy()
freqs = ds.target_freqs.astype(float)
room_id = ds.data["room_id"].cpu().numpy()
mode_id = ds.data["mode_id"].cpu().numpy()

# ── A0 ±12 인스턴스 + assert ─────────────────────────────────────────────────
reg = TF.build_registry()
model = reg["A0_Proposed"]["model"]
assert abs(model.gain_max - 12.0) < 1e-9, f"GAIN BOUND TRAP: gain_max={model.gain_max} (≠12)"
assert abs(model.fc_max - 16000.0) < 1e-9, f"fc_max={model.fc_max}"
print(f"[assert OK] A0 instance gain_max={model.gain_max}, fc_max={model.fc_max}")
ck = torch.load(SAVE / f"{cname('g12_f16k', SEED, 'A0')}.pt", map_location=device, weights_only=False)
model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
model.to(device).eval()
TF._REGISTRY_TARGET["A0_Proposed"] = "dual"

# ── forward 전체 → R_hat / fc,gain,q ─────────────────────────────────────────
preds, fcs, gains, qs = [], [], [], []
with torch.no_grad():
    for b in ds.iter_batches(512, shuffle=False):
        out = TF.model_forward("A0_Proposed", model, b)
        preds.append(out["pred_response_db"].cpu().numpy())
        fcs.append(out["fc"].cpu().numpy()); gains.append(out["gain"].cpu().numpy()); qs.append(out["q"].cpu().numpy())
R_hat = np.concatenate(preds); FC=np.concatenate(fcs); G=np.concatenate(gains); Q=np.concatenate(qs)
assert float(np.abs(G).max()) <= 12.0 + 1e-3, f"gain out of ±12: max|g|={np.abs(G).max()}"
print(f"[forward OK] max|gain|={float(np.abs(G).max()):.3f} (≤12 확인), gain>6 비율={float((np.abs(G)>6).mean())*100:.1f}%")

# ── median LSD_dual 샘플 선정 (cherry-pick 금지) ─────────────────────────────
lsd_dual = np.sqrt(np.mean((R_hat - dual) ** 2, axis=-1))
order = np.argsort(lsd_dual)
idx = int(order[len(order) // 2])     # median-rank sample
H_hat_all = R_hat - roomT
lsd_pref = np.sqrt(np.mean((H_hat_all - prefT) ** 2, axis=-1))
print(f"[median] sample_idx={idx}  LSD_dual={lsd_dual[idx]:.4f} (median={np.median(lsd_dual):.4f})  "
      f"LSD_pref={lsd_pref[idx]:.4f}  room_id={int(room_id[idx])}  preset(mode_id)={int(mode_id[idx])}")

# ── 7 개별 biquad band 렌더 (peq_response 단일 필터) ─────────────────────────
ft = torch.tensor(FC[idx:idx+1], device=device); gt = torch.tensor(G[idx:idx+1], device=device); qt = torch.tensor(Q[idx:idx+1], device=device)
bands = []
with torch.no_grad():
    for k in range(FC.shape[1]):
        r = model.peq_response(ft[:, k:k+1], gt[:, k:k+1], qt[:, k:k+1])  # (1,128)
        bands.append(r.cpu().numpy()[0].tolist())

# ── 저장 ─────────────────────────────────────────────────────────────────────
out = {
    "meta": {
        "model": "A0_Proposed", "gain_max": float(model.gain_max), "fc_max": float(model.fc_max),
        "seed": SEED, "split": "test_synth", "sample_idx": idx,
        "room_id": int(room_id[idx]), "preset_id_mode": int(mode_id[idx]),
        "LSD_dual": float(lsd_dual[idx]), "LSD_dual_median_of_set": float(np.median(lsd_dual)),
        "LSD_pref": float(lsd_pref[idx]),
        "selection": "median-rank of A0 test_synth LSD_dual (no cherry-pick)",
        "freq_axis": "log-spaced 20-24000 Hz, 128 bins",
        "sign_convention": "H_hat = R_hat - T_room (corrected/heard); R_hat tracks T_dual",
        "T_dual_definition": "clip(smooth(T_room + T_pref), +/-12 dB)  [generator L308-311] — NOT raw sum",
        "fc_Hz": FC[idx].tolist(), "gain_dB": G[idx].tolist(), "q": Q[idx].tolist(),
    },
    "freqs_Hz": freqs.tolist(),
    "T_room": roomT[idx].tolist(),
    "T_pref": prefT[idx].tolist(),
    "T_dual": dual[idx].tolist(),
    "R_hat": R_hat[idx].tolist(),
    "H_hat": H_hat_all[idx].tolist(),
    "T_room_plus_pref_raw": (roomT[idx] + prefT[idx]).tolist(),   # 참고(비교용, smooth/clip 전)
    "bands_band0to6": bands,                                       # 7 biquad 분해
}
p = HERE / "results" / "fig_overlay_sample.json"
p.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"\n저장: {p}")
print(f"배열: freqs/T_room/T_pref/T_dual/R_hat/H_hat (각 128) + bands(7×128) + T_room_plus_pref_raw(참고)")
