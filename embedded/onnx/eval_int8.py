"""eval_int8.py — int8 QDQ onnx vs fp32 ckpt, test_synth LSD/CosSim 저하 보고.

metric (논문 파이프라인 일치):
  LSD    = mean_sample sqrt(mean_f (pred - dual_target)^2)
  CosSim = mean_sample cos(heard, pref_target),  heard = pred - room_target
그래프 경계 출력(room_corr/fc/gain/q)에서 fp32 모델과 '동일 post-proc'로 pred 재구성:
  A0   : pred = room_corr + peq_response(fc,gain,q)
  AC*  : pred = room_corr + pref_curve(band) + peq_response(fc,gain,q)
  E3/E4: pred = response(fc,gain,q)
fp32 pred 은 PyTorch forward out['pred_response_db'] (동일 수식).
"""
import sys, glob
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
REV  = HERE.parent
ORIG = REV.parent
sys.path.insert(0, str(REV))
from model import DualObjectiveAdaptivePEQ
from baselines import E3_NercessianMLP, E4_PepeCNN
from arch_biquad import AC2_GRU_Biquad, AC3_Conformer_Biquad

TEST = ORIG / "data" / "dataset_v3" / "test_synth"

# ── load test_synth ─────────────────────────────────────────────────────────
A = {k: [] for k in ["features", "room_response", "mode_id", "band_gains",
                     "room_target", "pref_target", "dual_target"]}
for c in sorted(glob.glob(str(TEST / "chunk_*.npz"))):
    d = np.load(c, allow_pickle=False)
    for k in A:
        A[k].append(d[k])
T = {k: np.concatenate(v) for k, v in A.items()}
N = len(T["mode_id"])
feat = torch.from_numpy(T["features"]).float()
room = torch.from_numpy(T["room_response"]).float()
mid  = torch.from_numpy(T["mode_id"]).long()
band = torch.from_numpy(T["band_gains"]).float()
dual = T["dual_target"]; rt = T["room_target"]; pt = T["pref_target"]
mode_oh = np.eye(4, dtype=np.float32)[T["mode_id"]]
print(f"test_synth N={N}")


def lsd(pred, tgt):
    return np.sqrt(np.mean((pred - tgt) ** 2, axis=-1))

def cossim(a, b):
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return num / den

def metrics(pred):
    heard = pred - rt
    return float(lsd(pred, dual).mean()), float(cossim(heard, pt).mean())


SPECS = [
    ("A0",  lambda: DualObjectiveAdaptivePEQ(gain_max=12.0, fc_max=16000.0),
     REV/"checkpoints"/"A0_g12_f16k_s7.pt", "a0_int8.onnx", "a0", True),
    ("E3",  lambda: E3_NercessianMLP(),
     ORIG/"checkpoints"/"full"/"E3_Nercessian.pt", "e3_nercessian_int8.onnx", "e", False),
    ("E4",  lambda: E4_PepeCNN(),
     ORIG/"checkpoints"/"full"/"E4_Pepe.pt", "e4_pepe_int8.onnx", "e", False),
    ("AC2", lambda: AC2_GRU_Biquad(gain_max=12.0),
     REV/"checkpoints"/"AC2_GRU_Biquad_g12.pt", "ac2_gru_biquad_int8.onnx", "ac", True),
    ("AC3", lambda: AC3_Conformer_Biquad(gain_max=12.0),
     REV/"checkpoints"/"AC3_Conformer_Biquad_g12.pt", "ac3_conformer_biquad_int8.onnx", "ac", True),
]


def load(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(st, strict=False)
    return model.eval()


rows = []
for name, build, ckpt, onnx_name, kind, cond in SPECS:
    m = load(build(), ckpt)
    # ── fp32 (PyTorch full forward) ──
    preds = []
    with torch.no_grad():
        for s in range(0, N, 512):
            if cond:
                o = m(feat[s:s+512], room[s:s+512], mid[s:s+512], band[s:s+512])
            else:
                o = m(feat[s:s+512])
            preds.append(o["pred_response_db"].numpy())
    pred_fp = np.concatenate(preds)
    lsd_fp, cos_fp = metrics(pred_fp)

    # ── int8 (onnx per-sample) → reconstruct pred ──
    sess = ort.InferenceSession(str(HERE/onnx_name), providers=["CPUExecutionProvider"])
    innames = [i.name for i in sess.get_inputs()]
    rc8, fc8, g8, q8 = [], [], [], []
    for i in range(N):
        if cond:
            feed = {innames[0]: T["features"][i:i+1], innames[1]: T["room_response"][i:i+1],
                    innames[2]: mode_oh[i:i+1], innames[3]: T["band_gains"][i:i+1]}
        else:
            feed = {innames[0]: T["features"][i:i+1]}
        out = sess.run(None, feed)   # order: [room_corr,fc,gain,q] or [fc,gain,q]
        if cond:
            rc8.append(out[0][0]); fc8.append(out[1][0]); g8.append(out[2][0]); q8.append(out[3][0])
        else:
            fc8.append(out[0][0]); g8.append(out[1][0]); q8.append(out[2][0])
    fcT = torch.from_numpy(np.stack(fc8)).float()
    gT  = torch.from_numpy(np.stack(g8)).float()
    qT  = torch.from_numpy(np.stack(q8)).float()
    with torch.no_grad():
        if kind == "a0":
            peq = m.peq_response(fcT, gT, qT).numpy()
            pred_i8 = np.stack(rc8) + peq
        elif kind == "ac":
            peq = m.peq_response(fcT, gT, qT).numpy()
            prefc = m._pref_curve(band).numpy()
            pred_i8 = np.stack(rc8) + prefc + peq
        else:  # e
            pred_i8 = m.response(fcT, gT, qT).numpy()
    lsd_i8, cos_i8 = metrics(pred_i8)

    rows.append((name, lsd_fp, lsd_i8, lsd_i8 - lsd_fp, cos_fp, cos_i8, cos_i8 - cos_fp))
    print(f"  {name}: fp32 LSD={lsd_fp:.4f} cos={cos_fp:.4f} | int8 LSD={lsd_i8:.4f} cos={cos_i8:.4f} "
          f"| ΔLSD={lsd_i8-lsd_fp:+.4f} Δcos={cos_i8-cos_fp:+.4f}")

print("\n" + "=" * 78)
print(f"{'variant':<6}{'fp32 LSD':>10}{'int8 LSD':>10}{'ΔLSD':>9}{'ΔLSD%':>8}"
      f"{'fp32 cos':>10}{'int8 cos':>10}{'Δcos':>9}")
print("-" * 78)
for name, lf, li, dl, cf, ci, dc in rows:
    print(f"{name:<6}{lf:>10.4f}{li:>10.4f}{dl:>+9.4f}{dl/lf*100:>+7.1f}%{cf:>10.4f}{ci:>10.4f}{dc:>+9.4f}")
print("=" * 78)
print("LSD: dual-target, lower better. cos: heard vs pref_target, higher better. (test_synth, n=3000)")
