"""
export_dense_for_optionA.py — Option A(SciPy fitting) 입력 생성
==============================================================
ac_fitting_A.py 는 {model}_pred.npy(dense AC 예측) + targets_dual.npy 를 요구.
dense AC(AC2_GRU 등)를 원본 ckpt 로 forward 해 생성. test_synth.
"""
import argparse
from pathlib import Path
import numpy as np
import torch

import train_full as TF
from run_gain_freq_ablation import ORIG_ROOT, DEFAULT_DATA, HERE
from dataset_generator_v4_tracklevel import PEQDataset

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=["AC2_GRU", "AC1_BiLSTM", "AC3_Conformer"])
ap.add_argument("--stat_dir", default=str(HERE / "results" / "stats"))
args = ap.parse_args()

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
FULL = ORIG_ROOT / "checkpoints" / "full"
stat_dir = Path(args.stat_dir); stat_dir.mkdir(parents=True, exist_ok=True)

registry = TF.build_registry()
ds = PEQDataset(f"{DEFAULT_DATA}/test_synth", device=str(device))

# targets_dual 저장 (한 번)
duals = []
for batch in ds.iter_batches(512, shuffle=False):
    duals.append(batch["dual_target"].cpu().numpy())
dual = np.concatenate(duals)
np.save(stat_dir / "targets_dual.npy", dual)
print(f"saved targets_dual.npy {dual.shape}")

ckpt_file = {"AC1_BiLSTM": "AC1_BiLSTM.pt", "AC2_GRU": "AC2_GRU.pt", "AC3_Conformer": "AC3_Conformer.pt"}
for name in args.models:
    TF._REGISTRY_TARGET[name] = "dual"
    model = registry[name]["model"]
    ck = torch.load(FULL / ckpt_file[name], map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    preds = []
    with torch.no_grad():
        for batch in ds.iter_batches(512, shuffle=False):
            preds.append(TF.model_forward(name, model, batch)["pred_response_db"].cpu().numpy())
    pred = np.concatenate(preds)
    np.save(stat_dir / f"{name}_pred.npy", pred)
    lsd = np.sqrt(((pred - dual) ** 2).mean(axis=-1))
    np.save(stat_dir / f"{name}_lsd.npy", lsd)
    print(f"saved {name}_pred.npy {pred.shape}  dense LSD={lsd.mean():.4f}")
