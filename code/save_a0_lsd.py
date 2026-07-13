"""A0 g12 3-seed mean per-sample LSD 를 stat_dir/A0_Proposed_lsd.npy 로 저장 (fig 기준선용)."""
from pathlib import Path
import numpy as np
import torch
import train_full as TF
from run_gain_freq_ablation import HERE, DEFAULT_DATA, evaluate, cname
from dataset_generator_v4_tracklevel import PEQDataset

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
SAVE = HERE / "checkpoints"; STAT = HERE / "results" / "stats"
reg = TF.build_registry(); TF._REGISTRY_TARGET["A0_Proposed"] = "dual"
ds = PEQDataset(f"{DEFAULT_DATA}/test_synth", device=str(device))
arrs = []
for s in [42, 123, 7]:
    m = reg["A0_Proposed"]["model"]
    ck = torch.load(SAVE / f"{cname('g12_f16k', s, 'A0')}.pt", map_location=device, weights_only=False)
    m.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
    arrs.append(evaluate("A0_Proposed", None, m.to(device).eval(), ds, device)["lsd_arr"])
a0 = np.mean(arrs, axis=0)
np.save(STAT / "A0_Proposed_lsd.npy", a0)
print(f"saved A0_Proposed_lsd.npy (g12 3-seed mean) shape={a0.shape} mean={a0.mean():.4f}")
