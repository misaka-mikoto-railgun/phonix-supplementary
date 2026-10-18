"""
diag_room_corr.py — where does the low-frequency preference shape come from?
===========================================================================
For each A0 checkpoint: how far room_corr sits from T_room (does Stage A absorb
the preference shelf?), how much of the preference target below 200 Hz is
carried by room_corr vs. the PEQ, and the usual LSD / PA.

Hypothesis: in the good basin room_corr deviates MORE from T_room (it carries
the shelf the peaking sections cannot make); in the bad basin room_corr stays
close to T_room and the preference target is simply not reached.

    python diag_room_corr.py --data_dir <dataset_v3> --ckpts good=path/to/A0_g12_f20k_s42.pt bad=path/to/A0_g12_f16k_s42.pt
"""
import argparse
from pathlib import Path
import numpy as np
import torch

import train_full as TF
from export_track_level_predictions import TrackAwarePEQDataset
import ckpt_io

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)
ap.add_argument("--ckpts", nargs="+", required=True, help="label=path ...")
ap.add_argument("--fc_max", type=float, default=16000.0, help="fc_max the checkpoint was trained with (per label: label=path:fc_max)")
ap.add_argument("--split", default="test_synth")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = TrackAwarePEQDataset(f"{args.data_dir}/{args.split}", device=str(device))
d = ds.data
dual, room, pref = (d[k].cpu().numpy() for k in ("dual_target", "room_target", "pref_target"))
mode = d["mode_id"].cpu().numpy(); freqs = np.asarray(ds.target_freqs, float)
lo = freqs < 200.0

def lsd(a, b): return np.sqrt(np.mean((a - b) ** 2, -1))

reg = TF.build_registry(); TF._REGISTRY_TARGET["A0_Proposed"] = "dual"
print(f"{'ckpt':10s} {'LSD_dual':>8} {'PA':>6} | {'LSD(rc,Troom)':>13} {'LSD(peq,Tpref)':>14} | "
      f"{'<200Hz: rc-Troom':>16} {'peq':>6} {'Tpref':>6} | {'rc share':>8}")
for spec in args.ckpts:
    label, path = spec.split("=", 1)
    fc_max = args.fc_max
    if ":" in path and not Path(path).exists():
        path, fc_max = path.rsplit(":", 1); fc_max = float(fc_max)
    from model import DualObjectiveAdaptivePEQ
    m = DualObjectiveAdaptivePEQ(gain_max=12.0, fc_max=fc_max)
    ckpt_io.load_into(m, path, map_location=device, label=label)
    m.to(device).eval()
    rc, pq, pr = [], [], []
    with torch.no_grad():
        for b in ds.iter_batches(512, shuffle=False):
            o = TF.model_forward("A0_Proposed", m, b)
            rc.append(o["room_correction_db"].cpu().numpy()); pq.append(o["peq_response_db"].cpu().numpy())
            pr.append(o["pred_response_db"].cpu().numpy())
    rc, pq, pr = map(np.concatenate, (rc, pq, pr))
    heard = pr - room
    pa = np.mean((np.sign(heard) == np.sign(pref)).astype(float), -1)
    # low-frequency decomposition: preference shelf below 200 Hz
    rc_excess = (rc - room)[:, lo].mean(-1)        # what Stage A adds beyond pure room correction
    peq_lo = pq[:, lo].mean(-1)
    pref_lo = pref[:, lo].mean(-1)
    share = np.where(np.abs(pref_lo) > 1.0, rc_excess / np.where(np.abs(pref_lo) > 1e-6, pref_lo, 1e-6), np.nan)
    print(f"{label:10s} {lsd(pr,dual).mean():8.3f} {pa.mean():6.3f} | {lsd(rc,room).mean():13.3f} {lsd(pq, pref).mean():14.3f} | "
          f"{rc_excess.mean():16.2f} {peq_lo.mean():6.2f} {pref_lo.mean():6.2f} | {np.nanmedian(share):8.2f}")
    for mm in np.unique(mode):
        s = mode == mm
        print(f"    preset {int(mm)}: LSD {lsd(pr,dual)[s].mean():.3f}  PA {pa[s].mean():.3f}  "
              f"<200Hz rc-Troom {rc_excess[s].mean():+.2f}  peq {peq_lo[s].mean():+.2f}  Tpref {pref_lo[s].mean():+.2f}")
print("\nrc share = median over samples with |Tpref|>1 dB below 200 Hz of (room_corr - T_room)/T_pref: "
      "≈1 means Stage A carries the shelf, ≈0 means nobody does.")