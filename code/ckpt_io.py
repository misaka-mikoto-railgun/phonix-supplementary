"""
ckpt_io.py — checkpoint loading that fails instead of degrading
===============================================================
A wrong path used to be survivable: the loader printed a warning, kept the
randomly initialised weights, and the run produced a plausible-looking table
from an untrained model. Nothing downstream could tell the difference. These
helpers raise instead.

`load_into` keeps strict=False and then checks the returned key lists itself.
Switching to strict=True would also reject harmless buffer-registration
differences between torch versions; the explicit check reports exactly which
keys disagree, which is what one needs in order to fix the mismatch. Every
(model, checkpoint) pair this repository ships loads with missing=0 and
unexpected=0.

E1 (No Processing), E2 (Static Mode EQ) and E6 (DSP Analytical) are analytical
and hold no learned parameters, so they have no checkpoint at all; pass None
and nothing is loaded.
"""
from pathlib import Path

import torch

# Analytical baselines — no learned parameters, so no checkpoint exists.
NO_CHECKPOINT = {"E1", "E2", "E6", "E1_NoEQ", "E2_StaticEQ", "E6_DSP"}


def read_state(path, map_location="cpu"):
    """Return the state dict inside a checkpoint file, or raise."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    ck = torch.load(p, map_location=map_location, weights_only=False)
    return ck["model"] if isinstance(ck, dict) and "model" in ck else ck


def load_into(model, path, map_location="cpu", label=None):
    """Load weights into `model`, raising on a missing file or a key mismatch."""
    if path is None:
        return model
    missing, unexpected = model.load_state_dict(
        read_state(path, map_location), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch for {label or Path(path).name}: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}")
    return model
