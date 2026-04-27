import math

import numpy as np
import torch


DEFAULT_N_FREQS = 128
DEFAULT_F_MIN = 20.0
DEFAULT_F_MAX = 24000.0
DEFAULT_SPACING = "log"


def _validate_frequency_grid_args(n_freqs: int, f_min: float, f_max: float, spacing: str) -> None:
    if n_freqs <= 0:
        raise ValueError(f"n_freqs must be positive, got {n_freqs}")
    if f_min <= 0 or f_max <= 0:
        raise ValueError(f"f_min and f_max must be positive, got f_min={f_min}, f_max={f_max}")
    if f_max <= f_min:
        raise ValueError(f"f_max must be greater than f_min, got f_min={f_min}, f_max={f_max}")
    if spacing not in {"log", "linear"}:
        raise ValueError(f"Unsupported spacing '{spacing}'. Expected 'log' or 'linear'.")


def make_frequency_grid_torch(
    n_freqs: int = DEFAULT_N_FREQS,
    f_min: float = DEFAULT_F_MIN,
    f_max: float = DEFAULT_F_MAX,
    spacing: str = DEFAULT_SPACING,
    device=None,
    dtype=torch.float32,
):
    _validate_frequency_grid_args(n_freqs, f_min, f_max, spacing)
    if spacing == "log":
        grid = torch.exp(
            torch.linspace(
                math.log(float(f_min)),
                math.log(float(f_max)),
                int(n_freqs),
                device=device,
                dtype=torch.float64,
            )
        )
    else:
        grid = torch.linspace(
            float(f_min),
            float(f_max),
            int(n_freqs),
            device=device,
            dtype=torch.float64,
        )
    return grid.to(dtype=dtype)


def make_frequency_grid_np(
    n_freqs: int = DEFAULT_N_FREQS,
    f_min: float = DEFAULT_F_MIN,
    f_max: float = DEFAULT_F_MAX,
    spacing: str = DEFAULT_SPACING,
    dtype=np.float32,
):
    _validate_frequency_grid_args(n_freqs, f_min, f_max, spacing)
    if spacing == "log":
        grid = np.geomspace(float(f_min), float(f_max), int(n_freqs), dtype=np.float64)
    else:
        grid = np.linspace(float(f_min), float(f_max), int(n_freqs), dtype=np.float64)
    return grid.astype(dtype, copy=False)
