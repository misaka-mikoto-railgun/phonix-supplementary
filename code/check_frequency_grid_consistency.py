import argparse
import json
from pathlib import Path

import numpy as np

from arch_variants import AC1_TCNBiLSTM
from baselines import DifferentiablePEQResponse
from dataset_generator_v4_tracklevel import DatasetConfig, get_target_freqs
from frequency_grid import make_frequency_grid_np
from model import DualObjectiveAdaptivePEQ, DualObjectiveEQLoss


def _resolve_meta_path(data_dir: Path) -> Path:
    if (data_dir / "meta.json").exists():
        return data_dir / "meta.json"
    for split_name in ("test_synth", "val", "train", "test_real", "paired_mode_switch"):
        candidate = data_dir / split_name / "meta.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find meta.json under {data_dir}")


def _load_dataset_grid(meta_path: Path) -> tuple[dict, np.ndarray]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg_dict = dict(meta.get("config", {}))
    if "freq_spacing" in meta:
        cfg_dict["freq_spacing"] = meta["freq_spacing"]
    if "freq_min" in meta:
        cfg_dict["freq_min"] = meta["freq_min"]
    if "freq_max" in meta:
        cfg_dict["freq_max"] = meta["freq_max"]
    if "n_freqs" in meta:
        cfg_dict["n_freqs"] = meta["n_freqs"]

    cfg = DatasetConfig(**cfg_dict)
    assert cfg.freq_spacing == "log", f"Dataset freq_spacing must be 'log', got '{cfg.freq_spacing}'"

    dataset_grid = None
    if "target_freqs" in meta:
        dataset_grid = np.asarray(meta["target_freqs"], dtype=np.float32)
    else:
        chunk_files = sorted(meta_path.parent.glob("chunk_*.npz"))
        for chunk_path in chunk_files:
            with np.load(chunk_path, allow_pickle=False) as chunk:
                if "target_freqs" in chunk:
                    dataset_grid = np.asarray(chunk["target_freqs"], dtype=np.float32)
                    break

    generator_grid = get_target_freqs(cfg)
    if dataset_grid is None:
        dataset_grid = generator_grid
    else:
        assert np.allclose(dataset_grid, generator_grid), "Dataset target_freqs and generator grid do not match"

    utility_grid = make_frequency_grid_np(
        n_freqs=cfg.n_freqs,
        f_min=cfg.freq_min,
        f_max=cfg.freq_max,
        spacing=cfg.freq_spacing,
    )
    assert np.allclose(dataset_grid, utility_grid), "Dataset grid and frequency_grid utility do not match"
    return cfg_dict, dataset_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Check frequency-grid consistency across dataset/model/baselines.")
    parser.add_argument("--data_dir", required=True, help="Dataset root or split directory containing meta.json")
    args = parser.parse_args()

    meta_path = _resolve_meta_path(Path(args.data_dir))
    cfg_dict, dataset_grid = _load_dataset_grid(meta_path)

    n_freqs = int(cfg_dict.get("n_freqs", 128))
    f_min = float(cfg_dict.get("freq_min", 20.0))
    f_max = float(cfg_dict.get("freq_max", 24000.0))
    sample_rate = int(cfg_dict.get("sample_rate", 48000))
    freq_spacing = str(cfg_dict.get("freq_spacing", "log"))

    model = DualObjectiveAdaptivePEQ(
        n_freqs=n_freqs,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        freq_spacing=freq_spacing,
    )
    baseline_peq = DifferentiablePEQResponse(
        sample_rate=sample_rate,
        n_freqs=n_freqs,
        f_min=f_min,
        f_max=f_max,
        freq_spacing=freq_spacing,
    )
    ac_model = AC1_TCNBiLSTM(
        n_freqs=n_freqs,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        freq_spacing=freq_spacing,
    )
    loss = DualObjectiveEQLoss(
        n_freqs=n_freqs,
        f_min=f_min,
        f_max=f_max,
        freq_spacing=freq_spacing,
    )

    assert np.allclose(dataset_grid, model.target_freqs.detach().cpu().numpy()), "Model target_freqs mismatch"
    assert np.allclose(dataset_grid, model.peq_response.freqs.detach().cpu().numpy()), "Model PEQ freqs mismatch"
    assert np.allclose(dataset_grid, baseline_peq.freqs.detach().cpu().numpy()), "Baseline PEQ freqs mismatch"
    assert np.allclose(dataset_grid, ac_model.target_freqs.detach().cpu().numpy()), "AC model target_freqs mismatch"
    assert np.allclose(dataset_grid, loss.freqs.detach().cpu().numpy()), "Loss frequency grid mismatch"

    print("Frequency grid consistency check passed.")


if __name__ == "__main__":
    main()
