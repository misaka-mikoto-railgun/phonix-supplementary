from __future__ import annotations

"""
Export per-sample prediction dumps for track-level evaluation.

This script follows the model/checkpoint path used in `train_full.py` and the
evaluation flow used in `experiments_fixed_updated.py`, then saves `.npz`
payloads compatible with `track_level_eval.py`. It can also run the A0-vs-model
track-level paired tests needed for reviewer-facing ablation/baseline checks.

Required output keys:
    pred      : [N, F]
    target    : [N, F]
    track_id  : [N]

Optional keys saved when present:
    room_id, mode_id, pair_id, clip_start_sample, room_target, pref_target, heard_pred
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from frequency_grid import make_frequency_grid_np
from model_aliases import canonical_model_name, checkpoint_name_candidates
from train_full import build_registry


DEFAULT_REVIEWER_MODELS = [
    "A0_Proposed",
    "A1_NoRoomInput",
    "A2_withPrefLoss",
    "A3_NoPrefInput",
    "E3_Nercessian",
    "E4_Pepe",
]

DEFAULT_REVIEWER_CANDIDATES = [
    "A1_NoRoomInput",
    "A2_withPrefLoss",
    "A3_NoPrefInput",
    "E3_Nercessian",
    "E4_Pepe",
]

COMPARISON_LABELS = {
    "A1_NoRoomInput": "A0 vs A1 (Room Input)",
    "A2_withPrefLoss": "A0 vs A2 (with Pref Loss)",
    "A3_NoPrefInput": "A0 vs A3 (Pref Input)",
    "E3_Nercessian": "A0 vs E3 (Nercessian)",
    "E4_Pepe": "A0 vs E4 (Pepe)",
}


class TrackAwarePEQDataset:
    BASE_KEYS = [
        "features",
        "room_response",
        "mode_id",
        "band_gains",
        "room_target",
        "pref_target",
        "dual_target",
    ]
    OPTIONAL_KEYS = ["features_clean", "track_id", "room_id", "clip_start_sample", "pair_id"]
    INT_KEYS = {"mode_id", "track_id", "room_id", "clip_start_sample", "pair_id"}

    def __init__(self, split_dir: str | Path, device: str = "cpu"):
        self.split_dir = Path(split_dir)
        self.device = torch.device(device)

        meta_path = self.split_dir / "meta.json"
        self.meta = {}
        self.target_freqs: np.ndarray | None = None
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                self.meta = json.load(f)
            if "target_freqs" in self.meta:
                self.target_freqs = np.asarray(self.meta["target_freqs"], dtype=np.float32)
            elif all(k in self.meta for k in ("n_freqs", "freq_min", "freq_max", "freq_spacing")):
                self.target_freqs = make_frequency_grid_np(
                    n_freqs=int(self.meta["n_freqs"]),
                    f_min=float(self.meta["freq_min"]),
                    f_max=float(self.meta["freq_max"]),
                    spacing=str(self.meta["freq_spacing"]),
                )
            elif "config" in self.meta:
                cfg = self.meta["config"]
                self.target_freqs = make_frequency_grid_np(
                    n_freqs=int(cfg.get("n_freqs", 128)),
                    f_min=float(cfg.get("freq_min", 20.0)),
                    f_max=float(cfg.get("freq_max", 24000.0)),
                    spacing=str(cfg.get("freq_spacing", "log")),
                )

        chunk_files = sorted(self.split_dir.glob("chunk_*.npz"))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in: {self.split_dir}")

        arrays = {k: [] for k in self.BASE_KEYS + self.OPTIONAL_KEYS}
        for chunk_path in chunk_files:
            data = np.load(chunk_path, allow_pickle=False)
            if self.target_freqs is None and "target_freqs" in data:
                self.target_freqs = np.asarray(data["target_freqs"], dtype=np.float32)
            for key in self.BASE_KEYS:
                if key not in data:
                    raise KeyError(f"Missing required key '{key}' in {chunk_path}")
                arrays[key].append(data[key])
            for key in self.OPTIONAL_KEYS:
                if key in data:
                    arrays[key].append(data[key])

        self.data: Dict[str, torch.Tensor] = {}
        for key, parts in arrays.items():
            if not parts:
                continue
            tensor = torch.from_numpy(np.concatenate(parts, axis=0))
            if key in self.INT_KEYS:
                tensor = tensor.long()
            else:
                tensor = tensor.float()
            self.data[key] = tensor.to(self.device)

        self.n_samples = len(self.data["features"])
        mem_mb = sum(t.element_size() * t.numel() for t in self.data.values()) / (1024 * 1024)
        print(f"Loaded {self.n_samples:,} samples from {self.split_dir} ({mem_mb:.1f} MB on {self.device})")

    def __len__(self) -> int:
        return self.n_samples

    def iter_batches(self, batch_size: int = 512, shuffle: bool = False):
        indices = torch.arange(self.n_samples, device=self.device)
        for start in range(0, self.n_samples, batch_size):
            idx = indices[start:start + batch_size]
            yield {key: value[idx] for key, value in self.data.items()}


def build_export_registry() -> Dict[str, dict]:
    registry = build_registry()
    return registry


def _canonical_model_name(name: str) -> str:
    return canonical_model_name(name)


def parse_model_names(arg: str, registry: Dict[str, dict], default: List[str] | None = None) -> List[str]:
    if arg == "reviewer":
        names = list(default or DEFAULT_REVIEWER_MODELS)
    elif arg == "all":
        return list(registry.keys())
    else:
        names = [_canonical_model_name(name.strip()) for name in arg.split(",") if name.strip()]

    invalid = [name for name in names if name not in registry]
    if invalid:
        raise ValueError(f"Unknown model names: {invalid}")

    return names


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def model_forward(name: str, model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    features = batch["features"]
    features_clean = batch.get("features_clean")
    room_response = batch["room_response"]
    mode_id = batch["mode_id"]
    band_gains = batch["band_gains"]

    if name == "E1_NoEQ":
        return model(features)
    if name == "E2_StaticEQ":
        return model(features, mode_id=mode_id)
    if name in ("E3_Nercessian", "E4_Pepe"):
        return model(features)
    if name == "E5_Sequential":
        return model(features, mode_id=mode_id)
    if name == "E6_DSP":
        return model(features, room_response=room_response, mode_id=mode_id)
    if name == "A1_NoRoomInput":
        return model(features_clean if features_clean is not None else features, room_response, mode_id, band_gains)
    return model(features, room_response, mode_id, band_gains)


def load_checkpoint(
    model_name: str,
    model: nn.Module,
    ckpt_dir: Path,
    required: bool,
    checkpoint_name: str | None = None,
) -> None:
    candidate_names = [checkpoint_name] if checkpoint_name else checkpoint_name_candidates(model_name)
    ckpt_path = None
    for candidate in candidate_names:
        path = ckpt_dir / f"{candidate}.pt"
        if path.exists():
            ckpt_path = path
            break
    if ckpt_path is None:
        expected = ", ".join(f"{candidate}.pt" for candidate in candidate_names)
        if required:
            raise FileNotFoundError(f"Checkpoint not found for {model_name}. Tried: {expected}")
        print(f"Checkpoint not found for {model_name}; tried {expected}; using model as initialized")
        return

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"  missing keys: {len(missing)}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}")


@torch.no_grad()
def build_prediction_payload(
    name: str,
    model: nn.Module,
    dataset: TrackAwarePEQDataset,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    model.eval()
    pred_parts: List[np.ndarray] = []
    target_parts: List[np.ndarray] = []
    room_target_parts: List[np.ndarray] = []
    pref_target_parts: List[np.ndarray] = []
    track_id_parts: List[np.ndarray] = []
    mode_id_parts: List[np.ndarray] = []
    room_id_parts: List[np.ndarray] = []
    pair_id_parts: List[np.ndarray] = []
    clip_start_parts: List[np.ndarray] = []

    for batch in dataset.iter_batches(batch_size=batch_size, shuffle=False):
        out = model_forward(name, model, batch)
        pred = out["pred_response_db"]
        target = batch["dual_target"]

        pred_parts.append(pred.detach().cpu().numpy().astype(np.float32))
        target_parts.append(target.detach().cpu().numpy().astype(np.float32))
        room_target_parts.append(batch["room_target"].detach().cpu().numpy().astype(np.float32))
        pref_target_parts.append(batch["pref_target"].detach().cpu().numpy().astype(np.float32))

        if "track_id" not in batch:
            raise KeyError(
                f"Dataset split '{dataset.split_dir}' does not contain 'track_id'. "
                "Use a track-level dataset generated with track metadata."
            )
        track_id_parts.append(batch["track_id"].detach().cpu().numpy().astype(np.int32))
        mode_id_parts.append(batch["mode_id"].detach().cpu().numpy().astype(np.int32))

        if "room_id" in batch:
            room_id_parts.append(batch["room_id"].detach().cpu().numpy().astype(np.int32))
        if "pair_id" in batch:
            pair_id_parts.append(batch["pair_id"].detach().cpu().numpy().astype(np.int32))
        if "clip_start_sample" in batch:
            clip_start_parts.append(batch["clip_start_sample"].detach().cpu().numpy().astype(np.int32))

    pred = np.concatenate(pred_parts, axis=0)
    room_target = np.concatenate(room_target_parts, axis=0)
    payload: Dict[str, np.ndarray] = {
        "pred": pred,
        "target": np.concatenate(target_parts, axis=0),
        "dual_target": np.concatenate(target_parts, axis=0),
        "track_id": np.concatenate(track_id_parts, axis=0),
        "mode_id": np.concatenate(mode_id_parts, axis=0),
        "room_target": room_target,
        "pref_target": np.concatenate(pref_target_parts, axis=0),
        "heard_pred": (pred - room_target).astype(np.float32),
    }
    if dataset.target_freqs is not None:
        payload["target_freqs"] = dataset.target_freqs.astype(np.float32)

    if room_id_parts:
        payload["room_id"] = np.concatenate(room_id_parts, axis=0)
    if pair_id_parts:
        payload["pair_id"] = np.concatenate(pair_id_parts, axis=0)
    if clip_start_parts:
        payload["clip_start_sample"] = np.concatenate(clip_start_parts, axis=0)

    return payload


def save_prediction_dump_npz(
    model_name: str,
    model: nn.Module,
    dataset: TrackAwarePEQDataset,
    batch_size: int,
    out_path: Path,
) -> Dict[str, np.ndarray]:
    payload = build_prediction_payload(model_name, model, dataset, batch_size)
    np.savez_compressed(out_path, **payload)
    print(f"Saved prediction dump: {out_path}")
    return payload


def validate_group_key(dataset: TrackAwarePEQDataset, group_key: str) -> None:
    if group_key not in dataset.data:
        raise KeyError(
            f"'{group_key}' is missing from {dataset.split_dir}. "
            "Choose a compatible split/group-key pair for track-level evaluation."
        )


def summary_to_jsonable(summary) -> Dict:
    return {
        "group_key": summary.group_key,
        "sample_level_n": summary.sample_level_n,
        "n_groups": summary.n_groups,
        "metrics": {k: asdict(v) for k, v in summary.metrics.items()},
    }


def format_p_value(p: float | None) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def write_track_level_comparisons(
    payloads: Dict[str, Dict[str, np.ndarray]],
    baseline_name: str,
    candidate_names: List[str],
    group_key: str,
    n_boot: int,
    seed: int,
    out_dir: Path,
    baseline_seed: str | None = None,
) -> None:
    from track_level_eval import compare_two_prediction_sets

    if baseline_name not in payloads:
        raise KeyError(f"Baseline payload not exported: {baseline_name}")

    rows = []
    baseline = payloads[baseline_name]
    for candidate_name in candidate_names:
        if candidate_name not in payloads:
            raise KeyError(f"Candidate payload not exported: {candidate_name}")

        summary = compare_two_prediction_sets(
            baseline,
            payloads[candidate_name],
            group_key=group_key,
            n_boot=n_boot,
            seed=seed,
        )

        label = COMPARISON_LABELS.get(candidate_name, f"{baseline_name} vs {candidate_name}")
        safe_label = (
            label.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        json_path = out_dir / f"{safe_label}_{group_key}.json"
        payload = summary_to_jsonable(summary)
        if baseline_seed is not None:
            payload["baseline_seed"] = baseline_seed
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved paired stats: {json_path}")

        for metric_name in ["lsd", "dmr", "cossim"]:
            result = summary.metrics[metric_name]
            rows.append({
                "comparison": label,
                "metric": metric_name.upper() if metric_name != "dmr" else "DMR",
                "group_key": group_key,
                "n_groups": result.n_groups,
                "baseline_mean": f"{result.baseline_mean:.6f}",
                "candidate_mean": f"{result.candidate_mean:.6f}",
                "mean_diff": f"{result.mean_diff:+.6f}",
                "ci_low": f"{result.ci_low:+.6f}",
                "ci_high": f"{result.ci_high:+.6f}",
                "p_ttest": format_p_value(result.p_ttest),
                "p_wilcoxon": format_p_value(result.p_wilcoxon),
                "cohens_dz": "" if result.cohens_dz is None else f"{result.cohens_dz:+.6f}",
                "win_rate": f"{100.0 * result.win_rate:.2f}%",
                "conditional_win_rate": (
                    "" if result.conditional_win_rate is None else f"{100.0 * result.conditional_win_rate:.2f}%"
                ),
            })

    csv_path = out_dir / f"{baseline_name}_track_level_comparisons_{group_key}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved paired-stats table: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export track-level prediction dumps for track_level_eval.py")
    parser.add_argument("--data_dir", default="./data/dataset_v3", help="Dataset root containing split folders")
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test_synth", "test_real", "paired_mode_test"],
    )
    parser.add_argument("--ckpt_dir", default="./checkpoints/full", help="Checkpoint directory")
    parser.add_argument("--out_dir", default="./eval_reports/track_level_npz", help="Output directory")
    parser.add_argument(
        "--models",
        default="reviewer",
        help="Comma-separated model names, short names (A0,A1,A2,A3,E3,E4), 'reviewer', or 'all'",
    )
    parser.add_argument("--baseline", default="A0", help="Baseline model for paired stats")
    parser.add_argument("--baseline-seed", default=None,
                        help="Training seed of the baseline/candidate checkpoints, recorded in the JSON")
    parser.add_argument(
        "--candidates",
        default="A1,A2,A3,E3,E4",
        help="Comma-separated candidate models for paired stats, or 'none'",
    )
    parser.add_argument("--group-key", default="track_id", choices=["track_id", "room_id", "pair_id"])
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    split_dir = data_dir / args.split
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = TrackAwarePEQDataset(split_dir=split_dir, device=str(device))
    registry = build_export_registry()
    model_names = parse_model_names(args.models, registry, default=DEFAULT_REVIEWER_MODELS)
    baseline_name = _canonical_model_name(args.baseline)
    candidate_names = [] if args.candidates == "none" else parse_model_names(args.candidates, registry)

    missing_for_stats = [name for name in [baseline_name, *candidate_names] if name not in model_names]
    if missing_for_stats:
        model_names.extend(missing_for_stats)

    validate_group_key(dataset, "track_id")
    validate_group_key(dataset, args.group_key)

    payloads: Dict[str, Dict[str, np.ndarray]] = {}
    for model_name in model_names:
        entry = registry[model_name]
        model = entry["model"].to(device)
        requires_checkpoint = entry["loss"] is not None
        load_checkpoint(
            model_name,
            model,
            ckpt_dir,
            required=requires_checkpoint,
            checkpoint_name=entry.get("checkpoint_name"),
        )

        out_path = out_dir / f"{model_name}_{args.split}_preds.npz"
        payloads[model_name] = save_prediction_dump_npz(
            model_name=model_name,
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            out_path=out_path,
        )

    if candidate_names:
        write_track_level_comparisons(
            payloads=payloads,
            baseline_name=baseline_name,
            candidate_names=candidate_names,
            group_key=args.group_key,
            n_boot=args.n_boot,
            seed=args.seed,
            out_dir=out_dir,
            baseline_seed=args.baseline_seed,
        )


if __name__ == "__main__":
    main()
