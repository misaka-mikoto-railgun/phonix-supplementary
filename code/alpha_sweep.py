"""
alpha_sweep.py — α 민감도 분석 (사후 평가, 재학습 없음)

A0 체크포인트로 validation set을 한 번 추론한 뒤,
dual_target = α·room_target + (1-α)·pref_target 의 α 값만 바꿔가며
LSD / DMR / CosSim 을 측정.

결과:
  - checkpoints/full/alpha_sweep.json  → fig12_alpha_sensitivity()에 주입
  - figures/fig12_alpha_sensitivity.{png,pdf}

사용법:
  python alpha_sweep.py --data_dir ./data/dataset_v3 --ckpt_dir ./checkpoints/full
  python alpha_sweep.py --data_dir ./data/dataset_v3 --ckpt_dir ./checkpoints/full --out_dir ./figures
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import E3_NercessianMLP, E4_PepeCNN
from model import DualObjectiveAdaptivePEQ
from dataset_generator_v3 import PEQDataset
from model_aliases import checkpoint_name_candidates

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# ── 색상 ──────────────────────────────────────────────────────────────
C_PROPOSED  = "#C0392B"
C_BASELINE  = "#2980B9"
C_NEUTRAL   = "#555555"

FAIRNESS_MODELS = {
    "E3_Nercessian": E3_NercessianMLP,
    "E4_Pepe": E4_PepeCNN,
}

FAIRNESS_LABELS = {
    "E3_Nercessian": "E3 Nercessian",
    "E4_Pepe": "E4 Pepe",
}


# ── 메트릭 ────────────────────────────────────────────────────────────

def lsd_fn(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2, axis=-1)).mean())

def dmr_fn(pred: np.ndarray, pref: np.ndarray) -> float:
    return float(np.mean((np.sign(pred) == np.sign(pref)).astype(float)))

def cossim_fn(pred: np.ndarray, pref: np.ndarray) -> float:
    num   = np.sum(pred * pref, axis=-1)
    denom = np.linalg.norm(pred, axis=-1) * np.linalg.norm(pref, axis=-1) + 1e-8
    return float((num / denom).mean())


def expand_features(x: torch.Tensor, feature_dim: int) -> torch.Tensor:
    """
    Deterministically expand the saved 10-D acoustic features for E3/E4
    feature-dimension fairness checks.

    10: original features
    20: original + first temporal difference
    40: original + first difference + second difference + local temporal mean
    """
    base_dim = x.shape[-1]
    if feature_dim < base_dim:
        raise ValueError(f"feature_dim={feature_dim} is smaller than dataset feature dim={base_dim}")

    if feature_dim == base_dim:
        return x

    delta = torch.zeros_like(x)
    delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
    pieces = [x, delta]

    if feature_dim > base_dim * 2:
        delta2 = torch.zeros_like(x)
        delta2[:, 1:, :] = delta[:, 1:, :] - delta[:, :-1, :]

        local_mean = torch.empty_like(x)
        local_mean[:, 0, :] = (x[:, 0, :] + x[:, 1, :]) * 0.5
        local_mean[:, -1, :] = (x[:, -2, :] + x[:, -1, :]) * 0.5
        if x.shape[1] > 2:
            local_mean[:, 1:-1, :] = (x[:, :-2, :] + x[:, 1:-1, :] + x[:, 2:, :]) / 3.0
        pieces.extend([delta2, local_mean])

    expanded = torch.cat(pieces, dim=-1)
    if expanded.shape[-1] < feature_dim:
        pad = torch.zeros(*expanded.shape[:-1], feature_dim - expanded.shape[-1], device=x.device, dtype=x.dtype)
        expanded = torch.cat([expanded, pad], dim=-1)
    return expanded[..., :feature_dim]


def room_loss(pred: torch.Tensor, room_target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((pred - room_target) ** 2).mean(dim=-1) + 1e-8).mean()


def make_fairness_model(model_name: str, feature_dim: int, device: torch.device) -> nn.Module:
    if model_name not in FAIRNESS_MODELS:
        raise ValueError(f"Unsupported fairness model: {model_name}")
    return FAIRNESS_MODELS[model_name](in_dim=feature_dim, n_freqs=128, sample_rate=48000).to(device)


def forward_fairness_model(model: nn.Module, batch: dict, feature_dim: int) -> torch.Tensor:
    features = expand_features(batch["features"], feature_dim)
    return model(features)["pred_response_db"]


@torch.no_grad()
def evaluate_fairness_model(model: nn.Module, dataset: PEQDataset, feature_dim: int, batch_size: int) -> dict:
    model.eval()
    pred_parts, dual_parts, room_parts, pref_parts = [], [], [], []
    for batch in dataset.iter_batches(batch_size, shuffle=False):
        pred = forward_fairness_model(model, batch, feature_dim)
        pred_parts.append(pred.detach().cpu().numpy())
        dual_parts.append(batch["dual_target"].detach().cpu().numpy())
        room_parts.append(batch["room_target"].detach().cpu().numpy())
        pref_parts.append(batch["pref_target"].detach().cpu().numpy())

    pred = np.concatenate(pred_parts, axis=0)
    dual = np.concatenate(dual_parts, axis=0)
    room = np.concatenate(room_parts, axis=0)
    pref = np.concatenate(pref_parts, axis=0)
    heard = pred - room
    return {
        "lsd": lsd_fn(pred, dual),
        "lsd_room": lsd_fn(pred, room),
        "lsd_pref": lsd_fn(heard, pref),
        "dmr": dmr_fn(heard, pref),
        "cossim": cossim_fn(heard, pref),
        "n_samples": int(pred.shape[0]),
    }


def train_fairness_model(
    model_name: str,
    feature_dim: int,
    train_ds: PEQDataset,
    val_ds: PEQDataset,
    ckpt_path: Path,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
) -> tuple[nn.Module, dict]:
    model = make_fairness_model(model_name, feature_dim, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_lsd = float("inf")
    best_metrics = {}
    bad_epochs = 0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n = 0
        for batch in train_ds.iter_batches(batch_size, shuffle=True):
            pred = forward_fairness_model(model, batch, feature_dim)
            loss = room_loss(pred, batch["room_target"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bs = len(batch["room_target"])
            train_loss_sum += loss.item() * bs
            n += bs

        metrics = evaluate_fairness_model(model, val_ds, feature_dim, batch_size)
        train_loss_mean = train_loss_sum / max(n, 1)
        print(
            f"  {model_name} dim={feature_dim:>2} epoch={epoch:03d} "
            f"train_room_lsd={train_loss_mean:.4f} val_lsd={metrics['lsd']:.4f} "
            f"val_dmr={metrics['dmr']:.4f} val_cossim={metrics['cossim']:.4f}"
        )

        if metrics["lsd"] < best_lsd:
            best_lsd = metrics["lsd"]
            best_metrics = metrics
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": model_name,
                    "feature_dim": feature_dim,
                    "metrics": metrics,
                    "feature_expansion": "10: x; 20: x+delta; 40: x+delta+delta2+local_mean",
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop: no val_lsd improvement for {patience} epochs")
                break

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model, best_metrics


def run_fairness_table(args) -> None:
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    train_ds = PEQDataset(str(args.data_dir / "train"), device=str(device))
    val_ds = PEQDataset(str(args.data_dir / args.split), device=str(device))

    run_dir = args.ckpt_dir / "fairness_feature_dim"
    out_dir = args.table_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    raw = {
        "data_dir": str(args.data_dir),
        "split": args.split,
        "feature_dims": args.feature_dims,
        "reuse_main_checkpoints": not args.retrain_main,
        "feature_expansion": "10: x; 20: x+delta; 40: x+delta+delta2+local_mean",
        "models": {},
    }

    for model_name in args.fairness_models:
        raw["models"][model_name] = {}
        for feature_dim in args.feature_dims:
            main_ckpt = args.ckpt_dir / f"{model_name}.pt"
            fairness_ckpt = run_dir / f"{model_name}_in{feature_dim}.pt"
            use_main = feature_dim == 10 and not args.retrain_main and main_ckpt.exists()

            model = make_fairness_model(model_name, feature_dim, device)
            if use_main:
                state = torch.load(main_ckpt, map_location=device, weights_only=False)
                state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
                model.load_state_dict(state_dict)
                metrics = evaluate_fairness_model(model, val_ds, feature_dim, args.batch_size)
                source = str(main_ckpt)
                print(f"  evaluated existing checkpoint: {source}")
            elif fairness_ckpt.exists() and not args.force_retrain:
                state = torch.load(fairness_ckpt, map_location=device, weights_only=False)
                model.load_state_dict(state["model"])
                metrics = evaluate_fairness_model(model, val_ds, feature_dim, args.batch_size)
                source = str(fairness_ckpt)
                print(f"  evaluated cached checkpoint: {source}")
            else:
                model, metrics = train_fairness_model(
                    model_name=model_name,
                    feature_dim=feature_dim,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    ckpt_path=fairness_ckpt,
                    device=device,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    grad_clip=args.grad_clip,
                )
                source = str(fairness_ckpt)

            raw["models"][model_name][str(feature_dim)] = {
                "checkpoint": source,
                **{k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()},
            }
            rows.append({
                "Model": FAIRNESS_LABELS[model_name],
                "Feature Dim": f"{feature_dim} (main)" if feature_dim == 10 else str(feature_dim),
                "LSD ↓": f"{metrics['lsd']:.3f}",
                "dmr ↑": f"{metrics['dmr']:.3f}",
                "CosSim ↑": f"{metrics['cossim']:.3f}",
            })

    csv_path = out_dir / "tableA1_fairness.csv"
    tex_path = out_dir / "tableA1_fairness.tex"
    json_path = out_dir / "tableA1_fairness_raw.json"
    fieldnames = ["Model", "Feature Dim", "LSD ↓", "dmr ↑", "CosSim ↑"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    latex = "% Table A1: Feature dimension ablation for E3/E4 (generated from checkpoints)\n"
    latex += "\\begin{tabular}{llrrr}\n\\toprule\n"
    latex += "Model & Feature Dim & LSD $\\downarrow$ & dmr $\\uparrow$ & CosSim $\\uparrow$ \\\\\n\\midrule\n"
    for row in rows:
        latex += (
            f"{row['Model']} & {row['Feature Dim']} & {row['LSD ↓']} & "
            f"{row['dmr ↑']} & {row['CosSim ↑']} \\\\\n"
        )
    latex += "\\bottomrule\n\\end{tabular}\n"
    tex_path.write_text(latex, encoding="utf-8")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {tex_path}")


# ── 추론 ──────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, dataset, device, batch_size=512):
    """
    A0 모델로 val set 전체 추론 → pred / room_target / pref_target 수집
    """
    model.eval()
    preds, room_ts, pref_ts = [], [], []

    for batch in dataset.iter_batches(batch_size):
        f   = batch["features"].to(device)
        rr  = batch["room_response"].to(device)
        mi  = batch["mode_id"].to(device)
        bg  = batch["band_gains"].to(device)

        out  = model(f, rr, mi, bg)
        pred = out["pred_response_db"].cpu().numpy()

        preds.append(pred)
        room_ts.append(batch["room_target"].cpu().numpy())
        pref_ts.append(batch["pref_target"].cpu().numpy())

    return (
        np.concatenate(preds,   axis=0),   # (N, 128)
        np.concatenate(room_ts, axis=0),   # (N, 128)
        np.concatenate(pref_ts, axis=0),   # (N, 128)
    )


# ── sweep ─────────────────────────────────────────────────────────────

def run_sweep(pred, room_t, pref_t, alphas=ALPHAS):
    """
    α 마다 dual_target = α·room_t + (1-α)·pref_t 재계산 후 메트릭 측정.
    pred 는 고정(재추론 없음).
    """
    results = {"alphas": [], "lsd": [], "dmr": [], "cossim": []}
    print(f"\n{'α':>5}  {'LSD↓':>8}  {'DMR↑':>8}  {'CosSim↑':>10}")
    print("─" * 40)
    for alpha in alphas:
        dual_t = alpha * room_t + (1 - alpha) * pref_t
        l  = lsd_fn(pred, dual_t)
        d  = dmr_fn(pred, pref_t)   # pref_t 고정 (선호도 정렬 측정)
        cs = cossim_fn(pred, pref_t)
        results["alphas"].append(alpha)
        results["lsd"].append(round(l,  4))
        results["dmr"].append(round(d,  4))
        results["cossim"].append(round(cs, 4))
        print(f"{alpha:>5.2f}  {l:>8.4f}  {d:>8.4f}  {cs:>10.4f}")
    return results


# ── Figure ────────────────────────────────────────────────────────────

def plot_sweep(sweep: dict, out_dir: Path, chosen_alpha: float = 0.6):
    alphas   = np.array(sweep["alphas"])
    lsd_vals = np.array(sweep["lsd"])
    dmr_vals = np.array(sweep["dmr"])
    cos_vals = np.array(sweep["cossim"])
    chosen   = int(np.argmin(np.abs(alphas - chosen_alpha)))

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    fig.subplots_adjust(wspace=0.38)

    # (a) LSD vs α
    axes[0].plot(alphas, lsd_vals, "o-", color=C_BASELINE, lw=1.6, ms=5)
    axes[0].axvline(alphas[chosen], color=C_PROPOSED, ls="--", lw=1.2,
                    label=f"α={alphas[chosen]:.1f} (used)")
    axes[0].set_xlabel("α (room weight)", fontsize=9)
    axes[0].set_ylabel("Dual-target LSD (dB)", fontsize=9)
    axes[0].set_title("(a) LSD vs α", fontsize=9)
    axes[0].legend(fontsize=7.5)
    axes[0].grid(True, ls="--", alpha=0.4)

    # (b) DMR vs α
    axes[1].plot(alphas, dmr_vals, "o-", color=C_PROPOSED, lw=1.6, ms=5)
    axes[1].axvline(alphas[chosen], color=C_PROPOSED, ls="--", lw=1.2,
                    label=f"α={alphas[chosen]:.1f} (used)")
    axes[1].set_xlabel("α (room weight)", fontsize=9)
    axes[1].set_ylabel("DMR (Directional Match Rate)", fontsize=9)
    axes[1].set_title("(b) DMR vs α", fontsize=9)
    axes[1].legend(fontsize=7.5)
    axes[1].grid(True, ls="--", alpha=0.4)

    # (c) Trade-off (LSD vs DMR)
    axes[2].plot(lsd_vals, dmr_vals, "o-", color=C_NEUTRAL, lw=1.2, ms=4)
    for a, l, d in zip(alphas, lsd_vals, dmr_vals):
        axes[2].annotate(f"α={a:.1f}", (l, d),
                         fontsize=7, xytext=(3, 3),
                         textcoords="offset points")
    axes[2].scatter([lsd_vals[chosen]], [dmr_vals[chosen]],
                    c=C_PROPOSED, s=110, zorder=5, marker="*",
                    label=f"α={alphas[chosen]:.1f} (Pareto knee)")
    axes[2].set_xlabel("Dual-target LSD (dB)", fontsize=9)
    axes[2].set_ylabel("DMR", fontsize=9)
    axes[2].set_title("(c) LSD–DMR trade-off", fontsize=9)
    axes[2].legend(fontsize=7.5)
    axes[2].grid(True, ls="--", alpha=0.4)

    fig.suptitle(
        "Fig. 12  Sensitivity Analysis of Dual-Objective Weight α\n"
        "(A0 model fixed; dual target recomputed per α)",
        fontsize=9, y=1.03
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig12_alpha_sensitivity.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved → {out_dir / 'fig12_alpha_sensitivity.png'}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="α sensitivity sweep (no retraining)")
    parser.add_argument("--task", choices=["alpha", "fairness"], default="alpha",
                        help="alpha: A0 α sweep; fairness: train/evaluate E3/E4 feature-dim table")
    parser.add_argument("--data_dir",  type=Path, default=Path("./data/dataset_v3"))
    parser.add_argument("--ckpt_dir",  type=Path, default=Path("./checkpoints/full"))
    parser.add_argument("--out_dir",   type=Path, default=Path("./figures"))
    parser.add_argument("--split",     default="val", choices=["val", "test"])
    parser.add_argument("--batch_size",type=int,  default=512)
    parser.add_argument("--alphas",    nargs="+", type=float,
                        default=ALPHAS,
                        help="α 값 목록 (기본: 0.0 0.2 0.4 0.6 0.8 1.0)")
    parser.add_argument("--chosen_alpha", type=float, default=0.6,
                        help="Pareto knee 강조 표시할 α (기본: 0.6)")
    parser.add_argument("--no_cuda",   action="store_true")
    parser.add_argument("--feature_dims", nargs="+", type=int, default=[10, 20, 40])
    parser.add_argument("--fairness_models", nargs="+", default=["E3_Nercessian", "E4_Pepe"],
                        choices=sorted(FAIRNESS_MODELS))
    parser.add_argument("--table_out_dir", type=Path, default=Path("./paper_outputs/tables"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--retrain_main", action="store_true",
                        help="Also retrain 10-D E3/E4 instead of reusing main checkpoints")
    args = parser.parse_args()

    if args.task == "fairness":
        run_fairness_table(args)
        print("\nDone.")
        return

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available()
                          else "cuda")
    print(f"Device: {device}")

    # ── 데이터 로드 ──────────────────────────────────────────────────
    split_dir = args.data_dir / args.split
    print(f"\n데이터 로드: {split_dir}")
    dataset = PEQDataset(str(split_dir), device=str(device))

    # ── 모델 로드 ────────────────────────────────────────────────────
    ckpt_path = None
    for candidate in checkpoint_name_candidates("A0_Proposed"):
        path = args.ckpt_dir / f"{candidate}.pt"
        if path.exists():
            ckpt_path = path
            break
    if ckpt_path is None:
        tried = ", ".join(f"{candidate}.pt" for candidate in checkpoint_name_candidates("A0_Proposed"))
        raise FileNotFoundError(f"체크포인트 없음. 시도한 이름: {tried}")

    model = DualObjectiveAdaptivePEQ().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"A0_Proposed 로드: {ckpt_path}")

    # ── 추론 ─────────────────────────────────────────────────────────
    print("\n추론 중...")
    pred, room_t, pref_t = collect_predictions(
        model, dataset, device, args.batch_size
    )
    print(f"  pred shape: {pred.shape}")

    # ── sweep ────────────────────────────────────────────────────────
    print("\nα sweep:")
    sweep = run_sweep(pred, room_t, pref_t, args.alphas)

    # Pareto knee 자동 탐지 (LSD 감소 한계 + DMR 감소 시작점)
    lsd_arr = np.array(sweep["lsd"])
    dmr_arr = np.array(sweep["dmr"])
    alphas  = np.array(sweep["alphas"])
    # 각 α에서 LSD와 DMR의 정규화된 합
    lsd_norm = (lsd_arr - lsd_arr.min()) / (lsd_arr.max() - lsd_arr.min() + 1e-8)
    dmr_norm = (dmr_arr.max() - dmr_arr) / (dmr_arr.max() - dmr_arr.min() + 1e-8)
    knee_idx = int(np.argmin(lsd_norm + dmr_norm))
    print(f"\nPareto knee 추정: α={alphas[knee_idx]:.2f} "
          f"(LSD={lsd_arr[knee_idx]:.4f}, DMR={dmr_arr[knee_idx]:.4f})")

    # ── 저장 ─────────────────────────────────────────────────────────
    sweep["pareto_knee_alpha"] = float(alphas[knee_idx])
    sweep["pareto_knee_lsd"]   = float(lsd_arr[knee_idx])
    sweep["pareto_knee_dmr"]   = float(dmr_arr[knee_idx])

    # results.json에 주입할 수 있도록 별도 저장
    json_path = args.ckpt_dir / "alpha_sweep.json"
    with open(json_path, "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"\n결과 저장: {json_path}")

    # experiments.py의 results["A0_Proposed"]["alpha_sweep"] 에 넣으려면:
    results_path = args.ckpt_dir / "results_final.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        if "A0_Proposed" in results:
            results["A0_Proposed"]["alpha_sweep"] = sweep
        if "A0_Full" in results:
            results["A0_Full"]["alpha_sweep"] = sweep
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print("results_final.json 업데이트 완료")

    # ── Figure ───────────────────────────────────────────────────────
    plot_sweep(sweep, args.out_dir, chosen_alpha=args.chosen_alpha)
    print("\nDone.")


if __name__ == "__main__":
    main()
