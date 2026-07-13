"""
AC Fitting  --  Option C: Biquad-Constrained Retraining
========================================================
AC1/AC2/AC3 에서 dense 128-bin pref_res_head 를 7-band biquad head 로
교체하여 처음부터 재학습. biquad 제약이 각 아키텍처에 미치는 표현 패널티 측정.

동일 손실 (a0_proposed_loss: pref 손실 없음) + 동일 데이터 사용.

결과 해석
---------
  AC_Biquad LSD vs AC_raw LSD = representation penalty from biquad constraint
  AC_Biquad LSD vs A0 LSD     = 재학습 후에도 A0 대비 얼마나 남는지

Usage
-----
  # AC2_GRU_Biquad 학습 + 평가 (~1시간)
  python ac_fitting_C.py --data_dir ./data/dataset_v3 --ckpt_dir ./checkpoints/full

  # AC1, AC3 학습 (각 ~14분 예상)
  python ac_fitting_C.py --arch AC1_BiLSTM --data_dir ./data/dataset_v3
  python ac_fitting_C.py --arch AC3_Conformer --data_dir ./data/dataset_v3

  # 체크포인트 있으면 학습 건너뜀
  python ac_fitting_C.py --skip_train --arch AC2_GRU

  # 빠른 연기 테스트 (epochs=3)
  python ac_fitting_C.py --epochs 3 --data_dir ./data/dataset_v3
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from arch_biquad import BIQUAD_REGISTRY
from model import DualObjectiveEQLoss
from dataset_generator_v4_tracklevel import PEQDataset

ARCH_CHOICES = ["AC1_BiLSTM", "AC2_GRU", "AC3_Conformer"]
JND_THRESHOLD = 0.5   # dB (Toole & Olive 1988)


# ──────────────────────────────────────────────────────────
# 손실: A0_Proposed 와 동일 (pref 손실 없음)
# ──────────────────────────────────────────────────────────

def build_loss() -> DualObjectiveEQLoss:
    return DualObjectiveEQLoss(
        lambda_final=1.0,
        lambda_room=0.35,
        lambda_pref_res=0.0,
        lambda_shape=0.20,
        lambda_grad=0.20,
        lambda_curv=0.08,
        lambda_dir=0.0,
        lambda_mean=0.03,
        use_perceptual=True,
        mag_weight_alpha=0.10,
        grad_weight_beta=0.30,
    )


# ──────────────────────────────────────────────────────────
# 학습 / 검증 루프
# ──────────────────────────────────────────────────────────

def _fwd(model, batch):
    return model(
        batch["features"],
        batch["room_response"],
        batch["mode_id"],
        batch["band_gains"],
    )


def _lsd(pred, tgt):
    return torch.sqrt(((pred - tgt) ** 2).mean(dim=-1)).mean().item()


def train_epoch(model, criterion, dataset, optimizer, batch_size, grad_clip):
    model.train()
    total_loss = total_lsd = n = 0
    for batch in dataset.iter_batches(batch_size, shuffle=True):
        out  = _fwd(model, batch)
        pred = out["pred_response_db"]
        loss_dict = criterion(
            pred_response_db   = pred,
            dual_target_db     = batch["dual_target"],
            pref_target_db     = batch.get("pref_target"),
            room_target_db     = batch.get("room_target"),
            room_correction_db = out.get("room_correction_db"),
            peq_response_db    = out.get("peq_response_db"),
            pref_curve_db      = out.get("pref_curve_db"),
        )
        optimizer.zero_grad(set_to_none=True)
        loss_dict["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = len(batch["dual_target"])
        total_loss += loss_dict["loss"].item() * bs
        total_lsd  += _lsd(pred.detach(), batch["dual_target"]) * bs
        n += bs
    return {"train_loss": total_loss / max(n, 1), "train_lsd": total_lsd / max(n, 1)}


@torch.no_grad()
def validate(model, criterion, dataset, batch_size):
    model.eval()
    total_loss = total_lsd = n = 0
    for batch in dataset.iter_batches(batch_size, shuffle=False):
        out  = _fwd(model, batch)
        pred = out["pred_response_db"]
        ld = criterion(
            pred_response_db   = pred,
            dual_target_db     = batch["dual_target"],
            pref_target_db     = batch.get("pref_target"),
            room_target_db     = batch.get("room_target"),
            room_correction_db = out.get("room_correction_db"),
            peq_response_db    = out.get("peq_response_db"),
            pref_curve_db      = out.get("pref_curve_db"),
        )
        bs = len(batch["dual_target"])
        total_loss += ld["loss"].item() * bs
        total_lsd  += _lsd(pred, batch["dual_target"]) * bs
        n += bs
    return {"val_loss": total_loss / max(n, 1), "val_lsd": total_lsd / max(n, 1)}


@torch.no_grad()
def collect_preds(model, dataset, batch_size):
    model.eval()
    preds, targets = [], []
    for batch in dataset.iter_batches(batch_size, shuffle=False):
        out = _fwd(model, batch)
        preds.append(out["pred_response_db"].cpu().numpy())
        targets.append(batch["dual_target"].cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


# ──────────────────────────────────────────────────────────
# 메트릭
# ──────────────────────────────────────────────────────────

def lsd_np(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=-1))

def bootstrap_ci(arr, n_boot=2000, seed=42):
    rng   = np.random.default_rng(seed)
    means = np.array([rng.choice(arr, len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",       default="AC2_GRU", choices=ARCH_CHOICES,
                        help="재학습할 AC 아키텍처 (기본: AC2_GRU)")
    parser.add_argument("--data_dir",   default="./data/dataset_v3")
    parser.add_argument("--ckpt_dir",   default="./checkpoints/full")
    parser.add_argument("--stat_dir",   default="./paper_outputs/stats")
    parser.add_argument("--out_dir",    default="./paper_outputs")
    parser.add_argument("--epochs",     type=int,   default=300)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--patience",   type=int,   default=15)
    parser.add_argument("--skip_train", action="store_true",
                        help="체크포인트 있으면 학습 건너뜀 (평가만)")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gain_max",   type=float, default=6.0,
                        help="REVISION: per-section gain bound (기본 6.0, 완화 12.0)")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="REVISION: 원본엔 seed 고정 없음 → single-seed 재현성 위해 추가")
    parser.add_argument("--tag",        default="",
                        help="REVISION: ckpt 파일명 접미사 (예: _g12) — ±6 과 분리")
    args = parser.parse_args()

    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 모델명 결정
    arch_to_model = {
        "AC1_BiLSTM":   "AC1_BiLSTM_Biquad",
        "AC2_GRU":      "AC2_GRU_Biquad",
        "AC3_Conformer":"AC3_Conformer_Biquad",
    }
    model_name = arch_to_model[args.arch]

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    stat_dir = Path(args.stat_dir)
    out_dir  = Path(args.out_dir)
    tab_dir  = out_dir / "tables"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stat_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{model_name}{args.tag}.pt"
    device    = args.device

    # 모델 생성 (REVISION: gain_max 인자화)
    model     = BIQUAD_REGISTRY[model_name](gain_max=args.gain_max).to(device)
    criterion = build_loss().to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"AC Fitting Option C  --  {model_name}")
    print(f"Params: {n_params:,}  |  Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    # ── 학습 ────────────────────────────────────────────────
    if not (args.skip_train and ckpt_path.exists()):
        print(f"\nLoading datasets from {data_dir} ...")
        ds_train = PEQDataset(str(data_dir / "train"), device=device)
        ds_val   = PEQDataset(str(data_dir / "val"),   device=device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

        best_val_lsd = float("inf")
        patience_cnt = 0
        best_state   = None

        print(f"\nTraining up to {args.epochs} epochs (patience={args.patience})\n")
        t_start = time.perf_counter()

        for epoch in range(1, args.epochs + 1):
            tr = train_epoch(model, criterion, ds_train, optimizer,
                             args.batch_size, grad_clip=1.0)
            vl = validate(model, criterion, ds_val, args.batch_size)
            scheduler.step()

            print(f"Ep {epoch:3d}/{args.epochs}  "
                  f"loss={tr['train_loss']:.4f} lsd={tr['train_lsd']:.4f}  |  "
                  f"val_loss={vl['val_loss']:.4f} val_lsd={vl['val_lsd']:.4f}")

            if vl["val_lsd"] < best_val_lsd - 1e-5:
                best_val_lsd = vl["val_lsd"]
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch}  (best val_lsd={best_val_lsd:.4f})")
                    break

        elapsed = time.perf_counter() - t_start
        print(f"\nDone in {elapsed/60:.1f} min.  Best val_lsd={best_val_lsd:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")
    else:
        print(f"\nLoading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ── 테스트셋 평가 ────────────────────────────────────────
    print(f"\nEvaluating on test set ...")
    ds_test = PEQDataset(str(data_dir / "test_synth"), device=device)

    pred_biquad, dual_tgt = collect_preds(model, ds_test, args.batch_size)
    lsd_biquad = lsd_np(pred_biquad, dual_tgt)

    np.save(stat_dir / f"{model_name}_pred.npy", pred_biquad)
    np.save(stat_dir / f"{model_name}_lsd.npy",  lsd_biquad)

    # ── 비교 데이터 로드 ─────────────────────────────────────
    # AC raw 대응: arch_to_model 의 원본 모델
    raw_model_name  = args.arch.replace("BiLSTM", "BiLSTM").replace("Conformer", "Conformer")
    # e.g. AC2_GRU → AC2_GRU, AC1_BiLSTM → AC1_BiLSTM
    ac_raw_map = {"AC1_BiLSTM": "AC1_BiLSTM", "AC2_GRU": "AC2_GRU", "AC3_Conformer": "AC3_Conformer"}
    raw_name    = ac_raw_map[args.arch]

    ac_raw_lsd_path = stat_dir / f"{raw_name}_lsd.npy"
    a0_lsd_path     = stat_dir / "A0_Proposed_lsd.npy"

    N = len(lsd_biquad)
    lsd_raw = np.load(ac_raw_lsd_path).astype(np.float64)[:N] if ac_raw_lsd_path.exists() else None
    a0_lsd  = np.load(a0_lsd_path).astype(np.float64)[:N]     if a0_lsd_path.exists()     else None

    # ── 결과 출력 ────────────────────────────────────────────
    print()
    print(f"{'Metric':45s} | {'Mean':>8} | {'95% CI':>20}")
    print("-" * 79)

    if lsd_raw is not None:
        lo, hi = bootstrap_ci(lsd_raw)
        print(f"  {raw_name} dense (raw)           | {lsd_raw.mean():>8.4f} | [{lo:.4f}, {hi:.4f}]")

    lo, hi = bootstrap_ci(lsd_biquad)
    print(f"  {model_name} (retrained)   | {lsd_biquad.mean():>8.4f} | [{lo:.4f}, {hi:.4f}]")

    if lsd_raw is not None:
        gap = lsd_biquad - lsd_raw
        lo, hi = bootstrap_ci(gap)
        print(f"  Representation penalty (biq-raw)  | {gap.mean():>8.4f} | [{lo:.4f}, {hi:.4f}]")

    if a0_lsd is not None:
        lo, hi = bootstrap_ci(a0_lsd)
        print(f"  A0_Proposed (reference)           | {a0_lsd.mean():>8.4f} | [{lo:.4f}, {hi:.4f}]")

        margin = lsd_biquad - a0_lsd
        lo, hi = bootstrap_ci(margin)
        print(f"  {model_name} advantage over A0   | {margin.mean():>8.4f} | [{lo:.4f}, {hi:.4f}]")

    # ── JND 분석 ─────────────────────────────────────────────
    if a0_lsd is not None:
        diff = np.abs(lsd_biquad - a0_lsd)
        below_pct = (diff < JND_THRESHOLD).mean() * 100
        lo, hi = bootstrap_ci(diff)
        print(f"\nJND Analysis ({model_name} vs A0_Proposed):")
        print(f"  mean|dLSD|  = {diff.mean():.3f} dB")
        print(f"  95% CI      = [{lo:.4f}, {hi:.4f}]")
        print(f"  Below JND ({JND_THRESHOLD} dB) = {below_pct:.1f}%  "
              f"(N={int((diff < JND_THRESHOLD).sum())}/{N})")

    # ── CSV 저장 ─────────────────────────────────────────────
    rows = []
    if lsd_raw is not None:
        rows.append({
            "Model/Condition": f"{raw_name} dense (raw)",
            "LSD mean": f"{lsd_raw.mean():.4f}",
            "95% CI":   f"[{bootstrap_ci(lsd_raw)[0]:.4f},{bootstrap_ci(lsd_raw)[1]:.4f}]",
        })
    rows.append({
        "Model/Condition": f"{model_name} (retrained)",
        "LSD mean": f"{lsd_biquad.mean():.4f}",
        "95% CI":   f"[{bootstrap_ci(lsd_biquad)[0]:.4f},{bootstrap_ci(lsd_biquad)[1]:.4f}]",
    })
    if lsd_raw is not None:
        gap = lsd_biquad - lsd_raw
        rows.append({
            "Model/Condition": "Representation penalty",
            "LSD mean": f"{gap.mean():.4f}",
            "95% CI":   f"[{bootstrap_ci(gap)[0]:.4f},{bootstrap_ci(gap)[1]:.4f}]",
        })
    if a0_lsd is not None:
        rows.append({
            "Model/Condition": "A0_Proposed (reference)",
            "LSD mean": f"{a0_lsd.mean():.4f}",
            "95% CI":   f"[{bootstrap_ci(a0_lsd)[0]:.4f},{bootstrap_ci(a0_lsd)[1]:.4f}]",
        })
        diff = np.abs(lsd_biquad - a0_lsd)
        rows.append({
            "Model/Condition": f"JND below {JND_THRESHOLD}dB (%)",
            "LSD mean": f"{(diff < JND_THRESHOLD).mean()*100:.1f}",
            "95% CI":   "",
        })

    df = pd.DataFrame(rows)
    csv_path = tab_dir / f"table_ac_fitting_C_{args.arch}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {model_name}_pred.npy, {model_name}_lsd.npy")


if __name__ == "__main__":
    main()
