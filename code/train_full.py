"""
train_full.py — PHONIX Adaptive EQ: Full Training Suite v4
============================================================
학습 대상 (12개):

  External baselines (E):
    E1_NoEQ          — 학습 없음 (하한)
    E2_StaticEQ      — 학습 없음 (rule-based)
    E3_Nercessian    — MLP, room target만 학습
    E4_Pepe          — CNN, room target만 학습
    E5_Sequential    — E3→E2 2단계, room correction part만 학습
    E6_DSP           — 학습 없음 (analytical)

  Proposed (A):
    A0_Proposed      — 제안 모델 (w/o pref loss)

  Ablations (A):
    A1_NoRoomInput   — room_response 입력 제거
    A2_withPrefLoss  — 기존 full dual-objective loss (negative ablation)

  Architecture variants (AC):
    AC1_BiLSTM       — TCN×4 + BiLSTM aggregation
    AC2_GRU          — TCN×4 + GRU aggregation (causal)
    AC3_Conformer    — TCN×4 + Conformer×2

공통 학습 설정:
  optimizer : AdamW (lr=3e-4, weight_decay=1e-4)
  scheduler : CosineAnnealingLR (T_max=epochs, eta_min=1e-6)
  epochs    : 300
  patience  : 15 (early stopping, val_lsd 기준)
  batch     : 512

사용법:
  # 전체 학습
  python train_full.py --data_dir ./data/dataset_v3

  # 단일 모델
  python train_full.py --data_dir ./data/dataset_v3 --only A0_Proposed

  # 특정 모델 제외
  python train_full.py --data_dir ./data/dataset_v3 --skip E1_NoEQ E2_StaticEQ E6_DSP

  # 이어서 학습
  python train_full.py --data_dir ./data/dataset_v3 --resume
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DualObjectiveAdaptivePEQ, DualObjectiveEQLoss
from ablation import A1_NoRoomInput, A3_NoPrefInput
from baselines import (
    E1_NoProcessing, E2_StaticModeEQ,
    E3_NercessianMLP, E4_PepeCNN,
    E5_Sequential, E6_DSP_Analytical,
)
from arch_variants import AC1_TCNBiLSTM, AC2_TCNGRU, AC3_TCNConformer
from dataset_generator_v4_tracklevel import PEQDataset
from model_aliases import canonical_model_name


# ══════════════════════════════════════════════════════════════════════════════
# 학습 설정
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_CFG = dict(
    epochs=300,
    batch_size=512,
    lr=3e-4,
    weight_decay=1e-4,
    patience=15,          # early stopping: val_lsd 기준
    grad_clip=1.0,
    scheduler_eta_min=1e-6,
)

# room_response를 사용하지 않는 모델 집합
NO_ROOM_MODELS = {"E1_NoEQ", "E2_StaticEQ", "E3_Nercessian", "E4_Pepe", "E5_Sequential"}


# ══════════════════════════════════════════════════════════════════════════════
# 모델 레지스트리
# ══════════════════════════════════════════════════════════════════════════════

def build_registry() -> dict:
    """
    각 엔트리:
      model   : nn.Module
      loss    : DualObjectiveEQLoss | None (None → 학습 없음)
      target  : "dual" | "room"  (학습 타겟 선택)
      group   : "External" | "Proposed" | "Ablation" | "Architecture"
      note    : 설명
    """
    # ── A2 손실 (negative ablation: 기존 full dual-objective) ───────────────
    a2_with_pref_loss = DualObjectiveEQLoss(
        lambda_final=1.0,
        lambda_room=0.35,
        lambda_pref_res=0.25,
        lambda_shape=0.20,
        lambda_grad=0.20,
        lambda_curv=0.08,
        lambda_dir=0.15,
        lambda_mean=0.03,
        use_perceptual=True,
        mag_weight_alpha=0.10,
        grad_weight_beta=0.30,
    )

    # ── E3/E4/E5 손실 (룸 보정만, 지각 가중치 없음) ──────────────────────────
    room_only_loss = DualObjectiveEQLoss(
        lambda_final=1.0,
        lambda_room=0.0,
        lambda_pref_res=0.0,
        lambda_dir=0.0,
        use_perceptual=False,
    )

    # ── A0: 선호도 손실 제거 (λ_pref_res=0, λ_dir=0) ────────────────────────
    # 기존 A2_NoPrefLoss를 제안 모델로 승격
    a0_proposed_loss = DualObjectiveEQLoss(
        lambda_final=1.0,
        lambda_room=0.35,
        lambda_pref_res=0.0,   # ← 핵심: 선호도 잔차 감독 제거
        lambda_shape=0.20,
        lambda_grad=0.20,
        lambda_curv=0.08,
        lambda_dir=0.0,        # ← 핵심: 방향 손실 제거
        lambda_mean=0.03,
        use_perceptual=True,
        mag_weight_alpha=0.10,
        grad_weight_beta=0.30,
    )

    return {
        # ── External baselines ─────────────────────────────────────────────
        "E1_NoEQ": {
            "model":  E1_NoProcessing(n_freqs=128),
            "loss":   None,
            "target": "dual",
            "group":  "External",
            "note":   "No processing (lower bound)",
        },
        "E2_StaticEQ": {
            "model":  E2_StaticModeEQ(n_freqs=128),
            "loss":   None,
            "target": "dual",
            "group":  "External",
            "note":   "Rule-based static mode EQ",
        },
        "E3_Nercessian": {
            "model":  E3_NercessianMLP(in_dim=10, n_freqs=128, sample_rate=48000),
            "loss":   room_only_loss,
            "target": "room",   # room_target으로만 학습
            "group":  "External",
            "note":   "Nercessian 2020 MLP (room correction only)",
        },
        "E4_Pepe": {
            "model":  E4_PepeCNN(in_dim=10, n_freqs=128, sample_rate=48000),
            "loss":   room_only_loss,
            "target": "room",
            "group":  "External",
            "note":   "Pepe 2020 CNN (room correction only)",
        },
        "E5_Sequential": {
            "model":  E5_Sequential(in_dim=10, n_freqs=128, sample_rate=48000),
            "loss":   room_only_loss,
            "target": "room",   # room corrector(E3 part)만 학습, E2는 고정
            "group":  "External",
            "note":   "Sequential: E3(trained)→E2(fixed)",
        },
        "E6_DSP": {
            "model":  E6_DSP_Analytical(n_freqs=128),
            "loss":   None,
            "target": "dual",
            "group":  "External",
            "note":   "DSP analytical: -room + pref profile (no AI)",
        },

        # ── Proposed ───────────────────────────────────────────────────────
        "A0_Proposed": {
            # per-section output gain bound (dB); centre-frequency upper bound unchanged.
            "model":  DualObjectiveAdaptivePEQ(gain_max=12.0, fc_max=16000.0),
            "loss":   a0_proposed_loss,
            "target": "dual",
            "group":  "Proposed",
            "note":   "Proposed model: w/o preference loss",
        },

        # ── Ablations ──────────────────────────────────────────────────────
        # "A1_NoRoomInput": {
        #     "model":  A1_NoRoomInput(),
        #     "loss":   a0_loss,   # 손실은 동일, 입력만 다름
        #     "target": "dual",
        #     "group":  "Ablation",
        #     "note":   "w/o room_response input (RoomResponseEncoder removed)",
        # },
        "A1_NoRoomInput": {
            "model": A1_NoRoomInput(),
            "loss":  a0_proposed_loss,
            "target": "dual",
            "group": "Ablation",
            "note": "w/o room_response input + clean features",
        },
        "A2_withPrefLoss": {
            "model":  DualObjectiveAdaptivePEQ(gain_max=12.0, fc_max=16000.0),   # same architecture as A0
            "loss":   a2_with_pref_loss,
            "target": "dual",
            "group":  "Ablation",
            "note":   "with preference loss (negative ablation; former A0 full)",
        },
        "A3_NoPrefInput": {
            "model":  A3_NoPrefInput(),
            "loss":   a0_proposed_loss,
            "target": "dual",
            "group":  "Ablation",
            "note":   "w/o preference inputs (mode_id, band_gains removed)",
        },
        # ── Architecture variants ───────────────────────────────────────────
        "AC1_BiLSTM": {
            "model":  AC1_TCNBiLSTM(),
            "loss":   a0_proposed_loss,
            "target": "dual",
            "group":  "Architecture",
            "note":   "TCN×4 + BiLSTM aggregation",
        },
        "AC2_GRU": {
            "model":  AC2_TCNGRU(),
            "loss":   a0_proposed_loss,
            "target": "dual",
            "group":  "Architecture",
            "note":   "TCN×4 + GRU aggregation (causal)",
        },
        "AC3_Conformer": {
            "model":  AC3_TCNConformer(),
            "loss":   a0_proposed_loss,
            "target": "dual",
            "group":  "Architecture",
            "note":   "TCN×4 + Conformer×2 aggregation",
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Forward 디스패처
# ══════════════════════════════════════════════════════════════════════════════

def model_forward(name: str, model: nn.Module, batch: dict) -> dict:
    """
    모델별 입력 시그니처 차이를 여기서 흡수.
    항상 dict를 반환하고 pred_response_db key를 포함해야 함.
    """
    f   = batch["features"]       # (B, T, in_dim)
    f_c = batch.get("features_clean") # (B, T, in_dim), A1_NoRoomInput에서 우선 사용
    rr  = batch["room_response"]  # (B, n_room_bins)
    mi  = batch["mode_id"]        # (B,)
    bg  = batch["band_gains"]     # (B, n_bands)

    if name == "E1_NoEQ":
        return model(f)

    elif name == "E2_StaticEQ":
        return model(f, mode_id=mi)

    elif name in ("E3_Nercessian", "E4_Pepe"):
        # room_response는 받지 않음 (original 인터페이스)
        return model(f)

    elif name == "E5_Sequential":
        # E5는 pred_response_db = room_corr + pref_profile을 반환
        return model(f, mode_id=mi)

    elif name == "E6_DSP":
        return model(f, room_response=rr, mode_id=mi)
    elif name == "A1_NoRoomInput":
        return model(f_c if f_c is not None else f, rr, mi, bg)
    else:
        # A0, A1, A2, AC1~AC3 — 모두 동일한 풀 시그니처
        return model(f, rr, mi, bg)


def get_pred(name: str, out: dict, target: str) -> torch.Tensor:
    """
    학습/검증에 사용할 예측 텐서 선택.
    E5의 room corrector는 pred_response_db의 room correction 부분만 사용.
    """
    return out["pred_response_db"]


# ══════════════════════════════════════════════════════════════════════════════
# 메트릭 계산
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pred: torch.Tensor, batch: dict) -> dict:
    """per-batch 메트릭 (스칼라)"""
    dual_t = batch["dual_target"]
    room_t = batch["room_target"]
    pref_t = batch["pref_target"]

    def lsd(a, b):
        return torch.sqrt(((a - b) ** 2).mean(dim=-1)).mean().item()

    def pascore(a, b):
        return (a.sign() == b.sign()).float().mean().item()

    def cossim(a, b):
        return F.cosine_similarity(a, b, dim=-1).mean().item()

    # return {
    #     "lsd":      lsd(pred, dual_t),
    #     "lsd_room": lsd(pred, room_t),
    #     "lsd_pref": lsd(pred, pref_t),
    #     "pascore":  pascore(pred, pref_t),
    #     "cossim":   cossim(pred, pref_t),
    # }

    heard = pred - room_t

    return {
        "lsd":      lsd(pred, dual_t),       # 그대로 유지
        "lsd_room": lsd(pred, room_t),       # 그대로 유지
        "lsd_pref": lsd(heard, pref_t),      # 수정
        "pascore":  pascore(heard, pref_t),  # 수정
        "cossim":   cossim(heard, pref_t),   # 수정
    }


# ══════════════════════════════════════════════════════════════════════════════
# 학습 / 검증 루프
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    name: str, model: nn.Module, criterion: DualObjectiveEQLoss,
    dataset: PEQDataset, optimizer: torch.optim.Optimizer,
    batch_size: int, grad_clip: float,
) -> dict:
    model.train()
    total_loss = total_lsd = total_pa = n = 0
    print("Training...")
    for batch in dataset.iter_batches(batch_size, shuffle=True):
        out  = model_forward(name, model, batch)
        pred = get_pred(name, out, "")

        # 학습 타겟 선택
        entry_target = _REGISTRY_TARGET.get(name, "dual")
        train_target = batch["room_target"] if entry_target == "room" else batch["dual_target"]
        pref_for_loss = None if entry_target == "room" else batch["pref_target"]

        loss_dict = criterion(
            pred_response_db  = pred,
            dual_target_db    = train_target,
            pref_target_db    = pref_for_loss,
            room_target_db    = batch.get("room_target"),
            room_correction_db= out.get("room_correction_db"),
            peq_response_db = out.get("peq_response_db"),
            pref_curve_db   = out.get("pref_curve_db"),
        )

        optimizer.zero_grad(set_to_none=True)
        loss_dict["loss"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = len(batch["dual_target"])
        total_loss += loss_dict["loss"].item() * bs
        total_lsd  += compute_metrics(pred, batch)["lsd"] * bs
        total_pa   += compute_metrics(pred, batch)["pascore"] * bs
        n += bs

    return {
        "train_loss": total_loss / max(n, 1),
        "train_lsd":  total_lsd  / max(n, 1),
        "train_pa":   total_pa   / max(n, 1),
    }


@torch.no_grad()
def validate(
    name: str, model: nn.Module, criterion,
    dataset: PEQDataset, batch_size: int,
) -> dict:
    model.eval()
    agg = {k: 0.0 for k in ("loss", "lsd", "lsd_room", "lsd_pref", "pascore", "cossim")}
    n = 0

    for batch in dataset.iter_batches(batch_size, shuffle=False):
        out  = model_forward(name, model, batch)
        pred = get_pred(name, out, "")
        bs   = len(batch["dual_target"])

        if criterion is not None:
            entry_target  = _REGISTRY_TARGET.get(name, "dual")
            train_target  = batch["room_target"] if entry_target == "room" else batch["dual_target"]
            pref_for_loss = None if entry_target == "room" else batch["pref_target"]
            ld = criterion(
                pred_response_db  = pred,
                dual_target_db    = train_target,
                pref_target_db    = pref_for_loss,
                room_target_db    = batch.get("room_target"),
                room_correction_db= out.get("room_correction_db"),
                peq_response_db = out.get("peq_response_db"),
                pref_curve_db   = out.get("pref_curve_db"),
            )
            agg["loss"] += ld["loss"].item() * bs

        m = compute_metrics(pred, batch)
        for k in ("lsd", "lsd_room", "lsd_pref", "pascore", "cossim"):
            agg[k] += m[k] * bs
        n += bs

    return {f"val_{k}": v / max(n, 1) for k, v in agg.items()}


@torch.no_grad()
def evaluate_no_train(
    name: str, model: nn.Module, dataset: PEQDataset, batch_size: int,
) -> dict:
    """학습 없는 모델(E1, E2, E6) 검증 전용"""
    model.eval()
    agg = {k: 0.0 for k in ("lsd", "lsd_room", "lsd_pref", "pascore", "cossim")}
    n = 0
    for batch in dataset.iter_batches(batch_size, shuffle=False):
        out  = model_forward(name, model, batch)
        pred = get_pred(name, out, "")
        bs   = len(batch["dual_target"])
        m    = compute_metrics(pred, batch)
        for k in agg: agg[k] += m[k] * bs
        n += bs
    return {"val_loss": None, **{f"val_{k}": v / max(n, 1) for k, v in agg.items()}}


# ══════════════════════════════════════════════════════════════════════════════
# RTF 측정
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_rtf(name: str, model: nn.Module, n_warmup=10, n_trial=100) -> dict:
    """CPU 단일 스레드 기준 RTF 측정"""
    m = copy.deepcopy(model).cpu().eval()
    dummy = {
        "features":      torch.randn(1, 32, 10),
        "features_clean":torch.randn(1, 32, 10),
        "room_response": torch.randn(1, 128),
        "mode_id":       torch.zeros(1, dtype=torch.long),
        "band_gains":    torch.zeros(1, 10),
    }
    for _ in range(n_warmup):
        model_forward(name, m, dummy)
    t0 = time.perf_counter()
    for _ in range(n_trial):
        model_forward(name, m, dummy)
    elapsed_ms  = (time.perf_counter() - t0) / n_trial * 1000
    audio_ms    = 32 * 480 / 48000 * 1000  # 32 frames × hop / sr
    rtf         = elapsed_ms / audio_ms
    params      = sum(p.numel() for p in model.parameters())
    return {
        "latency_ms": round(elapsed_ms, 3),
        "rtf":        round(rtf, 4),
        "params":     params,
        "realtime":   rtf < 1.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 메인 학습 루프 (단일 모델)
# ══════════════════════════════════════════════════════════════════════════════

def train_one(
    name: str, model: nn.Module, criterion: DualObjectiveEQLoss,
    train_ds: PEQDataset, val_ds: PEQDataset,
    device: torch.device, cfg: dict, save_dir: Path,
    resume: bool = False,
) -> dict:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["scheduler_eta_min"]
    )

    ckpt_path = save_dir / f"{name}.pt"
    history   = []
    best_val  = float("inf")
    best_metrics = {}
    start_epoch  = 0
    no_improve   = 0

    # ── 이어서 학습 ──────────────────────────────────────────────────────────
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val     = ckpt["best_val"]
        start_epoch  = ckpt["epoch"] + 1
        no_improve   = ckpt.get("no_improve", 0)
        history      = ckpt.get("history", [])
        best_metrics = ckpt.get("best_metrics", {})
        print(f"  [Resume] epoch {start_epoch} | best_val_lsd={best_val:.4f}")

    t0 = time.time()
    for epoch in range(start_epoch, cfg["epochs"]):

        train_m = train_epoch(
            name, model, criterion, train_ds,
            optimizer, cfg["batch_size"], cfg["grad_clip"],
        )
        val_m = validate(name, model, criterion, val_ds, cfg["batch_size"])
        scheduler.step()

        # Early stopping 기준: val_lsd (낮을수록 좋음)
        monitor = val_m["val_lsd"]
        improved = monitor < best_val
        if improved:
            best_val     = monitor
            best_metrics = {**train_m, **val_m}
            no_improve   = 0
            torch.save({
                "epoch":        epoch,
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "best_val":     best_val,
                "no_improve":   no_improve,
                "best_metrics": best_metrics,
                "history":      history,
                "cfg":          cfg,
            }, ckpt_path)
        else:
            no_improve += 1

        history.append({"epoch": epoch, **train_m, **val_m})

        # 진행 출력
        if (epoch + 1) % 10 == 0 or improved or epoch == 0:
            mark  = " ★" if improved else ""
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Ep {epoch+1:4d}/{cfg['epochs']} | "
                f"loss {train_m['train_loss']:.4f} | "
                f"val_loss {val_m['val_loss']:.4f} | "
                f"LSD {val_m['val_lsd']:.4f} | "
                f"PA {val_m['val_pascore']:.3f} | "
                f"CosSim {val_m['val_cossim']:.3f} | "
                f"lr {lr_now:.2e} | "
                f"{time.time()-t0:.0f}s{mark}"
            )

        # Early stopping
        if no_improve >= cfg["patience"]:
            print(f"  [Early stop] epoch {epoch+1} (no improve for {cfg['patience']} epochs)")
            break

    # history 저장
    with open(save_dir / f"{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return best_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 결과 출력 테이블
# ══════════════════════════════════════════════════════════════════════════════

def print_table(results: dict):
    groups = ["External", "Proposed", "Ablation", "Architecture"]
    header = (
        f"  {'Model':<20} {'LSD↓':>7} {'LSD_room↓':>10} "
        f"{'PA↑':>7} {'CosSim↑':>9} {'RTF↓':>7} {'Params':>10} {'RT?':>5}"
    )
    sep = "  " + "─" * 80

    for group in groups:
        items = {k: v for k, v in results.items() if v.get("group") == group}
        if not items:
            continue
        print(f"\n{'═'*84}")
        print(f"  [{group}]")
        print(header)
        print(sep)
        for name, r in items.items():
            mark  = ""
            lsd   = f"{r['val_lsd']:.4f}"      if r.get("val_lsd")      else "—"
            lsr   = f"{r['val_lsd_room']:.4f}"  if r.get("val_lsd_room") else "—"
            pa    = f"{r['val_pascore']:.3f}"   if r.get("val_pascore")  else "—"
            cs    = f"{r['val_cossim']:.3f}"    if r.get("val_cossim")   else "—"
            rtf   = f"{r['rtf']:.3f}"           if r.get("rtf")          else "—"
            par   = f"{r['params']:,}"          if r.get("params")       else "—"
            rt    = "✓" if r.get("realtime") else "✗"
            print(f"  {name+mark:<20} {lsd:>7} {lsr:>10} {pa:>7} {cs:>9} {rtf:>7} {par:>10} {rt:>5}")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

# registry target 캐시 (train_epoch/validate에서 접근)
_REGISTRY_TARGET: dict = {}


def main():
    parser = argparse.ArgumentParser(description="PHONIX Adaptive EQ — Full Training Suite")
    parser.add_argument("--data_dir",    default="./data/dataset_v3")
    parser.add_argument("--save_dir",    default="./checkpoints/full")
    parser.add_argument("--epochs",      type=int,   default=TRAIN_CFG["epochs"])
    parser.add_argument("--batch_size",  type=int,   default=TRAIN_CFG["batch_size"])
    parser.add_argument("--lr",          type=float, default=TRAIN_CFG["lr"])
    parser.add_argument("--patience",    type=int,   default=TRAIN_CFG["patience"])
    parser.add_argument("--only",        type=str,   default=None,
                        help="단일 모델만 학습 (예: A0_Proposed)")
    parser.add_argument("--skip",        nargs="*",  default=[],
                        help="제외할 모델 목록")
    parser.add_argument("--resume",      action="store_true",
                        help="체크포인트에서 이어서 학습")
    parser.add_argument("--no_cuda",     action="store_true")
    args = parser.parse_args()

    # ── 디바이스 ──────────────────────────────────────────────────────────────
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    cfg = {
        **TRAIN_CFG,
        "epochs":     args.epochs,
        "batch_size": args.batch_size,
        "lr":         args.lr,
        "patience":   args.patience,
    }

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터셋 ──────────────────────────────────────────────────────────────
    print("\n데이터셋 로드 중...")
    train_ds = PEQDataset(f"{args.data_dir}/train", device=str(device))
    val_ds   = PEQDataset(f"{args.data_dir}/val",   device=str(device))
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    # ── 레지스트리 ────────────────────────────────────────────────────────────
    registry = build_registry()
    global _REGISTRY_TARGET
    _REGISTRY_TARGET = {k: v["target"] for k, v in registry.items()}

    if args.only:
        args.only = canonical_model_name(args.only)
        if args.only not in registry:
            print(f"[ERROR] '{args.only}' 없음. 사용 가능: {list(registry.keys())}")
            return
        registry = {args.only: registry[args.only]}

    skip_set = {canonical_model_name(name) for name in args.skip}

    # ── 기존 결과 로드 ────────────────────────────────────────────────────────
    results_path = save_dir / "results.json"
    results: dict = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    # ── 학습 루프 ──────────────────────────────────────────────────────────────
    total = len([k for k in registry if k not in skip_set])
    done  = 0

    for name, entry in registry.items():
        if name in skip_set:
            continue

        # resume 시 이미 완료된 모델 스킵
        if args.resume and not args.only and name in results:
            print(f"\n[SKIP] {name} (already done)")
            done += 1
            continue

        done += 1
        model, criterion = entry["model"], entry["loss"]
        print(f"\n{'═'*60}")
        print(f"[{done}/{total}]  {name}  |  {entry['group']}")
        print(f"  {entry['note']}")

        # RTF & 파라미터 수 측정
        rtf_info = measure_rtf(name, model)
        print(f"  Params : {rtf_info['params']:,}")
        print(f"  RTF    : {rtf_info['rtf']}  {'(✓ real-time)' if rtf_info['realtime'] else '(✗ NOT real-time)'}")

        # 학습 또는 평가
        if criterion is None:
            print("  [No training] evaluating fixed model...")
            metrics = evaluate_no_train(name, model.to(device), val_ds, cfg["batch_size"])
        else:
            metrics = train_one(
                name, model, criterion,
                train_ds, val_ds, device, cfg, save_dir, args.resume,
            )
            print(
                f"  Best → LSD={metrics.get('val_lsd',0):.4f}  "
                f"PA={metrics.get('val_pascore',0):.3f}  "
                f"CosSim={metrics.get('val_cossim',0):.3f}"
            )

        # 결과 저장
        results[name] = {
            "group": entry["group"],
            "note":  entry["note"],
            **rtf_info,
            **{k: v for k, v in metrics.items() if v is not None},
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    # ── 최종 출력 ──────────────────────────────────────────────────────────────
    print(f"\n\n{'═'*84}")
    print("  최종 결과 (val set)")
    print_table(results)

    final_path = save_dir / "results_final.json"
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  결과 저장: {final_path}")

    # A0 vs 베이스라인 핵심 요약
    if "A0_Proposed" in results:
        a0 = results["A0_Proposed"]
        print(f"\n  ── A0 vs 주요 모델 ──")
        for comp in ["E3_Nercessian", "E4_Pepe", "A1_NoRoomInput", "A2_withPrefLoss"]:
            if comp in results:
                r = results[comp]
                dlsd = a0.get("val_lsd", 0) - r.get("val_lsd", 0)
                dpa  = a0.get("val_pascore", 0) - r.get("val_pascore", 0)
                print(f"  A0 vs {comp:<20}  ΔLSD={dlsd:+.4f}  ΔPA={dpa:+.4f}")


if __name__ == "__main__":
    main()
