"""
run_gain_freq_ablation.py — gain-bound and centre-frequency ablation
====================================================================
Stage B bounds each of the K peaking sections to a per-section gain range and
to a centre-frequency range. Under the narrower +/-6 dB bound, 57.8 % of the
learned gains sat within 0.2 dB of the limit and 17.9 % of the centre
frequencies sat near the 16 kHz upper edge: the bounds were binding rather
than merely permissive, which is what motivates measuring the relaxed setting.

This driver trains and evaluates the 2x2 matrix
(gain in {6, 12} dB  x  fc_max in {16k, 20k} Hz) over several seeds, using a
model whose gain_max / fc_max are constructor arguments. Loss, optimiser,
scheduler, data and seed handling are identical across the cells, so the
bounds are the only thing that varies.

The dataset and checkpoint directories are only read; every checkpoint and
result this driver produces is written under --save_dir / --out_dir.

A checkpoint does not record the bound it was trained under. Loading a
+/-12 dB checkpoint into a model instantiated with gain_max=6.0 succeeds
without error and then silently clamps the output, which corrupts every
metric derived from it. All evaluation here therefore goes through this
driver's own instantiation, which sets gain_max / fc_max explicitly.

Usage:
  # relaxed-gain cells, three seeds
  python run_gain_freq_ablation.py --configs g12_f16k g12_f20k --seeds 42 123 7

  # +/-6 / 16k baseline metrics from the existing checkpoint (no training)
  python run_gain_freq_ablation.py --configs g6_f16k --eval_only

  # full 2x2
  python run_gain_freq_ablation.py --configs all --seeds 42 123 7

  # instantiate and run one batch without training, to check the bounds
  python run_gain_freq_ablation.py --dry_check
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import train_full as TF
from model import DualObjectiveAdaptivePEQ, DualObjectiveEQLoss
from dataset_generator_v4_tracklevel import PEQDataset


# ── 경로(절대) ────────────────────────────────────────────────────────────────
HERE          = Path(__file__).resolve().parent
ORIG_ROOT     = HERE.parent
DEFAULT_DATA  = ORIG_ROOT / "data" / "dataset_v3"            # read only
BASELINE_CKPT = ORIG_ROOT / "checkpoints" / "full" / "A0_Proposed.pt"  # ±6/16k 비교군 checkpoint


# ── 실험 매트릭스 (결정 4) ───────────────────────────────────────────────────
CONFIGS = {
    "g6_f16k":  dict(gain_max=6.0,  fc_max=16000.0),   # baseline (재학습 불필요)
    "g12_f16k": dict(gain_max=12.0, fc_max=16000.0),   # gain 만 완화  ← 우선순위 1
    "g6_f20k":  dict(gain_max=6.0,  fc_max=20000.0),   # freq 만 완화
    "g12_f20k": dict(gain_max=12.0, fc_max=20000.0),   # 둘 다 완화    ← 우선순위 1
}
PRIORITY_CONFIGS = ["g12_f16k", "g12_f20k"]
SEEDS_DEFAULT    = [42, 123, 7]
FIXED_GAIN_REF   = 6.0      # fixed ±6 dB reference for the saturation criterion
FIXED_FC_REF     = 16000.0  # fixed 16 kHz reference


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def a0_proposed_loss() -> DualObjectiveEQLoss:
    """train_full.build_registry 의 a0_proposed_loss 와 동일 구성."""
    return DualObjectiveEQLoss(
        lambda_final=1.0,
        lambda_room=0.35,
        lambda_pref_res=0.0,   # proposed: 선호 잔차 감독 제거
        lambda_shape=0.20,
        lambda_grad=0.20,
        lambda_curv=0.08,
        lambda_dir=0.0,        # proposed: 방향 손실 제거
        lambda_mean=0.03,
        use_perceptual=True,
        mag_weight_alpha=0.10,
        grad_weight_beta=0.30,
    )


def build_model(cfg: str) -> DualObjectiveAdaptivePEQ:
    """제약(gain_max/fc_max)만 다른 동일 구조 모델. fc_min=80 은 유지."""
    return DualObjectiveAdaptivePEQ(**CONFIGS[cfg])


def a2_with_pref_loss() -> DualObjectiveEQLoss:
    """train_full.build_registry 의 a2_with_pref_loss 와 동일 (negative ablation: full dual-objective)."""
    return DualObjectiveEQLoss(
        lambda_final=1.0,
        lambda_room=0.35,
        lambda_pref_res=0.25,   # A2: 선호 잔차 감독 켬
        lambda_shape=0.20,
        lambda_grad=0.20,
        lambda_curv=0.08,
        lambda_dir=0.15,        # A2: 방향 손실 켬
        lambda_mean=0.03,
        use_perceptual=True,
        mag_weight_alpha=0.10,
        grad_weight_beta=0.30,
    )


def variant_loss(variant: str) -> DualObjectiveEQLoss:
    return a2_with_pref_loss() if variant == "A2" else a0_proposed_loss()


def cname(cfg: str, seed: int, variant: str = "A0") -> str:
    """체크포인트 이름 규칙: {variant}_{cfg}_s{seed} (결정 3). 기본 variant=A0(하위호환)."""
    return f"{variant}_{cfg}_s{seed}"


def lsd_per_sample(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - target) ** 2, axis=-1))


# ══════════════════════════════════════════════════════════════════════════════
# 평가 (드라이버 내장 — ±12 도 올바른 bound 로 forward)
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(name, cfg, model, ds_test, device, batch_size=512) -> dict:
    """test set 에서 LSD/DMR/CosSim + per-section gain/fc 수집.

    gain/fc 는 saturation 두 기준 산출용. model.forward 가 out['gain'],
    out['fc'] 를 그대로 반환하므로 별도 재구현 불필요(model.py).
    """
    model.eval().to(device)
    preds, duals, prefs, rooms, gains, fcs = [], [], [], [], [], []
    for batch in ds_test.iter_batches(batch_size, shuffle=False):
        out = TF.model_forward(name, model, batch)
        preds.append(out["pred_response_db"].cpu().numpy())
        duals.append(batch["dual_target"].cpu().numpy())
        prefs.append(batch["pref_target"].cpu().numpy())
        rooms.append(batch["room_target"].cpu().numpy())
        if out.get("gain") is not None:   # A1/A3 는 "gain":None (dense) → 스킵
            gains.append(out["gain"].cpu().numpy())
            fcs.append(out["fc"].cpu().numpy())

    pred = np.concatenate(preds); dual = np.concatenate(duals)
    pref = np.concatenate(prefs); room = np.concatenate(rooms)

    lsd_arr = lsd_per_sample(pred, dual)
    heard = pred - room
    dmr_arr = np.mean((np.sign(heard) == np.sign(pref)).astype(float), axis=-1)
    num = np.sum(heard * pref, axis=-1)
    den = np.linalg.norm(heard, axis=-1) * np.linalg.norm(pref, axis=-1) + 1e-8
    cos_arr = num / den

    res = dict(
        lsd_arr=lsd_arr, dmr_arr=dmr_arr, cos_arr=cos_arr,
        lsd_mean=float(lsd_arr.mean()),
        dmr_mean=float(dmr_arr.mean()),
        cos_mean=float(cos_arr.mean()),
    )

    # ── saturation / boundary 두 기준 (cfg 가 CONFIGS 키일 때만) ─────────────
    if gains and cfg in CONFIGS:
        gain_all = np.abs(np.concatenate(gains))   # |gain| (N,K)
        fc_all   = np.concatenate(fcs)             # fc     (N,K)
        gmax = CONFIGS[cfg]["gain_max"]; fmax = CONFIGS[cfg]["fc_max"]
        res.update(
            # (a) 자기 boundary 기준: 이 설정에서도 막혔나
            gain_sat_self=float(np.mean(gain_all > (gmax - 0.2))),
            fc_hi_self=float(np.mean(fc_all > 0.875 * fmax)),
            # (b) ±6 / 16k 고정 기준: ±6/16k 였으면 막혔을 비율 (핵심 증거)
            gain_over6=float(np.mean(gain_all > FIXED_GAIN_REF)),
            fc_over16k=float(np.mean(fc_all > FIXED_FC_REF)),
            fc_lo=float(np.mean(fc_all < 100.0)),
        )
    return res


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (float("nan"), float("nan"), 0)
    return (float(np.mean(vals)), float(np.std(vals)), len(vals))


# ══════════════════════════════════════════════════════════════════════════════
# dry-run 검증: 학습/저장 없이 인스턴스화 + 1배치 forward + 제약값 확인
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def dry_check(data_dir: Path, device):
    print("=" * 64)
    print("DRY-CHECK: 인스턴스화 + 제약값 + 1배치 forward (학습/저장 없음)")
    print("=" * 64)
    ds = PEQDataset(f"{data_dir}/test_synth", device=str(device))
    batch = next(iter(ds.iter_batches(8, shuffle=False)))
    ok = True
    for cfg in CONFIGS:
        m = build_model(cfg).to(device).eval()
        name = cname(cfg, 42)
        TF._REGISTRY_TARGET[name] = "dual"
        out = TF.model_forward(name, m, batch)
        g = out["gain"]; fc = out["fc"]
        gmax = CONFIGS[cfg]["gain_max"]; fmax = CONFIGS[cfg]["fc_max"]
        attr_ok = (abs(m.gain_max - gmax) < 1e-9 and abs(m.fc_max - fmax) < 1e-9
                   and abs(m.fc_min - 80.0) < 1e-9)
        # forward 출력이 실제 bound 안에 들어오는지 (tanh*gmax, sigmoid 구간)
        rng_ok = (float(g.abs().max()) <= gmax + 1e-4
                  and float(fc.min()) >= 80.0 - 1e-3
                  and float(fc.max()) <= fmax + 1e-3)
        ok = ok and attr_ok and rng_ok
        print(f"  [{cfg:>9}] gain_max={m.gain_max:>5.1f} fc_min={m.fc_min:>5.1f} "
              f"fc_max={m.fc_max:>7.1f} | gain∈[{float(g.min()):+.2f},{float(g.max()):+.2f}] "
              f"fc∈[{float(fc.min()):.0f},{float(fc.max()):.0f}] "
              f"| attr={'OK' if attr_ok else 'FAIL'} range={'OK' if rng_ok else 'FAIL'}")
    print("-" * 64)
    print(f"DRY-CHECK 결과: {'ALL PASS' if ok else 'FAIL'}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="JAES revision: gain/freq 제약 완화 재실험")
    ap.add_argument("--data_dir",   default=str(DEFAULT_DATA),
                    help="원본 데이터셋(절대경로, read-only)")
    ap.add_argument("--save_dir",   default=str(HERE / "checkpoints"))
    ap.add_argument("--out_dir",    default=str(HERE / "results"))
    ap.add_argument("--configs",    nargs="*", default=PRIORITY_CONFIGS,
                    help="config 목록 또는 'all'. 예: g12_f16k g12_f20k")
    ap.add_argument("--seeds",      type=int, nargs="*", default=SEEDS_DEFAULT)
    ap.add_argument("--epochs",     type=int, default=TF.TRAIN_CFG["epochs"])
    ap.add_argument("--batch_size", type=int, default=TF.TRAIN_CFG["batch_size"])
    ap.add_argument("--lr",         type=float, default=TF.TRAIN_CFG["lr"])
    ap.add_argument("--patience",   type=int, default=TF.TRAIN_CFG["patience"])
    ap.add_argument("--test_split", default="test_synth", help="평가 split (test_synth/test_real)")
    ap.add_argument("--variant",    default="A0", choices=["A0", "A2"],
                    help="A0=proposed loss, A2=with-pref-loss(negative ablation). 둘 다 gain_max=12 동일 구조.")
    ap.add_argument("--eval_only",  action="store_true")
    ap.add_argument("--dry_check",  action="store_true", help="셋업 검증만 수행")
    ap.add_argument("--no_cuda",    action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    data_dir = Path(args.data_dir)

    if args.dry_check:
        ok = dry_check(data_dir, device)
        raise SystemExit(0 if ok else 1)

    configs = list(CONFIGS.keys()) if args.configs == ["all"] else args.configs
    for c in configs:
        if c not in CONFIGS:
            raise SystemExit(f"[ERROR] unknown config '{c}'. choices={list(CONFIGS)}")

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    out_dir  = Path(args.out_dir);  out_dir.mkdir(parents=True, exist_ok=True)
    V = args.variant
    print(f"Device: {device}  |  variant={V}  configs={configs}  seeds={args.seeds}")
    print(f"Data(read-only): {data_dir}")
    print(f"Save(new only):  {save_dir}")

    cfg = {**TF.TRAIN_CFG, "epochs": args.epochs, "batch_size": args.batch_size,
           "lr": args.lr, "patience": args.patience}

    # 모든 (config,seed) 이름을 dual 타겟으로 등록 (TF.model_forward 디스패치용)
    for c in configs:
        for s in args.seeds:
            TF._REGISTRY_TARGET[cname(c, s, V)] = "dual"

    # ── 데이터 ───────────────────────────────────────────────────────────────
    print("\n데이터 로드...")
    if not args.eval_only:
        train_ds = PEQDataset(f"{data_dir}/train", device=str(device))
        val_ds   = PEQDataset(f"{data_dir}/val",   device=str(device))
    ds_test = PEQDataset(f"{data_dir}/{args.test_split}", device=str(device))

    # ── 학습 루프 (config × seed) ────────────────────────────────────────────
    if not args.eval_only:
        for c in configs:
            if V == "A0" and c == "g6_f16k":
                print(f"\n[{c}] baseline — 원본 A0_Proposed.pt 재사용, 재학습 생략")
                continue
            for seed in args.seeds:
                name = cname(c, seed, V)
                ckpt_path = save_dir / f"{name}.pt"
                if ckpt_path.exists():
                    print(f"\n[{name}] 이미 학습됨(스킵). 재학습하려면 {ckpt_path} 삭제.")
                    continue
                print(f"\n{'='*60}\n[{name}] 재학습 시작 "
                      f"(variant={V}, gain_max={CONFIGS[c]['gain_max']}, fc_max={CONFIGS[c]['fc_max']})\n{'='*60}")
                set_seed(seed)
                model = build_model(c)
                TF.train_one(name, model, variant_loss(V),
                             train_ds, val_ds, device, cfg, save_dir)

    # ── 평가 루프 (config × seed) ────────────────────────────────────────────
    print(f"\n{'='*60}\n평가 ({args.test_split})\n{'='*60}")
    per_seed = {c: {} for c in configs}   # per_seed[cfg][seed] = result dict
    for c in configs:
        for seed in args.seeds:
            name = cname(c, seed, V)
            if V == "A0" and c == "g6_f16k":
                ckpt_path = BASELINE_CKPT          # 원본 비교군
            else:
                ckpt_path = save_dir / f"{name}.pt"
            if not Path(ckpt_path).exists():
                print(f"  [{c} seed={seed}] 체크포인트 없음 — 스킵 ({ckpt_path})")
                continue
            model = build_model(c)
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            model.load_state_dict(state, strict=False)
            r = evaluate(name, c, model, ds_test, device, args.batch_size)
            per_seed[c][seed] = r
            print(f"  [{c:>9} s{seed:>3}] LSD={r['lsd_mean']:.4f} DMR={r['dmr_mean']:.4f} "
                  f"CosSim={r['cos_mean']:.4f} | gain>6={r.get('gain_over6',0)*100:4.1f}% "
                  f"gain_sat={r.get('gain_sat_self',0)*100:4.1f}% fc>16k={r.get('fc_over16k',0)*100:4.1f}%")
            if V == "A0" and c == "g6_f16k":
                break  # baseline 은 seed 무관(동일 ckpt) — 1회만

    # ── 집계 (LSD/DMR/CosSim + saturation 나란히) ───────────────────────────
    print(f"\n{'='*96}\nconfig 별 seed 집계 (mean ± std)  —  test={args.test_split}\n{'='*96}")
    hdr = (f"{'config':>9} {'LSD':>14} {'DMR':>8} {'CosSim':>8} "
           f"{'gain_sat%':>9} {'gain>6%':>8} {'fc_sat%':>8} {'fc>16k%':>8} {'n':>3}")
    print(hdr); print("-" * len(hdr))
    summary = {}
    for c in configs:
        seeds = list(per_seed[c].keys())
        if not seeds:
            continue
        lsd_m, lsd_s, n = _mean_std([per_seed[c][s]["lsd_mean"] for s in seeds])
        dmr_m, _, _ = _mean_std([per_seed[c][s]["dmr_mean"] for s in seeds])
        cos_m, _, _ = _mean_std([per_seed[c][s]["cos_mean"] for s in seeds])
        gsat_m, _, _ = _mean_std([per_seed[c][s].get("gain_sat_self") for s in seeds])
        g6_m,  _, _ = _mean_std([per_seed[c][s].get("gain_over6") for s in seeds])
        fsat_m, _, _ = _mean_std([per_seed[c][s].get("fc_hi_self") for s in seeds])
        f16_m, _, _ = _mean_std([per_seed[c][s].get("fc_over16k") for s in seeds])
        print(f"{c:>9} {lsd_m:>7.4f}±{lsd_s:<5.4f} {dmr_m:>8.4f} {cos_m:>8.4f} "
              f"{gsat_m*100:>8.1f} {g6_m*100:>7.1f} {fsat_m*100:>7.1f} {f16_m*100:>7.1f} {n:>3}")
        summary[c] = dict(
            gain_max=CONFIGS[c]["gain_max"], fc_max=CONFIGS[c]["fc_max"],
            seeds=seeds, n=n,
            lsd=(lsd_m, lsd_s), dmr=dmr_m, cossim=cos_m,
            gain_sat_self=gsat_m, gain_over6=g6_m,
            fc_hi_self=fsat_m, fc_over16k=f16_m,
        )

    print("\n판단 기준: saturation(특히 gain>6%, fc>16k%)이 줄고 LSD가 개선되면 제약 완화가 유효.")
    print("           saturation만 줄고 LSD 불변 → 완화가 성능엔 무관(이것도 보고 대상).")

    out_path = out_dir / f"gain_freq_summary_{V}_{args.test_split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
