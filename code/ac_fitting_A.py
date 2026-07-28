"""
AC Fitting  --  Option A: Per-Sample Scipy Optimization
========================================================
AC 모델의 128-bin dense output → 7-band biquad 로 per-sample fitting.
A0_Proposed 와 동일한 Gaussian PEQ 근사를 사용하므로 직접 비교 가능.

결과 해석:
  AC2_raw LSD       : AC2 원본 dense output vs ground truth
  AC2_biquad LSD    : biquad fitting 후 vs ground truth
  A0_Proposed LSD   : (참고) 학습 때부터 biquad 로 제약된 모델
  gap = AC2_biquad - AC2_raw : representation penalty (fitting loss)

Usage
-----
  # pred 배열 먼저 생성 (experiments_fixed_updated.py 실행 후)
  python ac_fitting_A.py --stat_dir ./paper_outputs/stats --out_dir ./paper_outputs

  # 빠른 테스트 (첫 N 샘플만)
  python ac_fitting_A.py --n_samples 200
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import pandas as pd

# ──────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────

SR        = 48000
N_FILTERS = 7
FC_MIN, FC_MAX   = 80.0, 16000.0
GAIN_MAX         = 12.0
Q_MIN,  Q_MAX    = 0.3, 8.0
TARGET_FREQS     = np.geomspace(20.0, 24000.0, 128).astype(np.float32)
OMEGA            = (2 * np.pi * TARGET_FREQS / SR).astype(np.float64)


# ──────────────────────────────────────────────────────────
# Gaussian PEQ 근사 (A0_Proposed 와 동일 공식)
# ──────────────────────────────────────────────────────────

def gaussian_peq_response(params: np.ndarray) -> np.ndarray:
    """
    params: (N_FILTERS * 3,) = [fc1, gain1, q1, ...]  (raw, unbounded)
    반환: (128,) dB  (bounded params 적용 후)
    """
    resp = np.zeros(128, dtype=np.float64)
    for i in range(N_FILTERS):
        fc_raw   = params[3*i]
        gain_raw = params[3*i + 1]
        q_raw    = params[3*i + 2]
        # sigmoid/tanh 경계 (A0 학습 시 동일)
        fc   = FC_MIN   + _sigmoid(fc_raw)   * (FC_MAX - FC_MIN)
        gain = _tanh(gain_raw) * GAIN_MAX
        q    = Q_MIN    + _sigmoid(q_raw)    * (Q_MAX  - Q_MIN)
        omega_0 = 2 * np.pi * fc / SR
        bw      = omega_0 / (q + 1e-9)
        delta   = np.abs(OMEGA - omega_0)
        w       = np.exp(-0.5 * (delta / (bw + 1e-9))**2)
        resp   += gain * w
    return resp


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def _tanh(x):
    return np.tanh(np.clip(x, -20, 20))


def fit_biquad(target_db: np.ndarray, n_restarts: int = 3) -> tuple[np.ndarray, float]:
    """
    target_db (128,) 에 대해 7-band Gaussian PEQ 파라미터 fitting.
    반환: (best_params, best_mse)
    """
    best_loss = np.inf
    best_params = None

    # 초기값 후보: fc를 주파수 축 균등 배치, gain=0, q=1.41
    fc_inits = np.geomspace(FC_MIN, FC_MAX, N_FILTERS)
    x0_base = np.zeros(N_FILTERS * 3)
    for i, fc in enumerate(fc_inits):
        # fc raw = logit((fc - fc_min)/(fc_max - fc_min))
        ratio = np.clip((fc - FC_MIN) / (FC_MAX - FC_MIN), 1e-6, 1-1e-6)
        x0_base[3*i]   = np.log(ratio / (1 - ratio))
        x0_base[3*i+1] = 0.0   # gain=0
        x0_base[3*i+2] = 0.0   # q≈1 (sigmoid(0)=0.5 → q=Q_MIN+0.5*(Q_MAX-Q_MIN))

    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        x0 = x0_base if restart == 0 else x0_base + rng.standard_normal(N_FILTERS * 3) * 0.5

        def objective(params):
            pred = gaussian_peq_response(params)
            return float(np.mean((pred - target_db)**2))

        res = minimize(objective, x0, method="L-BFGS-B",
                       options={"maxiter": 200, "ftol": 1e-10})
        if res.fun < best_loss:
            best_loss   = res.fun
            best_params = res.x

    return best_params, float(np.sqrt(best_loss))


# ──────────────────────────────────────────────────────────
# 메트릭
# ──────────────────────────────────────────────────────────

def lsd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((a - b)**2, axis=-1))

def bootstrap_ci(arr: np.ndarray, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(arr, len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat_dir",   default="./paper_outputs/stats")
    parser.add_argument("--out_dir",    default="./paper_outputs")
    parser.add_argument("--model",      default="AC2_GRU",
                        help="AC 모델명 (pred.npy 가 있어야 함)")
    parser.add_argument("--n_samples",  type=int, default=0,
                        help="0 = 전체 사용 (권장), N>0 이면 첫 N개만")
    parser.add_argument("--n_restarts", type=int, default=3)
    parser.add_argument("--gain_max",   type=float, default=12.0,
                        help="Per-section gain bound for biquad fitting (dB). "
                             "Default 12.0 reproduces the published Option-A result.")
    args = parser.parse_args()

    global GAIN_MAX                    # REVISION: 모듈 상수 오버라이드 (fit_biquad 가 참조)
    GAIN_MAX = args.gain_max
    print(f"[REVISION] GAIN_MAX = {GAIN_MAX}")

    stat_dir = Path(args.stat_dir)
    out_dir  = Path(args.out_dir)
    tab_dir  = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    # ── 로드 ────────────────────────────────────────────────
    pred_path   = stat_dir / f"{args.model}_pred.npy"
    dual_path   = stat_dir / "targets_dual.npy"
    a0_lsd_path = stat_dir / "A0_Proposed_lsd.npy"

    if not pred_path.exists():
        print(f"[ERROR] {pred_path} 없음.")
        print("  -> experiments_fixed_updated.py 를 먼저 실행하세요.")
        return
    if not dual_path.exists():
        print(f"[ERROR] {dual_path} 없음.")
        return

    pred_ac   = np.load(pred_path).astype(np.float64)   # (N, 128)
    dual_tgt  = np.load(dual_path).astype(np.float64)   # (N, 128)

    N = len(pred_ac)
    if args.n_samples > 0:
        N = min(args.n_samples, N)
        pred_ac  = pred_ac[:N]
        dual_tgt = dual_tgt[:N]

    print(f"AC Fitting Option A  --  {args.model}")
    print(f"N={N} samples, n_restarts={args.n_restarts}")
    print(f"Fitting {N_FILTERS}-band Gaussian PEQ per sample (same formula as A0_Proposed)")
    print()

    # ── Per-sample fitting ──────────────────────────────────
    lsd_raw    = lsd(pred_ac, dual_tgt)                 # AC dense vs GT
    lsd_fitted = np.zeros(N, dtype=np.float64)

    t0 = time.perf_counter()
    for i in range(N):
        best_p, _ = fit_biquad(pred_ac[i], args.n_restarts)
        fitted_resp = gaussian_peq_response(best_p)
        lsd_fitted[i] = float(np.sqrt(np.mean((fitted_resp - dual_tgt[i])**2)))

        if (i+1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i+1) * (N - i - 1)
            print(f"  [{i+1:4d}/{N}]  raw={lsd_raw[:i+1].mean():.3f}  "
                  f"fitted={lsd_fitted[:i+1].mean():.3f}  ETA {eta:.0f}s")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s  ({elapsed/N*1000:.1f} ms/sample)")

    # ── A0 참고값 ────────────────────────────────────────────
    a0_lsd = np.load(a0_lsd_path).astype(np.float64)[:N] if a0_lsd_path.exists() else None

    # ── 결과 출력 ────────────────────────────────────────────
    print()
    print(f"{'Metric':35s} | {'Mean':>8} | {'95% CI':>20}")
    print("-" * 68)

    lo, hi = bootstrap_ci(lsd_raw[:N])
    print(f"  {args.model} dense (raw)              | {lsd_raw.mean():>8.4f} | [{lo:.4f}, {hi:.4f}]")

    lo, hi = bootstrap_ci(lsd_fitted)
    print(f"  {args.model} biquad-fitted             | {lsd_fitted.mean():>8.4f} | [{lo:.4f}, {hi:.4f}]")

    gap = lsd_fitted - lsd_raw[:N]
    lo, hi = bootstrap_ci(gap)
    print(f"  Representation penalty (fitted-raw) | {gap.mean():>8.4f} | [{lo:.4f}, {hi:.4f}]")

    if a0_lsd is not None:
        lo, hi = bootstrap_ci(a0_lsd)
        print(f"  A0_Proposed (reference)             | {a0_lsd.mean():>8.4f} | [{lo:.4f}, {hi:.4f}]")
        margin = lsd_fitted - a0_lsd
        lo, hi = bootstrap_ci(margin)
        print(f"  AC_biquad advantage over A0         | {margin.mean():>8.4f} | [{lo:.4f}, {hi:.4f}]")

    # ── 저장 ────────────────────────────────────────────────
    np.save(stat_dir / f"{args.model}_biquad_fitted_lsd.npy", lsd_fitted)

    rows = [
        {"Model/Condition":             f"{args.model} dense (raw)",
         "LSD mean":                    f"{lsd_raw.mean():.4f}",
         "95% CI":                      f"[{bootstrap_ci(lsd_raw)[0]:.4f},{bootstrap_ci(lsd_raw)[1]:.4f}]"},
        {"Model/Condition":             f"{args.model} biquad-fitted",
         "LSD mean":                    f"{lsd_fitted.mean():.4f}",
         "95% CI":                      f"[{bootstrap_ci(lsd_fitted)[0]:.4f},{bootstrap_ci(lsd_fitted)[1]:.4f}]"},
        {"Model/Condition":             "Representation penalty",
         "LSD mean":                    f"{gap.mean():.4f}",
         "95% CI":                      f"[{bootstrap_ci(gap)[0]:.4f},{bootstrap_ci(gap)[1]:.4f}]"},
    ]
    if a0_lsd is not None:
        rows.append({"Model/Condition": "A0_Proposed (reference)",
                     "LSD mean":        f"{a0_lsd.mean():.4f}",
                     "95% CI":          f"[{bootstrap_ci(a0_lsd)[0]:.4f},{bootstrap_ci(a0_lsd)[1]:.4f}]"})

    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "table_ac_fitting_A.csv", index=False)
    print(f"\nSaved table_ac_fitting_A.csv")


if __name__ == "__main__":
    main()
