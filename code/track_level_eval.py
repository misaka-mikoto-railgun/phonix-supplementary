from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon


@dataclass
class MetricSummary:
    n_groups: int
    baseline_mean: float
    candidate_mean: float
    mean_diff: float
    ci_low: float
    ci_high: float
    p_ttest: float | None
    p_wilcoxon: float | None
    cohens_dz: float | None
    win_rate: float
    conditional_win_rate: float | None


@dataclass
class ComparisonSummary:
    group_key: str
    sample_level_n: int
    n_groups: int
    metrics: Dict[str, MetricSummary]


def _lsd(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - target) ** 2, axis=-1))


def _dmr(heard_pred: np.ndarray, pref_target: np.ndarray) -> np.ndarray:
    return np.mean((np.sign(heard_pred) == np.sign(pref_target)).astype(np.float32), axis=-1)


def _cossim(heard_pred: np.ndarray, pref_target: np.ndarray) -> np.ndarray:
    num = np.sum(heard_pred * pref_target, axis=-1)
    denom = np.linalg.norm(heard_pred, axis=-1) * np.linalg.norm(pref_target, axis=-1) + 1e-8
    return num / denom


def _heard_pred(payload: Dict[str, np.ndarray]) -> np.ndarray:
    if "heard_pred" in payload:
        return np.asarray(payload["heard_pred"], dtype=np.float32)
    if "pred" in payload and "room_target" in payload:
        return np.asarray(payload["pred"], dtype=np.float32) - np.asarray(payload["room_target"], dtype=np.float32)
    raise KeyError("Payload must contain 'heard_pred' or both 'pred' and 'room_target'.")


def _validate_payload_pair(baseline: Dict[str, np.ndarray], candidate: Dict[str, np.ndarray], group_key: str) -> None:
    required = ["pred", "target", "pref_target", group_key]
    for name, payload in (("baseline", baseline), ("candidate", candidate)):
        for key in required:
            if key not in payload:
                raise KeyError(f"{name} payload is missing required key '{key}'")

    if baseline["pred"].shape != candidate["pred"].shape:
        raise ValueError("Baseline/candidate prediction shapes do not match.")
    if np.asarray(baseline[group_key]).shape != np.asarray(candidate[group_key]).shape:
        raise ValueError(f"Baseline/candidate '{group_key}' arrays do not match in shape.")


def _aggregate_by_group(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    agg = np.zeros(len(unique_groups), dtype=np.float64)
    for idx, gid in enumerate(unique_groups):
        agg[idx] = float(values[groups == gid].mean())
    return unique_groups, agg


def _bootstrap_ci(diff: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        val = float(diff[0])
        return val, val

    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = diff[rng.integers(0, n, size=n)]
        means[i] = float(sample.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _cohens_dz(diff: np.ndarray) -> float | None:
    if len(diff) < 2:
        return None
    std = diff.std(ddof=1)
    if std < 1e-12:
        return None
    return float(diff.mean() / std)


def _paired_pvalues(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    if len(a) < 2:
        return None, None
    try:
        _, p_t = stats.ttest_rel(a, b)
        p_t = float(p_t)
    except Exception:
        p_t = None

    try:
        _, p_w = wilcoxon(a - b)
        p_w = float(p_w)
    except Exception:
        p_w = None
    return p_t, p_w


def _win_rates(diff: np.ndarray, higher_is_better: bool) -> tuple[float, float | None]:
    signed = diff if higher_is_better else -diff
    wins = signed > 0
    ties = signed == 0
    win_rate = float(wins.mean())
    if np.all(ties):
        return win_rate, None
    conditional = float(wins[~ties].mean())
    return win_rate, conditional


def _summarize_metric(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    n_boot: int,
    seed: int,
    higher_is_better: bool,
) -> MetricSummary:
    diff = candidate_values - baseline_values
    ci_low, ci_high = _bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    p_t, p_w = _paired_pvalues(candidate_values, baseline_values)
    dz = _cohens_dz(diff)
    win_rate, conditional = _win_rates(diff, higher_is_better=higher_is_better)
    return MetricSummary(
        n_groups=len(diff),
        baseline_mean=float(baseline_values.mean()),
        candidate_mean=float(candidate_values.mean()),
        mean_diff=float(diff.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        p_ttest=p_t,
        p_wilcoxon=p_w,
        cohens_dz=dz,
        win_rate=win_rate,
        conditional_win_rate=conditional,
    )


def compare_two_prediction_sets(
    baseline: Dict[str, np.ndarray],
    candidate: Dict[str, np.ndarray],
    group_key: str = "track_id",
    n_boot: int = 5000,
    seed: int = 42,
) -> ComparisonSummary:
    _validate_payload_pair(baseline, candidate, group_key)

    group_ids = np.asarray(baseline[group_key])
    if not np.array_equal(group_ids, np.asarray(candidate[group_key])):
        raise ValueError(f"Baseline and candidate '{group_key}' arrays must match exactly for paired comparison.")

    baseline_pred = np.asarray(baseline["pred"], dtype=np.float32)
    candidate_pred = np.asarray(candidate["pred"], dtype=np.float32)
    target = np.asarray(baseline["target"], dtype=np.float32)
    pref_target = np.asarray(baseline["pref_target"], dtype=np.float32)
    baseline_heard = _heard_pred(baseline)
    candidate_heard = _heard_pred(candidate)

    unique_groups, baseline_lsd = _aggregate_by_group(_lsd(baseline_pred, target), group_ids)
    _, candidate_lsd = _aggregate_by_group(_lsd(candidate_pred, target), group_ids)
    _, baseline_dmr = _aggregate_by_group(_dmr(baseline_heard, pref_target), group_ids)
    _, candidate_dmr = _aggregate_by_group(_dmr(candidate_heard, pref_target), group_ids)
    _, baseline_cossim = _aggregate_by_group(_cossim(baseline_heard, pref_target), group_ids)
    _, candidate_cossim = _aggregate_by_group(_cossim(candidate_heard, pref_target), group_ids)

    metrics = {
        "lsd": _summarize_metric(baseline_lsd, candidate_lsd, n_boot=n_boot, seed=seed, higher_is_better=False),
        "dmr": _summarize_metric(baseline_dmr, candidate_dmr, n_boot=n_boot, seed=seed + 1, higher_is_better=True),
        "cossim": _summarize_metric(
            baseline_cossim, candidate_cossim, n_boot=n_boot, seed=seed + 2, higher_is_better=True
        ),
    }

    return ComparisonSummary(
        group_key=group_key,
        sample_level_n=int(len(group_ids)),
        n_groups=int(len(unique_groups)),
        metrics=metrics,
    )
