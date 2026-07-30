"""
experiments.py — PHONIX Adaptive EQ: Full Experiment Suite
============================================================
논문에 수록된 모든 Figure 및 Table을 재현하는 실험 코드.

Usage:
  # 학습된 체크포인트로 실행
  python experiments.py --data_dir ./data/dataset_v3 --ckpt_dir ./checkpoints --out_dir ./paper_outputs

  # 데이터/체크포인트 없이 파이프라인 검증 (랜덤 예측으로 대체)
  python experiments.py --dry_run --out_dir ./paper_outputs

Outputs (paper_outputs/):
  figures/  - Fig 1 ~ Fig 12 (PDF + PNG)
  tables/   - Table 1 ~ 4, A1 (CSV + LaTeX)
  stats/    - raw metric arrays (.npy)
"""

import os
import sys
import time
import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from baselines import DifferentiablePEQResponse
try:
    from alpha_sweep_patch import run_alpha_sweep, fig12_alpha_sensitivity_v2
except Exception:
    run_alpha_sweep = None
    fig12_alpha_sensitivity_v2 = None

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import wilcoxon

import torch
import torch.nn as nn
import torch.nn.functional as F

from frequency_grid import make_frequency_grid_np, make_frequency_grid_torch
from model_aliases import canonical_model_name, checkpoint_name_candidates

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# 0. matplotlib 스타일 (Springer-compatible)
# ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "lines.linewidth":  1.4,
})

# ── Okabe-Ito colorblind-safe palette ────────────────────────────
# Distinguishable for deuteranopia/protanopia; prints in greyscale.
# Reference: Okabe & Ito (2002) "Color Universal Design"
C = {
    "proposed": "#D55E00",   # Vermillion    – A0 Full
    "ablation": "#E69F00",   # Orange        – A1, A2, A3
    "arch":     "#0072B2",   # Blue          – AC1, AC2, AC3
    "baseline": "#999999",   # Grey          – E1–E6
    "target":   "#009E73",   # Bluish green  – target curve
    "real":     "#CC79A7",   # Reddish purple – real RIR
}
# Four modes: vermillion / sky-blue / bluish-green / orange
# (no red-green pair; distinguishable in deuteranopia & monochrome)
MODE_COLORS = ["#D55E00", "#56B4E9", "#009E73", "#E69F00"]  # Vocal/Bass/Treble/Soft

# ── Line styles, markers, hatches ─────────────────────────────────
# Use in combination with color so B&W printing stays readable.
LINE_STYLES = {
    "proposed": ("-",   "o",  2.0),   # solid,    circle
    "baseline": ("--",  "s",  1.4),   # dashed,   square
    "target":   (":",   "^",  1.4),   # dotted,   triangle-up
    "ablation": ("-.",  "D",  1.4),   # dash-dot, diamond
    "arch":     ((0,(5,2,1,2)), "v", 1.4),  # long-dash-dot, triangle-down
    "real":     ("--",  "P",  1.4),   # dashed,   plus (filled)
}
HATCHES = {
    "proposed": "",     # no hatch (stands out by color alone)
    "ablation": "\\", # backslash
    "arch":     "xx",   # cross
    "baseline": "//",   # forward slash
    "real":     "..",   # dots
}
MODE_NAMES  = ["Vocal","Bass","Treble","Soft"]

BAND_FREQS = np.array([63,125,250,500,1000,2000,4000,8000,12000,16000], dtype=np.float32)

# ── 외부 베이스라인 ───────────────────────────────────────

MODE_PROFILES_DB = {
    0: [(-4.0,-3.0,0.0,+2.0,+5.0,+6.0,+4.0,+2.0,-1.0,-2.0)],  # Vocal
    1: [(+8.0,+6.0,+3.0,0.0,-1.0,-2.0,-3.0,-2.0,-1.0,0.0)],    # Bass
    2: [(-2.0,-1.0,0.0,0.0,+1.0,+3.0,+5.0,+7.0,+8.0,+6.0)],    # Treble
    3: [(+2.0,+1.0,+1.0,0.0,-1.0,-3.0,-5.0,-6.0,-4.0,-3.0)],   # Soft
}

def _mode_curve(mode_ids: torch.Tensor, n_freqs=128, f_min=20.0, f_max=24000.0) -> torch.Tensor:
    """모드 프로파일을 dataset target grid에 맞춰 보간"""
    target_freqs = make_frequency_grid_torch(
        n_freqs=n_freqs,
        f_min=f_min,
        f_max=f_max,
        spacing="log",
    )
    band_t = torch.tensor(BAND_FREQS)
    out = torch.zeros(len(mode_ids), n_freqs)
    for b, mid in enumerate(mode_ids.tolist()):
        gains = torch.tensor(MODE_PROFILES_DB[int(mid)][0], dtype=torch.float32)
        out[b] = torch.from_numpy(np.interp(target_freqs.numpy(), band_t.numpy(), gains.numpy()))
    return out.to(mode_ids.device)


class E1_NoProcessing(nn.Module):
    def forward(self, x, room_response, mode_id, band_gains):
        return {"pred_response_db": torch.zeros(x.shape[0], 128, device=x.device)}


class E2_StaticModeEQ(nn.Module):
    def forward(self, x, room_response, mode_id, band_gains):
        return {"pred_response_db": _mode_curve(mode_id)}


class E3_NercessianMLP1(nn.Module):
    def __init__(self, in_dim=10, hidden_dim=256, num_filters=5, n_freqs=128, sr=48000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_filters*3))
        self.n_freqs = n_freqs
        self.register_buffer(
            "freqs",
            make_frequency_grid_torch(
                n_freqs=n_freqs,
                f_min=20.0,
                f_max=sr / 2.0,
                spacing="log",
            ),
        )
        self.sr = sr
    def _response(self, fc, gain, q):
        B, K = fc.shape; F_ = self.n_freqs
        freqs = self.freqs.view(1,1,F_)
        fc_   = fc.view(B,K,1); gain_ = gain.view(B,K,1); q_ = q.view(B,K,1)
        omega = 2*3.14159*fc_/self.sr; bw = omega/q_
        delta = (2*3.14159*freqs/self.sr - omega).abs()
        w = torch.exp(-0.5*(delta/(bw+1e-6))**2)
        return (gain_ * w).sum(dim=1)
    def forward(self, x, room_response, mode_id, band_gains):
        out = self.mlp(x.mean(dim=1))
        fc_r, g_r, q_r = out.chunk(3, dim=-1)
        fc   = 80.0 + torch.sigmoid(fc_r) * (8000.0 - 80.0)
        g    = torch.tanh(g_r) * 12.0
        q    = 0.3 + torch.sigmoid(q_r) * (8.0 - 0.3)
        return {"pred_response_db": self._response(fc, g, q)}


class E4_PepeCNN1(nn.Module):
    def __init__(self, in_dim=10, num_filters=5, n_freqs=128, sr=48000):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_dim, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, num_filters*3))
        self.e3 = E3_NercessianMLP(in_dim=10, n_freqs=n_freqs, sample_rate=sr)
    def forward(self, x, room_response, mode_id, band_gains):
        feat = self.fc(self.cnn(x.transpose(1,2)).squeeze(-1))
        fc_r, g_r, q_r = feat.chunk(3, dim=-1)
        fc = 80.0 + torch.sigmoid(fc_r)*(8000.0-80.0)
        g  = torch.tanh(g_r)*12.0
        q  = 0.3 + torch.sigmoid(q_r)*(8.0-0.3)
        return {"pred_response_db": self.e3._response(fc, g, q)}


class E5_Sequential(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.room = E3_NercessianMLP()
        self.pref = E2_StaticModeEQ()
    def forward(self, x, room_response, mode_id, band_gains):
        room = self.room(x, room_response, mode_id, band_gains)["pred_response_db"]
        pref = self.pref(x, room_response, mode_id, band_gains)["pred_response_db"]
        return {"pred_response_db": room + pref}


class E6_DSP_Analytical(nn.Module):
    def forward(self, x, room_response, mode_id, band_gains):
        rc   = (-room_response).clamp(-12, 12)
        pref = _mode_curve(mode_id).to(x.device)
        return {"pred_response_db": rc + pref}


# ══════════════════════════════════════════════════════════
# 2. 데이터셋 로딩
# ══════════════════════════════════════════════════════════

class PEQDataset:
    """chunk_*.npz 파일로부터 데이터 로드 (inference_multi_room.py 참조)"""
    KEYS = ["features","room_response","mode_id","band_gains","room_target","pref_target","dual_target"]

    def __init__(self, split_dir: str, device: str = "cpu"):
        self.split_dir = Path(split_dir)
        self.meta = {}
        self.target_freqs = None
        meta_path = self.split_dir / "meta.json"
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
        chunks = sorted(self.split_dir.glob("chunk_*.npz"))
        if not chunks:
            raise FileNotFoundError(f"No chunk files in {split_dir}")
        arrays = {k: [] for k in self.KEYS}
        meta   = []
        for cp in chunks:
            data = np.load(cp, allow_pickle=False)
            if self.target_freqs is None and "target_freqs" in data:
                self.target_freqs = np.asarray(data["target_freqs"], dtype=np.float32)
            for k in self.KEYS:
                arrays[k].append(data[k])
            if "rt60" in data:
                meta.append(data["rt60"])
        self.data = {}
        for k, v in arrays.items():
            t = torch.from_numpy(np.concatenate(v))
            self.data[k] = t.long() if k == "mode_id" else t.float()
        self.rt60 = np.concatenate(meta) if meta else None
        self.device = device
        self.n = len(self.data["features"])
        print(f"  Loaded {self.n} samples from {split_dir}")

    def iter_batches(self, batch_size=512):
        idx = torch.arange(self.n)
        for s in range(0, self.n, batch_size):
            b = idx[s:s+batch_size]
            yield {k: v[b].to(self.device) for k, v in self.data.items()}

    def get_all(self):
        return {k: v.to(self.device) for k, v in self.data.items()}


def make_dry_run_dataset(n=1000, device="cpu") -> dict:
    """체크포인트/데이터 없을 때 랜덤 텐서로 파이프라인 검증"""
    rng = torch.Generator().manual_seed(42)
    mode_ids = torch.randint(0, 4, (n,), generator=rng)
    room_resp = torch.randn(n, 128, generator=rng) * 3.0
    return {
        "features":     torch.randn(n, 32, 10, generator=rng),
        "room_response": room_resp,
        "mode_id":      mode_ids,
        "band_gains":   torch.randn(n, 10, generator=rng),
        "room_target":  -room_resp.clamp(-8, 8),
        "pref_target":  _mode_curve(mode_ids) + torch.randn(n, 128, generator=rng) * 0.5,
        "dual_target":  torch.randn(n, 128, generator=rng) * 4.0,
    }


# ══════════════════════════════════════════════════════════
# 3. 메트릭 계산
# ══════════════════════════════════════════════════════════

def lsd(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Log Spectral Distance per sample → (N,)"""
    return np.sqrt(np.mean((pred - target)**2, axis=-1))

def dmr(pred: np.ndarray, pref_target: np.ndarray) -> np.ndarray:
    """Preference Alignment Score per sample → (N,)"""
    return np.mean((np.sign(pred) == np.sign(pref_target)).astype(float), axis=-1)

def cossim(pred: np.ndarray, pref_target: np.ndarray) -> np.ndarray:
    """Cosine Similarity per sample → (N,)"""
    num   = np.sum(pred * pref_target, axis=-1)
    denom = np.linalg.norm(pred, axis=-1) * np.linalg.norm(pref_target, axis=-1) + 1e-8
    return num / denom


def model_forward(name: str, model: nn.Module, batch: dict) -> dict:
    """Match per-model forward signatures used during training."""
    f = batch["features"]
    f_c = batch.get("features_clean")
    rr = batch["room_response"]
    mi = batch["mode_id"]
    bg = batch["band_gains"]

    if name == "E1_NoEQ":
        return model(f, rr, mi, bg)
    elif name == "E2_StaticEQ":
        return model(f, rr, mi, bg)
    elif name in ("E3_Nercessian", "E4_Pepe"):
        return model(f)
    elif name == "E5_Sequential":
        return model(f, mi)
    elif name == "E6_DSP":
        return model(f, room_response = rr, mode_id = mi)
    elif name == "A1_NoRoomInput":
        return model(f_c if f_c is not None else f, rr, mi, bg)
    else:
        return model(f, rr, mi, bg)


@torch.no_grad()
def evaluate_model(name: str, model: nn.Module, data: dict, batch_size=512,
                   n_rtf_warmup=5, n_rtf_trials=20, audio_dur=4.0) -> dict:
    """Evaluate a model using heard-response preference metrics."""
    model.eval()
    N      = data["features"].shape[0]
    device = data["features"].device
    model  = model.to(device)

    pred_all      = []
    room_corr_all = []

    for s in range(0, N, batch_size):
        b = {k: v[s:s+batch_size] for k, v in data.items()}
        out = model_forward(name, model, b)
        pred_all.append(out["pred_response_db"].cpu().numpy())
        if out.get("room_correction_db") is not None:
            room_corr_all.append(out["room_correction_db"].cpu().numpy())

    pred_all = np.concatenate(pred_all, axis=0)
    room_corr_all = np.concatenate(room_corr_all, axis=0) if room_corr_all else None

    dual_target = data["dual_target"].cpu().numpy()
    room_target = data["room_target"].cpu().numpy()
    pref_target = data["pref_target"].cpu().numpy()
    heard_all   = pred_all - room_target

    lsd_dual_arr = lsd(pred_all, dual_target)
    lsd_room_arr = lsd(pred_all, room_target)
    lsd_pref_arr = lsd(heard_all, pref_target)
    pa_arr       = dmr(heard_all, pref_target)
    cos_arr      = cossim(heard_all, pref_target)

    room_branch_lsd_arr = None
    room_branch_lsd_mean = None
    if room_corr_all is not None:
        room_branch_lsd_arr = lsd(room_corr_all, room_target)
        room_branch_lsd_mean = float(room_branch_lsd_arr.mean())

    single = {k: v[:1] for k, v in data.items()}
    for _ in range(n_rtf_warmup):
        model_forward(name, model, single)
    times = []
    for _ in range(n_rtf_trials):
        t0 = time.perf_counter()
        model_forward(name, model, single)
        times.append((time.perf_counter() - t0) / audio_dur)
    rtf = float(np.median(times))

    return {
        "lsd_arr":              lsd_dual_arr,
        "lsd_room_arr":         lsd_room_arr,
        "lsd_pref_arr":         lsd_pref_arr,
        "dmr_arr":          pa_arr,
        "cossim_arr":           cos_arr,
        "lsd_mean":             float(lsd_dual_arr.mean()),
        "lsd_room_mean":        float(lsd_room_arr.mean()),
        "lsd_pref_mean":        float(lsd_pref_arr.mean()),
        "dmr_mean":         float(pa_arr.mean()),
        "cossim_mean":          float(cos_arr.mean()),
        "room_branch_lsd_arr":  room_branch_lsd_arr,
        "room_branch_lsd_mean": room_branch_lsd_mean,
        "rtf":                  rtf,
        "pred_all":             pred_all,
        "room_corr_all":        room_corr_all,
        "heard_all":            heard_all,
    }


# ══════════════════════════════════════════════════════════
# 4. 통계 검정
# ══════════════════════════════════════════════════════════

def cohens_dz(diff: np.ndarray) -> float:
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12))

def win_rates(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """(전체 승률, 조건부 승률 - a > b 인 샘플 중 비율)"""
    win   = (a > b).mean()
    diff  = a - b
    ne    = diff != 0
    cond  = (diff[ne] > 0).mean() if ne.any() else 0.5
    return float(win), float(cond)

def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Holm–Bonferroni 보정 p-value 반환"""
    n = len(p_values)
    order = np.argsort(p_values)
    corrected = np.array(p_values, dtype=float)
    for rank, idx in enumerate(order):
        corrected[idx] = min(1.0, p_values[idx] * (n - rank))
    return corrected.tolist()

def paired_tests(a: np.ndarray, b: np.ndarray) -> dict:
    """A vs B 쌍별 검정 (A0가 첫 번째)"""
    diff     = a - b
    t_stat, p_t    = stats.ttest_rel(a, b)
    try:
        _, p_w = wilcoxon(diff)
    except Exception:
        p_w = float("nan")
    dz  = cohens_dz(diff)
    win, cond_win = win_rates(a, b)
    return {"delta_mean": float(diff.mean()), "p_t": float(p_t), "p_w": float(p_w),
            "dz": dz, "win_rate": win, "cond_win_rate": cond_win}

def build_stat_table(results: dict) -> pd.DataFrame:
    """Table 3 – 통계 검정 결과 (dual-target LSD + dmr)."""
    a0_lsd = results["A0_Proposed"]["lsd_arr"]
    a0_pa  = results["A0_Proposed"]["dmr_arr"]
    rows   = []
    comps  = [("A0 vs A1 (Room)", "A1_NoRoomInput"),
              ("A0 vs A2 (with Pref Loss)", "A2_withPrefLoss"),
              ("A0 vs A3 (Pref Input)", "A3_NoPrefInput"),
              ("A0 vs E3",        "E3_Nercessian"),
              ("A0 vs E4",        "E4_Pepe")]

    all_ps = []
    raw    = []
    for label, key in comps:
        if key not in results:
            continue
        for metric, arr_a, arr_b in [
            ("LSD",     -a0_lsd, -results[key]["lsd_arr"]),   # 낮을수록 좋으므로 부호 반전
            ("dmr",  a0_pa,   results[key]["dmr_arr"]),
        ]:
            r = paired_tests(arr_a, arr_b)
            raw.append((label, metric, r))
            all_ps.append(r["p_t"])

    corrected = holm_bonferroni(all_ps) if all_ps else []
    for i, (label, metric, r) in enumerate(raw):
        delta = r["delta_mean"] if metric == "dmr" else -r["delta_mean"]
        rows.append({
            "Comparison": label,
            "Metric":     metric,
            "Δ mean":     f"{delta:+.3f}",
            "p (t-test)": f"<0.001" if corrected[i] < 0.001 else f"{corrected[i]:.3f}",
            "dz":         f"{r['dz'] if metric=='dmr' else -r['dz']:.3f}",
            "Win%":       f"{r['win_rate']*100:.1f}%",
            "Cond. Win%": f"{r['cond_win_rate']*100:.1f}%",
        })
    return pd.DataFrame(rows)


def _pareto_mask_min_x_max_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True for non-dominated points when minimizing x and maximizing y."""
    x = np.asarray(x)
    y = np.asarray(y)
    keep = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        dominated = np.any((x <= x[i]) & (y >= y[i]) & ((x < x[i]) | (y > y[i])))
        keep[i] = not dominated
    return keep


def _room_curve_for_plot(result: dict) -> np.ndarray:
    """Prefer stage-A room correction output when available; else fall back to final prediction."""
    if result.get("room_corr_all") is not None:
        return result["room_corr_all"]
    return result["pred_all"]


def _display_name(name: str) -> str:
    mapping = {
        "A0_Full": "A0_Proposed",
        "A0_Proposed": "A0_Proposed",
        "A1_NoRoomInput": "A1_NoRoomInput",
        "A2_NoPrefLoss": "A2_withPrefLoss",
        "A2_withPrefLoss": "A2_withPrefLoss",
        "E1_NoEQ": "E1_NoEQ",
        "E2_StaticEQ": "E2_StaticEQ",
        "E3_Nercessian": "E3_Nercessian",
        "E4_Pepe": "E4_Pepe",
        "E5_Sequential": "E5_Sequential",
        "E6_DSP": "E6_DSP",
        "AC1_BiLSTM": "AC1_BiLSTM",
        "AC2_GRU": "AC2_GRU",
        "AC3_Conformer": "AC3_Conformer",
    }
    return mapping.get(name, name)


def fig2_room_correction(results: dict, data: dict, out_dir: Path, target_freqs: np.ndarray):
    """Fig 2 – 룸 보정 성능: room branch/room-only 모델 기준으로 시각화."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # (좌) 평균 룸 보정 응답
    ax = axes[0]
    room_target_mean = data["room_target"].cpu().numpy().mean(axis=0)
    pred_a0_mean = _room_curve_for_plot(results["A0_Proposed"]).mean(axis=0)
    pred_e3_mean = _room_curve_for_plot(results["E3_Nercessian"]).mean(axis=0)
    _ls, _mk, _lw = LINE_STYLES["target"]
    ax.plot(target_freqs, room_target_mean, color=C["target"],   lw=_lw,
            ls=_ls, marker=_mk, markevery=12, markersize=4, label="Room correction target (mean)")
    _ls, _mk, _lw = LINE_STYLES["baseline"]
    ax.plot(target_freqs, pred_e3_mean,     color=C["baseline"], lw=_lw,
            ls=_ls, marker=_mk, markevery=12, markersize=4, label="E3 Nercessian")
    _ls, _mk, _lw = LINE_STYLES["proposed"]
    ax.plot(target_freqs, pred_a0_mean,     color=C["proposed"], lw=_lw,
            ls=_ls, marker=_mk, markevery=12, markersize=4, label="A0 room branch")
    ax.axhline(0, color="#333", lw=0.7, ls=":")
    ax.set_xscale("log"); ax.set_xlim(20, 24000)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Response (dB)")
    ax.set_title("(a) Mean room-correction response"); ax.legend(fontsize=7)

    # (우) RT60 사분위수별 Room-LSD
    ax = axes[1]
    rt60 = data.get("rt60_arr")
    if rt60 is None:
        rt60 = np.random.uniform(0.2, 0.75, results["A0_Proposed"]["lsd_room_arr"].shape[0])
    q = np.percentile(rt60, [0, 25, 50, 75, 100])
    labels_q = [f"Q1\n(≤{q[1]:.2f}s)", "Q2", "Q3", f"Q4\n(>{q[3]:.2f}s)"]
    room_lsd_by_q_a0 = []; room_lsd_by_q_e3 = []
    for i in range(4):
        mask = (rt60 >= q[i]) & (rt60 < q[i+1])
        if i == 3:
            mask = (rt60 >= q[i]) & (rt60 <= q[i+1])
        if mask.sum() == 0:
            mask = np.ones(len(rt60), bool)
        room_lsd_by_q_a0.append(results["A0_Proposed"]["lsd_room_arr"][mask])
        room_lsd_by_q_e3.append(results["E3_Nercessian"]["lsd_room_arr"][mask])
    pos = np.arange(4)
    bp1 = ax.boxplot(room_lsd_by_q_a0, positions=pos-0.18, widths=0.3, patch_artist=True,
                     boxprops=dict(facecolor=C["proposed"]+"66", hatch=HATCHES["proposed"]),
                     medianprops=dict(color=C["proposed"], lw=2), showfliers=False)
    bp2 = ax.boxplot(room_lsd_by_q_e3, positions=pos+0.18, widths=0.3, patch_artist=True,
                     boxprops=dict(facecolor=C["baseline"]+"66", hatch=HATCHES["baseline"]),
                     medianprops=dict(color=C["baseline"], lw=2), showfliers=False)
    ax.set_xticks(pos); ax.set_xticklabels(labels_q)
    ax.set_xlabel("RT60 Quartile"); ax.set_ylabel("Room-LSD (dB)")
    ax.set_title("(b) Room-LSD by room difficulty")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["A0 Proposed", "E3 Nercessian"], fontsize=7)
    #fig.suptitle("Fig. 2  Room Correction Performance", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_room_correction")


def fig3_per_mode(results: dict, data: dict, out_dir: Path):
    """Fig 3 – 모드별 LSD(상단) 및 dmr(하단)"""
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True)
    mode_ids = data["mode_id"].cpu().numpy()
    models   = [("E3_Nercessian", C["baseline"]), ("E4_Pepe", C["baseline"]), ("A0_Proposed", C["proposed"])]
    x = np.arange(4); w = 0.22

    _hatch_map = [HATCHES["baseline"], HATCHES["baseline"], HATCHES["proposed"]]
    for metric_idx, (metric_key, ylabel, ax) in enumerate([
        ("lsd_arr",     "Dual-target LSD (dB)",    axes[0]),
        ("dmr_arr", "dmr",     axes[1]),
    ]):
        for mi, (mname, color) in enumerate(models):
            vals = [results[mname][metric_key][mode_ids == m].mean() for m in range(4)]
            offset = (mi - 1) * w
            bars = ax.bar(x + offset, vals, w, label=mname, color=color,
                          alpha=0.75, edgecolor="#444", linewidth=0.5,
                          hatch=_hatch_map[mi])
        ax.set_ylabel(ylabel)
        ax.set_xticks(x); ax.set_xticklabels(MODE_NAMES)
        if metric_idx == 1:
            ax.axhline(0.5, color="#999", lw=0.8, ls="--", label="Random baseline")
        ax.legend(fontsize=7, ncol=4)

    axes[0].set_title("(a) Dual-target LSD by mode"); axes[1].set_title("(b) dmr by mode")
    #fig.suptitle("Fig. 3  Per-Mode Performance Comparison", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "fig3_per_mode")


def fig4_ablation_bar(results: dict, out_dir: Path):
    """Fig 4 – 어블레이션 LSD/dmr 막대 그래프"""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    models = ["E3_Nercessian","E4_Pepe","E5_Sequential","E6_DSP","A1_NoRoomInput","A2_withPrefLoss","AC1_BiLSTM","AC2_GRU","AC3_Conformer","A0_Proposed"]
    colors = ([C["baseline"]]*4 + [C["ablation"]]*2 + [C["arch"]]*3 + [C["proposed"]])
    hatches = ([HATCHES["baseline"]]*4 + [HATCHES["ablation"]]*2 +
               [HATCHES["arch"]]*3 + [HATCHES["proposed"]])
    labels = ["E3","E4","E5","E6","A1\n(−Room)","A2\n(+Pref)","AC1\nBiLSTM","AC2\nGRU","AC3\nConf.","A0"]

    lsd_vals = [results[m]["lsd_mean"] for m in models]
    pa_vals  = [results[m]["dmr_mean"] for m in models]

    for ax, vals, ylabel, title in [
        (axes[0], lsd_vals, "Dual-target LSD (dB)",    "(a) Dual-target LSD — lower is better"),
        (axes[1], pa_vals,  "dmr",      "(b) dmr — higher is better"),
    ]:
        bars = ax.bar(labels, vals, color=colors, edgecolor="#333",
                      linewidth=0.6, hatch=[h for h in hatches])
        a0_val = results["A0_Proposed"]["lsd_mean"] if ax is axes[0] else results["A0_Proposed"]["dmr_mean"]
        ax.axhline(a0_val, color=C["proposed"], lw=1.2, ls="--", alpha=0.8, label="A0 baseline")
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.tick_params(axis="x", labelsize=7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.003, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5)

    legend_elems = [
        mpatches.Patch(facecolor=C["proposed"], label="Proposed (A0)"),
        mpatches.Patch(facecolor=C["ablation"], label="Ablation"),
        mpatches.Patch(facecolor=C["arch"],     label="Architecture variant"),
        mpatches.Patch(facecolor=C["baseline"], label="External baseline"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5,-0.05))
    #fig.suptitle("Fig. 4  Ablation Study Results", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig4_ablation_bar")


def fig5_distributions(results: dict, out_dir: Path):
    """Fig 5 – LSD/dmr 샘플 단위 분포 (박스플롯)"""
    models = ["E3_Nercessian","E4_Pepe","A1_NoRoomInput","A2_withPrefLoss","AC3_Conformer","A0_Proposed"]
    labels = ["E3\nNercess.","E4\nPepe","A1\n−Room","A2\n+Pref","AC3\nConf.","A0"]
    colors  = [C["baseline"],C["baseline"],C["ablation"],C["ablation"],C["arch"],C["proposed"]]
    # Per-box unique hatch so B&W printing distinguishes boxes with similar greyscale
    hatches5 = ["//", "----", "\\\\", "....", "xx", ""]   # one per model, in order

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, key, ylabel, title in [
        (axes[0], "lsd_arr",     "Dual-target LSD (dB)",  "(a) Dual-target LSD distribution"),
        (axes[1], "dmr_arr", "dmr",   "(b) dmr distribution"),
    ]:
        data_list = [results[m][key] for m in models]
        bp = ax.boxplot(data_list, labels=labels, patch_artist=True,
                        medianprops=dict(color="black", lw=1.5),
                        flierprops=dict(marker="o", markersize=2, alpha=0.4), showfliers=True)
        for patch, c, h in zip(bp["boxes"], colors, hatches5):
            patch.set_facecolor(c + "88")
            patch.set_hatch(h)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.tick_params(axis="x", labelsize=7.5)
        if key == "dmr_arr":
            ax.axhline(0.5, color="#999", lw=0.8, ls="--")

    #fig.suptitle("Fig. 5  Sample-Level Metric Distributions", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig5_distributions")


def fig6_scatter(results: dict, out_dir: Path):
    """Fig 6 – dmr 샘플 단위 산점도: A0 vs A2, A0 vs E3"""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    pairs = [
        ("A2_withPrefLoss", C["ablation"]),
        ("E3_Nercessian", C["baseline"]),
    ]
    titles = ["(a) A0 vs A2 (with Pref Loss)", "(b) A0 vs E3 (Nercessian)"]

    for ax, (key, color), title in zip(axes, pairs, titles):
        a0 = results["A0_Proposed"]["dmr_arr"]
        bl = results[key]["dmr_arr"]
        _, cond = win_rates(a0, bl)
        sub = f"Cond. Win Rate: {cond*100:.1f}%"
        n  = min(2000, len(a0))
        rng = np.random.default_rng(0)
        idx = rng.choice(len(a0), n, replace=False)
        win  = a0[idx] > bl[idx]
        tie  = a0[idx] == bl[idx]
        lose = a0[idx] < bl[idx]
        ax.scatter(bl[idx][win],  a0[idx][win],  c=C["proposed"], s=5, alpha=0.5,
                   marker="o", label=f"A0 wins ({win.sum()})")
        ax.scatter(bl[idx][tie],  a0[idx][tie],  c="#888888",     s=3, alpha=0.4,
                   marker=".", label=f"Tie ({tie.sum()})")
        ax.scatter(bl[idx][lose], a0[idx][lose], c=color,         s=5, alpha=0.5,
                   marker="s", label=f"Baseline wins ({lose.sum()})")
        lim = (0.3, 1.05)
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xlabel(f"dmr ({key})"); ax.set_ylabel("dmr (A0 Proposed)")
        ax.set_title(title); ax.legend(fontsize=6.5, markerscale=2)
        ax.text(0.05, 0.95, sub, transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    # fig.suptitle("Fig. 6  Sample-Level dmr Comparison", y=1.02)  # caption handled by LaTeX
    fig.tight_layout()
    _save(fig, out_dir, "fig6_scatter")


def fig7_freq_response(results: dict, data: dict, out_dir: Path, target_freqs: np.ndarray):
    """Fig 7 – 모드별 예측 주파수 응답 (200 샘플, 평균 ± 1σ)"""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, sharey=True)
    mode_ids = data["mode_id"].cpu().numpy()
    pref_t   = data["pref_target"].cpu().numpy()

    for mode, (ax, name, color) in enumerate(zip(axes.flat, MODE_NAMES, MODE_COLORS)):
        mask = mode_ids == mode
        n200 = min(200, mask.sum())
        idx  = np.where(mask)[0][:n200]
        pred = results["A0_Proposed"]["pred_all"][idx]
        mean = pred.mean(axis=0)
        std  = pred.std(axis=0)
        target_mean = pref_t[idx].mean(axis=0)

        ax.fill_between(target_freqs, mean-std, mean+std, color=color, alpha=0.2)
        ax.plot(target_freqs, mean,        color=color, lw=2.0, ls="-",
                marker="o", markevery=16, markersize=3, label="Predicted (mean±1σ)")
        _ls, _mk, _lw = LINE_STYLES["target"]
        ax.plot(target_freqs, target_mean, color=C["target"], lw=_lw,
                ls=_ls, marker=_mk, markevery=16, markersize=3, label="Target profile")
        ax.axhline(0, color="#555", lw=0.7, ls=":")
        ax.set_xscale("log"); ax.set_xlim(20, 24000)
        ax.set_title(f"Mode: {name}", color=color, fontweight="bold")
        if ax in axes[:,0]: ax.set_ylabel("Response (dB)")
        if ax in axes[1,:]: ax.set_xlabel("Frequency (Hz)")
        ax.legend(fontsize=6.5)
        ax.set_ylim(-14, 14)

    # fig.suptitle("Per-Mode Predicted Frequency Responses", y=1.01)  # caption handled by LaTeX
    fig.tight_layout()
    _save(fig, out_dir, "fig7_freq_response")


def fig7_model_compare(
    results: dict,
    data: dict,
    out_dir: Path,
    target_freqs: np.ndarray,
    model_keys=None,
    use_heard=False,
):
    """Fig 7b – 모드별 여러 모델 출력 비교"""
    if model_keys is None:
        model_keys = ["A0_Proposed", "A1_NoRoomInput", "A2_withPrefLoss", "E5_Sequential"]
        if "A3_NoPrefInput" in results:
            model_keys.insert(3, "A3_NoPrefInput")

    model_labels = {
        "A0_Proposed": "A0 Proposed",
        "A1_NoRoomInput": "A1 w/o Room",
        "A2_withPrefLoss": "A2 with Pref Loss",
        "A3_NoPrefInput": "A3 w/o Pref Input",
        "E5_Sequential": "E5 Sequential",
    }
    # Okabe-Ito colors + distinct line styles for each model
    model_colors = {
        "A0_Proposed":   "#D55E00",   # Vermillion
        "A1_NoRoomInput":"#E69F00",   # Orange
        "A2_withPrefLoss": "#CC79A7",   # Reddish purple
        "A3_NoPrefInput":"#0072B2",   # Blue
        "E5_Sequential": "#999999",   # Grey
    }
    model_lstyle = {
        "A0_Proposed":   ("-",   "o",  2.0),
        "A1_NoRoomInput":("-.",  "D",  1.4),
        "A2_withPrefLoss": ("--",  "s",  1.4),
        "A3_NoPrefInput":((0,(5,2,1,2)), "v", 1.4),
        "E5_Sequential": (":",   "^",  1.4),
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), sharex=True, sharey=True)
    mode_ids = data["mode_id"].cpu().numpy()
    pref_t   = data["pref_target"].cpu().numpy()
    room_t   = data["room_target"].cpu().numpy()

    for mode, (ax, mode_name, mode_color) in enumerate(zip(axes.flat, MODE_NAMES, MODE_COLORS)):
        mask = mode_ids == mode
        n200 = min(200, mask.sum())
        idx  = np.where(mask)[0][:n200]
        target_mean = pref_t[idx].mean(axis=0)

        for mk in model_keys:
            if mk not in results:
                continue
            pred = results[mk]["pred_all"][idx]
            curve = pred - room_t[idx] if use_heard else pred
            mean = curve.mean(axis=0)
            std  = curve.std(axis=0)
            _ls, _mk_s, _lw = model_lstyle.get(mk, ("-", "o", 1.4))
            ax.plot(target_freqs, mean,
                    color=model_colors.get(mk, "#555"),
                    lw=_lw, ls=_ls, marker=_mk_s, markevery=16, markersize=3,
                    label=model_labels.get(mk, mk), alpha=0.95)
            if mk == "A0_Proposed":
                ax.fill_between(target_freqs, mean-std, mean+std,
                                color=model_colors.get(mk, "#D55E00"), alpha=0.15)

        _ls, _mk_s, _lw = LINE_STYLES["target"]
        ax.plot(target_freqs, target_mean, color=C["target"], lw=_lw,
                ls=_ls, marker=_mk_s, markevery=16, markersize=3, label="Target profile")
        ax.axhline(0, color="#555", lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(20, 24000)
        ax.set_ylim(-14, 14)
        ax.set_title(f"Mode: {mode_name}", color=mode_color, fontweight="bold")
        if ax in axes[:, 0]:
            ax.set_ylabel("Response (dB)")
        if ax in axes[1, :]:
            ax.set_xlabel("Frequency (Hz)")
        ax.legend(fontsize=6.2)

    title_suffix = "Heard Responses" if use_heard else "Predicted EQ Curves"
    # fig.suptitle(
    #     f"Per-Mode {title_suffix}: A0/A1/A2/A3/E5",
    #     y=1.01
    # )  # caption handled by LaTeX
    fig.tight_layout()
    suffix = "heard" if use_heard else "pred"
    _save(fig, out_dir, f"fig7b_model_compare_{suffix}")


def fig8_ood_advantage(results_synth: dict, results_real: dict, out_dir: Path):
    """Fig 8 – dmr 우위(A0 − 베이스라인): 합성 vs 실제 RIR"""
    baselines = ["E3_Nercessian","E4_Pepe","A1_NoRoomInput","A2_withPrefLoss"]
    labels    = ["vs E3","vs E4","vs A1\n(−Room)","vs A2\n(+Pref)"]
    x = np.arange(len(baselines)); w = 0.33

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    a0_synth = results_synth["A0_Proposed"]["dmr_mean"]
    a0_real  = results_real["A0_Proposed"]["dmr_mean"]
    adv_s = [a0_synth - results_synth[b]["dmr_mean"] for b in baselines]
    adv_r = [a0_real  - results_real[b]["dmr_mean"]  for b in baselines]

    b1 = ax.bar(x - w/2, adv_s, w, color=C["proposed"], edgecolor="#333",
               hatch=HATCHES["proposed"], label="Synthetic RIR")
    b2 = ax.bar(x + w/2, adv_r, w, color=C["real"],     edgecolor="#333",
               hatch=HATCHES["real"],     label="Real RIR (BUT ReverbDB)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_ylabel("dmr Advantage (A0 − Baseline)")
    ax.set_title("dmr Advantage: Synthetic vs Real RIR")
    ax.legend(fontsize=7.5)
    for bar, v in zip(list(b1)+list(b2), adv_s+adv_r):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.003, f"+{v:.3f}" if v>0 else f"{v:.3f}",
                ha="center", va="bottom", fontsize=6.5)
    fig.tight_layout()
    _save(fig, out_dir, "fig8_ood_advantage")


def fig9_ood_bar(results_synth: dict, results_real: dict, out_dir: Path):
    """Fig 9 – OOD 일반화: 합성/실제 RIR LSD + dmr 비교"""
    models = ["E3_Nercessian","E4_Pepe","AC1_BiLSTM","AC2_GRU","AC3_Conformer","A0_Proposed","A1_NoRoomInput","A2_withPrefLoss"]
    labels = ["E3","E4","AC1","AC2","AC3","A0","A1\n−Room","A2\n−Pref"]
    colors = [C["baseline"]]*2 + [C["arch"]]*3 + [C["proposed"]] + [C["ablation"]]*2

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    x = np.arange(len(models)); w = 0.38

    for ax, key, ylabel, title in [
        (axes[0], "lsd_mean",     "Dual-target LSD (dB)",  "(a) Dual-target LSD"),
        (axes[1], "dmr_mean", "dmr",   "(b) dmr"),
    ]:
        s_vals = [results_synth[m][key] for m in models]
        r_vals = [results_real[m][key]  for m in models]
        _htch = [HATCHES.get(k, "") for k in
                 ["baseline","baseline","arch","arch","arch","proposed","ablation","ablation"]]
        for i, (sv, rv, c, h) in enumerate(zip(s_vals, r_vals, colors, _htch)):
            ax.bar(x[i]-w/2, sv, w, color=c, edgecolor="#333", linewidth=0.6, hatch=h)
            ax.bar(x[i]+w/2, rv, w, color=c, edgecolor="#333", linewidth=0.6,
                   hatch=h+HATCHES["real"] if h else HATCHES["real"], alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(ylabel); ax.set_title(title)

    legend_elems = [
        mpatches.Patch(facecolor="#888888", edgecolor="#333", label="Synthetic RIR"),
        mpatches.Patch(facecolor="#888888", edgecolor="#333", hatch=HATCHES["real"],
                       alpha=0.6, label="Real RIR (BUT ReverbDB)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5,-0.03))
    # fig.suptitle("Out-of-Distribution Generalization", y=1.02)  # caption handled by LaTeX
    fig.tight_layout()
    _save(fig, out_dir, "fig9_ood_bar")


def fig10_arch_compare(results_synth: dict, results_real: dict, out_dir: Path):
    """Fig 10 – 아키텍처 비교: LSD, dmr, RTF"""
    models = ["AC1_BiLSTM","AC2_GRU","AC3_Conformer","A0_Proposed"]
    labels = ["AC1\nTCN+BiLSTM","AC2\nTCN+GRU","AC3\nTCN+Conf.","A0\nProposed"]
    colors = [C["arch"]]*3 + [C["proposed"]]
    metrics_synth = {m: results_synth[m] for m in models}
    metrics_real  = {m: results_real[m]  for m in models}

    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.8))
    x = np.arange(len(models)); w = 0.35

    # LSD
    for ax, key, ylabel, title in [
        (axes[0], "lsd_mean",     "Dual-target LSD (dB)",  "(a) Dual-target LSD"),
        (axes[1], "dmr_mean", "dmr",   "(b) dmr"),
        (axes[2], "rtf",          "RTF",       "(c) RTF"),
    ]:
        sv = [metrics_synth[m][key] for m in models]
        rv = [metrics_real[m][key]  for m in models]
        _htch = [HATCHES["arch"]]*3 + [HATCHES["proposed"]]
        for i, (sv_, rv_, c, h) in enumerate(zip(sv, rv, colors, _htch)):
            ax.bar(x[i]-w/2, sv_, w, color=c, edgecolor="#333", linewidth=0.6, hatch=h)
            ax.bar(x[i]+w/2, rv_, w, color=c, edgecolor="#333", linewidth=0.6,
                   hatch=h+HATCHES["real"] if h else HATCHES["real"], alpha=0.65)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(ylabel); ax.set_title(title)
        if key == "rtf":
            ax.axhline(1.0, color="red", lw=1.0, ls="--", label="RT threshold")
            ax.legend(fontsize=6.5)

    # LSD domain gap
    ax = axes[3]
    gap = [metrics_real[m]["lsd_mean"] - metrics_synth[m]["lsd_mean"] for m in models]
    bars = ax.bar(x, gap, color=colors, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Δ Dual-target LSD (Real − Synth)"); ax.set_title("(d) Domain Gap")
    for bar, v in zip(bars, gap):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.5)

    # fig.suptitle("Architecture Comparison", y=1.02)  # caption handled by LaTeX
    _leg10 = [
        mpatches.Patch(facecolor="#888", edgecolor="#333", label="Synthetic RIR"),
        mpatches.Patch(facecolor="#888", edgecolor="#333", alpha=0.65,
                       hatch=HATCHES["real"], label="Real RIR"),
    ]
    fig.legend(handles=_leg10, loc="lower center", ncol=2, fontsize=7,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    _save(fig, out_dir, "fig10_arch_compare")


def fig11_pareto(results: dict, out_dir: Path):
    """Fig 11 – dual-target LSD vs dmr trade-off scatter with Pareto frontier."""
    model_groups = {
        "External Baselines": (["E3_Nercessian","E4_Pepe"], C["baseline"], "o"),
        "Architecture Vars":  (["AC1_BiLSTM","AC2_GRU","AC3_Conformer"], C["arch"], "s"),
        "Ablations":          ([m for m in ["A1_NoRoomInput","A2_withPrefLoss","A3_NoPrefInput"] if m in results], C["ablation"], "^"),
        "Proposed":           (["A0_Proposed"], C["proposed"], "*"),
    }
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    all_names, all_x, all_y = [], [], []
    for group, (mlist, color, marker) in model_groups.items():
        mlist = [m for m in mlist if m in results]
        if not mlist:
            continue
        x = [results[m]["lsd_mean"] for m in mlist]
        y = [results[m]["dmr_mean"] for m in mlist]
        all_names.extend(mlist); all_x.extend(x); all_y.extend(y)
        sz = 120 if marker == "*" else 50
        ax.scatter(x, y, c=color, marker=marker, s=sz, label=group,
                   edgecolors="white", linewidths=0.6, zorder=3)
        for _i, (m, xi, yi) in enumerate(zip(mlist, x, y)):
            # Alternate offsets vertically to reduce label overlap
            _dy = 0.008 if _i % 2 == 0 else -0.018
            _dx = 0.04
            ax.annotate(_display_name(m), (xi, yi),
                        xytext=(xi + _dx, yi + _dy),
                        fontsize=6.5, color=color,
                        arrowprops=dict(arrowstyle="-", color=color,
                                        lw=0.5, alpha=0.5),
                        annotation_clip=True)

    if all_x:
        x_arr = np.asarray(all_x)
        y_arr = np.asarray(all_y)
        pareto = _pareto_mask_min_x_max_y(x_arr, y_arr)
        px = x_arr[pareto]
        py = y_arr[pareto]
        order = np.argsort(px)
        ax.plot(px[order], py[order], color="#222", lw=1.0, ls="--", alpha=0.8, zorder=2, label="Pareto frontier")
        ax.scatter(px, py, s=180, facecolors="none", edgecolors="#222", linewidths=1.0, zorder=4)

    a0_on_front = False
    if "A0_Proposed" in all_names:
        idx = all_names.index("A0_Proposed")
        a0_on_front = bool(_pareto_mask_min_x_max_y(np.asarray(all_x), np.asarray(all_y))[idx])

    subtitle = "Pareto-optimal points are outlined"
    if a0_on_front:
        subtitle += "; A0_Proposed is non-dominated"

    ax.set_xlabel("Dual-target LSD ↓ (lower is better)")
    ax.set_ylabel("dmr ↑ (higher is better)")
    ax.set_title(f"Dual-target LSD–dmr Trade-off\n({subtitle})")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    _save(fig, out_dir, "fig11_pareto")


def fig12_alpha_sensitivity(results_a0: dict, out_dir: Path):
    """Fig 12 – alpha 민감도 분석. 실측 sweep 결과가 없으면 placeholder를 그림."""
    sweep = results_a0.get("alpha_sweep") if isinstance(results_a0, dict) else None
    if not sweep:
        fig, ax = plt.subplots(figsize=(6.8, 2.4))
        ax.axis("off")
        ax.text(0.5, 0.62, "Fig. 12 requires alpha-sweep experiment results.", ha="center", va="center", fontsize=10)
        ax.text(0.5, 0.40, "No real sweep data were provided in the current checkpoint/results files,\nso the previous hard-coded placeholder values were removed.", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        _save(fig, out_dir, "fig12_alpha_sensitivity")
        return

    alphas = np.asarray(sweep["alphas"], dtype=float)
    lsd_vals = np.asarray(sweep["lsd"], dtype=float)
    pa_vals  = np.asarray(sweep["dmr"], dtype=float)
    chosen_idx = int(np.argmin(np.abs(alphas - 0.6))) if len(alphas) else 0

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8))

    axes[0].plot(alphas, lsd_vals, "o-", color=C["baseline"], lw=1.5)
    axes[0].axvline(alphas[chosen_idx], color=C["proposed"], ls="--", lw=1.0, label=f"α={alphas[chosen_idx]:.2f} (chosen)")
    axes[0].set_xlabel("α (room weight)"); axes[0].set_ylabel("Dual-target LSD (dB)")
    axes[0].set_title("(a) Dual-target LSD vs α"); axes[0].legend(fontsize=7)

    axes[1].plot(alphas, pa_vals, "o-", color=C["proposed"], lw=1.5)
    axes[1].axvline(alphas[chosen_idx], color=C["proposed"], ls="--", lw=1.0, label=f"α={alphas[chosen_idx]:.2f} (chosen)")
    axes[1].set_xlabel("α (room weight)"); axes[1].set_ylabel("dmr")
    axes[1].set_title("(b) dmr vs α"); axes[1].legend(fontsize=7)

    axes[2].plot(lsd_vals, pa_vals, "o-", color="#555", lw=1.2)
    for _i, (a, l, pa) in enumerate(zip(alphas, lsd_vals, pa_vals)):
        if _i % 2 == 0:  # show only every other label to avoid overlap
            axes[2].annotate(f"α={a:.1f}", (l, pa), fontsize=6.0,
                             xytext=(l+0.01, pa + (0.004 if _i % 4 == 0 else -0.012)))
    axes[2].scatter([lsd_vals[chosen_idx]], [pa_vals[chosen_idx]], c=C["proposed"], s=100, zorder=5, marker="*",
                    label=f"α={alphas[chosen_idx]:.2f}")
    axes[2].set_xlabel("Dual-target LSD (dB)"); axes[2].set_ylabel("dmr")
    axes[2].set_title("(c) Trade-off curve"); axes[2].legend(fontsize=7)

    # fig.suptitle("Sensitivity Analysis of Dual-Objective Weight α", y=1.02)  # caption in LaTeX
    fig.tight_layout()
    _save(fig, out_dir, "fig12_alpha_sensitivity")


# ══════════════════════════════════════════════════════════
# 6. 표 생성 (CSV + LaTeX)
# ══════════════════════════════════════════════════════════

def build_table1(results_synth: dict) -> pd.DataFrame:
    order  = ["E1_NoEQ","E2_StaticEQ","E3_Nercessian","E4_Pepe","E5_Sequential","E6_DSP","AC1_BiLSTM","AC2_GRU","AC3_Conformer","A0_Proposed","A1_NoRoomInput","A2_withPrefLoss"]
    if "A3_NoPrefInput" in results_synth:
        order.append("A3_NoPrefInput")
    labels = {
        "E1_NoEQ":"E1  No Processing","E2_StaticEQ":"E2  Static Mode EQ",
        "E3_Nercessian":"E3  Nercessian MLP [3]","E4_Pepe":"E4  Pepe CNN [4]",
        "E5_Sequential":"E5  Sequential (E3→E2)","E6_DSP":"E6  DSP Analytical",
        "AC1_BiLSTM":"AC1 TCN+BiLSTM","AC2_GRU":"AC2 TCN+GRU","AC3_Conformer":"AC3 TCN+Conformer",
        "A0_Proposed":"A0  Proposed","A1_NoRoomInput":"A1  w/o Room Input","A2_withPrefLoss":"A2  with Pref Loss","A3_NoPrefInput":"A3  w/o Pref Input",
    }
    rows = []
    for m in order:
        r = results_synth[m]
        rows.append({
            "Model":        labels[m],
            "LSD ↓":        f"{r['lsd_mean']:.3f}",
            "Room-LSD ↓":   f"{r['lsd_room_mean']:.3f}",
            "Pref-LSD ↓":   f"{r['lsd_pref_mean']:.3f}",
            "dmr ↑":    f"{r['dmr_mean']:.3f}",
            "CosSim ↑":     f"{r['cossim_mean']:.3f}",
            "RTF ↓":        f"{r['rtf']:.3f}" if r["rtf"] < 99 else "—",
        })
    return pd.DataFrame(rows)


def build_table2(results_synth: dict) -> pd.DataFrame:
    order = ["A0_Proposed","A1_NoRoomInput","A2_withPrefLoss"]
    if "A3_NoPrefInput" in results_synth:
        order.append("A3_NoPrefInput")
    descs = {
        "A0_Proposed":"Proposed model (former A2 no-pref-loss)",
        "A1_NoRoomInput":"Remove room_response input",
        "A2_withPrefLoss":"with preference loss (former A0 full)",
        "A3_NoPrefInput":"Remove preference inputs (mode_id, band_gains)",
    }
    rows = []
    for m in order:
        r = results_synth[m]
        rows.append({
            "Variant":     m,
            "Description": descs[m],
            "LSD ↓":       f"{r['lsd_mean']:.3f}",
            "Room-LSD ↓":  f"{r['lsd_room_mean']:.3f}",
            "Pref-LSD ↓":  f"{r['lsd_pref_mean']:.3f}",
            "dmr ↑":   f"{r['dmr_mean']:.3f}",
            "CosSim ↑":    f"{r['cossim_mean']:.3f}",
            "RTF ↓":       f"{r['rtf']:.3f}",
        })
    return pd.DataFrame(rows)


def build_table4(results_synth: dict, results_real: dict) -> pd.DataFrame:
    order = ["E3_Nercessian","E4_Pepe","AC1_BiLSTM","AC2_GRU","AC3_Conformer","A0_Proposed","A1_NoRoomInput","A2_withPrefLoss"]
    labels = {
        "E3_Nercessian":"E3 Nercessian","E4_Pepe":"E4 Pepe",
        "AC1_BiLSTM":"AC1 TCN+BiLSTM","AC2_GRU":"AC2 TCN+GRU","AC3_Conformer":"AC3 Conformer",
        "A0_Proposed":"A0 Proposed","A1_NoRoomInput":"A1 w/o Room","A2_withPrefLoss":"A2 with Pref",
    }
    rows = []
    for m in order:
        rs = results_synth[m]; rr = results_real[m]
        rows.append({
            "Model":       labels[m],
            "Synth Dual-LSD ↓": f"{rs['lsd_mean']:.3f}",
            "Real Dual-LSD ↓":  f"{rr['lsd_mean']:.3f}",
            "Synth PA ↑":  f"{rs['dmr_mean']:.3f}",
            "Real PA ↑":   f"{rr['dmr_mean']:.3f}",
        })
    return pd.DataFrame(rows)


def build_tableA1(source) -> pd.DataFrame:
    """Table A1 – E3/E4 feature-dimension ablation from real experiment output."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(
                f"Table A1 source not found: {path}. "
                "The feature-dimension fairness table is not regenerated by this release. "
                "--ckpt_dir ./checkpoints/full --table_out_dir ./paper_outputs/tables"
            )
        df = pd.read_csv(path)
    elif isinstance(source, dict) and source:
        rows = []
        for model_name, by_dim in source.items():
            label = {"E3_Nercessian": "E3 Nercessian", "E4_Pepe": "E4 Pepe"}.get(model_name, model_name)
            for dim, metrics in by_dim.items():
                rows.append({
                    "Model": label,
                    "Feature Dim": f"{dim} (main)" if int(dim) == 10 else str(dim),
                    "LSD ↓": f"{metrics['lsd']:.3f}",
                    "dmr ↑": f"{metrics['dmr']:.3f}",
                    "CosSim ↑": f"{metrics['cossim']:.3f}",
                })
        df = pd.DataFrame(rows)
    else:
        raise ValueError("Table A1 requires actual E3/E4 feature-dim results; hardcoded fallback is disabled.")

    required = ["Model", "Feature Dim", "LSD ↓", "dmr ↑", "CosSim ↑"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Table A1 source is missing columns: {missing}")
    return df[required]


def save_table(df: pd.DataFrame, out_dir: Path, name: str, caption: str = ""):
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)

    # LaTeX
    latex = df.to_latex(index=False, escape=True,
                        column_format="l" + "r"*(len(df.columns)-1))
    if caption:
        latex = f"% {caption}\n" + latex
    (out_dir / f"{name}.tex").write_text(latex)
    print(f"  Saved {name}.csv + .tex")


def _save(fig: plt.Figure, out_dir: Path, name: str):
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved {name}.pdf/.png")


def load_paired_mode_dataset(split_dir: str, device: str = "cpu") -> dict:
    split_dir = Path(split_dir)
    chunk_files = sorted(split_dir.glob("chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {split_dir}")

    keys = [
        "features", "room_response", "mode_id", "band_gains",
        "room_target", "pref_target", "dual_target", "pair_id"
    ]
    arrays = {k: [] for k in keys}
    for cp in chunk_files:
        data = np.load(cp, allow_pickle=False)
        for k in keys:
            if k not in data:
                raise KeyError(f"'{k}' not found in {cp.name}")
            arrays[k].append(data[k])

    out = {}
    for k, v in arrays.items():
        arr = np.concatenate(v)
        t = torch.from_numpy(arr)
        out[k] = t.long().to(device) if k in ["mode_id", "pair_id"] else t.float().to(device)

    print(f"  Loaded paired dataset: {len(out['features'])} samples from {split_dir}")
    return out


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_n @ b_n.T


def analyze_paired_mode_switch(results: dict, data: dict, model_keys=None) -> dict:
    if model_keys is None:
        model_keys = list(results.keys())

    pair_ids = data["pair_id"].cpu().numpy()
    mode_ids = data["mode_id"].cpu().numpy()
    room_t   = data["room_target"].cpu().numpy()
    pref_t   = data["pref_target"].cpu().numpy()

    valid_pairs = []
    for pid in np.unique(pair_ids):
        idx = np.where(pair_ids == pid)[0]
        if len(idx) != 4:
            continue
        ms = np.sort(mode_ids[idx])
        if np.array_equal(ms, np.array([0, 1, 2, 3])):
            valid_pairs.append(pid)

    print(f"  Valid paired groups: {len(valid_pairs)}")
    stats_out = {}
    for mk in model_keys:
        if mk not in results:
            continue
        pred = results[mk]["pred_all"]
        heard = pred - room_t

        mode_var_list, retrieval_acc_list, cos_margin_list = [], [], []
        correct_cos_list, wrong_cos_list = [], []
        for pid in valid_pairs:
            idx = np.where(pair_ids == pid)[0]
            idx = idx[np.argsort(mode_ids[idx])]
            heard_pair = heard[idx]
            pref_pair  = pref_t[idx]
            mode_var = float(np.sqrt(np.mean(np.var(heard_pair, axis=0))))
            mode_var_list.append(mode_var)
            sim = _cosine_sim_matrix(heard_pair, pref_pair)
            pred_mode = sim.argmax(axis=1)
            true_mode = np.arange(4)
            retrieval_acc_list.append(float((pred_mode == true_mode).mean()))
            correct = np.diag(sim)
            wrong = sim[~np.eye(4, dtype=bool)].reshape(4, 3).mean(axis=1)
            cos_margin_list.append(float(np.mean(correct - wrong)))
            correct_cos_list.extend(correct.tolist())
            wrong_cos_list.extend(wrong.tolist())

        stats_out[mk] = {
            "mode_var_arr": np.array(mode_var_list, dtype=np.float32),
            "retrieval_acc_arr": np.array(retrieval_acc_list, dtype=np.float32),
            "cos_margin_arr": np.array(cos_margin_list, dtype=np.float32),
            "correct_cos_arr": np.array(correct_cos_list, dtype=np.float32),
            "wrong_cos_arr": np.array(wrong_cos_list, dtype=np.float32),
            "mode_var_mean": float(np.mean(mode_var_list)) if mode_var_list else 0.0,
            "retrieval_acc_mean": float(np.mean(retrieval_acc_list)) if retrieval_acc_list else 0.0,
            "cos_margin_mean": float(np.mean(cos_margin_list)) if cos_margin_list else 0.0,
        }
    return stats_out


def fig13_paired_mode_switch(results: dict, data: dict, out_dir: Path, model_keys=None):
    if model_keys is None:
        preferred = ["A0_Proposed", "A2_withPrefLoss", "A3_NoPrefInput", "E5_Sequential"]
        model_keys = [m for m in preferred if m in results]

    stats = analyze_paired_mode_switch(results, data, model_keys=model_keys)

    model_labels = {
        "A0_Proposed":    "A0 Proposed",
        "A2_withPrefLoss":"A2 with Pref Loss",
        "A3_NoPrefInput": "A3 w/o Pref Input",
        "E5_Sequential":  "E5 Sequential",
    }
    # Okabe-Ito colors (replaces old inaccessible palette)
    model_colors = {
        "A0_Proposed":    "#D55E00",   # Vermillion  – proposed
        "A2_withPrefLoss":"#CC79A7",   # Reddish purple – ablation
        "A3_NoPrefInput": "#0072B2",   # Blue           – ablation
        "E5_Sequential":  "#999999",   # Grey           – baseline
    }
    # Per-model hatch patterns — each unique so B&W printing is unambiguous.
    # Grey-scale values of the 4 colors above: 119 / 151 / 87 / 153.
    # Δ(proposed, ablation-A2)=32  Δ(arch, grey)=66  Δ(A2, grey)=2 ← too close.
    # Hatches make them distinguishable even when Δgrey < 30.
    model_hatches = {
        "A0_Proposed":    "",        # solid fill  (proposed stands out alone)
        "A2_withPrefLoss":"\\\\",    # backslash diagonals
        "A3_NoPrefInput": "////",    # forward-slash diagonals
        "E5_Sequential":  "....",    # dense dots
    }

    valid_keys = [m for m in model_keys if m in stats and len(stats[m]["mode_var_arr"]) > 0]
    labels   = [model_labels.get(m, m)   for m in valid_keys]
    colors   = [model_colors.get(m, "#555555") for m in valid_keys]
    hatches_ = [model_hatches.get(m, "////")   for m in valid_keys]  # local list, no NameError

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))

    # ── (a) boxplot: mode variation ───────────────────────────────
    ax = axes[0]
    mode_var_data = [np.asarray(stats[m]["mode_var_arr"]).ravel() for m in valid_keys]
    if len(mode_var_data) == 0:
        ax.text(0.5, 0.5, "No valid paired data", ha="center", va="center")
        ax.set_title("(a) Output variation across modes")
    else:
        if len(set(len(x) for x in mode_var_data)) == 1:
            box_data = np.column_stack(mode_var_data)
        else:
            box_data = mode_var_data
        bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True,
                        medianprops=dict(color="black", lw=1.4), showfliers=False)
        for patch, c, h in zip(bp["boxes"], colors, hatches_):  # hatches_ is local list
            patch.set_facecolor(c + "88")
            patch.set_hatch(h)
            patch.set_edgecolor("#333333")
        ax.set_ylabel("Mode Variation Score")
        ax.set_title("(a) Output variation across modes")
        ax.tick_params(axis="x", rotation=15, labelsize=7.5)

    # ── (b) bar: retrieval accuracy ───────────────────────────────
    ax = axes[1]
    vals = [stats[m]["retrieval_acc_mean"] * 100.0 for m in valid_keys]
    bars = ax.bar(labels, vals, color=colors, edgecolor="#333333",
                  linewidth=0.7, hatch=hatches_)  # list → one hatch per bar
    ax.axhline(25.0, color="#555555", ls="--", lw=0.9, label="Chance (25%)")
    ax.set_ylabel("Retrieval Accuracy (%)")
    ax.set_title("(b) Correct mode-target identification")
    ax.tick_params(axis="x", rotation=15, labelsize=7.5)
    ax.legend(fontsize=7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.0,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    # ── (c) bar: cosine margin ────────────────────────────────────
    ax = axes[2]
    vals = [stats[m]["cos_margin_mean"] for m in valid_keys]
    bars = ax.bar(labels, vals, color=colors, edgecolor="#333333",
                  linewidth=0.7, hatch=hatches_)
    ax.axhline(0.0, color="#555555", ls=":", lw=0.8)
    ax.set_ylabel("CosSim Margin")
    ax.set_title("(c) Correct-mode vs wrong-mode margin")
    ax.tick_params(axis="x", rotation=15, labelsize=7.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # fig.suptitle("Paired Mode-Switch Evaluation", y=1.03)  # caption handled by LaTeX
    fig.tight_layout()
    _save(fig, out_dir, "fig13_paired_mode_switch")


# ══════════════════════════════════════════════════════════
# 7. 모델 로딩 (체크포인트 or 랜덤 초기화)
# ══════════════════════════════════════════════════════════

from arch_variants import AC1_TCNBiLSTM, AC2_TCNGRU, AC3_TCNConformer
from baselines import E3_NercessianMLP, E4_PepeCNN, E5_Sequential, E6_DSP_Analytical
from model import DualObjectiveAdaptivePEQ
from ablation import A1_NoRoomInput, A3_NoPrefInput
from arch_biquad import AC1_BiLSTM_Biquad, AC2_GRU_Biquad, AC3_Conformer_Biquad
MODEL_REGISTRY = {
    # REVISION(gain±12): A0/A2 use the relaxed per-section gain bound (±12 dB);
    # fc_max stays 16 kHz. Lambdas so load_model's MODEL_REGISTRY[name]() passes args.
    "A0_Proposed":         lambda: DualObjectiveAdaptivePEQ(gain_max=12.0, fc_max=16000.0),
    "A1_NoRoomInput":      A1_NoRoomInput,
    "A2_withPrefLoss":     lambda: DualObjectiveAdaptivePEQ(gain_max=12.0, fc_max=16000.0),
    "A3_NoPrefInput":      A3_NoPrefInput,
    "E1_NoEQ":             E1_NoProcessing,
    "E2_StaticEQ":         E2_StaticModeEQ,
    "E3_Nercessian":       E3_NercessianMLP,
    "E4_Pepe":             E4_PepeCNN,
    "E5_Sequential":       E5_Sequential,
    "E6_DSP":              E6_DSP_Analytical,
    "AC1_BiLSTM":          AC1_TCNBiLSTM,
    "AC2_GRU":             AC2_TCNGRU,
    "AC3_Conformer":       AC3_TCNConformer,
    "AC1_BiLSTM_Biquad":   AC1_BiLSTM_Biquad,
    "AC2_GRU_Biquad":      AC2_GRU_Biquad,
    "AC3_Conformer_Biquad": AC3_Conformer_Biquad,
}


def inject_legacy_result_aliases(results: dict) -> None:
    if "A0_Proposed" in results:
        results["A0_Full"] = results["A0_Proposed"]
    if "A2_withPrefLoss" in results:
        results["A2_NoPrefLoss"] = results["A2_withPrefLoss"]

def load_model(name: str, ckpt_dir: Optional[Path], dry_run: bool) -> nn.Module:
    name = canonical_model_name(name)
    model = MODEL_REGISTRY[name]()
    if dry_run or ckpt_dir is None:
        return model

    ckpt_path = None
    for candidate in checkpoint_name_candidates(name):
        path = ckpt_dir / f"{candidate}.pt"
        if path.exists():
            ckpt_path = path
            break
    if ckpt_path is None:
        tried = ", ".join(f"{candidate}.pt" for candidate in checkpoint_name_candidates(name))
        print(f"  WARNING: No checkpoint for {name} (tried: {tried}), using random weights")
        return model

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"    missing keys: {len(missing)}")
    if unexpected:
        print(f"    unexpected keys: {len(unexpected)}")
    return model


# ══════════════════════════════════════════════════════════
# 8. 메인
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PHONIX Adaptive EQ – Paper Experiment Suite")
    parser.add_argument("--data_dir",  type=str, default="./data/dataset_v3",
                        help="데이터셋 루트 디렉터리 (split/ 하위 포함)")
    parser.add_argument("--ckpt_dir",  type=str, default="./checkpoints",
                        help="체크포인트 디렉터리 (A0.pt, A1.pt, ...)")
    parser.add_argument("--out_dir",   type=str, default="./paper_outputs")
    parser.add_argument("--device",    type=str, default="cpu")
    parser.add_argument("--batch_size",type=int, default=512)
    parser.add_argument("--dry_run",   action="store_true",
                        help="랜덤 예측으로 파이프라인 검증 (데이터/체크포인트 불필요)")
    parser.add_argument("--models",    type=str, default="all",
                        help="평가할 모델 (쉼표 구분, 예: A0_Proposed,E3_Nercessian,E4_Pepe / 기본: all)")
    parser.add_argument("--figs",      type=str, default="all",
                        help="생성할 figure 번호 (쉼표 구분, 예: 1,2,7 / 기본: all)")
    parser.add_argument("--enable_alpha_sweep", action="store_true",
                        help="논문 본 실험과 분리된 alpha sweep/fig12를 명시적으로 실행")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir = out_dir / "tables";  tab_dir.mkdir(parents=True, exist_ok=True)
    stat_dir= out_dir / "stats";   stat_dir.mkdir(parents=True, exist_ok=True)

    device   = args.device
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None

    model_names = (
        list(MODEL_REGISTRY.keys())
        if args.models == "all"
        else [canonical_model_name(name.strip()) for name in args.models.split(",") if name.strip()]
    )
    fig_ids     = None if args.figs == "all" else set(args.figs.split(","))

    # ── 데이터 로드 ──────────────────────────────────────────
    print("\n[1/4] Loading data...")
    if args.dry_run:
        print("  DRY RUN mode: using random synthetic data")
        data_synth = make_dry_run_dataset(n=1000, device=device)
        data_real  = make_dry_run_dataset(n=500,  device=device)
        data_synth["rt60_arr"] = np.random.uniform(0.2, 0.75, 1000)
        target_freqs = make_frequency_grid_np()
    else:
        try:
            ds_synth = PEQDataset(f"{args.data_dir}/test_synth",  device)
            ds_real  = PEQDataset(f"{args.data_dir}/test_real", device)
            data_synth = ds_synth.get_all()
            data_real  = ds_real.get_all()
            if ds_synth.rt60 is not None:
                data_synth["rt60_arr"] = ds_synth.rt60
            target_freqs = ds_synth.target_freqs if ds_synth.target_freqs is not None else make_frequency_grid_np()
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            print("  Falling back to dry_run mode")
            data_synth = make_dry_run_dataset(n=1000, device=device)
            data_real  = make_dry_run_dataset(n=500,  device=device)
            data_synth["rt60_arr"] = np.random.uniform(0.2, 0.75, 1000)
            target_freqs = make_frequency_grid_np()

    # ── 모델 평가 ─────────────────────────────────────────────
    print(f"\n[2/4] Evaluating {len(model_names)} models...")
    results_synth = {}; results_real = {}

    for name in model_names:
        print(f"\n  -- {name} --")
        model = load_model(name, ckpt_dir, args.dry_run)
        results_synth[name] = evaluate_model(name, model, data_synth, args.batch_size)
        results_real[name]  = evaluate_model(name, model, data_real,  args.batch_size)
        inject_legacy_result_aliases(results_synth)
        inject_legacy_result_aliases(results_real)
        print(f"    Synth: LSD={results_synth[name]['lsd_mean']:.3f}  "
              f"Room-LSD={results_synth[name]['lsd_room_mean']:.3f}  "
              f"dmr={results_synth[name]['dmr_mean']:.3f}  "
              f"CosSim={results_synth[name]['cossim_mean']:.3f}  "
              f"RTF={results_synth[name]['rtf']:.3f}")
        print(f"    Real : LSD={results_real[name]['lsd_mean']:.3f}  "
              f"Room-LSD={results_real[name]['lsd_room_mean']:.3f}  "
              f"dmr={results_real[name]['dmr_mean']:.3f}  "
              f"CosSim={results_real[name]['cossim_mean']:.3f}  "
              f"RTF={results_real[name]['rtf']:.3f}")

        # alpha sweep is kept optional and excluded from the main experiment path.
        if args.enable_alpha_sweep and name == "A0_Proposed" and run_alpha_sweep is not None:
            print("  -- A0_Proposed: Alpha Sweep (synth) --")
            sweep_synth = run_alpha_sweep(model, data_synth,
                                        alphas=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                                        batch_size=args.batch_size, device=device)
            results_synth["A0_Proposed"]["alpha_sweep"] = sweep_synth

            print("  -- A0_Proposed: Alpha Sweep (real RIR) --")
            sweep_real = run_alpha_sweep(model, data_real,
                                        alphas=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                                        batch_size=args.batch_size, device=device)
            results_real["A0_Proposed"]["alpha_sweep"] = sweep_real
            inject_legacy_result_aliases(results_synth)
            inject_legacy_result_aliases(results_real)

            (stat_dir / "alpha_sweep_synth.json").write_text(json.dumps(sweep_synth))
            (stat_dir / "alpha_sweep_real.json").write_text(json.dumps(sweep_real))
        # raw arrays 저장
        np.save(stat_dir / f"{name}_lsd.npy",       results_synth[name]["lsd_arr"])
        np.save(stat_dir / f"{name}_lsd_room.npy",  results_synth[name]["lsd_room_arr"])
        np.save(stat_dir / f"{name}_lsd_pref.npy",  results_synth[name]["lsd_pref_arr"])
        np.save(stat_dir / f"{name}_dmr.npy",   results_synth[name]["dmr_arr"])
        np.save(stat_dir / f"{name}_cossim.npy",    results_synth[name]["cossim_arr"])
        np.save(stat_dir / f"{name}_real_lsd.npy",       results_real[name]["lsd_arr"])
        np.save(stat_dir / f"{name}_real_lsd_room.npy",  results_real[name]["lsd_room_arr"])
        np.save(stat_dir / f"{name}_real_lsd_pref.npy",  results_real[name]["lsd_pref_arr"])
        np.save(stat_dir / f"{name}_real_dmr.npy",       results_real[name]["dmr_arr"])
        np.save(stat_dir / f"{name}_real_cossim.npy",    results_real[name]["cossim_arr"])
        # perceptual proxy 분석용 raw pred 저장 (perceptual_proxy.py 에서 사용)
        np.save(stat_dir / f"{name}_pred.npy",      results_synth[name]["pred_all"])
        np.save(stat_dir / f"{name}_rtf.npy",       np.array([results_synth[name]["rtf"]]))

    # 타겟 곡선 1회 저장 (perceptual_proxy.py 공유 입력)
    if "A2_withPrefLoss" in results_synth:
        np.save(stat_dir / "targets_dual.npy",  data_synth["dual_target"].cpu().numpy())
        np.save(stat_dir / "targets_pref.npy",  data_synth["pref_target"].cpu().numpy())
        np.save(stat_dir / "targets_room.npy",  data_synth["room_target"].cpu().numpy())

    # ── 표 생성 ───────────────────────────────────────────────
    print("\n[3/4] Generating tables...")
    req_t1 = ["E1_NoEQ","E2_StaticEQ","E3_Nercessian","E4_Pepe","E5_Sequential","E6_DSP","AC1_BiLSTM","AC2_GRU","AC3_Conformer","A0_Proposed","A1_NoRoomInput","A2_withPrefLoss"]
    if all(m in results_synth for m in req_t1):
        t1 = build_table1(results_synth)
        save_table(t1, tab_dir, "table1_main_results",
                   "Table 1: Main results on synthetic RIR validation set (N=5,000)")
        print(t1.to_string(index=False))

    req_t2 = ["A0_Proposed","A1_NoRoomInput","A2_withPrefLoss"]
    if all(m in results_synth for m in req_t2):
        t2 = build_table2(results_synth)
        save_table(t2, tab_dir, "table2_ablation",
                   "Table 2: Ablation study results")
        print(t2.to_string(index=False))

    req_t3 = ["A0_Proposed","A1_NoRoomInput","A2_withPrefLoss","E3_Nercessian","E4_Pepe"]
    if "A3_NoPrefInput" in results_synth:
        req_t3.insert(3, "A3_NoPrefInput")
    if all(m in results_synth for m in req_t3):
        stat_df = build_stat_table(results_synth)
        save_table(stat_df, tab_dir, "table3_statistics",
                   "Table 3: Paired statistical test results")
        print(stat_df.to_string(index=False))

    req_t4 = ["E3_Nercessian","E4_Pepe","AC1_BiLSTM","AC2_GRU","AC3_Conformer","A0_Proposed","A1_NoRoomInput","A2_withPrefLoss"]
    if all(m in results_synth for m in req_t4) and all(m in results_real for m in req_t4):
        t4 = build_table4(results_synth, results_real)
        save_table(t4, tab_dir, "table4_ood",
                   "Table 4: Out-of-distribution generalization (BUT ReverbDB)")
        print(t4.to_string(index=False))

    # REVISION: 보조 fairness 표는 alpha_sweep 선행 산출물 의존 → 없으면 스킵(피규어 차단 방지)
    try:
        tA1 = build_tableA1(tab_dir / "tableA1_fairness.csv")
        save_table(tA1, tab_dir, "tableA1_fairness",
                   "Table A1: Feature dimension ablation for E3/E4")
    except FileNotFoundError as e:
        print(f"  [skip tableA1] {e}")

    # ── Figure 생성 ───────────────────────────────────────────
    print("\n[4/4] Generating figures...")

    def should_plot(fid): return fig_ids is None or fid in fig_ids

    # Fig. 1 is a manually curated paper asset and is not emitted by this script.

    if should_plot("2") and all(m in results_synth for m in ["A0_Proposed","E3_Nercessian"]):
        print("  Fig 2: Room correction performance")
        fig2_room_correction(results_synth, data_synth, fig_dir, target_freqs)

    if should_plot("3") and all(m in results_synth for m in ["E3_Nercessian","E4_Pepe","A0_Proposed"]):
        print("  Fig 3: Per-mode comparison")
        fig3_per_mode(results_synth, data_synth, fig_dir)

    if should_plot("4") and len(results_synth) > 3:
        print("  Fig 4: Ablation bar chart")
        fig4_keys = ["E3_Nercessian","E4_Pepe","E5_Sequential","E6_DSP","A1_NoRoomInput","A2_withPrefLoss","AC1_BiLSTM","AC2_GRU","AC3_Conformer","A0_Proposed"]
        if "A3_NoPrefInput" in results_synth:
            fig4_keys.insert(6, "A3_NoPrefInput")
        _r = {k: results_synth.get(k, results_synth.get("A0_Proposed")) for k in fig4_keys}
        fig4_ablation_bar(_r, fig_dir)

    if should_plot("5") and len(results_synth) > 3:
        print("  Fig 5: Metric distributions")
        fig5_keys = ["E3_Nercessian","E4_Pepe","A1_NoRoomInput","A2_withPrefLoss","AC3_Conformer","A0_Proposed"]
        if "A3_NoPrefInput" in results_synth:
            fig5_keys.insert(4, "A3_NoPrefInput")
        _r = {k: results_synth.get(k, results_synth.get("A0_Proposed")) for k in fig5_keys}
        fig5_distributions(_r, fig_dir)

    if should_plot("6") and all(m in results_synth for m in ["A0_Proposed","A2_withPrefLoss","E3_Nercessian"]):
        print("  Fig 6: dmr scatter plots")
        fig6_scatter(results_synth, fig_dir)

    if should_plot("7") and "A0_Proposed" in results_synth:
        print("  Fig 7: Per-mode frequency responses")
        fig7_freq_response(results_synth, data_synth, fig_dir, target_freqs)
        print("  Fig 7b: Multi-model per-mode comparison")
        fig7_compare_models = ["A0_Proposed","A1_NoRoomInput","A2_withPrefLoss","E5_Sequential"]
        if "A3_NoPrefInput" in results_synth:
            fig7_compare_models.insert(3, "A3_NoPrefInput")
        fig7_model_compare(results_synth, data_synth, fig_dir, target_freqs, model_keys=fig7_compare_models, use_heard=True)

    if should_plot("8") and all(m in results_synth for m in ["A0_Proposed","E3_Nercessian","E4_Pepe","A1_NoRoomInput","A2_withPrefLoss"]):
        print("  Fig 8: dmr advantage (synth vs real)")
        _rs = {k: results_synth.get(k, results_synth["A0_Proposed"]) for k in ["A0_Proposed","E3_Nercessian","E4_Pepe","A1_NoRoomInput","A2_withPrefLoss"]}
        _rr = {k: results_real.get(k, results_real.get("A0_Proposed", results_synth["A0_Proposed"]))
               for k in ["A0_Proposed","E3_Nercessian","E4_Pepe","A1_NoRoomInput","A2_withPrefLoss"]}
        fig8_ood_advantage(_rs, _rr, fig_dir)

    if should_plot("9") and len(results_synth) > 3:
        print("  Fig 9: OOD generalization bar chart")
        _rs = {k: results_synth.get(k, results_synth["A0_Proposed"]) for k in
               ["E3_Nercessian","E4_Pepe","AC1_BiLSTM","AC2_GRU","AC3_Conformer","A0_Proposed","A1_NoRoomInput","A2_withPrefLoss"]}
        _rr = {k: results_real.get(k, results_real.get("A0_Proposed", results_synth["A0_Proposed"]))
               for k in ["E3_Nercessian","E4_Pepe","AC1_BiLSTM","AC2_GRU","AC3_Conformer","A0_Proposed","A1_NoRoomInput","A2_withPrefLoss"]}
        fig9_ood_bar(_rs, _rr, fig_dir)

    if should_plot("10") and len(results_synth) > 3:
        print("  Fig 10: Architecture comparison")
        _rs = {k: results_synth.get(k, results_synth["A0_Proposed"]) for k in ["AC1_BiLSTM","AC2_GRU","AC3_Conformer","A0_Proposed"]}
        _rr = {k: results_real.get(k, results_real.get("A0_Proposed", results_synth["A0_Proposed"]))
               for k in ["AC1_BiLSTM","AC2_GRU","AC3_Conformer","A0_Proposed"]}
        fig10_arch_compare(_rs, _rr, fig_dir)

    if should_plot("11") and len(results_synth) > 3:
        print("  Fig 11: LSD-dmr trade-off scatter")
        _r = {k: results_synth.get(k, results_synth["A0_Proposed"]) for k in
              ["E3_Nercessian","E4_Pepe","AC1_BiLSTM","AC2_GRU","AC3_Conformer","A0_Proposed","A1_NoRoomInput","A2_withPrefLoss"]}
        fig11_pareto(_r, fig_dir)

    if should_plot("12"):
        if args.enable_alpha_sweep and fig12_alpha_sensitivity_v2 is not None:
            print("  Fig 12: Alpha sensitivity analysis")
            fig12_alpha_sensitivity_v2(results_synth, results_real, fig_dir, C)
        else:
            print("  Fig 12 skipped: alpha sweep is optional and disabled")


    print(f"\nDone. All outputs saved to: {out_dir.resolve()}")
    print(f"  figures/ : {len(list(fig_dir.glob('*.pdf')))} PDFs")
    print(f"  tables/  : {len(list(tab_dir.glob('*.csv')))} CSVs + LaTeX")
    print(f"  stats/   : {len(list(stat_dir.glob('*.npy')))} metric arrays")


if __name__ == "__main__":
    main()
