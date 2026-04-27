"""
arch_variants.py — Architecture Comparison Models (AC1~AC3)

A0(Full)와 동일한 조건(same loss, same data)에서 아키텍처만 교체한 대조군.
모두 RoomResponseEncoder + UserPreferenceEncoder concat 조건화는 유지.

  AC1: TCN×4 + BiLSTM  (bidirectional LSTM for sequence aggregation)
  AC2: TCN×4 + GRU     (단방향 GRU - causal, 실시간 가능성 테스트)
  AC3: TCN×4 + Conformer×2 (Conformer block for global context)

공통 인터페이스: forward(x, room_response, mode_id, band_gains) → dict
필수 출력 key: pred_response_db
선택 출력 key: room_correction_db, pref_residual_db, pref_curve_db (없으면 loss에서 skip)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from frequency_grid import make_frequency_grid_torch
from model import (
    RoomResponseEncoder,
    UserPreferenceEncoder,
    CausalConv1d,
    TCNBlock,
    AttentionPooling,
)

BAND_FREQS = torch.tensor(
    [63, 125, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000], dtype=torch.float32
)


# ── 공통 베이스 ────────────────────────────────────────────────────────────────

class _ACBase(nn.Module):
    """
    AC 변형 공통 베이스.
    TCN×4 인코더 + RoomResponseEncoder + UserPreferenceEncoder 는 A0와 동일.
    aggregate() 메서드만 각 서브클래스에서 오버라이드.
    """
    def __init__(
        self,
        in_dim=10, n_room_bins=128, room_dim=32,
        hidden_dim=64, n_tcn_blocks=4,
        n_modes=4, n_bands=10, pref_dim=64,
        n_freqs=128, sample_rate=48000, f_min=20.0, f_max=None,
        freq_spacing="log", **kw,
    ):
        super().__init__()
        if f_max is None:
            f_max = sample_rate / 2.0
        self.room_encoder = RoomResponseEncoder(n_room_bins, room_dim)
        self.pref_encoder = UserPreferenceEncoder(n_modes, n_bands, pref_dim=pref_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(hidden_dim, dilation=2**i) for i in range(n_tcn_blocks)
        ])

        self.hidden_dim = hidden_dim
        self.room_dim   = room_dim
        self.pref_dim   = pref_dim
        self.n_freqs    = n_freqs
        self.sample_rate = sample_rate

        self.register_buffer("band_freqs",   BAND_FREQS)
        self.register_buffer(
            "target_freqs",
            make_frequency_grid_torch(
                n_freqs=n_freqs,
                f_min=f_min,
                f_max=f_max,
                spacing=freq_spacing,
            ),
        )

    def _tcn_out(self, x):
        """x: (B, T, in_dim) → (B, T, hidden_dim)"""
        h = self.input_proj(x)
        for blk in self.tcn_blocks:
            h = blk(h)
        return h

    def aggregate(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, hidden) → ctx: (B, hidden)  — 서브클래스에서 구현"""
        raise NotImplementedError

    @staticmethod
    def _interp(xp, fp, x):
        x   = torch.clamp(x, xp[0], xp[-1])
        idx = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
        t   = (x - xp[idx - 1]) / (xp[idx] - xp[idx - 1] + 1e-6)
        return fp[idx - 1] + t * (fp[idx] - fp[idx - 1])

    def _pref_curve(self, band_gains):
        B = band_gains.shape[0]
        pref_curve = torch.zeros(B, len(self.target_freqs), device=band_gains.device)
        for i in range(B):
            pref_curve[i] = self._interp(self.band_freqs, band_gains[i], self.target_freqs)
        return pref_curve

    def forward(self, x, room_response, mode_id, band_gains):
        room_vec = self.room_encoder(room_response)       # (B, room_dim)
        pref_vec = self.pref_encoder(mode_id, band_gains) # (B, pref_dim)

        h   = self._tcn_out(x)          # (B, T, hidden)
        ctx = self.aggregate(h)          # (B, hidden) — 서브클래스

        pref_curve = self._pref_curve(band_gains)  # (B, n_freqs)

        # ── Stage A: room correction prior ──────────────────────────────
        room_input = torch.cat([ctx, room_vec], dim=-1)
        room_mean  = torch.tanh(self.room_mean_head(room_input)) * self.room_mean_scale
        room_shape = self.room_shape_head(room_input) * self.room_shape_scale
        room_shape = room_shape - room_shape.mean(dim=-1, keepdim=True)
        room_corr  = room_mean + room_shape

        # ── Stage B: preference-conditioned residual ─────────────────────
        pref_input = torch.cat([ctx, room_vec, pref_vec, room_corr, pref_curve], dim=-1)
        pref_res   = self.pref_res_head(pref_input) * self.pref_res_scale

        pred = room_corr + pref_curve + pref_res

        return {
            "pred_response_db":   pred,
            "room_correction_db": room_corr,
            "pref_residual_db":   pref_res,
            "pref_curve_db":      pref_curve,
        }

    def _build_heads(self):
        """서브클래스 __init__ 마지막에 호출"""
        hd, rd, pd, nf = self.hidden_dim, self.room_dim, self.pref_dim, self.n_freqs

        self.room_mean_head = nn.Sequential(
            nn.Linear(hd + rd, hd), nn.GELU(), nn.Linear(hd, 1)
        )
        self.room_shape_head = nn.Sequential(
            nn.Linear(hd + rd, hd * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hd * 2, hd), nn.GELU(), nn.Linear(hd, nf), nn.Tanh()
        )
        pref_in = hd + rd + pd + nf + nf
        self.pref_res_head = nn.Sequential(
            nn.Linear(pref_in, hd * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hd * 2, hd), nn.GELU(), nn.Linear(hd, nf), nn.Tanh()
        )
        self.room_mean_scale = 4.0
        self.room_shape_scale = 12.0
        self.pref_res_scale   = 6.0


# ── AC1: TCN×4 + BiLSTM ───────────────────────────────────────────────────────

class AC1_TCNBiLSTM(_ACBase):
    """
    Sequence aggregation: Bidirectional LSTM.
    Non-causal (bidirectional) → RTF가 A0보다 높을 가능성 있음.
    """
    def __init__(self, hidden_dim=64, **kw):
        super().__init__(hidden_dim=hidden_dim, **kw)
        self.bilstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self._build_heads()

    def aggregate(self, h):
        # BiLSTM: (B, T, hidden) → mean pooling over T
        out, _ = self.bilstm(h)
        return out.mean(dim=1)


# ── AC2: TCN×4 + GRU ─────────────────────────────────────────────────────────

class AC2_TCNGRU(_ACBase):
    """
    Sequence aggregation: 단방향 Causal GRU.
    Causal이므로 실시간 배포 가능하지만 파라미터가 많아 RTF가 올라갈 수 있음.
    """
    def __init__(self, hidden_dim=64, **kw):
        super().__init__(hidden_dim=hidden_dim, **kw)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=2, batch_first=True, bidirectional=False
        )
        self._build_heads()

    def aggregate(self, h):
        # GRU 마지막 hidden state 사용 (causal: 미래 정보 없음)
        _, hn = self.gru(h)
        return hn[-1]  # (B, hidden)


# ── AC3: TCN×4 + Conformer×2 ─────────────────────────────────────────────────

class _ConformerBlock(nn.Module):
    """
    Conformer block (Gulati et al. 2020 기반 간소화 버전).
    FF → Self-Attn → Depthwise-Conv → FF → LayerNorm
    """
    def __init__(self, dim, n_heads=4, ffn_mult=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        ffn_dim = dim * ffn_mult

        # Feed-Forward 1
        self.ff1_norm = nn.LayerNorm(dim)
        self.ff1      = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout)
        )

        # Multi-Head Self-Attention
        self.attn_norm = nn.LayerNorm(dim)
        self.attn      = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)

        # Depthwise Convolution
        assert conv_kernel % 2 == 1, "conv_kernel must be odd"
        self.conv_norm  = nn.LayerNorm(dim)
        self.conv_pw1   = nn.Linear(dim, dim * 2)  # pointwise expand
        self.conv_dw    = nn.Conv1d(dim, dim, conv_kernel,
                                    padding=conv_kernel // 2, groups=dim)
        self.conv_bn    = nn.BatchNorm1d(dim)
        self.conv_act   = nn.SiLU()
        self.conv_pw2   = nn.Linear(dim, dim)
        self.conv_drop  = nn.Dropout(dropout)

        # Feed-Forward 2
        self.ff2_norm = nn.LayerNorm(dim)
        self.ff2      = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout)
        )

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # FF1
        x = x + 0.5 * self.ff1(self.ff1_norm(x))

        # Self-Attention
        xn = self.attn_norm(x)
        a, _ = self.attn(xn, xn, xn)
        x = x + self.attn_drop(a)

        # Depthwise Conv
        xn = self.conv_norm(x)
        xn = self.conv_pw1(xn)                         # (B, T, 2D)
        xn, gate = xn.chunk(2, dim=-1)
        xn = xn * torch.sigmoid(gate)                  # GLU
        xn = self.conv_dw(xn.transpose(1, 2))          # (B, D, T)
        xn = self.conv_act(self.conv_bn(xn))
        xn = self.conv_pw2(xn.transpose(1, 2))
        x  = x + self.conv_drop(xn)

        # FF2
        x = x + 0.5 * self.ff2(self.ff2_norm(x))

        return self.out_norm(x)


class AC3_TCNConformer(_ACBase):
    """
    Sequence aggregation: 2-layer Conformer → Attention Pooling.
    전역 문맥 포착 능력이 가장 강하지만 파라미터도 가장 많음.
    """
    def __init__(self, hidden_dim=64, n_conformer_blocks=2, **kw):
        super().__init__(hidden_dim=hidden_dim, **kw)
        self.conformer = nn.Sequential(*[
            _ConformerBlock(hidden_dim) for _ in range(n_conformer_blocks)
        ])
        self.pool = AttentionPooling(hidden_dim)
        self._build_heads()

    def aggregate(self, h):
        h = self.conformer(h)
        ctx, _ = self.pool(h)
        return ctx


# ── 파라미터 수 확인 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from model import DualObjectiveAdaptivePEQ

    B = 4
    x  = torch.randn(B, 32, 10)
    rr = torch.randn(B, 128)
    mi = torch.randint(0, 4, (B,))
    bg = torch.randn(B, 10)

    models = {
        "A0 (proposed)": DualObjectiveAdaptivePEQ(),
        "AC1 TCN+BiLSTM": AC1_TCNBiLSTM(),
        "AC2 TCN+GRU":    AC2_TCNGRU(),
        "AC3 TCN+Conformer": AC3_TCNConformer(),
    }

    print(f"{'Model':<22} {'Params':>10}  Output shape")
    print("─" * 50)
    for name, m in models.items():
        m.eval()
        with torch.no_grad():
            out = m(x, rr, mi, bg)
        p = sum(pp.numel() for pp in m.parameters())
        print(f"{name:<22} {p:>10,}  {tuple(out['pred_response_db'].shape)}")
