"""
Ablation Study v3 (Simplified)

Perceptual 평가 결과 유의미한 것만 유지:
  A0. Proposed            (former A2 no-pref-loss)
  A1. w/o Room Input      (ΔPLSD=2.39dB, 확실히 들림)
  A2. with Pref Loss      (former A0 full; negative ablation)

제거된 ablation (모두 ΔPLSD < 0.01dB):
  - FiLM vs concat → 차이 없음 → concat 채택
  - TCN vs Transformer → 차이 없음 → TCN만 채택
  - smooth/temporal/boost loss → 차이 없음 → 제거
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    RoomResponseEncoder,
    UserPreferenceEncoder,
    TCNBlock,
    AttentionPooling,
    DualObjectiveEQLoss,
)


# ──────────────────────────────────────────────
# A1. w/o Room Input
# room_response 제거 → pref만으로 동작
# 확인: room 정보 없이 dual-objective가 가능한가
# ──────────────────────────────────────────────

class A1_NoRoomInput(nn.Module):
    """Room response 입력 제거. pref_vec만으로 conditioning."""
    def __init__(self, in_dim=10, hidden_dim=64, n_tcn_blocks=4,
                 n_modes=4, n_bands=10, pref_dim=64, n_freqs=128, **kw):
        super().__init__()
        self.pref_encoder = UserPreferenceEncoder(n_modes, n_bands, pref_dim=pref_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(hidden_dim, dilation=2**i) for i in range(n_tcn_blocks)])
        self.pool = AttentionPooling(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + pref_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_freqs), nn.Tanh())
        self.output_scale = 18.0

    def forward(self, x, room_response, mode_id, band_gains):
        # room_response 무시
        pref_vec = self.pref_encoder(mode_id, band_gains)
        x = self.input_proj(x)
        for tcn in self.tcn_blocks:
            x = tcn(x)
        ctx, attn = self.pool(x)
        pred = self.output_head(torch.cat([ctx, pref_vec], dim=-1)) * self.output_scale
        return {"pred_response_db": pred, "fc": None, "gain": None, "q": None,
                "attn_weights": attn, "cond_vec": pref_vec}


# ──────────────────────────────────────────────
# A2. with Pref Loss (negative ablation)
# 기존 full dual-objective loss를 negative ablation으로 재정의
# ──────────────────────────────────────────────

class A2_NoPrefLoss(DualObjectiveEQLoss):
    """Legacy name for the promoted proposed-loss setting."""
    def __init__(self, **kw):
        super().__init__(lambda_pref_res=0.0, lambda_dir=0.0, **kw)


class A0_ProposedLoss(A2_NoPrefLoss):
    pass


class A2_withPrefLossLoss(DualObjectiveEQLoss):
    """Canonical negative-ablation loss: retains the preference terms."""
    pass

# ──────────────────────────────────────────────
# A3. w/o Pref Input
# pref_vec 입력 제거 → room_response + audio features만으로 동작
# 확인: pref_vec 없이 dual-objective가 가능한가
# ──────────────────────────────────────────────

class A3_NoPrefInput(nn.Module):
    """
    Preference 입력(mode_id, band_gains) 제거.
    room_response + audio features만 사용하여 dual target을 맞추는 통제실험.
    """
    def __init__(
        self,
        in_dim=10,
        n_room_bins=128,
        room_dim=32,
        hidden_dim=64,
        n_tcn_blocks=4,
        n_freqs=128,
        **kw
    ):
        super().__init__()
        self.room_encoder = RoomResponseEncoder(n_room_bins, room_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(hidden_dim, dilation=2**i) for i in range(n_tcn_blocks)
        ])
        self.pool = AttentionPooling(hidden_dim)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + room_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_freqs),
            nn.Tanh(),
        )
        self.output_scale = 18.0

    def forward(self, x, room_response, mode_id, band_gains):
        # mode_id, band_gains는 완전히 무시
        room_vec = self.room_encoder(room_response)

        x = self.input_proj(x)
        for tcn in self.tcn_blocks:
            x = tcn(x)
        ctx, attn = self.pool(x)

        pred = self.output_head(torch.cat([ctx, room_vec], dim=-1)) * self.output_scale

        return {
            "pred_response_db": pred,
            "fc": None,
            "gain": None,
            "q": None,
            "attn_weights": attn,
            "cond_vec": room_vec,
        }

# ──────────────────────────────────────────────
# 레지스트리
# ──────────────────────────────────────────────

def get_ablation_loss(variant: str) -> DualObjectiveEQLoss:
    if variant in {"A0_Proposed", "A2_NoPrefLoss"}:
        return A0_ProposedLoss()
    if variant in {"A2_withPrefLoss", "A0_Full"}:
        return A2_withPrefLossLoss()
    return DualObjectiveEQLoss()


def get_ablation_registry() -> dict:
    from model import DualObjectiveAdaptivePEQ
    return {
        "A0_Proposed":    DualObjectiveAdaptivePEQ(),
        "A1_NoRoomInput": A1_NoRoomInput(),
        "A2_withPrefLoss": DualObjectiveAdaptivePEQ(),
        "A3_NoPrefInput": A3_NoPrefInput(),
    }


if __name__ == "__main__":
    registry = get_ablation_registry()
    B = 4
    x  = torch.randn(B, 32, 10)
    rr = torch.randn(B, 128)
    mi = torch.randint(0, 4, (B,))
    bg = torch.randn(B, 10)

    for name, model in registry.items():
        model.eval()
        with torch.no_grad():
            out = model(x, rr, mi, bg)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:<20} params={params:>8,}  shape={tuple(out['pred_response_db'].shape)}")
