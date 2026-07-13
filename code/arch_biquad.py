"""
arch_biquad.py — Biquad-constrained AC variants (AC1/AC2/AC3)
=============================================================
_ACBase 의 Stage B (dense 128-bin pref_res_head) 를
7-band Gaussian PEQ head 로 교체한 모델 정의.

Stage A: room correction (dense, free) — 원본과 동일
Stage B: DifferentiablePEQResponse 로 제약된 biquad pref residual

사용처
------
  ac_fitting_C.py              — Option C 재학습 실험
  experiments_fixed_updated.py — MODEL_REGISTRY 등록
  measure_rtf.py               — RTF 측정
"""

import torch
import torch.nn as nn

from arch_variants import _ACBase, _ConformerBlock
from model import AttentionPooling
from baselines import DifferentiablePEQResponse

N_FILTERS = 7
FC_MIN, FC_MAX = 80.0, 16000.0
GAIN_MAX        = 6.0
Q_MIN,  Q_MAX   = 0.3, 8.0


class _ACBiquadBase(_ACBase):
    """
    _ACBase 에서 Stage B (pref_res_head) 를 7-band biquad head 로 교체.
    서브클래스: 시퀀스 aggregator 추가 후 _build_biquad_heads() 호출.
    """

    def _build_biquad_heads(self, n_filters: int = N_FILTERS, gain_max: float = GAIN_MAX):
        # REVISION(gain±12): gain_max 생성자 인자화. 기본 6.0(원본 동일). GAIN_MAX는
        # state_dict 에 없으므로 ±6 ckpt 를 ±12 인스턴스에 그대로 로드 가능.
        self._gain_max = gain_max
        hd, rd, pd, nf = self.hidden_dim, self.room_dim, self.pref_dim, self.n_freqs

        # Stage A heads (원본과 동일)
        self.room_mean_head = nn.Sequential(
            nn.Linear(hd + rd, hd), nn.GELU(), nn.Linear(hd, 1)
        )
        self.room_shape_head = nn.Sequential(
            nn.Linear(hd + rd, hd * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hd * 2, hd), nn.GELU(), nn.Linear(hd, nf), nn.Tanh()
        )
        self.room_mean_scale  = 4.0
        self.room_shape_scale = 12.0

        # Stage B: biquad param head
        pref_in = hd + rd + pd + nf + nf
        self.peq_params_head = nn.Sequential(
            nn.Linear(pref_in, hd * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hd * 2, hd), nn.GELU(), nn.Linear(hd, n_filters * 3),
        )
        self.peq_response = DifferentiablePEQResponse(
            sample_rate=self.sample_rate, n_freqs=nf,
        )
        self._n_filters = n_filters

    def forward(self, x, room_response, mode_id, band_gains):
        room_vec = self.room_encoder(room_response)
        pref_vec = self.pref_encoder(mode_id, band_gains)

        h   = self._tcn_out(x)
        ctx = self.aggregate(h)

        pref_curve = self._pref_curve(band_gains)

        # Stage A: room correction (unchanged)
        room_input = torch.cat([ctx, room_vec], dim=-1)
        room_mean  = torch.tanh(self.room_mean_head(room_input)) * self.room_mean_scale
        room_shape = self.room_shape_head(room_input) * self.room_shape_scale
        room_shape = room_shape - room_shape.mean(dim=-1, keepdim=True)
        room_corr  = room_mean + room_shape

        # Stage B: biquad-constrained pref residual
        pref_input = torch.cat([ctx, room_vec, pref_vec, room_corr, pref_curve], dim=-1)
        raw  = self.peq_params_head(pref_input)           # (B, n_filters*3)
        raw3 = raw.view(raw.shape[0], self._n_filters, 3)

        fc   = FC_MIN + torch.sigmoid(raw3[..., 0]) * (FC_MAX - FC_MIN)
        gain = torch.tanh(raw3[..., 1]) * self._gain_max   # REVISION: 인자화된 bound
        q    = Q_MIN  + torch.sigmoid(raw3[..., 2]) * (Q_MAX  - Q_MIN)

        pref_res = self.peq_response(fc, gain, q)          # (B, n_freqs)

        pred = room_corr + pref_curve + pref_res

        return {
            "pred_response_db":   pred,
            "room_correction_db": room_corr,
            "pref_residual_db":   pref_res,
            "pref_curve_db":      pref_curve,
            "fc": fc, "gain": gain, "q": q,                # REVISION: verify/saturation 용
        }


class AC1_BiLSTM_Biquad(_ACBiquadBase):
    """BiLSTM aggregation + biquad Stage B"""
    def __init__(self, hidden_dim=64, gain_max=GAIN_MAX, **kw):
        super().__init__(hidden_dim=hidden_dim, **kw)
        self.bilstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self._build_biquad_heads(gain_max=gain_max)

    def aggregate(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.bilstm(h)
        return out.mean(dim=1)


class AC2_GRU_Biquad(_ACBiquadBase):
    """GRU (causal) aggregation + biquad Stage B"""
    def __init__(self, hidden_dim=64, gain_max=GAIN_MAX, **kw):
        super().__init__(hidden_dim=hidden_dim, **kw)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=2, batch_first=True, bidirectional=False,
        )
        self._build_biquad_heads(gain_max=gain_max)

    def aggregate(self, h: torch.Tensor) -> torch.Tensor:
        _, hn = self.gru(h)
        return hn[-1]


class AC3_Conformer_Biquad(_ACBiquadBase):
    """Conformer×2 aggregation + biquad Stage B"""
    def __init__(self, hidden_dim=64, n_conformer=2, gain_max=GAIN_MAX, **kw):
        super().__init__(hidden_dim=hidden_dim, **kw)
        self.conformer_blocks = nn.ModuleList([
            _ConformerBlock(hidden_dim) for _ in range(n_conformer)
        ])
        self.pool = AttentionPooling(hidden_dim)
        self._build_biquad_heads(gain_max=gain_max)

    def aggregate(self, h: torch.Tensor) -> torch.Tensor:
        for blk in self.conformer_blocks:
            h = blk(h)
        ctx, _ = self.pool(h)
        return ctx


BIQUAD_REGISTRY = {
    "AC1_BiLSTM_Biquad":   AC1_BiLSTM_Biquad,
    "AC2_GRU_Biquad":      AC2_GRU_Biquad,
    "AC3_Conformer_Biquad": AC3_Conformer_Biquad,
}
