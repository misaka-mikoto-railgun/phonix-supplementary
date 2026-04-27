import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines import DifferentiablePEQResponse
import math

from frequency_grid import make_frequency_grid_torch


class RoomResponseEncoder(nn.Module):
    def __init__(self, n_room_bins=128, room_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_room_bins, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, room_dim), nn.LayerNorm(room_dim), nn.GELU(),
        )
    def forward(self, room_response):
        return self.encoder(room_response)


class UserPreferenceEncoder(nn.Module):
    def __init__(self, n_modes=4, n_bands=10,
                 mode_embed_dim=16, band_embed_dim=64, pref_dim=64):
        super().__init__()
        self.mode_embedding = nn.Embedding(n_modes, mode_embed_dim)
        self.band_proj = nn.Sequential(
            nn.Linear(n_bands, band_embed_dim), nn.LayerNorm(band_embed_dim), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(mode_embed_dim + band_embed_dim, pref_dim),
            nn.LayerNorm(pref_dim), nn.GELU())

    def forward(self, mode_id, band_gains):
        return self.fusion(torch.cat([
            self.mode_embedding(mode_id), self.band_proj(band_gains)], dim=-1))


class CausalConv1d(nn.Module):
    def __init__(self, ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel_size, dilation=dilation, padding=0)
    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = CausalConv1d(dim, kernel_size, dilation)
        self.conv2 = CausalConv1d(dim, kernel_size, dilation * 2)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop  = nn.Dropout(0.1)

    def forward(self, x):
        r = x
        x = self.drop(F.gelu(self.norm1(self.conv1(x.transpose(1,2)).transpose(1,2))))
        x = self.drop(F.gelu(self.norm2(self.conv2(x.transpose(1,2)).transpose(1,2))))
        return x + r


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)
    def forward(self, x):
        w = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return (w.unsqueeze(-1) * x).sum(dim=1), w

class DualObjectiveAdaptivePEQ(nn.Module):
    def __init__(self, in_dim=10, n_room_bins=128, room_dim=32,
                 hidden_dim=64, n_tcn_blocks=4, n_modes=4, n_bands=10, 
                 pref_dim=64, n_freqs=128, sample_rate=48000, n_peq_filters=7,
                 f_min=20.0, f_max=None, freq_spacing="log", **kw):
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
        self.pool = AttentionPooling(hidden_dim)

        # Stage A: room correction prior
        self.room_mean_head = nn.Sequential(
            #nn.Linear(hidden_dim + room_dim, hidden_dim), 
            nn.Linear(room_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.room_shape_head = nn.Sequential(
            #nn.Linear(hidden_dim + room_dim, hidden_dim * 2),
            nn.Linear(room_dim, hidden_dim * 2),
            nn.GELU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_freqs), nn.Tanh(),
        )

        # Stage B: preference-conditioned residual refinement
        pref_input_dim = hidden_dim + room_dim + pref_dim + n_freqs + n_freqs

        self.peq_head = nn.Sequential(
            nn.Linear(pref_input_dim, hidden_dim * 2),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_peq_filters * 3)  # fc, gain, Q
        )
        self.peq_response = DifferentiablePEQResponse(
            sample_rate=sample_rate,
            n_freqs=n_freqs,
            f_min=f_min,
            f_max=f_max,
            freq_spacing=freq_spacing,
        )

        self.fc_min, self.fc_max = 80.0, 16000.0
        self.gain_max = 6.0
        self.q_min, self.q_max = 0.3, 8.0

        self.room_mean_scale = 4.0
        self.room_shape_scale = 12.0
        self.pref_res_scale = 6.0
        self.pref_anchor_scale = 1.0

        self.register_buffer(
            "band_freqs",
            torch.tensor([63,125,250,500,1000,2000,4000,8000,12000,16000], dtype=torch.float32)
        )
        self.register_buffer(
            "target_freqs",
            make_frequency_grid_torch(
                n_freqs=n_freqs,
                f_min=f_min,
                f_max=f_max,
                spacing=freq_spacing,
            )
        )

    @staticmethod
    def _interp(xp, fp, x):
        x = torch.clamp(x, xp[0], xp[-1])
        idx = torch.searchsorted(xp, x).clamp(1, len(xp)-1)
        x0, x1 = xp[idx-1], xp[idx]
        f0, f1 = fp[idx-1], fp[idx]
        t = (x - x0) / (x1 - x0 + 1e-6)
        return f0 + t * (f1 - f0)

    def forward(self, x, room_response, mode_id, band_gains):
        room_vec = self.room_encoder(room_response)
        pref_vec = self.pref_encoder(mode_id, band_gains)

        x = self.input_proj(x)
        for tcn in self.tcn_blocks:
            x = tcn(x)
        ctx, attn = self.pool(x)

        # ---- room correction prior ----
        #room_input = torch.cat([ctx, room_vec], dim=-1)
        room_input = room_vec  # room_response만으로 Stage A 추정

        room_mean = torch.tanh(self.room_mean_head(room_input)) * self.room_mean_scale
        room_shape = self.room_shape_head(room_input) * self.room_shape_scale
        room_shape = room_shape - room_shape.mean(dim=-1, keepdim=True)
        room_corr = room_mean + room_shape

        # ---- preference curve (anchor) ----
        pref_curve = torch.zeros(
            band_gains.shape[0], len(self.target_freqs),
            device=band_gains.device
        )
        for i in range(band_gains.shape[0]):
            pref_curve[i] = self._interp(self.band_freqs, band_gains[i], self.target_freqs)

        # ---- preference-conditioned residual refinement ----
        pref_input = torch.cat([ctx, room_vec, pref_vec, room_corr, pref_curve], dim=-1)
        peq_params = self.peq_head(pref_input)

        fc_raw, gain_raw, q_raw = peq_params.chunk(3, dim=-1)
        fc   = self.fc_min + torch.sigmoid(fc_raw) * (self.fc_max - self.fc_min)
        gain = torch.tanh(gain_raw) * self.gain_max
        q    = self.q_min + torch.sigmoid(q_raw) * (self.q_max - self.q_min)

        peq_resp = self.peq_response(fc, gain, q)  # (B, n_freqs)

        pred = room_corr + peq_resp

        return {
            "pred_response_db":   pred,
            "room_correction_db": room_corr,
            "peq_response_db":    peq_resp,
            "fc": fc, "gain": gain, "q": q,
            "pref_curve_db":      pref_curve,
            "attn_weights":       attn,
        }
    
class DualObjectiveEQLoss(nn.Module):
    """
    Hierarchical Dual-Objective EQ Loss
    - final prediction: pred_response_db -> dual_target_db
    - room branch:      room_correction_db -> room_target_db
    - pref branch:      pref_residual_db -> (dual_target_db - room_target_db)
    - perceptual heard alignment: (pred - room_target) -> pref_target_db

    목적:
      1) room branch가 LSD/세밀한 룸 보정을 담당
      2) pref residual branch가 취향 반영을 담당
      3) 최종 출력은 dual target을 맞춤

    기본 설정은 A0_Proposed 손실이다.
    A2_withPrefLoss는 lambda_pref_res / lambda_dir를 명시적으로 켜서 생성한다.
    """
    def __init__(
        self,
        lambda_final=1.0,
        lambda_room=0.35,
        lambda_pref_res=0.0,
        lambda_shape=0.20,
        lambda_grad=0.20,
        lambda_curv=0.08,
        lambda_dir=0.0,
        lambda_mean=0.03,
        n_freqs=128,
        f_min=20.0,
        f_max=24000.0,
        freq_spacing="log",
        use_perceptual=True,
        mag_weight_alpha=0.10,
        grad_weight_beta=0.30,
        low_edge_boost=1.30,
        high_edge_boost=1.15,
        edge_low_hz=300.0,
        edge_high_hz=8000.0,
    ):
        super().__init__()
        self.lambda_final = lambda_final
        self.lambda_room = lambda_room
        self.lambda_pref_res = lambda_pref_res
        self.lambda_shape = lambda_shape
        self.lambda_grad = lambda_grad
        self.lambda_curv = lambda_curv
        self.lambda_dir = lambda_dir
        self.lambda_mean = lambda_mean
        self.use_perceptual = use_perceptual
        self.pref_anchor_scale = 1.0

        self.mag_weight_alpha = mag_weight_alpha
        self.grad_weight_beta = grad_weight_beta

        freqs = make_frequency_grid_torch(
            n_freqs=n_freqs,
            f_min=f_min,
            f_max=f_max,
            spacing=freq_spacing,
        )
        f2 = freqs ** 2

        a_weight_db = 20 * torch.log10(
            (12194**2 * f2**2) /
            ((f2 + 20.6**2) * torch.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2))
        ) + 2.0
        perceptual_weight = 10 ** (a_weight_db / 20.0)
        perceptual_weight = perceptual_weight / perceptual_weight.mean()

        edge_weight = torch.ones_like(freqs)
        edge_weight = torch.where(freqs <= edge_low_hz, edge_weight * low_edge_boost, edge_weight)
        edge_weight = torch.where(freqs >= edge_high_hz, edge_weight * high_edge_boost, edge_weight)

        self.register_buffer("freqs", freqs)
        self.register_buffer("perceptual_weight", perceptual_weight)
        self.register_buffer("edge_weight", edge_weight)

    @staticmethod
    def _first_diff(x):
        return x[:, 1:] - x[:, :-1]

    @staticmethod
    def _second_diff(x):
        return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]

    def _weighted_curve_loss(self, pred, target):
        device = pred.device
        err = pred - target
        diff_sq = err ** 2

        mag_weight = 1.0 + self.mag_weight_alpha * torch.abs(target)

        target_grad_mag = torch.zeros_like(target)
        target_grad_mag[:, 1:] = torch.abs(target[:, 1:] - target[:, :-1])
        grad_weight = 1.0 + self.grad_weight_beta * target_grad_mag

        if self.use_perceptual:
            base_weight = self.perceptual_weight.to(device) * self.edge_weight.to(device)
        else:
            base_weight = self.edge_weight.to(device)

        total_weight = base_weight.unsqueeze(0) * mag_weight * grad_weight
        total_weight = total_weight / (total_weight.mean(dim=-1, keepdim=True) + 1e-8)

        main = (diff_sq * total_weight).mean()

        pred_centered = pred - pred.mean(dim=-1, keepdim=True)
        target_centered = target - target.mean(dim=-1, keepdim=True)
        shape = F.mse_loss(pred_centered, target_centered)

        pred_grad = self._first_diff(pred)
        target_grad = self._first_diff(target)
        grad = F.mse_loss(pred_grad, target_grad)

        pred_curv = self._second_diff(pred)
        target_curv = self._second_diff(target)
        curv = F.mse_loss(pred_curv, target_curv)

        mean_penalty = F.mse_loss(
            pred.mean(dim=-1),
            target.mean(dim=-1)
        )

        total = (
            main
            + self.lambda_shape * shape
            + self.lambda_grad * grad
            + self.lambda_curv * curv
            + self.lambda_mean * mean_penalty
        )

        return {
            "total": total,
            "main": main,
            "shape": shape,
            "grad": grad,
            "curv": curv,
            "mean": mean_penalty,
        }

    def forward(
        self,
        pred_response_db,
        dual_target_db,
        gain=None,
        pref_target_db=None,
        room_target_db=None,
        prev_gain=None,
        room_correction_db=None,
        peq_response_db=None,
        pref_curve_db=None,
    ):
        device = pred_response_db.device

        # -----------------------------------
        # 1) Final dual-target loss
        # -----------------------------------
        final_dict = self._weighted_curve_loss(pred_response_db, dual_target_db)
        final_loss = final_dict["total"]

        # -----------------------------------
        # 2) Room branch supervision
        # -----------------------------------
        if room_correction_db is not None and room_target_db is not None and self.lambda_room > 0:
            room_dict = self._weighted_curve_loss(room_correction_db, room_target_db)
            room_loss = room_dict["total"]
        else:
            room_dict = None
            room_loss = torch.tensor(0.0, device=device)

        # -----------------------------------
        # 3) PEQ response supervision
        #    peq_target = dual_target - room_target
        #    (pref_curve는 conditioning 입력으로만 사용,
        #     출력에 직접 더하지 않으므로 빼지 않음)
        # -----------------------------------
        if peq_response_db is not None and room_target_db is not None and self.lambda_pref_res > 0:
            peq_target = dual_target_db - room_target_db
            peq_dict = self._weighted_curve_loss(peq_response_db, peq_target)
            pref_res_loss = peq_dict["total"]
        else:
            peq_dict = None
            pref_res_loss = torch.tensor(0.0, device=device)

        # -----------------------------------
        # 4) Direction loss on "heard" response
        #    heard = pred + room_clean = pred - room_target
        # -----------------------------------
        if pref_target_db is not None and room_target_db is not None and self.lambda_dir > 0:
            heard = pred_response_db - room_target_db
            dir_loss = 1.0 - F.cosine_similarity(heard, pref_target_db, dim=-1).mean()
        elif pref_target_db is not None and self.lambda_dir > 0:
            dir_loss = 1.0 - F.cosine_similarity(pred_response_db, pref_target_db, dim=-1).mean()
        else:
            dir_loss = torch.tensor(0.0, device=device)

        total = (
            self.lambda_final * final_loss
            + self.lambda_room * room_loss
            + self.lambda_pref_res * pref_res_loss  # 이름 그대로 유지 (lambda는 재사용)
            + self.lambda_dir * dir_loss
        )

        out = {
            "loss": total,
            "final_loss": final_loss,
            "final_main_loss": final_dict["main"],
            "final_shape_loss": final_dict["shape"],
            "final_grad_loss": final_dict["grad"],
            "final_curv_loss": final_dict["curv"],
            "final_mean_loss": final_dict["mean"],
            "room_loss": room_loss,
            "peq_loss": pref_res_loss,      # pref_res_loss → peq_loss
            "dir_loss": dir_loss,
        }

        if room_dict is not None:
            out.update({
                "room_main_loss": room_dict["main"],
                "room_shape_loss": room_dict["shape"],
                "room_grad_loss": room_dict["grad"],
                "room_curv_loss": room_dict["curv"],
                "room_mean_loss": room_dict["mean"],
            })

        if peq_dict is not None:            # pref_res_dict → peq_dict
            out.update({
                "peq_main_loss": peq_dict["main"],
                "peq_shape_loss": peq_dict["shape"],
                "peq_grad_loss": peq_dict["grad"],
                "peq_curv_loss": peq_dict["curv"],
                "peq_mean_loss": peq_dict["mean"],
            })

        return out
    
def count_parameters(model):
    t = sum(p.numel() for p in model.parameters())
    return {"total_params": t, "trainable_params": t, "size_MB": t*4/1024/1024}


if __name__ == "__main__":
    m = DualObjectiveAdaptivePEQ()
    B = 4
    out = m(torch.randn(B,32,10), torch.randn(B,128),
            torch.randint(0,4,(B,)), torch.randn(B,10))
    print(f"pred: {out['pred_response_db'].shape}")
    loss = DualObjectiveEQLoss()(out["pred_response_db"], torch.randn(B,128),
                                  pref_target_db=torch.randn(B,128))
    print(f"loss: {loss['loss'].item():.2f}")
    print(f"params: {count_parameters(m)['total_params']:,}")
