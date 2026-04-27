"""
External Baselines v3 (Simplified)

E1. No Processing
E2. Static Mode EQ (user-pref only)
E3. Nercessian 2020 (MLP, room correction only)
E4. Pepe 2020 (CNN, room correction only)
E5. Sequential (E3→E2, 2-step)
E6. DSP Analytical (no AI, 수학 역전)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from frequency_grid import make_frequency_grid_torch


# ── 모드 프로파일 (확대 ±8dB) ──────────────────

MODE_PROFILES_DB = {
    0: {63:-4.0, 125:-3.0, 250:0.0, 500:+2.0, 1000:+5.0,
        2000:+6.0, 4000:+4.0, 8000:+2.0, 12000:-1.0, 16000:-2.0},
    1: {63:+8.0, 125:+6.0, 250:+3.0, 500:0.0, 1000:-1.0,
        2000:-2.0, 4000:-3.0, 8000:-2.0, 12000:-1.0, 16000:0.0},
    2: {63:-2.0, 125:-1.0, 250:0.0, 500:0.0, 1000:+1.0,
        2000:+3.0, 4000:+5.0, 8000:+7.0, 12000:+8.0, 16000:+6.0},
    3: {63:+2.0, 125:+1.0, 250:+1.0, 500:0.0, 1000:-1.0,
        2000:-3.0, 4000:-5.0, 8000:-6.0, 12000:-4.0, 16000:-3.0},
}


def build_mode_response_tensor(mode_id, freqs):
    """모드 프로파일 → (B, F) 벡터화 보간"""
    device = freqs.device
    pf = torch.tensor(list(MODE_PROFILES_DB[0].keys()), dtype=torch.float32, device=device)
    all_gains = torch.stack([
        torch.tensor(list(MODE_PROFILES_DB[m].values()), dtype=torch.float32, device=device)
        for m in range(4)])
    idx = torch.searchsorted(pf, freqs).clamp(1, len(pf)-1)
    f0, f1 = pf[idx-1], pf[idx]
    t = (freqs - f0) / (f1 - f0 + 1e-6)
    g0, g1 = all_gains[:, idx-1], all_gains[:, idx]
    return (g0 + t.unsqueeze(0) * (g1 - g0))[mode_id]


# ── PEQ Response (E3/E4용) ──────────────────

class DifferentiablePEQResponse(nn.Module):
    """Biquad peaking EQ의 differentiable Gaussian 근사"""
    def __init__(
        self,
        sample_rate=48000,
        n_freqs=128,
        f_min=20.0,
        f_max=24000.0,
        freq_spacing="log",
    ):
        super().__init__()
        if f_max is None:
            f_max = sample_rate / 2.0
        self.register_buffer(
            "freqs",
            make_frequency_grid_torch(
                n_freqs=n_freqs,
                f_min=f_min,
                f_max=f_max,
                spacing=freq_spacing,
            ),
        )
        self.sample_rate = sample_rate

    def forward(self, fc, gain, q):
        B, K = fc.shape
        F_ = self.freqs.shape[0]
        freqs = self.freqs.view(1, 1, F_)
        fc_, gain_, q_ = fc.view(B,K,1), gain.view(B,K,1), q.view(B,K,1)
        omega_0 = 2*math.pi*fc_/self.sample_rate
        bw = omega_0 / q_
        delta = (2*math.pi*freqs/self.sample_rate - omega_0).abs()
        w = torch.exp(-0.5*(delta/(bw+1e-6))**2)
        return (gain_ * w).sum(dim=1)


# ── E1. No Processing ──────────────────

class E1_NoProcessing(nn.Module):
    def __init__(self, n_freqs=128):
        super().__init__()
        self.n_freqs = n_freqs
    def forward(self, x, **kw):
        return {"pred_response_db": torch.zeros(x.shape[0], self.n_freqs, device=x.device)}


# ── E2. Static Mode EQ ──────────────────

class E2_StaticModeEQ(nn.Module):
    def __init__(self, n_freqs=128, f_min=20.0, f_max=24000.0, freq_spacing="log"):
        super().__init__()
        self.register_buffer(
            "freqs",
            make_frequency_grid_torch(
                n_freqs=n_freqs,
                f_min=f_min,
                f_max=f_max,
                spacing=freq_spacing,
            ),
        )
    def forward(self, x, mode_id=None, **kw):
        if mode_id is None:
            return {"pred_response_db": torch.zeros(x.shape[0], len(self.freqs), device=x.device)}
        return {"pred_response_db": build_mode_response_tensor(mode_id, self.freqs)}


# ── E3. Nercessian MLP ──────────────────

class E3_NercessianMLP(nn.Module):
    def __init__(self, in_dim=10, hidden_dim=256, num_filters=5,
                 n_freqs=128, sample_rate=48000, f_min=20.0, f_max=24000.0,
                 freq_spacing="log", **kw):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_filters*3))
        self.response = DifferentiablePEQResponse(
            sample_rate=sample_rate,
            n_freqs=n_freqs,
            f_min=f_min,
            f_max=f_max,
            freq_spacing=freq_spacing,
        )
        self.f_min, self.f_max = 80.0, 24000.0
        self.g_min, self.g_max = -12.0, 12.0
        self.q_min, self.q_max = 0.3, 8.0

    def forward(self, x, **kw):
        out = self.mlp(x.mean(dim=1))
        fc_r, g_r, q_r = out.chunk(3, dim=-1)
        fc = self.f_min + torch.sigmoid(fc_r)*(self.f_max-self.f_min)
        g  = torch.tanh(g_r)*self.g_max
        q  = self.q_min + torch.sigmoid(q_r)*(self.q_max-self.q_min)
        return {"pred_response_db": self.response(fc,g,q), "fc":fc, "gain":g, "q":q}


# ── E4. Pepe CNN ──────────────────

class E4_PepeCNN(nn.Module):
    def __init__(self, in_dim=10, num_filters=5, n_freqs=128, sample_rate=48000,
                 f_min=20.0, f_max=24000.0, freq_spacing="log", **kw):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_dim,32,3,padding=1), nn.ReLU(),
            nn.Conv1d(32,64,3,padding=1), nn.ReLU(),
            nn.Conv1d(64,128,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Linear(128,128), nn.ReLU(), nn.Linear(128,num_filters*3))
        self.response = DifferentiablePEQResponse(
            sample_rate=sample_rate,
            n_freqs=n_freqs,
            f_min=f_min,
            f_max=f_max,
            freq_spacing=freq_spacing,
        )
        self.f_min, self.f_max = 80.0, 24000.0
        self.g_min, self.g_max = -12.0, 12.0
        self.q_min, self.q_max = 0.3, 8.0

    def forward(self, x, **kw):
        out = self.fc(self.cnn(x.transpose(1,2)).squeeze(-1))
        fc_r, g_r, q_r = out.chunk(3, dim=-1)
        fc = self.f_min + torch.sigmoid(fc_r)*(self.f_max-self.f_min)
        g  = torch.tanh(g_r)*self.g_max
        q  = self.q_min + torch.sigmoid(q_r)*(self.q_max-self.q_min)
        return {"pred_response_db": self.response(fc,g,q), "fc":fc, "gain":g, "q":q}


# ── E5. Sequential ──────────────────

class E5_Sequential(nn.Module):
    def __init__(self, in_dim=10, n_freqs=128, sample_rate=48000, f_min=20.0,
                 f_max=24000.0, freq_spacing="log", **kw):
        super().__init__()
        self.room_corrector = E3_NercessianMLP(
            in_dim,
            n_freqs=n_freqs,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            freq_spacing=freq_spacing,
        )
        self.pref_applier = E2_StaticModeEQ(
            n_freqs=n_freqs,
            f_min=f_min,
            f_max=f_max,
            freq_spacing=freq_spacing,
        )
    def forward(self, x, mode_id=None, **kw):
        room = self.room_corrector(x)
        pref = self.pref_applier(x, mode_id=mode_id)
        combined = room["pred_response_db"] + pref["pred_response_db"]
        return {"pred_response_db": combined, "room_response_db": room["pred_response_db"],
                "fc": room.get("fc"), "gain": room.get("gain"), "q": room.get("q")}


# ── E6. DSP Analytical ──────────────────

class E6_DSP_Analytical(nn.Module):
    """AI 없이 -room_response + pref_profile. 파라미터 0."""
    def __init__(self, n_freqs=128, f_min=20.0, f_max=24000.0, freq_spacing="log"):
        super().__init__()
        self.register_buffer(
            "freqs",
            make_frequency_grid_torch(
                n_freqs=n_freqs,
                f_min=f_min,
                f_max=f_max,
                spacing=freq_spacing,
            ),
        )
    def forward(self, x, room_response=None, mode_id=None, **kw):
        B, device = x.shape[0], x.device
        rc = (-room_response).clamp(-12,12) if room_response is not None else torch.zeros(B,len(self.freqs),device=device)
        pr = build_mode_response_tensor(mode_id, self.freqs) if mode_id is not None else torch.zeros(B,len(self.freqs),device=device)
        return {"pred_response_db": rc+pr, "fc":None, "gain":None, "q":None}


if __name__ == "__main__":
    B = 4
    x, rr = torch.randn(B,32,10), torch.randn(B,128)
    mi, bg = torch.randint(0,4,(B,)), torch.randn(B,10)

    models = {"E1":E1_NoProcessing(), "E2":E2_StaticModeEQ(), "E3":E3_NercessianMLP(),
              "E4":E4_PepeCNN(), "E5":E5_Sequential(), "E6":E6_DSP_Analytical()}
    for name, m in models.items():
        if name in ["E2","E5"]: out = m(x, mode_id=mi)
        elif name == "E6": out = m(x, room_response=rr, mode_id=mi)
        elif name == "E1": out = m(x)
        else: out = m(x)
        p = sum(pp.numel() for pp in m.parameters())
        print(f"{name}: {tuple(out['pred_response_db'].shape)}  params={p:,}")
