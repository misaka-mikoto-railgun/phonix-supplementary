"""
exact_peq_response.py — differentiable exact RBJ biquad magnitude response
===========================================================================
Drop-in replacement for baselines.DifferentiablePEQResponse (same constructor,
same forward (fc, gain, q) -> (B, F) dB, same grid), computing the exact
magnitude of Audio-EQ-Cookbook sections. Each of the K sections has a type:

    'P'  peaking EQ          'L'  low shelf          'H'  high shelf

given as a string, e.g. section_types="LPPPPPH" (1 low shelf + 5 peaks +
1 high shelf; A0/A2 and the Option C heads) or None / "PPPPPPP" (peaking only;
E3/E4/E5 via baselines.DifferentiablePEQResponse).

All three are evaluated through their analog prototypes under the bilinear
transform with the prewarped corner frequency, which is exactly how the
cookbook coefficients are derived, so the result equals the digital filter
the device runs (verified to <1e-7 dB in float64, <1e-4 dB in float32 against
the cookbook z-domain transfer functions, for fc ∈ [80, 16000], gain ∈ [-12, 12],
Q ∈ [0.3, 8]):

    A = 10^(gain/40),  t0 = tan(w0/2),  u = cos(w/2),  v = sin(w/2)
    peaking:    |H|^2 = [ (u²t0² − v²)² + (A/Q)²  v²u²t0² ] / [ (u²t0² − v²)² + (1/(AQ))² v²u²t0² ]
    low shelf:  |H|^2 = A² [ (A u²t0² − v²)² + (A/Q²) v²u²t0² ] / [ (u²t0² − A v²)² + (A/Q²) v²u²t0² ]
    high shelf: |H|^2 = A² [ (u²t0² − A v²)² + (A/Q²) v²u²t0² ] / [ (A u²t0² − v²)² + (A/Q²) v²u²t0² ]

Every term is a product of bounded, directly computed factors, so the forms
are float32-safe down to the 20 Hz bin (no large terms cancelling to a tiny
result). Gain semantics: a peaking section reaches `gain` dB at fc; a shelf
reaches `gain` dB on its plateau (DC for 'L', Nyquist for 'H'). Q has the
cookbook meaning for all three (shelf Q > 1/√2 adds a resonant lip).
Sections are summed in dB (cascade), exactly as on the device.

Usage (model.py / arch_biquad.py):
    from exact_peq_response import ExactPEQResponse
    self.peq_response = ExactPEQResponse(sample_rate=sample_rate, n_freqs=n_freqs,
                                         f_min=f_min, f_max=f_max, freq_spacing=freq_spacing,
                                         section_types="LPPPPPH")
"""
import math
import torch
import torch.nn as nn

try:
    from frequency_grid import make_frequency_grid_torch   # the repo's grid helper (no circular import via baselines)
except Exception:  # standalone fallback: log-spaced grid like the repo's
    def make_frequency_grid_torch(n_freqs=128, f_min=20.0, f_max=24000.0, spacing="log"):
        if spacing == "log":
            return torch.logspace(math.log10(f_min), math.log10(f_max), n_freqs)
        return torch.linspace(f_min, f_max, n_freqs)


class ExactPEQResponse(nn.Module):
    """Sum (cascade, in dB) of K exact RBJ sections (peaking / low shelf / high shelf)."""

    def __init__(self, sample_rate=48000, n_freqs=128, f_min=20.0, f_max=24000.0,
                 freq_spacing="log", section_types=None, eps=1e-20):
        super().__init__()
        if f_max is None:
            f_max = sample_rate / 2.0
        freqs = make_frequency_grid_torch(n_freqs=n_freqs, f_min=f_min, f_max=f_max, spacing=freq_spacing)
        self.register_buffer("freqs", freqs)
        # kept only so checkpoints written by earlier versions (which stored w) still load;
        # forward recomputes everything from freqs in the current dtype.
        self.register_buffer("w", 2 * math.pi * freqs / sample_rate)
        self.sample_rate = sample_rate
        self.eps = eps
        self.section_types = None if section_types is None else str(section_types).upper()
        if self.section_types is not None and set(self.section_types) - set("PLH"):
            raise ValueError(f"section_types must use only P/L/H, got {section_types!r}")

    def extra_repr(self):
        return f"section_types={self.section_types or 'P*'}"

    def forward(self, fc, gain, q, section_types=None):
        """fc (Hz), gain (dB), q: each (B, K). Returns (B, F) in dB.
        section_types: optional override (e.g. a single section's type when rendering
        one section at a time: model.peq_response(fc[:, k:k+1], ..., section_types=model.peq_response.section_types[k]))."""
        B, K = fc.shape
        types = (section_types.upper() if section_types is not None else self.section_types) or "P" * K
        if len(types) != K:
            raise ValueError(f"section_types has {len(types)} entries but K={K} "
                             f"(pass section_types=... when rendering a subset of sections)")
        w = (2 * math.pi * self.freqs / self.sample_rate).view(1, 1, -1)      # (1,1,F)
        u2 = torch.cos(0.5 * w) ** 2
        v2 = torch.sin(0.5 * w) ** 2
        A = torch.pow(10.0, gain / 40.0).unsqueeze(-1)                         # (B,K,1)
        A2 = A * A
        t02 = torch.tan(math.pi * fc / self.sample_rate).unsqueeze(-1) ** 2     # tan²(w0/2)
        qq = q.unsqueeze(-1)
        S = v2 * u2 * t02                                                       # shared term

        # peaking
        Xp = u2 * t02 - v2
        Np = Xp * Xp + (A / qq) ** 2 * S
        Dp = Xp * Xp + (1.0 / (A * qq)) ** 2 * S
        # low shelf
        Sq = (A / (qq * qq)) * S
        Nl = A2 * ((A * u2 * t02 - v2) ** 2 + Sq)
        Dl = (u2 * t02 - A * v2) ** 2 + Sq
        # high shelf
        Nh = A2 * ((u2 * t02 - A * v2) ** 2 + Sq)
        Dh = (A * u2 * t02 - v2) ** 2 + Sq

        isL = torch.tensor([c == "L" for c in types], device=fc.device).view(1, K, 1)
        isH = torch.tensor([c == "H" for c in types], device=fc.device).view(1, K, 1)
        N = torch.where(isL, Nl, torch.where(isH, Nh, Np))
        D = torch.where(isL, Dl, torch.where(isH, Dh, Dp))
        mag_db = 10.0 * (torch.log10(N + self.eps) - torch.log10(D + self.eps))
        return mag_db.sum(dim=1)                                                # (B,F)


if __name__ == "__main__":  # self-test against the cookbook z-domain transfer functions
    import numpy as np
    torch.manual_seed(0)
    fs = 48000; B, K = 256, 7
    types = "LPPPPPH"
    fc = 80 + torch.rand(B, K) * (16000 - 80); g = (torch.rand(B, K) * 2 - 1) * 12; q = 0.3 + torch.rand(B, K) * 7.7
    fc[0] = 80.0; g[0] = 12.0; q[0] = 8.0
    fc[1] = 80.0; g[1] = -12.0; q[1] = 0.3
    fc[2] = 16000.0; g[2] = 12.0; q[2] = 0.3
    m = ExactPEQResponse(section_types=types)
    f = m.freqs.double().numpy(); w = 2 * np.pi * f / fs; z = np.exp(-1j * w)

    def cookbook(kind, fc_, g_, q_):
        A = 10 ** (g_ / 40); w0 = 2 * np.pi * fc_ / fs; al = np.sin(w0) / (2 * q_); c = np.cos(w0); sA = np.sqrt(A)
        if kind == "P":
            b = [1 + al * A, -2 * c, 1 - al * A]; a = [1 + al / A, -2 * c, 1 - al / A]
        elif kind == "L":
            b = [A * ((A + 1) - (A - 1) * c + 2 * sA * al), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - 2 * sA * al)]
            a = [(A + 1) + (A - 1) * c + 2 * sA * al, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - 2 * sA * al]
        else:
            b = [A * ((A + 1) + (A - 1) * c + 2 * sA * al), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - 2 * sA * al)]
            a = [(A + 1) - (A - 1) * c + 2 * sA * al, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - 2 * sA * al]
        H = (b[0] + b[1] * z + b[2] * z ** 2) / (a[0] + a[1] * z + a[2] * z ** 2)
        return 20 * np.log10(np.abs(H))

    ref = np.zeros((B, len(f)))
    for i in range(B):
        for k in range(K):
            ref[i] += cookbook(types[k], fc[i, k].item(), g[i, k].item(), q[i, k].item())
    for dtype in (torch.float64, torch.float32):
        mm = ExactPEQResponse(section_types=types).to(dtype)
        p = [t.to(dtype).clone().requires_grad_(True) for t in (fc, g, q)]
        out = mm(*p); out.sum().backward()
        assert all(torch.isfinite(t.grad).all() for t in p), "non-finite gradient"
        err = np.abs(out.detach().double().numpy() - ref).max()
        print(f"{str(dtype):14s} section_types={types}  max |torch - cookbook| = {err:.2e} dB, grads finite")
        assert err < (1e-6 if dtype == torch.float64 else 1e-2)
    # peaking-only path must equal the old behaviour
    m2 = ExactPEQResponse().double(); out2 = m2(fc.double(), g.double(), q.double())
    ref2 = np.zeros((B, len(f)))
    for i in range(B):
        for k in range(K):
            ref2[i] += cookbook("P", fc[i, k].item(), g[i, k].item(), q[i, k].item())
    assert np.abs(out2.detach().numpy() - ref2).max() < 1e-6
    print("self-test OK")
