"""
exact_peq_np.py — numpy exact RBJ biquad response (for the non-torch scripts)
================================================================================
Same closed forms as exact_peq_response.py (torch) — peaking / low-shelf /
high-shelf cookbook sections via their prewarped analog prototypes — for
ac_fitting_A.py (Option A per-sample SciPy fitting), the Option D naive
seven-point rendering, and any other numpy-side rendering. Float64 throughout.
section_types: K-char string of P/L/H (default: all peaking).
"""
import numpy as np

SR = 48000
TARGET_FREQS = np.geomspace(20.0, 24000.0, 128)
OMEGA = 2 * np.pi * TARGET_FREQS / SR


def rbj_peq_response_db(fc, gain_db, q, omega=OMEGA, sr=SR, section_types=None):
    """fc/gain_db/q: (K,) or (N,K). section_types: str of K chars in {P,L,H} (default all peaking).
    Returns (F,) or (N,F) — sum of the K exact cookbook sections in dB."""
    fc = np.atleast_2d(np.asarray(fc, np.float64))[..., None]
    g = np.atleast_2d(np.asarray(gain_db, np.float64))[..., None]
    q = np.atleast_2d(np.asarray(q, np.float64))[..., None]
    K = fc.shape[1]
    types = (section_types or "P" * K).upper()
    assert len(types) == K, f"section_types has {len(types)} entries but K={K}"
    w = np.asarray(omega, np.float64)[None, None, :]
    u2 = np.cos(0.5 * w) ** 2; v2 = np.sin(0.5 * w) ** 2
    A = 10.0 ** (g / 40.0); A2 = A * A
    t02 = np.tan(np.pi * fc / sr) ** 2
    S = v2 * u2 * t02
    Xp = u2 * t02 - v2
    Np = Xp * Xp + (A / q) ** 2 * S;             Dp = Xp * Xp + (1.0 / (A * q)) ** 2 * S
    Sq = (A / (q * q)) * S
    Nl = A2 * ((A * u2 * t02 - v2) ** 2 + Sq);  Dl = (u2 * t02 - A * v2) ** 2 + Sq
    Nh = A2 * ((u2 * t02 - A * v2) ** 2 + Sq);  Dh = (A * u2 * t02 - v2) ** 2 + Sq
    isL = np.array([c == "L" for c in types])[None, :, None]
    isH = np.array([c == "H" for c in types])[None, :, None]
    N = np.where(isL, Nl, np.where(isH, Nh, Np)); D = np.where(isL, Dl, np.where(isH, Dh, Dp))
    out = (10.0 * (np.log10(N + 1e-30) - np.log10(D + 1e-30))).sum(axis=1)
    return out[0] if out.shape[0] == 1 and np.ndim(gain_db) == 1 else out


# ── drop-in for ac_fitting_A.gaussian_peq_response(params) ─────────────────────
def make_raw_param_renderer(n_filters, fc_min, fc_max, gain_max_getter, q_min, q_max, omega=OMEGA, sr=SR, section_types=None):
    """Returns f(params_raw) -> (F,) dB with the same sigmoid/tanh bounding A0 uses.
    gain_max_getter is a zero-arg callable so the module-level GAIN_MAX override in
    ac_fitting_A.main() is honoured."""
    def _sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    def render(params):
        p = np.asarray(params, np.float64).reshape(n_filters, 3)
        fc = fc_min + _sig(p[:, 0]) * (fc_max - fc_min)
        gain = np.tanh(np.clip(p[:, 1], -20, 20)) * gain_max_getter()
        q = q_min + _sig(p[:, 2]) * (q_max - q_min)
        return rbj_peq_response_db(fc, gain, q, omega, sr, section_types)
    return render


# ── Option D: naive seven-point sampling of the target, rendered with exact biquads ──
def option_d_naive7pt(target_db, fc_points=None, q=1.41, omega=OMEGA, sr=SR, freqs=TARGET_FREQS, section_types=None):
    """target_db: (N,F). Sample the target at K centre frequencies (default:
    geomspace 80–16000 Hz, K=7), use the sampled value as each section's gain, fixed Q
    (scalar, or one value per section), and render with exact RBJ sections of the given
    types (default all peaking; pass "LPPPPPH" to match the A0 / Option A / Option C head —
    for a shelf the sampled value becomes the plateau gain). Returns (N,F) dB."""
    if fc_points is None:
        fc_points = np.geomspace(80.0, 16000.0, 7)
    fc_points = np.asarray(fc_points, np.float64); K = len(fc_points)
    target_db = np.atleast_2d(np.asarray(target_db, np.float64))
    N = target_db.shape[0]
    logf = np.log(freqs)
    gains = np.stack([np.interp(np.log(fc_points), logf, target_db[i]) for i in range(N)])   # (N,K)
    fc = np.broadcast_to(fc_points, (N, K))
    qq = np.broadcast_to(np.asarray(q, np.float64).reshape(-1) if np.ndim(q) else np.full(K, float(q)), (N, K))
    return rbj_peq_response_db(fc, gains, qq, omega, sr, section_types)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fc = 80 + rng.random((50, 7)) * 15920; g = (rng.random((50, 7)) * 2 - 1) * 12; q = 0.3 + rng.random((50, 7)) * 7.7
    A = 10 ** (g / 40)[..., None]; w0 = (2 * np.pi * fc / SR)[..., None]; al = np.sin(w0) / (2 * q[..., None])
    z = np.exp(-1j * OMEGA)[None, None]; cw = np.cos(w0)
    H = (1 + al * A - 2 * cw * z + (1 - al * A) * z ** 2) / (1 + al / A - 2 * cw * z + (1 - al / A) * z ** 2)
    ref = (20 * np.log10(np.abs(H))).sum(1)
    err = np.abs(rbj_peq_response_db(fc, g, q) - ref).max()
    print(f"peaking: max |closed form - complex| = {err:.2e} dB"); assert err < 1e-6
    # shelves against cookbook coefficients
    def cb(kind, fc_, g_, q_):
        A = 10 ** (g_ / 40); w0 = 2 * np.pi * fc_ / SR; al = np.sin(w0) / (2 * q_); c = np.cos(w0); sA = np.sqrt(A)
        if kind == "L":
            b = [A * ((A + 1) - (A - 1) * c + 2 * sA * al), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - 2 * sA * al)]
            a = [(A + 1) + (A - 1) * c + 2 * sA * al, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - 2 * sA * al]
        else:
            b = [A * ((A + 1) + (A - 1) * c + 2 * sA * al), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - 2 * sA * al)]
            a = [(A + 1) - (A - 1) * c + 2 * sA * al, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - 2 * sA * al]
        zz = np.exp(-1j * OMEGA); H = (b[0] + b[1] * zz + b[2] * zz ** 2) / (a[0] + a[1] * zz + a[2] * zz ** 2)
        return 20 * np.log10(np.abs(H))
    types = "LPPPPPH"
    ref2 = np.zeros((50, 128))
    for i in range(50):
        for k in range(7):
            ref2[i] += cb(types[k], fc[i, k], g[i, k], q[i, k]) if types[k] != "P" else 0.0
    peak_only = rbj_peq_response_db(fc[:, 1:6], g[:, 1:6], q[:, 1:6])
    err2 = np.abs(rbj_peq_response_db(fc, g, q, section_types=types) - (ref2 + peak_only)).max()
    print(f"LPPPPPH: max |closed form - cookbook| = {err2:.2e} dB"); assert err2 < 1e-6
    r = make_raw_param_renderer(7, 80.0, 16000.0, lambda: 12.0, 0.3, 8.0)
    assert r(np.zeros(21)).shape == (128,)
    print("self-test OK")