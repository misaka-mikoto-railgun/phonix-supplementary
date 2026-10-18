"""
eval_exact_biquad.py — Gaussian surrogate vs. exact RBJ biquad rendering
=========================================================================
For every model that emits (fc, gain, Q), re-render the 7-band (or 5-band)
PEQ response with the exact RBJ biquad magnitude |H(e^jw)| at fs = 48 kHz
(each section with the type the model's head declares, peaking by default)
and compare it with the response the model was trained and evaluated with.

With the v1.1 checkpoints that response was the Gaussian surrogate and the gap
measures the surrogate error. Since v1.2 the heads are exact themselves, so
the two columns must agree (gap ~ 0); run_exact_pipeline.ps1 uses this as its
final consistency check. The "gauss" labels below are kept for continuity.

    pred_gauss = <model output as published>           (room_corr + Gaussian PEQ [+ pref_curve])
    pred_exact = pred_gauss - Gaussian_PEQ + RBJ_PEQ    (same params, exact biquad response)

Metrics (per split): LSD_dual, LSD_pref, DMR, CosSim for both renderings,
the per-sample LSD gap (exact - gauss), the RMS difference between the two
PEQ responses, and per-preset / per-source breakdowns. A0 and A2 use the
three ±12 dB seeds; everything else is single-seed.

Place next to the other generator scripts (code/) and run:

    python code/eval_exact_biquad.py --data_dir <dataset_v3> --rev_ckpt_dir <±12 ckpts> \
        [--eval_ckpt_dir <staging, for E3/E4/AC>] --out_dir results/exact_biquad

Outputs: exact_biquad_summary.csv, exact_biquad_summary.json,
         exact_biquad_persample_<model>_<split>.npz (per-sample arrays, A0 seed 7 only)
"""
import sys, json, csv
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_full as TF
from run_gain_freq_ablation import cname
from export_track_level_predictions import TrackAwarePEQDataset
import cli_paths
import ckpt_io

try:
    from arch_biquad import BIQUAD_REGISTRY
except Exception:
    BIQUAD_REGISTRY = {}

_P, _ = cli_paths.parse("Gaussian surrogate vs exact RBJ biquad rendering",
                        require=("data_dir", "rev_ckpt_dir"))
DATA, SAVE, CKE, OUT = _P.data_dir, _P.rev_ckpt_dir, _P.eval_ckpt_dir, _P.out_dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 7]
FS = 48000.0

# ── exact RBJ magnitude (dB) from cookbook z-domain coefficients, (N, K, F) ───
def rbj_db(fc, gain_db, q, freqs, section_types=None, fs=FS):
    """Audio-EQ-Cookbook sections ('P' peaking, 'L' low shelf, 'H' high shelf; default all
    peaking). fc/gain/q: (N,K); freqs: (F,). Returns (N,F) sum in dB. Evaluated from the
    z-domain coefficients, independently of the closed forms in exact_peq_response."""
    fc = np.asarray(fc, np.float64)[..., None]; g = np.asarray(gain_db, np.float64)[..., None]
    q = np.asarray(q, np.float64)[..., None]; w = (2 * np.pi * np.asarray(freqs, np.float64) / fs)[None, None, :]
    K = fc.shape[1]
    types = (section_types or "P" * K).upper()
    assert len(types) == K, f"section_types has {len(types)} entries but K={K}"
    A = 10 ** (g / 40.0); w0 = 2 * np.pi * fc / fs; al = np.sin(w0) / (2 * q); c = np.cos(w0); sA = np.sqrt(A)
    pk = [(1 + al * A, -2 * c, 1 - al * A), (1 + al / A, -2 * c, 1 - al / A)]
    ls = [(A * ((A + 1) - (A - 1) * c + 2 * sA * al), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - 2 * sA * al)),
          ((A + 1) + (A - 1) * c + 2 * sA * al, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - 2 * sA * al)]
    hs = [(A * ((A + 1) + (A - 1) * c + 2 * sA * al), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - 2 * sA * al)),
          ((A + 1) - (A - 1) * c + 2 * sA * al, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - 2 * sA * al)]
    isL = np.array([t == "L" for t in types])[None, :, None]; isH = np.array([t == "H" for t in types])[None, :, None]
    pick = lambda i, j: np.where(isL, ls[i][j], np.where(isH, hs[i][j], pk[i][j]))
    z1 = np.exp(-1j * w); z2 = z1 * z1
    H = (pick(0, 0) + pick(0, 1) * z1 + pick(0, 2) * z2) / (pick(1, 0) + pick(1, 1) * z1 + pick(1, 2) * z2)
    return (20 * np.log10(np.abs(H) + 1e-12)).sum(axis=1)

# ── data ─────────────────────────────────────────────────────────────────────
def load_split(name):
    ds = TrackAwarePEQDataset(f"{DATA}/{name}", device=str(device))
    d = ds.data
    meta = dict(dual=d["dual_target"].cpu().numpy(), room=d["room_target"].cpu().numpy(),
                pref=d["pref_target"].cpu().numpy(), mode=d["mode_id"].cpu().numpy(),
                room_id=d["room_id"].cpu().numpy(), freqs=np.asarray(ds.target_freqs, np.float64))
    return ds, meta

splits = {"test_synth": load_split("test_synth")}
if (DATA / "test_real").is_dir():
    splits["test_real"] = load_split("test_real")
    try:  # BUT / OpenAIR split, same rule as table5_ood.py
        rmap = json.load(open(DATA / "test_real" / "rir_map.json", encoding="utf-8"))
        id2path = {r["id"]: r["value"].lower() for r in rmap["rirs"]}
        def src(i):
            p = id2path.get(int(i), "")
            return "BUT" if ("but_reverb" in p or "but_" in p) else ("OpenAIR" if ("openair" in p or "open_air" in p) else "other")
        splits["test_real"][1]["source"] = np.array([src(i) for i in splits["test_real"][1]["room_id"]])
    except Exception as e:
        print(f"[warn] rir_map.json not parsed ({e}); no BUT/OpenAIR split")

# ── metrics ──────────────────────────────────────────────────────────────────
def lsd(p, t): return np.sqrt(np.mean((p - t) ** 2, -1))
def dmr(h, q): return np.mean((np.sign(h) == np.sign(q)).astype(float), -1)
def cossim(h, q):
    return np.sum(h * q, -1) / (np.linalg.norm(h, axis=-1) * np.linalg.norm(q, axis=-1) + 1e-8)

@torch.no_grad()
def run(model, name, ds, meta, forward):
    """Returns dict of per-sample arrays for gauss and exact renderings."""
    preds, fcs, gs, qs, gauss_peq = [], [], [], [], []
    resp_fn = getattr(model, "peq_response", None) or getattr(model, "response")
    for b in ds.iter_batches(512, shuffle=False):
        out = forward(b)
        fc, g, q = out["fc"], out["gain"], out["q"]
        preds.append(out["pred_response_db"].cpu().numpy())
        gauss_peq.append(resp_fn(fc, g, q).cpu().numpy())
        fcs.append(fc.cpu().numpy()); gs.append(g.cpu().numpy()); qs.append(q.cpu().numpy())
    pred = np.concatenate(preds); gauss = np.concatenate(gauss_peq)
    FC, G, Q = map(np.concatenate, (fcs, gs, qs))
    rbj = rbj_db(FC, G, Q, meta["freqs"], getattr(resp_fn, "section_types", None))
    exact = pred - gauss + rbj          # swap surrogate for exact response, keep everything else
    res = {}
    for tag, P in (("gauss", pred), ("exact", exact)):
        heard = P - meta["room"]
        res[f"lsd_dual_{tag}"] = lsd(P, meta["dual"])
        res[f"lsd_pref_{tag}"] = lsd(heard, meta["pref"])
        res[f"dmr_{tag}"] = dmr(heard, meta["pref"])
        res[f"cos_{tag}"] = cossim(heard, meta["pref"])
    res["peq_rms_diff"] = lsd(gauss, rbj)                  # Gaussian PEQ vs RBJ PEQ, per sample
    res["gap"] = res["lsd_dual_exact"] - res["lsd_dual_gauss"]
    res["fc"], res["gain"], res["q"] = FC, G, Q
    res["pred_gauss"], res["pred_exact"] = pred, exact
    return res

def summarize(res, meta, label, split):
    row = dict(model=label, split=split, N=len(res["gap"]))
    for k in ("lsd_dual_gauss", "lsd_dual_exact", "lsd_pref_gauss", "lsd_pref_exact",
              "dmr_gauss", "dmr_exact", "cos_gauss", "cos_exact", "peq_rms_diff"):
        row[k] = float(res[k].mean())
    row["gap_mean"] = float(res["gap"].mean()); row["gap_median"] = float(np.median(res["gap"]))
    row["gap_p90"] = float(np.percentile(res["gap"], 90))
    row["frac_exact_worse_1dB"] = float((res["gap"] > 1.0).mean())
    Q, G = res["q"], res["gain"]
    row["frac_filters_Q<0.5"] = float((Q < 0.5).mean())
    row["frac_filters_|g|>6"] = float((np.abs(G) > 6).mean())
    row["frac_filters_Q<0.5_and_|g|>6"] = float(((Q < 0.5) & (np.abs(G) > 6)).mean())
    # per preset
    for m in np.unique(meta["mode"]):
        sel = meta["mode"] == m
        row[f"preset{int(m)}_gauss"] = float(res["lsd_dual_gauss"][sel].mean())
        row[f"preset{int(m)}_exact"] = float(res["lsd_dual_exact"][sel].mean())
    if "source" in meta:
        for s in ("BUT", "OpenAIR"):
            sel = meta["source"] == s
            if sel.any():
                row[f"{s}_gauss"] = float(res["lsd_dual_gauss"][sel].mean())
                row[f"{s}_exact"] = float(res["lsd_dual_exact"][sel].mean())
    return row

# ── model list ───────────────────────────────────────────────────────────────
reg = TF.build_registry()
rows = []

def std_forward(name, model):
    TF._REGISTRY_TARGET[name] = reg[name]["target"] if name in reg else "dual"
    return lambda b: TF.model_forward(name, model, b)

# A0 / A2 — three ±12 seeds
for regname, var, label in (("A0_Proposed", "A0", "A0 (proposed)"), ("A2_withPrefLoss", "A2", "A2 with preference loss")):
    for split, (ds, meta) in splits.items():
        seed_rows = []
        for s in SEEDS:
            ck = SAVE / f"{cname('g12_f16k', s, var)}.pt"
            if not ck.exists():
                print(f"[skip] {ck} not found"); continue
            model = reg[regname]["model"]
            assert abs(model.gain_max - 12.0) < 1e-9, f"gain_max={model.gain_max} (≠12)"
            ckpt_io.load_into(model, ck, map_location=device, label=regname)
            model.to(device).eval()
            res = run(model, regname, ds, meta, std_forward(regname, model))
            r = summarize(res, meta, f"{label} s{s}", split); seed_rows.append(r); rows.append(r)
            print(f"{label} s{s} {split}: LSD_dual gauss {r['lsd_dual_gauss']:.3f} → exact {r['lsd_dual_exact']:.3f}  "
                  f"(gap mean {r['gap_mean']:+.3f}, p90 {r['gap_p90']:+.3f}; PEQ RMS diff {r['peq_rms_diff']:.3f} dB)")
            if var == "A0" and s == 7:
                np.savez_compressed(OUT / f"exact_biquad_persample_A0_s7_{split}.npz",
                                    freqs=meta["freqs"], **{k: v for k, v in res.items()}, dual=meta["dual"],
                                    room=meta["room"], pref=meta["pref"], mode=meta["mode"])
        if seed_rows:
            mean_row = {k: (float(np.mean([r[k] for r in seed_rows])) if isinstance(seed_rows[0][k], float) else seed_rows[0][k])
                        for k in seed_rows[0]}
            mean_row["model"] = f"{label} (3-seed mean)"
            mean_row["lsd_dual_exact_sd"] = float(np.std([r["lsd_dual_exact"] for r in seed_rows], ddof=1)) if len(seed_rows) > 1 else 0.0
            rows.append(mean_row)

# Option C (AC *_Biquad, ±12) — pred = room_corr + pref_curve + PEQ
for bname in BIQUAD_REGISTRY:
    ck = SAVE / f"{bname}_g12.pt"
    if not ck.exists():
        print(f"[skip] {ck} not found"); continue
    model = BIQUAD_REGISTRY[bname](gain_max=12.0).to(device).eval()
    ckpt_io.load_into(model, ck, map_location=device, label=bname)
    fwd = lambda b, m=model: m(b["features"], b["room_response"], b["mode_id"], b["band_gains"])
    for split, (ds, meta) in splits.items():
        res = run(model, bname, ds, meta, fwd)
        r = summarize(res, meta, f"{bname} (option C, ±12)", split); rows.append(r)
        print(f"{bname} {split}: gauss {r['lsd_dual_gauss']:.3f} → exact {r['lsd_dual_exact']:.3f} (gap {r['gap_mean']:+.3f})")

# E3 / E4 external baselines (5-band peaking head) — from staging dir if present
for regname in ("E3_Nercessian", "E4_Pepe"):
    ck = CKE / f"{regname}.pt"
    if not (CKE.is_dir() and ck.exists()):
        print(f"[skip] {ck} not found"); continue
    model = reg[regname]["model"]
    ckpt_io.load_into(model, ck, map_location=device, label=regname)
    model.to(device).eval()
    for split, (ds, meta) in splits.items():
        res = run(model, regname, ds, meta, std_forward(regname, model))
        r = summarize(res, meta, regname, split); rows.append(r)
        print(f"{regname} {split}: gauss {r['lsd_dual_gauss']:.3f} → exact {r['lsd_dual_exact']:.3f} (gap {r['gap_mean']:+.3f})")

# ── output ───────────────────────────────────────────────────────────────────
keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "split", "N"), k))
with open(OUT / "exact_biquad_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for r in rows: w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
(OUT / "exact_biquad_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

print("\n" + "=" * 100)
print(f"{'model':34}{'split':11}{'LSD gauss':>10}{'LSD exact':>10}{'gap':>8}{'p90':>8}{'PEQ Δ':>8}{'Q<.5&|g|>6':>11}")
print("-" * 100)
for r in rows:
    print(f"{r['model']:34}{r['split']:11}{r['lsd_dual_gauss']:>10.3f}{r['lsd_dual_exact']:>10.3f}"
          f"{r['gap_mean']:>+8.3f}{r['gap_p90']:>+8.3f}{r['peq_rms_diff']:>8.3f}{r['frac_filters_Q<0.5_and_|g|>6']:>11.3f}")
print(f"\nsaved: {OUT / 'exact_biquad_summary.csv'}, .json, per-sample npz (A0 s7)")