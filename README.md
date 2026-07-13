# PHONIX Supplementary — Revision: per-section gain bound ±6 → ±12 dB

Major-revision update to the JAES submission. Reviewer R1-8 noted the Stage-B
hyperparameter bounds appeared arbitrary; we verified that the per-section gain
bound of **±6 dB was binding** (57.8 % of learned gains saturated) and **relaxed
it to ±12 dB** (center frequency kept at [80, 16000] Hz). All A0/A2 results were
re-trained; gain-independent models (E1–E6, A1, A3, AC1–AC3 dense) were re-evaluated
unchanged.

**Loss function and architecture are unchanged** — only the output gain bound was relaxed.

## Headline changes (±6 → ±12)
- A0 synthetic LSD **1.442 → 1.095 dB**, real-RIR LSD **1.941 → 1.792 dB** (both improve → not overfitting).
- gain saturation **57.8 % → 16.6 %**; 72.8 % of learned gains now exceed 6 dB (the old bound was suppressing them).
- A0 vs retrained biquad-AC gap **~0.40 → ~0.08 dB** (deployment parity under identical 7-band ±12 constraint).
- synthetic→real domain gap 0.499 → 0.697 dB — **within the AC variant range (0.646–0.706)**, no longer "smallest" (reframed as deployment parity).
- A0 vs A2 (preference-loss ablation): comparable mean, but A0 is **3.2× more stable across seeds** (σ 0.14 vs 0.45).

## Reproducibility notes
- A0 / A2 reported over **3 seeds (42, 123, 7)**; all other models retain the original single-seed (42) checkpoints (seed asymmetry noted in captions).
- AC_Biquad follows the original single-seed + bootstrap-CI methodology.
- **Gain-bound trap**: A0/A2 must be instantiated with `gain_max=12.0` when loading the ±12 checkpoints; a default `gain_max=6.0` instance silently clamps the output. `verify_patch.py` / `verify_biquad_patch.py` assert this.
- Model checkpoints and the dataset are **not** included here (size); the evaluation scripts reference them by path.

## Layout
```
code/          patched core (model, train_full, experiments, arch_biquad, ac_fitting_*) + new analysis scripts
figures/       F1–F6 + fig_ac_fitting + fig_response_overlay + param_dist  (PNG + vector PDF)
tables/        T1–T7 (consolidated) + table2–7 (per-metric CSVs)
results_json/  raw metric outputs (gain_freq_summary, paired/track stats, ac_biquad, param_dist, fig_overlay_sample)
docs/          inline_number_changes.md (manuscript value diffs), loss_equations.md (code-faithful loss)
```

## Key scripts
| script | produces |
|--------|----------|
| `run_gain_freq_ablation.py` | ±12 A0/A2 training + gain/freq ablation (multi-seed) |
| `verify_patch.py`, `verify_biquad_patch.py` | gain-bound assertion gates |
| `table2_revision.py` … `table7_perceptual.py` | Tables 2–7 (synthetic/real, paired, track-level, OOD, perceptual) |
| `ac_biquad_table.py`, `ac_gap_eval.py` | biquad-constrained comparison (§4.5) |
| `param_dist_gain_freq.py` | gain/fc saturation two-criteria (Fig 1 / F1) |
| `consolidate.py` | colorblind/B&W-safe figures F1–F6 |
| `extract_overlay_sample.py`, `make_overlay_figure.py` | median-sample response overlay (fig_response_overlay) |

See `docs/inline_number_changes.md` for the full list of manuscript numeric values that change under ±12.
