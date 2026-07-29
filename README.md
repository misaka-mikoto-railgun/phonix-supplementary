# PHONIX — Supplementary material

Code, generated tables and figures, and embedded-deployment artefacts for the
JAES engineering report on real-time adaptive parametric equalisation for joint
room correction and tonal-shape tracking.

This release describes the system as reported: a two-stage model whose Stage-B
parametric-EQ head emits seven biquad sections with a **per-section gain bound of
±12 dB** and centre frequencies in **[80, 16000] Hz**.

> Manuscript details (ID, DOI) are filled in on acceptance.

## What is and is not included

| | |
|---|---|
| **Included** | model and training code, evaluation and table/figure scripts, generated tables (CSV) and figures (PNG + vector PDF), raw metric outputs (JSON), track-level paired statistics, ONNX exports and the embedded MFCC front-end |
| **Not included** | the audio dataset and the trained checkpoints (size). Scripts take their locations as command-line arguments |

## Layout

```
code/            model, training, evaluation and table/figure scripts
figures/         F1–F6, fig_ac_fitting, fig_response_overlay, param_dist (PNG + PDF)
tables/          consolidated tables (T1–T7) and per-metric CSVs
results_json/    raw metric outputs (per-seed summaries, paired/track statistics,
                 saturation distributions, response-overlay sample)
results/
  track_level/   per-track paired statistics for three evaluation splits
docs/            loss_equations.md — the training loss, transcribed from the code
embedded/
  onnx/          ONNX exports (opset 13, FP32 + INT8) and the export/quantisation scripts
  cmsis_mfcc/    CMSIS-DSP MFCC front-end and its constant generator
  latency/       DWT benchmark harness and MEASUREMENT_LOG.md
  build_reports/ ST Edge AI Core validation results
scripts/         run_all_refresh.{sh,ps1} — regenerate tables, figures and track-level exports
```

## Dataset generation — note on the `dataset_v3` path name

The dataset directory is called `dataset_v3` throughout the scripts
(`--data_dir ./data/dataset_v3`), but it is produced by
**`code/dataset_generator_v4_tracklevel.py`**. The v4 generator writes the same
directory layout as the earlier v3 generator and additionally stores per-clip
`track_id`, `room_id`, `pair_id` and clean (pre-convolution) features, which the
track-level analyses require. The path name was kept for continuity; there is no
separate v3 dataset, and the v3 entry point has been removed from this release.

```
python code/dataset_generator_v4_tracklevel.py --fma_dir ./fma_audio --output_dir ./data/dataset_v3
```

Splits: `train`, `val`, `test_synth`, `test_real` (BUT ReverbDB + OpenAIR),
`paired_mode_test`.

## Seeds

A0 and A2 were trained with three seeds (42, 123, 7); every other model retains
its original single-seed (42) checkpoint. Headline numbers are the three-seed
aggregate.

**Track-level comparisons** use the seed-42 checkpoints for both A0 and A2 so that
every paired comparison is seed-matched with the single-seed candidates
(AC1–AC3, E1–E6, A1, A3); pairing a multi-seed baseline against single-seed
candidates would reintroduce the between-run variation that this aggregate is
meant to control for. Seed 42 also matches the convention of the earlier ±6 dB
files. Note that seed 42 is A0's best of the three revision seeds (0.999 vs. the
3-seed mean of 1.095) and A2's best (1.031 vs. 1.329); at this matched seed the
two are within 0.03 dB, consistent with the near-parity in mean accuracy reported
in the manuscript. Headline figures use the 3-seed aggregate.

Each track-level JSON records the seed in a `baseline_seed` field. The sign
convention is `mean_diff = candidate − baseline`, with A0 as the baseline.

## Reproducing

```bash
DATA_DIR=/path/to/dataset_v3 CKPT_DIR=/path/to/checkpoints bash scripts/run_all_refresh.sh
```

Individual entry points:

| script | produces |
|---|---|
| `run_gain_freq_ablation.py` | A0/A2 training and the gain/centre-frequency ablation (multi-seed) |
| `verify_patch.py`, `verify_biquad_patch.py` | gain-bound assertion gates (see below) |
| `table2_revision.py` … `table7_perceptual.py` | main, ablation, paired, OOD and perceptual tables |
| `ac_biquad_table.py`, `ac_fitting_A.py`, `ac_fitting_C.py` | biquad-constrained architecture comparison |
| `param_dist_gain_freq.py` | gain / centre-frequency saturation statistics |
| `consolidate.py` | figures F1–F6 (colour-blind safe, legible in greyscale) |
| `extract_overlay_sample.py`, `make_overlay_figure.py` | median-sample frequency-response overlay |
| `export_track_level_predictions.py` | per-track paired statistics |
| `embedded/onnx/export_onnx.py`, `verify_fp32_weights.py` | ONNX export and weight-size check |

### Gain-bound caveat

`gain_max` is not stored in the checkpoint. A model instantiated with the wrong
bound loads the weights without error and then silently clamps its output, so a
±12 dB checkpoint evaluated through a ±6 dB instance produces wrong numbers.
All entry points here instantiate the model explicitly, and `verify_patch.py` /
`verify_biquad_patch.py` assert the bound before any evaluation. Keep those
assertions if you adapt the code.

## Line endings

Text files are normalised to LF (`.gitattributes`, `* text=auto`); the
normalisation touched files whose content is otherwise unchanged. When comparing
this release against an earlier tag, use `git diff -w` so that the whitespace
change does not obscure the substantive differences.

## Licence

See `LICENSE`.
