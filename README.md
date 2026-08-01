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
| **Not included** | the audio dataset and the trained checkpoints. Every script takes their locations as command-line arguments, and `checkpoints_manifest.json` / `dataset_manifest.json` describe what is published alongside the release |

## Layout

```
code/            model, training, evaluation and table/figure scripts
figures/         F1–F6, fig_ac_fitting, fig_response_overlay, param_dist (PNG + PDF)
tables/          consolidated tables (T1–T7) and per-metric CSVs
results_json/    raw metric outputs (per-seed summaries, paired/track statistics,
                 saturation distributions, response-overlay sample) and the
                 ac_fitting_*.csv inputs that Table 4 is assembled from
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

## Table mapping

| Repository file | Paper |
|---|---|
| `tables/table2_revision_synth.csv` | Table 1 (main, synthetic) and Table 2 (ablation) |
| `tables/table5_ood.csv` | Table 3 (out-of-distribution) |
| `tables/table6_biquad.csv`, `tables/T5_ac_fitting.csv` | Table 4 (biquad-constrained comparison) |
| `tables/table7_perceptual.csv` | Table 5 (perceptual proxy) |
| `tables/T2_paired_stats_3seed.csv`, `results_json/paired_stats_3seed_*.json` | paired diagnostics quoted in Section 3.2 |
| `tables/T3_tracklevel_3seed.csv`, `results_json/track_stats_3seed_*.json` | track-level aggregate quoted in Section 3.2 |
| `tables/T4_saturation.csv`, `figures/F1_saturation_A0.*` | gain-saturation statistics (Section 3.4) |
| `embedded/` | Tables 6 and 7 (deployment footprint and on-chip latency) |

### How Table 4 is put together

`table6_biquad.py` performs no computation. It reads already-generated files and
assembles them, so it runs without a GPU, a dataset or a checkpoint:

| column | comes from |
|---|---|
| Option C rows — LSD, CI, vs A0, repr. penalty | `results_json/ac_biquad_table.json` (`ac_biquad_table.py`) |
| `% < JND` | `tables/table7_perceptual.csv` (`table7_perceptual.py`) — the only place this quantity is computed |
| `dense AC2 (raw)` | `tables/table2_revision_synth.csv`, the same figure as the AC2 row of Table 1 |
| Option A, ±12 and ±6 | `results_json/ac_fitting_A_g12.csv`, `ac_fitting_A_g6_orig.csv` (`ac_fitting_A.py`) |
| Option D | `results_json/ac_fitting_D_naive7pt.csv` (`ac_fitting_C.py`) |
| A0 reference row | `a0_reference` in `results_json/ac_biquad_table.json` |

The A0 reference interval `[1.083, 1.107]` is the bootstrap (`n_boot=2000`,
`seed=42`) over the per-sample LSD averaged across the three training seeds
(N=3000). Pooling the three seeds into one N=9000 resample instead treats a seed
as if it were a sample and narrows the interval to `[1.086, 1.104]`; that is not
what the paper reports, and no table here carries it.

**Option A ±6 and Option D come from the pre-revision run.** Option D samples the
target response at seven points and never evaluates a trained model, so the
Stage-B gain bound plays no part in it and there is nothing for the ±12 relaxation
to change. The Option A ±6 row is the deliberate control against which the ±12
refit is read, and is labelled `orig` for that reason. Everything else in the
table was recomputed under the ±12 bound.

`% < JND` compares `|LSD_model − LSD_A0|` against 0.5 dB with the ±12 A0 as the
reference, so it is defined only for ±12 configurations. In
`tables/T5_ac_fitting.csv` the ±6 rows are therefore blank in that column; they
are kept because their `gain_sat(self)%` and `gain>6%` columns are the evidence
that the narrower bound was binding.

Specific figures quoted in Section 3.2 come from
`tables/T2_paired_stats_3seed.csv`: the `|d_z| = 1.5–4.4` range and the
`78–100 %` win rates are the A1/A3/E3/E4 rows, and `about 0.23 dB lower LSD`
is the **AC1–AC3** block (LSD deltas +0.242 / +0.235 / +0.225, mean 0.234, in
favour of the dense variants). Note that the A2 row happens to have the same
magnitude with the opposite sign (−0.234, in favour of A0); the two are easy to
confuse.

Real-time factors are machine-specific and are reported only in the paper
(AMD Ryzen 7 9800X3D, single thread; RTF = inference latency / 4000 ms), so the
CSVs here carry no RTF column. On-chip latency for the embedded target is in
`embedded/latency/MEASUREMENT_LOG.md`.

## Sign conventions

Sign conventions differ between the two families of paired statistics:

- `tables/T2`, `T3` and `results_json/*_3seed_*.json` (`paired_stats.py`,
  `track_stats.py`): **Δ = A0 − comparison**, so a negative value means A0 is
  better.
- `results/track_level/**` (`export_track_level_predictions.py`):
  **mean_diff = comparison − baseline (A0)**, so a positive value means A0 is
  better.

Check the field name before comparing. The two families also differ in seed
handling: T2/T3 aggregate three seeds, while `results/track_level/` uses a
matched single seed (42). Figures quoted in the paper come from the three-seed
family — see the table mapping above.

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

Uncertainty is reported as a 95 % bootstrap CI for single-seed rows and as
±1 seed standard deviation for the three-seed A0/A2 rows, matching the paper.
The A0 confidence interval quoted in the paper's Table 4, [1.083, 1.107], is the
bootstrap over the per-sample three-seed mean and is the one carried in
`tables/table6_biquad.csv`.

Each track-level JSON records the seed in a `baseline_seed` field. The sign
convention is `mean_diff = candidate − baseline`, with A0 as the baseline.

## Reproducing

`scripts/run_all_refresh.sh` (or `.ps1` on Windows) runs every generator behind
the paper's tables and figures, in order, and stops at the first failure.

```bash
DATA_DIR=/path/to/dataset_v3 \
CKPT_DIR=/path/to/checkpoints_original \
REV_CKPT_DIR=/path/to/checkpoints_revision \
    bash scripts/run_all_refresh.sh
```

Output goes to `reruns/refresh/` and nothing already committed is touched. To
compare against what is published here, diff that directory against `tables/`,
`figures/` and `results_json/`; to overwrite them, add `PUBLISH=1`. That copy is
the step between `code/results/` and the published directories, and it is part
of the script rather than something to do by hand.

Environment: the pinned versions in `requirements.txt` are the ones this was
verified with — Python 3.13.5 on Windows 11, CUDA 12.8 on an RTX 3060 Ti. Every
script falls back to CPU.

`figures/fig_ac_fitting_AC2_GRU.*` is the one figure the script cannot rebuild
from what is shipped: it plots per-sample arrays written by `ac_fitting_A.py` /
`ac_fitting_C.py`, which are refit and retrain runs. Point `STAT_DIR` at their
`.npy` dumps to regenerate it; otherwise the script says it is skipping it.

Individual entry points. All of them take `--data_dir`, `--ckpt_dir`,
`--rev_ckpt_dir`, `--eval_ckpt_dir` and `--out_dir`; the defaults match this
repository's layout.

| script | produces |
|---|---|
| `run_gain_freq_ablation.py` | A0/A2 training and the gain/centre-frequency ablation (multi-seed) |
| `verify_patch.py`, `verify_biquad_patch.py` | gain-bound assertion gates (see below) |
| `table2_revision.py` … `table7_perceptual.py` | main, ablation, paired, OOD and perceptual tables |
| `ac_biquad_table.py`, `ac_fitting_A.py`, `ac_fitting_C.py` | biquad-constrained architecture comparison |
| `table6_biquad.py` | assembles Table 4 from the files above; no GPU or checkpoint needed |
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
assertions if you adapt the code. `checkpoints_manifest.json` records the bound
each published checkpoint was trained under, because the file itself does not.

### Failing rather than degrading

A wrong path used to be survivable. A missing dataset fell back to synthetic
curves, a missing checkpoint left the model randomly initialised, and either way
a complete set of result files came out looking normal. Both now raise:

- the path options are checked before any work starts;
- `ckpt_io.load_into` refuses a checkpoint whose keys do not match the model,
  reporting which keys disagree (every checkpoint here loads with none missing
  and none unexpected);
- a `(config, seed)` cell with no checkpoint stops the run instead of quietly
  shrinking the seed aggregate — pass `--allow_missing_ckpt` to evaluate only
  the cells that exist;
- `--dry_run` output goes to `dry_run_outputs/` with a `DRY_RUN_` prefix, so it
  cannot be mistaken for or overwrite a real result.

E1 (No Processing), E2 (Static Mode EQ) and E6 (DSP Analytical) are analytical
and have no checkpoint; they are exempt by name, not by silence.

## Checkpoints and dataset

Neither is in this repository. `checkpoints_manifest.json` and
`dataset_manifest.json` describe them.

The checkpoints are published as two directories, because the same basename can
mean two different files: `checkpoints_original/A0_Proposed.pt` is the
pre-revision model, while the ±12 dB seed-7 model that the single-checkpoint
figures use is `checkpoints_revision/A0_g12_f16k_s7.pt`.

```
checkpoints_original/    15 files, pre-revision
checkpoints_revision/    12 files, retrained under the ±12 dB bound
```

The working tree applied the "representative seed 7" convention by substituting
files inside a staging directory, which is what produced the name collision.
The release states it as the `evaluation_staging` mapping in the manifest
instead, and does not ship the staging directory. Checkpoints are published
byte-for-byte as trained — optimiser state and history included — so that their
SHA-256 in the manifest identifies exactly what was evaluated.

The audio is third-party (FMA, BUT ReverbDB, OpenAIR) and is not redistributed.
`dataset_generator_v4_tracklevel.py` sorts its file list before splitting, so a
different local corpus produces a different split even at seed 42;
`dataset_manifest.json` therefore records the per-split track and RIR lists
along with the full generation config.

## Line endings

Text files are normalised to LF (`.gitattributes`, `* text=auto`); the
normalisation touched files whose content is otherwise unchanged. When comparing
this release against an earlier tag, use `git diff -w` so that the whitespace
change does not obscure the substantive differences.

## Licence

See `LICENSE`.
