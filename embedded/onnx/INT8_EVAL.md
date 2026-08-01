# INT8 PTQ accuracy cost — test_synth (n = 3000)

QDQ INT8 ONNX (run under onnxruntime) against the FP32 checkpoint, on the same
test split, with the same metrics and the same post-processing.

- **LSD** — against the dual target: per sample, `sqrt(mean((pred − dual_target)^2))`, then averaged. Lower is better.
- **CosSim** — heard response (`pred − room_target`) against `pref_target`. Higher is better.
- `pred` is reconstructed from the graph boundary outputs (`room_corr`, `fc`, `gain`, `q`) with the same expression the FP32 path uses.

| variant | fp32 LSD | int8 LSD | ΔLSD | ΔLSD % | fp32 cos | int8 cos | Δcos |
|---|---|---|---|---|---|---|---|
| A0 | 1.0279 | 1.1128 | +0.0850 | +8.3 % | 0.9772 | 0.9737 | −0.0035 |
| AC2 (GRU) | 1.0172 | 1.0561 | +0.0389 | +3.8 % | 0.9798 | 0.9783 | −0.0015 |
| AC3 (Conformer) | 1.0103 | 1.0393 | +0.0291 | +2.9 % | 0.9802 | 0.9788 | −0.0014 |
| E3 (Nercessian) | 5.9641 | 5.9657 | +0.0016 | +0.0 % | 0.0108 | 0.0104 | −0.0004 |
| E4 (Pepe) | 5.8086 | 5.8085 | −0.0001 | −0.0 % | 0.0319 | 0.0325 | +0.0006 |

## Cross-check of the FP32 baseline against the reported figures

- A0: 1.028 here is the single seed-7 checkpoint on test_synth; the reported
  three-seed figure is 1.095 ± 0.142 dB (mean ± sample standard deviation over
  seeds 42/123/7), validation 1.43.
- E3 / E4: 5.96 / 5.81 on test, against 5.94 / 5.78 reported on validation.
- AC2 / AC3 biquad at ±12 dB: 1.02 / 1.01, against ≈1.01 from
  `ac_biquad_table.py`.

The FP32 path therefore agrees with the reported baselines, so each Δ above is
the quantisation effect alone, measured inside one pipeline.

## Reading the numbers

- **A0** loses the most in relative terms (+8.3 %, +0.085 dB), but at an absolute
  LSD of 1.11 it remains far ahead of every baseline, and −0.0035 in cosine
  similarity is negligible. Its dense room-correction head feeding the PEQ stage
  makes it the most quantisation-sensitive of the variants.
- **AC2 / AC3**: +2.9 to +3.8 % (+0.03 dB), cosine −0.0015. Effectively lossless.
- **E3 / E4**: no measurable change — at LSD ≈ 6 and cosine ≈ 0.01 these models
  are already far from the target, so quantisation has nothing left to degrade.
  The rows are reported for completeness rather than as evidence.
- Overall: INT8 post-training quantisation costs 0.03–0.09 dB LSD and under
  0.004 cosine on the proposed model and the main variants. Together with the
  flash-fit result for the heavier variants under INT8, that is what the
  appendix draws on.

## Why INT8 is slower on this part

Accuracy is not the only thing quantisation changes, and on the F405 it does not
buy speed: A0 measures 288.3 ms under INT8 against 251.1 ms in FP32
(`embedded/latency/MEASUREMENT_LOG.md`). The operation-type tables say why, and
they split the models into two groups.

| variant | s8 | f32 | MACC vs its FP32 graph |
|---|---|---|---|
| E3 INT8 | 99.7 % | 0.2 % | 138,655 vs 139,523 (−0.6 %) |
| E4 INT8 | 100.0 % | 0.0 % | 1,036,567 vs 1,044,003 (−0.7 %) |
| A0 INT8 | 15.7 % | 80.1 % | 4,031,544 vs 3,901,505 (**+3.3 %**) |
| AC2 INT8 | 11.4 % | 85.5 % | 5,627,201 vs 5,493,259 (**+2.4 %**) |
| AC3 INT8 | 15.8 % | 83.4 % | 44,699,530 vs 44,383,435 (**+0.7 %**) |

The external baselines are plain feed-forward stacks and quantise essentially
completely, and their MACC falls. The proposed model and the architecture
comparators do not: they keep 80–86 % of their operations in float, so the graph
is mixed precision and pays for a quantise/dequantise pair at every boundary.
The MACC count rises instead of falling, and the Cortex-M4F has no SIMD path
that would repay the integer kernels that do exist. For AC3 the attention
MatMuls in particular are left in float.

So the accuracy figures above and the latency are answering different questions.
INT8 is worth keeping here for flash footprint — which is what decides whether
AC2 and AC3 fit at all — and not as a latency measure.
`[file: ../build_reports/ST_VALIDATION.md]` collects the per-variant memory and
operation-type figures; the reports themselves are `analyze_*.txt` beside it.

## Notes

- The large absolute LSD of E3 / E4 is those baselines' own performance, as
  reported; what this table measures is Δ.
- INT8 graphs kept: `a0_int8`, `e3_nercessian_int8`, `e4_pepe_int8`,
  `ac2_gru_biquad_int8`, `ac3_conformer_biquad_int8`.
- To reproduce: `python eval_int8.py` (per-sample onnxruntime execution against
  the FP32 PyTorch model).
