# Embedded deployment (STM32F405)

`onnx/` holds the exported graphs, `cmsis_mfcc/` the MFCC front-end,
`latency/` the DWT measurements and `build_reports/` the ST Edge AI Core
results behind Tables 6 and 7. `onnx/manifest.md` documents each graph's
inputs, outputs and the post-processing that runs on the host.

## PEQ section types (v1.2)

The graphs end at the PEQ parameters: A0, A2 and AC1–AC3 `*_Biquad` output
`room_corr[1,128]` and `fc/gain/q[1,7]`; E3/E4/E5 output `fc/gain/q[1,5]`.
The biquad coefficients are computed on the host with the Audio-EQ-Cookbook
closed forms, and since v1.2 the section type depends on the index:

| model | section 0 | sections 1–5 | section 6 |
|---|---|---|---|
| A0, A2, AC1–AC3 `*_Biquad` | low shelf | peaking | high shelf |
| E3, E4, E5 (5 sections) | peaking | peaking | — |

For a shelf, `gain` is the plateau gain (DC for the low shelf, Nyquist for the
high shelf) and `q` is the cookbook shelf Q. Every type is the same 5-coefficient
biquad, so the per-section cycle count is unchanged.

The device response is `room_corr` plus the cascade of these sections. In v1.2
the AC `*_Biquad` models no longer add `pref_curve` to their output (it remains
a network input), so step (1) in the `w7` post-processing note of
`onnx/manifest.md` is not part of their output path any more.
`code/exact_peq_response.py` is the reference for these magnitudes.

## Which checkpoints the `.onnx` files come from

The `.onnx` files here were exported from the v1.1 checkpoints. Re-exporting
from the v1.2 checkpoints gives graphs with the same operators, initializer
sizes and inputs/outputs for all thirteen files, so the flash/RAM footprint and
the NN-only cycle counts in Tables 6 and 7 apply to v1.2 unchanged. The weights,
and therefore the parameters a graph emits for a given input, are v1.1's.

INT8 accuracy does depend on the weights. With the v1.2 checkpoints
(`export_onnx.py` → `build_calib.py` → `quantize_ptq.py` → `eval_int8.py`,
test_synth, n = 3000) the INT8 − FP32 LSD is +0.050 dB for A0 (seed 7,
0.9235 → 0.9733), +0.049 for AC2, +0.052 for AC3 and below 0.003 for E3/E4;
cosine drops by at most 0.0016. `onnx/INT8_EVAL.md` gives both runs; the graphs
shipped in `onnx/` stay the v1.1 exports, because the ST reports and the latency
log were made from those exact files.
