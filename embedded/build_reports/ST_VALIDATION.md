# ST Edge AI Core 2.2.0 — validation results (STM32F405 latency benchmark)

Tool: X-CUBE-AI 10.2.0 (= ST Edge AI Core v2.2.0-20266), the bundled
`stedgeai.exe`, `--target stm32`. This document is maintained by hand and is
separate from `manifest.md`, which is overwritten by every export.

**Target of this export**: opset 13 (`embedded/onnx/`, manifest generated
2026-06-22T20:40:02).

**Target device**: STM32F405RGT6 — 1024 KiB flash, 192 KiB total SRAM
(128 KiB main SRAM = 112 KiB SRAM1 + 16 KiB SRAM2, contiguous and
DMA-accessible, plus 64 KiB CCM on the D-bus with no DMA access; the separate
4 KiB backup SRAM is not counted in the 192 KiB).

**Source labels**: `[tool]` is a value read out of an `stedgeai analyze`
report that this repository carries. `[session record]` was read during a
measurement session and has no report file here — including the AC3 RAM
breakdown and the linker RAM region, which would become `[file:]` if the
`analyze` reports and the `MEMORY` block of `STM32F405RGTX_FLASH.ld` were
committed to this directory. See `embedded/latency/MEASUREMENT_LOG.md`.

## Final state — 8/8 analyze + generate PASS on the FP32 graphs (AC1 as the statically unrolled graph)

The RAM columns are separated because they are not interchangeable.
**Activations** is the tensor arena; **Runtime RAM** is what the X-CUBE-AI
runtime needs alongside it; **Total RAM** is the `Total Ram:` line in the report
footer and is the figure a link has to fit. Reporting activations alone
understates the requirement by a factor of two to four.

| variant | onnx | import (native) | analyze | flash weights | build flash | fits 1024 KiB | MACC (tool) | Activations | Runtime RAM | Total RAM | validate (c-model vs ref) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_Proposed | a0.onnx | PASS | PASS | 804 KiB `[session record]` | 896 KB `[session record]` | ✓ (78 %) | 3,901,505 `[tool]` | 27.05 KiB `[tool]` | 54.88 KiB `[session record]` | 81.93 KiB `[session record]` | cos ≈ 1.0, nse ≈ 1.0 (rmse ≤ 8e-4) |
| A2_withPrefLoss | a2.onnx | PASS | PASS | 804 KiB `[session record]` | 896 KB `[session record]` | ✓ (78 %) | 3,901,505 `[tool]` | 27.05 KiB `[tool]` | (≡ A0) | (≡ A0) | same architecture as A0 |
| E3_Nercessian | e3_nercessian.onnx | PASS | PASS | — | — | ✓ | 139,523 `[tool]` | 2.00 KiB `[tool]` | — | — | — |
| E4_Pepe | e4_pepe.onnx | PASS | PASS | — | — | ✓ | 1,044,003 `[tool]` | 18.50 KiB `[tool]` | — | — | — |
| E5_Sequential | e5_sequential.onnx | PASS | PASS | — | — | ✓ | 139,523 `[tool]` | 2.00 KiB `[tool]` | — | — | — |
| AC1_BiLSTM_Biquad | ac1_bilstm_biquad.onnx | **FAIL**³ → unrolled graph PASS¹ | PASS¹ | ~940 KiB `[session record]` | ~1.0–1.1 MB `[session record]` | ✗ over | 4,815,819 `[tool]` | 25.94 KiB `[tool]` | — | — | room_corr cos ≈ 0.93²; fc/gain/q cos ≥ 0.9996 |
| AC2_GRU_Biquad | ac2_gru_biquad.onnx | PASS | PASS | 1.02 MiB `[session record]` | 1.12 MB `[session record]` | ✗ over | 5,493,259 `[tool]` | 25.94 KiB `[tool]` | — | — | — |
| AC3_Conformer_Biquad | ac3_conformer_biquad.onnx | PASS | PASS | 1.58 MiB (+58 %) `[session record]` | — | ✗ over | 44,383,435 `[tool]` | 96.94 KiB `[tool]` | 103.79 KiB (derived) | 200.73 KiB `[session record]` | — |
| AC3_Conformer_Biquad **INT8** | ac3_conformer_biquad_int8.onnx | PASS⁴ | PASS⁴ | 667.1 KiB `[session record]` | 910.8 KiB `[session record]` | ✓ flash | — | 45.41 KiB `[session record]` | 139.64 KiB `[session record]` | 185.05 KiB `[session record]` | — |

Blank Runtime/Total cells are not zero — the footer figure was not captured for
those variants in the session, and nothing here reconstructs it. AC3 FP32's
runtime is the arithmetic difference of its two recorded figures.

**The AC3 INT8 row is the one Table 6 footnote e rests on.** Its total AI RAM of
185.05 KiB overflows the 112 KiB linker RAM region the build declares
(`RAM (xrw) : LENGTH = 112K`) by ≈ 78 KB once the application's own ~5 KiB is
counted; the derivation is in `embedded/latency/MEASUREMENT_LOG.md`. The overrun
is in the runtime, not the arena: at 45.41 KiB the activations fit easily, while
the runtime needs 139.64 KiB — 2.5× A0's — because the QDQ graph carries
per-tensor quantisation tables and mixed-precision conversion buffers.

¹ AC1 here is the statically unrolled LSTM graph (see §AC1). The analyze PASS in
the table is the state **after** unrolling.

² Float accumulation difference across the 32-step recursion, on random
validation input. It does not bear on latency, which is what this export is
for; the LSD accuracy reported in the paper is measured in PyTorch.

³ The `nn.LSTM` original fails at the ST import stage (`dl_remapping`). See §AC1.

⁴ The AC3 INT8 import and analyze complete, but the report carries four
E-level errors on the attention MatMul conversion (I/O 0 and 1). The heading's
"8/8 PASS" is about the FP32 graphs; this row is not covered by it, and the row
is included because Table 6 footnote e depends on its RAM figures rather than on
a clean conversion. `[session record]`

**On the fit column**: the flash figure from `analyze` is essentially weights,
while an actual build adds roughly 100 KiB of runtime and HAL. **Fit is judged on
build flash**, cross-checked on two variants (A0 804 KiB → 896 KB, AC2 1.02 MiB →
1.12 MB). `[session record]`

**Total RAM is still not the application's footprint**: it is what the AI model
needs, before the application's own stack, heap and HAL static data. Those add
roughly 5 KiB in this project, which is why the AC3 INT8 shortfall against the
112 KiB region works out at ≈ 78 KB rather than 73.05 KB.

## Transformations applied for ST compatibility (all parity-preserving; weights unchanged)

- **opset 13**: `nn.LayerNorm` decomposed into primitive operators — 2.2.0 has no
  fused `LayerNormalization`.
- **mode embedding → one-hot matmul**: `nn.Embedding` (Gather) had the embedding
  table's leading dimension (4) mistaken for a batch dimension, so it is replaced
  by `one-hot @ weight`.
- **pref_curve interpolation → constant matmul**: removes `searchsorted`, which
  needs opset 17.
- **Pad folded into Conv.pads**: works around a standalone-Pad codegen bug
  (`_Pad_output_0_value_data[]={[]}`). Causal `pads = [(k-1)*dilation, 0]`.
- **AC2 two-layer GRU → two one-layer GRUs in series**: avoids a failure to
  identify the stacked-GRU transpose.
- **ORT BASIC constant folding on**, with
  `disabled_optimizers=[LayerNormFusion family, MatMulAddFusion]`. Disabling
  MatMulAddFusion stops the unrolled LSTM from fusing a variable tensor into a
  Gemm bias; it has no effect on the other models.

## AC1 — static BiLSTM unrolling (the substantive one)

The `nn.LSTM` version cannot be imported: ST fails to remap the transpose from
TCN (Conv1d) output to LSTM input. The original bidirectional form, a
unidirectional decomposition, sequence-first layout, a barrier, and the ONNX
simplifier all fail the same way. Observed error `[session record]`:

```
INTERNAL ERROR: _m_lstm_f_LSTM_output_0_in_transpose of type Transpose
has not parameter dl_remapping
```

**Isolating the cause (control experiment)** `[session record]`

- control `Linear → bi-LSTM`: **PASS** — bi-LSTM itself imports
- this model `TCN (Conv1d) → LSTM`: **FAIL**

so the failure is not the LSTM operator but the missing remapping for the
transpose between Conv1d output and LSTM input.

**Fix — unroll at the fixed seq_len = 32**: the recursion is expanded into
Gemm/Sigmoid/Tanh/Mul/Add and the LSTM operator disappears. The F405 has no LSTM
accelerator and would execute it this way regardless, so the latency remains
representative. Implementation constraints imposed by ST:

- cell gates as `addmm(const_bias, cat([x_t, h]), [W_ih|W_hh]^T)` — a constant
  bias only, no variable-tensor Add, since ST does not support a multi-dimensional
  or variable Gemm bias;
- 2-D Gemm throughout (no 3-D), and batch-agnostic `h`/`c` initialisation via
  `x.new_zeros(x.shape[0], H)`.

Self-check against the original BiLSTM: error ≈ 3e-8, i.e. bit-identical.

## Notes

- The `_m_Mul_*_0` output labels in the report are internal `stedgeai` node
  names. The ONNX files' own outputs are `room_corr` / `fc` / `gain` / `q`. When
  integrating the generated C, the output order is
  `[room_corr(128), fc(7), gain(7), q(7)]`; the E-series models use `[fc, gain, q]`
  with 5 bands.
- `duration ms/sample` in the report is a host (x86) figure. The F405 latency is
  measured on the board with the DWT cycle counter.
- FP32 weight sizes are reproducible as `params × 4 / 1024`; see
  `embedded/onnx/verify_fp32_weights.py`.
