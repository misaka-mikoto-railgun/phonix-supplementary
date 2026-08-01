# ST Edge AI Core 2.2.0 — analyze results for the STM32F405 deployment

Tool: X-CUBE-AI 10.2.0 (= ST Edge AI Core v2.2.0-20266), the bundled
`stedgeai.exe`, `analyze --target stm32f4 --compression none`. This document is
maintained by hand; `../onnx/manifest.md` is overwritten by every export.

**Target device**: STM32F405RGT6 — 1024 KiB flash, 192 KiB total SRAM
(128 KiB main SRAM = 112 KiB SRAM1 + 16 KiB SRAM2, contiguous and
DMA-accessible, plus 64 KiB CCM on the D-bus with no DMA access; the separate
4 KiB backup SRAM is not counted in the 192 KiB).

**The linker region is 112 KiB, not 192.** The CubeMX project declares
`RAM (xrw) : LENGTH = 112K` — SRAM1 alone — with CCM as a separate region at a
disjoint address, so it cannot back the same allocation. The declaration is in
`STM32F405RGTX_FLASH.ld.MEMORY`, and that is the budget the fit column is
judged against.

**Dates.** The on-board latency figures were measured on 2026-06-23. The ONNX
graphs were regenerated on 2026-07-29, and the twelve `analyze` reports carried
here were produced on 2026-08-02 against those regenerated graphs. Anything
marked `[tool]` comes from the latter and may differ slightly from a June note.

**Source labels**: `[tool]` is read out of one of the `analyze_*.txt` files in
this directory — each an excerpt of the report header, the operation-type table
and the memory footer, with the per-layer dump removed for size.
`[session record]` was read during a measurement session and has no report file
here; see `../latency/MEASUREMENT_LOG.md` for why the on-board figures leave no
artefact.

## Memory, per variant

Three RAM figures, because they are not interchangeable. **Activations** is the
tensor arena. **Runtime** is what the X-CUBE-AI runtime needs beside it.
**Total** is the report's own `TOTAL` line and is the number a link has to fit;
quoting activations alone understates the requirement by two to ten times.

`Fit RAM` is against the 112 KiB (114,688 B) linker region, before the
application's own stack and heap (0x800 each in this project).

| variant | format | MACC | weights (B) | flash total (B) | fit flash | activations (B) | runtime (B) | **RAM total (B)** | fit RAM |
|---|---|---|---|---|---|---|---|---|---|
| A0 / A2 | float | 3,901,505 | 823,080 | 905,127 | ✓ | 27,704 | 56,192 | **83,896** | ✓ |
| A0 INT8 | ss/sa per channel | 4,031,544 | 467,884 | 626,617 | ✓ | 25,856 | 80,628 | **106,484** | ✓ |
| E3 (Nercessian) | float | 139,523 | 553,140 | 568,651 | ✓ | 2,048 | 4,624 | **6,672** | ✓ |
| E3 INT8 | ss/sa per channel | 138,655 | 140,704 | 171,051 | ✓ | 3,584 | 5,216 | **8,800** | ✓ |
| E4 (Pepe) | float | 1,044,003 | 201,524 | 220,059 | ✓ | 18,944 | 5,576 | **24,520** | ✓ |
| E4 INT8 | sa/sa per tensor | 1,036,567 | 51,472 | 84,043 | ✓ | 12,032 | 5,936 | **17,968** | ✓ |
| E5 (Sequential) | float | 139,523 | 553,140 | 568,651 | ✓ | 2,048 | 4,624 | **6,672** | ✓ |
| AC1 (BiLSTM, unrolled) | float | 4,815,819 | 971,692 | 1,277,193 | ✗ | 26,560 | 251,520 | **278,080** | ✗ |
| AC2 (GRU) | float | 5,493,259 | 1,072,056 | 1,164,688 | ✗ | 26,560 | 57,980 | **84,540** | ✓ |
| AC2 INT8 | ss/sa per channel | 5,627,201 | 680,312 | 832,410 | ✓ | 25,600 | 82,292 | **107,892** | ✓ |
| AC3 (Conformer) | float | 44,383,435 | 1,656,164 | 1,796,380 | ✗ | 99,264 | 106,288 | **205,552** | ✗ |
| AC3 INT8 | sa/sa per tensor | 44,699,530 | 683,084 | 932,644 | ✓ | 46,500 | 142,988 | **189,488** | ✗ |

All `[tool]`. Every row satisfies activations + runtime = total, which is how the
report footer is built. E5 and E3 share a graph and therefore a row of figures.
A2 is architecturally identical to A0.

**A0 is the only proposed-family variant that fits in FP32**, on both flash and
RAM, and it fits with the runtime already counted.

**AC1 fails on RAM as well as flash.** Its runtime alone is 251,520 B — more
than twice the 112 KiB region — which the earlier record did not show, because
it reported activations only (26,560 B) and those fit easily. Flash was the
blocker that got recorded; RAM was a second one.

**AC3 INT8 is the row Table 6 footnote e rests on.** INT8 brings its flash
inside budget (932,644 B against 1024 KiB) but its RAM total to 189,488 B:

```
189,488 B  AI RAM total   (46,500 activations + 142,988 runtime)
−114,688 B  linker RAM region (112 KiB)
=  74,800 B  =  73.05 KiB
+   4,096 B  application stack + heap (0x800 each), before HAL static data
≈  78 KB    → the figure quoted in the paper
```

Against this part's 128 KiB of contiguous SRAM it is still 57.05 KiB over. The
overrun is in the runtime, not the arena: at 46,500 B the activations fit
easily, while the runtime needs 142,988 B — 2.5× A0's 56,192 B.

## Why INT8 costs latency here rather than saving it

The operation-type tables separate the variants into two groups.

| variant | s8 | f32 | MACC vs its FP32 graph |
|---|---|---|---|
| E3 INT8 | 99.7 % | 0.2 % | 138,655 vs 139,523 (−0.6 %) |
| E4 INT8 | 100.0 % | 0.0 % | 1,036,567 vs 1,044,003 (−0.7 %) |
| A0 INT8 | 15.7 % | 80.1 % | 4,031,544 vs 3,901,505 (**+3.3 %**) |
| AC2 INT8 | 11.4 % | 85.5 % | 5,627,201 vs 5,493,259 (**+2.4 %**) |
| AC3 INT8 | 15.8 % | 83.4 % | 44,699,530 vs 44,383,435 (**+0.7 %**) |

The external baselines are plain feed-forward stacks and quantise essentially
completely. The proposed model and the architecture comparators do not: they
keep 80–86 % of their operations in float, so the graph is mixed precision and
pays for a quantise/dequantise pair at every boundary. The MACC count rises
rather than falls, and the Cortex-M4F has no SIMD path that would repay the
integer kernels that do exist.

That is the measured explanation for A0's INT8 latency of 288.3 ms against
251.1 ms in FP32 (`../latency/MEASUREMENT_LOG.md`). INT8 is worth keeping here
for flash footprint — which is what decides whether AC2 and AC3 fit at all — and
not as a latency measure.

For AC3 specifically the attention MatMuls are left in float, which is both why
its s8 share is 15.8 % and why its runtime RAM is the largest of the set.

## Import: what needed changing, and what could not be fixed

Eight FP32 graphs import and analyze. AC1 needed the LSTM unrolled first (§AC1
below); the rest needed the transformations listed after it. Those are
parity-preserving and leave the weights untouched.

The INT8 graphs analyze but are not clean conversions, as the table above shows.
No claim is made here that they are.

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

## AC1 — static BiLSTM unrolling

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

Self-check against the original BiLSTM: error ≈ 3e-8, i.e. bit-identical. Its
c-model validation gives room_corr cos ≈ 0.93 with fc/gain/q cos ≥ 0.9996; the
room_corr difference is float accumulation across the 32-step recursion on random
validation input and does not bear on latency, which is what this export is for.
The LSD accuracy reported in the paper is measured in PyTorch.

## Notes

- A0's c-model validation: cos ≈ 1.0, nse ≈ 1.0, rmse ≤ 8e-4.
- The `_m_Mul_*_0` output labels in a report are internal `stedgeai` node names.
  The ONNX files' own outputs are `room_corr` / `fc` / `gain` / `q`. When
  integrating the generated C, the output order is
  `[room_corr(128), fc(7), gain(7), q(7)]`; the E-series models use `[fc, gain, q]`
  with 5 bands.
- `duration ms/sample` in a report is a host (x86) figure. The F405 latency is
  measured on the board with the DWT cycle counter.
- **Reconciling with the paper's Table 6.** Its FP32 *Weights* column is
  `params × 4 / 1024` from the PyTorch parameter count, not the weight segment
  above. `analyze` counts slightly more parameters because export-time graph
  surgery injects constants — A0 204,738 against 203,447 (+0.63 %), E3 138,260
  against 138,255, AC1 241,890 against 240,758, AC2 267,013 against 265,590,
  AC3 412,998 against 411,959. No fit decision changes. The parameter-count
  figures are reproducible with `../onnx/verify_fp32_weights.py`.
- The paper's *RAM* column is the **total** above, and its Fit column is decided
  on total flash and total RAM together, which is why AC1 fails on both.
