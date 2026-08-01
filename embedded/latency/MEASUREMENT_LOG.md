# On-device latency measurement — provenance record

STM32F405 latency figures reported for this work were measured on hardware on
**2026-06-23**. This file documents *how* they were obtained. It is a provenance
record, **not** a substitute for a raw log.

## Why there is no raw log file

The ST-Link V2 clone used for flashing has **no VCP (USB CDC COM port)**, so UART
logging from the target was not possible. All counters were read **through SWD with
STM32CubeIDE Live Expressions** while the target was halted at a breakpoint after the
benchmark loop. A debugger read-out leaves no file artefact — the absence of a log
file is therefore expected and does not indicate an absence of measurement.

## Source labels

- `[file: <path>]` — value verified in a file in this repository.
- `[session record]` — value from the 2026-06-23 measurement session; no file in this
  repository contains it.

## Hardware

| item | value | source |
|---|---|---|
| board | WeAct Studio STM32F405RGT6 v1.1 | `[session record]` |
| core clock | 168 MHz (SYSCLK), ART prefetch + I/D cache ON, FPU on | `[file: ../cmsis_mfcc/README_MFCC_BENCH.md]` |
| probe | ST-Link V2 clone (no VCP) | `[session record]` |
| power | USB-C | `[session record]` |
| flash / SRAM | 1024 KiB / 192 KB (F405RG) | `[datasheet]` |
| SRAM composition | 112 KB SRAM1 + 16 KB SRAM2 (128 KB contiguous, DMA-accessible) + 64 KB CCM (D-bus, no DMA) + 4 KB backup | `[datasheet]` |

## Toolchain

| item | value | source |
|---|---|---|
| IDE | STM32CubeIDE 2.1.1 | `[session record]` |
| AI pack | X-CUBE-AI 10.2.0 = ST Edge AI Core v2.2.0-20266 | `[file: ../build_reports/ST_VALIDATION.md]` |
| ONNX opset | **13** | `[file: ../onnx/manifest.json]` |
| graph surgery | Pad→Conv.pads fold; searchsorted→constant matmul; nn.Embedding→one-hot matmul; LayerNorm decomposed | `[file: ../onnx/manifest.md]`, `[file: ../onnx/export_onnx.py]` |
| project template | CubeMX **ApplicationTemplate** | `[session record]` |

The **SystemPerformance** template could not be used: it requires `bsp_ai.h` and a COM
port for its report output, neither of which was available on this board/probe
combination. `[session record]`

## Method

- Cycle counter: **DWT->CYCCNT** (`CoreDebug->DEMCR |= TRCENA`, `DWT->CTRL |= CYCCNTENA`)
  `[file: bench_mfcc.c]`
- Statistic: **minimum of 50 runs** `[file: bench_mfcc.c]`, `[file: ../cmsis_mfcc/README_MFCC_BENCH.md]`
- Read-out: CubeIDE **Live Expressions over SWD** on the global `g_*` variables
  `[file: bench_mfcc.c]`, `[session record]`
- Time conversion: `ms = cycles / 168e6 * 1000` `[session record]`
- Harness: `bench_mfcc.c` (MFCC path); the NN path was measured in the same project and
  the same optimisation settings (`-O2`, float32). `[file: bench_mfcc.c]`

## Neural inference (FP32)

| variant | cycles | latency | source |
|---|---|---|---|
| A0_Proposed | **42.18 M** | **251.09 ms** | `[session record]` |
| A2_withPrefLoss | `[cycle count not recorded]` | 251.4 ms | `[session record]` |
| E3_Nercessian | `[cycle count not recorded]` | 6.98 ms | `[session record]` |
| E4_Pepe | `[cycle count not recorded]` | 58.4 ms | `[session record]` |
| E5_Sequential | `[cycle count not recorded]` | ≡ E3 (same graph) | `[session record]` |

A search of this repository and of the working tree found **no file** containing the
per-variant cycle counts other than A0; those rows are marked
`[cycle count not recorded]` rather than back-computed from the millisecond figures.

## Neural inference (INT8 PTQ)

| variant | latency | note | source |
|---|---|---|---|
| A0_Proposed | 288.3 ms | slower than FP32 on this MCU | `[session record]` |
| AC2_GRU_Biquad | 389.6 ms | | `[session record]` |
| AC3_Conformer_Biquad | N/A | activation buffer over budget by ≈ 78 KB | `[session record]` |

INT8 accuracy impact is in `[file: ../onnx/INT8_EVAL.md]` (that file contains accuracy
only — no size or latency figures).

The ≈ 78 KB overrun for AC3 is a session record: the linker `.map` and the
CubeIDE project are not part of this repository, so neither the region the
activation buffer was placed in nor the exact shortfall can be re-derived here.
For scale, the ST Edge AI `analyze` step reports 96.94 KiB RAM(total) for the
FP32 graph `[file: ../build_reports/ST_VALIDATION.md]`; that figure covers the
analysed graph only, not the built application, and is not the number the
overrun was measured against. The other rows in this file are cycle-counter
measurements and do not depend on it.

## Feature front-end (CMSIS-DSP MFCC)

| item | value | source |
|---|---|---|
| MFCC (32 frames) | **106.3 ms** | `[session record]` |
| parity vs librosa | **8.9e-5** max abs err (on-device, float32 + CMSIS rFFT) | `[session record]` |
| host parity (float64) | 1.2e-6 | `[file: ../cmsis_mfcc/README_MFCC_BENCH.md]` |
| parity gate | on-device expected ≤ ~1e-3; ≥1e-2 indicates misconfiguration | `[file: ../cmsis_mfcc/README_MFCC_BENCH.md]` |

## Compute total

| item | value | source |
|---|---|---|
| MFCC + A0 inference | **357.4 ms** | `[session record]` |

## Flash footprint and fit

| variant | analyze (weights) | build flash | fit (1024 KiB) | source |
|---|---|---|---|---|
| A0 / A2 | 804 KiB | **896 KB** | ✓ (78%) | `[session record]` |
| AC1_BiLSTM_Biquad | ~940 KiB | ~1.0–1.1 MB (unrolled) | ✗ over | `[session record]` |
| AC2_GRU_Biquad | 1.02 MiB | 1.12 MB | ✗ over | `[session record]` |
| AC3_Conformer_Biquad | 1.58 MiB (+58%), MACC 44.8 M | — | ✗ over | `[session record]` |

`analyze` reports weight-dominated size; a real build adds roughly 100 KiB of runtime
and HAL. **fit / no-fit is decided on the build figure**, cross-checked by
A0 804 KiB → 896 KB and AC2 1.02 MiB → 1.12 MB. `[session record]`

FP32 weight sizes are reproducible from the parameter counts:
`embedded/onnx/verify_fp32_weights.py` (`KiB = n_params × 4 / 1024`).
`[file: ../onnx/verify_fp32_weights.py]`

## Reproducing

1. Generate the constant header and check host parity:
   `python ../cmsis_mfcc/gen_mfcc_const.py`
2. Add `mfcc_f405.c/.h`, `mfcc_const.h` and `bench_mfcc.c` to a CubeIDE
   **ApplicationTemplate** project for STM32F405RGT6 (168 MHz, ART on, `-O2`).
3. Import the ONNX model from `embedded/onnx/` with X-CUBE-AI 10.2.0 and generate the
   C model.
4. Run, halt after the benchmark loop, and read `g_*` in **Live Expressions**.
   Convert cycles to milliseconds with the 168 MHz core clock.
5. Check the parity gate (`g_mfcc_parity_maxabs`) before trusting any latency number —
   a mismatched MFCC makes the latency meaningless.
