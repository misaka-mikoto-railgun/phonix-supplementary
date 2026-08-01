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
- `[datasheet]` — a published property of the part.
- `[session record]` — read out during a measurement session; no file in this
  repository contains it. This covers the DWT cycle-counter figures of
  2026-06-23, which were read over SWD as described above.

## Hardware

| item | value | source |
|---|---|---|
| board | WeAct Studio STM32F405RGT6 v1.1 | `[session record]` |
| core clock | 168 MHz (SYSCLK), ART prefetch + I/D cache ON, FPU on | `[file: ../cmsis_mfcc/README_MFCC_BENCH.md]` |
| probe | ST-Link V2 clone (no VCP) | `[session record]` |
| power | USB-C | `[session record]` |
| flash / SRAM | 1024 KiB flash / 192 KiB total SRAM (F405RG) | `[datasheet]` |
| SRAM composition | 128 KiB main SRAM (112 KiB SRAM1 + 16 KiB SRAM2, contiguous, DMA-accessible) + 64 KiB CCM (D-bus, no DMA); the separate 4 KiB backup SRAM is excluded from the 192 KiB | `[datasheet]` |

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
| A0_Proposed | 288.3 ms | slower than the 251.09 ms FP32 figure: only 15.7 % of its operations quantise, and MACC rises 3.3 % | `[session record]`, `[file: ../build_reports/analyze_a0_int8.txt]` |
| AC2_GRU_Biquad | 389.6 ms | | `[session record]` |
| AC3_Conformer_Biquad | N/A | AI RAM 189,488 B (185.05 KiB) overflows the 112 KiB linker RAM region by ≈ 78 KB at link time | `[file: ../build_reports/analyze_ac3_int8.txt]`, `[file: ../build_reports/STM32F405RGTX_FLASH.ld.MEMORY]` |

INT8 accuracy impact is in `[file: ../onnx/INT8_EVAL.md]` (that file contains accuracy
only — no size or latency figures).

**The AC3 overrun is not an activation-buffer overflow.** The INT8 activations
are 46,500 B (45.41 KiB) and fit comfortably. What overflows is the total AI
RAM, of which the X-CUBE-AI runtime is 142,988 B (139.64 KiB) — 2.5× A0's
56,192 B — because the QDQ graph carries per-tensor quantisation tables and
mixed-precision conversion buffers.

The build declares the CubeMX default RAM region, which is SRAM1 only:

```
MEMORY {
  CCMRAM (xrw) : ORIGIN = 0x10000000, LENGTH =  64K
  RAM    (xrw) : ORIGIN = 0x20000000, LENGTH = 112K
  FLASH  (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
}
_Min_Heap_Size  = 0x800    _Min_Stack_Size = 0x800
```

```
189,488 B  AI RAM total   (46,500 activations + 142,988 runtime)
−114,688 B  RAM region     (112 KiB)
=  74,800 B  =  73.05 KiB
+   4,096 B  application stack + heap (0x800 each), before HAL static data
≈  78 KB    → the figure quoted in the paper
```

Even against this part's 128 KiB of contiguous SRAM the graph is 57.05 KiB over,
and the 64 KiB CCM is a separate region at a disjoint address, so it cannot back
the same allocation.

The three RAM figures come from
`[file: ../build_reports/analyze_ac3_int8.txt]` and the region from
`[file: ../build_reports/STM32F405RGTX_FLASH.ld.MEMORY]`. Every other row in
this file is a DWT cycle-counter read-out and does not depend on them.

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

Two different quantities, kept apart. `analyze weights` is the weight segment;
`analyze total` adds the ST runtime and generated code and is what a link has to
fit. The June column is what the CubeIDE build actually produced at the time and
is a session record; the analyze columns are from the 2026-08-02 reports against
the graphs regenerated on 2026-07-29, and are the figures the paper quotes.

| variant | analyze weights | **analyze total** | June build | fit (1024 KiB) |
|---|---|---|---|---|
| A0 / A2 | 823,080 B (803.79 KiB) | **905,127 B (883.91 KiB)** | 896 KB `[session record]` | ✓ |
| AC1_BiLSTM_Biquad (unrolled) | 971,692 B (948.92 KiB) | **1,277,193 B (1247.26 KiB)** | ~1.0–1.1 MB `[session record]` | ✗ |
| AC2_GRU_Biquad | 1,072,056 B (1046.93 KiB) | **1,164,688 B (1137.39 KiB)** | 1.12 MB `[session record]` | ✗ |
| AC3_Conformer_Biquad | 1,656,164 B (1617.35 KiB) | **1,796,380 B (1754.28 KiB)** | — | ✗ |
| AC3_Conformer_Biquad INT8 | 683,084 B (667.07 KiB) | **932,644 B (910.79 KiB)** | — | ✓ |

Analyze figures `[file: ../build_reports/analyze_a0.txt]`,
`[file: ../build_reports/analyze_ac1.txt]`,
`[file: ../build_reports/analyze_ac2.txt]`,
`[file: ../build_reports/analyze_ac3.txt]`,
`[file: ../build_reports/analyze_ac3_int8.txt]`; the full set including the
E-series and the other INT8 graphs is tabulated in
`[file: ../build_reports/ST_VALIDATION.md]`.

The June build figures corroborate the analyze totals to within about 1 %
(A0 896 KB against 905 KB, AC2 1.12 MB against 1.16 MB) — close enough that the
fit decisions are the same either way, which is why the earlier record could be
read as consistent. **Flash is not the only budget**, though: AC1 also fails on
RAM, needing 278,080 B against the 112 KiB region, and that was not visible while
only activations were being recorded.

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
