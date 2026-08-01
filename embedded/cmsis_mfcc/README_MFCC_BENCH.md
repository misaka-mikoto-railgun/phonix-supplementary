# On-device MFCC latency measurement (STM32F405RGT6, 168 MHz, DWT)

Purpose: break the full pipeline into its parts — 4 s buffering / **MFCC** /
NN forward — and measure them. The NN forward for A0 was measured separately at
251.1 ms. What is measured here is the whole front end, from the audio buffer up
to the model input `x[32,10]`, using the DWT cycle counter.

## Files

| file | role |
|---|---|
| `gen_mfcc_const.py` | generates the constant header and checks parity against librosa (on the host). Run this first. |
| `mfcc_const.h` | generated: `HANN_MFCC`/`HANN_FULL`, sparse mel (`MEL_W`/`START`/`COUNT`/`OFF`), DCT (8×128), `CFREQ` (1025), **`FRAMES_SEG` (32×2048, the measurement clip)**, `X_REF` (32×10, the librosa reference). |
| `mfcc_f405.c` / `.h` | CMSIS-DSP MFCC, producing the same features as the librosa path used in training. |
| `bench_mfcc.c` | DWT harness (minimum of 50 runs) plus the parity check. Exposes the `g_*` globals. |

## Prerequisite (host)

```
python gen_mfcc_const.py   # expect [host parity] max abs err ~1.2e-6, then mfcc_const.h is written
```

If that parity check fails, stop there: the latency of a wrong MFCC means
nothing.

## STM32CubeIDE project

1. **Target**: STM32F405RGT6, in the **same configuration** as the NN
   measurement — SYSCLK **168 MHz**, **ART (prefetch + I/D cache) on**, FPU on,
   `-O2` (or whichever optimisation the NN run used), float32.
2. **Add CMSIS-DSP**: CubeMX → Software Packs → enable CMSIS DSP, or add the
   `Drivers/CMSIS/DSP` sources and link `arm_cortexM4lf_math`. Put `arm_math.h`
   on the include path and define `__FPU_PRESENT=1` and `ARM_MATH_CM4`.
3. Add `mfcc_const.h` (~290 KB of flash), `mfcc_f405.c` and `bench_mfcc.c` to
   the project.
4. In `main()`, after clock and ART initialisation:
   ```c
   extern void bench_mfcc_run(void);
   bench_mfcc_run();   // halts in an infinite loop when finished
   ```
5. Run under the debugger, and once it halts read these in **Live Expressions**:
   - `g_core_mhz` — must be **168**; anything else means the clock is misconfigured
   - `g_mfcc_parity_maxabs` — the parity gate, see below
   - `g_mfcc_cyc_min`, `g_mfcc_ms` — the result

## Parity gate (required; the latency is only meaningful if this passes)

`g_mfcc_parity_maxabs` is the maximum absolute error between the on-device
`x[32,10]` and the librosa reference `X_REF`.

- Host parity (float64) is 1.2e-6. The device runs float32 with the CMSIS rFFT,
  so its error is larger; **≤ ~1e-3 is normal** and means the same feature.
- At 1e-2 or above, suspect the configuration: CMSIS rFFT init (2048), window
  indexing, the sparse mel offsets, or the z-score population standard deviation
  (`/32`).

## What the benchmark encloses (everything up to the NN input)

```
for 32 frames {
    seg -> Hann-1200 -> rFFT -> power -> sparse mel (128) -> log -> DCT (8)
    + log_RMS(seg[424:1624])
    + centroid(seg -> Hann-2048 -> rFFT, magnitude-weighted)
}
-> (10,32) -> per-dimension z-score -> transpose -> x[32,10]
```

It ends when `x[32,10]` is complete, immediately before the NN is fed. The 4 s
of buffering is not included, since that is the adaptation period rather than
computation.

## Reporting format

```
MFCC (32-frame), F405 @ 168 MHz, ART on, float32, DWT 50-run min:
  g_mfcc_ms = ____ ms   (g_mfcc_cyc_min = ____ cyc)
  parity max abs err = ____  (vs librosa, z-scored x[32,10])
```

Full-pipeline table:

| stage | time | share of the 4 s budget | note |
|---|---|---|---|
| 4 s buffering | 4000 ms | (adaptation period) | not computation |
| MFCC (32 frames) | **g_mfcc_ms** | g_mfcc_ms / 40 % | DWT measurement |
| NN forward (A0) | 251.1 ms | 6.3 % | measured separately |
| computation total | 251.1 + g_mfcc_ms | — | MFCC + NN |

## Notes

- **Two FFTs per frame.** The MFCC (Hann-1200) and the spectral centroid (full
  Hann-2048) use different windows and therefore need separate rFFTs, exactly as
  in the librosa path used for training. That is 64 rFFTs per clip — more than
  the "1 FFT per frame" a specification sheet would assume.
- **32 frames rather than 401.** Training computes a 401-frame STFT and then
  subsamples 32. Here only those 32 frames are computed, which gives the same
  output for about 12.5× (401/32) less work. If the full STFT were required,
  MFCC would be roughly `g_mfcc_ms × 12.5` — an upper bound, for reference.
- **`top_db` is skipped.** Its global maximum over 32 frames differs from the one
  over 401, so applying it would not mean the same thing; it is nearly inactive
  either way and does not affect latency. The 1.2e-6 host parity confirms no
  effect on the features.
- **`FRAMES_SEG` is an LCG-generated measurement clip** kept in flash. Its values
  do not affect the result, only the amount of computation, which is identical.
  To measure on real audio, replace this array alone.
- A whole clip does not fit in SRAM (4 s of float32 is 768 KiB against 192 KiB
  total), so the segments are baked into flash. Flash reads are absorbed by the
  ART cache.
