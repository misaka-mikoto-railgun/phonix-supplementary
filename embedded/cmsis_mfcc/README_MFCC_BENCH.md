# MFCC latency on-device 측정 (STM32F405RGT6, 168MHz, DWT)

목적: full-pipeline 분해(4s 버퍼링 / **MFCC** / NN forward)로 reviewer R1-11 닫기.
NN A0 = 251.1ms 기측정. 여기선 **버퍼 → 모델 입력 x[32,10]** 까지 전처리 전부를 DWT로 측정.

## 파일
| 파일 | 역할 |
|---|---|
| `gen_mfcc_const.py` | 상수 헤더 생성 + **librosa parity 검증**(host). 먼저 1회 실행. |
| `mfcc_const.h` | 자동생성: HANN_MFCC/HANN_FULL, sparse mel(MEL_W/START/COUNT/OFF), DCT(8×128), CFREQ(1025), **FRAMES_SEG(32×2048, 측정 클립)**, X_REF(32×10 librosa 기준). |
| `mfcc_f405.c/.h` | CMSIS-DSP MFCC (학습 librosa 동일 산출). |
| `bench_mfcc.c` | DWT 하니스(50회 최솟값) + parity. 전역 `g_*` 노출. |

## 사전 (host)
```
python gen_mfcc_const.py   # [host parity] max abs err ~1.2e-6 확인 → mfcc_const.h 생성
```
이 host parity가 깨지면 그 자리에서 멈추세요(틀린 MFCC의 latency는 무의미).

## STM32CubeIDE 프로젝트
1. **타겟**: STM32F405RGT6. NN(A0) 측정과 **동일 환경**: SYSCLK **168MHz**, **ART(Prefetch+I/D cache) ON**, FPU on, `-O2`(또는 NN 측정 때 쓴 동일 최적화), float32.
2. **CMSIS-DSP** 추가: CubeMX Software Packs → `X-CUBE-... / CMSIS DSP` 활성 (또는 `Drivers/CMSIS/DSP` 소스+`arm_cortexM4lf_math` 링크). `arm_math.h` include 경로, `__FPU_PRESENT=1`, `ARM_MATH_CM4` 정의.
3. `mfcc_const.h`(~290KB flash), `mfcc_f405.c`, `bench_mfcc.c` 를 프로젝트에 추가.
4. `main()` 에서 클럭/ART 초기화 후:
   ```c
   extern void bench_mfcc_run(void);
   bench_mfcc_run();   // 끝에서 무한루프 정지
   ```
5. Debug 실행 → 정지 후 **Live Expressions** 에 추가해 읽기:
   - `g_core_mhz`  → **168** 확인 (아니면 클럭 설정 틀림)
   - `g_mfcc_parity_maxabs` → **parity gate** (아래)
   - `g_mfcc_cyc_min`, `g_mfcc_ms` → **결과**

## Parity gate (필수, latency 유효성 조건)
`g_mfcc_parity_maxabs` = on-device x[32,10] vs librosa X_REF 최대절대오차.
- host(float64) parity는 1.2e-6. 디바이스는 **float32 + CMSIS rFFT**라 더 큼 — 보통 **~1e-3 이하**면 정상(같은 feature).
- 1e-2 이상이면 설정 의심: CMSIS rFFT init(2048), window 인덱싱, mel sparse offset, z-score 모집단 std(/32) 등 점검.

## 측정 구간 (bench가 감싸는 것 = NN 입력 직전까지 전부)
`for 32 frames { seg→Hann1200·rFFT·power→sparse mel(128)→log→DCT(8); +log_RMS(seg[424:1624]); +centroid(seg·Hann2048·rFFT, mag 가중) } → (10,32) → per-dim z-score → transpose → x[32,10]`.
종료점 = x[32,10] 완성(NN feed 직전). 4s 버퍼링은 미포함(적응 주기).

## 보고 양식
```
MFCC (32-frame), F405@168MHz, ART on, float32, DWT 50-run min:
  g_mfcc_ms = ____ ms   (g_mfcc_cyc_min = ____ cyc)
  parity max abs err = ____  (vs librosa, z-scored x[32,10])
```
full-pipeline 표:
| 단계 | 시간 | 4s 예산 | 비고 |
|---|---|---|---|
| 4s 버퍼링 | 4000 ms | (적응 주기) | 계산 아님 |
| MFCC (32프레임) | **g_mfcc_ms** | g_mfcc_ms/40 % | DWT 측정 |
| NN forward (A0) | 251.1 ms | 6.3% | 기측정 |
| 계산 합계 | 251.1+g_mfcc_ms | — | MFCC+NN |

## 주의/참고
- **프레임당 FFT 2회**: MFCC(Hann-1200)와 centroid(full Hann-2048)는 창이 달라 별도 rFFT 필요(학습 librosa 동일). 총 64 rFFT/clip. (스펙의 "1 FFT/frame" 가정보다 정확.)
- **32프레임 vs 401프레임**: 학습은 401프레임 STFT 후 32 서브샘플. 여기선 동일 출력을 주는 32프레임만 계산(≈ **12.5× (401/32) 절감**). 전체 STFT 가정 시 MFCC ≈ `g_mfcc_ms × ~12.5` (참고용 상한).
- **top_db 무시**: 32프레임 전역 max가 401과 달라 의미없어 생략(거의 비활성, latency 무관). host parity 1.2e-6 로 영향 없음 확인.
- **FRAMES_SEG = 측정용 LCG 클립**(flash, 결과값 무관·compute 동일). 실제 오디오로 바꾸려면 이 배열만 교체.
- 클립이 RAM(192KB)에 안 들어가므로(4s float=768KB) 세그먼트를 flash에 baked. flash 읽기는 ART 캐시로 흡수.
