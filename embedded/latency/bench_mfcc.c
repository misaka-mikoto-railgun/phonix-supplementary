/* bench_mfcc.c — MFCC latency를 DWT 사이클 카운터로 on-device 측정 (NN 측정과 동일 환경).
 *   - 168MHz, ART on, float32.
 *   - 50회 반복 최솟값 → g_mfcc_cyc_min, g_mfcc_ms.
 *   - librosa reference(X_REF)와 parity max abs err → g_mfcc_parity_maxabs.
 * 사용: main()에서 SystemClock 168MHz/ART 설정 후 bench_mfcc_run() 1회 호출,
 *       STM32CubeIDE Live Expressions 로 아래 g_* 전역 읽기.
 */
#include "mfcc_f405.h"
#include "stm32f4xx.h"      /* CoreDebug, DWT, SystemCoreClock */
#include <math.h>

volatile uint32_t g_mfcc_cyc_min   = 0xFFFFFFFFu;  /* 최소 사이클 (50회) */
volatile float    g_mfcc_ms        = 0.0f;         /* = cyc_min / 168000 */
volatile float    g_mfcc_parity_maxabs = 0.0f;     /* vs X_REF (librosa) */
volatile uint32_t g_core_mhz       = 0;            /* SystemCoreClock 확인용 */

static float x_out[SEQ][FEATD];

static void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
#ifdef DWT_LAR
    DWT->LAR = 0xC5ACCE55;   /* 일부 코어 unlock 필요 */
#endif
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

void bench_mfcc_run(void) {
    g_core_mhz = SystemCoreClock / 1000000u;   /* 168 이어야 함 */
    mfcc_init();
    dwt_init();

    /* warm-up (ART/flash 캐시 채우기) */
    for (int i = 0; i < 5; i++) mfcc_compute(x_out);

    /* 50회 측정, 최솟값 */
    g_mfcc_cyc_min = 0xFFFFFFFFu;
    for (int it = 0; it < 50; it++) {
        DWT->CYCCNT = 0;
        mfcc_compute(x_out);
        uint32_t c = DWT->CYCCNT;
        if (c < g_mfcc_cyc_min) g_mfcc_cyc_min = c;
    }
    g_mfcc_ms = (float)g_mfcc_cyc_min / 168000.0f;   /* 168MHz */

    /* parity: 마지막 x_out vs X_REF(librosa) max abs err */
    float me = 0.0f;
    for (int f = 0; f < SEQ; f++)
        for (int d = 0; d < FEATD; d++) {
            float e = fabsf(x_out[f][d] - X_REF[f * FEATD + d]);
            if (e > me) me = e;
        }
    g_mfcc_parity_maxabs = me;

    /* 여기서 정지하고 Live Expressions 로:
     *   g_core_mhz(=168), g_mfcc_cyc_min, g_mfcc_ms, g_mfcc_parity_maxabs */
    for (;;) { __NOP(); }
}
