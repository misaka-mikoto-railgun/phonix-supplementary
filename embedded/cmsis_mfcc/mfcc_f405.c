/* mfcc_f405.c — STM32F405 (Cortex-M4F) MFCC 전처리, CMSIS-DSP.
 * 학습(librosa 0.11.0)과 동일 산출 (host parity max abs err ~1e-6, gen_mfcc_const.py 검증).
 * 32프레임만 계산(서브샘플 idx). mel 은 sparse. centroid 는 별도 full-Hann FFT.
 */
#include "mfcc_f405.h"
#include "arm_math.h"
#include <math.h>

static arm_rfft_fast_instance_f32 S_fft;

static float win_buf[NFFT];     /* windowed segment (rfft 입력, 파괴됨) */
static float fft_buf[NFFT];     /* rfft 출력 (CMSIS packed) */
static float power[NBIN];       /* |FFT|^2 */
static float mel_db[NMELS];     /* 10log10(mel energy) */
static float F[FEATD][SEQ];     /* (10,32) 누적 후 z-score */

void mfcc_init(void) {
    arm_rfft_fast_init_f32(&S_fft, NFFT);
}

/* CMSIS packed rfft 출력(fft_buf) → power[NBIN].
 * fft_buf[0]=DC(real), fft_buf[1]=Nyquist(real), 이후 [Re_k, Im_k] k=1..N/2-1 */
static void rfft_power(const float *seg, const float *window) {
    for (int j = 0; j < NFFT; j++) win_buf[j] = seg[j] * window[j];
    arm_rfft_fast_f32(&S_fft, win_buf, fft_buf, 0);
    power[0]        = fft_buf[0] * fft_buf[0];
    power[NBIN - 1] = fft_buf[1] * fft_buf[1];
    for (int k = 1; k < NBIN - 1; k++)
        power[k] = fft_buf[2 * k] * fft_buf[2 * k] + fft_buf[2 * k + 1] * fft_buf[2 * k + 1];
}

void mfcc_compute(float x_out[SEQ][FEATD]) {
    for (int f = 0; f < SEQ; f++) {
        const float *seg = &CLIP[FRAME_OFF[f]];   /* 8192 측정버퍼 내 32 overlapping frame */

        /* ---- MFCC: Hann-1200(2048 pad) → power → sparse mel → log → DCT-II(8) ---- */
        rfft_power(seg, HANN_MFCC);
        for (int m = 0; m < NMELS; m++) {
            const float *w = &MEL_W[MEL_OFF[m]];
            const int    s = MEL_START[m];
            const int    c = MEL_COUNT[m];
            float acc = 0.0f;
            for (int j = 0; j < c; j++) acc += w[j] * power[s + j];
            if (acc < 1e-10f) acc = 1e-10f;
            mel_db[m] = 10.0f * log10f(acc);
        }
        for (int k = 0; k < NMFCC; k++) {
            const float *dk = &DCT[k * NMELS];
            float acc = 0.0f;
            for (int n = 0; n < NMELS; n++) acc += dk[n] * mel_db[n];
            F[k][f] = acc;
        }

        /* ---- log_RMS: seg[RMS_OFF .. RMS_OFF+1200) (= librosa frame=1200,center 동일) ---- */
        {
            float ss = 0.0f;
            for (int j = 0; j < WINLEN; j++) { float v = seg[RMS_OFF + j]; ss += v * v; }
            float rms = sqrtf(ss / (float)WINLEN);
            F[NMFCC][f] = logf(rms + 1e-8f);
        }

        /* ---- spectral_centroid: full Hann-2048, magnitude 가중, /24000 ---- */
        rfft_power(seg, HANN_FULL);
        {
            float num = 0.0f, den = 0.0f;
            for (int k = 0; k < NBIN; k++) { float mag = sqrtf(power[k]); num += CFREQ[k] * mag; den += mag; }
            F[NMFCC + 1][f] = (num / (den + 1e-12f)) / 24000.0f;
        }
    }

    /* ---- per-dim z-score (32프레임 기준), population std + 1e-8 ---- */
    for (int d = 0; d < FEATD; d++) {
        float mean = 0.0f;
        for (int f = 0; f < SEQ; f++) mean += F[d][f];
        mean /= (float)SEQ;
        float var = 0.0f;
        for (int f = 0; f < SEQ; f++) { float e = F[d][f] - mean; var += e * e; }
        float std = sqrtf(var / (float)SEQ) + 1e-8f;
        for (int f = 0; f < SEQ; f++) F[d][f] = (F[d][f] - mean) / std;
    }

    /* ---- transpose → x[32][10] (NN 입력 직전 = 측정 종료점) ---- */
    for (int f = 0; f < SEQ; f++)
        for (int d = 0; d < FEATD; d++) x_out[f][d] = F[d][f];
}
