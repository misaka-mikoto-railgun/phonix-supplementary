#ifndef MFCC_F405_H
#define MFCC_F405_H
#include "mfcc_const.h"     // SR/NFFT/WINLEN/HOP/NMELS/NMFCC/SEQ/NBIN/RMS_OFF + const tables
#define FEATD (NMFCC + 2)   // 10

// 1회 초기화 (CMSIS rFFT 인스턴스).
void mfcc_init(void);

// FRAMES_SEG(flash) 로부터 모델 입력 x[SEQ][FEATD] = x[32][10] 계산.
// 학습 librosa 파이프라인과 동일: MFCC(Hann1200)+log_RMS+centroid(Hann2048 full),
// per-dim z-score(32프레임). NN 입력 직전까지 전부 포함.
void mfcc_compute(float x_out[SEQ][FEATD]);

#endif
