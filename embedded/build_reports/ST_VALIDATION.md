# ST Edge AI Core 2.2.0 — 검증 결과 (STM32F405 latency benchmark)

툴: X-CUBE-AI 10.2.0 (= ST Edge AI Core v2.2.0-20266), 동봉 `stedgeai.exe`, `--target stm32`.
이 문서는 수동 유지(매 export 마다 덮어쓰이는 `manifest.md` 와 별개).

**대상 export**: opset 13 (`embedded/onnx/`, manifest generated 2026-06-22T20:40:02).

**출처 라벨**: `[tool]` = stedgeai analyze 리포트에서 확인된 값.
`[session record]` = 2026-06-23 측정 세션의 기록이며, 이 저장소에 원본 리포트 파일이
존재하지 않는 값(사유는 `embedded/latency/MEASUREMENT_LOG.md` 참조).

## 최종 상태 — 8/8 analyze + generate PASS (analyze 기준; AC1 은 정적 언롤본)

| variant | onnx | import(native) | analyze | flash weights | build flash | fit (1024 KiB) | MACC(tool) | RAM(total) | validate(c-model vs ref) |
|---|---|---|---|---|---|---|---|---|---|
| A0_Proposed | a0.onnx | PASS | PASS | 804 KiB `[session record]` | 896 KB `[session record]` | ✓ (78%) | 3,901,505 `[tool]` | 27.05 KiB `[tool]` | cos≈1.0, nse≈1.0 (rmse≤8e-4) |
| A2_withPrefLoss | a2.onnx | PASS | PASS | 804 KiB `[session record]` | 896 KB `[session record]` | ✓ (78%) | 3,901,505 `[tool]` | 27.05 KiB `[tool]` | (A0 동일 구조) |
| E3_Nercessian | e3_nercessian.onnx | PASS | PASS | — | — | ✓ | 139,523 `[tool]` | 2.00 KiB `[tool]` | — |
| E4_Pepe | e4_pepe.onnx | PASS | PASS | — | — | ✓ | 1,044,003 `[tool]` | 18.50 KiB `[tool]` | — |
| E5_Sequential | e5_sequential.onnx | PASS | PASS | — | — | ✓ | 139,523 `[tool]` | 2.00 KiB `[tool]` | — |
| AC1_BiLSTM_Biquad | ac1_bilstm_biquad.onnx | **FAIL**³ → 언롤본 PASS¹ | PASS¹ | ~940 KiB `[session record]` | ~1.0–1.1 MB `[session record]` | ✗ over | 4,815,819 `[tool]` | 25.94 KiB `[tool]` | room_corr cos≈0.93²; fc/gain/q cos≥0.9996 |
| AC2_GRU_Biquad | ac2_gru_biquad.onnx | PASS | PASS | 1.02 MiB `[session record]` | 1.12 MB `[session record]` | ✗ over | 5,493,259 `[tool]` | 25.94 KiB `[tool]` | — |
| AC3_Conformer_Biquad | ac3_conformer_biquad.onnx | PASS | PASS | 1.58 MiB (+58%) `[session record]` | — | ✗ over | 44,383,435 `[tool]` | 96.94 KiB `[tool]` | — |

¹ AC1 은 LSTM 정적 언롤본(아래 §AC1). 표의 analyze PASS 는 **언롤 적용 후** 상태다.
² 32-step 재귀의 float 누적 차(랜덤 validate 입력). **latency 목적엔 무관** — 논문 LSD 정확도는 PyTorch 기준이며 STM32 export 는 latency 전용.
³ nn.LSTM 기반 원본은 ST import 단계에서 실패(`dl_remapping`). 아래 §AC1 참조.

**fit 판정 주의**: `analyze` 의 flash 값은 가중치 위주이고, 실제 빌드는 런타임+HAL 로
약 100 KiB 가 추가된다. **fit/no-fit 판정은 build flash 기준**이다
(A0 804 KiB → 896 KB, AC2 1.02 MiB → 1.12 MB 로 교차확인). `[session record]`

## ST 호환을 위해 적용한 변환 (전부 parity 보존, 가중치 동일)
- **opset 13**: nn.LayerNorm 을 기본연산 분해(2.2.0 은 fused `LayerNormalization` 미지원).
- **mode embedding → one-hot matmul**: nn.Embedding(Gather)이 embedding-table leading dim(4)을 batch 로 오인 → one-hot @ weight 로 치환.
- **pref_curve interp → 상수 matmul**: searchsorted(opset17 미지원) 제거.
- **Pad → Conv.pads 흡수**: standalone Pad codegen 버그(`_Pad_output_0_value_data[]={[]}`) 회피. causal pads=[(k-1)*dilation, 0].
- **AC2 2-layer GRU → 1-layer GRU ×2 직렬**: stacked-GRU transpose 식별 실패 회피.
- **ORT BASIC 상수폴딩 ON** + disabled_optimizers=[LayerNormFusion 계열, MatMulAddFusion].
  MatMulAddFusion 비활성은 언롤 LSTM 이 가변텐서를 Gemm bias 로 융합하는 것 방지(다른 모델 무영향).

## AC1 — BiLSTM 정적 언롤 (핵심)
nn.LSTM 기반은 ST 가 TCN(Conv1d)→LSTM transpose 를 remap 못 해 임포트 불가
(원본 bidirectional / uni 분해 / seq-first / barrier / simplifier 전부 `dl_remapping` 실패).
관측된 오류 `[session record]`:
```
INTERNAL ERROR: _m_lstm_f_LSTM_output_0_in_transpose of type Transpose
has not parameter dl_remapping
```

**원인 격리 (대조군 실험)** `[session record]`
- 대조군 `Linear → bi-LSTM`: **PASS** — bi-LSTM 자체는 import 가능
- 본 모델 `TCN(Conv1d) → LSTM`: **FAIL**
→ 실패 원인이 LSTM 연산 자체가 아니라 **Conv1d 출력에서 LSTM 입력으로 가는
transpose 의 remap 부재**임이 격리됨.

→ **seq_len=32 고정 언롤**: LSTM 재귀를 Gemm/Sigmoid/Tanh/Mul/Add 로 펼침(LSTM op 제거).
F405 엔 LSTM 가속기가 없어 실제로도 이렇게 구동되므로 latency 대표성 유지.
구현 주의(ST 제약):
- 셀 게이트 = `addmm(const_bias, cat([x_t, h]), [W_ih|W_hh]^T)` — 가변텐서 Add 없이 상수 bias 만
  (ST 의 multi-dim/variable Gemm bias 미지원 회피).
- 3D Gemm 회피(2D 유지), h/c init 은 batch-agnostic(`x.new_zeros(x.shape[0],H)`).
- 원본 BiLSTM 대비 self-check err ≈ 3e-8 (bit-identical).

## 비고
- 리포트의 `_m_Mul_*_0` 출력 라벨은 stedgeai 내부 노드명일 뿐, onnx 파일 출력명은 `room_corr/fc/gain/q` 정상. C 통합 시 출력 순서 `[room_corr(128), fc(7), gain(7), q(7)]` (E 계열은 `[fc,gain,q]`, 5-band).
- `duration ms/sample` 은 호스트(x86) 표시값 — 실제 F405 latency 는 보드 DWT 사이클 카운터로 측정.
- FP32 weights 는 `params × 4 / 1024` 로 재현 가능: `embedded/onnx/verify_fp32_weights.py` 참조.
