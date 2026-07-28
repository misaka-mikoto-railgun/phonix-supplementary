# ONNX Export Manifest — STM32F405 latency benchmark

- generated: 2026-06-22T20:40:02
- opset: **13**, batch=1, dynamic_axes=False, dtype=float32
- backbone input: `feat [1, 32, 10]` (seq_len=32=4.0s, in_dim=10)
- conditional inputs: room_response[1,128], **mode_onehot[1,4] float**, band_gains[1,10]
- pref_curve `_interp` → 상수 matmul 치환(searchsorted 제거, parity 보존)
- mode embedding(nn.Embedding Gather) → one-hot matmul 치환(ST Edge AI Core 의 embedding-table batch 오인 회피; host에서 one-hot 인코딩)
- standalone Pad → Conv.pads 흡수(fold_pad_into_conv): X-CUBE-AI 10.2.0 의 Pad codegen 버그 회피. causal pads=[(k-1)*dilation, 0]

| variant | group | onnx | ckpt | seed | gain_max | params | parity max|err| | max\|gain\| | gate | MACC(thop) | HARD blocker | VERIFY ops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_Proposed | A0 | a0.onnx | A0_g12_f16k_s7.pt | 7 | 12.0 | 203,447 | 1.95e-03 | 12.00 | PASS | 3344256 | - | ['Erf', 'ReduceSum', 'Softmax'] |
| A2_withPrefLoss | A2 | a2.onnx | A2_g12_f16k_s7.pt | 7 | 12.0 | 203,447 | 1.22e-03 | 12.00 | PASS | 3344256 | - | ['Erf', 'ReduceSum', 'Softmax'] |
| E3_Nercessian | E | e3_nercessian.onnx | E3_Nercessian.pt | 42 | 12.0 | 138,255 | 1.46e-03 | 7.19 | n/a (구조적 ±12, trap 없음) | 137472 | - | - |
| E4_Pepe | E | e4_pepe.onnx | E4_Pepe.pt | 42 | 12.0 | 50,351 | 1.46e-03 | 11.65 | n/a (구조적 ±12, trap 없음) | 1036288 | - | - |
| E5_Sequential | E | e5_sequential.onnx | E5_Sequential.pt | 42 | 12.0 | 138,255 | 1.95e-03 | 5.61 | n/a (구조적 ±12, trap 없음) | 137472 | - | - |
| AC1_BiLSTM_Biquad | AC | ac1_bilstm_biquad.onnx | AC1_BiLSTM_Biquad_g12.pt | None | 12.0 | 240,758 | 1.71e-03 | 10.81 | PASS | 3354496 | - | ['Erf'] |
| AC2_GRU_Biquad | AC | ac2_gru_biquad.onnx | AC2_GRU_Biquad_g12.pt | None | 12.0 | 315,510 | 6.10e-04 | 11.42 | PASS | 4980608 | - | ['Erf', 'GRU'] |
| AC3_Conformer_Biquad | AC | ac3_conformer_biquad.onnx | AC3_Conformer_Biquad_g12.pt | None | 12.0 | 411,959 | 9.77e-04 | 11.34 | PASS | 8562560 | - | ['Erf', 'ReduceSum', 'Softmax'] |

## 그래프 경계 & 제외 post-processing

### A0_Proposed (`a0.onnx`)
- output format: room_corr[1,128] (dense room-correction dB) + fc/gain/q [1,7] (7-band biquad params)
- output shapes: {'room_corr': [1, 128], 'fc': [1, 7], 'gain': [1, 7], 'q': [1, 7]}
- graph boundary: input → {room_corr, fc, gain, q}.  peq_response(가우시안)·biquad 계수계산·pref_curve 합성은 그래프 밖.
- excluded post-proc (C에서 측정 = 비교표 2nd 컬럼): host/C: ①pref_curve = band_gains→128 선형보간, ②7-band closed-form biquad(RBJ) 계수계산 후 room_corr(dense)와 합성. 가우시안 재구성은 학습전용이라 미사용.
- onnx ops: ['Add', 'Concat', 'Conv', 'Div', 'Erf', 'Gemm', 'MatMul', 'Mul', 'Pow', 'ReduceMean', 'ReduceSum', 'Sigmoid', 'Slice', 'Softmax', 'Sqrt', 'Squeeze', 'Sub', 'Tanh', 'Transpose', 'Unsqueeze']

### A2_withPrefLoss (`a2.onnx`)
- output format: room_corr[1,128] (dense room-correction dB) + fc/gain/q [1,7] (7-band biquad params)
- output shapes: {'room_corr': [1, 128], 'fc': [1, 7], 'gain': [1, 7], 'q': [1, 7]}
- graph boundary: input → {room_corr, fc, gain, q}.  peq_response(가우시안)·biquad 계수계산·pref_curve 합성은 그래프 밖.
- excluded post-proc (C에서 측정 = 비교표 2nd 컬럼): host/C: ①pref_curve = band_gains→128 선형보간, ②7-band closed-form biquad(RBJ) 계수계산 후 room_corr(dense)와 합성. 가우시안 재구성은 학습전용이라 미사용.
- onnx ops: ['Add', 'Concat', 'Conv', 'Div', 'Erf', 'Gemm', 'MatMul', 'Mul', 'Pow', 'ReduceMean', 'ReduceSum', 'Sigmoid', 'Slice', 'Softmax', 'Sqrt', 'Squeeze', 'Sub', 'Tanh', 'Transpose', 'Unsqueeze']

### E3_Nercessian (`e3_nercessian.onnx`)
- output format: fc/gain/q [1,5] (5-band parametric PEQ params)
- output shapes: {'fc': [1, 5], 'gain': [1, 5], 'q': [1, 5]}
- graph boundary: input → {fc, gain, q}.  response(가우시안)·biquad 계수계산은 그래프 밖.
- excluded post-proc (C에서 측정 = 비교표 2nd 컬럼): host/C: 5-band closed-form biquad 계수계산. 가우시안 재구성은 학습전용.
- onnx ops: ['Add', 'Gemm', 'Mul', 'ReduceMean', 'Relu', 'Sigmoid', 'Slice', 'Tanh']

### E4_Pepe (`e4_pepe.onnx`)
- output format: fc/gain/q [1,5] (5-band parametric PEQ params)
- output shapes: {'fc': [1, 5], 'gain': [1, 5], 'q': [1, 5]}
- graph boundary: input → {fc, gain, q}.  response(가우시안)·biquad 계수계산은 그래프 밖.
- excluded post-proc (C에서 측정 = 비교표 2nd 컬럼): host/C: 5-band closed-form biquad 계수계산. 가우시안 재구성은 학습전용.
- onnx ops: ['Add', 'Conv', 'Gemm', 'GlobalAveragePool', 'Mul', 'Relu', 'Sigmoid', 'Slice', 'Squeeze', 'Tanh', 'Transpose']

### E5_Sequential (`e5_sequential.onnx`)
- output format: fc/gain/q [1,5] (내부 E3 room-corrector; E2 pref는 고정 테이블/비신경망)
- output shapes: {'fc': [1, 5], 'gain': [1, 5], 'q': [1, 5]}
- graph boundary: input → {fc, gain, q}(E3).  E2 테이블 합산은 그래프 밖.
- excluded post-proc (C에서 측정 = 비교표 2nd 컬럼): host/C: 5-band closed-form biquad(E3) + 고정 모드 프로파일(E2) 합산.
- onnx ops: ['Add', 'Gemm', 'Mul', 'ReduceMean', 'Relu', 'Sigmoid', 'Slice', 'Tanh']

### AC1_BiLSTM_Biquad (`ac1_bilstm_biquad.onnx`)
- output format: room_corr[1,128] (dense room-correction dB) + fc/gain/q [1,7] (7-band biquad params)
- output shapes: {'room_corr': [1, 128], 'fc': [1, 7], 'gain': [1, 7], 'q': [1, 7]}
- graph boundary: input → {room_corr, fc, gain, q}.  peq_response(가우시안)·biquad 계수계산·pref_curve 합성은 그래프 밖.
- excluded post-proc (C에서 측정 = 비교표 2nd 컬럼): host/C: ①pref_curve = band_gains→128 선형보간, ②7-band closed-form biquad(RBJ) 계수계산 후 room_corr(dense)와 합성. 가우시안 재구성은 학습전용이라 미사용.
- onnx ops: ['Add', 'Concat', 'Conv', 'Div', 'Erf', 'Gather', 'Gemm', 'MatMul', 'Mul', 'Pow', 'ReduceMean', 'Reshape', 'Sigmoid', 'Slice', 'Sqrt', 'Sub', 'Tanh', 'Transpose']

### AC2_GRU_Biquad (`ac2_gru_biquad.onnx`)
- output format: room_corr[1,128] (dense room-correction dB) + fc/gain/q [1,7] (7-band biquad params)
- output shapes: {'room_corr': [1, 128], 'fc': [1, 7], 'gain': [1, 7], 'q': [1, 7]}
- graph boundary: input → {room_corr, fc, gain, q}.  peq_response(가우시안)·biquad 계수계산·pref_curve 합성은 그래프 밖.
- excluded post-proc (C에서 측정 = 비교표 2nd 컬럼): host/C: ①pref_curve = band_gains→128 선형보간, ②7-band closed-form biquad(RBJ) 계수계산 후 room_corr(dense)와 합성. 가우시안 재구성은 학습전용이라 미사용.
- onnx ops: ['Add', 'Concat', 'Conv', 'Div', 'Erf', 'GRU', 'Gather', 'Gemm', 'MatMul', 'Mul', 'Pow', 'ReduceMean', 'Reshape', 'Sigmoid', 'Sqrt', 'Squeeze', 'Sub', 'Tanh', 'Transpose']

### AC3_Conformer_Biquad (`ac3_conformer_biquad.onnx`)
- output format: room_corr[1,128] (dense room-correction dB) + fc/gain/q [1,7] (7-band biquad params)
- output shapes: {'room_corr': [1, 128], 'fc': [1, 7], 'gain': [1, 7], 'q': [1, 7]}
- graph boundary: input → {room_corr, fc, gain, q}.  peq_response(가우시안)·biquad 계수계산·pref_curve 합성은 그래프 밖.
- excluded post-proc (C에서 측정 = 비교표 2nd 컬럼): host/C: ①pref_curve = band_gains→128 선형보간, ②7-band closed-form biquad(RBJ) 계수계산 후 room_corr(dense)와 합성. 가우시안 재구성은 학습전용이라 미사용.
- onnx ops: ['Add', 'Concat', 'Conv', 'Div', 'Erf', 'Gather', 'Gemm', 'MatMul', 'Mul', 'Pow', 'ReduceMean', 'ReduceSum', 'Reshape', 'Sigmoid', 'Slice', 'Softmax', 'Sqrt', 'Squeeze', 'Sub', 'Tanh', 'Transpose', 'Unsqueeze']
