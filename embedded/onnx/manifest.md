# ONNX Export Manifest — STM32F405 latency benchmark

- generated: 2026-06-22T20:40:02
- opset: **13**, batch=1, dynamic_axes=False, dtype=float32
- backbone input: `feat [1, 32, 10]` (seq_len=32=4.0s, in_dim=10)
- conditional inputs: room_response[1,128], **mode_onehot[1,4] float**, band_gains[1,10]
- pref_curve `_interp` replaced by a constant matmul (removes searchsorted; parity preserved)
- mode embedding (nn.Embedding Gather) replaced by a one-hot matmul (ST Edge AI Core mistook the embedding table's leading dimension for a batch; the one-hot encoding is done on the host)
- standalone Pad folded into Conv.pads (fold_pad_into_conv), working around the Pad codegen bug in X-CUBE-AI 10.2.0. Causal pads=[(k-1)*dilation, 0]

| variant | group | onnx | ckpt | seed | gain_max | params | parity max|err| | max\|gain\| | gate | MACC(thop) | HARD blocker | VERIFY ops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_Proposed | A0 | a0.onnx | A0_g12_f16k_s7.pt | 7 | 12.0 | 203,447 | 1.95e-03 | 12.00 | PASS | 3344256 | - | ['Erf', 'ReduceSum', 'Softmax'] |
| A2_withPrefLoss | A2 | a2.onnx | A2_g12_f16k_s7.pt | 7 | 12.0 | 203,447 | 1.22e-03 | 12.00 | PASS | 3344256 | - | ['Erf', 'ReduceSum', 'Softmax'] |
| E3_Nercessian | E | e3_nercessian.onnx | E3_Nercessian.pt | 42 | 12.0 | 138,255 | 1.46e-03 | 7.19 | n/a (structurally +/-12 dB, so the clamp trap cannot apply) | 137472 | - | - |
| E4_Pepe | E | e4_pepe.onnx | E4_Pepe.pt | 42 | 12.0 | 50,351 | 1.46e-03 | 11.65 | n/a (structurally +/-12 dB, so the clamp trap cannot apply) | 1036288 | - | - |
| E5_Sequential | E | e5_sequential.onnx | E5_Sequential.pt | 42 | 12.0 | 138,255 | 1.95e-03 | 5.61 | n/a (structurally +/-12 dB, so the clamp trap cannot apply) | 137472 | - | - |
| AC1_BiLSTM_Biquad | AC | ac1_bilstm_biquad.onnx | AC1_BiLSTM_Biquad_g12.pt | None | 12.0 | 240,758 | 1.71e-03 | 10.81 | PASS | 3354496 | - | ['Erf'] |
| AC2_GRU_Biquad | AC | ac2_gru_biquad.onnx | AC2_GRU_Biquad_g12.pt | None | 12.0 | 265,590 | 6.10e-04 | 11.42 | PASS | 4980608 | - | ['Erf', 'GRU'] |
| AC3_Conformer_Biquad | AC | ac3_conformer_biquad.onnx | AC3_Conformer_Biquad_g12.pt | None | 12.0 | 411,959 | 9.77e-04 | 11.34 | PASS | 8562560 | - | ['Erf', 'ReduceSum', 'Softmax'] |

## Graph boundary and excluded post-processing

### A0_Proposed (`a0.onnx`)
- output format: room_corr[1,128] (dense room-correction dB) + fc/gain/q [1,7] (7-band biquad params)
- output shapes: {'room_corr': [1, 128], 'fc': [1, 7], 'gain': [1, 7], 'q': [1, 7]}
- graph boundary: input -> {room_corr, fc, gain, q}. peq_response (Gaussian), biquad coefficient computation and pref_curve synthesis are outside the graph.
- excluded post-processing (measured in C; second column of the comparison table): host/C: (1) pref_curve = band_gains linearly interpolated to 128 bins; (2) closed-form RBJ coefficients for the 7 bands, then combined with the dense room_corr. The Gaussian reconstruction is training-only and is not used here.
- onnx ops: ['Add', 'Concat', 'Conv', 'Div', 'Erf', 'Gemm', 'MatMul', 'Mul', 'Pow', 'ReduceMean', 'ReduceSum', 'Sigmoid', 'Slice', 'Softmax', 'Sqrt', 'Squeeze', 'Sub', 'Tanh', 'Transpose', 'Unsqueeze']

### A2_withPrefLoss (`a2.onnx`)
- output format: room_corr[1,128] (dense room-correction dB) + fc/gain/q [1,7] (7-band biquad params)
- output shapes: {'room_corr': [1, 128], 'fc': [1, 7], 'gain': [1, 7], 'q': [1, 7]}
- graph boundary: input -> {room_corr, fc, gain, q}. peq_response (Gaussian), biquad coefficient computation and pref_curve synthesis are outside the graph.
- excluded post-processing (measured in C; second column of the comparison table): host/C: (1) pref_curve = band_gains linearly interpolated to 128 bins; (2) closed-form RBJ coefficients for the 7 bands, then combined with the dense room_corr. The Gaussian reconstruction is training-only and is not used here.
- onnx ops: ['Add', 'Concat', 'Conv', 'Div', 'Erf', 'Gemm', 'MatMul', 'Mul', 'Pow', 'ReduceMean', 'ReduceSum', 'Sigmoid', 'Slice', 'Softmax', 'Sqrt', 'Squeeze', 'Sub', 'Tanh', 'Transpose', 'Unsqueeze']

### E3_Nercessian (`e3_nercessian.onnx`)
- output format: fc/gain/q [1,5] (5-band parametric PEQ params)
- output shapes: {'fc': [1, 5], 'gain': [1, 5], 'q': [1, 5]}
- graph boundary: input -> {fc, gain, q}. response (Gaussian) and biquad coefficient computation are outside the graph.
- excluded post-processing (measured in C; second column of the comparison table): host/C: closed-form biquad coefficients for the 5 bands. The Gaussian reconstruction is training-only.
- onnx ops: ['Add', 'Gemm', 'Mul', 'ReduceMean', 'Relu', 'Sigmoid', 'Slice', 'Tanh']

### E4_Pepe (`e4_pepe.onnx`)
- output format: fc/gain/q [1,5] (5-band parametric PEQ params)
- output shapes: {'fc': [1, 5], 'gain': [1, 5], 'q': [1, 5]}
- graph boundary: input -> {fc, gain, q}. response (Gaussian) and biquad coefficient computation are outside the graph.
- excluded post-processing (measured in C; second column of the comparison table): host/C: closed-form biquad coefficients for the 5 bands. The Gaussian reconstruction is training-only.
- onnx ops: ['Add', 'Conv', 'Gemm', 'GlobalAveragePool', 'Mul', 'Relu', 'Sigmoid', 'Slice', 'Squeeze', 'Tanh', 'Transpose']

### E5_Sequential (`e5_sequential.onnx`)
- output format: fc/gain/q [1,5] (the internal E3 room corrector; the E2 preference part is a fixed table, not a network)
- output shapes: {'fc': [1, 5], 'gain': [1, 5], 'q': [1, 5]}
- graph boundary: input -> {fc, gain, q} (E3). The E2 table sum is outside the graph.
- excluded post-processing (measured in C; second column of the comparison table): host/C: closed-form biquad coefficients for the 5 bands (E3), summed with the fixed mode profile (E2).
- onnx ops: ['Add', 'Gemm', 'Mul', 'ReduceMean', 'Relu', 'Sigmoid', 'Slice', 'Tanh']

### AC1_BiLSTM_Biquad (`ac1_bilstm_biquad.onnx`)
- output format: room_corr[1,128] (dense room-correction dB) + fc/gain/q [1,7] (7-band biquad params)
- output shapes: {'room_corr': [1, 128], 'fc': [1, 7], 'gain': [1, 7], 'q': [1, 7]}
- graph boundary: input -> {room_corr, fc, gain, q}. peq_response (Gaussian), biquad coefficient computation and pref_curve synthesis are outside the graph.
- excluded post-processing (measured in C; second column of the comparison table): host/C: (1) pref_curve = band_gains linearly interpolated to 128 bins; (2) closed-form RBJ coefficients for the 7 bands, then combined with the dense room_corr. The Gaussian reconstruction is training-only and is not used here.
- onnx ops: ['Add', 'Concat', 'Conv', 'Div', 'Erf', 'Gather', 'Gemm', 'MatMul', 'Mul', 'Pow', 'ReduceMean', 'Reshape', 'Sigmoid', 'Slice', 'Sqrt', 'Sub', 'Tanh', 'Transpose']

### AC2_GRU_Biquad (`ac2_gru_biquad.onnx`)
- output format: room_corr[1,128] (dense room-correction dB) + fc/gain/q [1,7] (7-band biquad params)
- output shapes: {'room_corr': [1, 128], 'fc': [1, 7], 'gain': [1, 7], 'q': [1, 7]}
- graph boundary: input -> {room_corr, fc, gain, q}. peq_response (Gaussian), biquad coefficient computation and pref_curve synthesis are outside the graph.
- excluded post-processing (measured in C; second column of the comparison table): host/C: (1) pref_curve = band_gains linearly interpolated to 128 bins; (2) closed-form RBJ coefficients for the 7 bands, then combined with the dense room_corr. The Gaussian reconstruction is training-only and is not used here.
- onnx ops: ['Add', 'Concat', 'Conv', 'Div', 'Erf', 'GRU', 'Gather', 'Gemm', 'MatMul', 'Mul', 'Pow', 'ReduceMean', 'Reshape', 'Sigmoid', 'Sqrt', 'Squeeze', 'Sub', 'Tanh', 'Transpose']

### AC3_Conformer_Biquad (`ac3_conformer_biquad.onnx`)
- output format: room_corr[1,128] (dense room-correction dB) + fc/gain/q [1,7] (7-band biquad params)
- output shapes: {'room_corr': [1, 128], 'fc': [1, 7], 'gain': [1, 7], 'q': [1, 7]}
- graph boundary: input -> {room_corr, fc, gain, q}. peq_response (Gaussian), biquad coefficient computation and pref_curve synthesis are outside the graph.
- excluded post-processing (measured in C; second column of the comparison table): host/C: (1) pref_curve = band_gains linearly interpolated to 128 bins; (2) closed-form RBJ coefficients for the 7 bands, then combined with the dense room_corr. The Gaussian reconstruction is training-only and is not used here.
- onnx ops: ['Add', 'Concat', 'Conv', 'Div', 'Erf', 'Gather', 'Gemm', 'MatMul', 'Mul', 'Pow', 'ReduceMean', 'ReduceSum', 'Reshape', 'Sigmoid', 'Slice', 'Softmax', 'Sqrt', 'Squeeze', 'Sub', 'Tanh', 'Transpose', 'Unsqueeze']
