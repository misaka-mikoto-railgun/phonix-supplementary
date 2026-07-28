"""quantize_ptq.py — calib.npz 로 onnxruntime static(QDQ) INT8 양자화 → ST Edge AI Core import용.

ST Edge AI Core 2.2.0 흐름: 외부 onnxruntime 으로 QDQ int8 모델 생성 → stedgeai 가 그 QDQ import.
스킴: ss/sa per-channel (activation asym int8, weight sym int8) = ST 권장.
calibrate_method = MinMax (calib 에 의도적으로 넣은 극단까지 범위로 반영; percentile 은 극단 clip 우려).

사용: python quantize_ptq.py a0 [ac2_gru_biquad ...]
"""
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (quantize_static, QuantType, QuantFormat,
                                      CalibrationMethod, CalibrationDataReader)
from onnxruntime.quantization.shape_inference import quant_pre_process

HERE = Path(__file__).resolve().parent
CAL = np.load(HERE / "calib.npz")


class NpzReader(CalibrationDataReader):
    """calib.npz 를 모델 입력명에 맞춰 feed. (E 모델은 feat 만 사용.)"""
    def __init__(self, model_path):
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.names = [i.name for i in sess.get_inputs()]
        n = CAL[self.names[0]].shape[0]
        self.data = [{nm: CAL[nm][k:k + 1] for nm in self.names} for k in range(n)]
        self.it = iter(self.data)
    def get_next(self):
        return next(self.it, None)
    def rewind(self):
        self.it = iter(self.data)


def quantize(model):
    inp = str(HERE / f"{model}.onnx")
    pre = str(HERE / f"_{model}_infer.onnx")
    out = str(HERE / f"{model}_int8.onnx")
    quant_pre_process(inp, pre, skip_symbolic_shape=True)
    quantize_static(
        pre, out, NpzReader(pre),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        per_channel=True, calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    Path(pre).unlink(missing_ok=True)
    return out


if __name__ == "__main__":
    models = sys.argv[1:] or ["a0"]
    for m in models:
        o = quantize(m)
        print(f"  {m} -> {Path(o).name}")
