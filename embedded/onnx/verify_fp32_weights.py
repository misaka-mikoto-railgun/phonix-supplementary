"""Reproduce the FP32 weight sizes reported for the STM32 deployment table,
and regression-check the parameter counts against the model definitions.

FP32 weights [KiB] = n_params * 4 / 1024, with n_params taken from manifest.json.

Regression (AC2 double-count guard):
  export_onnx.py must capture n_params BEFORE export-time graph surgery.
  install_gru_split (AC2) adds nn.GRU gru_a/gru_b while keeping the original
  model.gru, so counting parameters AFTER surgery double-counts the GRU
  (265,590 + 49,920 = 315,510). install_bilstm_unroll (AC1) stores unrolled
  weights as register_buffer (not parameters) so it is not double-counted.
  This test instantiates the models directly and asserts the true counts,
  then checks manifest.json agrees.

    python verify_fp32_weights.py
"""
import json
import sys
from pathlib import Path

# ── (1) 모델 정의로부터 실제 파라미터 수 (surgery 이전 = 정답) ───────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from arch_biquad import AC1_BiLSTM_Biquad, AC2_GRU_Biquad, AC3_Conformer_Biquad  # noqa: E402

EXPECTED = {
    "AC1_BiLSTM_Biquad":    (lambda: AC1_BiLSTM_Biquad(gain_max=12.0),    240_758),
    "AC2_GRU_Biquad":       (lambda: AC2_GRU_Biquad(gain_max=12.0),       265_590),  # ← not 315,510
    "AC3_Conformer_Biquad": (lambda: AC3_Conformer_Biquad(gain_max=12.0), 411_959),
}

print("[regression] instantiated params (gain_max=12):")
model_np = {}
for name, (build, exp) in EXPECTED.items():
    n = sum(p.numel() for p in build().parameters())
    model_np[name] = n
    status = "OK" if n == exp else "FAIL"
    print(f"  {name:24} {n:>10,}  (expected {exp:,})  {status}")
    assert n == exp, f"{name}: {n} != expected {exp} (export-time surgery double-count?)"

# ── (2) manifest.json 의 n_params 가 정의와 일치하는지 ──────────────────────
manifest = json.load(open("manifest.json", encoding="utf-8"))
print("\n[manifest] n_params  ->  FP32 KiB (= n_params*4/1024):")
for v in manifest["variants"]:
    kib = v["n_params"] * 4 / 1024
    print(f"  {v['name']:24} {v['n_params']:>10,} params  ->  {kib:8.1f} KiB")
    if v["name"] in model_np:
        assert v["n_params"] == model_np[v["name"]], (
            f"manifest {v['name']} n_params={v['n_params']} != model {model_np[v['name']]}")

print("\nAll parameter-count checks passed.")
