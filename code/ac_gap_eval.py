"""
ac_gap_eval.py — AC1/AC2/AC3 의 synth/real gap 재산출 (재학습 없음)
====================================================================
목적: 원본 Table 5 의 AC gap(0.646~0.706)이 이 클린 환경 평가 코드로 재현되는지
확인하고, g12_f16k 의 gap 0.697 과 동일 기준에서 비교.

AC 는 dense 출력(fc/gain 없음) → gain_max/fc_max 무관. TF.build_registry() 로
원본과 동일하게 인스턴스화하고, 원본 checkpoints/full/AC*.pt 를 read-only 로 로드.
드라이버 내장 evaluate(동일 LSD 정의) 재사용.
"""
import torch

import train_full as TF
from run_gain_freq_ablation import DEFAULT_DATA, ORIG_ROOT, evaluate
from dataset_generator_v4_tracklevel import PEQDataset

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
FULL_CKPT = ORIG_ROOT / "checkpoints" / "full"

# 평가 대상: dense AC 체크포인트 (Biquad 변형 아님)
AC = [
    ("AC1_BiLSTM",    "AC1_BiLSTM.pt"),
    ("AC2_GRU",       "AC2_GRU.pt"),
    ("AC3_Conformer", "AC3_Conformer.pt"),
]

registry = TF.build_registry()
for name, _ in AC:
    TF._REGISTRY_TARGET[name] = "dual"

ds_synth = PEQDataset(f"{DEFAULT_DATA}/test_synth", device=str(device))
ds_real  = PEQDataset(f"{DEFAULT_DATA}/test_real",  device=str(device))


def lsd_of(name, ckpt_file, ds):
    model = registry[name]["model"]
    cp = FULL_CKPT / ckpt_file
    ck = torch.load(cp, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=False)
    # cfg=None: AC 는 out['gain'] 없음 → evaluate 의 saturation 블록 미실행(CONFIGS 미참조)
    return evaluate(name, None, model, ds, device)


print("=" * 64)
print("AC variants — synth/real gap (LSD)   [원본 ckpt read-only, 재학습 없음]")
print("=" * 64)
print(f"{'model':>14} {'synth LSD':>10} {'real LSD':>10} {'gap':>8}")
print("-" * 46)
results = {}
for name, ckpt_file in AC:
    syn = lsd_of(name, ckpt_file, ds_synth)["lsd_mean"]
    rea = lsd_of(name, ckpt_file, ds_real)["lsd_mean"]
    gap = rea - syn
    results[name] = (syn, rea, gap)
    print(f"{name:>14} {syn:>10.4f} {rea:>10.4f} {gap:>8.4f}")
print("-" * 46)
print(f"{'g12_f16k(mean)':>14} {1.0950:>10.4f} {1.7922:>10.4f} {0.6972:>8.4f}  ← 비교 대상")
print(f"{'g6_f16k(base)':>14} {1.4423:>10.4f} {1.9412:>10.4f} {0.4989:>8.4f}")
print("\n원본 Table 5 AC gap 범위: 0.646 ~ 0.706")
