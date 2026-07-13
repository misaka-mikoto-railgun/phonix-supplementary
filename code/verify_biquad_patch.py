"""
verify_biquad_patch.py — AC_Biquad gain±12 패치 게이트
=====================================================
검증:
  1. AC{1,2,3}_*_Biquad(gain_max=12.0) 인스턴스 → self._gain_max==12.0
  2. forward 반환 dict 에 'gain' 키 존재(패치 확인)
  3. ±6 체크포인트를 gain_max=12 인스턴스에 로드 → forward → max|gain| > 6 AND ≤ 12
     (±6 로 학습된 모델도 ±12 bound 에서는 6 초과 출력 = ±6 가 binding 이었던 증거,
      동시에 ±12 로 clamp 안 됨을 양성 확인)
실패 시 exit 1.
"""
import sys
import torch

from arch_biquad import BIQUAD_REGISTRY
from run_gain_freq_ablation import ORIG_ROOT, DEFAULT_DATA
from dataset_generator_v4_tracklevel import PEQDataset

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
FULL = ORIG_ROOT / "checkpoints" / "full"
CKPT = {
    "AC1_BiLSTM_Biquad":   "AC1_BiLSTM_Biquad.pt",
    "AC2_GRU_Biquad":      "AC2_GRU_Biquad.pt",
    "AC3_Conformer_Biquad":"AC3_Conformer_Biquad.pt",
}

ds = PEQDataset(f"{DEFAULT_DATA}/test_synth", device=str(device))
batch = next(iter(ds.iter_batches(512, shuffle=False)))

print("=" * 70)
print("VERIFY BIQUAD PATCH: gain±12 (생성자 인자화 + forward dict)")
print("=" * 70)
ok = True
for name, ckf in CKPT.items():
    m = BIQUAD_REGISTRY[name](gain_max=12.0).to(device).eval()
    attr_ok = abs(getattr(m, "_gain_max", -1) - 12.0) < 1e-9
    ck = torch.load(FULL / ckf, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    m.load_state_dict(state, strict=False)
    with torch.no_grad():
        out = m(batch["features"], batch["room_response"], batch["mode_id"], batch["band_gains"])
    has_gain = out.get("gain") is not None
    g = out["gain"]
    gmax = float(g.abs().max()); over6 = float((g.abs() > 6.0).float().mean())
    not_clamped = gmax > 6.0; within12 = gmax <= 12.0 + 1e-3
    good = attr_ok and has_gain and not_clamped and within12
    ok = ok and good
    print(f"  [{name:>22}] _gain_max={getattr(m,'_gain_max',None)} gain_key={'Y' if has_gain else 'N'} "
          f"max|gain|={gmax:.3f} (>6={'Y' if not_clamped else 'N'},<=12={'Y' if within12 else 'N'}) "
          f"|gain|>6={over6*100:.1f}%  {'OK' if good else 'FAIL'}")
print("-" * 70)
print(f"VERIFY BIQUAD PATCH 결과: {'ALL PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
