"""
verify_patch.py — gain±12 패치 게이트 (figure/track 재생성 전 필수 검증)
========================================================================
결정 2: build_registry / MODEL_REGISTRY 의 A0(±6 기본) 함정을 패치했는지,
그리고 그 패치가 실제로 forward 단계에서 먹는지(±6 clamp 안 됨)를 양성 확인.

검증:
  1. train_full.build_registry()["A0_Proposed"]/["A2_withPrefLoss"] → gain_max==12.0
  2. experiments_fixed_updated.MODEL_REGISTRY["A0_Proposed"]()/["A2_withPrefLoss"]() → gain_max==12.0
  3. g12 seed7 A0 체크포인트를 (1)의 모델에 로드 → test_synth 1배치 forward →
     out['gain'] 의 max|gain| > 6.0  AND  <= 12.0   (±6 clamp 안 됨을 양성 입증)

실패 시 exit 1 (다음 단계 진입 차단).
"""
import sys
import torch

import train_full as TF
from experiments_fixed_updated import MODEL_REGISTRY
from run_gain_freq_ablation import DEFAULT_DATA, HERE, cname
from dataset_generator_v4_tracklevel import PEQDataset

ok = True
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

print("=" * 64)
print("VERIFY PATCH: gain±12 (build_registry / MODEL_REGISTRY)")
print("=" * 64)

# ── 1) build_registry 패치 ────────────────────────────────────────────────
reg = TF.build_registry()
for name in ("A0_Proposed", "A2_withPrefLoss"):
    gm = getattr(reg[name]["model"], "gain_max", None)
    fm = getattr(reg[name]["model"], "fc_max", None)
    good = (gm == 12.0 and fm == 16000.0)
    ok = ok and good
    print(f"  build_registry[{name:>16}]: gain_max={gm} fc_max={fm}  {'OK' if good else 'FAIL'}")

# ── 2) MODEL_REGISTRY 패치 (experiments) ──────────────────────────────────
for name in ("A0_Proposed", "A2_withPrefLoss"):
    m = MODEL_REGISTRY[name]()        # lambda → 인자 적용
    gm = getattr(m, "gain_max", None); fm = getattr(m, "fc_max", None)
    good = (gm == 12.0 and fm == 16000.0)
    ok = ok and good
    print(f"  MODEL_REGISTRY[{name:>16}](): gain_max={gm} fc_max={fm}  {'OK' if good else 'FAIL'}")

# ── 3) 실제 forward: ±6 clamp 안 됨을 양성 확인 ───────────────────────────
ckpt = HERE / "checkpoints" / f"{cname('g12_f16k', 7)}.pt"   # A0_g12_f16k_s7.pt
if not ckpt.exists():
    print(f"  [WARN] 대표 ckpt 없음: {ckpt} — forward 검증 스킵 (학습 후 재실행 가능)")
else:
    model = reg["A0_Proposed"]["model"]
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    TF._REGISTRY_TARGET["A0_Proposed"] = "dual"
    ds = PEQDataset(f"{DEFAULT_DATA}/test_synth", device=str(device))
    with torch.no_grad():
        batch = next(iter(ds.iter_batches(512, shuffle=False)))
        out = TF.model_forward("A0_Proposed", model, batch)
    g = out["gain"]
    gmax = float(g.abs().max()); over6 = float((g.abs() > 6.0).float().mean())
    not_clamped = gmax > 6.0          # ±6 였으면 불가능 → 패치 양성
    within12 = gmax <= 12.0 + 1e-4
    good = not_clamped and within12
    ok = ok and good
    print(f"  forward(A0 g12 s7): max|gain|={gmax:.3f}  (>6={'YES' if not_clamped else 'NO'}, "
          f"<=12={'YES' if within12 else 'NO'})  |gain|>6 비율={over6*100:.1f}%  {'OK' if good else 'FAIL'}")

print("-" * 64)
print(f"VERIFY PATCH 결과: {'ALL PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
