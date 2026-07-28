# INT8 PTQ 정확도 저하 — test_synth (n=3000)

QDQ int8 onnx(onnxruntime 실행) vs fp32 체크포인트. 동일 test_synth·동일 metric·동일 post-proc.
- **LSD** = dual-target, sample별 √mean((pred−dual_target)²) 평균 (낮을수록 좋음)
- **CosSim** = heard(=pred−room_target) vs pref_target (높을수록 좋음)
- pred 는 그래프 경계 출력(room_corr/fc/gain/q)에서 fp32와 동일 수식으로 재구성.

| variant | fp32 LSD | int8 LSD | ΔLSD | ΔLSD% | fp32 cos | int8 cos | Δcos |
|---|---|---|---|---|---|---|---|
| A0  | 1.0279 | 1.1128 | +0.0850 | +8.3% | 0.9772 | 0.9737 | −0.0035 |
| AC2 (GRU)      | 1.0172 | 1.0561 | +0.0389 | +3.8% | 0.9798 | 0.9783 | −0.0015 |
| AC3 (Conformer)| 1.0103 | 1.0393 | +0.0291 | +2.9% | 0.9802 | 0.9788 | −0.0014 |
| E3 (Nercessian)| 5.9641 | 5.9657 | +0.0016 | +0.0% | 0.0108 | 0.0104 | −0.0004 |
| E4 (Pepe)      | 5.8086 | 5.8085 | −0.0001 | −0.0% | 0.0319 | 0.0325 | +0.0006 |

## fp32 기준선 교차검증 (논문/revision 수치와 일치)
- A0: 1.028 (test_synth, s7) ↔ revision g12 3-seed 1.095±0.116, val 1.43.
- E3/E4: 5.96/5.81 (test) ↔ 논문 val 5.94/5.78.
- AC2/AC3 biquad g12: 1.02/1.01 ↔ ac_biquad_table g12 ≈1.01.
→ fp32 파이프라인이 논문 기준선과 정합. Δ는 동일 test 파이프라인 내 순수 양자화 효과.

## 해석
- **A0**: 상대 저하 최대(+8.3%, +0.085 dB)지만 절대 LSD 1.11로 여전히 baseline 대비 압도적. cos −0.0035로 무시 가능. (room_corr dense head + PEQ로 양자화 민감도가 변종 중 가장 큼.)
- **AC2/AC3**: +2.9~3.8% (+0.03 dB), cos −0.0015 수준. 거의 무손실.
- **E3/E4**: 저하 ≈0 — 이미 LSD~6/cos~0.01의 부적합 모델이라 양자화가 바꿀 여지가 없음(정보성 낮음).
- 종합: **INT8 PTQ는 제안모델/주요 변종에서 LSD 0.03~0.09 dB, cos <0.004의 사실상 무손실**. 앞서의 "양자화 시 무거운 변종도 F405 적재" 결과와 묶어 부록 근거로 적합.

## 비고
- E3/E4 절대 LSD가 큰 건 원래 그 baseline들의 성능(논문과 동일). 양자화 저하 평가의 대상은 Δ.
- int8 onnx 유지: a0_int8 / e3_nercessian_int8 / e4_pepe_int8 / ac2_gru_biquad_int8 / ac3_conformer_biquad_int8.
- 재현: `python eval_int8.py` (onnxruntime per-sample 실행 + fp32 PyTorch).
