# Inline number change list — A0 = gain ±12 (fc 16 kHz 유지)

대상 원고: `jaes_optimized.tex` (read-only, **본 목록은 수정 지시서일 뿐 원고를 고치지 않음**)

## 신규값 출처 (모두 revision_gain_freq/results/)
- **A0 3-seed 집계**: `gain_freq_summary_A0_test_synth.json`, `gain_freq_summary_A0_test_real.json`
  - synth: LSD **1.095±0.116**, DMR **0.929**, CosSim **0.974**
  - real:  LSD **1.792±0.182**, DMR **0.891**, CosSim **0.930**
  - domain gap = real−synth = **0.697** (원본 0.499)
- **A2 3-seed 집계**: `gain_freq_summary_A2_test_*.json`
  - synth LSD **1.329±0.368**, DMR **0.933**, CosSim **0.958**;  real LSD **1.981±0.301**, DMR **0.889**
- **대표 seed 7 단일값**(figure/experiments 표/단일-cell용): A0 synth LSD **1.028** / DMR 0.940 / CosSim 0.977, real LSD **1.687** / DMR 0.904; A2 synth **1.852** / real **2.404**
- **Paired 3-seed**: `paired_stats_3seed_test_synth.json`
- **Track-level 3-seed (N=1306)**: `track_stats_3seed_test_synth.json`

> 표기 규칙: 표 cell 의 A0/A2 행은 **3-seed mean±std** 를 primary 로 권장. 단, 원본이 단일값+CI 형식([1.423,1.462])인 cell 은 대표 seed7 단일값으로 바꾸고 캡션에 seed 정책 명시하는 방안도 가능(둘 중 택일은 저자 판단). 아래 "신규값"은 3-seed 기준으로 적되, 단일-cell 대안을 병기.

---

## A. gain 제약 서술 (필수, 1곳)
| line | 항목 | 기존 | 신규 |
|------|------|------|------|
| 310 | Stage B 필터 파라미터 범위 | `gain $\in [-6, +6]$~dB` | **`gain $\in [-12, +12]$~dB`** (`$f_c \in [80,16000]$`·`$Q\in[0.3,8.0]$` **그대로**) |

---

## B. 직접 A0 수치 — LSD/DMR/CosSim (synth)
| line | 항목 | 기존 | 신규(3-seed) | 단일-cell 대안(s7) |
|------|------|------|--------------|---------------------|
| 70–71 | abstract | LSD 1.442 / DMR 0.928 / CosSim 0.960 | **1.095±0.116 / 0.929 / 0.974** | 1.028/0.940/0.977 |
| 534 | 본문 결과 | LSD 1.442, DMR 0.928, CosSim 0.960 | 동일 신규 | — |
| 577 | **tab:main** A0 행 | `1.442 & [1.423,1.462] & 0.928 & 0.960 & 3.4e-4 & 203,447` | LSD **1.095±0.116**, DMR 0.929, CosSim 0.974 (RTF 3.4e-4·params 203,447 **불변**) | `1.028 & [CI] & 0.940 & 0.977` (table1_main_results.tex 참조) |
| 591 | "LSD rises from 1.442 to 5.329" | 1.442 | **1.095** (5.329=A3 불변) |
| 622 | **tab:ablation** A0 행 | `1.442 & 0.928 & 0.960` | **1.095±0.116 / 0.929 / 0.974** | 1.028/0.940/0.977 |
| 776 | "0.853–0.870 vs.\ 1.442"·"0.945–0.947 vs.\ 0.928" | 1.442 / 0.928 | **1.095 / 0.929** |
| 844 | "Relative to A0 (LSD 1.442)" | 1.442 | **1.095** |
| 987 | "collapses DMR from 0.928 to 0.509" | 0.928 | **0.929** (0.509=A3 불변) |
| 1117–1118 | conclusion DMR 0.928, LSD 1.442 | 0.928 / 1.442 | **0.929 / 1.095** (latency 1.354ms·params **불변**) |

## C. 직접 A0 수치 — real / OOD
| line | 항목 | 기존 | 신규 |
|------|------|------|------|
| 271 | "4.875 vs 1.941 for A0" | real 1.941 | **1.792±0.182** |
| 596–597 | "A0: 1.941" | 1.941 | **1.792** |
| 713 | "real LSD 1.941~dB and real DMR 0.885" | 1.941 / 0.885 | **1.792 / 0.891** |
| 1124 | "4.875 vs 1.941 with them" | 1.941 | **1.792** |
| 762 | **tab:ood** A0 행 `1.442 & 0.928 & 1.941 & 0.885 & 1.781 & 1.843` | synth/DMR/real/realDMR/BUT/OpenAIR | **1.095 / 0.929 / 1.792 / 0.891 / ⟨BUT⟩ / ⟨OpenAIR⟩** — source-wise BUT/OpenAIR(1.781/1.843)은 **RECOMPUTE 필요**(미산출, 아래 G) |

## D. domain gap
| line | 항목 | 기존 | 신규 | 출처 |
|------|------|------|------|------|
| 711 | "domain gap (0.499~dB)" | **0.499** | **0.697** | A0 real−synth (gain_freq_summary_A0) |
| 717–718 | A1 gap (1.978), 2.5× | A1 항목 — A0 불변, 단 "2.5× A0's"의 분모(real 1.941→1.792)로 **배수 재계산**: 4.875/1.792 = **2.72×** | 4.875/1.792 |
| 713 | AC gap 0.540 | 0.540 (AC, 불변) | — |

---

## E. A0 vs A2 비교 (★ 관계 변화 — paired_stats_3seed_test_synth.json)
| line | 항목 | 기존 | 신규(3-seed) |
|------|------|------|--------------|
| 579 | tab:main A2 행 | `1.703 & [1.678,1.730] & 0.884 & 0.944` | LSD **1.329±0.368**, DMR 0.933, CosSim 0.958 |
| 601–603 | "A2 LSD 1.703, DMR 0.884 … d_z=−0.486(LSD), +0.404(DMR)" | 1.703/0.884; d_z −0.486/+0.404 | LSD **1.329±0.368**, DMR **0.933**; **d_z −0.19±0.56 (LSD)**, **+0.02 (DMR)** |
| 626 | tab:ablation A2 행 | `1.703 & 0.884 & 0.944` | **1.329±0.368 / 0.933 / 0.958** |
| 645–646 | "A0 vs A2 … d_z −0.486(LSD), +0.404(DMR), Win 67.3%" | −0.486 / +0.404 / 67.3% | **d_z −0.19±0.56 / +0.02±0.35**, **Win 57.9±24.5% (LSD)** / 41.7±13.2% (DMR) |
| 686 | **tab:stats** A0 vs A2 LSD | `−0.261 & −0.486 & 67.3%` | **Δ −0.234±0.519 / d_z −0.19±0.56 / Win 57.9±24.5%** |
| (tab:stats) | A0 vs A2 DMR | `+0.044 & +0.404 & 49.6%` | **Δ −0.004±0.021 / d_z +0.02±0.35 / Win 41.7±13.2%** (★ DMR 부호 역전: A2 미세 우위) |

> 서사 전환: "A0가 A2보다 낫다" → **"평균상 동등(LSD 차 0.23, d_z 약함), 단 A2는 seed 불안정(std 0.37 vs A0 0.12)"**.

## F. A0 vs AC 비교 (★ 격차 절반 축소 — paired_stats_3seed_test_synth.json)
| line | 항목 | 기존 | 신규(3-seed) |
|------|------|------|--------------|
| 651 | "AC1–AC3 attain 0.57–0.59 dB lower LSD (d_z ≈ +1.16 to +1.19)" | 0.57–0.59 / +1.16~+1.19 | **0.22–0.24 dB lower / d_z +0.62~+0.70** |
| 664 | track "0.574–0.592 dB lower per-track LSD than A0" | 0.574–0.592 | **0.224–0.243** (track_stats json) |
| 694 | **tab:stats** A0 vs AC1–3 LSD | `+0.572~+0.589 & +1.16~+1.19 & 7.6–8.7%` | **Δ +0.225~+0.242 / d_z +0.62~+0.70 / Win 20.5–23.5%** |
| (tab:stats) | A0 vs AC1–3 DMR | `−0.017~−0.019 & −0.54~−0.56 & 22.8–24.2%` | **Δ −0.016~−0.018 / d_z −0.23~−0.28 / Win 29.8–32.6%** |
| 776 | "0.853–0.870 vs 1.442" | 1.442 | **1.095** (B에도 기재) |
| 887–892 | **fig:arch_compare 표** A0 1.442; AC Δ −0.589/−0.582/−0.572; Win 52.2/52.0/53.1% | A0 1.442, AC Δ vs A0 | A0 **1.095**(또는 s7 1.028); AC Δ **−0.242/−0.235/−0.225**; Win% **RECOMPUTE**(아래 G) |

## G. RECOMPUTE 필요 (아직 미산출 — 추가 실행 시 채움)
| line | 항목 | 기존 | 비고 |
|------|------|------|------|
| 762 | tab:ood A0 source-wise BUT/OpenAIR real LSD | 1.781 / 1.843 | test_real를 BUT/OpenAIR로 분할 평가 필요(rir_map.json) |
| 764, 963–964 | A2 source-wise(2.243 등)·tab(paired-mode) A2/A0 행 | — | A2 g12 기준 재평가 필요 |
| 887–892 | fig:arch_compare AC Win% (52%대) | 52.2/52.0/53.1% | per-sample DMR-tie 기반 Win 재계산 |
| 961 | (mode-switch/per-mode 표) A0 1.442/1.850/1.411 | — | 해당 평가 스크립트 재실행 필요 |
| 1122 | "d_z = +2.376 vs removing it" (A0 vs A3 DMR) | +2.376 | 신규 paired: **+2.32** (paired json, A0 vs A3 DMR d_z) |

## H. 불변 (변경 금지 — A0 영향 없음)
- params 203,447 (line 577,579,794,887,1071–1072) — gain 변경은 파라미터 수 무관
- RTF/latency 3.4e-4 / 1.354 ms (line 577,1072,1118) — 출력단 변환만 바뀜
- E3/E4/A1/A3/AC 자체 행, A3 5.329·DMR 0.509, AC LSD 0.853–0.870
- fc 범위 [80,16000], Q [0.3,8.0]

## I. 기타 A0-파생 (강화 방향, 표시만)
| line | 항목 | 기존 | 신규 | 출처 |
|------|------|------|------|------|
| 643 | "LSD d_z −1.08 to −4.23" (A0 vs ablations 범위) | −1.08~−4.23 | **−1.47~−4.32** (A0 vs A1~A3) | paired json |
| 661 | track "1.254 dB lower per-track LSD than A1" | 1.254 | **1.604** | track_stats json |
| 690,692 | tab:stats A0 vs E3/E4 LSD | −4.522/−4.366 | **−4.869 / −4.714** (d_z −4.03/−4.37) | paired json |
| 689 | A0 vs A3 DMR d_z +2.376 | +2.376 | **+2.32** | paired json |

---

## J. sec:ac_fitting (Biquad-Constrained Comparison) — AC_Biquad ±12 (출처: `ac_biquad_table.json`)
Option C AC1/2/3 단일-seed(42) ±12 재학습, "vs A0"는 A0 3-seed mean=1.095. %<JND = per-sample |AC_Biquad_LSD − A0_LSD|<0.5(원본 정의 동일).

| line | 항목 | 기존(±6) | 신규(±12) |
|------|------|----------|-----------|
| tab:ac_fitting [901] | AC1_Biquad LSD / vsA0 / %<JND | 1.039 [1.027,1.051] / −0.403 / 58.1% | **1.010 [0.999,1.021] / −0.085 / 88.8%** |
| [902] | AC2_Biquad | 1.005 [0.993,1.016] / −0.438 / 57.5% | **1.017 [1.006,1.029] / −0.078 / 88.7%** |
| [903] | AC3_Biquad | 1.009 [0.997,1.020] / −0.434 / 56.9% | **1.010 [0.999,1.021] / −0.085 / 88.3%** |
| [897] Option A | SciPy-fit LSD (AC2_GRU) | 1.026 [—] / penalty 0.166 | **0.918 [0.908,0.929] / penalty 0.058** (dense raw 0.860) |
| 본문 [834–836] | "1.00±0.02"; AC별 1.039/1.005/1.009; penalty 0.14–0.19 | — | **Option C mean≈1.01; AC별 1.010/1.017/1.010; penalty 0.14–0.16** |
| 본문 [838–840] ★ | "Option A·C가 0.02dB 내 일치 = architecture-invariant ceiling≈1.0" | 0.02dB 일치 | **⚠️ ±12에서 불일치: Option A 0.918 vs Option C ~1.01 (~0.09dB 차). 서사 수정 필요** (아래 주석) |
| 본문 [844] | "Relative to A0 (LSD 1.442)" | 1.442 | **1.095** |
| 본문 [847–848] | "56.9–58.1% below JND" (=%<JND vs A0) | 56.9–58.1% | **88.3–88.8%** (A0가 1.095로 낮아져 biquad와 더 근접) |
| 본문 [858] | latency 1.61–2.41 ms | — | **불변** |
| tab:perceptual biquad %<JND | (동일 정의) | 56.9–58.1% | **88.3–88.8%** |
| **캡션(seed 비대칭)** | — | 추가: *"AC_Biquad: single-seed(42)+bootstrap-CI (원본 방법론); A0/A2: 3 seeds. 'vs A0' uses A0 3-seed mean (1.095)."* |

### ★ 핵심: A0 vs AC_Biquad ±12 격차
| | LSD | A0(1.095) 대비 격차 |
|--|-----|---------------------|
| AC1_Biquad ±12 | 1.010 | **0.085** |
| AC2_Biquad ±12 | 1.017 | **0.078** |
| AC3_Biquad ±12 | 1.010 | **0.085** |

원본(±6, A0=1.442): 격차 0.40–0.44dB → **신규(둘 다 ±12): 0.08–0.09dB** (격차 ~5배 축소). **동일 ±12 + 7-band biquad deployment 제약 하에서 A0가 AC 대비 0.1dB 미만**.

### saturation 교차검증 (A0 패턴과 동일한가)
| AC_Biquad | ±6 자기경계 \|g\|>5.8 | ±12 \|g\|>6 | ±12 자기경계 \|g\|>11.8 |
|-----------|----------------------|-------------|------------------------|
| AC1 | 9.7% | **38.0%** | 0.0% |
| AC2 | 2.8% | **46.0%** | 0.0% |
| AC3 | 3.3% | **64.0%** | 0.0% |

→ ±6에서 자기경계 saturation은 낮았으나, ±12 허용 시 **38–64%가 6 초과 사용**(A0와 동일하게 ±6가 binding). 단 Option C LSD는 거의 불변(~1.01) = "end-to-end retrain ceiling은 gain bound에 robust". A0(±6→±12 LSD 1.442→1.095 크게 개선)와 대조 — A0는 dense Stage A+B 구조라 gain 완화 이득이 크고, AC_Biquad(C)는 이미 ceiling 근처라 추가 이득 작음.

### ★ Option A vs C ceiling — 원본 "0.02dB 내 일치" 주장 ±12에서 깨짐 (서사 수정 필요)
| | ±6 (원본) | ±12 (신규) | 출처 |
|--|-----------|------------|------|
| Option A (post-hoc SciPy fit, AC2_GRU) | 1.026 (penalty 0.166) | **0.918** [0.908,0.929] (penalty 0.058) | biquad_a_optionA.log |
| Option C (end-to-end retrain, AC2_GRU) | 1.005 | **1.017** | ac_biquad_table.json |
| 둘 차이 | ~0.02dB (일치) | **~0.10dB (불일치)** | |

해석: ±6에서는 둘 다 gain-bound에 막혀 같은 ~1.0 ceiling에 수렴 → "architecture-invariant biquad ceiling" 주장 성립. **±12에서는 post-hoc 최적화(Option A 0.918)가 gain 여유를 더 적극 활용해 end-to-end 학습(Option C ~1.01)보다 ~0.1dB 낮음** → "Option A·C가 동일 ceiling" 교차검증 주장은 ±12에서 약화. 원고 [838–840]을 "post-hoc fitting이 ±12에서 retrain보다 우수 = 학습이 biquad 표현력을 완전히 활용하지 못함" 취지로 수정 권장. (단 fig:ac_fitting/표 모두 갱신됨: `fig_ac_fitting_AC2_GRU.{png,pdf}`)

---
**요약**: A0가 ±12로 개선되어 (i) E3/E4/A1/A3 대비 우위 강화, (ii) AC 대비 격차 절반 축소(샘플 LSD)·**AC_Biquad 대비는 0.4→0.08dB로 ~5배 축소**, (iii) A2 대비 동등~약우위(+분산 작아 안정). 모든 신규 effect size 는 json 에서 추적. source-wise/mode-switch/AC-Win%/Option-A 항목은 추가 평가 필요(Option A 진행중).
