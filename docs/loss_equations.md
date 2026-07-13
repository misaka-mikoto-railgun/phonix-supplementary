# M5 — Loss function equations (code-faithful, for R2 "specify as equations")

**무결성 원칙**: 아래 수식은 전부 `model.py`의 `DualObjectiveEQLoss`에서 **그대로 전사**한 것이며,
prose에서 역산하거나 임의로 작성한 항은 **없다**. 각 식에 원본 코드 라인을 병기해 1:1 검증 가능.
손실 함수형은 gain ±6→±12 revision에서 **변경 없음**(같은 `a0_proposed_loss` 사용).

코드 출처: `../model.py` (`DualObjectiveEQLoss`, lines 204–437), 하이퍼파라미터: `../train_full.py:127–139`.

---

## 0. 표기 / 그리드
- $\hat{R}\in\mathbb{R}^{F}$: 예측 응답(dB), $T$: 타깃(dB). bin index $k=1,\dots,F$, $F=128$.
- 주파수 그리드: log-spaced $f_k\in[20,\,24000]$ Hz (`model.py:257–262`, `f_max=24000` default = Nyquist). 타깃·예측·loss 그리드 동일.
- $\langle\cdot\rangle_{b,k}$ = batch·bin 평균, $\bar{x}=\tfrac1F\sum_k x_k$ (per-sample bin 평균).

## 1. Top-level (`forward`, lines 399–404)
$$
\mathcal{L}=\lambda_\text{final}\,\mathcal{L}_\text{final}(\hat{R},T_\text{dual})
+\lambda_\text{room}\,\mathcal{L}_\text{room}(\hat{R}_\text{room},T_\text{room})
+\lambda_\text{pref}\mathcal{L}_\text{pref}+\lambda_\text{dir}\mathcal{L}_\text{dir}
$$
A0_Proposed: $(\lambda_\text{final},\lambda_\text{room},\lambda_\text{pref},\lambda_\text{dir})=(1.0,\;0.35,\;0,\;0)$.

> $\mathcal{L}_\text{room}$은 **동일 weighted-curve 함수형**을 $(\hat{R}_\text{room\,corr},T_\text{room})$에 적용 (line 367). plain MSE/L1 아님.

## 2. Weighted-curve loss (`_weighted_curve_loss`, lines 288–341) — final·room 공통
$$
\mathcal{L}_\bullet=\mathcal{L}_\text{mag}+\lambda_\text{shape}\mathcal{L}_\text{shape}
+\lambda_\text{grad}\mathcal{L}_\text{grad}+\lambda_\text{curv}\mathcal{L}_\text{curv}
+\lambda_\text{mean}\mathcal{L}_\text{mean}
$$
$(\lambda_\text{shape},\lambda_\text{grad},\lambda_\text{curv},\lambda_\text{mean})=(0.20,0.20,0.08,0.03)$; $\mathcal{L}_\text{mag}$ 계수 $=1$.

### (1) Magnitude `main` (lines 290–307)
$$
\mathcal{L}_\text{mag}=\big\langle \widetilde{W}_k\,(\hat{R}_k-T_k)^2\big\rangle_{b,k},\qquad
\widetilde{W}_k=\frac{W_k}{\frac1F\sum_j W_j+\varepsilon},\ \ \varepsilon=10^{-8}
$$
$$
W_k=w^A_k\cdot w^E_k\cdot(1+\alpha|T_k|)\cdot\big(1+\beta\,|T_k-T_{k-1}|\big),\quad \alpha=0.10,\ \beta=0.30
$$
(grad항 첫 bin $k{=}1$은 0; `target_grad_mag[:,0]=0`, line 295–296. 정규화는 per-sample, line 305.)

**A-weighting** $w^A_k$ (IEC 61672, 선형진폭 변환 후 평균정규화; lines 265–270):
$$
A_\text{dB}(f)=20\log_{10}\!\frac{12194^2\,f^4}{(f^2+20.6^2)\sqrt{(f^2+107.7^2)(f^2+737.9^2)}\,(f^2+12194^2)}+2.0
$$
$$
w^A_k=\frac{10^{A_\text{dB}(f_k)/20}}{\big\langle 10^{A_\text{dB}/20}\big\rangle_k}
$$
(A0: `use_perceptual=True` → base$=w^A_k w^E_k$. room-only baseline E3/E4: `use_perceptual=False` → base$=w^E_k$, lines 299–302.)

**Edge boost** $w^E_k$ (lines 272–274):
$$
w^E_k=\begin{cases}1.30 & f_k\le 300\,\text{Hz}\\[2pt] 1.15 & f_k\ge 8000\,\text{Hz}\\[2pt] 1.0 & \text{otherwise}\end{cases}
$$

### (2) Shape (lines 309–311) — mean 제거 후 MSE
$$
\mathcal{L}_\text{shape}=\big\langle\big((\hat{R}_k-\bar{\hat{R}})-(T_k-\bar{T})\big)^2\big\rangle_{b,k}
$$
### (3) Gradient (lines 313–315) — 1차 차분 MSE
$$
\mathcal{L}_\text{grad}=\big\langle\big((\hat{R}_{k+1}-\hat{R}_k)-(T_{k+1}-T_k)\big)^2\big\rangle_{b,k}
$$
### (4) Curvature (lines 317–319) — 2차 차분 MSE
$$
\mathcal{L}_\text{curv}=\big\langle\big((\hat{R}_{k+1}-2\hat{R}_k+\hat{R}_{k-1})-(T_{k+1}-2T_k+T_{k-1})\big)^2\big\rangle_{b,k}
$$
### (5) Mean penalty (lines 321–324) — broadband offset MSE
$$
\mathcal{L}_\text{mean}=\big\langle(\bar{\hat{R}}-\bar{T})^2\big\rangle_{b}
$$

(A2 전용: $\mathcal{L}_\text{pref}=\mathcal{L}_\bullet(\hat{R}_\text{peq},\,T_\text{dual}-T_\text{room})$ line 380–382; $\mathcal{L}_\text{dir}=1-\cos(\hat{R}-T_\text{room},\,T_\text{pref})$ line 392–393. A0는 둘 다 $\lambda=0$.)

---

## 3. 본문용 안전 축약본 (권장 — principal in text, full weighting in code)

> Both $\mathcal{L}_\text{final}$ and $\mathcal{L}_\text{room}$ use a weighted log-spectral form. For a target $T$ ($T_\text{dual}$ and $T_\text{room}$, respectively),
> $$\mathcal{L}_\bullet=\big\langle \widetilde{W}_k(\hat{R}_k-T_k)^2\big\rangle+\lambda_\text{shape}\mathcal{L}_\text{shape}+\lambda_\text{grad}\mathcal{L}_\text{grad}+\lambda_\text{curv}\mathcal{L}_\text{curv}+\lambda_\text{mean}\mathcal{L}_\text{mean},$$
> where $W_k=w^A_k\,w^E_k(1+\alpha|T_k|)(1+\beta|\Delta T_k|)$ combines an A-weighting $w^A_k$, low/high band-edge boosts $w^E_k$, and target magnitude/gradient adaptivity, normalized to unit mean per sample ($\widetilde{W}_k$). The auxiliary terms are MSEs on the mean-removed curve ($\mathcal{L}_\text{shape}$), its first ($\mathcal{L}_\text{grad}$) and second ($\mathcal{L}_\text{curv}$) finite differences, and the broadband offset ($\mathcal{L}_\text{mean}$). The total objective is $\mathcal{L}=\lambda_\text{final}\mathcal{L}_\text{final}+\lambda_\text{room}\mathcal{L}_\text{room}$ with
> $(\lambda_\text{final},\lambda_\text{room},\lambda_\text{shape},\lambda_\text{grad},\lambda_\text{curv},\lambda_\text{mean})=(1,0.35,0.20,0.20,0.08,0.03)$.
> *The exact constants ($\alpha=0.1$, $\beta=0.3$; edge boosts $1.30/1.15$ at $300$ Hz $/\,8$ kHz) and the IEC-61672 A-weighting $w^A_k$ are given in the released training code.*

## 4. λ / 상수 표 (코드 확인됨)
| 기호 | 값 | 코드 |
|------|----|------|
| $\lambda_\text{final}$ | 1.0 | train_full:128 |
| $\lambda_\text{room}$ | 0.35 | train_full:129 |
| $\lambda_\text{pref},\lambda_\text{dir}$ | 0, 0 (A0) | train_full:130,134 |
| $\lambda_\text{shape},\lambda_\text{grad},\lambda_\text{curv},\lambda_\text{mean}$ | 0.20, 0.20, 0.08, 0.03 | train_full:131–135 |
| $\alpha$ (mag), $\beta$ (grad) | 0.10, 0.30 | train_full:137–138 |
| edge boost low/high | 1.30 / 1.15 | model.py:237–238 (default) |
| edge band | $\le300$ / $\ge8000$ Hz | model.py:239–240 (default) |
| grid | 128 bins, log, 20–24000 Hz | model.py:230–233 (default) |
| `use_perceptual` | True (A0) | train_full:136 |

## 5. rebuttal 문구
"The principal magnitude term and its per-bin weight $W_k$ are now defined explicitly in §2; the full A-weighting and the shape/gradient/curvature sub-terms are specified in the released training code (`model.py`, `DualObjectiveEQLoss`). The loss is unchanged from the original submission — only the per-section gain bound was relaxed (±6→±12 dB)."
