# The training loss, as equations

Everything below is transcribed from `DualObjectiveEQLoss` in `model.py`. No term
is reconstructed from prose or written by hand, and each carries the symbol that
defines it so it can be checked against the code one for one.

References are by symbol rather than line number. Line numbers drift as soon as
the code moves, and a stale one would undermine the only thing this document
claims.

The loss is unchanged by the ±6 → ±12 dB relaxation: the same
`a0_proposed_loss` is used before and after.

- Code: `model.py::DualObjectiveEQLoss`
- Hyperparameters: `a0_proposed_loss` in `train_full.py::build_registry` — the
  loss instance A0 uses, shared with A1 and A3. A2 uses a separate
  `a2_with_pref_loss`; the only difference is $\lambda_\text{pref}$ and
  $\lambda_\text{dir}$.

---

## 0. Notation and grid

- $\hat{R}\in\mathbb{R}^{F}$ is the predicted response in dB and $T$ the target in dB, over bins $k=1,\dots,F$ with $F=128$.
- The frequency grid is log-spaced, $f_k\in[20,\,24000]$ Hz (`DualObjectiveEQLoss.__init__`; `f_max=24000` by default, the Nyquist frequency). Target, prediction and loss share it.
- $\langle\cdot\rangle_{b,k}$ is the mean over batch and bins; $\bar{x}=\tfrac1F\sum_k x_k$ is the per-sample bin mean.

## 1. Top level (`DualObjectiveEQLoss.forward`)

$$
\mathcal{L}=\lambda_\text{final}\,\mathcal{L}_\text{final}(\hat{R},T_\text{dual})
+\lambda_\text{room}\,\mathcal{L}_\text{room}(\hat{R}_\text{room},T_\text{room})
+\lambda_\text{pref}\mathcal{L}_\text{pref}+\lambda_\text{dir}\mathcal{L}_\text{dir}
$$

For A0_Proposed, $(\lambda_\text{final},\lambda_\text{room},\lambda_\text{pref},\lambda_\text{dir})=(1.0,\;0.35,\;0,\;0)$.

> $\mathcal{L}_\text{room}$ applies the **same weighted-curve form** to
> $(\hat{R}_\text{room\,corr},T_\text{room})$ — the room term inside `forward`.
> It is not a plain MSE or L1.

## 2. Weighted-curve loss (`DualObjectiveEQLoss._weighted_curve_loss`) — shared by the final and room terms

$$
\mathcal{L}_\bullet=\mathcal{L}_\text{mag}+\lambda_\text{shape}\mathcal{L}_\text{shape}
+\lambda_\text{grad}\mathcal{L}_\text{grad}+\lambda_\text{curv}\mathcal{L}_\text{curv}
+\lambda_\text{mean}\mathcal{L}_\text{mean}
$$

with $(\lambda_\text{shape},\lambda_\text{grad},\lambda_\text{curv},\lambda_\text{mean})=(0.20,0.20,0.08,0.03)$ and a coefficient of $1$ on $\mathcal{L}_\text{mag}$.

### (1) Magnitude term

$$
\mathcal{L}_\text{mag}=\big\langle \widetilde{W}_k\,(\hat{R}_k-T_k)^2\big\rangle_{b,k},\qquad
\widetilde{W}_k=\frac{W_k}{\frac1F\sum_j W_j+\varepsilon},\ \ \varepsilon=10^{-8}
$$

$$
W_k=w^A_k\cdot w^E_k\cdot(1+\alpha|T_k|)\cdot\big(1+\beta\,|T_k-T_{k-1}|\big),\quad \alpha=0.10,\ \beta=0.30
$$

The gradient factor is zero at the first bin ($k{=}1$): `target_grad_mag[:, 0] = 0`. The normalisation is per sample.

**A-weighting** $w^A_k$ (IEC 61672, converted to linear amplitude and then mean-normalised; the `a_weight` buffer in `__init__`):

$$
A_\text{dB}(f)=20\log_{10}\!\frac{12194^2\,f^4}{(f^2+20.6^2)\sqrt{(f^2+107.7^2)(f^2+737.9^2)}\,(f^2+12194^2)}+2.0
$$

$$
w^A_k=\frac{10^{A_\text{dB}(f_k)/20}}{\big\langle 10^{A_\text{dB}/20}\big\rangle_k}
$$

A0 sets `use_perceptual=True`, giving a base of $w^A_k w^E_k$. The room-only baselines E3 and E4 set `use_perceptual=False`, leaving a base of $w^E_k$.

**Edge boost** $w^E_k$ (the `edge_weight` buffer in `__init__`):

$$
w^E_k=\begin{cases}1.30 & f_k\le 300\,\text{Hz}\\[2pt] 1.15 & f_k\ge 8000\,\text{Hz}\\[2pt] 1.0 & \text{otherwise}\end{cases}
$$

### (2) Shape — MSE after removing the mean

$$
\mathcal{L}_\text{shape}=\big\langle\big((\hat{R}_k-\bar{\hat{R}})-(T_k-\bar{T})\big)^2\big\rangle_{b,k}
$$

### (3) Gradient — MSE on the first difference

$$
\mathcal{L}_\text{grad}=\big\langle\big((\hat{R}_{k+1}-\hat{R}_k)-(T_{k+1}-T_k)\big)^2\big\rangle_{b,k}
$$

### (4) Curvature — MSE on the second difference

$$
\mathcal{L}_\text{curv}=\big\langle\big((\hat{R}_{k+1}-2\hat{R}_k+\hat{R}_{k-1})-(T_{k+1}-2T_k+T_{k-1})\big)^2\big\rangle_{b,k}
$$

### (5) Mean penalty — MSE on the broadband offset

$$
\mathcal{L}_\text{mean}=\big\langle(\bar{\hat{R}}-\bar{T})^2\big\rangle_{b}
$$

A2 only: $\mathcal{L}_\text{pref}=\mathcal{L}_\bullet(\hat{R}_\text{peq},\,T_\text{dual}-T_\text{room})$ and $\mathcal{L}_\text{dir}=1-\cos(\hat{R}-T_\text{room},\,T_\text{pref})$, both inside `forward`. A0 sets $\lambda=0$ on both.

---

## 3. Condensed form for the text

> Both $\mathcal{L}_\text{final}$ and $\mathcal{L}_\text{room}$ use a weighted log-spectral form. For a target $T$ ($T_\text{dual}$ and $T_\text{room}$, respectively),
> $$\mathcal{L}_\bullet=\big\langle \widetilde{W}_k(\hat{R}_k-T_k)^2\big\rangle+\lambda_\text{shape}\mathcal{L}_\text{shape}+\lambda_\text{grad}\mathcal{L}_\text{grad}+\lambda_\text{curv}\mathcal{L}_\text{curv}+\lambda_\text{mean}\mathcal{L}_\text{mean},$$
> where $W_k=w^A_k\,w^E_k(1+\alpha|T_k|)(1+\beta|\Delta T_k|)$ combines an A-weighting $w^A_k$, low/high band-edge boosts $w^E_k$, and target magnitude/gradient adaptivity, normalized to unit mean per sample ($\widetilde{W}_k$). The auxiliary terms are MSEs on the mean-removed curve ($\mathcal{L}_\text{shape}$), its first ($\mathcal{L}_\text{grad}$) and second ($\mathcal{L}_\text{curv}$) finite differences, and the broadband offset ($\mathcal{L}_\text{mean}$). The total objective is $\mathcal{L}=\lambda_\text{final}\mathcal{L}_\text{final}+\lambda_\text{room}\mathcal{L}_\text{room}$ with
> $(\lambda_\text{final},\lambda_\text{room},\lambda_\text{shape},\lambda_\text{grad},\lambda_\text{curv},\lambda_\text{mean})=(1,0.35,0.20,0.20,0.08,0.03)$.
> *The exact constants ($\alpha=0.1$, $\beta=0.3$; edge boosts $1.30/1.15$ at $300$ Hz $/\,8$ kHz) and the IEC-61672 A-weighting $w^A_k$ are given in the released training code.*

## 4. Constants, as verified against the code

| symbol | value | defined in |
|---|---|---|
| $\lambda_\text{final}$ | 1.0 | `build_registry` → `a0_proposed_loss` |
| $\lambda_\text{room}$ | 0.35 | `build_registry` → `a0_proposed_loss` |
| $\lambda_\text{pref},\lambda_\text{dir}$ | 0, 0 (A0) | `build_registry` → `a0_proposed_loss` |
| $\lambda_\text{shape},\lambda_\text{grad},\lambda_\text{curv},\lambda_\text{mean}$ | 0.20, 0.20, 0.08, 0.03 | `build_registry` → `a0_proposed_loss` |
| $\alpha$ (magnitude), $\beta$ (gradient) | 0.10, 0.30 | `build_registry` → `a0_proposed_loss` |
| edge boost, low / high | 1.30 / 1.15 | `DualObjectiveEQLoss.__init__` (default) |
| edge bands | $\le300$ / $\ge8000$ Hz | `DualObjectiveEQLoss.__init__` (default) |
| grid | 128 bins, log-spaced, 20–24000 Hz | `DualObjectiveEQLoss.__init__` (default) |
| `use_perceptual` | True (A0) | `build_registry` → `a0_proposed_loss` |

## 5. Summary

The principal magnitude term and its per-bin weight $W_k$ are given in §2; the
A-weighting and the shape, gradient and curvature sub-terms are in
`DualObjectiveEQLoss`. The ±6 → ±12 dB relaxation changed the Stage-B
per-section gain bound alone — the loss itself is unchanged.
