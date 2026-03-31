# Interpretation Guide: GLMM False Discovery on Pure Noise

## What This Simulation Tests

This project asks a simple but devastating question: **when there is zero signal in the data, do Generalised Linear Mixed Models (GLMMs) still report seemingly meaningful results?** The answer, as with neural networks, is yes — but the mechanisms and diagnostics differ.

## The Null Truth

Every dataset in the noise-only experiments is generated from:

$$
\text{logit}(P(y_{ij} = 1)) = b_i, \quad b_i \sim N(0, 0.25)
$$

All $\beta$ coefficients for the fixed-effect covariates are **exactly zero**. Any apparent relationship between $X$ and $y$ is pure noise. The only real structure is the random intercept $b_i$, which creates within-subject correlation but carries no covariate-driven signal.

---

## Experiment-by-Experiment Interpretation

### 1. Input-Size Sweep (02)

**Design**: Fit `glmer(y ~ X1 + ... + Xp + (1|subject))` for $p \in \{2, 5, 10, ..., 50\}$ noise covariates.

**Actual results**:

| Metric | Behaviour as $p$ increases |
|--------|---------------------------|
| **Training accuracy** | Rises from ~64% (p=2) to ~72% (p=50) — the model memorises noise patterns in the training clusters |
| **Test accuracy** | Drifts slightly above 50% to ~58% at p=50, but remains far below training accuracy — noise patterns mostly don't generalise |
| **Marginal R²** | Rises from ~0.006 to ~0.25 — the model attributes spurious variance to the noise covariates, inflating fixed-effect R² |
| **Conditional R²** | Rises from ~0.09 to ~0.32 — the gap between marginal and conditional R² persists throughout |

**Key insight**: GLMMs overfit to noise at least as aggressively as neural networks — training accuracy reaches 72% with 50 pure noise features. The Laplace approximation in `glmer` does not provide meaningful implicit regularisation when $p$ is large relative to $n_{\text{clusters}}$. Marginal R² is *not* immune to inflation: with many noise covariates, the model attributes substantial spurious variance to fixed effects. The train/test gap and inflated R² values are the key red flags.

### 2. Sample-Size Sweep (03)

**Design**: Fix $p = 25$ noise covariates, vary the number of subjects from 10 to 120.

**Actual results**:
- **Small $n$ (10–20 subjects)**: Severe overfitting. Train accuracy reaches ~77% (n=10) and ~72% (n=20), while test accuracy stays at ~53–57%. Marginal R² is highly inflated (~0.25 at n=10).
- **Large $n$ (80–120 subjects)**: Overfitting shrinks. Train accuracy drops to ~61–64%, test accuracy remains at ~52%. The train/test gap narrows from ~24pp to ~12pp.
- **Marginal R²** drops from ~0.25 (n=10) to ~0.03 (n=120), converging toward zero with more subjects.

**Key insight**: In mixed models, effective sample size is determined by the number of **clusters** (subjects), not the total number of observations. This is the $n_2$ vs $n_1$ distinction that many applied researchers miss.

### 3. Complexity Sweep (04)

**Design**: Fix $p = 25$, vary random-effect structure:
- `(1 | subject)` — random intercept only (4 parameters: 25 fixed + intercept + σ_b + residual)
- `(1 + X1 | subject)` — add one random slope (adds covariance terms)
- `(1 + X1 + X2 | subject)` — two random slopes
- `(1 + X1 + X2 + X3 | subject)` — three random slopes (covariance matrix grows as $k(k+1)/2$)

**Actual results**:
- Richer random-effect structures produce **higher conditional R²** (0.19 → 0.28) and slightly higher marginal R² (0.11 → 0.13), while test accuracy stays flat (~54%) or even decreases marginally. Train accuracy rises from ~67% to ~71%.
- Convergence degrades severely with complexity: random-intercept only converges 97% of the time, but `rs_X1` drops to 25%, and `rs_X1_X2` and `rs_X1_X2_X3` essentially never converge (0–0.6%).
- The model is fitting random covariance structure to noise — the mixed-model analogue of adding hidden layers to a neural network.

**Key insight**: "Maximal random effects" (Barr et al., 2013) is only justified when the design actually supports it. On pure noise, maximal models don't improve prediction — they just overfit the variance components. The near-zero convergence rates for complex random-effect structures are themselves a diagnostic signal that the data cannot support the model.

### 4. Regularisation Sweep (05)

**Design**: Bayesian GLMM via `rstanarm::stan_glmer` with increasingly informative priors on fixed effects:
- $N(0, 10)$ — nearly flat (≈ ML estimate)
- $N(0, 1)$ — moderate shrinkage
- $N(0, 0.5)$ — strong shrinkage
- $N(0, 0.1)$ — very strong (pushes $\beta$ toward zero)

**Actual results** (50 seeds per config):
- **Flat prior $N(0, 10)$**: Failed completely — Stan could not initialise on any of the 50 seeds. With 25 noise covariates and a near-flat prior, the posterior is too diffuse for the sampler to find a starting point.
- **Moderate $N(0, 1)$, strong $N(0, 0.5)$, very strong $N(0, 0.1)$**: All three produced similar train accuracy (~67.5%) and test accuracy (~53%). The train/test gap persisted across all prior strengths.
- Marginal R² showed the clearest differentiation: moderate (0.11) → strong (0.11) → very strong (0.03), with the strongest prior successfully shrinking fixed-effect variance toward zero.

**Key insight**: The flat prior failure is itself informative — it shows that with many noise covariates, even a Bayesian approach cannot rescue an underidentified model without meaningful regularisation. Among the working priors, shrinkage primarily affects R² rather than accuracy, suggesting that accuracy is a blunt diagnostic. The prior $N(0, \sigma^2)$ on $\beta$ is mathematically equivalent to an L2 penalty $\lambda \|\beta\|^2$ with $\lambda = 1/(2\sigma^2)$.

### 5. Signal Injection (06)

**Design**: Among 25 total covariates, make $k \in \{0, 1, 2, 5, 10, 15, 25\}$ of them truly informative with $\beta = 0.5$.

**Actual results**:
- $k = 0$: Pure noise baseline — train ~70%, test ~55%, marginal R² ~0.12. The train/test gap confirms overfitting.
- $k = 1\text{–}2$: Test accuracy rises to ~58–60%, marginal R² to ~0.19–0.21. The model begins to detect real signal.
- $k = 5\text{–}10$: Both train and test accuracy rise substantially (test reaches ~68–75%). The train/test gap narrows because the model is learning real signal. Marginal R² rises to 0.39–0.56.
- $k = 25$: Train ~87%, test ~82%, marginal R² ~0.78. Nearly all variance is explained by real signal, and the gap is minimal.

**Key insight**: This is the sanity check. It proves the simulation framework can detect real effects and that the train/test gap shrinks when signal is present. The smooth monotonic increase in test accuracy and marginal R² with $k$ confirms the framework is well-calibrated.

---

## How to Use These Results in Practice

### The Diagnostic Checklist

When fitting a GLMM to real data, apply these checks:

1. **Compare marginal vs conditional R²**. If marginal R² is near zero but conditional R² is high, your fixed effects aren't explaining anything — the random effects are doing all the work. Ask: is that scientifically meaningful, or is the model just capturing cluster-level noise?

2. **Cross-validate at the cluster level**. Never evaluate GLMM performance on observations from subjects seen during training. Always hold out entire subjects/clusters. Within-subject prediction is contaminated by the random intercept.

3. **Check the ratio $p / n_{\text{clusters}}$**. If you have 25 covariates and 30 subjects, you're in the danger zone — the same regime where our simulation shows inflated training accuracy.

4. **Simplify random effects**. If a random-slopes model doesn't improve out-of-sample prediction over a random-intercept model, the extra complexity is fitting noise.

5. **Use informative priors or penalisation** when $p$ is large relative to $n_{\text{clusters}}$. Bayesian GLMMs with $N(0, 1)$ priors on standardised predictors are a reasonable default.

---

## Connection to the ML Experiments

| ML concept | GLMM analogue |
|---|---|
| Training accuracy > 50% on noise | In-sample accuracy > 50% on noise |
| Test accuracy ≈ 50% | Out-of-sample (new clusters) accuracy ≈ 50% |
| Hidden units / depth | Random-effect structure complexity |
| Weight decay / dropout | Bayesian priors / penalised likelihood |
| Number of features | Number of fixed-effect covariates |
| Number of training samples | Number of clusters (not observations) |
| Early stopping on validation | AIC/BIC model selection |

The fundamental lesson is identical: **any sufficiently flexible model will find patterns in noise. The only protection is out-of-sample evaluation — and in mixed models, "out-of-sample" means out-of-cluster.**
