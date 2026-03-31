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

**Expected results**:

| Metric | Behaviour as $p$ increases |
|--------|---------------------------|
| **Training accuracy** | Rises above 50% — the model memorises noise patterns in the training clusters |
| **Test accuracy** | Stays at ~50% — noise patterns don't generalise to held-out subjects |
| **Marginal R²** | Stays near 0 — correctly reflects that fixed effects explain nothing |
| **Conditional R²** | Inflated — the random intercept absorbs real between-subject variance, but the model attributes some noise to it too |

**Key insight**: Unlike neural networks where training accuracy can reach 65%+ with 50 noise features, GLMMs are more constrained because `glmer` uses maximum likelihood with a Laplace approximation — an implicit regulariser. The train/test gap will be smaller but still present. The gap between marginal and conditional R² is the GLMM-specific red flag.

### 2. Sample-Size Sweep (03)

**Design**: Fix $p = 25$ noise covariates, vary the number of subjects from 10 to 120.

**Expected results**:
- **Small $n$ (10–20 subjects)**: Severe overfitting. With 25 covariates and only 200–400 rows, the model has far too many parameters relative to the effective sample size (which is the number of clusters, not rows).
- **Large $n$ (80–120 subjects)**: Overfitting shrinks. More clusters → better estimation of the zero fixed effects → train/test gap closes toward zero.
- **Marginal R²** converges to 0 faster with more subjects.

**Key insight**: In mixed models, effective sample size is determined by the number of **clusters** (subjects), not the total number of observations. This is the $n_2$ vs $n_1$ distinction that many applied researchers miss.

### 3. Complexity Sweep (04)

**Design**: Fix $p = 25$, vary random-effect structure:
- `(1 | subject)` — random intercept only (4 parameters: 25 fixed + intercept + σ_b + residual)
- `(1 + X1 | subject)` — add one random slope (adds covariance terms)
- `(1 + X1 + X2 | subject)` — two random slopes
- `(1 + X1 + X2 + X3 | subject)` — three random slopes (covariance matrix grows as $k(k+1)/2$)

**Expected results**:
- Richer random-effect structures produce **higher conditional R²** but not higher marginal R².
- More random slopes → more convergence failures (singular fits, boundary estimates).
- The model is fitting random covariance structure to noise — the mixed-model analogue of adding hidden layers to a neural network.

**Key insight**: "Maximal random effects" (Barr et al., 2013) is only justified when the design actually supports it. On pure noise, maximal models don't improve prediction — they just overfit the variance components.

### 4. Regularisation Sweep (05)

**Design**: Bayesian GLMM via `rstanarm::stan_glmer` with increasingly informative priors on fixed effects:
- $N(0, 10)$ — nearly flat (≈ ML estimate)
- $N(0, 1)$ — moderate shrinkage
- $N(0, 0.5)$ — strong shrinkage
- $N(0, 0.1)$ — very strong (pushes $\beta$ toward zero)

**Expected results**:
- **Flat prior**: Behaves like `lme4`, same overfitting pattern.
- **Strong priors**: Shrink all $\hat{\beta}$ toward zero → training accuracy drops toward 50%, test accuracy stays at 50%, and the train/test gap disappears.
- Marginal R² under strong priors will be nearly exactly 0.

**Key insight**: This is the direct analogue of weight decay in neural networks. The prior $N(0, \sigma^2)$ on $\beta$ is mathematically equivalent to an L2 penalty $\lambda \|\beta\|^2$ with $\lambda = 1/(2\sigma^2)$. Stronger prior = more regularisation = less overfitting to noise.

### 5. Signal Injection (06)

**Design**: Among 25 total covariates, make $k \in \{0, 1, 2, 5, 10, 15, 25\}$ of them truly informative with $\beta = 0.5$.

**Expected results**:
- $k = 0$: Pure noise baseline — train/test gap, marginal R² ≈ 0.
- $k = 1\text{–}2$: Slight improvement in test accuracy, marginal R² begins to rise.
- $k = 10\text{–}25$: Both train and test accuracy rise substantially. The gap shrinks because the model is learning real signal. Marginal R² rises proportionally.

**Key insight**: This is the sanity check. It proves the simulation framework can detect real effects. If your GLMM can't distinguish $k = 0$ from $k = 10$, something is wrong with the setup, not with the method.

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
