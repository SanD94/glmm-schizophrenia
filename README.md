# GLMM False Discovery on Pure Noise

A Monte Carlo simulation study demonstrating how Generalised Linear Mixed Models (GLMMs) can overfit to pure noise — the statistical modelling analogue of the [ML accuracy check](../acc-check/) project.

## Motivation

When all covariates are random noise ($\beta = 0$), a GLMM will still report:
- In-sample accuracy above chance
- Inflated conditional R² (Nakagawa & Schielzeth)
- Apparently "significant" fixed-effect estimates

This project quantifies the problem across 5 experimental dimensions and provides a practical diagnostic checklist.

## Quick Start

```bash
# Prerequisites: R with lme4, performance, rstanarm, tidyverse, MASS, parallel
./run_all.sh
```

The pipeline runs:
1. **Data generation** → `data/`
2. **5 experiments in parallel** → `results/*.csv`
3. **Visualisation** → `plots/*.png`

## Experiments

| # | Script | Sweep variable | Analogue in ML project |
|---|--------|---------------|----------------------|
| 01 | `01_data_generation.R` | — | `generate_data.py` |
| 02 | `02_glmm_experiment.R` | Number of noise covariates (2–50) | `pytorch_experiment.py` |
| 03 | `03_sample_size.R` | Number of subjects (10–120) | `sample_size_experiment.py` |
| 04 | `04_complexity.R` | Random-effect structure (intercept → slopes) | `complexity_experiment.py` |
| 05 | `05_regularisation.R` | Bayesian prior strength (N(0,10) → N(0,0.1)) | `regularisation_experiment.py` |
| 06 | `06_signal.R` | Number of informative features (0–25) | `signal_experiment.py` |
| 07 | `07_visualisation.R` | — | `r_visualization.r` |

Each experiment runs **500 bootstrap seeds** and records:
- Training accuracy, test accuracy (cluster-level holdout)
- Nakagawa marginal R² (fixed effects only)
- Nakagawa conditional R² (fixed + random effects)
- AIC and convergence status

## Data Generation

```
y_ij ~ Bernoulli(logit⁻¹(b_i))
b_i  ~ N(0, 0.25)
X    ~ N(0, 1)  ← 100 pure noise covariates
β    = 0         ← no signal
```

30 subjects × 20 observations = 600 rows per dataset. Train/test split at the **subject level** (24 train / 6 test).

## Key Results (Expected)

| Metric | Under pure noise | With signal |
|--------|-----------------|-------------|
| Training accuracy | > 50% (inflated) | > 50% (real) |
| Test accuracy | ≈ 50% (chance) | > 50% (generalises) |
| Marginal R² | ≈ 0 | > 0 |
| Conditional R² | > 0 (inflated) | > marginal R² |

## Project Structure

```
glmm-acc-check/
├── run_all.sh              # Pipeline: 01 → [02-06 parallel] → 07
├── 01_data_generation.R
├── 02_glmm_experiment.R
├── 03_sample_size.R
├── 04_complexity.R
├── 05_regularisation.R
├── 06_signal.R
├── 07_visualisation.R
├── interpretation.md       # Detailed statistical interpretation
├── README.md
├── data/                   # Generated data (RDS)
├── results/                # CSV outputs
├── plots/                  # PNG visualisations
└── logs/                   # Rscript logs
```

## R Dependencies

```r
install.packages(c("lme4", "performance", "rstanarm", "tidyverse", "MASS"))
```

## Interpretation

See [interpretation.md](interpretation.md) for a detailed breakdown of expected results, the connection to ML overfitting, and a practical diagnostic checklist for applied GLMM users.

## Licence

MIT
