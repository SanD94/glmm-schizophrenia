library(MASS)

# ── Configuration ──────────────────────────────────────────────────────────────
N_FEATURES      <- 100
N_SUBJ          <- 30
N_OBS_PER_SUBJ  <- 20
SIGMA_B         <- 0.5
MASTER_SEED     <- 42
OUT_DIR         <- "data"

# ── Generate pure-noise GLMM data ─────────────────────────────────────────────
set.seed(MASTER_SEED)
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

n_total <- N_SUBJ * N_OBS_PER_SUBJ

subject <- rep(1:N_SUBJ, each = N_OBS_PER_SUBJ)
b_i     <- rnorm(N_SUBJ, mean = 0, sd = SIGMA_B)

X <- matrix(rnorm(n_total * N_FEATURES), nrow = n_total, ncol = N_FEATURES)
colnames(X) <- paste0("X", seq_len(N_FEATURES))

eta <- b_i[subject]
prob <- plogis(eta)
y <- rbinom(n_total, size = 1, prob = prob)

dat <- data.frame(
  subject = subject,
  y       = y,
  X
)

saveRDS(dat, file.path(OUT_DIR, "glmm_pool.rds"))

cat("Data saved to", OUT_DIR, "\n")
cat("  Subjects:", N_SUBJ, "\n")
cat("  Obs per subject:", N_OBS_PER_SUBJ, "\n")
cat("  Total rows:", n_total, "\n")
cat("  Features:", N_FEATURES, "\n")
cat("  True betas: all zero\n")
cat("  sigma_b:", SIGMA_B, "\n")
cat("  Outcome mean:", round(mean(y), 3), "\n")
