library(rstanarm)
library(performance)
library(parallel)

# ── Configuration ──────────────────────────────────────────────────────────────
N_SEEDS    <- 50
P          <- 25
N_SUBJ     <- 30
N_TRAIN    <- 24
DATA_FILE  <- "data/glmm_pool.rds"
OUT_FILE   <- "results/regularisation_sweep.csv"
N_CORES    <- parallel::detectCores() - 1

PRIOR_CONFIGS <- list(
  flat        = normal(location = 0, scale = 10),
  moderate    = normal(location = 0, scale = 1),
  strong      = normal(location = 0, scale = 0.5),
  very_strong = normal(location = 0, scale = 0.1)
)

# ── Load data ──────────────────────────────────────────────────────────────────
dat <- readRDS(DATA_FILE)
dir.create("results", showWarnings = FALSE, recursive = TRUE)

# ── Single seed worker ─────────────────────────────────────────────────────────
run_one_seed <- function(seed, config_name, prior_obj, dat) {
  set.seed(seed)

  all_subj   <- unique(dat$subject)
  boot_subj  <- sample(all_subj, replace = TRUE)
  train_subj <- boot_subj[1:N_TRAIN]
  test_subj  <- boot_subj[(N_TRAIN + 1):N_SUBJ]

  remap_subject <- function(subj_ids, data) {
    pieces <- lapply(seq_along(subj_ids), function(k) {
      d <- data[data$subject == subj_ids[k], , drop = FALSE]
      d$subject_new <- k
      d
    })
    do.call(rbind, pieces)
  }

  train_dat <- remap_subject(train_subj, dat)
  test_dat  <- remap_subject(test_subj, dat)

  covs <- paste0("X", 1:P)
  frm  <- as.formula(paste0("y ~ ", paste(covs, collapse = " + "), " + (1 | subject_new)"))

  fit <- tryCatch(
    stan_glmer(frm, data = train_dat, family = binomial,
               prior = prior_obj,
               chains = 2, iter = 1000, warmup = 500,
               seed = seed, refresh = 0,
               adapt_delta = 0.95),
    error = function(e) NULL
  )

  if (is.null(fit)) {
    return(data.frame(seed = seed, config = config_name,
                      train_acc = NA, test_acc = NA,
                      marginal_R2 = NA, conditional_R2 = NA, converged = FALSE))
  }

  pred_train <- ifelse(posterior_linpred(fit, newdata = train_dat,
                                         transform = TRUE, draws = 100) |>
                          colMeans() > 0.5, 1, 0)
  train_acc  <- mean(pred_train == train_dat$y)

  pred_test <- ifelse(posterior_linpred(fit, newdata = test_dat,
                                        transform = TRUE, draws = 100,
                                        allow.new.levels = TRUE) |>
                         colMeans() > 0.5, 1, 0)
  test_acc  <- mean(pred_test == test_dat$y)

  r2 <- tryCatch(r2_nakagawa(fit), error = function(e) list(R2_marginal = NA, R2_conditional = NA))

  data.frame(seed = seed, config = config_name,
             train_acc = train_acc, test_acc = test_acc,
             marginal_R2 = r2$R2_marginal, conditional_R2 = r2$R2_conditional,
             converged = TRUE)
}

# ── Run (sequential due to Stan's own parallelism) ─────────────────────────────
cat("Running regularisation sweep:", length(PRIOR_CONFIGS), "configs ×", N_SEEDS, "seeds\n")
cat("NOTE: Stan models run sequentially (Stan uses internal parallelism)\n")

results <- list()
for (cfg_name in names(PRIOR_CONFIGS)) {
  cat("  config =", cfg_name, "...")
  prior_obj <- PRIOR_CONFIGS[[cfg_name]]
  res <- lapply(1:N_SEEDS, function(s) {
    if (s %% 50 == 0) cat(" seed", s)
    run_one_seed(s, cfg_name, prior_obj, dat)
  })
  results[[cfg_name]] <- do.call(rbind, res)
  cat(" done\n")
}

out <- do.call(rbind, results)
write.csv(out, OUT_FILE, row.names = FALSE)
cat("Saved", nrow(out), "rows →", OUT_FILE, "\n")
