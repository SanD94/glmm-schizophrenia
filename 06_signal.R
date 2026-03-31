library(lme4)
library(performance)
library(parallel)

# ── Configuration ──────────────────────────────────────────────────────────────
N_SEEDS         <- 500
P_TOTAL         <- 25
N_SUBJ          <- 30
N_OBS_PER_SUBJ  <- 20
N_TRAIN         <- 24
SIGMA_B         <- 0.5
SIGNAL_BETA     <- 0.5
N_SIGNAL_LEVELS <- c(0, 1, 2, 5, 10, 15, 25)
MASTER_SEED     <- 42
OUT_FILE        <- "results/signal_sweep.csv"
N_CORES         <- parallel::detectCores() - 1

dir.create("results", showWarnings = FALSE, recursive = TRUE)

# ── Generate data with k informative features ─────────────────────────────────
generate_signal_data <- function(n_signal, base_seed) {
  set.seed(base_seed)
  n_total <- N_SUBJ * N_OBS_PER_SUBJ
  subject <- rep(1:N_SUBJ, each = N_OBS_PER_SUBJ)
  b_i     <- rnorm(N_SUBJ, 0, SIGMA_B)

  X <- matrix(rnorm(n_total * P_TOTAL), nrow = n_total, ncol = P_TOTAL)
  colnames(X) <- paste0("X", 1:P_TOTAL)

  eta <- b_i[subject]
  if (n_signal > 0) {
    for (j in 1:n_signal) {
      eta <- eta + SIGNAL_BETA * X[, j]
    }
  }

  prob <- plogis(eta)
  y    <- rbinom(n_total, 1, prob)
  data.frame(subject = subject, y = y, X)
}

# ── Single seed worker ─────────────────────────────────────────────────────────
run_one_seed <- function(seed, n_signal, dat) {
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

  covs <- paste0("X", 1:P_TOTAL)
  frm  <- as.formula(paste0("y ~ ", paste(covs, collapse = " + "), " + (1 | subject_new)"))

  fit <- tryCatch(
    glmer(frm, data = train_dat, family = binomial,
          control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))),
    error = function(e) NULL
  )

  if (is.null(fit)) {
    return(data.frame(seed = seed, n_signal = n_signal,
                      train_acc = NA, test_acc = NA,
                      marginal_R2 = NA, conditional_R2 = NA, converged = FALSE))
  }

  conv <- is.null(fit@optinfo$conv$lme4)

  pred_train <- ifelse(predict(fit, train_dat, type = "response") > 0.5, 1, 0)
  train_acc  <- mean(pred_train == train_dat$y)

  pred_test <- ifelse(predict(fit, test_dat, type = "response",
                              allow.new.levels = TRUE) > 0.5, 1, 0)
  test_acc  <- mean(pred_test == test_dat$y)

  r2 <- tryCatch(r2_nakagawa(fit), error = function(e) list(R2_marginal = NA, R2_conditional = NA))

  data.frame(seed = seed, n_signal = n_signal,
             train_acc = train_acc, test_acc = test_acc,
             marginal_R2 = r2$R2_marginal, conditional_R2 = r2$R2_conditional,
             converged = conv)
}

# ── Run ───────────────────────────────────────────────────────────────────────
cat("Running signal sweep:", length(N_SIGNAL_LEVELS), "levels ×", N_SEEDS, "seeds\n")

cl <- makeCluster(N_CORES)
clusterEvalQ(cl, { library(lme4); library(performance) })

results <- list()
for (ns in N_SIGNAL_LEVELS) {
  cat("  n_signal =", ns, "...")
  pool <- generate_signal_data(ns, MASTER_SEED + ns)
  clusterExport(cl, c("pool", "P_TOTAL", "N_TRAIN", "N_SUBJ", "run_one_seed"), envir = environment())
  res <- parLapply(cl, 1:N_SEEDS, function(s) run_one_seed(s, ns, pool))
  results[[as.character(ns)]] <- do.call(rbind, res)
  cat(" done\n")
}

stopCluster(cl)

out <- do.call(rbind, results)
write.csv(out, OUT_FILE, row.names = FALSE)
cat("Saved", nrow(out), "rows →", OUT_FILE, "\n")
