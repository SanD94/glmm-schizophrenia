library(lme4)
library(performance)
library(parallel)

# ── Configuration ──────────────────────────────────────────────────────────────
N_SEEDS         <- 500
N_SUBJ          <- 30
N_TRAIN_SUBJ    <- 24
N_TEST_SUBJ     <- 6
INPUT_SIZES     <- c(2, 5, 10, 15, 20, 25, 30, 40, 50)
DATA_FILE       <- "data/glmm_pool.rds"
OUT_FILE        <- "results/input_size_sweep.csv"
N_CORES         <- parallel::detectCores() - 1

# ── Load data ──────────────────────────────────────────────────────────────────
dat <- readRDS(DATA_FILE)
dir.create("results", showWarnings = FALSE, recursive = TRUE)

# ── Single seed worker ─────────────────────────────────────────────────────────
run_one_seed <- function(seed, p, dat) {
  set.seed(seed)

  all_subj   <- unique(dat$subject)
  boot_subj  <- sample(all_subj, replace = TRUE)
  train_subj <- boot_subj[1:N_TRAIN_SUBJ]
  test_subj  <- boot_subj[(N_TRAIN_SUBJ + 1):N_SUBJ]

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

  covs <- paste0("X", 1:p)
  formula_str <- paste0("y ~ ", paste(covs, collapse = " + "), " + (1 | subject_new)")
  frm <- as.formula(formula_str)

  fit <- tryCatch(
    glmer(frm, data = train_dat, family = binomial,
          control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))),
    error = function(e) NULL
  )

  if (is.null(fit)) {
    return(data.frame(
      seed = seed, p = p,
      train_acc = NA, test_acc = NA,
      marginal_R2 = NA, conditional_R2 = NA,
      AIC = NA, converged = FALSE
    ))
  }

  conv <- length(fit@optinfo$conv$lme4) == 0

  pred_train <- ifelse(predict(fit, train_dat, type = "response") > 0.5, 1, 0)
  train_acc  <- mean(pred_train == train_dat$y)

  pred_test <- ifelse(predict(fit, test_dat, type = "response",
                              allow.new.levels = TRUE) > 0.5, 1, 0)
  test_acc  <- mean(pred_test == test_dat$y)

  r2 <- tryCatch(r2_nakagawa(fit), error = function(e) list(R2_marginal = NA, R2_conditional = NA))

  data.frame(
    seed           = seed,
    p              = p,
    train_acc      = train_acc,
    test_acc       = test_acc,
    marginal_R2    = r2$R2_marginal,
    conditional_R2 = r2$R2_conditional,
    AIC            = AIC(fit),
    converged      = conv
  )
}

# ── Run all seeds × input sizes ───────────────────────────────────────────────
cat("Running input-size sweep:", length(INPUT_SIZES), "sizes ×", N_SEEDS, "seeds\n")
cat("Using", N_CORES, "cores\n")

cl <- makeCluster(N_CORES)
clusterEvalQ(cl, { library(lme4); library(performance) })
clusterExport(cl, c("dat", "N_TRAIN_SUBJ", "N_SUBJ", "run_one_seed"))

results <- list()
for (p_val in INPUT_SIZES) {
  cat("  p =", p_val, "...")
  clusterExport(cl, "p_val", envir = environment())
  res <- parLapply(cl, 1:N_SEEDS, function(s) run_one_seed(s, p_val, dat))
  results[[as.character(p_val)]] <- do.call(rbind, res)
  cat(" done\n")
}

stopCluster(cl)

out <- do.call(rbind, results)
write.csv(out, OUT_FILE, row.names = FALSE)
cat("Saved", nrow(out), "rows →", OUT_FILE, "\n")
