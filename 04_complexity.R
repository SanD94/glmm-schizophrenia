library(lme4)
library(performance)
library(parallel)

# ── Configuration ──────────────────────────────────────────────────────────────
N_SEEDS    <- 500
P          <- 25
N_SUBJ     <- 30
N_TRAIN    <- 24
DATA_FILE  <- "data/glmm_pool.rds"
OUT_FILE   <- "results/complexity_sweep.csv"
N_CORES    <- parallel::detectCores() - 1

RE_CONFIGS <- list(
  ri_only    = "(1 | subject_new)",
  rs_X1      = "(1 + X1 | subject_new)",
  rs_X1_X2   = "(1 + X1 + X2 | subject_new)",
  rs_X1_X2_X3 = "(1 + X1 + X2 + X3 | subject_new)"
)

# ── Load data ──────────────────────────────────────────────────────────────────
dat <- readRDS(DATA_FILE)
dir.create("results", showWarnings = FALSE, recursive = TRUE)

# ── Single seed worker ─────────────────────────────────────────────────────────
run_one_seed <- function(seed, config_name, re_str, dat) {
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
  frm  <- as.formula(paste0("y ~ ", paste(covs, collapse = " + "), " + ", re_str))

  fit <- tryCatch(
    glmer(frm, data = train_dat, family = binomial,
          control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))),
    error = function(e) NULL
  )

  if (is.null(fit)) {
    return(data.frame(seed = seed, config = config_name,
                      train_acc = NA, test_acc = NA,
                      marginal_R2 = NA, conditional_R2 = NA, converged = FALSE))
  }

  conv <- length(fit@optinfo$conv$lme4) == 0

  pred_train <- ifelse(predict(fit, train_dat, type = "response") > 0.5, 1, 0)
  train_acc  <- mean(pred_train == train_dat$y)

  pred_test <- ifelse(predict(fit, test_dat, type = "response",
                              allow.new.levels = TRUE) > 0.5, 1, 0)
  test_acc  <- mean(pred_test == test_dat$y)

  r2 <- tryCatch(r2_nakagawa(fit), error = function(e) list(R2_marginal = NA, R2_conditional = NA))

  data.frame(seed = seed, config = config_name,
             train_acc = train_acc, test_acc = test_acc,
             marginal_R2 = r2$R2_marginal, conditional_R2 = r2$R2_conditional,
             converged = conv)
}

# ── Run ───────────────────────────────────────────────────────────────────────
cat("Running complexity sweep:", length(RE_CONFIGS), "configs ×", N_SEEDS, "seeds\n")

cl <- makeCluster(N_CORES)
clusterEvalQ(cl, { library(lme4); library(performance) })
clusterExport(cl, c("dat", "N_TRAIN", "N_SUBJ", "P", "run_one_seed"))

results <- list()
for (cfg_name in names(RE_CONFIGS)) {
  re_str <- RE_CONFIGS[[cfg_name]]
  cat("  config =", cfg_name, "(", re_str, ") ...")
  clusterExport(cl, c("re_str", "cfg_name"), envir = environment())
  res <- parLapply(cl, 1:N_SEEDS, function(s) run_one_seed(s, cfg_name, re_str, dat))
  results[[cfg_name]] <- do.call(rbind, res)
  cat(" done\n")
}

stopCluster(cl)

out <- do.call(rbind, results)
write.csv(out, OUT_FILE, row.names = FALSE)
cat("Saved", nrow(out), "rows →", OUT_FILE, "\n")
