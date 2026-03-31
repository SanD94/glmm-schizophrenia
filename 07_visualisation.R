library(tidyverse)

# ── Theme & helpers ────────────────────────────────────────────────────────────
base_theme <- theme_minimal(base_size = 13) +
  theme(
    axis.title       = element_text(face = "bold"),
    axis.text.x      = element_text(angle = 45, hjust = 1),
    panel.grid.minor = element_blank(),
    legend.position  = "top",
    legend.title     = element_blank()
  )

acc_colours <- c(
  "Train"  = "#E07B54",
  "Test"   = "#6DBF8A"
)

r2_colours <- c(
  "Marginal R²"    = "#5B8DB8",
  "Conditional R²" = "#E07B54"
)

pivot_acc <- function(df) {
  df |>
    filter(converged == TRUE | is.na(converged)) |>
    pivot_longer(
      cols      = c(train_acc, test_acc),
      names_to  = "split",
      values_to = "accuracy"
    ) |>
    mutate(split = factor(split,
                          levels = c("train_acc", "test_acc"),
                          labels = c("Train", "Test")))
}

pivot_r2 <- function(df) {
  df |>
    filter(converged == TRUE | is.na(converged)) |>
    pivot_longer(
      cols      = c(marginal_R2, conditional_R2),
      names_to  = "r2_type",
      values_to = "R2"
    ) |>
    mutate(r2_type = factor(r2_type,
                            levels = c("marginal_R2", "conditional_R2"),
                            labels = c("Marginal R²", "Conditional R²")))
}

make_acc_plot <- function(data, title, subtitle, x_var, y_limits = c(0.3, 1.0)) {
  ggplot(data, aes(x = .data[[x_var]], y = accuracy, fill = split)) +
    geom_violin(alpha = 0.6, trim = TRUE, scale = "width", width = 0.7,
                position = position_dodge(width = 0.75)) +
    geom_hline(yintercept = 0.5, linetype = "dashed",
               colour = "grey40", linewidth = 0.7) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                       limits = y_limits) +
    scale_fill_manual(values = acc_colours) +
    labs(title = title, subtitle = subtitle, x = x_var, y = "Accuracy") +
    base_theme
}

make_r2_plot <- function(data, title, subtitle, x_var, y_limits = c(0, 0.5)) {
  ggplot(data, aes(x = .data[[x_var]], y = R2, fill = r2_type)) +
    geom_violin(alpha = 0.6, trim = TRUE, scale = "width", width = 0.7,
                position = position_dodge(width = 0.75)) +
    geom_hline(yintercept = 0, linetype = "dashed",
               colour = "grey40", linewidth = 0.7) +
    scale_y_continuous(limits = y_limits) +
    scale_fill_manual(values = r2_colours) +
    labs(title = title, subtitle = subtitle, x = x_var, y = "R²") +
    base_theme
}

# ── 1. Input-size sweep ───────────────────────────────────────────────────────
df_inp <- read_csv("results/input_size_sweep.csv") |>
  mutate(p = factor(p, levels = c(2, 5, 10, 15, 20, 25, 30, 40, 50)))

p1_acc <- make_acc_plot(
  pivot_acc(df_inp),
  "GLMM false discovery: accuracy vs number of noise covariates",
  "Dashed line = 50% chance | 500 bootstrap seeds per p", "p"
)

p1_r2 <- make_r2_plot(
  pivot_r2(df_inp),
  "GLMM R² inflation: marginal vs conditional R²",
  "Marginal R² should stay ~0 under null | 500 seeds per p", "p"
)

# ── 2. Sample-size sweep ──────────────────────────────────────────────────────
df_ss <- read_csv("results/sample_size_sweep.csv") |>
  mutate(n_subj = factor(n_subj, levels = c(10, 20, 30, 50, 80, 120)))

p2_acc <- make_acc_plot(
  pivot_acc(df_ss),
  "Effect of number of subjects on GLMM overfitting (25 noise covariates)",
  "Dashed line = 50% chance | 500 seeds per n_subj", "n_subj"
)

p2_r2 <- make_r2_plot(
  pivot_r2(df_ss),
  "R² vs number of subjects (25 noise covariates)",
  "500 seeds per n_subj", "n_subj"
)

# ── 3. Complexity sweep ───────────────────────────────────────────────────────
df_cx <- read_csv("results/complexity_sweep.csv") |>
  mutate(config = factor(config,
    levels = c("ri_only", "rs_X1", "rs_X1_X2", "rs_X1_X2_X3")))

p3_acc <- make_acc_plot(
  pivot_acc(df_cx),
  "Effect of random-effect complexity on GLMM overfitting",
  "ri = random intercept, rs = random slopes | 500 seeds per config", "config"
)

p3_r2 <- make_r2_plot(
  pivot_r2(df_cx),
  "R² vs random-effect complexity",
  "500 seeds per config", "config"
)

# ── 4. Regularisation sweep ───────────────────────────────────────────────────
df_rg <- read_csv("results/regularisation_sweep.csv") |>
  mutate(config = factor(config,
    levels = c("flat", "moderate", "strong", "very_strong")))

p4_acc <- make_acc_plot(
  pivot_acc(df_rg),
  "Bayesian GLMM: effect of prior strength on overfitting",
  "Priors: N(0,10) → N(0,0.1) | 500 seeds per config", "config"
)

p4_r2 <- make_r2_plot(
  pivot_r2(df_rg),
  "Bayesian GLMM: R² vs prior strength",
  "500 seeds per config", "config"
)

# ── 5. Signal sweep ──────────────────────────────────────────────────────────
df_sg <- read_csv("results/signal_sweep.csv") |>
  mutate(n_signal = factor(n_signal, levels = c(0, 1, 2, 5, 10, 15, 25)))

p5_acc <- make_acc_plot(
  pivot_acc(df_sg),
  "From noise to signal: GLMM accuracy with informative features",
  "25 total features, β = 0.5 per signal feature | 500 seeds",
  "n_signal", y_limits = c(0.3, 1.0)
)

p5_r2 <- make_r2_plot(
  pivot_r2(df_sg),
  "From noise to signal: R² with informative features",
  "25 total features, β = 0.5 | 500 seeds",
  "n_signal", y_limits = c(0, 1.0)
)

# ── Save all plots ─────────────────────────────────────────────────────────────
plots_dir <- "plots"
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)

save_plot <- function(p, name, w = 16, h = 6) {
  ggsave(file.path(plots_dir, paste0(name, ".png")), p,
         width = w, height = h, dpi = 300, bg = "white")
}

save_plot(p1_acc, "01_input_size_accuracy")
save_plot(p1_r2,  "01_input_size_r2")
save_plot(p2_acc, "02_sample_size_accuracy")
save_plot(p2_r2,  "02_sample_size_r2")
save_plot(p3_acc, "03_complexity_accuracy")
save_plot(p3_r2,  "03_complexity_r2")
save_plot(p4_acc, "04_regularisation_accuracy")
save_plot(p4_r2,  "04_regularisation_r2")
save_plot(p5_acc, "05_signal_accuracy")
save_plot(p5_r2,  "05_signal_r2")

message("Saved all 10 plots to ", plots_dir, "/")
