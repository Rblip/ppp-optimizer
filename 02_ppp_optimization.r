# ==============================================================================
#                 PARAMETRIC PORTFOLIO POLICY (PPP) OPTIMIZATION
# ==============================================================================
# Author: Peter Paul Dimke
# Date:   January 2026
# Course: Portfolio Optimization
# Purpose: Implement the Brandt, Santa-Clara, & Valkanov (2009) PPP framework
#          at monthly rebalancing (end-of-month portfolio formation)
#
#                            All rights reserved.
# ==============================================================================
# Reference:
#   Brandt, M. W., Santa-Clara, P., & Valkanov, R. (2009).
#   "Parametric Portfolio Policies: Exploiting Characteristics in the 
#    Cross-Section of Equity Returns."
#   Review of Financial Studies, 22(9), 3411-3447.
#
# ==============================================================================
#
# THE PPP IDEA IN A NUTSHELL
# ==========================
#
# Traditional portfolio optimization (Markowitz):
#   - Estimate expected returns μ and covariance matrix Σ
#   - Solve: max w'μ - (λ/2) w'Σw
#   - Problem: Very sensitive to estimation errors!
#
# PPP approach:
#   - Don't estimate μ directly
#   - Instead, model portfolio WEIGHTS as function of characteristics
#   - Weight of stock i at time t:
#
#         w_{i,t} = w^{benchmark}_{i,t} + (1/N_t) * Σ_k θ_k * z_{i,k,t}
#
#   Where:
#     - w^{benchmark} = benchmark weight (e.g., equal-weight or value-weight)
#     - θ_k = tilt parameter for characteristic k (to be optimized)
#     - z_{i,k,t} = cross-sectional z-score of characteristic k for stock i
#     - N_t = number of stocks at time t
#
# Why z-scores?
#   - Standardizes characteristics to comparable scale
#   - Mean zero: tilts cancel out, keeping weights summing to ~1
#
# ==============================================================================

# ==============================================================================
# STEP 0: LOAD PACKAGES AND DATA
# ==============================================================================

library(data.table)
library(dplyr)
library(DEoptim)   # Differential Evolution optimization

cat("=" |> rep(70) |> paste(collapse = ""), "\n")
cat("  PARAMETRIC PORTFOLIO POLICY OPTIMIZATION\n")
cat("=" |> rep(70) |> paste(collapse = ""), "\n\n")

# Load the panel data created by 01_data_loading.r
df <- fread("yf_panel_data.csv")
df$date <- as.Date(df$date)

cat("Data loaded (MONTHLY frequency):\n")
cat("  Observations:", format(nrow(df), big.mark = ","), "\n")
cat("  Stocks:      ", n_distinct(df$ticker), "\n")
cat("  Date range:  ", as.character(min(df$date)), "to", 
    as.character(max(df$date)), "\n")
cat("  Months:      ", n_distinct(df$year_month), "\n\n")

# ==============================================================================
# STEP 1: DEFINE THE HELPER FUNCTION FOR Z-SCORES
# ==============================================================================
#
# Z-score formula: z = (x - mean(x)) / sd(x)
#
# Properties:
#   - Mean of z-scores = 0 (by construction)
#   - Standard deviation of z-scores = 1 (by construction)
#   - Allows comparing characteristics on the same scale
#
# We calculate z-scores CROSS-SECTIONALLY (across stocks on each date)
# ==============================================================================

zscore <- function(x) {
  m <- mean(x, na.rm = TRUE)
  s <- sd(x, na.rm = TRUE)
  if (is.finite(s) && s > 0) {
    (x - m) / s
  } else {
    rep(0, length(x))  # If no variation, set z = 0
  }
}

# ==============================================================================
# STEP 2: COMPUTE CROSS-SECTIONAL Z-SCORES
# ==============================================================================
#
# For each date, we standardize each characteristic across all stocks.
# This is the CROSS-SECTIONAL dimension of panel data.
#
# Example: If AAPL has higher momentum than average on 2020-01-02,
#          its mom_z will be positive on that date.
#
# ==============================================================================

cat("Computing cross-sectional z-scores...\n")

df <- df %>%
  group_by(date) %>%  # Group by date = cross-sectional standardization
  mutate(
    mom_12m_z  = zscore(mom_12m),          # Momentum
    vol_z      = zscore(vol_1m),           # Volatility
    illiq_z    = zscore(illiq_1m),         # Illiquidity (Amihud)
    prc_high_z = zscore(price_to_high_12m) # Price relative to 12m high
  ) %>%
  ungroup()

cat("  ✓ Z-scores computed for: mom_12m, vol_1m, illiq_1m, price_to_high_12m\n\n")

# ==============================================================================
# STEP 3: COMPUTE BENCHMARK WEIGHTS
# ==============================================================================
#
# The benchmark is your "neutral" portfolio before applying tilts.
#
# Common choices:
#   - Equal-weight: w_i = 1/N (simple, easy to interpret)
#   - Value-weight: w_i = market_cap_i / Σ market_cap
#
# For this example, we use EQUAL WEIGHTS for simplicity.
#
# ==============================================================================

cat("Computing benchmark weights...\n")

df <- df %>%
  group_by(date) %>%
  mutate(
    N = n(),                # Number of stocks on this date
    w_benchmark = 1 / N     # Equal weight
  ) %>%
  ungroup()

cat("  ✓ Using equal-weight benchmark (w = 1/N)\n\n")

# ==============================================================================
# STEP 4: DEFINE THE THETA PARAMETERS (MANUAL EXAMPLE)
# ==============================================================================
#
# θ (theta) parameters control HOW MUCH we tilt toward each characteristic.
#
# Interpretation:
#   θ > 0: Overweight stocks with HIGH values of the characteristic
#   θ < 0: Overweight stocks with LOW values of the characteristic
#   θ = 0: No tilt (stay at benchmark)
#
# Example tilts (before optimization):
#   θ_mom_12m  = +0.5  → Overweight past winners (momentum strategy)
#   θ_vol      = -0.3  → Overweight low volatility stocks
#   θ_illiq    = +0.2  → Overweight illiquid stocks (illiquidity premium)
#   θ_prc_high = +0.3  → Overweight stocks near 52-week high
#
# ==============================================================================

cat("=" |> rep(70) |> paste(collapse = ""), "\n")
cat("  PART A: MANUAL PPP (BEFORE OPTIMIZATION)\n")
cat("=" |> rep(70) |> paste(collapse = ""), "\n\n")

# Define example theta values (we will optimize these later)
theta <- c(
  mom_12m_z  = 0.5,   # Positive tilt on 12-month momentum
  vol_z      = -0.3,  # Negative tilt on volatility (low-vol strategy)
  illiq_z    = 0.2,   # Positive tilt on illiquidity (illiquidity premium)
  prc_high_z = 0.3    # Positive tilt on price near high
)

cat("Example theta values:\n")
print(theta)
cat("\n")

# ==============================================================================
# STEP 5: COMPUTE PPP PORTFOLIO WEIGHTS
# ==============================================================================
#
# The PPP weight formula:
#
#   w_{i,t} = w^{benchmark}_{i,t} + (1/N_t) * Σ_k θ_k * z_{i,k,t}
#
# Breaking it down:
#   1. Start with benchmark weight
#   2. Add a tilt based on (θ × z-score) for each characteristic
#   3. Divide by N to keep weights roughly summing to 1
#
# ==============================================================================

df <- df %>%
  group_by(date) %>%
  mutate(
    # Compute the sum of (theta × z-score) for each stock
    theta_x_sum = (mom_12m_z  * theta["mom_12m_z"]) +
                  (vol_z      * theta["vol_z"]) +
                  (illiq_z    * theta["illiq_z"]) +
                  (prc_high_z * theta["prc_high_z"]),
    
    # Divide by N (number of stocks)
    theta_x = theta_x_sum / N,
    
    # Final portfolio weight = benchmark + tilt
    w = w_benchmark + theta_x
  ) %>%
  ungroup()

cat("Portfolio weights computed.\n")
cat("  Sum of weights check (should be ≈ 1):\n")
weight_sums <- df %>% group_by(date) %>% summarise(sum_w = sum(w), .groups = "drop")
cat("    Mean:  ", round(mean(weight_sums$sum_w), 4), "\n")
cat("    Min:   ", round(min(weight_sums$sum_w), 4), "\n")
cat("    Max:   ", round(max(weight_sums$sum_w), 4), "\n\n")

# ==============================================================================
# STEP 6: COMPUTE PORTFOLIO RETURNS
# ==============================================================================
#
# Portfolio return at time t:
#
#   R_{portfolio,t} = Σ_i w_{i,t} × r_{i,t+1}
#
# Where r_{i,t+1} is the FORWARD return (next period's return).
#
# This is the return we would have earned by holding the portfolio
# constructed at time t until time t+1.
#
# ==============================================================================

df <- df %>%
  mutate(
    weighted_return = w * fwd_ret  # Weight × forward return
  )

# Aggregate to portfolio level (one return per date)
portfolio_returns <- df %>%
  group_by(date) %>%
  summarise(
    portfolio_return = sum(weighted_return, na.rm = TRUE),
    benchmark_return = sum(w_benchmark * fwd_ret, na.rm = TRUE),
    .groups = "drop"
  )

# Performance summary
mean_ret   <- mean(portfolio_returns$portfolio_return, na.rm = TRUE)
sd_ret     <- sd(portfolio_returns$portfolio_return, na.rm = TRUE)
sharpe     <- (mean_ret / sd_ret) * sqrt(12)  # Annualized Sharpe (monthly data)

cat("Portfolio Performance (Manual Theta):\n")
cat("-" |> rep(50) |> paste(collapse = ""), "\n")
cat("  Mean monthly return:     ", round(mean_ret * 100, 4), "%\n")
cat("  Std dev monthly return:  ", round(sd_ret * 100, 4), "%\n")
cat("  Annualized Sharpe Ratio: ", round(sharpe, 3), "\n\n")

# ==============================================================================
# ==============================================================================
#  PART B: OPTIMIZATION
# ==============================================================================
# ==============================================================================
#
# Now we find the OPTIMAL theta values that maximize the Sharpe Ratio.
#
# We use Differential Evolution (DEoptim), a global optimization algorithm
# that works well for non-convex problems.
#
# ==============================================================================

cat("=" |> rep(70) |> paste(collapse = ""), "\n")
cat("  PART B: PPP OPTIMIZATION\n")
cat("=" |> rep(70) |> paste(collapse = ""), "\n\n")

# ==============================================================================
# STEP 7: DEFINE THE OBJECTIVE FUNCTION
# ==============================================================================
#
# We want to MAXIMIZE the Sharpe Ratio.
# DEoptim MINIMIZES, so we return the NEGATIVE Sharpe Ratio.
#
# The function:
#   1. Takes a vector of theta values
#   2. Computes portfolio weights using PPP formula
#   3. Computes portfolio returns
#   4. Returns negative Sharpe Ratio
#
# ==============================================================================

calculate_neg_sharpe <- function(theta_vec, data) {
  # Name the theta vector
  names(theta_vec) <- c("mom_12m_z", "vol_z", "illiq_z", "prc_high_z")
  
  # Compute portfolio weights and returns
  portfolio_data <- data %>%
    group_by(date) %>%
    mutate(
      theta_x_sum = (mom_12m_z  * theta_vec["mom_12m_z"]) +
                    (vol_z      * theta_vec["vol_z"]) +
                    (illiq_z    * theta_vec["illiq_z"]) +
                    (prc_high_z * theta_vec["prc_high_z"]),
      theta_x = theta_x_sum / N,
      w = w_benchmark + theta_x
    ) %>%
    summarise(
      portfolio_return = sum(w * fwd_ret, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Calculate Sharpe Ratio
  mean_ret <- mean(portfolio_data$portfolio_return, na.rm = TRUE)
  sd_ret   <- sd(portfolio_data$portfolio_return, na.rm = TRUE)
  
  # Handle edge cases
  if (is.na(sd_ret) || sd_ret == 0) {
    return(1e9)  # Return large penalty
  }
  
  # Annualized Sharpe Ratio (monthly data: 12 months per year)
  sharpe_ratio <- (mean_ret / sd_ret) * sqrt(12)
  
  # Return NEGATIVE because DEoptim minimizes
  return(-sharpe_ratio)
}

# ==============================================================================
# STEP 8: SET OPTIMIZATION BOUNDS
# ==============================================================================
#
# We constrain theta to be between -2 and +2.
# This prevents extreme tilts that might be unrealistic.
#
# ==============================================================================

n_params <- 4  # Number of theta parameters
lower_bounds <- rep(-2, n_params)
upper_bounds <- rep(2, n_params)

cat("Optimization settings:\n")
cat("  Parameters:", n_params, "\n")
cat("  Bounds: [", lower_bounds[1], ",", upper_bounds[1], "] for each theta\n\n")

# ==============================================================================
# STEP 9: RUN THE OPTIMIZATION
# ==============================================================================

cat("Running Differential Evolution optimization...\n")
cat("-" |> rep(50) |> paste(collapse = ""), "\n")

set.seed(123)  # For reproducibility

optim_result <- DEoptim(
  fn = calculate_neg_sharpe,
  lower = lower_bounds,
  upper = upper_bounds,
  control = DEoptim.control(
    trace = 10,        # Print progress every 10 iterations
    itermax = 50,      # Maximum iterations (increase for better results)
    steptol = 20,      # Stop if no improvement for 20 iterations
    NP = n_params * 10 # Population size
  ),
  data = df
)

cat("-" |> rep(50) |> paste(collapse = ""), "\n\n")

# ==============================================================================
# STEP 10: EXTRACT AND INTERPRET RESULTS
# ==============================================================================

optimal_theta <- optim_result$optim$bestmem
names(optimal_theta) <- c("mom_12m_z", "vol_z", "illiq_z", "prc_high_z")

cat("=" |> rep(70) |> paste(collapse = ""), "\n")
cat("  OPTIMIZATION RESULTS\n")
cat("=" |> rep(70) |> paste(collapse = ""), "\n\n")

cat("Optimal Theta Values:\n")
cat("-" |> rep(50) |> paste(collapse = ""), "\n")
for (i in seq_along(optimal_theta)) {
  name <- names(optimal_theta)[i]
  value <- optimal_theta[i]
  interpretation <- if (value > 0.1) {
    "→ Overweight high values"
  } else if (value < -0.1) {
    "→ Overweight low values"
  } else {
    "→ Minimal tilt"
  }
  cat(sprintf("  %-12s: %+7.4f  %s\n", name, value, interpretation))
}
cat("\n")

# Optimal Sharpe Ratio
optimal_sharpe <- -optim_result$optim$bestval
cat("Optimal Annualized Sharpe Ratio:", round(optimal_sharpe, 3), "\n\n")

# ==============================================================================
# STEP 11: BACKTEST WITH OPTIMAL WEIGHTS
# ==============================================================================

cat("Backtesting with optimal theta...\n")

backtest <- df %>%
  group_by(date) %>%
  mutate(
    theta_x_sum_opt = (mom_12m_z  * optimal_theta["mom_12m_z"]) +
                      (vol_z      * optimal_theta["vol_z"]) +
                      (illiq_z    * optimal_theta["illiq_z"]) +
                      (prc_high_z * optimal_theta["prc_high_z"]),
    theta_x_opt = theta_x_sum_opt / N,
    w_optimal = w_benchmark + theta_x_opt
  ) %>%
  summarise(
    ppp_return = sum(w_optimal * fwd_ret, na.rm = TRUE),
    benchmark_return = sum(w_benchmark * fwd_ret, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(date) %>%
  filter(!is.na(ppp_return) & !is.na(benchmark_return)) %>%
  mutate(
    cum_ppp = cumprod(1 + ppp_return),
    cum_benchmark = cumprod(1 + benchmark_return)
  )

# ==============================================================================
# STEP 12: PERFORMANCE COMPARISON
# ==============================================================================

# Calculate statistics
n_years <- as.numeric(difftime(max(backtest$date), min(backtest$date), units = "days")) / 365.25

ppp_total_ret   <- (tail(backtest$cum_ppp, 1) - 1) * 100
bench_total_ret <- (tail(backtest$cum_benchmark, 1) - 1) * 100

ppp_ann_ret   <- ((tail(backtest$cum_ppp, 1))^(1/n_years) - 1) * 100
bench_ann_ret <- ((tail(backtest$cum_benchmark, 1))^(1/n_years) - 1) * 100

ppp_vol   <- sd(backtest$ppp_return, na.rm = TRUE) * sqrt(12) * 100
bench_vol <- sd(backtest$benchmark_return, na.rm = TRUE) * sqrt(12) * 100

ppp_sharpe   <- (mean(backtest$ppp_return) / sd(backtest$ppp_return)) * sqrt(12)
bench_sharpe <- (mean(backtest$benchmark_return) / sd(backtest$benchmark_return)) * sqrt(12)

cat("\n")
cat("=" |> rep(70) |> paste(collapse = ""), "\n")
cat("  PERFORMANCE COMPARISON\n")
cat("=" |> rep(70) |> paste(collapse = ""), "\n\n")

cat(sprintf("%-30s %15s %15s\n", "Metric", "PPP Strategy", "Benchmark"))
cat("-" |> rep(60) |> paste(collapse = ""), "\n")
cat(sprintf("%-30s %14.2f%% %14.2f%%\n", "Total Return", ppp_total_ret, bench_total_ret))
cat(sprintf("%-30s %14.2f%% %14.2f%%\n", "Annualized Return", ppp_ann_ret, bench_ann_ret))
cat(sprintf("%-30s %14.2f%% %14.2f%%\n", "Annualized Volatility", ppp_vol, bench_vol))
cat(sprintf("%-30s %14.3f  %14.3f\n", "Sharpe Ratio", ppp_sharpe, bench_sharpe))
cat("\n")

# ==============================================================================
# STEP 13: SAVE RESULTS
# ==============================================================================

cat("Saving results...\n")

# Save backtest results
fwrite(backtest, "ppp_backtest_results.csv")

# Save optimal theta
theta_df <- data.frame(
  characteristic = names(optimal_theta),
  theta = as.numeric(optimal_theta)
)
fwrite(theta_df, "optimal_theta.csv")

cat("  ✓ Backtest results saved to: ppp_backtest_results.csv\n")
cat("  ✓ Optimal theta saved to: optimal_theta.csv\n\n")

cat("=" |> rep(70) |> paste(collapse = ""), "\n")
cat("  OPTIMIZATION COMPLETE\n")
cat("=" |> rep(70) |> paste(collapse = ""), "\n")


