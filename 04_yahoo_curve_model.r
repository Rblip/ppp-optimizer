# ==============================================================================
#            YAHOO FINANCE — EARNINGS YIELD & ROI CURVE PARAMETERIZATION
# ==============================================================================
# Author: Peter Paul Dimke
# Date:   January 2026
# Course: Portfolio Optimization
# Purpose: Fetch EP (earnings yield) and ROI from Yahoo Finance and use them
#          to parameterize the empirical curve model from 03_curve_model.r.
#
#                            All rights reserved.
# ==============================================================================
#
# FINANCIAL PARAMETERIZATION
# ==========================
#
# The curve P(x) = E*(1-x) / (r - x*(r+a))  is given financial meaning by:
#
#   E  →  EP  (earnings yield = EPS / Price, i.e. E/P ratio)
#
#   r  →  r(b) = (1-b)*EP + b*ROI
#
# Interpretation of r(b):
#   b = 0  →  r = EP   pure earnings-yield view of expected return
#   b = 1  →  r = ROI  pure return-on-investment view
#   0 < b < 1  →  convex blend of both views
#
# The residual l (from 03_curve_model.r) then captures the gap between the
# theoretical curve and any empirically observed point (b, p).
#
# ==============================================================================

cat("=" |> rep(70) |> paste(collapse = ""), "\n")
cat("  YAHOO FINANCE CURVE PARAMETERIZATION\n")
cat("=" |> rep(70) |> paste(collapse = ""), "\n\n")

# ==============================================================================
# SECTION 1: PACKAGES
# ==============================================================================

suppressPackageStartupMessages({
  library(httr)
  library(jsonlite)
  library(data.table)
  library(dplyr)
})

# Null-coalescing: return lhs if non-NULL, else rhs (built-in from R 4.4; defined here for compatibility)
`%||%` <- function(lhs, rhs) if (!is.null(lhs)) lhs else rhs

# Ensure curve model functions are available
if (!exists("curve_P")) source("03_curve_model.r")

# ==============================================================================
# SECTION 2: YAHOO FINANCE FETCH
# ==============================================================================

# Fetch EP (earnings yield) and ROI for a single ticker via Yahoo Finance
# quoteSummary API.
#
# EP  — priority order:
#   1. Yahoo's own earningsYield field (defaultKeyStatistics)
#   2. 1 / trailingPE            (summaryDetail)
#   3. trailingEps / marketPrice (fallback)
#
# ROI — priority order:
#   1. returnOnEquity  (financialData)
#   2. returnOnAssets  (financialData)
#
# Returns a one-row data.frame with columns: ticker, EP, ROI, price.
# Returns NA for any metric that cannot be retrieved.
fetch_yahoo_financials <- function(ticker, pause = 0.4) {
  Sys.sleep(pause)   # polite rate-limiting

  url <- paste0(
    "https://query1.finance.yahoo.com/v10/finance/quoteSummary/",
    URLencode(ticker, reserved = TRUE),
    "?modules=defaultKeyStatistics%2CfinancialData%2CsummaryDetail"
  )

  ua <- paste0(
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ",
    "AppleWebKit/537.36 (KHTML, like Gecko) ",
    "Chrome/124.0 Safari/537.36"
  )

  empty <- data.frame(ticker  = ticker,
                      EP      = NA_real_,
                      ROI     = NA_real_,
                      price   = NA_real_,
                      stringsAsFactors = FALSE)

  tryCatch({
    resp <- GET(url,
                add_headers(`User-Agent` = ua, Accept = "application/json"),
                timeout(12))

    if (status_code(resp) != 200L) {
      message(sprintf("  [WARN] %-6s HTTP %d", ticker, status_code(resp)))
      return(empty)
    }

    parsed  <- fromJSON(content(resp, "text", encoding = "UTF-8"),
                        simplifyVector = FALSE)
    result  <- parsed$quoteSummary$result[[1]]
    ks      <- result$defaultKeyStatistics
    fd      <- result$financialData
    sd_     <- result$summaryDetail

    # -- Earnings Yield (EP) --------------------------------------------------
    safe <- function(x) if (is.numeric(x) && is.finite(x)) x else NULL
    EP <-
      safe(ks$earningsYield$raw)    %||%
      { pe <- safe(sd_$trailingPE$raw);   if (!is.null(pe) && pe > 0) 1/pe else NULL } %||%
      { eps <- safe(ks$trailingEps$raw);
        px  <- safe(sd_$regularMarketPrice$raw);
        if (!is.null(eps) && !is.null(px) && px > 0) eps/px else NULL } %||%
      NA_real_

    # -- ROI ------------------------------------------------------------------
    ROI <-
      safe(fd$returnOnEquity$raw) %||%
      safe(fd$returnOnAssets$raw) %||%
      NA_real_

    price <- safe(sd_$regularMarketPrice$raw) %||% NA_real_

    data.frame(ticker  = ticker,
               EP      = as.numeric(EP),
               ROI     = as.numeric(ROI),
               price   = as.numeric(price),
               stringsAsFactors = FALSE)

  }, error = function(e) {
    message(sprintf("  [WARN] %-6s %s", ticker, conditionMessage(e)))
    empty
  })
}

# Vectorised version: fetch a vector of tickers with progress reporting
fetch_financials_batch <- function(tickers, pause = 0.4) {
  n <- length(tickers)
  cat(sprintf("Fetching EP and ROI from Yahoo Finance for %d ticker(s)...\n", n))

  results <- vector("list", n)
  for (i in seq_len(n)) {
    if (i %% 20 == 0 || i == n)
      cat(sprintf("  %d / %d done\n", i, n))
    results[[i]] <- fetch_yahoo_financials(tickers[i], pause = pause)
  }

  out <- rbindlist(results, use.names = TRUE, fill = TRUE)
  setDF(out)
  out
}

# ==============================================================================
# SECTION 3: FINANCIAL PARAMETERIZATION OF r
# ==============================================================================

# r(b) = (1-b)*EP + b*ROI
# Linear blend: at b=0 expected return equals EP; at b=1 it equals ROI.
compute_r_financial <- function(b, EP, ROI) {
  (1 - b) * EP + b * ROI
}

# Evaluate the financially-parameterized curve for a stock.
# E <- EP (earnings yield plays the role of the scale parameter).
# r <- compute_r_financial(b, EP, ROI).
curve_P_financial <- function(x, EP, ROI, a0, ab, l = 0) {
  r_val <- compute_r_financial(x, EP, ROI)
  curve_P(x, E = EP, r = r_val, a0 = a0, ab = ab, l = l)
}

# Given an observed point (b_obs, p_obs) for a stock with known EP and ROI,
# compute the residual l so that curve_P_financial(b_obs, ..., l) == p_obs.
compute_residual_financial <- function(b_obs, p_obs, EP, ROI, a0, ab) {
  r_val <- compute_r_financial(b_obs, EP, ROI)
  compute_residual(b = b_obs, p = p_obs, E = EP, r = r_val, a0 = a0, ab = ab)
}

# ==============================================================================
# SECTION 4: DEMONSTRATION WITH REAL TICKERS
# ==============================================================================

demo_tickers <- c("AAPL", "MSFT", "NVDA", "JPM", "JNJ", "XOM")

cat(sprintf("Demo subset: %s\n\n", paste(demo_tickers, collapse = ", ")))

fin_data <- fetch_financials_batch(demo_tickers, pause = 0.5)

cat("\nRaw Yahoo Finance output:\n")
cat(sprintf("  %-6s  %10s  %10s  %12s\n", "Ticker", "EP", "ROI", "Price"))
cat(strrep("-", 46), "\n")
for (i in seq_len(nrow(fin_data))) {
  cat(sprintf("  %-6s  %10.4f  %10.4f  %12.2f\n",
              fin_data$ticker[i],
              ifelse(is.na(fin_data$EP[i]),  NA, fin_data$EP[i]),
              ifelse(is.na(fin_data$ROI[i]), NA, fin_data$ROI[i]),
              ifelse(is.na(fin_data$price[i]), NA, fin_data$price[i])))
}
cat("\n")

# --- 4a. r(b) at selected b values -------------------------------------------
a0_demo <- 0.98
ab_demo <- 0.15
b_seq   <- c(0.0, 0.25, 0.50, 0.75, 1.0)

cat("--- r(b) = (1-b)*EP + b*ROI for each ticker ---\n")
cat(sprintf("  %-6s", "b"))
for (tk in fin_data$ticker) cat(sprintf("  %10s", tk))
cat("\n")
cat(strrep("-", 6 + 12 * nrow(fin_data)), "\n")

for (b in b_seq) {
  cat(sprintf("  %-6.2f", b))
  for (i in seq_len(nrow(fin_data))) {
    r_val <- compute_r_financial(b, fin_data$EP[i], fin_data$ROI[i])
    cat(sprintf("  %10.4f", r_val))
  }
  cat("\n")
}
cat("\n")

# --- 4b. Curve P(x) at b = 0.5 for each ticker ------------------------------
b_eval <- 0.5
cat(sprintf("--- Curve P(x) evaluated at x = %.2f ---\n", b_eval))
cat(sprintf("  %-6s  %8s  %8s  %10s  %10s\n",
            "Ticker", "EP", "ROI", "r(b=0.5)", "P(b=0.5)"))
cat(strrep("-", 50), "\n")

for (i in seq_len(nrow(fin_data))) {
  EP_i  <- fin_data$EP[i]
  ROI_i <- fin_data$ROI[i]
  r_i   <- compute_r_financial(b_eval, EP_i, ROI_i)
  P_i   <- curve_P_financial(b_eval, EP_i, ROI_i, a0 = a0_demo, ab = ab_demo, l = 0)
  cat(sprintf("  %-6s  %8.4f  %8.4f  %10.4f  %10.4f\n",
              fin_data$ticker[i], EP_i, ROI_i, r_i, P_i))
}
cat("\n")

# --- 4c. Residual for a synthetic empirical observation ----------------------
cat("--- Residual l for synthetic empirical observations ---\n")
cat("    (p_obs = P_theoretical * 1.10 — a +10% deviation)\n\n")
cat(sprintf("  %-6s  %10s  %10s  %12s\n",
            "Ticker", "P_theory", "p_obs", "residual_l"))
cat(strrep("-", 46), "\n")

for (i in seq_len(nrow(fin_data))) {
  EP_i   <- fin_data$EP[i]
  ROI_i  <- fin_data$ROI[i]
  P_th   <- curve_P_financial(b_eval, EP_i, ROI_i, a0_demo, ab_demo, l = 0)
  p_obs  <- P_th * 1.10   # synthetic: observed 10% above theoretical
  l_res  <- compute_residual_financial(b_eval, p_obs, EP_i, ROI_i, a0_demo, ab_demo)
  cat(sprintf("  %-6s  %10.4f  %10.4f  %12.6f\n",
              fin_data$ticker[i], P_th, p_obs, l_res))
}
cat("\n")

cat("=" |> rep(70) |> paste(collapse = ""), "\n")
cat("  Yahoo Finance curve parameterization complete.\n")
cat("  Financial functions available:\n")
cat("    fetch_yahoo_financials(ticker)                        – single fetch\n")
cat("    fetch_financials_batch(tickers)                       – batch fetch\n")
cat("    compute_r_financial(b, EP, ROI)                       – r(b) blend\n")
cat("    curve_P_financial(x, EP, ROI, a0, ab, l)             – curve eval\n")
cat("    compute_residual_financial(b, p, EP, ROI, a0, ab)    – residual l\n")
cat("=" |> rep(70) |> paste(collapse = ""), "\n\n")
