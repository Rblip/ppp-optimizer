#!/usr/bin/env Rscript
# Minimal pricing-curve fit & plot for MSFT — data pulled live from Yahoo Finance.
#
#   P(b) = E * (1 - b) / (r - b * (r + a(b)))      a(b) = a0 * (1 - b / ab)
#
# (a0, ab, r) are estimated jointly from observed prices; r is a single
# constant (cost of equity). Because E enters only as a scale factor,
# every year's curve has the same shape and the same optimum b*.

suppressMessages(library(jsonlite))

curve_P <- function(b, E, r, a0, ab) {
  a <- a0 * (1 - b / ab)
  E * (1 - b) / (r - b * (r + a))
}

# Closed-form link between the firm's chosen plowback b and the perceived-
# disadvantage residual l. If management maximises its OWN (perceived)
# curve, using perceived reinvestment return a(b) = a0*(1 - b/ab) - l rather
# than the true a0*(1 - b/ab), the optimal b solves d/db P = 0, giving:
#
#   b*(l) = 1 - sqrt(1 - ab * (1 - l/a0))
#
# Inverting this lets us read l directly off an OBSERVED b — i.e. treat the
# firm's actual plowback choice as revealed-optimal under its own perceived
# curve, and back out the perception gap that would make it so:
#
#   l(b)  = a0 * (1 - b*(2 - b) / ab)
b_from_l <- function(l, a0, ab) 1 - sqrt(1 - ab * (1 - l / a0))
l_from_b <- function(b, a0, ab) a0 * (1 - b * (2 - b) / ab)


# ---- Pull MSFT annual fundamentals + fiscal-year-end prices from Yahoo -----
fetch_msft_panel <- function(ticker = "MSFT") {
  now <- as.integer(Sys.time())
  since <- now - 6L * 365L * 86400L

  # Annual fundamentals: diluted EPS, net income, equity, dividends, buybacks
  types <- c("annualDilutedEPS", "annualNetIncome", "annualStockholdersEquity",
             "annualCashDividendsPaid", "annualRepurchaseOfCapitalStock")
  fund_url <- sprintf(
    "https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/%s?symbol=%s&type=%s&period1=%d&period2=%d",
    ticker, ticker, paste(types, collapse = ","), since, now)
  series <- fromJSON(fund_url, simplifyVector = FALSE)$timeseries$result

  get_series <- function(field) {
    for (s in series) if (field %in% names(s)) return(s[[field]])
    NULL
  }
  to_named <- function(entries) {
    entries <- Filter(Negate(is.null), entries)
    setNames(vapply(entries, function(e) e$reportedValue$raw, numeric(1)),
             vapply(entries, function(e) e$asOfDate, character(1)))
  }
  eps <- to_named(get_series("annualDilutedEPS"))
  ni  <- to_named(get_series("annualNetIncome"))
  eq  <- to_named(get_series("annualStockholdersEquity"))
  div <- to_named(get_series("annualCashDividendsPaid"))
  bb  <- to_named(get_series("annualRepurchaseOfCapitalStock"))

  # Daily close prices -> nearest fiscal-year-end close
  chart_url <- sprintf(
    "https://query1.finance.yahoo.com/v8/finance/chart/%s?period1=%d&period2=%d&interval=1d",
    ticker, since, now)
  chart  <- fromJSON(chart_url, simplifyVector = FALSE)$chart$result[[1]]
  pdates <- as.Date(as.POSIXct(unlist(chart$timestamp), origin = "1970-01-01", tz = "UTC"))
  pclose <- unlist(chart$indicators$quote[[1]]$close)
  ok <- !is.na(pclose)
  pdates <- pdates[ok]; pclose <- pclose[ok]
  nearest_close <- function(d) pclose[which.min(abs(pdates - as.Date(d)))]

  fy <- Reduce(intersect, list(names(eps), names(ni), names(eq), names(div), names(bb)))
  fy <- sort(fy)

  # Economic plowback: fraction of earnings retained after dividends + buybacks
  payout <- abs(div[fy]) + abs(bb[fy])
  b <- pmin(pmax(1 - payout / ni[fy], 0.01), 0.99)

  data.frame(
    year  = as.integer(format(as.Date(fy), "%Y")),
    eps   = unname(eps[fy]),
    price = vapply(fy, nearest_close, numeric(1)),
    b     = unname(b)
  )
}

cat("Fetching MSFT data from Yahoo Finance...\n")
msft <- fetch_msft_panel()
print(msft, row.names = FALSE)

# ---- Fit (a0, ab, r) jointly by minimising relative squared price error ----
# The surface has multiple local minima, so try several starting points
# (L-BFGS-B with box constraints) and keep the best.
sse <- function(par) {
  P_hat <- curve_P(msft$b, msft$eps, r = par[3], a0 = par[1], ab = par[2])
  if (any(!is.finite(P_hat) | P_hat <= 0)) return(1e12)
  sum((P_hat / msft$price - 1)^2)
}
set.seed(42)
starts <- cbind(runif(20, 0, 1), runif(20, 0, 1), runif(20, 0.01, 0.5))
fits <- lapply(seq_len(nrow(starts)), function(i)
  optim(starts[i, ], sse, method = "L-BFGS-B",
        lower = c(0, 0, 0.01), upper = c(1, 1, 0.5)))
fit <- fits[[which.min(sapply(fits, `[[`, "value"))]]
a0 <- fit$par[1]; ab <- fit$par[2]; r <- fit$par[3]

# b* = b*(l = 0) — value-maximising plowback under the TRUE curve
# (shape is identical for every year, since E only rescales it)
b_star <- b_from_l(0, a0, ab)

# Per-year sentiment residual l, read directly off each year's CHOSEN b:
# l > 0  -> b sits below b* -> management perceives a disadvantage in
#           retaining (and so retains less than the curve would reward)
# l < 0  -> b sits above b* -> management perceives an advantage in
#           retaining (and so retains more than the curve alone implies)
msft$l <- l_from_b(msft$b, a0, ab)

cat(sprintf("\na0 = %.4f   ab = %.4f   r = %.4f   b* = %.4f\n", a0, ab, r, b_star))
cat("\nPer-year residuals l:\n")
print(msft[, c("year", "b", "price", "l")], row.names = FALSE)

# ---- Plot 1: pricing curves, observations, and shared optimum b* -----------
# FY2022 and FY2023 have almost identical EPS (9.65 vs 9.68), so their curves
# nearly coincide -- distinct line types keep both visible, and drawing in
# reverse order puts FY2022 on top.
png("msft_curve_R.png", width = 1000, height = 700, res = 120)
cols <- c("#2E86AB", "#A23B72", "#F18F01", "#C73E1D")
ltys <- c(1, 2, 1, 1)

plot(NULL, xlim = c(0, 1), ylim = c(0, max(msft$price) * 2.2),
     xlab = "Plowback ratio  b", ylab = "Price  P ($)",
     main = "MSFT pricing curves   P(b) = E * shape(b)")
for (i in rev(seq_len(nrow(msft)))) {
  curve(curve_P(x, msft$eps[i], r, a0, ab), from = 0.01, to = 0.97,
        add = TRUE, col = cols[i], lwd = 2, lty = ltys[i])
  points(msft$b[i], msft$price[i], pch = 19, col = cols[i], cex = 1.3)
}
abline(v = b_star, lty = 3, col = "gray40")
legend("topleft", bty = "n",
       legend = c(paste0("FY", msft$year), sprintf("b* = %.3f", b_star)),
       col = c(cols, "gray40"), lwd = c(rep(2, 4), 1), lty = c(ltys, 3))
invisible(dev.off())

# ---- Plot 2: per-year residual l (management-perceived (dis)advantage) -----
png("msft_residuals_R.png", width = 700, height = 500, res = 120)
bar_cols <- ifelse(msft$l > 0, "#C0392B", "#1E8449")
pad <- max(abs(msft$l)) * 0.3
bp <- barplot(msft$l, names.arg = paste0("FY", msft$year), col = bar_cols,
              ylim = range(msft$l) + c(-pad, pad),
              ylab = "Residual  l  (read off observed b)",
              main = "Per-year sentiment residual l\nimplied by the firm's chosen plowback b")
abline(h = 0)
text(bp, msft$l, sprintf("%+.4f", msft$l), pos = ifelse(msft$l > 0, 3, 1), cex = 0.8)
invisible(dev.off())
