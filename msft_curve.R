#!/usr/bin/env Rscript
# Minimal pricing-curve fit & plot for MSFT.
#
#   P(b) = E * (1 - b) / (r - b * (r + a(b)))      a(b) = a0 * (1 - b / ab)
#
# (a0, ab, r) are estimated jointly from observed prices; r is a single
# constant (cost of equity). Because E enters only as a scale factor,
# every year's curve has the same shape and the same optimum b*.

curve_P <- function(b, E, r, a0, ab) {
  a <- a0 * (1 - b / ab)
  E * (1 - b) / (r - b * (r + a))
}

# MSFT panel: FY2022-FY2025 (plowback b, price P, EPS)
msft <- data.frame(
  year  = 2022:2025,
  b     = c(0.3012, 0.4190, 0.5572, 0.5826),
  price = c(248.49, 332.67, 440.03, 493.47),
  eps   = c(9.65,   9.68,   11.80,  13.64)
)

# Fit (a0, ab, r) by minimising relative squared pricing error.
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

# b* — value-maximising plowback (shape is identical for every year)
b_grid <- seq(0.01, 0.99, length.out = 2000)
b_star <- b_grid[which.max(curve_P(b_grid, E = 1, r, a0, ab))]

cat(sprintf("a0 = %.4f   ab = %.4f   r = %.4f   b* = %.4f\n", a0, ab, r, b_star))

# Plot: each year's curve P(b) = E * shape(b), observations, shared b*
png("msft_curve_R.png", width = 1000, height = 700, res = 120)
cols <- c("#2E86AB", "#A23B72", "#F18F01", "#C73E1D")
plot(NULL, xlim = c(0, 1), ylim = c(0, max(msft$price) * 2.2),
     xlab = "Plowback ratio  b", ylab = "Price  P ($)",
     main = "MSFT pricing curves   P(b) = E * shape(b)")
for (i in seq_len(nrow(msft))) {
  curve(curve_P(x, msft$eps[i], r, a0, ab), from = 0.01, to = 0.97,
        add = TRUE, col = cols[i], lwd = 2)
  points(msft$b[i], msft$price[i], pch = 19, col = cols[i], cex = 1.3)
}
abline(v = b_star, lty = 2, col = "gray40")
legend("topleft", bty = "n",
       legend = c(paste0("FY", msft$year), sprintf("b* = %.3f", b_star)),
       col = c(cols, "gray40"), lwd = c(rep(2, 4), 1), lty = c(rep(1, 4), 2))
invisible(dev.off())
