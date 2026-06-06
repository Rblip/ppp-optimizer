# ==============================================================================
#                     EMPIRICAL CURVE MODEL WITH RESIDUAL
# ==============================================================================
# Author: Peter Paul Dimke
# Date:   January 2026
# Course: Portfolio Optimization
# Purpose: Model a parametric curve P(x) and compute the residual l when an
#          empirical point (b, p) does not lie exactly on the theoretical curve.
#
#                            All rights reserved.
# ==============================================================================
#
# MODEL DEFINITION
# ================
#
# Curve:
#
#         P(x) = E * (1 - x) / (r - x * (r + a(x)))
#
# with adaptive parameter a depending on x:
#
#         a(x) = a0 * (1 - x / ab) - l
#
# Parameters:
#   E   – scale / expected-value parameter
#   r   – base rate
#   a0  – initial slope of the a-function  (0 < a0 < 1)
#   ab  – breakpoint of a (where a would reach 0 without residual)
#   l   – residual (0 for the theoretical curve)
#
# EMPIRICAL FITTING
# =================
#
# Given an observed point (b, p) that may not lie on the theoretical curve
# (l = 0), compute l such that P(b) = p exactly.
#
# Derivation:
#   p = E*(1-b) / (r - b*(r + a))
#   => r - b*(r+a) = E*(1-b)/p
#   => r + a = r/b - E*(1-b)/(p*b)
#   => a = (p*r - E*(1-b)) / (p*b) - r
#
#   Since a = a0*(1 - b/ab) - l:
#   => l = a0*(1 - b/ab) - (p*r - E*(1-b)) / (p*b) + r
#
# ==============================================================================

cat("=" |> rep(70) |> paste(collapse = ""), "\n")
cat("  EMPIRICAL CURVE MODEL: POINT VALIDATION & RESIDUAL ESTIMATION\n")
cat("=" |> rep(70) |> paste(collapse = ""), "\n\n")

# ==============================================================================
# SECTION 1: CORE CURVE FUNCTIONS
# ==============================================================================

# Compute the adaptive a-parameter at a given x value
compute_a <- function(x, a0, ab, l = 0) {
  a0 * (1 - x / ab) - l
}

# Evaluate the curve P(x) at one or more x values
curve_P <- function(x, E, r, a0, ab, l = 0) {
  a   <- compute_a(x, a0, ab, l)
  denom <- r - x * (r + a)
  ifelse(abs(denom) < .Machine$double.eps, NA_real_, E * (1 - x) / denom)
}

# ==============================================================================
# SECTION 2: RESIDUAL COMPUTATION (EMPIRICAL FITTING)
# ==============================================================================

# Given empirical point (b, p), return l so that curve_P(b, ..., l) == p.
# Returns NA when the system has no finite solution (p = 0 or b = 0).
compute_residual <- function(b, p, E, r, a0, ab) {
  if (b == 0 || p == 0) {
    warning("compute_residual: b and p must be nonzero")
    return(NA_real_)
  }
  a_required <- (p * r - E * (1 - b)) / (p * b) - r
  l <- a0 * (1 - b / ab) - a_required
  l
}

# ==============================================================================
# SECTION 3: POINT-ON-CURVE VALIDATION
# ==============================================================================

# Check whether the empirical point (b, p) lies on the curve (l ≈ 0).
# Returns a named list with:
#   on_curve  – logical
#   residual  – computed l
#   P_theoretical – P(b) with l = 0
is_on_curve <- function(b, p, E, r, a0, ab, tol = 1e-6) {
  l_emp     <- compute_residual(b, p, E, r, a0, ab)
  P_theory  <- curve_P(b, E, r, a0, ab, l = 0)
  list(
    on_curve      = isTRUE(abs(l_emp) < tol),
    residual      = l_emp,
    P_theoretical = P_theory,
    P_empirical   = p
  )
}

# ==============================================================================
# SECTION 4: DEMONSTRATION
# ==============================================================================

cat("Default parameters (matching the Desmos model):\n")
cat("  a0 =", 0.98, "  ab =", 0.15, "  E =", 1.1, "  r =", 0.2, "\n\n")

E_val  <- 1.1
r_val  <- 0.2
a0_val <- 0.98
ab_val <- 0.15

# --- 4a. Theoretical curve (l = 0) ------------------------------------------
cat("--- Theoretical curve (l = 0) at selected x values ---\n")
x_demo <- c(0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
P_demo <- curve_P(x_demo, E_val, r_val, a0_val, ab_val, l = 0)

cat(sprintf("  %6s  %10s  %10s\n", "x", "a(x)", "P(x)"))
cat(strrep("-", 32), "\n")
for (i in seq_along(x_demo)) {
  a_i <- compute_a(x_demo[i], a0_val, ab_val, l = 0)
  cat(sprintf("  %6.3f  %10.4f  %10.4f\n", x_demo[i], a_i, P_demo[i]))
}
cat("\n")

# --- 4b. Empirical point exactly on the curve --------------------------------
b_exact <- 0.10
p_exact <- curve_P(b_exact, E_val, r_val, a0_val, ab_val, l = 0)
cat("--- Point exactly on the curve ---\n")
cat(sprintf("  b = %.4f,  P(b) = %.4f  (l = 0 theoretical)\n", b_exact, p_exact))
chk_exact <- is_on_curve(b_exact, p_exact, E_val, r_val, a0_val, ab_val)
cat(sprintf("  on_curve = %s,  residual l = %.2e\n\n",
            chk_exact$on_curve, chk_exact$residual))

# --- 4c. Empirical point NOT on the curve ------------------------------------
b_obs  <- 0.10
p_obs  <- 8.00   # Arbitrary empirical observation
cat("--- Empirical point NOT on the theoretical curve ---\n")
cat(sprintf("  b = %.4f,  p_observed = %.4f,  P_theoretical(b) = %.4f\n",
            b_obs, p_obs, curve_P(b_obs, E_val, r_val, a0_val, ab_val, l = 0)))

chk <- is_on_curve(b_obs, p_obs, E_val, r_val, a0_val, ab_val)
cat(sprintf("  on_curve  = %s\n",  chk$on_curve))
cat(sprintf("  residual l = %.6f  (adjusts the curve to pass through the point)\n",
            chk$residual))

# Verify: curve with computed l passes exactly through (b_obs, p_obs)
P_with_l <- curve_P(b_obs, E_val, r_val, a0_val, ab_val, l = chk$residual)
cat(sprintf("  Verification: P(b, l = %.6f) = %.6f  (should equal p = %.4f)\n\n",
            chk$residual, P_with_l, p_obs))

# --- 4d. Multiple empirical points -------------------------------------------
cat("--- Residuals for multiple empirical observations ---\n")
empirical_points <- data.frame(
  b = c(0.05, 0.10, 0.20, 0.30),
  p = c(6.00, 8.00, 4.50, 3.00)
)
empirical_points$P_theoretical <- mapply(
  curve_P, empirical_points$b,
  MoreArgs = list(E = E_val, r = r_val, a0 = a0_val, ab = ab_val, l = 0)
)
empirical_points$l_residual <- mapply(
  compute_residual, empirical_points$b, empirical_points$p,
  MoreArgs = list(E = E_val, r = r_val, a0 = a0_val, ab = ab_val)
)

cat(sprintf("  %6s  %8s  %14s  %12s\n",
            "b", "p_obs", "P_theoretical", "residual_l"))
cat(strrep("-", 46), "\n")
for (i in seq_len(nrow(empirical_points))) {
  cat(sprintf("  %6.3f  %8.4f  %14.4f  %12.6f\n",
              empirical_points$b[i],
              empirical_points$p[i],
              empirical_points$P_theoretical[i],
              empirical_points$l_residual[i]))
}
cat("\n")

cat("=" |> rep(70) |> paste(collapse = ""), "\n")
cat("  Curve model loaded. Functions available:\n")
cat("    curve_P(x, E, r, a0, ab, l)         – evaluate P(x)\n")
cat("    compute_a(x, a0, ab, l)             – evaluate a(x)\n")
cat("    compute_residual(b, p, E, r, a0, ab) – find l for empirical point\n")
cat("    is_on_curve(b, p, E, r, a0, ab, tol) – validate point vs curve\n")
cat("=" |> rep(70) |> paste(collapse = ""), "\n\n")
