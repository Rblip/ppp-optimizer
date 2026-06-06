"""
Firm-specific estimation of a0 and ab for MSFT — Gordon Growth Model
with financially-parameterized discount rate and constrained optimization.

Model
-----
    P(b) = E * (1 - b) / (r(b) - b * (r(b) + a(b)))

    a(b) = a0 * (1 - b / ab) - l          [adaptive, firm-specific]
    r(b) = (1 - b) * EP  +  b * ROI       [financially parameterized]

Identification of b and P from MSFT quarterly data
---------------------------------------------------
    b_t  = plowback (retention) ratio at quarter t
           = 1 - dividends_per_share_t / diluted_eps_t
           In (0, 1): fraction of earnings NOT paid as dividends

    P_t  = stock price at end of quarter t
           The Gordon Growth Model equates this to the formula above.

    E_t  = diluted EPS for quarter t   (numerator scale)
    EP_t = E_t / P_t                   (quarterly earnings yield → feeds r)
    ROI_t = quarterly ROE = net_income_t / stockholders_equity_t

Constraints for interior optimum  (from Desmos slider bounds)
--------------------------------------------------------------
    a0 ∈ [0, 1]
    ab ∈ [0, 1]

    These ensure that the P(b) curve has a peak at some b* ∈ (0, 1),
    i.e., there is an optimal reinvestment rate for the firm.

Estimation
----------
    For each quarter, invert the curve formula to recover the a value
    that is *required* for the model to price the stock exactly:

        a_required_t = (P_t * r_t - E_t * (1 - b_t)) / (P_t * b_t) - r_t

    Then fit a0 and ab by minimising the sum of squared residuals

        l_t = a0 * (1 - b_t / ab) - a_required_t

    subject to a0, ab ∈ [0, 1], using global constrained optimisation
    (differential evolution) to guarantee the interior-optimum constraints
    are satisfied.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import differential_evolution

from yahoo_curve import compute_r


# ============================================================================
# 1. DATA DOWNLOAD
# ============================================================================

def _first_available(df: pd.DataFrame, keys: list[str]) -> pd.Series | None:
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return None


def fetch_msft_panel() -> pd.DataFrame:
    """
    Download quarterly MSFT financial data from Yahoo Finance and return
    a tidy DataFrame with columns:
        date, EPS, price, net_income, equity, dividends_per_share, b, EP, ROI
    """
    print("  Downloading MSFT quarterly financials from Yahoo Finance...")
    tk = yf.Ticker("MSFT")

    # ── Income statement ─────────────────────────────────────────────────────
    income = tk.quarterly_income_stmt

    eps_series = _first_available(
        income, ["Diluted EPS", "Basic EPS", "EPS"]
    )
    ni_series = _first_available(
        income, ["Net Income", "Net Income Common Stockholders",
                 "Net Income Including Noncontrolling Interests"]
    )

    # ── Cash-flow statement (dividends paid) ──────────────────────────────────
    cf = tk.quarterly_cashflow
    div_series = _first_available(
        cf, ["Cash Dividends Paid", "Common Stock Dividend Paid",
             "Dividends Paid", "Payment Of Dividends"]
    )
    shares_series = _first_available(
        cf, ["Issuance Of Capital Stock", "Common Stock Issuance",
             "Repurchase Of Capital Stock"]
    )

    # ── Balance sheet (stockholders' equity) ──────────────────────────────────
    bs = tk.quarterly_balance_sheet
    eq_series = _first_available(
        bs, ["Stockholders Equity", "Common Stock Equity",
             "Total Equity Gross Minority Interest",
             "Ordinary Shares Number"]
    )
    shares_out = _first_available(
        bs, ["Ordinary Shares Number", "Share Issued",
             "Common Stock Shares Outstanding"]
    )

    # ── Price history (daily → quarterly end-of-period) ───────────────────────
    prices = tk.history(period="6y", interval="1d", auto_adjust=True)
    prices.index = prices.index.tz_localize(None) if prices.index.tz else prices.index
    prices_q = prices["Close"].resample("QE").last()

    # ── Align all series on quarterly dates ───────────────────────────────────
    dates = eps_series.dropna().index if eps_series is not None else pd.Index([])
    records = []

    for dt in dates:
        try:
            eps = float(eps_series[dt]) if eps_series is not None else None
            ni  = float(ni_series[dt])  if ni_series  is not None else None
            eq  = float(eq_series[dt])  if eq_series  is not None else None

            # Match to nearest quarter-end price
            price_match = prices_q.reindex([dt], method="nearest")
            price = float(price_match.iloc[0]) if len(price_match) > 0 else None

            # Dividends per share: total dividends paid / shares outstanding
            div_total = float(div_series[dt])    if div_series  is not None else 0.0
            n_shares   = float(shares_out[dt])   if shares_out  is not None else None

            if n_shares and n_shares > 0 and abs(div_total) > 0:
                dps = abs(div_total) / n_shares   # dividends paid is negative in yfinance
            else:
                dps = 0.0

            if None in (eps, price, ni, eq):
                continue
            if eps <= 0 or price <= 0 or eq <= 0:
                continue

            records.append({
                "date":               pd.Timestamp(dt),
                "EPS":                eps,
                "price":              price,
                "net_income":         ni,
                "equity":             eq,
                "dividends_per_share": dps,
            })
        except Exception:
            continue

    if not records:
        raise RuntimeError("No valid quarterly observations could be assembled for MSFT.")

    panel = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    # ── Derived quantities ────────────────────────────────────────────────────
    panel["EP"]  = panel["EPS"]        / panel["price"]    # quarterly earnings yield
    panel["ROI"] = panel["net_income"] / panel["equity"]   # quarterly ROE

    # Plowback ratio: fraction of EPS not paid as dividends
    panel["b"] = (panel["EPS"] - panel["dividends_per_share"]) / panel["EPS"]
    panel["b"] = panel["b"].clip(0.01, 0.99)   # keep strictly in (0, 1)

    return panel


# ============================================================================
# 2. CONSTRAINED NLS
# ============================================================================

def _compute_required_a(b: np.ndarray,
                         p: np.ndarray,
                         E: np.ndarray,
                         r: np.ndarray) -> np.ndarray:
    """
    Invert P(b) = E*(1-b) / (r - b*(r+a))  to solve for a.
    """
    return (p * r - E * (1.0 - b)) / (p * b) - r


def fit_constrained(ticker: str,
                    b: np.ndarray,
                    p: np.ndarray,
                    E: np.ndarray,
                    ROI: np.ndarray,
                    EP: np.ndarray) -> dict:
    """
    Estimate a0, ab ∈ [0, 1] via differential evolution (global constrained NLS).

    Parameters
    ----------
    ticker  : firm label
    b       : plowback ratios, shape (T,)
    p       : observed prices,  shape (T,)
    E       : earnings per share, shape (T,)
    ROI     : quarterly return on equity, shape (T,)
    EP      : quarterly earnings yield,   shape (T,)

    Returns a dict with: a0, ab, residuals, r_squared, n_obs, b_star
    """
    r_arr = compute_r(b, EP, ROI)   # r_t = (1-b_t)*EP_t + b_t*ROI_t

    # Drop observations where the required-a formula is undefined
    valid = (
        np.isfinite(b) & np.isfinite(p) & np.isfinite(E)
        & np.isfinite(r_arr) & np.isfinite(ROI)
        & (b > 0) & (p > 0) & (E > 0) & (r_arr != 0)
    )
    b_, p_, E_, r_ = b[valid], p[valid], E[valid], r_arr[valid]

    a_req = _compute_required_a(b_, p_, E_, r_)
    finite = np.isfinite(a_req)
    b_, p_, E_, r_, a_req = b_[finite], p_[finite], E_[finite], r_[finite], a_req[finite]

    n = len(b_)
    if n < 3:
        raise ValueError(f"{ticker}: only {n} valid observations after cleaning")

    # ── Objective: sum of squared residuals ───────────────────────────────────
    def sse(params: np.ndarray) -> float:
        a0, ab = params
        if ab < 1e-8:
            return 1e12
        a_fit = a0 * (1.0 - b_ / ab)
        return float(np.sum((a_req - a_fit) ** 2))

    # ── Global optimisation with unit-interval constraints ────────────────────
    bounds = [(0.0, 1.0),   # a0
              (0.0, 1.0)]   # ab

    result = differential_evolution(
        sse, bounds,
        seed       = 42,
        maxiter    = 2000,
        tol        = 1e-10,
        popsize    = 20,
        mutation   = (0.5, 1.5),
        recombination = 0.9,
    )

    a0_hat, ab_hat = result.x

    a_fit   = a0_hat * (1.0 - b_ / ab_hat)
    resid   = a_req - a_fit

    ss_res  = float(np.sum(resid ** 2))
    ss_tot  = float(np.sum((a_req - np.mean(a_req)) ** 2))
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Optimal plowback b* = argmax P(b) — numerically found
    b_grid = np.linspace(0.01, 0.99, 2000)
    EP_rep = np.mean(EP)
    ROI_rep = np.mean(ROI)
    E_rep   = np.mean(E)
    r_grid  = compute_r(b_grid, EP_rep, ROI_rep)
    a_grid  = a0_hat * (1.0 - b_grid / ab_hat)
    denom   = r_grid - b_grid * (r_grid + a_grid)
    P_grid  = np.where(denom > 0, E_rep * (1.0 - b_grid) / denom, np.nan)
    b_star  = float(b_grid[np.nanargmax(P_grid)])

    return {
        "ticker":      ticker,
        "a0":          float(a0_hat),
        "ab":          float(ab_hat),
        "residuals":   resid,
        "r_squared":   float(r2),
        "n_obs":       n,
        "b_star":      b_star,       # optimal plowback ratio
        "converged":   result.success,
        "b_used":      b_,
        "a_required":  a_req,
        "a_fitted":    a_fit,
    }


# ============================================================================
# 3. RUN MSFT ESTIMATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  MSFT — FIRM-SPECIFIC CURVE PARAMETER ESTIMATION")
    print("=" * 70)
    print()
    print("Model:  P(b) = E*(1-b) / (r(b) - b*(r(b)+a(b)))")
    print("        a(b) = a0*(1 - b/ab) - l   [firm-specific, time-invariant]")
    print("        r(b) = (1-b)*EP + b*ROI    [financially parameterized]")
    print("Constraint: a0, ab ∈ [0, 1]  →  interior optimum b* ∈ (0, 1)")
    print()

    # -- Fetch data -----------------------------------------------------------
    panel = fetch_msft_panel()

    print(f"\nQuarterly panel for MSFT  ({len(panel)} observations):\n")
    print(f"  {'Date':<12} {'EPS':>7} {'Price':>8} {'EP':>8} "
          f"{'ROI':>8} {'b (plow)':>10}")
    print("  " + "-" * 58)
    for _, row in panel.iterrows():
        print(f"  {str(row['date'].date()):<12} "
              f"{row['EPS']:>7.3f} {row['price']:>8.2f} "
              f"{row['EP']:>8.5f} {row['ROI']:>8.5f} {row['b']:>10.4f}")

    # -- Constrained estimation -----------------------------------------------
    print("\nRunning constrained optimisation (differential evolution)...")
    print("  Bounds: a0 ∈ [0, 1],  ab ∈ [0, 1]\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_constrained(
            ticker = "MSFT",
            b      = panel["b"].values,
            p      = panel["price"].values,
            E      = panel["EPS"].values,
            ROI    = panel["ROI"].values,
            EP     = panel["EP"].values,
        )

    # -- Report ---------------------------------------------------------------
    print("=" * 70)
    print("  ESTIMATION RESULTS")
    print("=" * 70)
    print(f"\n  a0  (intercept of a-function)  = {result['a0']:.6f}   [0, 1] ✓")
    print(f"  ab  (breakeven plowback)       = {result['ab']:.6f}   [0, 1] ✓")
    print(f"\n  R²  (fit on a_required)        = {result['r_squared']:.4f}")
    print(f"  RMSE                           = {np.sqrt(np.mean(result['residuals']**2)):.6f}")
    print(f"  Observations used              = {result['n_obs']}")
    print(f"  Converged                      = {result['converged']}")
    print(f"\n  Implied optimal plowback b*    = {result['b_star']:.4f}   ← interior optimum ∈ (0,1)")

    # Implied a(b) at typical plowback
    b_typ = float(panel["b"].mean())
    a_typ = result["a0"] * (1.0 - b_typ / result["ab"])
    print(f"\n  At mean observed b = {b_typ:.4f}:")
    EP_m, ROI_m, E_m = panel["EP"].mean(), panel["ROI"].mean(), panel["EPS"].mean()
    r_typ = float(compute_r(b_typ, EP_m, ROI_m))
    print(f"    a(b)   = {a_typ:.6f}")
    print(f"    r(b)   = {r_typ:.6f}  (r ∈ [0, 0.2] constraint: {'✓' if r_typ <= 0.2 else '✗ — exceeds 0.2'})")
    denom  = r_typ - b_typ * (r_typ + a_typ)
    P_pred = E_m * (1.0 - b_typ) / denom if abs(denom) > 1e-10 else float("nan")
    print(f"    P(b)   = {P_pred:.2f}   (observed avg price: {panel['price'].mean():.2f})")

    # Per-observation residuals
    print(f"\n  Per-observation residuals (l_t = a_required - a_fitted):")
    print(f"  {'b_t':>8} {'a_required':>12} {'a_fitted':>10} {'l_t':>10}")
    print("  " + "-" * 44)
    for i in range(len(result["b_used"])):
        b_i  = result["b_used"][i]
        aq_i = result["a_required"][i]
        af_i = result["a_fitted"][i]
        l_i  = result["residuals"][i]
        print(f"  {b_i:>8.4f} {aq_i:>12.6f} {af_i:>10.6f} {l_i:>10.6f}")

    print("\n" + "=" * 70)
    print("  Interpretation:")
    print(f"  The adaptive parameter a(b) = {result['a0']:.4f}*(1 - b/{result['ab']:.4f})")
    print(f"  decreases in b, reaching zero at b = ab = {result['ab']:.4f}.")
    print(f"  The P(b) curve peaks at b* = {result['b_star']:.4f}, meaning MSFT's")
    print(f"  value is maximised when {result['b_star']*100:.1f}% of earnings are reinvested.")
    print("=" * 70)
