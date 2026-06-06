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
    Download ANNUAL MSFT financial data from Yahoo Finance (4 fiscal years)
    and return a tidy DataFrame.

    Plowback ratio uses the ECONOMIC definition:
        b = 1 - (dividends + buybacks) / net_income

    This is wider-ranging than the dividend-only ratio because MSFT returns
    large amounts via buybacks, giving b meaningful variation across years.
    """
    print("  Downloading MSFT annual financials from Yahoo Finance...")
    tk = yf.Ticker("MSFT")

    # ── Annual income statement ───────────────────────────────────────────────
    income = tk.income_stmt        # ANNUAL: 4 fiscal years (FY end = June 30)

    eps_series = _first_available(income, ["Diluted EPS", "Basic EPS"])
    ni_series  = _first_available(income, ["Net Income",
                                           "Net Income Common Stockholders"])

    # ── Annual cash flow ──────────────────────────────────────────────────────
    cf = tk.cashflow

    div_series  = _first_available(cf, ["Cash Dividends Paid",
                                        "Common Stock Dividend Paid"])
    bb_series   = _first_available(cf, ["Repurchase Of Capital Stock",
                                        "Common Stock Payments"])

    # ── Annual balance sheet ──────────────────────────────────────────────────
    bs = tk.balance_sheet

    eq_series = _first_available(bs, ["Stockholders Equity",
                                      "Common Stock Equity"])

    # ── Fiscal-year-end prices ────────────────────────────────────────────────
    prices = tk.history(period="6y", interval="1d", auto_adjust=True)
    prices.index = prices.index.tz_localize(None) if prices.index.tz else prices.index
    prices_annual = prices["Close"].resample("YE-JUN").last()   # fiscal year = July–June

    if eps_series is None or ni_series is None or eq_series is None:
        raise RuntimeError("Could not retrieve core annual financials from Yahoo Finance.")

    dates   = eps_series.dropna().index
    records = []

    for dt in dates:
        try:
            eps = float(eps_series.get(dt, float("nan")))
            ni  = float(ni_series.get(dt,  float("nan")))
            eq  = float(eq_series.get(dt,  float("nan")))

            # Fiscal-year-end price (nearest available trading day)
            idx = prices_annual.index.get_indexer([dt], method="nearest")[0]
            price = float(prices_annual.iloc[idx]) if idx >= 0 else float("nan")

            # Total capital returned to shareholders (both negative in yfinance)
            div     = abs(float(div_series.get(dt, 0) or 0)) if div_series is not None else 0.0
            buyback = abs(float(bb_series.get(dt,  0) or 0)) if bb_series  is not None else 0.0
            total_payout = div + buyback

            if not all(map(np.isfinite, [eps, ni, eq, price])):
                continue
            if eps <= 0 or price <= 0 or eq <= 0 or ni <= 0:
                continue

            # Economic plowback: fraction of earnings NOT returned to shareholders
            b_economic = (ni - total_payout) / ni
            b_economic = float(np.clip(b_economic, 0.01, 0.99))

            records.append({
                "date":        pd.Timestamp(dt),
                "EPS":         eps,
                "price":       price,
                "net_income":  ni,
                "equity":      eq,
                "dividends":   div,
                "buybacks":    buyback,
                "b":           b_economic,
            })
        except Exception:
            continue

    if len(records) < 2:
        raise RuntimeError("Fewer than 2 valid annual observations assembled for MSFT.")

    panel = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    # Annual EP and ROE
    panel["EP"]  = panel["EPS"]       / panel["price"]    # earnings yield (annual)
    panel["ROI"] = panel["net_income"] / panel["equity"]   # return on equity (annual)

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

    print(f"\nAnnual panel for MSFT  ({len(panel)} fiscal years):\n")
    print(f"  {'FY end':<12} {'EPS':>7} {'Price':>8} {'Divs $B':>9} "
          f"{'BB $B':>7} {'EP':>8} {'ROI':>8} {'b (econ)':>10}")
    print("  " + "-" * 74)
    for _, row in panel.iterrows():
        print(f"  {str(row['date'].date()):<12} "
              f"{row['EPS']:>7.2f} {row['price']:>8.2f} "
              f"{row['dividends']/1e9:>9.1f} {row['buybacks']/1e9:>7.1f} "
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
