"""
Firm-specific estimation of a0 and ab for MSFT — Gordon Growth Model
with financially-parameterized discount rate and constrained optimization.

Model
-----
    P(b) = E * (1 - b) / (r(b) - b * (r(b) + a(b)))

    a(b) = a0 * (1 - b / ab) - l          [adaptive, firm-specific]
    r_t  = (1 - b_t) * EP_t  +  b_t * ROI_t   [given each year from observables — not a parameter]

Interpretation of l  — management's perceived disadvantage of b
---------------------------------------------------------------
    l is the residual between the theoretical reinvestment premium a0*(1-b/ab)
    and the premium actually required to price the stock at its observed level.

    l > 0  Management perceives the current plowback ratio b as disadvantageous:
           the effective reinvestment premium is discounted below the theoretical
           curve.  They earn less from reinvestment than the model expects.

    l = 0  Management is neutral about b; the stock is priced exactly on the
           theoretical curve.

    l < 0  Management perceives an advantage in b: they earn above the
           theoretical curve at the current reinvestment rate.

    Because l is estimated residually from observed prices, it is a revealed
    preference — it reflects what management's capital-allocation decisions
    imply about their own assessment of the return on retained earnings.

Identification of b and P from MSFT annual data
------------------------------------------------
    b_t  = economic plowback ratio for fiscal year t
           = 1 - (dividends_t + buybacks_t) / net_income_t
           Fraction of earnings truly retained (not returned via any channel).

    P_t  = stock price at fiscal-year end.
    E_t  = annual diluted EPS.
    EP_t = E_t / P_t   (earnings yield → feeds r).
    ROI_t = net_income_t / equity_t   (annual ROE → feeds r).

Constraints for interior optimum  (from Desmos slider bounds)
--------------------------------------------------------------
    a0 ∈ [0, 1],  ab ∈ [0, 1]

    Guarantee P(b) has a peak at b* ∈ (0, 1): an optimal reinvestment rate.
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
    print("Model:  P(b) = E*(1-b) / (r_t - b*(r_t+a(b)))")
    print("        a(b) = a0*(1 - b/ab) - l   [estimated, firm-specific, time-invariant]")
    print("        r_t  = (1-b_t)*EP_t + b_t*ROI_t  [given each year from observables]")
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

    # Per-observation table — r_t is given each year from observables, not a parameter
    years = panel["date"].dt.year.tolist()
    print(f"\n  r_t is given each year:  r_t = (1 - b_t)*EP_t + b_t*ROI_t  (not estimated)")
    print(f"  l_t = management's perceived disadvantage of b_t")
    print(f"  (l > 0: disadvantage — earns below curve;  l < 0: advantage — earns above curve)\n")
    print(f"  {'FY':>6} {'b_t':>8} {'r_t (given)':>13} {'a_required':>12} "
          f"{'a_fitted':>10} {'l_t':>10}  {'signal':>20}")
    print("  " + "-" * 86)
    for i in range(len(result["b_used"])):
        b_i   = result["b_used"][i]
        r_i   = float(compute_r(b_i, panel["EP"].iloc[i], panel["ROI"].iloc[i]))
        aq_i  = result["a_required"][i]
        af_i  = result["a_fitted"][i]
        l_i   = result["residuals"][i]
        signal = "disadvantage (l > 0)" if l_i > 1e-4 else \
                 "advantage   (l < 0)" if l_i < -1e-4 else "neutral"
        print(f"  {years[i]:>6} {b_i:>8.4f} {r_i:>13.6f} {aq_i:>12.6f} "
              f"{af_i:>10.6f} {l_i:>10.6f}  {signal:>20}")

    print("\n" + "=" * 70)
    print("  Interpretation:")
    print(f"  a(b) = {result['a0']:.4f}*(1 - b/{result['ab']:.4f})  — theoretical reinvestment premium")
    print(f"  ab = {result['ab']:.4f}: plowback at which the premium reaches zero.")
    print(f"  b* = {result['b_star']:.4f}: value-maximising plowback (interior optimum).")
    print()
    print("  l by year captures whether management's actual capital-allocation")
    print("  decisions reveal a perceived disadvantage (l > 0) or advantage (l < 0)")
    print("  in the plowback ratio chosen that year relative to the fitted curve.")
    print("=" * 70)
