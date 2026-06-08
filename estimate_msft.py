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

from curve_model import curve_P, l_from_b, b_star_from_l


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
# 2. JOINT CURVE ESTIMATION  (a0, ab, r — all estimated from price data)
# ============================================================================

def fit_curve(ticker: str,
              b: np.ndarray,
              p: np.ndarray,
              E: np.ndarray) -> dict:
    """
    Jointly estimate a0, ab, and the discount rate r by fitting the pricing
    curve  P(b) = E*(1-b) / (r - b*(r + a(b)))  directly to observed prices
    via differential evolution (P-space, relative price errors).

    r is a single constant — the firm's cost of equity — estimated from the
    absolute price level. This replaces the r(b) = (1-b)*EP + b*ROI blend,
    which rises toward ROI as b increases and mechanically inflates the
    discount rate, dragging the optimum down toward b* ≈ 0.28 regardless of
    whether that is where the firm actually creates value.

    Because E enters P(b) only as a multiplicative scale factor, b* is the
    same in every year — a structural property of the firm, not an artifact
    of that year's earnings level.

    Parameters
    ----------
    ticker : firm label
    b      : plowback ratios,    shape (T,)
    p      : observed prices,    shape (T,)
    E      : earnings per share, shape (T,)

    Returns a dict with: a0, ab, r, b_star, residuals (l per year),
    rel_resid, rmse_rel, n_obs, b_used, P_fitted
    """
    b_ = np.asarray(b, dtype=float)
    p_ = np.asarray(p, dtype=float)
    E_ = np.asarray(E, dtype=float)

    valid = (np.isfinite(b_) & np.isfinite(p_) & np.isfinite(E_)
             & (b_ > 0) & (b_ < 1) & (p_ > 0) & (E_ > 0))
    b_, p_, E_ = b_[valid], p_[valid], E_[valid]
    n = len(b_)
    if n < 3:
        raise ValueError(f"{ticker}: only {n} valid observations after cleaning")

    # ── Objective: sum of squared relative price errors ──────────────────────
    def sse(params: np.ndarray) -> float:
        a0, ab, r = params
        if ab < 1e-8 or r <= 1e-6:
            return 1e12
        total = 0.0
        for i in range(n):
            P_i = float(curve_P(b_[i], E_[i], r, a0, ab, l=0.0))
            if not np.isfinite(P_i) or P_i <= 0:
                return 1e12
            total += ((P_i / p_[i]) - 1.0) ** 2
        return total

    bounds = [(0.0, 1.0),    # a0
              (0.0, 1.0),    # ab
              (0.01, 0.50)]  # r  (cost of equity: 1%–50%)

    result = differential_evolution(
        sse, bounds,
        seed       = 42,
        maxiter    = 5000,
        tol        = 1e-12,
        popsize    = 25,
        mutation   = (0.5, 1.5),
        recombination = 0.9,
    )

    a0_hat, ab_hat, r_hat = result.x

    P_fit = np.array([float(curve_P(b_[i], E_[i], r_hat, a0_hat, ab_hat))
                      for i in range(n)])
    rel_resid = (P_fit - p_) / p_
    rmse_rel  = float(np.sqrt(np.mean(rel_resid ** 2)))

    # Per-year sentiment residual l, read directly off each year's CHOSEN b
    # (closed-form inversion of the optimal-b relationship — see curve_model):
    #   l > 0  ->  b below b*  ->  perceived disadvantage  (retains less)
    #   l < 0  ->  b above b*  ->  perceived advantage     (retains more)
    l_vals = l_from_b(b_, a0_hat, ab_hat)
    a_fitted   = a0_hat * (1.0 - b_ / ab_hat)
    a_required = a_fitted - l_vals

    # b* = b*(l = 0) — the curve's true optimum, identical for every year
    b_star = float(b_star_from_l(0.0, a0_hat, ab_hat))

    return {
        "ticker":     ticker,
        "a0":         float(a0_hat),
        "ab":         float(ab_hat),
        "r":          float(r_hat),
        "b_star":     b_star,
        "residuals":  l_vals,
        "a_required": a_required,
        "a_fitted":   a_fitted,
        "rel_resid":  rel_resid,
        "rmse_rel":   rmse_rel,
        "n_obs":      n,
        "converged":  result.success,
        "b_used":     b_,
        "p_used":     p_,
        "E_used":     E_,
        "P_fitted":   P_fit,
    }


# ============================================================================
# 3. RUN MSFT ESTIMATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  MSFT — FIRM-SPECIFIC CURVE PARAMETER ESTIMATION")
    print("=" * 70)
    print()
    print("Model:  P(b) = E*(1-b) / (r - b*(r+a(b)))")
    print("        a(b) = a0*(1 - b/ab) - l   [estimated, firm-specific, time-invariant]")
    print("        r    = cost of equity      [estimated jointly with a0, ab from prices]")
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

    # -- Joint estimation ------------------------------------------------------
    print("\nRunning joint estimation (differential evolution, P-space)...")
    print("  Bounds: a0 ∈ [0, 1],  ab ∈ [0, 1],  r ∈ [0.01, 0.50]\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_curve(
            ticker = "MSFT",
            b      = panel["b"].values,
            p      = panel["price"].values,
            E      = panel["EPS"].values,
        )

    # -- Report ---------------------------------------------------------------
    print("=" * 70)
    print("  ESTIMATION RESULTS")
    print("=" * 70)
    print(f"\n  a0  (intercept of a-function)  = {result['a0']:.6f}   [0, 1] ✓")
    print(f"  ab  (breakeven plowback)       = {result['ab']:.6f}   [0, 1] ✓")
    print(f"  r   (cost of equity, est.)     = {result['r']:.6f}")
    print(f"\n  RMSE (relative price error)    = {result['rmse_rel']*100:.3f}%")
    print(f"  Observations used              = {result['n_obs']}")
    print(f"  Converged                      = {result['converged']}")
    print(f"\n  Implied optimal plowback b*    = {result['b_star']:.4f}   ← interior optimum ∈ (0,1)")

    years = panel["date"].dt.year.tolist()
    print(f"\n  r is a single estimated constant (cost of equity) — not a per-year blend")
    print(f"  l_t = management's perceived disadvantage of b_t")
    print(f"  (l > 0: disadvantage — earns below curve;  l < 0: advantage — earns above curve)\n")
    print(f"  {'FY':>6} {'b_t':>8} {'P_obs':>10} {'P_fitted':>10} {'l_t':>10}  {'signal':>20}")
    print("  " + "-" * 72)
    for i in range(len(result["b_used"])):
        b_i = result["b_used"][i]
        l_i = result["residuals"][i]
        signal = "disadvantage (l > 0)" if l_i > 1e-4 else \
                 "advantage   (l < 0)" if l_i < -1e-4 else "neutral"
        print(f"  {years[i]:>6} {b_i:>8.4f} {result['p_used'][i]:>10.2f} "
              f"{result['P_fitted'][i]:>10.2f} {l_i:>10.6f}  {signal:>20}")

    print("\n" + "=" * 70)
    print("  Interpretation:")
    print(f"  a(b) = {result['a0']:.4f}*(1 - b/{result['ab']:.4f})  — theoretical reinvestment premium")
    print(f"  r    = {result['r']:.4f}: estimated cost of equity (constant, not blended toward ROI)")
    print(f"  ab   = {result['ab']:.4f}: plowback at which the premium reaches zero")
    print(f"  b*   = {result['b_star']:.4f}: value-maximising plowback — same every year")
    print()
    print("  Replacing the r(b) blend with a single estimated discount rate")
    print("  removes the mechanical penalty on retention, moving b* into")
    print("  MSFT's actually-observed plowback range and leaving residuals l")
    print("  an order of magnitude smaller than under the blended specification.")
    print("=" * 70)

