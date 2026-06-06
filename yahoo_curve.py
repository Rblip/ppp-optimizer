"""
Yahoo Finance parameterization of the empirical curve model.

Financial mapping
-----------------
    E  →  EP  (earnings yield = EPS / Price, i.e. E/P ratio from Yahoo Finance)

    r(x) = (1 - x) * EP  +  x * ROI

        x = 0  →  r = EP   (pure earnings-yield view)
        x = 1  →  r = ROI  (pure return-on-investment view)
        0 < x < 1  →  convex blend

The residual l then captures how far an empirical observation (b, p) sits
from the curve that is fully driven by real financial data.
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np
import yfinance as yf

from curve_model import compute_a, compute_residual, curve_P


# ---------------------------------------------------------------------------
# Yahoo Finance fetch
# ---------------------------------------------------------------------------

def fetch_financials(ticker: str, pause: float = 0.3) -> dict:
    """
    Fetch EP (earnings yield) and ROI for *ticker* from Yahoo Finance.

    EP  priority:
        1. info["earningsYield"]          (Yahoo's pre-computed field)
        2. 1 / info["trailingPE"]
        3. info["trailingEps"] / info["currentPrice"]

    ROI priority:
        1. info["returnOnEquity"]
        2. info["returnOnAssets"]

    Returns a dict with keys: ticker, EP, ROI, price.
    Missing values are None.
    """
    time.sleep(pause)

    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:
        warnings.warn(f"{ticker}: {exc}")
        return {"ticker": ticker, "EP": None, "ROI": None, "price": None}

    def _finite(val) -> Optional[float]:
        try:
            v = float(val)
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    # -- Earnings Yield -------------------------------------------------------
    EP = (
        _finite(info.get("earningsYield"))
        or (_finite(info.get("trailingPE")) and
            (lambda pe: 1.0 / pe if pe and pe > 0 else None)(_finite(info.get("trailingPE"))))
        or None
    )
    # tighter fallback: EPS / price
    if EP is None:
        eps   = _finite(info.get("trailingEps"))
        price = _finite(info.get("currentPrice") or info.get("regularMarketPrice"))
        if eps is not None and price and price > 0:
            EP = eps / price

    # -- ROI ------------------------------------------------------------------
    ROI = _finite(info.get("returnOnEquity")) or _finite(info.get("returnOnAssets"))

    # -- Price ----------------------------------------------------------------
    price = _finite(info.get("currentPrice") or info.get("regularMarketPrice"))

    return {"ticker": ticker, "EP": EP, "ROI": ROI, "price": price}


def fetch_financials_batch(tickers: list[str],
                           pause: float = 0.3,
                           verbose: bool = True) -> list[dict]:
    """Fetch EP and ROI for a list of tickers with progress output."""
    results = []
    n = len(tickers)
    for i, tk in enumerate(tickers, 1):
        if verbose and (i % 10 == 0 or i == n):
            print(f"  {i}/{n} fetched")
        results.append(fetch_financials(tk, pause=pause))
    return results


# ---------------------------------------------------------------------------
# Financial parameterization
# ---------------------------------------------------------------------------

def compute_r(x: float | np.ndarray,
              EP: float,
              ROI: float) -> float | np.ndarray:
    """
    r(x) = (1 - x)*EP + x*ROI

    Linear blend: earnings-yield view at x=0, ROI view at x=1.
    """
    return (1.0 - x) * EP + x * ROI


def curve_P_financial(x: float | np.ndarray,
                      EP: float,
                      ROI: float,
                      a0: float,
                      ab: float,
                      l: float = 0.0) -> float | np.ndarray:
    """
    Evaluate the curve where E = EP and r(x) = (1-x)*EP + x*ROI.

    Both E and r are grounded in real financial data fetched from Yahoo Finance.
    """
    x = np.asarray(x, dtype=float)
    r = compute_r(x, EP, ROI)
    a = compute_a(x, a0, ab, l)
    denom = r - x * (r + a)
    return np.where(np.abs(denom) < np.finfo(float).eps, np.nan, EP * (1.0 - x) / denom)


def compute_residual_financial(b: float,
                               p: float,
                               EP: float,
                               ROI: float,
                               a0: float,
                               ab: float) -> float:
    """
    Residual l such that curve_P_financial(b, EP, ROI, a0, ab, l) == p.

    r is evaluated at x = b before delegating to the core residual solver.
    """
    r_b = float(compute_r(b, EP, ROI))
    return compute_residual(b=b, p=p, E=EP, r=r_b, a0=a0, ab=ab)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ", "XOM"]
    A0, AB  = 0.98, 0.15

    print("=" * 70)
    print("  YAHOO FINANCE CURVE PARAMETERIZATION")
    print("=" * 70)
    print(f"\nFetching EP and ROI for: {', '.join(TICKERS)}\n")

    rows = fetch_financials_batch(TICKERS)

    # ---- Raw data -----------------------------------------------------------
    print(f"\n{'Ticker':<8} {'EP':>10} {'ROI':>10} {'Price':>12}")
    print("-" * 44)
    for r in rows:
        ep    = f"{r['EP']:.4f}"   if r["EP"]    is not None else "N/A"
        roi   = f"{r['ROI']:.4f}"  if r["ROI"]   is not None else "N/A"
        price = f"{r['price']:.2f}" if r["price"] is not None else "N/A"
        print(f"{r['ticker']:<8} {ep:>10} {roi:>10} {price:>12}")

    # ---- r(b) at selected blend values -------------------------------------
    print("\n--- r(b) = (1-b)*EP + b*ROI ---")
    b_vals = [0.0, 0.25, 0.50, 0.75, 1.0]
    header = f"{'b':>6}" + "".join(f"{r['ticker']:>12}" for r in rows)
    print(header)
    print("-" * len(header))
    for b in b_vals:
        row_str = f"{b:>6.2f}"
        for r in rows:
            if r["EP"] is not None and r["ROI"] is not None:
                val = compute_r(b, r["EP"], r["ROI"])
                row_str += f"{val:>12.4f}"
            else:
                row_str += f"{'N/A':>12}"
        print(row_str)

    # ---- Curve P(x) at b = 0.5 --------------------------------------------
    B_EVAL = 0.5
    print(f"\n--- Curve P(x=b) evaluated at b = {B_EVAL} ---")
    print(f"{'Ticker':<8} {'EP':>8} {'ROI':>8} {'r(b)':>10} {'P(b,l=0)':>12}")
    print("-" * 52)
    for r in rows:
        if r["EP"] is None or r["ROI"] is None:
            print(f"{r['ticker']:<8} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>12}")
            continue
        r_b = compute_r(B_EVAL, r["EP"], r["ROI"])
        P_b = curve_P_financial(B_EVAL, r["EP"], r["ROI"], A0, AB, l=0.0)
        print(f"{r['ticker']:<8} {r['EP']:>8.4f} {r['ROI']:>8.4f} {r_b:>10.4f} {float(P_b):>12.4f}")

    # ---- Residual for synthetic observations (+10% above theoretical) ------
    print(f"\n--- Residual l for +10% empirical deviation at b = {B_EVAL} ---")
    print(f"{'Ticker':<8} {'P_theory':>12} {'p_obs':>10} {'residual_l':>12}")
    print("-" * 48)
    for r in rows:
        if r["EP"] is None or r["ROI"] is None:
            print(f"{r['ticker']:<8} {'N/A':>12} {'N/A':>10} {'N/A':>12}")
            continue
        P_th  = float(curve_P_financial(B_EVAL, r["EP"], r["ROI"], A0, AB, l=0.0))
        p_obs = P_th * 1.10
        try:
            l_res = compute_residual_financial(B_EVAL, p_obs, r["EP"], r["ROI"], A0, AB)
            print(f"{r['ticker']:<8} {P_th:>12.4f} {p_obs:>10.4f} {l_res:>12.6f}")
        except ValueError as e:
            print(f"{r['ticker']:<8} {P_th:>12.4f} {'N/A':>10} {str(e):>12}")

    print("\n" + "=" * 70)
