"""
Firm-specific, time-invariant estimation of a0 and ab via OLS.

Background
----------
The adaptive parameter is:

    a(x) = a0 * (1 - x/ab) - l                          ... (1)

Rearranged in terms of what the data *require*:

    a_required_t  =  (p_t * r_t - EP * (1 - b_t)) / (p_t * b_t) - r_t   ... (2)

where for observation t we observe (b_t, p_t) and r_t = (1-b_t)*EP + b_t*ROI.

Substituting (1) into (2) with the residual l_t as the error term gives:

    a_required_t  =  a0 - (a0/ab) * b_t  +  l_t

This is a standard OLS regression with two coefficients:

    γ1 = a0          (intercept)
    γ2 = a0 / ab     (slope on  -b_t)

so the design matrix is X_t = [1, -b_t].  Recovery:

    a0  = γ1
    ab  = γ1 / γ2

The OLS residuals {l_t} are exactly the per-period residuals of the curve model.

Firm-specificity
----------------
The regression is run separately for each firm using that firm's panel of
observations {(b_t, p_t)}.  Because we pool all time periods of a single
firm, a0 and ab are time-invariant by construction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from yahoo_curve import compute_r, curve_P_financial


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FirmParams:
    """Firm-specific curve parameters estimated by OLS."""
    ticker:    str
    a0:        float
    ab:        float
    residuals: np.ndarray          # l_t for each observation
    r_squared: float
    n_obs:     int
    b_obs:     np.ndarray = field(repr=False)
    p_obs:     np.ndarray = field(repr=False)

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals ** 2)))


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _compute_required_a(b: np.ndarray,
                         p: np.ndarray,
                         EP: float,
                         r: np.ndarray) -> np.ndarray:
    """
    Solve for a such that P(b) = p given E=EP, r (vectorised).

        p = EP*(1-b) / (r - b*(r+a))
        => a = (p*r - EP*(1-b)) / (p*b) - r
    """
    return (p * r - EP * (1.0 - b)) / (p * b) - r


# ---------------------------------------------------------------------------
# OLS fit
# ---------------------------------------------------------------------------

def fit_firm_params(ticker: str,
                    b_obs: np.ndarray,
                    p_obs: np.ndarray,
                    EP: float,
                    ROI: float,
                    min_obs: int = 3) -> FirmParams | None:
    """
    Estimate firm-specific a0 and ab by OLS given panel observations.

    Parameters
    ----------
    ticker  : firm identifier (for reporting only)
    b_obs   : array of observed x/b values, shape (T,)
    p_obs   : array of observed P values,   shape (T,)
    EP      : firm earnings yield (time-invariant, from Yahoo Finance)
    ROI     : firm return on investment    (time-invariant, from Yahoo Finance)
    min_obs : minimum observations required; returns None if not met

    Returns
    -------
    FirmParams or None if estimation fails.

    Notes
    -----
    Observations where b == 0, p == 0, or r == 0 are dropped because they
    produce undefined a_required values.
    """
    b_obs = np.asarray(b_obs, dtype=float)
    p_obs = np.asarray(p_obs, dtype=float)

    r_obs = compute_r(b_obs, EP, ROI)

    # Drop invalid observations
    valid = (
        np.isfinite(b_obs) & np.isfinite(p_obs) & np.isfinite(r_obs)
        & (b_obs != 0.0) & (p_obs != 0.0) & (r_obs != 0.0)
    )
    b, p, r = b_obs[valid], p_obs[valid], r_obs[valid]

    if len(b) < min_obs:
        warnings.warn(f"{ticker}: only {len(b)} valid observations (need {min_obs}); skipping")
        return None

    # Required a for each observation
    a_req = _compute_required_a(b, p, EP, r)

    # Design matrix  X = [1, -b]  →  a_req = γ1 - γ2*b + l
    X = np.column_stack([np.ones(len(b)), -b])

    # OLS via least squares
    try:
        coeffs, _, rank, _ = np.linalg.lstsq(X, a_req, rcond=None)
    except np.linalg.LinAlgError as exc:
        warnings.warn(f"{ticker}: OLS failed — {exc}")
        return None

    if rank < 2:
        warnings.warn(f"{ticker}: design matrix rank {rank} < 2; cannot identify both parameters")
        return None

    gamma1, gamma2 = coeffs

    # Recover structural parameters
    a0 = float(gamma1)
    if abs(gamma2) < 1e-10:
        warnings.warn(f"{ticker}: γ2 ≈ 0, ab is not identified")
        ab = np.nan
    else:
        ab = float(gamma1 / gamma2)

    # OLS residuals (= curve model residuals l_t)
    l_t = a_req - (gamma1 - gamma2 * b)

    # R²
    ss_res = float(np.sum(l_t ** 2))
    ss_tot = float(np.sum((a_req - np.mean(a_req)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return FirmParams(
        ticker    = ticker,
        a0        = a0,
        ab        = ab,
        residuals = l_t,
        r_squared = r2,
        n_obs     = len(b),
        b_obs     = b,
        p_obs     = p,
    )


def fit_cross_section(records: list[dict],
                      b_key: str = "b",
                      p_key: str = "p") -> list[FirmParams]:
    """
    Fit a0 and ab for each firm in a cross-sectional panel.

    Parameters
    ----------
    records : list of dicts, one per firm, each containing:
                  ticker, EP, ROI, and arrays under b_key / p_key
    b_key   : key for the b-observation array inside each record
    p_key   : key for the p-observation array inside each record

    Returns
    -------
    List of FirmParams (one per firm, None entries omitted).
    """
    results = []
    for rec in records:
        fp = fit_firm_params(
            ticker = rec["ticker"],
            b_obs  = rec[b_key],
            p_obs  = rec[p_key],
            EP     = rec["EP"],
            ROI    = rec["ROI"],
        )
        if fp is not None:
            results.append(fp)
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yfinance as yf
    from yahoo_curve import fetch_financials

    np.random.seed(42)

    TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ", "XOM"]
    T       = 24   # synthetic monthly observations per firm

    print("=" * 70)
    print("  FIRM-SPECIFIC OLS: estimating a0 and ab per firm")
    print("=" * 70)

    # Fetch real EP and ROI from Yahoo Finance
    print("\nFetching EP / ROI from Yahoo Finance...")
    fin = {r["ticker"]: r for r in [fetch_financials(tk) for tk in TICKERS]}

    # Build synthetic panel: for each firm generate T observations (b_t, p_t)
    # where b_t is drawn from Uniform(0.05, 0.45) and p_t is the theoretical
    # curve value plus Gaussian noise (so we recover a known a0, ab).
    TRUE_A0, TRUE_AB = 0.85, 0.20   # planted truth for all firms in this demo

    records = []
    for tk in TICKERS:
        f = fin[tk]
        if f["EP"] is None or f["ROI"] is None:
            print(f"  {tk}: missing EP or ROI, skipped")
            continue

        b_t = np.random.uniform(0.05, 0.45, size=T)
        p_t = (
            curve_P_financial(b_t, f["EP"], f["ROI"], TRUE_A0, TRUE_AB, l=0.0)
            + np.random.normal(0, 0.05, size=T)   # observation noise → residuals
        )
        # Mask any NaN / infinite values from the curve
        ok  = np.isfinite(p_t) & (p_t > 0)
        records.append({"ticker": tk, "EP": f["EP"], "ROI": f["ROI"],
                         "b": b_t[ok], "p": p_t[ok]})

    # Fit firm-specific parameters
    print(f"\nFitting a0 and ab for {len(records)} firms (planted truth: "
          f"a0={TRUE_A0}, ab={TRUE_AB})...\n")

    results = fit_cross_section(records)

    # Report
    print(f"{'Ticker':<8} {'a0':>8} {'ab':>8} {'R²':>8} {'RMSE':>10} {'N':>5}")
    print("-" * 50)
    for fp in results:
        print(f"{fp.ticker:<8} {fp.a0:>8.4f} {fp.ab:>8.4f} "
              f"{fp.r_squared:>8.4f} {fp.rmse:>10.6f} {fp.n_obs:>5d}")

    print("\nNote: a0 and ab recover the planted values up to observation noise.")
    print("=" * 70)
