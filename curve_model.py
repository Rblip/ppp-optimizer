"""
Empirical curve model with residual estimation.

Curve:
    P(x) = E * (1 - x) / (r - x * (r + a(x)))

Adaptive parameter:
    a(x) = a0 * (1 - x / ab) - l

Residual l
    Given an empirical point (b, p) that may not lie on the theoretical curve
    (l = 0), compute l so that P(b) = p exactly.

    Derivation:
        p = E*(1-b) / (r - b*(r + a))
        => a = (p*r - E*(1-b)) / (p*b) - r
        => l = a0*(1 - b/ab) - a_required
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core curve
# ---------------------------------------------------------------------------

def compute_a(x: float | np.ndarray,
              a0: float,
              ab: float,
              l: float = 0.0) -> float | np.ndarray:
    """Adaptive parameter a(x) = a0*(1 - x/ab) - l."""
    return a0 * (1.0 - x / ab) - l


def curve_P(x: float | np.ndarray,
            E: float,
            r: float,
            a0: float,
            ab: float,
            l: float = 0.0) -> float | np.ndarray:
    """Evaluate P(x) = E*(1-x) / (r - x*(r + a(x)))."""
    x = np.asarray(x, dtype=float)
    a = compute_a(x, a0, ab, l)
    denom = r - x * (r + a)
    return np.where(np.abs(denom) < np.finfo(float).eps, np.nan, E * (1.0 - x) / denom)


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------

def compute_residual(b: float,
                     p: float,
                     E: float,
                     r: float,
                     a0: float,
                     ab: float) -> float:
    """
    Return l such that curve_P(b, E, r, a0, ab, l) == p.

    Raises ValueError when b == 0 or p == 0 (no finite solution).
    """
    if b == 0.0 or p == 0.0:
        raise ValueError("b and p must both be non-zero")
    a_required = (p * r - E * (1.0 - b)) / (p * b) - r
    return a0 * (1.0 - b / ab) - a_required


def b_star_from_l(l: float | np.ndarray,
                  a0: float,
                  ab: float) -> float | np.ndarray:
    """
    Closed-form value-maximising plowback under management's PERCEIVED
    curve a(b) = a0*(1 - b/ab) - l (i.e. solving d/db P(b) = 0):

        b*(l) = 1 - sqrt(1 - ab * (1 - l/a0))

    At l = 0 this is the curve's true optimum b*.
    """
    return 1.0 - np.sqrt(1.0 - ab * (1.0 - l / a0))


def l_from_b(b: float | np.ndarray,
             a0: float,
             ab: float) -> float | np.ndarray:
    """
    Inverse of b_star_from_l: read the sentiment residual l directly off an
    OBSERVED plowback b, by treating that choice as revealed-optimal under
    management's own perceived curve:

        l(b) = a0 * (1 - b*(2 - b) / ab)

    l > 0  ->  b sits below the true optimum b* — management perceives a
               disadvantage in retaining (and so retains less than the
               curve alone would reward).
    l < 0  ->  b sits above b* — management perceives an advantage in
               retaining (and so retains more).
    """
    b = np.asarray(b, dtype=float)
    return a0 * (1.0 - b * (2.0 - b) / ab)


def is_on_curve(b: float,
                p: float,
                E: float,
                r: float,
                a0: float,
                ab: float,
                tol: float = 1e-6) -> dict:
    """
    Check whether the empirical point (b, p) lies on the curve.

    Returns a dict with:
        on_curve        – bool
        residual        – float, the computed l
        P_theoretical   – float, P(b) with l=0
        P_empirical     – float, the supplied p
    """
    l = compute_residual(b, p, E, r, a0, ab)
    return {
        "on_curve":      abs(l) < tol,
        "residual":      l,
        "P_theoretical": float(curve_P(b, E, r, a0, ab, l=0.0)),
        "P_empirical":   p,
    }
