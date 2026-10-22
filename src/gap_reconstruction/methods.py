"""Gap-reconstruction methods.

Every method has the same signature::

    reconstruct(t_known, y_known, t_query) -> y_hat

where ``t_known`` holds the strictly increasing times of the readings that
were *not* hidden, ``y_known`` their values, and ``t_query`` the times of the
hidden readings to estimate. All times are in seconds.

Information used by each method
-------------------------------
- `forward_fill`: **causal**. Uses only the last reading before each query
  time, so it could run in real time. This is the baseline.
- `linear`: **retrospective**. Interpolates between the readings either side
  of the gap, so it needs the first reading *after* the gap.
- `cubic_spline`: **retrospective**. Fits a C2-continuous piecewise cubic
  through all supplied known readings (before and after the gap).
- `pchip`: **retrospective**. Shape-preserving piecewise cubic (C1) through
  the known readings; between two consecutive readings it never leaves
  their range, so it cannot overshoot.

This project only reconstructs gaps after the fact, from a log, so the
retrospective methods are legitimate here. They would not be usable for
real-time prediction.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

Method = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _validate(t_known, y_known, t_query, min_points: int, need_right: bool):
    """Convert inputs to float arrays and check the method's preconditions.

    Raises
    ------
    ValueError
        If arrays are not 1-D, have different lengths, contain non-finite
        values, ``t_known`` is not strictly increasing, there are fewer than
        ``min_points`` known readings, or a query lies outside the supported
        range: before the first known reading, or (when ``need_right``) after
        the last one. Extrapolation is refused on purpose.
    """
    t_known = np.asarray(t_known, dtype=float)
    y_known = np.asarray(y_known, dtype=float)
    t_query = np.asarray(t_query, dtype=float)
    if t_known.ndim != 1 or y_known.ndim != 1 or t_query.ndim != 1:
        raise ValueError("t_known, y_known and t_query must be 1-D arrays")
    if t_known.size != y_known.size:
        raise ValueError("t_known and y_known must have the same length")
    if t_known.size < min_points:
        raise ValueError(f"need at least {min_points} known readings, got {t_known.size}")
    if not (np.all(np.isfinite(t_known)) and np.all(np.isfinite(y_known)) and np.all(np.isfinite(t_query))):
        raise ValueError("inputs must be finite (no NaN or inf)")
    if np.any(np.diff(t_known) <= 0):
        raise ValueError("t_known must be strictly increasing (no duplicates)")
    if t_query.size and t_query.min() < t_known[0]:
        raise ValueError("query time before the first known reading")
    if need_right and t_query.size and t_query.max() > t_known[-1]:
        raise ValueError("query time after the last known reading (extrapolation not supported)")
    return t_known, y_known, t_query


def forward_fill(t_known, y_known, t_query) -> np.ndarray:
    """Baseline: carry the last known reading forward (zero-order hold).

    Parameters
    ----------
    t_known, y_known : array_like
        Known reading times (s, strictly increasing) and values.
    t_query : array_like
        Times to estimate; each must be >= ``t_known[0]``.

    Returns
    -------
    numpy.ndarray
        For each query, the value of the latest known reading at or before it.
    """
    t_known, y_known, t_query = _validate(t_known, y_known, t_query, 1, need_right=False)
    idx = np.searchsorted(t_known, t_query, side="right") - 1
    return y_known[idx]


def linear(t_known, y_known, t_query) -> np.ndarray:
    """Piecewise-linear interpolation between neighbouring known readings.

    Inside a gap this is the straight line joining the last reading before
    the gap and the first reading after it. Error is O(h^2 * max|y''|) for
    a smooth signal sampled at spacing h.

    Parameters
    ----------
    t_known, y_known : array_like
        Known reading times (s, strictly increasing) and values; at least 2.
    t_query : array_like
        Times to estimate, within ``[t_known[0], t_known[-1]]``.

    Returns
    -------
    numpy.ndarray
        Interpolated values.
    """
    t_known, y_known, t_query = _validate(t_known, y_known, t_query, 2, need_right=True)
    return np.interp(t_query, t_known, y_known)


def cubic_spline(t_known, y_known, t_query) -> np.ndarray:
    """Cubic-spline interpolation with not-a-knot end conditions.

    The spline is twice continuously differentiable and passes exactly
    through every known reading. Fitting needs one tridiagonal linear solve,
    which is O(n) for n known readings. For a smooth signal the error is
    O(h^4 * max|y''''|). The spline can overshoot near sharp changes and
    across a wide gap, because it has no shape constraint. Values are not
    clipped, so overshoot appears in the error metrics.

    Times are shifted to start at zero before fitting. Raw Unix seconds
    (about 1.7e9) are used only through differences, but shifting keeps the
    polynomial coefficients well scaled.

    Parameters
    ----------
    t_known, y_known : array_like
        Known reading times (s, strictly increasing) and values; at least 4.
    t_query : array_like
        Times to estimate, within ``[t_known[0], t_known[-1]]``.

    Returns
    -------
    numpy.ndarray
        Spline values at ``t_query``.
    """
    t_known, y_known, t_query = _validate(t_known, y_known, t_query, 4, need_right=True)
    t0 = t_known[0]
    spline = CubicSpline(t_known - t0, y_known, bc_type="not-a-knot")
    return spline(t_query - t0)


def pchip(t_known, y_known, t_query) -> np.ndarray:
    """Shape-preserving piecewise cubic Hermite interpolation (PCHIP).

    Like `cubic_spline` this passes through every known reading with a
    piecewise cubic, but the slope at each reading is chosen to avoid
    overshoot (Fritsch-Carlson, via `scipy.interpolate.PchipInterpolator`).
    The slope is zero where a reading is a local maximum or minimum of its
    neighbours, and otherwise a weighted harmonic mean of the two
    neighbouring secant slopes. The curve is therefore monotone between any
    two consecutive known readings and never leaves their range: inside a
    gap it stays between the two boundary readings.

    The interpolant is C1 (continuous slope) but not C2, and on smooth data
    it is generally less accurate than a not-a-knot spline. Each slope depends
    only on the neighbouring readings, so the estimate inside a gap depends
    only on the two known readings on each side; more context has no effect.

    Times are shifted to start at zero before fitting, as in `cubic_spline`.

    Parameters
    ----------
    t_known, y_known : array_like
        Known reading times (s, strictly increasing) and values; at least 2.
    t_query : array_like
        Times to estimate, within ``[t_known[0], t_known[-1]]``.

    Returns
    -------
    numpy.ndarray
        PCHIP values at ``t_query``.
    """
    t_known, y_known, t_query = _validate(t_known, y_known, t_query, 2, need_right=True)
    t0 = t_known[0]
    return PchipInterpolator(t_known - t0, y_known, extrapolate=False)(t_query - t0)


#: Registry used by the experiment; the order is the reporting order.
METHODS: dict[str, Method] = {
    "forward_fill": forward_fill,
    "linear": linear,
    "cubic_spline": cubic_spline,
    "pchip": pchip,
}
