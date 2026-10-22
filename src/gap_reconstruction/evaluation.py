"""Artificial-gap evaluation of reconstruction methods.

Procedure
---------
1. Split the series chronologically: an earlier **development** period
   (used only to choose settings) and a later **test** period (reported once).
2. Within a period, hide a contiguous block of genuine readings. A gap of
   nominal duration ``D`` starting at reading ``s`` hides every reading with
   ``t[s] <= t < t[s] + D - Δ/2``, where Δ is the nominal 600 s cadence.
3. Give each method only the ``context`` known readings immediately before
   and after the block. The hidden values are removed from the input arrays
   and an assertion checks this.
4. Score every method on the *same* hidden readings.

Three different quantities describe a gap
-----------------------------------------
- **Nominal duration D** (the label: 10, 30, 60, 180 or 360 min). This is
  the length of the time window, starting at a genuine reading, from which
  readings are removed.
- **Hidden readings.** The number of genuine readings inside that window.
  Under perfectly regular sampling this would be ``k = D/Δ``. The half-step
  margin keeps the count stable under the ±2 s timestamp jitter. Where a
  native (pre-existing) missing reading falls inside the window, fewer
  readings are hidden. The sensor used here already misses about one
  reading in ten (consecutive dropouts are most often 13 or 3 readings
  apart), so long windows usually hide fewer than ``k``.
- **Boundary interval.** The elapsed time between the known reading just
  before the gap and the known reading just after it, i.e. the hole an
  interpolator must bridge. It is about ``D + Δ``, plus about Δ for each
  native missing reading directly bordering the window. For example, a
  10-minute gap hides one reading and normally leaves known readings
  20 minutes apart, but 30 minutes apart when the neighbouring reading was
  already missing.

`gap_statistics` reports the distribution of the second and third quantities
for any set of gaps. All methods are scored on identical gaps, so these
variations affect every method equally.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .methods import METHODS, Method

NOMINAL_DT_S = 600.0


@dataclass(frozen=True)
class Gap:
    """One artificial gap.

    Attributes
    ----------
    hidden : numpy.ndarray of int
        Indices (into the full series) of the hidden readings, contiguous.
    known : numpy.ndarray of int
        Indices of the context readings given to the methods.
    """

    hidden: np.ndarray
    known: np.ndarray


def chronological_split(n: int, test_fraction: float = 0.3) -> int:
    """Return the index where the test period starts.

    Readings ``[0, split)`` form the development period and ``[split, n)``
    the test period. No reading belongs to both.

    Raises
    ------
    ValueError
        If ``test_fraction`` is not in (0, 1) or either period would be empty.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    split = int(round(n * (1.0 - test_fraction)))
    if split <= 0 or split >= n:
        raise ValueError("both periods must contain readings")
    return split


def make_gap(t: np.ndarray, start: int, duration_s: float, context: int,
             lo: int, hi: int, dt_nominal: float = NOMINAL_DT_S) -> Gap | None:
    """Build the gap that starts at reading ``start``, or return None if it does not fit.

    The gap fits only if the hidden block plus ``context`` readings on each
    side lie entirely inside ``[lo, hi)``, so a test gap never borrows
    readings from the development period and vice versa.

    Parameters
    ----------
    t : numpy.ndarray
        Strictly increasing reading times (s) of the whole series.
    start : int
        Index of the first hidden reading.
    duration_s : float
        Nominal gap duration ``D`` in seconds (> 0).
    context : int
        Number of known readings supplied on each side (>= 2).
    lo, hi : int
        Period bounds as indices.
    dt_nominal : float
        Nominal sampling interval, used for the half-step margin.
    """
    if duration_s <= 0 or context < 2:
        raise ValueError("duration_s must be > 0 and context >= 2")
    end_time = t[start] + duration_s - dt_nominal / 2.0
    stop = int(np.searchsorted(t, end_time, side="left"))  # first index not hidden
    stop = max(stop, start + 1)  # always hide at least the start reading
    left, right = start - context, stop + context
    if left < lo or right > hi:
        return None
    hidden = np.arange(start, stop)
    known = np.concatenate([np.arange(left, start), np.arange(stop, right)])
    return Gap(hidden=hidden, known=known)


def sample_gaps(t: np.ndarray, duration_s: float, context: int, lo: int, hi: int,
                n_max: int, rng: np.random.Generator) -> list[Gap]:
    """Randomly place up to ``n_max`` non-overlapping gaps inside ``[lo, hi)``.

    Start positions are visited in random order and a gap is accepted only
    if its hidden block does not overlap a previously accepted one. This
    keeps the gaps' errors closer to independent for the bootstrap. The
    context of one gap may overlap another gap's hidden block. That is
    harmless, because each gap is reconstructed separately from the full
    genuine data.
    """
    occupied = np.zeros(t.size, dtype=bool)
    gaps: list[Gap] = []
    for s in rng.permutation(np.arange(lo, hi)):
        g = make_gap(t, int(s), duration_s, context, lo, hi)
        if g is None or occupied[g.hidden].any():
            continue
        occupied[g.hidden] = True
        gaps.append(g)
        if len(gaps) >= n_max:
            break
    gaps.sort(key=lambda g: int(g.hidden[0]))
    return gaps


def with_context(gap: Gap, context: int) -> Gap:
    """Return the same hidden block with ``context`` known readings on each side.

    Used to compare context sizes on identical hidden readings. The caller
    must ensure the wider context still lies inside the period. This holds
    automatically for gaps drawn with `common_gaps`.

    Raises
    ------
    ValueError
        If ``context < 2``.
    """
    if context < 2:
        raise ValueError("context must be >= 2")
    first, last = int(gap.hidden[0]), int(gap.hidden[-1])
    known = np.concatenate([np.arange(first - context, first), np.arange(last + 1, last + 1 + context)])
    return Gap(hidden=gap.hidden, known=known)


def common_gaps(t: np.ndarray, duration_s: float, contexts, lo: int, hi: int,
                n_max: int, rng: np.random.Generator) -> dict[int, list[Gap]]:
    """Draw one set of gaps valid for every context size, for a fair comparison.

    Gaps are placed with `sample_gaps` using the *largest* context, so every
    smaller context also fits inside ``[lo, hi)``. Each context then receives
    the same hidden blocks; only the known readings supplied differ.

    Returns
    -------
    dict
        ``{context: [Gap, ...]}`` with identical ``hidden`` arrays, in the
        same order, for every context.
    """
    contexts = sorted(set(int(c) for c in contexts))
    base = sample_gaps(t, duration_s, contexts[-1], lo, hi, n_max, rng)
    return {c: [with_context(g, c) for g in base] for c in contexts}


def gap_statistics(t: np.ndarray, gaps: list[Gap], duration_s: float,
                   dt_nominal: float = NOMINAL_DT_S, gap_factor: float = 1.5) -> dict:
    """Describe what a set of gaps actually removed.

    Parameters
    ----------
    t : numpy.ndarray
        Reading times (s) of the whole series.
    gaps : list of Gap
        Gaps to describe (non-empty).
    duration_s : float
        Their nominal duration ``D``.
    dt_nominal : float
        Nominal sampling interval Δ.
    gap_factor : float
        An interval longer than ``gap_factor * dt_nominal`` counts as a native
        missing reading.

    Returns
    -------
    dict
        Nominal ``k = D/Δ``; the distribution of hidden-reading counts; the
        fraction of gaps hiding fewer than ``k``; boundary-interval statistics
        (minutes) and their distribution rounded to the nearest 10 min; and
        the fractions of gaps with a native missing reading immediately
        before or after the hidden block.
    """
    if not gaps:
        raise ValueError("no gaps to describe")
    k = int(round(duration_s / dt_nominal))
    hidden = np.array([g.hidden.size for g in gaps])
    first = np.array([g.hidden[0] for g in gaps])
    last = np.array([g.hidden[-1] for g in gaps])
    boundary_min = (t[last + 1] - t[first - 1]) / 60.0
    left_native = (t[first] - t[first - 1]) > gap_factor * dt_nominal
    right_native = (t[last + 1] - t[last]) > gap_factor * dt_nominal
    fmt = lambda c: ";".join(f"{key}:{c[key]}" for key in sorted(c))  # noqa: E731
    return {
        "nominal_duration_min": duration_s / 60.0,
        "nominal_k": k,
        "n_gaps": len(gaps),
        "hidden_min": int(hidden.min()),
        "hidden_mean": float(hidden.mean()),
        "hidden_max": int(hidden.max()),
        "hidden_count_distribution": fmt(Counter(hidden.tolist())),
        "frac_fewer_than_nominal_k": float(np.mean(hidden < k)),
        "boundary_min_min": float(boundary_min.min()),
        "boundary_mean_min": float(boundary_min.mean()),
        "boundary_max_min": float(boundary_min.max()),
        "boundary_distribution_10min": fmt(Counter((np.round(boundary_min / 10.0) * 10).astype(int).tolist())),
        "frac_native_missing_before": float(left_native.mean()),
        "frac_native_missing_after": float(right_native.mean()),
    }


def gap_set_hash(gaps: list[Gap]) -> str:
    """Short fingerprint of the hidden blocks, to show two runs used the same gaps."""
    h = hashlib.sha1()
    for g in gaps:
        h.update(np.asarray(g.hidden, dtype=np.int64).tobytes())
        h.update(b"|")
    return h.hexdigest()[:12]


def reconstruct_gap(method: Method, t: np.ndarray, y: np.ndarray, gap: Gap) -> np.ndarray:
    """Reconstruct one gap using only its known context readings.

    The method receives copies of ``t[gap.known]`` and ``y[gap.known]``,
    which contain no hidden reading. This is asserted rather than assumed.
    """
    assert not np.intersect1d(gap.known, gap.hidden).size, "hidden reading leaked into method input"
    return method(t[gap.known].copy(), y[gap.known].copy(), t[gap.hidden].copy())


def mae(err: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(err)))


def rmse(err: np.ndarray) -> float:
    """Root-mean-square error."""
    return float(np.sqrt(np.mean(np.square(err))))


def bootstrap_ci(per_gap_errors: list[np.ndarray], stat, n_boot: int = 1000,
                 rng: np.random.Generator | None = None, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of ``stat`` (MAE or RMSE), resampling whole gaps.

    Whole gaps are resampled, not individual hidden readings, because the
    readings within one gap are strongly correlated.
    """
    rng = rng or np.random.default_rng(0)
    m = len(per_gap_errors)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, m, size=m)
        vals[b] = stat(np.concatenate([per_gap_errors[i] for i in pick]))
    a = (1.0 - level) / 2.0
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1.0 - a))


def day_blocks(start_times_s: np.ndarray, utc_offset_hours: float = 10.0, block_days: int = 1) -> np.ndarray:
    """Assign each gap to a block of ``block_days`` local calendar days.

    Parameters
    ----------
    start_times_s : numpy.ndarray
        Unix time (s, UTC) of each gap's first hidden reading.
    utc_offset_hours : float
        Local offset from UTC. Melbourne is UTC+10 (AEST) throughout the
        14 May – 13 Jun 2024 test period; daylight saving ended on 7 Apr 2024.
    block_days : int
        Block length in days (>= 1).

    Returns
    -------
    numpy.ndarray of int
        Block index per gap. Gaps starting in the same block share a label.
        Blocks are fixed consecutive windows of ``block_days`` local days
        counted from 1 Jan 1970 (local time), so with ``block_days > 1`` the
        first and last blocks of a period may be partial. A gap is assigned
        by its start, even if it runs past midnight.
    """
    if block_days < 1:
        raise ValueError("block_days must be >= 1")
    local = np.asarray(start_times_s, dtype=float) + utc_offset_hours * 3600.0
    return np.floor(local / (86400.0 * block_days)).astype(np.int64)


def paired_block_bootstrap(abs_err_sum: dict[str, np.ndarray], n_hidden: np.ndarray, blocks: np.ndarray,
                           reference: str, n_boot: int = 2000, rng: np.random.Generator | None = None,
                           levels: tuple[float, ...] = (0.95,)) -> dict:
    """Paired block bootstrap of MAE differences between methods on the same gaps.

    Each replicate draws the blocks (e.g. days) with replacement, as many as
    there are blocks, and keeps every gap in each drawn block. All methods
    are evaluated on that same resampled set of gaps, so the comparison stays
    paired. The statistic is the MAE pooled over hidden readings,
    ``sum |error| / number of hidden readings``, the same definition as the
    point estimates. Resampling whole blocks keeps the dependence between
    gaps that share a block inside each replicate.

    Parameters
    ----------
    abs_err_sum : dict of str -> numpy.ndarray
        Per method, the sum of absolute errors of each gap (same gap order
        for every method).
    n_hidden : numpy.ndarray
        Number of hidden readings in each gap (> 0).
    blocks : numpy.ndarray
        Block label of each gap, e.g. from `day_blocks`.
    reference : str
        Method that differences are taken against: ``MAE[other] - MAE[reference]``.
        A positive difference means the reference has the lower error.
    n_boot : int
        Number of bootstrap replicates.
    rng : numpy.random.Generator, optional
        Random generator (seed it for reproducibility).
    levels : tuple of float
        Confidence levels for percentile intervals.

    Returns
    -------
    dict
        ``n_gaps``, ``n_blocks``; per method the pooled MAE and its block-
        bootstrap intervals; per non-reference method the point difference,
        its intervals at each level, and ``frac_boot_diff_le_0`` (the share of
        replicates in which the other method was at least as good).

    Raises
    ------
    ValueError
        On empty input, mismatched lengths, unknown reference or non-positive
        hidden counts.
    """
    n_hidden = np.asarray(n_hidden, dtype=float)
    blocks = np.asarray(blocks)
    m = n_hidden.size
    if m == 0:
        raise ValueError("no gaps")
    if blocks.shape != (m,) or any(np.asarray(v).shape != (m,) for v in abs_err_sum.values()):
        raise ValueError("abs_err_sum, n_hidden and blocks must all have one entry per gap")
    if reference not in abs_err_sum:
        raise ValueError(f"unknown reference method {reference!r}")
    if np.any(n_hidden <= 0):
        raise ValueError("every gap must hide at least one reading")
    rng = rng or np.random.default_rng(0)

    labels, inv = np.unique(blocks, return_inverse=True)
    nb = labels.size
    block_n = np.bincount(inv, weights=n_hidden, minlength=nb)
    block_s = {k: np.bincount(inv, weights=np.asarray(v, dtype=float), minlength=nb) for k, v in abs_err_sum.items()}
    # weights[b, j] = how many times block j is drawn in replicate b
    draws = rng.integers(0, nb, size=(n_boot, nb))
    weights = np.zeros((n_boot, nb))
    np.add.at(weights, (np.repeat(np.arange(n_boot), nb), draws.ravel()), 1.0)
    denom = weights @ block_n
    boot_mae = {k: (weights @ s) / denom for k, s in block_s.items()}
    point_mae = {k: float(s.sum() / block_n.sum()) for k, s in block_s.items()}

    def ci(x):
        return {str(lv): (float(np.quantile(x, (1 - lv) / 2)), float(np.quantile(x, 1 - (1 - lv) / 2))) for lv in levels}

    out = {"n_gaps": int(m), "n_blocks": int(nb), "n_boot": int(n_boot), "reference": reference,
           "methods": {k: {"mae": point_mae[k], "mae_ci": ci(boot_mae[k])} for k in abs_err_sum},
           "differences": {}}
    for k in abs_err_sum:
        if k == reference:
            continue
        d = boot_mae[k] - boot_mae[reference]
        out["differences"][k] = {"point": point_mae[k] - point_mae[reference], "ci": ci(d),
                                 "frac_boot_diff_le_0": float(np.mean(d <= 0))}
    return out


def evaluate(t: np.ndarray, y: np.ndarray, gaps: list[Gap],
             methods: dict[str, Method] = METHODS, timing_repeats: int = 5,
             n_boot: int = 1000, seed: int = 0) -> dict:
    """Score every method on the same set of gaps.

    Returns
    -------
    dict
        For each method: pooled MAE/RMSE (ppm) with bootstrap 95 % CIs, the
        mean absolute error at the gap midpoint, the maximum overshoot
        beyond the range of the context readings, and the median runtime per
        gap in microseconds, and the list of per-gap MAEs (same gap order
        for every method, for paired comparisons). Also shared gap descriptors: number of gaps,
        mean hidden readings per gap, and the mean and maximum boundary
        interval (time between the known readings either side of the gap).
    """
    if not gaps:
        raise ValueError("no gaps to evaluate")
    boundary = np.array([t[g.hidden[-1] + 1] - t[g.hidden[0] - 1] for g in gaps], dtype=float)
    out = {
        "n_gaps": len(gaps),
        "mean_hidden_readings": float(np.mean([g.hidden.size for g in gaps])),
        "min_hidden_readings": int(min(g.hidden.size for g in gaps)),
        "max_hidden_readings": int(max(g.hidden.size for g in gaps)),
        "mean_boundary_interval_min": float(boundary.mean() / 60.0),
        "max_boundary_interval_min": float(boundary.max() / 60.0),
        "methods": {},
    }
    for name, method in methods.items():
        per_gap, mid_err, overshoot = [], [], []
        for g in gaps:
            y_hat = reconstruct_gap(method, t, y, g)
            e = y_hat - y[g.hidden]
            per_gap.append(e)
            mid_err.append(abs(e[e.size // 2]))
            ctx = y[g.known]
            overshoot.append(max(0.0, float(y_hat.max() - ctx.max()), float(ctx.min() - y_hat.min())))
        # Timing: only the method call, on inputs sliced in advance (the leak
        # check and slicing are harness overhead shared by all methods).
        # Repeated; the median per-gap time is reported.
        inputs = [(t[g.known], y[g.known], t[g.hidden]) for g in gaps]
        runs = []
        for _ in range(timing_repeats):
            t0 = time.perf_counter()
            for tk, yk, tq in inputs:
                method(tk, yk, tq)
            runs.append((time.perf_counter() - t0) / len(gaps))
        rng = np.random.default_rng(seed)
        allerr = np.concatenate(per_gap)
        out["methods"][name] = {
            "bias": float(np.mean(allerr)),  # mean signed error: + means overestimate
            "mae": mae(allerr),
            "mae_ci95": bootstrap_ci(per_gap, mae, n_boot, rng),
            "rmse": rmse(allerr),
            "rmse_ci95": bootstrap_ci(per_gap, rmse, n_boot, rng),
            "midpoint_mae": float(np.mean(mid_err)),
            "max_overshoot": float(max(overshoot)),
            "us_per_gap": float(np.median(runs) * 1e6),
            "per_gap_mae": [float(np.mean(np.abs(e))) for e in per_gap],
        }
    return out
