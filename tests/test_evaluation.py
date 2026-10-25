"""Tests for the artificial-gap evaluation: leakage, placement rules and metrics."""

import numpy as np
import pytest

from gap_reconstruction import evaluation as ev
from gap_reconstruction.evaluation import (
    Gap, bootstrap_ci, chronological_split, evaluate, make_gap, mae, reconstruct_gap, rmse, sample_gaps,
)
from gap_reconstruction.methods import linear


def regular(n, dt=600.0, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return 1.7e9 + dt * np.arange(n) + rng.uniform(-jitter, jitter, n)


def test_chronological_split_is_disjoint_and_ordered():
    split = chronological_split(100, 0.3)
    assert split == 70
    with pytest.raises(ValueError):
        chronological_split(100, 0.0)
    with pytest.raises(ValueError):
        chronological_split(1, 0.3)


@pytest.mark.parametrize("minutes, k", [(10, 1), (30, 3), (60, 6), (180, 18), (360, 36)])
def test_gap_hides_k_readings_under_jitter(minutes, k):
    t = regular(200, jitter=2.0)
    g = make_gap(t, 50, minutes * 60.0, context=4, lo=0, hi=200)
    assert g.hidden.size == k
    assert g.hidden.tolist() == list(range(50, 50 + k))
    # boundary interval is (k + 1) * dt, not k * dt
    assert t[g.hidden[-1] + 1] - t[g.hidden[0] - 1] == pytest.approx((k + 1) * 600.0, abs=5)


def test_gap_with_native_missing_reading_hides_fewer():
    t = np.delete(regular(50), 22)  # reading at original index 22 never existed
    g = make_gap(t, 20, 3600.0, context=3, lo=0, hi=t.size)
    assert g.hidden.size == 5  # one of the six nominal slots was already empty


def test_gap_must_fit_inside_period():
    t = regular(100)
    assert make_gap(t, 2, 600.0, context=4, lo=0, hi=100) is None    # left context too short
    assert make_gap(t, 95, 3600.0, context=4, lo=0, hi=100) is None  # runs past period end
    assert make_gap(t, 70, 600.0, context=4, lo=70, hi=100) is None  # context would cross split


def test_context_excludes_hidden_and_has_requested_size():
    t = regular(100)
    g = make_gap(t, 40, 1800.0, context=5, lo=0, hi=100)
    assert not np.intersect1d(g.known, g.hidden).size
    assert g.known.size == 10
    assert g.known.max() < 100 and g.known.min() >= 0


def test_reconstruct_gap_never_sees_hidden_values():
    t = regular(60)
    y = np.full(60, 400.0)
    g = make_gap(t, 30, 1800.0, context=4, lo=0, hi=60)
    y[g.hidden] = 1e6  # poison: any leak would show up in the output
    seen = {}

    def spy(tk, yk, tq):
        seen["t"], seen["y"] = tk, yk
        return linear(tk, yk, tq)

    out = reconstruct_gap(spy, t, y, g)
    assert np.all(seen["y"] < 1e6)
    assert not np.isin(seen["t"], t[g.hidden]).any()
    np.testing.assert_allclose(out, 400.0)


def test_leak_assertion_fires():
    t, y = regular(10), np.zeros(10)
    bad = Gap(hidden=np.array([4, 5]), known=np.array([2, 3, 4, 6, 7]))
    with pytest.raises(AssertionError):
        reconstruct_gap(linear, t, y, bad)


def test_sample_gaps_non_overlapping_within_bounds_and_reproducible():
    t = regular(500)
    a = sample_gaps(t, 3600.0, 4, 100, 400, 50, np.random.default_rng(1))
    b = sample_gaps(t, 3600.0, 4, 100, 400, 50, np.random.default_rng(1))
    assert [g.hidden.tolist() for g in a] == [g.hidden.tolist() for g in b]
    hidden = np.concatenate([g.hidden for g in a])
    assert hidden.size == np.unique(hidden).size
    assert min(g.known.min() for g in a) >= 100 and max(g.known.max() for g in a) < 400


def test_metrics_match_hand_calculation():
    e = np.array([3.0, -4.0, 0.0, 1.0])
    assert mae(e) == pytest.approx(2.0)
    assert rmse(e) == pytest.approx(np.sqrt(26 / 4))


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    errs = [rng.normal(0, 5, 6) for _ in range(40)]
    lo, hi = bootstrap_ci(errs, mae, 500, np.random.default_rng(1))
    assert lo < mae(np.concatenate(errs)) < hi


def test_evaluate_same_gaps_for_all_methods_and_exact_on_line():
    t = regular(300)
    y = 400.0 + 0.01 * (t - t[0])  # straight line: linear and spline are exact
    gaps = sample_gaps(t, 3600.0, 6, 0, 300, 20, np.random.default_rng(0))
    res = evaluate(t, y, gaps, timing_repeats=1, n_boot=50)
    assert res["n_gaps"] == len(gaps)
    assert res["methods"]["linear"]["mae"] < 1e-8
    assert res["methods"]["cubic_spline"]["mae"] < 1e-6
    # forward-fill lags a rising line by on average 3.5 steps of 6 ppm
    assert res["methods"]["forward_fill"]["mae"] == pytest.approx(0.01 * 600 * 3.5, rel=1e-6)
    assert res["mean_boundary_interval_min"] == pytest.approx(70.0)


def test_evaluate_rejects_empty():
    with pytest.raises(ValueError):
        evaluate(regular(10), np.zeros(10), [])


def test_nominal_dt_constant():
    assert ev.NOMINAL_DT_S == 600.0


# --- Fix 1: gap-definition statistics on irregular sampling -------------------

def test_gap_statistics_regular_series():
    t = regular(300, jitter=2.0)
    gaps = sample_gaps(t, 3600.0, 4, 0, 300, 20, np.random.default_rng(0))
    s = ev.gap_statistics(t, gaps, 3600.0)
    assert s["nominal_k"] == 6
    assert s["hidden_count_distribution"] == f"6:{len(gaps)}"
    assert s["frac_fewer_than_nominal_k"] == 0.0
    assert s["boundary_distribution_10min"] == f"70:{len(gaps)}"  # (k + 1) * 10 min


def test_gap_statistics_native_missing_reading_beside_10min_gap():
    t = np.delete(regular(40), 19)  # native missing reading just before index 19 of the new series
    g = make_gap(t, 19, 600.0, context=2, lo=0, hi=t.size)
    s = ev.gap_statistics(t, [g], 600.0)
    assert s["hidden_count_distribution"] == "1:1"
    assert s["boundary_max_min"] == pytest.approx(30.0)  # 20 min + the pre-existing 10-min hole
    assert s["frac_native_missing_before"] == 1.0 and s["frac_native_missing_after"] == 0.0


def test_gap_statistics_native_missing_reading_inside_window():
    t = np.delete(regular(60), 33)
    g = make_gap(t, 30, 3600.0, context=2, lo=0, hi=t.size)
    s = ev.gap_statistics(t, [g], 3600.0)
    assert s["hidden_count_distribution"] == "5:1"   # fewer than nominal k = 6
    assert s["frac_fewer_than_nominal_k"] == 1.0
    assert s["boundary_max_min"] == pytest.approx(70.0)  # elapsed time is unchanged


# --- Fix 2: every context evaluated on identical gaps -----------------------------

def test_common_gaps_identical_hidden_blocks_for_every_context():
    t = regular(1000)
    sets = ev.common_gaps(t, 3600.0, (2, 4, 8, 16, 32), 0, 700, 30, np.random.default_rng(3))
    ref = [g.hidden.tolist() for g in sets[32]]
    assert len(ref) == 30
    for c, gaps in sets.items():
        assert [g.hidden.tolist() for g in gaps] == ref
        assert ev.gap_set_hash(gaps) == ev.gap_set_hash(sets[32])
        for g in gaps:
            assert g.known.size == 2 * c
            assert g.known.min() >= 0 and g.known.max() < 700          # never leaves the period
            assert not np.intersect1d(g.known, g.hidden).size


def test_with_context_rejects_too_small_context():
    with pytest.raises(ValueError):
        ev.with_context(Gap(np.array([5]), np.array([3, 4, 6, 7])), 1)


def test_tune_context_uses_same_gaps_and_dev_period_only():
    from gap_reconstruction.experiment import tune_context

    t = regular(1500)
    y = 450.0 + 20.0 * np.sin((t - t[0]) / 7200.0)
    split = 1000
    # Poison the test period: a NaN reaching a method's inputs raises ValueError, and a
    # NaN among the hidden values would make the MAE non-finite. Either would fail this test.
    y[split:] = np.nan
    rows, spline_mean, context, dev_sets = tune_context(t, y, split, contexts=(2, 4, 8),
                                                        durations=(10, 60), n_max=15)
    for d in (10, 60):
        hashes = {r["gap_set_hash"] for r in rows if r["duration_min"] == d}
        assert len(hashes) == 1
        assert max(int(g.known.max()) for g in dev_sets[d]) < split
    assert all(np.isfinite(r["mae"]) for r in rows)
    # forward-fill and linear ignore extra context, so their scores are identical across contexts
    for m in ("forward_fill", "linear"):
        assert len({round(r["mae"], 12) for r in rows if r["method"] == m and r["duration_min"] == 60}) == 1
    assert context == min(spline_mean, key=lambda c: (spline_mean[c], c))
