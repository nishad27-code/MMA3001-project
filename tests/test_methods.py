"""Verification of the reconstruction methods against known analytic answers."""

import numpy as np
import pytest

from gap_reconstruction.evaluation import make_gap, reconstruct_gap, with_context
from gap_reconstruction.methods import METHODS, cubic_spline, forward_fill, linear, pchip

T_EPOCH = 1.7e9  # realistic Unix-second magnitudes, to exercise conditioning


def test_forward_fill_returns_last_known_value():
    t = np.array([0.0, 600.0, 1800.0])
    y = np.array([400.0, 450.0, 500.0])
    q = np.array([0.0, 599.0, 600.0, 1200.0, 1800.0, 5000.0])
    np.testing.assert_array_equal(forward_fill(t, y, q), [400, 400, 450, 450, 500, 500])


def test_linear_reproduces_straight_line_exactly():
    t = T_EPOCH + np.array([0.0, 600.0, 3600.0, 4200.0])
    y = 2.5e-3 * (t - T_EPOCH) + 410.0
    q = T_EPOCH + np.arange(600.0, 3601.0, 600.0)
    np.testing.assert_allclose(linear(t, y, q), 2.5e-3 * (q - T_EPOCH) + 410.0, rtol=0, atol=1e-9)


def test_cubic_spline_reproduces_cubic_exactly():
    # A not-a-knot spline is exact for any cubic polynomial.
    f = lambda s: 420.0 + 3.0 * s - 0.8 * s**2 + 0.05 * s**3  # s in hours
    s_known = np.array([0, 1, 2, 3, 7, 8, 9, 10], dtype=float)  # gap between 3 h and 7 h
    t = T_EPOCH + 3600.0 * s_known
    q_s = np.linspace(3.0, 7.0, 25)
    out = cubic_spline(t, f(s_known), T_EPOCH + 3600.0 * q_s)
    np.testing.assert_allclose(out, f(q_s), rtol=0, atol=1e-8)


def test_linear_is_not_exact_for_cubic_but_spline_is():
    f = lambda s: s**3
    s_known = np.array([0, 1, 2, 4, 5, 6], dtype=float)
    q = np.array([3.0])
    assert abs(linear(s_known, f(s_known), q)[0] - 27.0) > 1.0
    assert cubic_spline(s_known, f(s_known), q)[0] == pytest.approx(27.0, abs=1e-9)


def test_spline_error_converges_at_fourth_order():
    # Interpolating sin on a refined grid: error should fall by ~2^4 per halving.
    errs = []
    for n in (16, 32, 64):
        x = np.linspace(0, np.pi, n + 1)
        q = (x[:-1] + x[1:]) / 2
        errs.append(np.max(np.abs(cubic_spline(x, np.sin(x), q) - np.sin(q))))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 3.5)


def test_linear_error_converges_at_second_order():
    errs = []
    for n in (16, 32, 64):
        x = np.linspace(0, np.pi, n + 1)
        q = (x[:-1] + x[1:]) / 2
        errs.append(np.max(np.abs(linear(x, np.sin(x), q) - np.sin(q))))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(np.abs(rates - 2.0) < 0.2)


def test_methods_pass_through_known_points():
    t = np.array([0.0, 1.0, 2.5, 4.0, 5.0])
    y = np.array([410.0, 430.0, 425.0, 470.0, 460.0])
    for name, m in METHODS.items():
        np.testing.assert_allclose(m(t, y, t), y, atol=1e-9, err_msg=name)


@pytest.mark.parametrize("method", list(METHODS.values()))
@pytest.mark.parametrize("t, y, q", [
    ([0, 2, 1, 3], [1, 2, 3, 4], [1.5]),             # unsorted
    ([0, 1, 1, 3], [1, 2, 3, 4], [1.5]),             # duplicate time
    ([0, 1, 2, 3], [1, 2, np.nan, 4], [1.5]),        # NaN value
    ([0, 1, 2, 3], [1, 2, 3], [1.5]),                # length mismatch
    ([0, 1, 2, 3], [1, 2, 3, 4], [-1.0]),            # before first reading
    ([[0, 1], [2, 3]], [1, 2, 3, 4], [1.5]),         # not 1-D
])
def test_invalid_inputs_raise(method, t, y, q):
    with pytest.raises(ValueError):
        method(t, y, q)


def test_pchip_reproduces_straight_line_exactly():
    t = T_EPOCH + np.array([0.0, 600.0, 3600.0, 4200.0])
    y = 2.5e-3 * (t - T_EPOCH) + 410.0
    q = T_EPOCH + np.arange(600.0, 3601.0, 600.0)
    np.testing.assert_allclose(pchip(t, y, q), 2.5e-3 * (q - T_EPOCH) + 410.0, rtol=0, atol=1e-9)


def test_pchip_never_leaves_boundary_range_on_noisy_data():
    rng = np.random.default_rng(0)
    spline_overshoots = 0
    for _ in range(300):
        t = 600.0 * np.arange(12)
        y = 450.0 + rng.normal(0.0, 15.0, t.size)
        hide = np.arange(4, 8)                      # gap between readings 3 and 8
        tk, yk = np.delete(t, hide), np.delete(y, hide)
        q = np.linspace(t[3], t[8], 60)
        lo, hi = min(y[3], y[8]), max(y[3], y[8])
        out = pchip(tk, yk, q)
        assert out.min() >= lo - 1e-9 and out.max() <= hi + 1e-9
        s = cubic_spline(tk, yk, q)
        spline_overshoots += bool(s.min() < lo - 1e-9 or s.max() > hi + 1e-9)
    assert spline_overshoots > 0   # the same data does make the spline overshoot


def test_pchip_gap_prediction_same_for_context_2_and_8():
    rng = np.random.default_rng(1)
    t = 1.7e9 + 600.0 * np.arange(400)
    y = 450.0 + np.cumsum(rng.normal(0.0, 5.0, t.size))
    for start, minutes in ((50, 10), (120, 60), (250, 360)):
        g8 = make_gap(t, start, minutes * 60.0, context=8, lo=0, hi=t.size)
        g2 = with_context(g8, 2)
        np.testing.assert_allclose(reconstruct_gap(pchip, t, y, g2), reconstruct_gap(pchip, t, y, g8),
                                   rtol=0, atol=1e-9)


def test_interpolators_refuse_extrapolation():
    t, y = [0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError):
        linear(t, y, [3.5])
    with pytest.raises(ValueError):
        cubic_spline(t, y, [3.5])
    with pytest.raises(ValueError):
        pchip(t, y, [3.5])
    with pytest.raises(ValueError):
        pchip(t, y, [-0.5])
    assert forward_fill(t, y, [3.5])[0] == 4.0  # causal method may run past the end


def test_minimum_point_counts():
    with pytest.raises(ValueError):
        linear([0.0], [1.0], [0.0])
    with pytest.raises(ValueError):
        cubic_spline([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [0.5])
    with pytest.raises(ValueError):
        pchip([0.0], [1.0], [0.0])
