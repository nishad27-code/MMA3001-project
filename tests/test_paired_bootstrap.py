"""Tests for the paired day-block bootstrap (evaluation.day_blocks, evaluation.paired_block_bootstrap)."""

import numpy as np
import pytest

from gap_reconstruction.evaluation import day_blocks, paired_block_bootstrap

DAY = 86400.0
MIDNIGHT_AEST = 1715608800.0  # 2024-05-14 00:00 AEST = 2024-05-13 14:00 UTC


def synthetic(n_days=30, per_day=6, seed=0, shift=0.0):
    """Errors with a strong shared day effect, so dependence within a day matters."""
    rng = np.random.default_rng(seed)
    blocks = np.repeat(np.arange(n_days), per_day)
    day_effect = rng.gamma(2.0, 3.0, n_days)[blocks]
    n_hidden = rng.integers(1, 7, blocks.size).astype(float)
    base = (day_effect + rng.gamma(2.0, 1.0, blocks.size)) * n_hidden   # per-gap sums of |error|
    other = base + (shift + rng.normal(0, 0.5, blocks.size)) * n_hidden
    return {"linear": base, "other": np.abs(other)}, n_hidden, blocks


# --- day_blocks -----------------------------------------------------------------

def test_day_blocks_use_local_midnight():
    t = MIDNIGHT_AEST + np.array([-1.0, 0.0, 10 * 3600.0, DAY - 1.0, DAY])
    b = day_blocks(t, utc_offset_hours=10.0)
    assert b[0] == b[1] - 1          # one second before local midnight is the previous day
    assert b[1] == b[2] == b[3]      # same local day, even though UTC date changes at 10:00 local
    assert b[4] == b[1] + 1


def test_day_blocks_multi_day_and_validation():
    t = MIDNIGHT_AEST + DAY * np.arange(9)
    b = day_blocks(t, block_days=3)
    assert len(np.unique(b)) in (3, 4)
    assert all(np.diff(b) >= 0)
    with pytest.raises(ValueError):
        day_blocks(t, block_days=0)


# --- paired_block_bootstrap: correctness -----------------------------------------

def test_point_estimates_are_pooled_mae():
    sums, n, blocks = synthetic()
    res = paired_block_bootstrap(sums, n, blocks, "linear", n_boot=200, rng=np.random.default_rng(1))
    for m in sums:
        assert res["methods"][m]["mae"] == pytest.approx(sums[m].sum() / n.sum())
    assert res["differences"]["other"]["point"] == pytest.approx(
        res["methods"]["other"]["mae"] - res["methods"]["linear"]["mae"])
    assert res["n_blocks"] == 30 and res["n_gaps"] == 180


def test_identical_methods_give_zero_difference_in_every_replicate():
    sums, n, blocks = synthetic()
    res = paired_block_bootstrap({"linear": sums["linear"], "copy": sums["linear"].copy()}, n, blocks,
                                 "linear", n_boot=500, rng=np.random.default_rng(2))
    assert res["differences"]["copy"]["ci"]["0.95"] == (0.0, 0.0)
    assert res["differences"]["copy"]["frac_boot_diff_le_0"] == 1.0


def test_pairing_constant_per_reading_offset_is_exact():
    # other = linear + 2.5 per hidden reading in every gap. A paired bootstrap must give exactly 2.5
    # in every replicate. An unpaired one (methods resampled separately) would not.
    sums, n, blocks = synthetic()
    res = paired_block_bootstrap({"linear": sums["linear"], "other": sums["linear"] + 2.5 * n}, n, blocks,
                                 "linear", n_boot=500, rng=np.random.default_rng(3))
    lo, hi = res["differences"]["other"]["ci"]["0.95"]
    assert lo == pytest.approx(2.5) and hi == pytest.approx(2.5)


def test_single_block_has_degenerate_interval():
    sums, n, _ = synthetic()
    blocks = np.zeros(n.size, dtype=int)
    res = paired_block_bootstrap(sums, n, blocks, "linear", n_boot=300, rng=np.random.default_rng(4))
    lo, hi = res["differences"]["other"]["ci"]["0.95"]
    assert lo == pytest.approx(hi) == pytest.approx(res["differences"]["other"]["point"])


def test_resamples_whole_blocks():
    # Each block has a constant per-reading error, so a replicate MAE is a weighted mean of block
    # constants. With 2 blocks, only 3 replicate values are possible (AA, AB, BB).
    n = np.array([1.0, 2.0, 1.0, 3.0])
    blocks = np.array([0, 0, 1, 1])
    lin = np.array([1.0, 2.0, 5.0, 15.0])           # block 0: 1 per reading, block 1: 5 per reading
    res = paired_block_bootstrap({"linear": lin, "other": 2 * lin}, n, blocks, "linear", n_boot=2000,
                                 rng=np.random.default_rng(5), levels=(0.5, 0.9))
    possible = {1.0, (3 + 20) / 7, 5.0}           # diff = MAE(linear) here since other = 2 * linear
    for lv in ("0.5", "0.9"):
        for v in res["differences"]["other"]["ci"][lv]:
            assert any(abs(v - p) < 1e-9 for p in possible)


# --- reproducibility and interval validity ----------------------------------------

def test_reproducible_with_same_seed_and_differs_with_other_seed():
    sums, n, blocks = synthetic()
    a = paired_block_bootstrap(sums, n, blocks, "linear", 400, np.random.default_rng(7), levels=(0.95, 0.995))
    b = paired_block_bootstrap(sums, n, blocks, "linear", 400, np.random.default_rng(7), levels=(0.95, 0.995))
    c = paired_block_bootstrap(sums, n, blocks, "linear", 400, np.random.default_rng(8), levels=(0.95, 0.995))
    assert a == b
    assert a["differences"]["other"]["ci"] != c["differences"]["other"]["ci"]


def test_intervals_are_ordered_nested_and_contain_point():
    sums, n, blocks = synthetic(shift=1.0)
    res = paired_block_bootstrap(sums, n, blocks, "linear", 3000, np.random.default_rng(9), levels=(0.95, 0.995))
    d = res["differences"]["other"]
    lo95, hi95 = d["ci"]["0.95"]
    lo99, hi99 = d["ci"]["0.995"]
    assert lo99 <= lo95 <= d["point"] <= hi95 <= hi99
    for m in sums:
        lo, hi = res["methods"][m]["mae_ci"]["0.95"]
        assert lo <= res["methods"][m]["mae"] <= hi


def test_detects_clear_difference_and_reports_inconclusive_null():
    sums, n, blocks = synthetic(shift=3.0)
    res = paired_block_bootstrap(sums, n, blocks, "linear", 2000, np.random.default_rng(10))
    assert res["differences"]["other"]["ci"]["0.95"][0] > 0
    sums0, n0, blocks0 = synthetic(shift=0.0, seed=11)
    res0 = paired_block_bootstrap(sums0, n0, blocks0, "linear", 2000, np.random.default_rng(12))
    lo, hi = res0["differences"]["other"]["ci"]["0.95"]
    assert lo < 0.3 and hi > -0.3   # a no-effect case should not show a large, clear difference


def test_block_bootstrap_wider_than_gap_bootstrap_under_day_dependence():
    # With a strong shared day effect, treating gaps as independent understates uncertainty.
    sums, n, blocks = synthetic(n_days=30, per_day=10, shift=0.5, seed=13)
    day = paired_block_bootstrap(sums, n, blocks, "linear", 3000, np.random.default_rng(14))
    gap = paired_block_bootstrap(sums, n, np.arange(n.size), "linear", 3000, np.random.default_rng(14))
    w = lambda r: np.diff(r["methods"]["linear"]["mae_ci"]["0.95"])[0]  # noqa: E731
    assert w(day) > 1.5 * w(gap)


# --- input validation -----------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(abs_err_sum={"linear": np.array([])}, n_hidden=np.array([]), blocks=np.array([])),
    dict(abs_err_sum={"linear": np.ones(3)}, n_hidden=np.ones(2), blocks=np.zeros(3)),
    dict(abs_err_sum={"linear": np.ones(3), "x": np.ones(2)}, n_hidden=np.ones(3), blocks=np.zeros(3)),
    dict(abs_err_sum={"linear": np.ones(3)}, n_hidden=np.array([1.0, 0.0, 1.0]), blocks=np.zeros(3)),
    dict(abs_err_sum={"spline": np.ones(3)}, n_hidden=np.ones(3), blocks=np.zeros(3)),
])
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        paired_block_bootstrap(reference="linear", n_boot=10, **kwargs)


# --- integration: paired step uses exactly the main evaluation's test gaps ---------

def test_paired_step_matches_main_evaluation_and_guards(tmp_path):
    import csv
    import json

    from gap_reconstruction import experiment as exp

    processed, out = tmp_path / "processed", tmp_path / "results"
    processed.mkdir()
    t = MIDNIGHT_AEST - 40 * DAY + 600.0 * np.arange(6000)
    t = np.delete(t, np.arange(5, t.size, 13))
    y = 450.0 + 30.0 * np.sin((t - t[0]) / 9000.0) + np.random.default_rng(0).normal(0, 4, t.size)
    with (processed / "co2_X.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch_s", "time_utc", "co2_ppm"])
        w.writerows([int(a), "", repr(float(b))] for a, b in zip(t, y))

    exp.run("co2", "X", processed, out)  # full run: writes test metrics and paired CSV, cross-checked inside
    rows = list(csv.DictReader((out / "co2_paired_bootstrap.csv").open(encoding="utf-8")))
    stored = {(r["duration_min"], r["method"]): float(r["mae"])
              for r in csv.DictReader((out / "co2_test_metrics.csv").open(encoding="utf-8"))}
    assert {r["comparison"] for r in rows} == {"forward_fill - linear", "cubic_spline - linear"}
    assert {int(r["block_days"]) for r in rows} == set(exp.PAIRED_BLOCK_DAYS)
    for r in rows:
        assert float(r["mae_linear"]) == pytest.approx(stored[(r["duration_min"], "linear")], rel=1e-5)
        assert float(r["ci95_lo"]) <= float(r["diff"]) <= float(r["ci95_hi"])
        assert float(r["ci_bonf_lo"]) <= float(r["ci95_lo"]) and float(r["ci95_hi"]) <= float(r["ci_bonf_hi"])

    again = exp.run_paired("co2", "X", processed, out)  # re-run is reproducible
    assert [{k: str(v) if not isinstance(v, float) else f"{v:.6g}" for k, v in r.items()} for r in again] == rows

    info = json.loads((out / "co2_run_info.json").read_text(encoding="utf-8"))
    info["test_results_stale"] = True
    (out / "co2_run_info.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(SystemExit):
        exp.run_paired("co2", "X", processed, out)


def test_check_against_test_metrics_detects_mismatch(tmp_path):
    from gap_reconstruction import experiment as exp

    p = tmp_path / "m.csv"
    p.write_text("duration_min,method,mae\n10,linear,7.0\n10,cubic_spline,9.0\n", encoding="utf-8")
    ok = [{"duration_min": 10, "comparison": "cubic_spline - linear", "mae_method": 9.0, "mae_linear": 7.0}]
    exp._check_against_test_metrics(ok, p)
    with pytest.raises(SystemExit):
        exp._check_against_test_metrics([{**ok[0], "mae_linear": 7.5}], p)
