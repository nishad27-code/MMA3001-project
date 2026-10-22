"""Tests for the runtime benchmark harness (timing logic, not machine speed)."""

import csv
import json
import time

import numpy as np
import pytest

from gap_reconstruction import benchmark as bm
from gap_reconstruction import experiment as exp
from gap_reconstruction.evaluation import reconstruct_gap
from gap_reconstruction.methods import METHODS


def test_method_order_rotates_and_balances_positions():
    names = ["a", "b", "c"]
    orders = [bm.method_order(r, names) for r in range(6)]
    assert orders[0] == ["a", "b", "c"] and orders[1] == ["b", "c", "a"] and orders[3] == orders[0]
    for pos in range(3):  # every method occupies every position equally often
        assert sorted(o[pos] for o in orders) == ["a", "a", "b", "b", "c", "c"]


def test_duration_order_alternates_between_runs():
    assert bm.duration_order(1, [60, 10, 360]) == [10, 60, 360]
    assert bm.duration_order(2, [60, 10, 360]) == [360, 60, 10]
    assert bm.duration_order(3) == sorted(exp.DURATIONS_MIN)


def test_time_methods_counts_rounds_excludes_warmup_and_interleaves():
    calls = []

    def make(name, delay):
        def f(tk, yk, tq):
            calls.append(name)
            time.sleep(delay)
        return f

    inputs = [(None, None, None)] * 2
    out = bm.time_methods(inputs, {"fast": make("fast", 0.0), "slow": make("slow", 0.002)}, rounds=4, warmup=1)
    assert {k: v.size for k, v in out.items()} == {"fast": 4, "slow": 4}
    assert len(calls) == 2 * 2 * 5  # 2 methods x 2 inputs x (4 + 1 warm-up) rounds
    assert calls[:4] == ["fast", "fast", "slow", "slow"] and calls[4:8] == ["slow", "slow", "fast", "fast"]
    assert np.all(out["slow"] > out["fast"]) and np.all(out["slow"] >= 0.0019)  # per-gap seconds


def test_time_methods_restores_gc_and_rejects_empty():
    import gc

    gc.enable()
    bm.time_methods([(None, None, None)], {"x": lambda *a: None}, rounds=1, warmup=0)
    assert gc.isenabled()
    with pytest.raises(ValueError):
        bm.time_methods([], {"x": lambda *a: None})


def test_summarise_matches_numpy():
    v = np.array([1e-6, 2e-6, 3e-6, 10e-6])
    s = bm.summarise(v)
    assert s["median_us"] == pytest.approx(2.5)
    assert s["p25_us"] == pytest.approx(np.quantile(v * 1e6, 0.25))
    assert s["min_us"] == pytest.approx(1.0) and s["max_us"] == pytest.approx(10.0)


def test_run_benchmark_end_to_end_and_guards(tmp_path):
    processed, out = tmp_path / "processed", tmp_path / "results"
    processed.mkdir()
    out.mkdir()
    t = 1.7e9 + 600.0 * np.arange(3000)
    t = np.delete(t, np.arange(5, t.size, 13))
    y = 450.0 + 20.0 * np.sin((t - t[0]) / 9000.0)
    with (processed / "co2_X.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch_s", "time_utc", "co2_ppm"])
        w.writerows([int(a), "", repr(float(b))] for a, b in zip(t, y))
    with pytest.raises(SystemExit):  # no run_info yet
        bm.run_benchmark("co2", "X", processed, out, rounds=2, warmup=0, runs=1)
    (out / "co2_run_info.json").write_text(json.dumps({"chosen_context": 2, "test_results_stale": False}),
                                           encoding="utf-8")
    with pytest.raises(SystemExit):  # no stored test metrics yet
        bm.run_benchmark("co2", "X", processed, out, rounds=2, warmup=0, runs=1)
    split = exp.chronological_split(t.size, exp.TEST_FRACTION)
    counts = {d: len(exp._test_gaps(t, split, 2, d)) for d in exp.DURATIONS_MIN}

    true_mae = {}
    for d in exp.DURATIONS_MIN:
        gs = exp._test_gaps(t, split, 2, d)
        n = sum(g.hidden.size for g in gs)
        for m, f in METHODS.items():
            true_mae[(d, m)] = sum(float(np.abs(reconstruct_gap(f, t, y, g) - y[g.hidden]).sum()) for g in gs) / n

    def write_metrics(n_by_duration, mae=true_mae):
        with (out / "co2_test_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["duration_min", "n_gaps", "method", "mae"])
            for d, k in n_by_duration.items():
                for m in METHODS:
                    w.writerow([d, k, m, repr(mae[(d, m)])])

    write_metrics({**counts, 60: counts[60] + 1})  # one wrong count must stop the benchmark
    with pytest.raises(SystemExit, match="60-min"):
        bm.run_benchmark("co2", "X", processed, out, rounds=2, warmup=0, runs=1)
    write_metrics(counts, {**true_mae, (180, "linear"): true_mae[(180, "linear")] * 1.01})  # wrong MAE
    with pytest.raises(SystemExit, match="linear MAE at 180 min"):
        bm.run_benchmark("co2", "X", processed, out, rounds=2, warmup=0, runs=1)
    write_metrics(counts)
    meta = bm.run_benchmark("co2", "X", processed, out, rounds=3, warmup=1, runs=2)
    rows = list(csv.DictReader((out / "co2_runtime_benchmark.csv").open(encoding="utf-8")))
    assert len(rows) == 2 * len(exp.DURATIONS_MIN) * len(METHODS)
    assert all(float(r["min_us"]) <= float(r["median_us"]) <= float(r["max_us"]) for r in rows)
    assert all(float(r["median_ratio_to_linear"]) == 1.0 for r in rows if r["method"] == "linear")
    assert meta["context"] == 2 and meta["runs"] == 2 and meta["rounds"] == 3
    # Issue 1: run 1 shortest-first, run 2 longest-first, recorded in JSON and CSV
    assert meta["duration_order_by_run"] == {"1": sorted(exp.DURATIONS_MIN),
                                             "2": sorted(exp.DURATIONS_MIN, reverse=True)}
    for r in rows:
        order = meta["duration_order_by_run"][r["run"]]
        assert order[int(r["duration_position"]) - 1] == int(r["duration_min"])
    # Issue 2: a gap-set hash is recorded for every duration
    assert set(meta["test_gap_set_hash_by_duration"]) == {str(d) for d in exp.DURATIONS_MIN}
    assert (out / "figures" / "co2_runtime_benchmark.png").is_file()
    with pytest.raises(ValueError):
        bm.run_benchmark("co2", "X", processed, out, rounds=0, warmup=0, runs=1)
    (out / "co2_run_info.json").write_text(json.dumps({"chosen_context": 2, "test_results_stale": True}),
                                           encoding="utf-8")
    with pytest.raises(SystemExit):
        bm.run_benchmark("co2", "X", processed, out, rounds=2, warmup=0, runs=1)
