"""Command-line step 3: runtime benchmark of the reconstruction methods on the test gaps.

Run from the repository root, after the main experiment::

    python -m gap_reconstruction.benchmark                 # CO2, 2 runs x 30 rounds

Why a separate benchmark
------------------------
The ``us_per_gap`` column written by `gap_reconstruction.experiment` is timed
inside the evaluation. Methods are timed one after another with only 9
repeats, while the laptop's clock speed drifts. Methods measured at different
moments therefore saw different machine states, and one run showed all
methods 3.5–4x faster for the last duration only. This module measures more
fairly:

- **Same inputs as the evaluation.** The test gaps of the chosen context
  (`experiment._test_gaps`) are used, with the known/query arrays sliced in
  advance. Only the method call is timed: the method's own input validation
  is included because it is part of the method as implemented, while data
  slicing, the leak check, metrics, plotting and file I/O are excluded.
- **Warm-up.** ``WARMUP_ROUNDS`` untimed rounds run first (imports, caches,
  CPU frequency ramp-up).
- **Interleaving.** In every round each method reconstructs all gaps of one
  duration once, and the method order rotates from round to round. Each
  method is therefore measured as often in each position, close in time to
  the others, so slow drift in machine speed affects all methods alike.
- **Repetition.** ``ROUNDS`` timed rounds per duration and method. The
  median, quartiles, minimum and maximum of the per-gap time are reported.
- **Reproducibility.** The whole benchmark is repeated ``RUNS`` times; the
  spread between runs is reported, not hidden.
- **Duration order.** Odd runs time the durations from shortest to longest
  gap and even runs in reverse (`duration_order`). Slow machine drift then
  pushes the runs in opposite directions instead of looking like an effect
  of gap duration. The order used is recorded per run.
- **Same gaps as the scored results.** Before any timing, the rebuilt test
  gaps are checked against ``results/<var>_test_metrics.csv``: the gap count
  per duration (`check_test_gaps`) and every method's MAE (`check_test_maes`)
  must match. A hash of each duration's gap set is recorded.
- The garbage collector is paused during each timed loop, and
  `time.perf_counter_ns` is used.

Outputs (no raw readings):

- ``results/<var>_runtime_benchmark.csv``: one row per run, duration and method
- ``results/<var>_runtime_benchmark.json``: settings, environment, summary
- ``results/figures/<var>_runtime_benchmark.png``

Absolute times depend on the machine and its power state. Ratios between
methods are the more transferable result.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402

from . import experiment as exp  # noqa: E402
from .evaluation import chronological_split, gap_set_hash, reconstruct_gap  # noqa: E402
from .methods import METHODS  # noqa: E402

ROUNDS = 30
WARMUP_ROUNDS = 3
RUNS = 2


def method_order(round_index: int, names: list[str]) -> list[str]:
    """Rotate the method order by one place per round (round-robin)."""
    k = round_index % len(names)
    return names[k:] + names[:k]


def duration_order(run: int, durations=None) -> list[int]:
    """Order in which run ``run`` (1-based) times the gap durations.

    Odd runs go from the shortest to the longest gap, even runs in reverse,
    so that slow drift in machine speed cannot masquerade as an effect of
    gap duration.
    """
    d = sorted(exp.DURATIONS_MIN if durations is None else durations)
    return d if run % 2 == 1 else d[::-1]


def check_test_gaps(gaps_by_duration: dict, metrics_path: Path) -> None:
    """Stop if the rebuilt test gaps do not match those scored in the test metrics.

    Compares the number of gaps per duration with ``n_gaps`` in
    ``<var>_test_metrics.csv``. Raises SystemExit if that file is missing or
    any count differs, so the benchmark never times a different set of gaps
    from the one that was evaluated.
    """
    if not metrics_path.is_file():
        raise SystemExit(f"{metrics_path} not found - run the main experiment first.")
    stored: dict[int, set] = {}
    with metrics_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            stored.setdefault(int(r["duration_min"]), set()).add(int(r["n_gaps"]))
    for d, gaps in gaps_by_duration.items():
        if stored.get(d) != {len(gaps)}:
            raise SystemExit(f"{d}-min test gaps: rebuilt {len(gaps)}, stored {sorted(stored.get(d, set()))} "
                             f"in {metrics_path}; the benchmark would not time the evaluated gaps.")


def check_test_maes(t: np.ndarray, y: np.ndarray, gaps_by_duration: dict, metrics_path: Path) -> None:
    """Stop if the methods' MAEs on the rebuilt gaps differ from the stored test metrics.

    For every duration and method, the MAE pooled over all hidden readings is
    recomputed and compared with ``mae`` in ``<var>_test_metrics.csv``, using
    the same tolerance as `experiment._check_against_test_metrics`:
    ``|recomputed - stored| <= 1e-5 * max(1, |stored|)``. This is stronger than
    comparing gap counts, because a different set of gaps with the same
    count would still change the MAEs.
    """
    with metrics_path.open(encoding="utf-8") as fh:
        stored = {(int(r["duration_min"]), r["method"]): float(r["mae"]) for r in csv.DictReader(fh)}
    for d, gaps in gaps_by_duration.items():
        n = sum(g.hidden.size for g in gaps)
        for m, f in METHODS.items():
            if (d, m) not in stored:
                raise SystemExit(f"{metrics_path} has no MAE for {m} at {d} min.")
            value = sum(float(np.abs(reconstruct_gap(f, t, y, g) - y[g.hidden]).sum()) for g in gaps) / n
            ref = stored[(d, m)]
            if abs(value - ref) > 1e-5 * max(1.0, abs(ref)):
                raise SystemExit(f"{m} MAE at {d} min: recomputed {value:.6g}, stored {ref:.6g} in "
                                 f"{metrics_path}; the benchmark would not time the evaluated gaps.")


def time_methods(inputs: list[tuple], methods: dict, rounds: int = ROUNDS,
                 warmup: int = WARMUP_ROUNDS) -> dict[str, np.ndarray]:
    """Time each method over all ``inputs``, interleaved and repeated.

    Parameters
    ----------
    inputs : list of (t_known, y_known, t_query)
        Pre-sliced arrays, one tuple per gap.
    methods : dict of str -> callable
        Methods with the signature ``f(t_known, y_known, t_query)``.
    rounds, warmup : int
        Number of timed and untimed rounds.

    Returns
    -------
    dict of str -> numpy.ndarray
        For each method, ``rounds`` values of the mean time per gap in
        seconds (total loop time / number of gaps).
    """
    if not inputs:
        raise ValueError("no inputs to time")
    names = list(methods)
    out = {m: [] for m in names}
    for r in range(warmup + rounds):
        for m in method_order(r, names):
            f = methods[m]
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                t0 = time.perf_counter_ns()
                for tk, yk, tq in inputs:
                    f(tk, yk, tq)
                elapsed = time.perf_counter_ns() - t0
            finally:
                if gc_was_enabled:
                    gc.enable()
            if r >= warmup:
                out[m].append(elapsed * 1e-9 / len(inputs))
    return {m: np.array(v) for m, v in out.items()}


def summarise(per_gap_s: np.ndarray) -> dict:
    """Median, quartiles, minimum and maximum of per-gap times, in microseconds."""
    us = np.asarray(per_gap_s) * 1e6
    return {"median_us": float(np.median(us)), "p25_us": float(np.quantile(us, 0.25)),
            "p75_us": float(np.quantile(us, 0.75)), "min_us": float(us.min()), "max_us": float(us.max())}


def run_benchmark(variable: str, sensor: str, processed: Path, out: Path, rounds: int = ROUNDS,
                  warmup: int = WARMUP_ROUNDS, runs: int = RUNS) -> dict:
    """Benchmark all methods on the existing test gaps and write the outputs."""
    if rounds < 1 or runs < 1:
        raise ValueError("rounds and runs must both be >= 1")
    spec = exp.VARIABLES[variable]
    info_path = out / f"{variable}_run_info.json"
    if not info_path.is_file():
        raise SystemExit(f"{info_path} not found - run the main experiment first.")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("test_results_stale"):
        raise SystemExit("test results are flagged stale - rerun the main experiment first.")
    context = int(info["chosen_context"])
    t, y = exp.load_processed(processed / spec["file"].format(sensor=sensor), spec["column"])
    split = chronological_split(t.size, exp.TEST_FRACTION)
    gaps = {d: exp._test_gaps(t, split, context, d) for d in exp.DURATIONS_MIN}
    check_test_gaps(gaps, out / f"{variable}_test_metrics.csv")
    check_test_maes(t, y, gaps, out / f"{variable}_test_metrics.csv")
    inputs = {d: [(t[g.known].copy(), y[g.known].copy(), t[g.hidden].copy())
                  for g in gaps[d]] for d in exp.DURATIONS_MIN}

    rows, orders, wall0 = [], {}, time.perf_counter()
    for run in range(1, runs + 1):
        orders[str(run)] = duration_order(run)
        for position, d in enumerate(orders[str(run)], start=1):
            times = time_methods(inputs[d], METHODS, rounds, warmup)
            lin = times["linear"]
            for m, v in times.items():
                rows.append({"run": run, "duration_min": d, "duration_position": position, "method": m,
                             "n_gaps": len(inputs[d]), "rounds": rounds, **summarise(v),
                             # paired within each round, so the ratio is insensitive to drift
                             "median_ratio_to_linear": float(np.median(v / lin))})
    exp._write_csv(out / f"{variable}_runtime_benchmark.csv", rows)

    summary = {}
    for m in METHODS:
        med = np.array([r["median_us"] for r in rows if r["method"] == m])
        ratio = np.array([r["median_ratio_to_linear"] for r in rows if r["method"] == m])
        summary[m] = {"median_us_range_over_runs_and_durations": [float(med.min()), float(med.max())],
                      "max_us_any_round": float(max(r["max_us"] for r in rows if r["method"] == m)),
                      "ratio_to_linear_range": [float(ratio.min()), float(ratio.max())]}
    between_runs = {}
    for d in exp.DURATIONS_MIN:
        for m in METHODS:
            meds = [r["median_us"] for r in rows if r["method"] == m and r["duration_min"] == d]
            between_runs[f"{d}_{m}"] = float(max(meds) / min(meds))
    meta = {
        "variable": variable, "sensor_id": sensor, "context": context, "rounds": rounds,
        "test_gaps_checked_against": f"{variable}_test_metrics.csv (n_gaps per duration and pooled MAE "
                                     "per duration and method, tolerance 1e-5 * max(1, |stored|))",
        "test_gap_set_hash_by_duration": {str(d): gap_set_hash(g) for d, g in gaps.items()},
        "warmup_rounds": warmup, "runs": runs, "timer": "time.perf_counter_ns",
        "garbage_collector": "paused during each timed loop",
        "interleaving": "method order rotated every round",
        "duration_order_by_run": orders,
        "timed": "method call only (includes the method's input validation)",
        "excluded": "data slicing, leak check, metric computation, plotting, file I/O",
        "summary": summary,
        "max_between_run_ratio_of_medians": float(max(between_runs.values())),
        "between_run_ratio_of_medians": between_runs,
        "wall_time_s": round(time.perf_counter() - wall0, 1),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "machine": platform.processor() or platform.machine(), "platform": platform.platform(),
        "note": "absolute times depend on the machine and its power state; ratios are more transferable",
    }
    (out / f"{variable}_runtime_benchmark.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _plot(rows, out / "figures" / f"{variable}_runtime_benchmark.png", rounds, runs)
    return meta


def _plot(rows: list[dict], path: Path, rounds: int, runs: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for m in METHODS:
        for run in range(1, runs + 1):
            ls = ("-", "--", ":", "-.")[(run - 1) % 4]
            r = [x for x in rows if x["method"] == m and x["run"] == run]
            d = np.array([x["duration_min"] for x in r], float)
            med = np.array([x["median_us"] for x in r])
            ax.errorbar(d, med, yerr=[med - [x["p25_us"] for x in r], np.array([x["p75_us"] for x in r]) - med],
                        marker="o", capsize=3, ls=ls, color=exp.COLORS[m],
                        label=f"{exp.LABELS[m]} (run {run})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(list(exp.DURATIONS_MIN), [str(x) for x in exp.DURATIONS_MIN])
    ax.set_xlabel("Gap duration (min)")
    ax.set_ylabel("Time per gap (µs): median, 25–75 %")
    ax.set_title(f"Interleaved benchmark: {rounds} rounds x {runs} runs", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    """Parse arguments and run the benchmark."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variable", choices=sorted(exp.VARIABLES), default="co2")
    p.add_argument("--sensor", default="6012002000869")
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument("--runs", type=int, default=RUNS)
    args = p.parse_args(argv)
    meta = run_benchmark(args.variable, args.sensor, args.processed, args.out, args.rounds, WARMUP_ROUNDS, args.runs)
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    main()
