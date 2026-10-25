"""Command-line step 2: run the gap-reconstruction experiments.

Run from the repository root, after `gap_reconstruction.extract`::

    python -m gap_reconstruction.experiment                # CO2 (primary)
    python -m gap_reconstruction.experiment --variable temperature
    python -m gap_reconstruction.experiment --tuning-only    # dev tuning + gap stats only
    python -m gap_reconstruction.experiment --paired-only    # paired day-block bootstrap only

Protocol (fixed before looking at any test-period result):

1. Chronological split: the first 70 % of readings form the development
   period and the last 30 % the test period.
2. **Development period only**: the spline's context size (known readings
   supplied on each side of a gap) is chosen from `CONTEXT_CANDIDATES` by
   the lowest mean spline MAE over all gap durations. For each duration,
   one common set of gaps, eligible for the largest context, is used for
   every context size, so context is the only thing that varies. This is
   the sensitivity study of the method setting.
3. **Test period**: with that context fixed, every method is scored on the
   same non-overlapping random gaps for each duration in `DURATIONS_MIN`.
   This is the sensitivity study of gap duration. Runtime is measured, and
   one configuration is profiled with cProfile.
4. **Paired comparison (test period, no selection)**: on the same test gaps,
   the MAE difference of every other method minus linear is given a
   paired day-block bootstrap interval (`paired_comparison`). This step
   only quantifies uncertainty; nothing is chosen from it.

Outputs (aggregate statistics only, safe to commit):

- ``results/<var>_dev_context_sensitivity.csv``
- ``results/<var>_test_metrics.csv``
- ``results/<var>_gap_definition_stats.csv`` (hidden-reading counts and
  boundary intervals actually produced, per period and nominal duration)
- ``results/<var>_paired_bootstrap.csv`` (paired MAE differences vs linear
  with day-block bootstrap intervals)
- ``results/<var>_run_info.json``
- ``results/<var>_profile.txt``
- ``results/figures/<var>_*.png``

``results/local/`` (git-ignored) holds figures that plot raw readings.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import io
import json
import platform
import pstats
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, reproducible rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402

from .evaluation import (  # noqa: E402
    chronological_split, common_gaps, day_blocks, evaluate, gap_set_hash, gap_statistics, paired_block_bootstrap,
    reconstruct_gap, sample_gaps,
)
from .loader import epoch_to_utc_iso  # noqa: E402
from .methods import METHODS, cubic_spline  # noqa: E402

DURATIONS_MIN = (10, 30, 60, 180, 360)
CONTEXT_CANDIDATES = (2, 4, 8, 16, 32)
N_GAPS_MAX = 200
TEST_FRACTION = 0.3
SEED = 2024
PAIRED_REFERENCE = "linear"
PAIRED_N_BOOT = 5000
PAIRED_BLOCK_DAYS = (1, 3)  # 1 day = primary analysis; 3 days = robustness check
UTC_OFFSET_HOURS = 10.0  # AEST; no daylight-saving change in the test period
CI_LEVEL = 0.95
# Bonferroni over every (method vs linear) x duration comparison per variable:
# with 4 methods, 3 comparisons x 5 durations -> 1 - 0.05 / 15
N_PAIRED_COMPARISONS = (len(METHODS) - 1) * len(DURATIONS_MIN)
CI_LEVEL_BONFERRONI = 1.0 - (1.0 - CI_LEVEL) / N_PAIRED_COMPARISONS

VARIABLES = {
    "co2": {"file": "co2_{sensor}.csv", "column": "co2_ppm", "unit": "ppm", "label": "CO$_2$"},
    "temperature": {"file": "temperature_{sensor}.csv", "column": "temperature_c", "unit": "°C",
                    "label": "Temperature"},
}
LABELS = {"forward_fill": "Forward-fill (baseline)", "linear": "Linear", "cubic_spline": "Cubic spline",
          "pchip": "PCHIP"}
COLORS = {"forward_fill": "#7f7f7f", "linear": "#1f77b4", "cubic_spline": "#d62728", "pchip": "#2ca02c"}


def load_processed(path: Path, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a processed series written by `gap_reconstruction.extract`."""
    if not path.is_file():
        raise SystemExit(f"{path} not found - run `python -m gap_reconstruction.extract` first.")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    t = np.array([float(r["epoch_s"]) for r in rows])
    y = np.array([float(r[column]) for r in rows])
    if t.size < 100 or np.any(np.diff(t) <= 0):
        raise SystemExit(f"{path}: series too short or not strictly increasing")
    return t, y


def _rng(duration_min: int, context: int, period: str) -> np.random.Generator:
    """Independent, reproducible stream per (duration, context, period)."""
    return np.random.default_rng([SEED, duration_min, context, 0 if period == "dev" else 1])


def _test_gaps(t: np.ndarray, split: int, context: int, duration_min: int) -> list:
    """Test-period gaps for one duration (placement depends on the chosen context)."""
    return sample_gaps(t, duration_min * 60.0, context, split, t.size, N_GAPS_MAX,
                       _rng(duration_min, context, "test"))


def tune_context(t: np.ndarray, y: np.ndarray, split: int, contexts=CONTEXT_CANDIDATES,
                 durations=DURATIONS_MIN, n_max: int = N_GAPS_MAX) -> tuple[list[dict], dict, int, dict]:
    """Choose the spline context on the development period ``[0, split)`` only.

    For each duration, one set of gaps eligible for the largest context is
    drawn (`common_gaps`), and every context size is scored on exactly those
    hidden readings. The criterion is the lowest spline MAE averaged over the
    durations. Ties are broken towards the smaller context.

    Returns
    -------
    rows : list of dict
        One row per (context, duration, method): number of gaps, a hash of
        the gap set (identical across contexts by construction, and
        asserted), MAE, RMSE, and for the spline the largest absolute
        difference between its predictions and those made with the largest
        context.
    spline_mean : dict
        Mean spline MAE over durations, per context.
    context : int
        The selected context.
    dev_sets : dict
        ``{duration_min: gaps}`` of the common development gaps.
    """
    contexts = sorted(contexts)
    cmax = contexts[-1]
    rows, dev_sets = [], {}
    for d in durations:
        sets = common_gaps(t, d * 60.0, contexts, 0, split, n_max, _rng(d, cmax, "dev"))
        hashes = {c: gap_set_hash(sets[c]) for c in contexts}
        assert len(set(hashes.values())) == 1, "contexts were not evaluated on identical gaps"
        assert all(int(g.known.max()) < split for g in sets[cmax]), "tuning gap uses test-period data"
        dev_sets[d] = sets[cmax]
        ref = [reconstruct_gap(cubic_spline, t, y, g) for g in sets[cmax]]
        for c in contexts:
            r = evaluate(t, y, sets[c], timing_repeats=1, n_boot=200, seed=SEED)
            pred = [reconstruct_gap(cubic_spline, t, y, g) for g in sets[c]]
            diff = max(float(np.max(np.abs(p - q))) for p, q in zip(pred, ref))
            for m, mr in r["methods"].items():
                rows.append({"context": c, "duration_min": d, "n_gaps": r["n_gaps"], "gap_set_hash": hashes[c],
                             "method": m, "mae": mr["mae"], "rmse": mr["rmse"],
                             "spline_max_abs_diff_vs_largest_context": diff if m == "cubic_spline" else ""})
    spline_mean = {c: float(np.mean([r["mae"] for r in rows if r["context"] == c and r["method"] == "cubic_spline"]))
                   for c in contexts}
    context = min(contexts, key=lambda c: (spline_mean[c], c))
    return rows, spline_mean, int(context), dev_sets


def _gap_stat_rows(t: np.ndarray, split: int, context: int, dev_sets: dict) -> list[dict]:
    """Gap-definition statistics for the development (tuning) and test gaps."""
    rows = []
    for d, gaps in dev_sets.items():
        rows.append({"period": "dev_tuning", "eligibility_context": max(CONTEXT_CANDIDATES),
                     **gap_statistics(t, gaps, d * 60.0)})
    for d in DURATIONS_MIN:
        rows.append({"period": "test", "eligibility_context": context,
                     **gap_statistics(t, _test_gaps(t, split, context, d), d * 60.0)})
    return rows


def _tuning_info(spline_mean: dict, context: int, dev_sets: dict) -> dict:
    """Tuning provenance recorded in ``<var>_run_info.json``."""
    return {
        "tuning_design": "common development gaps per duration, eligible for the largest context; "
                         "every context scored on identical hidden readings",
        "dev_mean_spline_mae_by_context": {str(k): v for k, v in spline_mean.items()},
        "dev_gap_set_hash_by_duration": {str(d): gap_set_hash(g) for d, g in dev_sets.items()},
        "dev_n_gaps_by_duration": {str(d): len(g) for d, g in dev_sets.items()},
        "chosen_context": context,
    }


def _verdict(lo: float, hi: float) -> str:
    """Plain-language reading of an interval for ``MAE[other] - MAE[linear]``."""
    if lo > 0:
        return "linear lower MAE"
    if hi < 0:
        return "other method lower MAE"
    return "inconclusive"


def paired_comparison(t: np.ndarray, y: np.ndarray, split: int, context: int, unit: str) -> list[dict]:
    """Paired day-block bootstrap of MAE differences versus linear on the test gaps.

    Uses exactly the test gaps of the main evaluation (`_test_gaps` with the
    chosen ``context``), so every method is compared on identical hidden
    readings. Rows report, per duration, block length and comparison: the
    point difference ``MAE[method] - MAE[linear]``, its 95 % and
    Bonferroni-adjusted intervals, the share of replicates in which the
    method was at least as good as linear, and a verdict.
    """
    rows = []
    for d in DURATIONS_MIN:
        gaps = _test_gaps(t, split, context, d)
        n_hidden = np.array([g.hidden.size for g in gaps], dtype=float)
        abs_sum = {m: np.array([float(np.abs(reconstruct_gap(f, t, y, g) - y[g.hidden]).sum()) for g in gaps])
                   for m, f in METHODS.items()}
        start = t[np.array([g.hidden[0] for g in gaps])]
        for bd in PAIRED_BLOCK_DAYS:
            res = paired_block_bootstrap(abs_sum, n_hidden, day_blocks(start, UTC_OFFSET_HOURS, bd),
                                         PAIRED_REFERENCE, PAIRED_N_BOOT,
                                         np.random.default_rng([SEED, d, bd, 7]),
                                         levels=(CI_LEVEL, CI_LEVEL_BONFERRONI))
            for m, diff in res["differences"].items():
                lo, hi = diff["ci"][str(CI_LEVEL)]
                blo, bhi = diff["ci"][str(CI_LEVEL_BONFERRONI)]
                rows.append({
                    "duration_min": d, "block_days": bd, "n_gaps": res["n_gaps"], "n_blocks": res["n_blocks"],
                    "comparison": f"{m} - {PAIRED_REFERENCE}", "unit": unit,
                    "mae_method": res["methods"][m]["mae"], "mae_linear": res["methods"][PAIRED_REFERENCE]["mae"],
                    # unpaired block-bootstrap intervals of each MAE, for comparison with the
                    # gap-level intervals in <var>_test_metrics.csv
                    "mae_method_block_ci95_lo": res["methods"][m]["mae_ci"][str(CI_LEVEL)][0],
                    "mae_method_block_ci95_hi": res["methods"][m]["mae_ci"][str(CI_LEVEL)][1],
                    "mae_linear_block_ci95_lo": res["methods"][PAIRED_REFERENCE]["mae_ci"][str(CI_LEVEL)][0],
                    "mae_linear_block_ci95_hi": res["methods"][PAIRED_REFERENCE]["mae_ci"][str(CI_LEVEL)][1],
                    "diff": diff["point"], "ci95_lo": lo, "ci95_hi": hi,
                    "ci_bonferroni_level": round(CI_LEVEL_BONFERRONI, 4), "ci_bonf_lo": blo, "ci_bonf_hi": bhi,
                    "frac_boot_method_at_least_as_good": diff["frac_boot_diff_le_0"],
                    "verdict_95": _verdict(lo, hi), "verdict_bonferroni": _verdict(blo, bhi),
                    "n_boot": res["n_boot"],
                })
    return rows


def _check_against_test_metrics(rows: list[dict], path: Path) -> None:
    """Fail loudly if the paired step's MAEs disagree with the stored test metrics."""
    if not path.is_file():
        return
    stored = {(int(r["duration_min"]), r["method"]): float(r["mae"]) for r in csv.DictReader(path.open(encoding="utf-8"))}
    for r in rows:
        m = r["comparison"].split(" - ")[0]
        for method, value in ((m, r["mae_method"]), (PAIRED_REFERENCE, r["mae_linear"])):
            ref = stored[(r["duration_min"], method)]
            if abs(value - ref) > 1e-5 * max(1.0, abs(ref)):
                raise SystemExit(f"paired MAE {value} != stored {ref} for {method}, {r['duration_min']} min: "
                                 "test gaps differ from the main evaluation")


def run_paired(variable: str, sensor: str, processed: Path, out: Path) -> list[dict]:
    """Re-run only the paired day-block bootstrap for an existing test evaluation.

    Reads the chosen context from ``<var>_run_info.json`` (it must exist and
    not be flagged stale), rebuilds the identical test gaps, and checks the
    recomputed MAEs against ``<var>_test_metrics.csv`` before writing
    ``<var>_paired_bootstrap.csv`` and its figure.
    """
    spec = VARIABLES[variable]
    info_path = out / f"{variable}_run_info.json"
    if not info_path.is_file():
        raise SystemExit(f"{info_path} not found - run the full experiment first.")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("test_results_stale"):
        raise SystemExit("test results are flagged stale - rerun the full experiment first.")
    t, y = load_processed(processed / spec["file"].format(sensor=sensor), spec["column"])
    split = chronological_split(t.size, TEST_FRACTION)
    rows = paired_comparison(t, y, split, int(info["chosen_context"]), spec["unit"])
    _check_against_test_metrics(rows, out / f"{variable}_test_metrics.csv")
    _write_csv(out / f"{variable}_paired_bootstrap.csv", rows)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _plot_paired(rows, spec, out / "figures" / f"{variable}_paired_differences.png")
    return rows


def run_tuning(variable: str, sensor: str, processed: Path, out: Path) -> dict:
    """Re-run only the development-period tuning and gap statistics.

    The test-period gaps are *placed* (to describe them) but never scored.
    The existing test outputs are left untouched. If the selected context
    differs from the one recorded in ``<var>_run_info.json``, those test
    outputs are stale, and this is flagged.
    """
    spec = VARIABLES[variable]
    t, y = load_processed(processed / spec["file"].format(sensor=sensor), spec["column"])
    split = chronological_split(t.size, TEST_FRACTION)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    dev_rows, spline_mean, context, dev_sets = tune_context(t, y, split)
    _write_csv(out / f"{variable}_dev_context_sensitivity.csv", dev_rows)
    _write_csv(out / f"{variable}_gap_definition_stats.csv", _gap_stat_rows(t, split, context, dev_sets))
    _plot_context(dev_rows, spec, out / "figures" / f"{variable}_dev_context_sensitivity.png", context)
    info_path = out / f"{variable}_run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    previous = info.get("chosen_context")
    info.update(_tuning_info(spline_mean, context, dev_sets))
    info["test_results_stale"] = previous is not None and previous != context
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    if info["test_results_stale"]:
        print(f"WARNING: selected context {context} differs from {previous} used for the test results; "
              "rerun the full experiment.")
    return info


def run(variable: str, sensor: str, processed: Path, out: Path) -> dict:
    """Execute the full protocol for one variable and write all outputs."""
    spec = VARIABLES[variable]
    t, y = load_processed(processed / spec["file"].format(sensor=sensor), spec["column"])
    n = t.size
    split = chronological_split(n, TEST_FRACTION)
    fig_dir, local_dir = out / "figures", out / "local"
    fig_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()

    # --- 1. development period: choose spline context -------------------------
    dev_rows, spline_mean, context, dev_sets = tune_context(t, y, split)
    _write_csv(out / f"{variable}_dev_context_sensitivity.csv", dev_rows)
    _write_csv(out / f"{variable}_gap_definition_stats.csv", _gap_stat_rows(t, split, context, dev_sets))

    # --- 2. test period: fixed context, every duration -------------------------
    test_rows, test_raw = [], {}
    for d in DURATIONS_MIN:
        gaps = _test_gaps(t, split, context, d)
        r = evaluate(t, y, gaps, timing_repeats=9, n_boot=1000, seed=SEED)
        test_raw[d] = (gaps, r)
        lin, spl = np.array(r["methods"]["linear"]["per_gap_mae"]), np.array(r["methods"]["cubic_spline"]["per_gap_mae"])
        ff = np.array(r["methods"]["forward_fill"]["per_gap_mae"])
        pch = np.array(r["methods"]["pchip"]["per_gap_mae"])
        for m, mr in r["methods"].items():
            pg = np.array(mr["per_gap_mae"])
            test_rows.append({
                "duration_min": d, "n_gaps": r["n_gaps"],
                "mean_hidden_readings": round(r["mean_hidden_readings"], 3),
                "min_hidden_readings": r["min_hidden_readings"], "max_hidden_readings": r["max_hidden_readings"],
                "mean_boundary_interval_min": round(r["mean_boundary_interval_min"], 2),
                "max_boundary_interval_min": round(r["max_boundary_interval_min"], 2),
                "method": m, "unit": spec["unit"],
                "bias": mr["bias"], "mae": mr["mae"], "mae_ci_lo": mr["mae_ci95"][0], "mae_ci_hi": mr["mae_ci95"][1],
                "rmse": mr["rmse"], "rmse_ci_lo": mr["rmse_ci95"][0], "rmse_ci_hi": mr["rmse_ci95"][1],
                "midpoint_mae": mr["midpoint_mae"], "p95_gap_mae": float(np.quantile(pg, 0.95)),
                "max_gap_mae": float(pg.max()), "max_overshoot": mr["max_overshoot"],
                "us_per_gap": mr["us_per_gap"],
                "frac_gaps_spline_better_than_linear": float(np.mean(spl < lin)),
                "frac_gaps_linear_better_than_ffill": float(np.mean(lin < ff)),
                "frac_gaps_pchip_better_than_linear": float(np.mean(pch < lin)),
            })
    _write_csv(out / f"{variable}_test_metrics.csv", test_rows)
    paired_rows = paired_comparison(t, y, split, context, spec["unit"])
    _check_against_test_metrics(paired_rows, out / f"{variable}_test_metrics.csv")
    _write_csv(out / f"{variable}_paired_bootstrap.csv", paired_rows)

    # --- 3. profile one representative configuration ---------------------------
    gaps60, _ = test_raw[60]
    prof = cProfile.Profile()
    prof.enable()
    for m in METHODS.values():
        for g in gaps60:
            reconstruct_gap(m, t, y, g)
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(15)
    (out / f"{variable}_profile.txt").write_text(
        f"cProfile: all {len(METHODS)} methods on the {len(gaps60)} test gaps of 60 min, context={context}\n\n"
        + buf.getvalue().replace(str(Path.cwd()) + "\\", "").replace(str(Path.cwd()), "."),
        encoding="utf-8")

    # --- 4. figures --------------------------------------------------------------
    _plot_error_vs_duration(test_rows, spec, fig_dir / f"{variable}_error_vs_duration.png", context)
    _plot_runtime(test_rows, fig_dir / f"{variable}_runtime.png")
    _plot_context(dev_rows, spec, fig_dir / f"{variable}_dev_context_sensitivity.png", context)
    _plot_paired(paired_rows, spec, fig_dir / f"{variable}_paired_differences.png")
    _plot_examples(t, y, test_raw, context, spec, local_dir / f"{variable}_example_reconstructions.png")

    info = {
        "variable": variable, "unit": spec["unit"], "sensor_id": sensor, "seed": SEED,
        "n_readings": int(n), "test_fraction": TEST_FRACTION,
        "dev_period_utc": [epoch_to_utc_iso(t[0]), epoch_to_utc_iso(t[split - 1])],
        "test_period_utc": [epoch_to_utc_iso(t[split]), epoch_to_utc_iso(t[-1])],
        "n_dev_readings": int(split), "n_test_readings": int(n - split),
        "durations_min": list(DURATIONS_MIN), "context_candidates": list(CONTEXT_CANDIDATES),
        **_tuning_info(spline_mean, context, dev_sets), "test_results_stale": False,
        "n_gaps_max": N_GAPS_MAX,
        "test_signal": _signal_stats(t[split:], y[split:]),
        "wall_time_s": round(time.perf_counter() - wall0, 1),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "machine": platform.processor() or platform.machine(),
    }
    (out / f"{variable}_run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def _signal_stats(t: np.ndarray, y: np.ndarray) -> dict:
    """Describe the signal, to put reconstruction errors in context.

    The typical change between consecutive ~10-min readings sets a noise
    floor: no method can be expected to beat it for a single missing reading.
    """
    regular = np.abs(np.diff(t) - 600.0) <= 30.0
    step = np.diff(y)[regular]
    return {
        "mean": float(y.mean()), "std": float(y.std()), "min": float(y.min()), "max": float(y.max()),
        "p05": float(np.quantile(y, 0.05)), "p95": float(np.quantile(y, 0.95)),
        "consecutive_step_mean_abs": float(np.mean(np.abs(step))),
        "consecutive_step_std": float(step.std()),
        "resolution": float(np.min(np.diff(np.unique(y)))),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def _plot_error_vs_duration(rows, spec, path, context):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric in zip(axes, ("mae", "rmse")):
        for m in METHODS:
            r = [x for x in rows if x["method"] == m]
            d = np.array([x["duration_min"] for x in r], float)
            v = np.array([x[metric] for x in r])
            lo = v - np.array([x[f"{metric}_ci_lo"] for x in r])
            hi = np.array([x[f"{metric}_ci_hi"] for x in r]) - v
            ax.errorbar(d, v, yerr=[lo, hi], marker="o", capsize=3, label=LABELS[m], color=COLORS[m])
        ax.set_xscale("log")
        ax.set_xticks(list(DURATIONS_MIN), [str(x) for x in DURATIONS_MIN])
        ax.set_xlabel("Gap duration (min)")
        ax.set_ylabel(f"{metric.upper()} ({spec['unit']})")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"{spec['label']} reconstruction error, test period (95 % bootstrap CI; spline context = {context})")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_runtime(rows, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
    for m in METHODS:
        r = [x for x in rows if x["method"] == m]
        ax.plot([x["duration_min"] for x in r], [x["us_per_gap"] for x in r], marker="o",
                label=LABELS[m], color=COLORS[m])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(list(DURATIONS_MIN), [str(x) for x in DURATIONS_MIN])
    ax.set_xlabel("Gap duration (min)")
    ax.set_ylabel("Median time per gap (µs)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_context(rows, spec, path, chosen):
    fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
    for d in DURATIONS_MIN:
        r = [x for x in rows if x["method"] == "cubic_spline" and x["duration_min"] == d]
        ax.plot([x["context"] for x in r], [x["mae"] for x in r], marker="o", label=f"{d} min")
    ax.axvline(chosen, color="k", ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(CONTEXT_CANDIDATES), [str(x) for x in CONTEXT_CANDIDATES])
    ax.set_xlabel("Spline context (known readings each side)")
    ax.set_ylabel(f"Spline MAE ({spec['unit']})")
    ax.set_title("Development period only; same gaps for every context")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Gap", fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_paired(rows, spec, path):
    """Paired MAE differences vs linear (1-day blocks) with 95 % and Bonferroni intervals."""
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    r1 = [r for r in rows if r["block_days"] == PAIRED_BLOCK_DAYS[0]]
    for k, (m, offset) in enumerate((("forward_fill", 0.92), ("cubic_spline", 1.0), ("pchip", 1.08))):
        r = [x for x in r1 if x["comparison"].startswith(m)]
        d = np.array([x["duration_min"] for x in r], float) * offset
        v = np.array([x["diff"] for x in r])
        ax.errorbar(d, v, yerr=[v - [x["ci_bonf_lo"] for x in r], np.array([x["ci_bonf_hi"] for x in r]) - v],
                    fmt="none", ecolor=COLORS[m], alpha=0.35, elinewidth=4)
        ax.errorbar(d, v, yerr=[v - [x["ci95_lo"] for x in r], np.array([x["ci95_hi"] for x in r]) - v],
                    marker="o", capsize=3, ls="-", color=COLORS[m], label=f"{LABELS[m]} − linear")
    ax.axhline(0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(list(DURATIONS_MIN), [str(x) for x in DURATIONS_MIN])
    ax.set_xlabel("Gap duration (min)")
    ax.set_ylabel(f"MAE difference vs linear ({spec['unit']})")
    ax.set_title(f"{spec['label']}: paired day-block bootstrap (thin: 95 %, wide: Bonferroni {CI_LEVEL_BONFERRONI:.1%})",
                 fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_examples(t, y, test_raw, context, spec, path):
    """Typical and worst-case spline gaps (raw readings; written to results/local)."""
    gaps, r = test_raw[180]
    pg = np.array(r["methods"]["cubic_spline"]["per_gap_mae"])
    picks = [("median-error gap", int(np.argsort(pg)[len(pg) // 2])), ("worst spline gap", int(np.argmax(pg)))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, (title, i) in zip(axes, picks):
        g = gaps[i]
        lo, hi = g.known.min(), g.known.max() + 1
        h = (t[lo:hi] - t[g.hidden[0]]) / 3600.0
        ax.plot(h, y[lo:hi], "k.-", lw=0.8, label="genuine readings")
        ax.plot((t[g.hidden] - t[g.hidden[0]]) / 3600.0, y[g.hidden], "o", mfc="none", color="k", label="hidden (truth)")
        # draw each reconstruction across the whole gap, boundary to boundary
        q = np.linspace(t[g.hidden[0] - 1], t[g.hidden[-1] + 1], 200)
        for m, f in METHODS.items():
            ax.plot((q - t[g.hidden[0]]) / 3600.0, f(t[g.known], y[g.known], q), color=COLORS[m], label=LABELS[m])
        ax.set_title(f"180-min {title}: {epoch_to_utc_iso(t[g.hidden[0]])}")
        ax.set_xlabel("Hours from gap start")
        ax.set_ylabel(f"{spec['label']} ({spec['unit']})")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    """Parse arguments and run the experiment."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variable", choices=sorted(VARIABLES), default="co2")
    p.add_argument("--sensor", default="6012002000869")
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--tuning-only", action="store_true",
                   help="re-run only development tuning and gap statistics; leave test outputs untouched")
    p.add_argument("--paired-only", action="store_true",
                   help="re-run only the paired day-block bootstrap on the existing test gaps")
    args = p.parse_args(argv)
    if args.tuning_only and args.paired_only:
        p.error("choose at most one of --tuning-only and --paired-only")
    runner = run_tuning if args.tuning_only else run_paired if args.paired_only else run
    info = runner(args.variable, args.sensor, args.processed, args.out)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return info


if __name__ == "__main__":
    main()
