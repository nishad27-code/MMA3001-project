"""End-to-end check of the temperature tuning path (`experiment.run_tuning`) on synthetic data."""

import csv
import json

import numpy as np

from gap_reconstruction import experiment as exp


def _write_processed(path, t, y):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch_s", "time_utc", "temperature_c"])
        for a, b in zip(t, y):
            w.writerow([int(a), "", repr(float(b))])


def test_temperature_run_tuning_fair_dev_only_and_leaves_test_outputs(tmp_path):
    processed, out = tmp_path / "processed", tmp_path / "results"
    processed.mkdir()
    out.mkdir()
    # 2,000 regular 10-min readings with every 13th dropped, like the real sensor's native gaps
    t = 1.7e9 + 600.0 * np.arange(2000)
    t = np.delete(t, np.arange(5, t.size, 13))
    y = np.round((22.0 + 1.0 * np.sin((t - t[0]) / 20000.0)) * 4) / 4  # quarter-degree steps
    split = exp.chronological_split(t.size, exp.TEST_FRACTION)
    y[split:] = np.nan  # poison: any use of test-period values would break the tuning
    _write_processed(processed / "temperature_X.csv", t, y)

    sentinel = "sentinel test metrics - must not change\n"
    (out / "temperature_test_metrics.csv").write_text(sentinel, encoding="utf-8")
    (out / "temperature_run_info.json").write_text(json.dumps({"chosen_context": 2}), encoding="utf-8")

    info = exp.run_tuning("temperature", "X", processed, out)

    rows = list(csv.DictReader((out / "temperature_dev_context_sensitivity.csv").open(encoding="utf-8")))
    assert {r["context"] for r in rows} == {str(c) for c in exp.CONTEXT_CANDIDATES}
    for d in exp.DURATIONS_MIN:
        assert len({r["gap_set_hash"] for r in rows if r["duration_min"] == str(d)}) == 1
        assert len({r["n_gaps"] for r in rows if r["duration_min"] == str(d)}) == 1
    assert all(np.isfinite(float(r["mae"])) for r in rows)          # no test-period (NaN) value used
    assert info["dev_gap_set_hash_by_duration"] == {
        str(d): next(r["gap_set_hash"] for r in rows if r["duration_min"] == str(d)) for d in exp.DURATIONS_MIN}
    assert info["chosen_context"] in exp.CONTEXT_CANDIDATES
    assert info["test_results_stale"] == (info["chosen_context"] != 2)
    assert (out / "temperature_test_metrics.csv").read_text(encoding="utf-8") == sentinel
    assert (out / "figures" / "temperature_dev_context_sensitivity.png").is_file()
    stats = list(csv.DictReader((out / "temperature_gap_definition_stats.csv").open(encoding="utf-8")))
    assert {r["period"] for r in stats} == {"dev_tuning", "test"}
