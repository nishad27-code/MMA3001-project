"""Tests for gap_reconstruction.loader using small fixtures in the real CSV format."""

import csv
import json

import numpy as np
import pytest

from gap_reconstruction.loader import (
    SeriesExtractor,
    epoch_to_utc_iso,
    extract_series,
    interval_summary,
    iter_sensor_rows,
    parse_variable,
)

HEADER = ["id", "sensorid", "devicetype", "jsondata", "status",
          "processing_errors", "createdate", "processdate"]


def js(time=1709186696, value=465, unit="ppm", name="Carbon dioxide", extra=None):
    """Build a jsondata string shaped like the real export."""
    sample = {"time": time}
    if value is not None:
        sample["value"] = value
    if extra:
        sample.update(extra)
    return json.dumps([
        {"values": [{"time": time, "value": 23.8}],
         "variable": {"id": 84, "name": "Temperature", "unit": "°C"}},
        {"values": [sample], "variable": {"id": 67, "name": name, "unit": unit}},
    ])


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_NONNUMERIC)  # matches the "" escaping of the export
        w.writerow(HEADER)
        for i, (sensor, j) in enumerate(rows):
            w.writerow([i, sensor, "env", j, "Processed", "NULL",
                        "2024-02-29 17:49:25.51818+11", "2024-02-29 17:49:26+11"])


# --- parse_variable -------------------------------------------------------

def test_parse_valid_reading():
    assert parse_variable(js(), "Carbon dioxide", "ppm", (250, 5000)) == (1709186696, 465.0, None)


@pytest.mark.parametrize("jsondata, reason", [
    ("not json", "bad_json"),
    ('{"a": 1}', "bad_json"),
    (js(name="Humidity"), "variable_absent"),
    (js(unit="mg/m3"), "unit_mismatch"),
    (json.dumps([{"values": [], "variable": {"name": "Carbon dioxide", "unit": "ppm"}}]), "no_values"),
    (js(value=None, extra={"errorCode": 3}), "error_code"),
    (js(value=None), "missing_value"),
    (js(value="465"), "non_numeric"),
    (js(value=True), "non_numeric"),
    (js(value=float("nan")), "non_finite"),
    (js(time="yesterday"), "bad_timestamp"),
    (js(time=0), "bad_timestamp"),
    (js(value=12), "out_of_range"),
    (js(value=99999), "out_of_range"),
])
def test_parse_rejections(jsondata, reason):
    t, v, r = parse_variable(jsondata, "Carbon dioxide", "ppm", (250, 5000))
    assert (t, v, r) == (None, None, reason)


def test_temperature_unit_with_degree_sign():
    assert parse_variable(js(), "Temperature", "°C") == (1709186696, 23.8, None)


# --- SeriesExtractor: stale repeats, duplicates, ordering ----------------------

def test_stale_repeats_are_dropped_and_counted():
    rows = [js(100, 450), js(100, 450), js(100, 450), js(700, 460), js(700, 460)]
    s = extract_series(rows, "Carbon dioxide", "ppm")
    assert s.epoch_s.tolist() == [100, 700]
    assert s.values.tolist() == [450.0, 460.0]
    assert s.stats["stale_repeats"] == 3
    assert s.stats["source_rows"] == 5
    assert s.stats["valid_unique_readings"] == 2


def test_conflicting_duplicate_keeps_first_and_counts():
    s = extract_series([js(100, 450), js(100, 999)], "Carbon dioxide", "ppm")
    assert s.values.tolist() == [450.0]
    assert s.stats["conflicting_duplicates"] == 1
    assert s.stats["stale_repeats"] == 0


def test_out_of_order_readings_are_sorted():
    s = extract_series([js(1300, 3.0e2), js(100, 4.0e2), js(700, 5.0e2)], "Carbon dioxide", "ppm")
    assert s.epoch_s.tolist() == [100, 700, 1300]
    assert s.values.tolist() == [400.0, 500.0, 300.0]
    assert s.stats["out_of_order_new_readings"] == 2


def test_rejections_counted_by_reason():
    s = extract_series([js(100, 450), js(200, 10), js(300, None)], "Carbon dioxide", "ppm")
    assert s.stats["rejected"] == {"out_of_range": 1, "missing_value": 1}
    assert s.epoch_s.size == 1


def test_default_plausible_range_applies():
    assert SeriesExtractor("Carbon dioxide", "ppm").value_range == (250.0, 5000.0)


# --- iter_sensor_rows --------------------------------------------------------

def test_iter_sensor_rows_filters_sensor(tmp_path):
    p = tmp_path / "env.csv"
    write_csv(p, [("6012002000869", js(100)), ("6012002000241", js(200)), ("6012002000869", js(700))])
    rows = list(iter_sensor_rows(p, "6012002000869"))
    assert len(rows) == 2
    assert [parse_variable(r, "Carbon dioxide", "ppm")[0] for r in rows] == [100, 700]


def test_iter_sensor_rows_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_sensor_rows(tmp_path / "nope.csv", "x"))


def test_iter_sensor_rows_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("id,sensorid\n1,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsondata"):
        list(iter_sensor_rows(p, "x"))


# --- timestamps and cadence -------------------------------------------------

def test_epoch_to_utc_iso():
    assert epoch_to_utc_iso(1709186696) == "2024-02-29T06:04:56Z"
    assert epoch_to_utc_iso(0) == "1970-01-01T00:00:00Z"


def test_interval_summary_detects_known_gap():
    t = np.array([0, 600, 1200, 1800, 3000, 3600])  # one reading missing at 2400
    s = interval_summary(t)
    assert s["median_interval_s"] == 600
    assert s["native_gaps"] == 1
    assert s["longest_native_gaps"][0]["interval_s"] == 1200
    assert s["coverage_fraction"] == pytest.approx(6 / 7)


@pytest.mark.parametrize("t", [np.array([5]), np.array([0, 600, 600]), np.array([0, 600, 300])])
def test_interval_summary_rejects_bad_input(t):
    with pytest.raises(ValueError):
        interval_summary(t)
