"""Command-line step 1: extract genuine readings for one sensor.

Run from the repository root::

    python -m gap_reconstruction.extract

This makes a single streaming pass over the raw environmental CSV. It writes:

- ``data/processed/co2_<sensor>.csv`` with columns ``epoch_s,time_utc,co2_ppm``
- ``data/processed/temperature_<sensor>.csv`` with columns ``epoch_s,time_utc,temperature_c``
- ``data/processed/extraction_summary_<sensor>.json`` with extraction counts and cadence checks

Later steps read these small files, so the 353 MB source is not re-scanned.
Processed files are git-ignored because the dataset's redistribution terms
are unconfirmed.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .loader import SeriesExtractor, epoch_to_utc_iso, interval_summary, iter_sensor_rows

DEFAULT_SENSOR = "6012002000869"
DEFAULT_INPUT = Path(
    "data/building_sensors/MMA3001 Dataset - MSI Occupancy and Environmental/"
    "5EnvSensor_MayToDec2024_180kRows/5EnvSensor_MayToDec2024_180kRows.csv"
)
DEFAULT_OUT = Path("data/processed")


def write_series(path: Path, epoch_s: np.ndarray, values: np.ndarray, column: str) -> None:
    """Write a time series as CSV with epoch seconds, ISO UTC time and value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch_s", "time_utc", column])
        for t, v in zip(epoch_s.tolist(), values.tolist()):
            w.writerow([t, epoch_to_utc_iso(t), repr(v)])


def main(argv: list[str] | None = None) -> dict:
    """Run the extraction and return the summary dictionary."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="raw environmental CSV")
    p.add_argument("--sensor", default=DEFAULT_SENSOR, help="sensor id (exact string)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    co2 = SeriesExtractor("Carbon dioxide", "ppm")
    temp = SeriesExtractor("Temperature", "°C")
    t0 = time.perf_counter()
    for js in iter_sensor_rows(args.input, args.sensor):
        co2.add(js)
        temp.add(js)
    elapsed = time.perf_counter() - t0

    co2_s, temp_s = co2.result(), temp.result()
    if co2_s.epoch_s.size < 2:
        raise SystemExit(f"Sensor {args.sensor}: fewer than two valid CO2 readings; nothing to analyse.")

    summary = {
        "sensor_id": args.sensor,
        "source_file": str(args.input).replace("\\", "/"),
        "scan_seconds": round(elapsed, 1),
        "co2": {**co2_s.stats, "cadence": interval_summary(co2_s.epoch_s)},
        "temperature": {**temp_s.stats,
                        "cadence": interval_summary(temp_s.epoch_s) if temp_s.epoch_s.size >= 2 else None},
        "temperature_timestamps_match_co2": bool(np.array_equal(co2_s.epoch_s, temp_s.epoch_s)),
    }
    tag = args.sensor
    write_series(args.out_dir / f"co2_{tag}.csv", co2_s.epoch_s, co2_s.values, "co2_ppm")
    write_series(args.out_dir / f"temperature_{tag}.csv", temp_s.epoch_s, temp_s.values, "temperature_c")
    (args.out_dir / f"extraction_summary_{tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
