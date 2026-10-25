"""Streaming extraction of one variable from the MSI environmental sensor CSV.

Raw file format
---------------
The environmental export is a CSV with columns::

    id, sensorid, devicetype, jsondata, status, processing_errors,
    createdate, processdate

``jsondata`` holds a JSON list with one entry per measured variable::

    [{"values": [{"time": 1709186696, "value": 465}],
      "variable": {"id": 67, "name": "Carbon dioxide", "unit": "ppm", ...}},
     ...]

``time`` is the *measurement* time in Unix seconds (UTC). ``createdate`` is
the time the server stored the row. The server keeps re-storing the last
packet it received when a sensor goes quiet, so many rows are stale copies.
A reading counts as a genuinely new observation only when its own
measurement ``time`` has not been seen before for that sensor and variable.

Using the JSON ``time`` also avoids the mixed ``+10``/``+11`` (daylight
saving) offsets in ``createdate``.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

#: Columns that must be present in the raw environmental CSV.
REQUIRED_COLUMNS = ("sensorid", "jsondata")

#: Physically plausible ranges used to reject corrupt values. CO2: outdoor
#: air is about 420 ppm and the sensor's reported range is 400 ppm upward,
#: so values below 250 ppm or above 5000 ppm (the usual NDIR full scale) are
#: treated as invalid. Temperature: indoor conditions.
PLAUSIBLE_RANGE = {
    "Carbon dioxide": (250.0, 5000.0),
    "Temperature": (-10.0, 50.0),
}

#: Rejection reasons reported by `parse_variable`.
REJECTION_REASONS = (
    "bad_json",
    "variable_absent",
    "unit_mismatch",
    "no_values",
    "error_code",
    "missing_value",
    "non_numeric",
    "non_finite",
    "bad_timestamp",
    "out_of_range",
)


def iter_sensor_rows(path: str | Path, sensor_id: str) -> Iterator[str]:
    """Stream the ``jsondata`` field of every row belonging to one sensor.

    The file is read row by row, so memory use does not depend on file size.
    Rows are filtered on the ``sensorid`` string *before* any JSON parsing,
    which keeps a full pass over the 353 MB file to well under a minute.

    Parameters
    ----------
    path : str or Path
        Path to the raw environmental CSV.
    sensor_id : str
        Exact sensor identifier, e.g. ``"6012002000869"``.

    Yields
    ------
    str
        The raw ``jsondata`` text of each matching row, in file order.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the header lacks a required column.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Environmental CSV not found: {path}")
    # jsondata fields are several kB long; the csv default limit is 128 kB,
    # so raise it to be safe against unusually long rows.
    csv.field_size_limit(2**31 - 1)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV file: {path}")
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"CSV is missing required column(s): {missing}")
        i_sensor = header.index("sensorid")
        i_json = header.index("jsondata")
        for row in reader:
            if len(row) > max(i_sensor, i_json) and row[i_sensor] == sensor_id:
                yield row[i_json]


def parse_variable(
    jsondata: str,
    name: str,
    expected_unit: str,
    value_range: tuple[float, float] | None = None,
) -> tuple[int | None, float | None, str | None]:
    """Extract one variable's measurement time and value from a ``jsondata`` field.

    Parameters
    ----------
    jsondata : str
        JSON text from one raw row.
    name : str
        Variable name as it appears in the JSON, e.g. ``"Carbon dioxide"``.
    expected_unit : str
        Required unit string, e.g. ``"ppm"``. A different unit is rejected
        rather than silently mixed in.
    value_range : (float, float), optional
        Inclusive plausible range. Values outside it are rejected.

    Returns
    -------
    (time, value, reason) : tuple
        ``(epoch_seconds, value, None)`` for a valid reading, otherwise
        ``(None, None, reason)`` with ``reason`` taken from `REJECTION_REASONS`.
    """
    try:
        entries = json.loads(jsondata)
    except (json.JSONDecodeError, TypeError):
        return None, None, "bad_json"
    if not isinstance(entries, list):
        return None, None, "bad_json"

    entry = next(
        (e for e in entries
         if isinstance(e, dict) and isinstance(e.get("variable"), dict)
         and e["variable"].get("name") == name),
        None,
    )
    if entry is None:
        return None, None, "variable_absent"
    if entry["variable"].get("unit") != expected_unit:
        return None, None, "unit_mismatch"

    values = entry.get("values")
    if not isinstance(values, list) or not values or not isinstance(values[0], dict):
        return None, None, "no_values"
    sample = values[0]
    if "errorCode" in sample:
        return None, None, "error_code"
    if "value" not in sample:
        return None, None, "missing_value"

    value = sample["value"]
    # bool is a subclass of int in Python, so exclude it explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, None, "non_numeric"
    value = float(value)
    if not math.isfinite(value):
        return None, None, "non_finite"

    t = sample.get("time")
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0:
        return None, None, "bad_timestamp"

    if value_range is not None and not (value_range[0] <= value <= value_range[1]):
        return None, None, "out_of_range"
    return int(t), value, None


@dataclass
class SeriesExtractor:
    """Accumulate genuine readings of one variable, removing stale repeats.

    Feed raw ``jsondata`` strings with `add`, then call `result`. Several
    extractors can share one pass over the file.

    Duplicate measurement timestamps are handled as follows:

    - **stale repeat**: same timestamp and same value as an earlier row. This
      is the server re-storing an old packet; it is dropped and counted.
    - **conflicting duplicate**: same timestamp but a different value. The
      first value is kept and the conflict is counted, because neither copy
      can be shown to be correct.

    Parameters
    ----------
    name : str
        Variable name in the JSON.
    unit : str
        Expected unit.
    value_range : (float, float), optional
        Plausible range; defaults to `PLAUSIBLE_RANGE` for ``name`` if present.
    """

    name: str
    unit: str
    value_range: tuple[float, float] | None = None
    _seen: dict[int, float] = field(default_factory=dict, repr=False)
    _latest_time: int | None = field(default=None, repr=False)
    source_rows: int = 0
    stale_repeats: int = 0
    conflicting_duplicates: int = 0
    out_of_order: int = 0
    rejected: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        if self.value_range is None:
            self.value_range = PLAUSIBLE_RANGE.get(self.name)

    def add(self, jsondata: str) -> None:
        """Process one raw ``jsondata`` string."""
        self.source_rows += 1
        t, v, reason = parse_variable(jsondata, self.name, self.unit, self.value_range)
        if reason is not None:
            self.rejected[reason] += 1
            return
        if t in self._seen:
            if self._seen[t] == v:
                self.stale_repeats += 1
            else:
                self.conflicting_duplicates += 1
            return
        # Out of order = earlier than the latest measurement time seen so far.
        if self._latest_time is not None and t < self._latest_time:
            self.out_of_order += 1
        self._seen[t] = v
        if self._latest_time is None or t > self._latest_time:
            self._latest_time = t

    def result(self) -> "ExtractedSeries":
        """Return the genuine readings sorted by measurement time."""
        times = np.array(sorted(self._seen), dtype=np.int64)
        values = np.array([self._seen[t] for t in times.tolist()], dtype=float)
        stats = {
            "variable": self.name,
            "unit": self.unit,
            "plausible_range": list(self.value_range) if self.value_range else None,
            "source_rows": self.source_rows,
            "valid_unique_readings": int(times.size),
            "stale_repeats": self.stale_repeats,
            "conflicting_duplicates": self.conflicting_duplicates,
            "out_of_order_new_readings": self.out_of_order,
            "rejected": dict(self.rejected),
        }
        return ExtractedSeries(times, values, stats)


@dataclass
class ExtractedSeries:
    """A cleaned time series with its extraction statistics.

    Attributes
    ----------
    epoch_s : numpy.ndarray of int64
        Strictly increasing measurement times, Unix seconds (UTC).
    values : numpy.ndarray of float
        Measured values in the variable's unit.
    stats : dict
        Counts of source rows, valid readings, stale repeats, conflicts and
        rejections by reason.
    """

    epoch_s: np.ndarray
    values: np.ndarray
    stats: dict


def extract_series(
    jsondata_rows: Iterable[str],
    name: str,
    unit: str,
    value_range: tuple[float, float] | None = None,
) -> ExtractedSeries:
    """Convenience wrapper: run one `SeriesExtractor` over an iterable of rows."""
    ex = SeriesExtractor(name, unit, value_range)
    for js in jsondata_rows:
        ex.add(js)
    return ex.result()


def epoch_to_utc_iso(epoch_s: int | float) -> str:
    """Format Unix seconds as an ISO-8601 UTC string, e.g. ``2024-02-29T06:04:56Z``."""
    return dt.datetime.fromtimestamp(int(epoch_s), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def interval_summary(epoch_s: np.ndarray, gap_factor: float = 1.5, top: int = 8) -> dict:
    """Describe sampling cadence and coverage of a time series from the data itself.

    Parameters
    ----------
    epoch_s : numpy.ndarray
        Strictly increasing timestamps in seconds; at least two values.
    gap_factor : float
        An interval longer than ``gap_factor`` times the median interval is
        reported as a native (pre-existing) gap.
    top : int
        Number of most common intervals and longest gaps to list.

    Returns
    -------
    dict
        Start/end, median interval, most common intervals, fraction of
        intervals within 5 % of the median, number and total duration of
        native gaps, the longest gaps, and coverage = readings / expected
        readings at the median cadence.

    Raises
    ------
    ValueError
        If fewer than two timestamps are given or they are not strictly increasing.
    """
    t = np.asarray(epoch_s, dtype=np.int64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("interval_summary needs at least two timestamps")
    d = np.diff(t)
    if np.any(d <= 0):
        raise ValueError("timestamps must be strictly increasing")
    med = float(np.median(d))
    gap_mask = d > gap_factor * med
    gap_idx = np.flatnonzero(gap_mask)
    longest = gap_idx[np.argsort(d[gap_idx])[::-1][:top]]
    span = float(t[-1] - t[0])
    vals, counts = np.unique(d, return_counts=True)
    order = np.argsort(counts)[::-1][:top]
    return {
        "n_readings": int(t.size),
        "start_utc": epoch_to_utc_iso(t[0]),
        "end_utc": epoch_to_utc_iso(t[-1]),
        "span_days": span / 86400.0,
        "median_interval_s": med,
        "most_common_intervals_s": {int(vals[i]): int(counts[i]) for i in order},
        "fraction_within_5pct_of_median": float(np.mean(np.abs(d - med) <= 0.05 * med)),
        "native_gaps": int(gap_mask.sum()),
        "native_gap_total_hours": float((d[gap_mask] - med).sum() / 3600.0),
        "longest_native_gaps": [
            {"from_utc": epoch_to_utc_iso(t[i]), "to_utc": epoch_to_utc_iso(t[i + 1]),
             "interval_s": int(d[i])}
            for i in longest
        ],
        "coverage_fraction": float(t.size / (span / med + 1.0)),
    }
