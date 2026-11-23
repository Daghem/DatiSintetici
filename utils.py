"""Small helpers shared by the generator: RNG handling, index and periods."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["resolve_rng", "make_index", "sampling_minutes", "seasonal_periods"]


def resolve_rng(seed: int | None) -> np.random.Generator:
    """Return a numpy random generator from an optional seed."""
    return np.random.default_rng(seed)


def make_index(start: str, freq: str, n: int) -> pd.DatetimeIndex:
    """Build the datetime index of a generated series."""
    try:
        return pd.date_range(start=start, periods=n, freq=freq)
    except ValueError as exc:  # pragma: no cover - defensive, message from pandas
        raise ValueError(
            f"invalid frequency {freq!r}: {exc}. Use pandas offset aliases such as "
            "'D', 'h', '15min', 'W', 'ME', 'QS'."
        ) from exc


def sampling_minutes(freq: str) -> float | None:
    """Return the length of one sampling step in minutes.

    ``None`` is returned for calendar frequencies (month, quarter, year) whose
    duration is not constant.
    """
    offset = pd.tseries.frequencies.to_offset(freq)
    code = offset.rule_code.upper()
    try:
        return offset.nanos / 60e9
    except ValueError:
        # Non-fixed offsets (MonthEnd, QuarterEnd, YearEnd, ...) have no single
        # duration in nanoseconds.
        if code.startswith("M"):
            return None
        if code.startswith("Q"):
            return None
        if code.startswith("Y") or code.startswith("A"):
            return None
        raise


def seasonal_periods(freq: str) -> dict[str, float]:
    """Return candidate seasonal periods (in samples) for a given frequency.

    Example:
        >>> seasonal_periods("D")
        {'weekly': 7.0, 'yearly': 365.25}
        >>> seasonal_periods("h")
        {'daily': 24.0, 'weekly': 168.0, 'yearly': 8766.0}
    """
    minutes = sampling_minutes(freq)
    if minutes is None:
        code = pd.tseries.frequencies.to_offset(freq).rule_code.upper()
        if code.startswith("M"):
            return {"yearly": 12.0}
        if code.startswith("Q"):
            return {"yearly": 4.0}
        return {}

    if minutes < 60:  # sub-hourly sampling
        return {
            "hourly": 60.0 / minutes,
            "daily": 1440.0 / minutes,
            "weekly": 10080.0 / minutes,
            "yearly": 525600.0 / minutes,
        }
    if minutes < 1440:  # hourly (or coarser intra-day) sampling
        return {
            "daily": 1440.0 / minutes,
            "weekly": 10080.0 / minutes,
            "yearly": 525600.0 / minutes,
        }
    if abs(minutes - 1440.0) < 1.0:  # daily
        return {"weekly": 7.0, "yearly": 365.25}
    if abs(minutes - 10080.0) < 1.0:  # weekly
        return {"yearly": 52.18}
    if abs(minutes - 43200.0) < 1.0:  # semi-monthly
        return {"yearly": 24.0}
    return {}
