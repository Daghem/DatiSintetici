"""Feature engineering helpers that turn raw series into ML-ready tables.

Naming convention
-----------------
Every engineered feature is named ``"<column>_lag<L>"`` where ``L`` is the
lag in observations:

* ``lag0`` -> value observed at time ``t`` (always available when predicting
  ``t + horizon``);
* ``lag1`` -> value observed at ``t - 1``;
* ``rollmean7`` -> rolling mean of the last 7 observations (past only).

The target of a row is the value at ``t + horizon``, so no feature ever
contains information from the future.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .utils import sampling_minutes

__all__ = [
    "add_lagged_features",
    "add_rolling_features",
    "add_time_features",
    "make_supervised",
    "make_supervised_frame",
    "make_classification",
    "time_train_test_split",
]


def _day_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    """Day of the week (Monday = 0) without relying on deprecated attributes."""
    if hasattr(index, "day_of_week"):  # pandas >= 3.0
        return index.day_of_week.to_numpy(dtype=float)
    return index.dayofweek.to_numpy(dtype=float)  # pragma: no cover - pandas 2.x


def _day_of_year(index: pd.DatetimeIndex) -> np.ndarray:
    """Day of the year (1-based) without relying on deprecated attributes."""
    if hasattr(index, "day_of_year"):  # pandas >= 3.0
        return index.day_of_year.to_numpy(dtype=float)
    return index.dayofyear.to_numpy(dtype=float)  # pragma: no cover - pandas 2.x


def _target_columns(
    frame: pd.DataFrame, columns: Sequence[str] | None
) -> list[str]:
    if columns is None:
        return list(frame.columns)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"columns not found in the frame: {missing}")
    return list(columns)


def add_lagged_features(
    frame: pd.DataFrame,
    lags: Iterable[int] = (1, 2, 3),
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Append lagged copies of ``columns`` to the frame.

    Lag 0 is accepted and means "the current value".  The usual convention for
    a forecasting task is to use lags >= 1 only.
    """
    cols = _target_columns(frame, columns)
    out = frame.copy()
    for col in cols:
        for lag in lags:
            if lag < 0:
                raise ValueError("lags must be non-negative")
            out[f"{col}_lag{lag}"] = frame[col].shift(lag)
    return out


def add_rolling_features(
    frame: pd.DataFrame,
    window: int = 7,
    stats: Sequence[str] = ("mean", "std"),
    columns: Sequence[str] | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Append rolling statistics of ``columns`` (computed on the past only)."""
    cols = _target_columns(frame, columns)
    out = frame.copy()
    min_periods = window if min_periods is None else min_periods
    for col in cols:
        rolling = frame[col].rolling(window=window, min_periods=min_periods)
        for stat in stats:
            out[f"{col}_roll{stat}{window}"] = getattr(rolling, stat)()
    return out


def add_time_features(frame: pd.DataFrame, freq: str | None = None) -> pd.DataFrame:
    """Append cyclical encodings of the calendar position of each timestamp.

    The encodings are ``sin``/``cos`` pairs, so that a model can learn the
    periodicity without extrapolating wrongly between the end and the start of
    a cycle.
    """
    out = frame.copy()
    index = out.index

    minutes = sampling_minutes(freq) if freq else None
    if minutes is not None and minutes < 1440:
        hour = index.hour.to_numpy(dtype=float)
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    dow = _day_of_week(index)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    doy = _day_of_year(index)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def make_supervised_frame(
    frame: pd.DataFrame,
    target: str,
    lags: Sequence[int] = (1, 2, 3, 7),
    rolling_window: int | None = None,
    rolling_stats: Sequence[str] = ("mean", "std"),
    horizon: int = 1,
    feature_columns: Sequence[str] | None = None,
    include_target: bool = True,
    lag_sources: bool = False,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build a supervised dataset: features at time ``t``, target at ``t+horizon``.

    Args:
        frame: The source data, one column per variable.
        target: Name of the column to predict.
        lags: Lags (in observations) used as features.
        rolling_window: Size of the rolling window, ``None`` to disable.
        rolling_stats: Statistics computed on the rolling window.
        horizon: How many steps ahead the target is.
        feature_columns: Columns used as features; by default every column
            except the target.
        include_target: When ``True`` the past values of the target itself are
            used as features (they are known at prediction time).
        lag_sources: When ``True`` the auxiliary columns also get lagged and
            rolling features; by default only their current value is used.
        dropna: Drop the rows with an incomplete history.

    Returns:
        A DataFrame with the engineered features and a ``target`` column.

    Note:
        The value observed at time ``t`` is always added as ``"<col>_lag0"``:
        it is known when the prediction for ``t + horizon`` is made.
    """
    if target not in frame.columns:
        raise KeyError(f"target column {target!r} not found in the frame")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if any(lag < 0 for lag in lags):
        raise ValueError("lags must be non-negative")

    if feature_columns is None:
        sources = [c for c in frame.columns if c != target]
    else:
        sources = _target_columns(frame, feature_columns)

    derived = list(sources)
    if include_target and target not in derived:
        derived.append(target)

    X = pd.DataFrame(index=frame.index)
    for col in derived:
        # lag 0: the value observed at time t, always available at prediction
        # time and leak-free because the target lives at t + horizon.
        X[f"{col}_lag0"] = frame[col]

    # lagged and rolling features: on the target by default, on the auxiliary
    # columns only when explicitly requested.
    lagged = list(derived) if lag_sources else ([target] if include_target else [])
    for col in lagged:
        for lag in lags:
            if lag:
                X[f"{col}_lag{lag}"] = frame[col].shift(lag)
        if rolling_window:
            rolling = frame[col].rolling(
                window=rolling_window, min_periods=rolling_window
            )
            for stat in rolling_stats:
                X[f"{col}_roll{stat}{rolling_window}"] = getattr(rolling, stat)()

    y = frame[target].shift(-horizon).rename("target")
    data = X.join(y)
    if dropna:
        data = data.dropna()
    return data


def make_supervised(
    frame: pd.DataFrame,
    target: str,
    lags: Sequence[int] = (1, 2, 3, 7),
    rolling_window: int | None = None,
    rolling_stats: Sequence[str] = ("mean", "std"),
    horizon: int = 1,
    feature_columns: Sequence[str] | None = None,
    include_target: bool = True,
    lag_sources: bool = False,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Same as :func:`make_supervised_frame` but returning ``(X, y)``."""
    data = make_supervised_frame(
        frame,
        target=target,
        lags=lags,
        rolling_window=rolling_window,
        rolling_stats=rolling_stats,
        horizon=horizon,
        feature_columns=feature_columns,
        include_target=include_target,
        lag_sources=lag_sources,
        dropna=dropna,
    )
    return data.drop(columns="target"), data["target"]


def make_classification(
    frame: pd.DataFrame,
    target: str,
    label_fn: Callable[[pd.Series], pd.Series] | None = None,
    quantile: float | None = 0.8,
    lags: Sequence[int] = (1, 2, 3, 7),
    rolling_window: int | None = None,
    rolling_stats: Sequence[str] = ("mean", "std"),
    horizon: int = 1,
    feature_columns: Sequence[str] | None = None,
    include_target: bool = True,
    lag_sources: bool = False,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a binary classification dataset.

    The label is by default ``1`` when the future value exceeds the given
    quantile of the series, ``0`` otherwise; pass a custom ``label_fn`` for a
    different rule.
    """
    if label_fn is None:
        if quantile is None:
            raise ValueError("provide either label_fn or quantile")
        threshold = frame[target].quantile(quantile)

        def label_fn(series: pd.Series) -> pd.Series:  # type: ignore[misc]
            return (series > threshold).astype(int)

    data = make_supervised_frame(
        frame,
        target=target,
        lags=lags,
        rolling_window=rolling_window,
        rolling_stats=rolling_stats,
        horizon=horizon,
        feature_columns=feature_columns,
        include_target=include_target,
        lag_sources=lag_sources,
        dropna=dropna,
    )
    labels = label_fn(frame[target].shift(-horizon)).reindex(data.index)
    data = data.dropna(subset=["target"])
    labels = labels.reindex(data.index).astype(int)
    return data.drop(columns="target"), labels


def time_train_test_split(
    frame: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the last ``test_size`` of the rows become the test set.

    No shuffling is applied, so the temporal order is preserved and the split
    is a valid proxy for training on the past and testing on the future.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    split = int(len(frame) * (1 - test_size))
    if split <= 0 or split >= len(frame):
        raise ValueError("not enough rows to perform the split")
    return frame.iloc[:split].copy(), frame.iloc[split:].copy()
