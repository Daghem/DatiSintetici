"""dati_sintetici — synthetic time-series generation for machine learning.

Quick start:
    >>> from dati_sintetici import generate_time_series, ChannelSpec, TrendSpec
    >>> df = generate_time_series(n=100, channels=[
    ...     ChannelSpec(name="y", trend=TrendSpec.linear(slope=0.5)),
    ... ])
    >>> df.shape
    (100, 1)
"""

from __future__ import annotations

from .config import (
    AnomalySpec,
    ChannelSpec,
    NoiseSpec,
    SeasonalitySpec,
    SeriesSpec,
    TrendSpec,
)
from .features import (
    add_lagged_features,
    add_rolling_features,
    add_time_features,
    make_classification,
    make_supervised,
    make_supervised_frame,
    time_train_test_split,
)
from .generator import generate_series, generate_time_series, summarize
from .presets import energy_demand, iot_sensor, stock_price, web_traffic
from .utils import make_index, resolve_rng, seasonal_periods

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # configuration
    "SeriesSpec",
    "ChannelSpec",
    "TrendSpec",
    "SeasonalitySpec",
    "NoiseSpec",
    "AnomalySpec",
    # generation
    "generate_series",
    "generate_time_series",
    "summarize",
    # presets
    "energy_demand",
    "iot_sensor",
    "web_traffic",
    "stock_price",
    # feature engineering / ML
    "add_lagged_features",
    "add_rolling_features",
    "add_time_features",
    "make_supervised",
    "make_supervised_frame",
    "make_classification",
    "time_train_test_split",
    # utilities
    "seasonal_periods",
    "make_index",
    "resolve_rng",
]
