"""Ready-to-use channel presets.

The presets are ordinary :class:`ChannelSpec` objects: they are a convenience
layer, and can be freely modified before being passed to the generator.
"""

from __future__ import annotations

from .config import AnomalySpec, ChannelSpec, NoiseSpec, SeasonalitySpec, TrendSpec
from .utils import seasonal_periods

__all__ = ["energy_demand", "iot_sensor", "web_traffic", "stock_price"]


def _available(freq: str) -> dict[str, float]:
    return seasonal_periods(freq)


def energy_demand(
    name: str = "demand",
    freq: str = "h",
    level: float = 100.0,
    trend_slope: float = 0.05,
    noise_scale: float = 8.0,
    anomaly_rate: float = 0.005,
) -> ChannelSpec:
    """Electricity demand: daily + weekly seasonality, smooth drift, AR noise."""
    periods = _available(freq)
    seas_periods = [p for name_, p in periods.items() if name_ in ("daily", "weekly")]
    amplitudes = [1.0] * len(seas_periods)
    if len(seas_periods) == 2:
        amplitudes = [8.0, 15.0]  # the weekly cycle dominates the daily one
    return ChannelSpec(
        name=name,
        trend=TrendSpec.linear(slope=trend_slope, intercept=level),
        seasonality=SeasonalitySpec.seasonal(seas_periods, amplitudes),
        noise=NoiseSpec.ar1(scale=noise_scale, rho=0.85),
        anomalies=AnomalySpec(kind="spikes", rate=anomaly_rate, magnitude=40.0),
    )


def iot_sensor(
    name: str = "sensor",
    freq: str = "5min",
    level: float = 20.0,
    noise_scale: float = 1.5,
    drift: float = 6.0,
) -> ChannelSpec:
    """A sensor with a strong daily cycle and a slow drift (sensor ageing)."""
    periods = _available(freq)
    daily = periods.get("daily")
    seas_periods = [daily] if daily else []
    return ChannelSpec(
        name=name,
        trend=TrendSpec.linear(slope=0.0, intercept=level),
        seasonality=SeasonalitySpec.seasonal(seas_periods, [3.0]),
        noise=NoiseSpec.gaussian(scale=noise_scale),
        anomalies=AnomalySpec(kind="drift", magnitude=drift),
    )


def web_traffic(
    name: str = "visits",
    freq: str = "h",
    level: float = 500.0,
    trend_slope: float = 0.2,
    noise_scale: float = 40.0,
    anomaly_rate: float = 0.01,
) -> ChannelSpec:
    """Web traffic: strong weekly cycle, marketing-like spikes."""
    periods = _available(freq)
    weekly = periods.get("weekly")
    daily = periods.get("daily")
    seas_periods = [p for p in (daily, weekly) if p]
    amplitudes = [60.0 if p == daily else 200.0 for p in seas_periods]
    return ChannelSpec(
        name=name,
        trend=TrendSpec.linear(slope=trend_slope, intercept=level),
        seasonality=SeasonalitySpec.seasonal(seas_periods, amplitudes),
        noise=NoiseSpec.ar1(scale=noise_scale, rho=0.6),
        anomalies=AnomalySpec(kind="spikes", rate=anomaly_rate, magnitude=300.0),
    )


def stock_price(
    name: str = "price",
    freq: str = "D",
    start_price: float = 100.0,
    noise_scale: float = 1.5,
) -> ChannelSpec:
    """A price-like series: random walk with a mild upward drift."""
    return ChannelSpec(
        name=name,
        trend=TrendSpec.linear(slope=0.0, intercept=start_price),
        seasonality=SeasonalitySpec(kind="none"),
        noise=NoiseSpec(kind="random_walk", scale=noise_scale),
    )
