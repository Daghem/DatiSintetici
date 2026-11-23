"""Orchestration: turn a :class:`SeriesSpec` into a pandas DataFrame."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .components import (
    generate_anomalies,
    generate_latent_factor,
    generate_noise,
    generate_seasonality,
    generate_trend,
)
from .config import ChannelSpec, SeriesSpec
from .utils import make_index, resolve_rng

__all__ = ["generate_series", "generate_time_series", "summarize"]


def _build_channel(
    channel: ChannelSpec,
    n: int,
    rng: np.random.Generator,
    latent: np.ndarray,
) -> np.ndarray:
    trend = generate_trend(n, rng, channel.trend)
    season = generate_seasonality(n, rng, channel.seasonality)
    noise = generate_noise(n, rng, channel.noise)
    anomalies = (
        generate_anomalies(n, rng, channel.anomalies)
        if channel.anomalies is not None
        else np.zeros(n)
    )
    factor = channel.factor_loading * latent

    if channel.seasonality.multiplicative:
        base = (trend + noise + anomalies + factor) * (1.0 + season)
    else:
        base = trend + season + noise + anomalies + factor

    return channel.offset + channel.scale * base


def generate_series(spec: SeriesSpec) -> pd.DataFrame:
    """Generate a synthetic dataset described by ``spec``.

    Returns:
        A DataFrame indexed by a ``DatetimeIndex`` with one column per channel.
    """
    rng = resolve_rng(spec.seed)
    index = make_index(spec.start, spec.freq, spec.n)
    latent = generate_latent_factor(spec.n, rng, spec.latent_rho, spec.latent_scale)

    data = {
        channel.name: _build_channel(channel, spec.n, rng, latent)
        for channel in spec.channels
    }
    frame = pd.DataFrame(data, index=index)
    frame.index.name = "timestamp"
    return frame


def generate_time_series(
    n: int = 365,
    freq: str = "D",
    start: str = "2020-01-01",
    seed: int | None = 42,
    channels: Sequence[ChannelSpec] | None = None,
    *,
    latent_rho: float = 0.8,
    latent_scale: float = 1.0,
) -> pd.DataFrame:
    """Convenience wrapper around :func:`generate_series`.

    Example:
        >>> from dati_sintetici import ChannelSpec, TrendSpec, NoiseSpec
        >>> df = generate_time_series(
        ...     n=100,
        ...     channels=[
        ...         ChannelSpec(
        ...             name="y",
        ...             trend=TrendSpec.linear(slope=0.2),
        ...             noise=NoiseSpec.ar1(scale=0.5, rho=0.7),
        ...         )
        ...     ],
        ... )
        >>> list(df.columns)
        ['y']
    """
    if channels is None:
        channels = [ChannelSpec(name="y")]
    spec = SeriesSpec(
        n=n,
        freq=freq,
        start=start,
        seed=seed,
        channels=list(channels),
        latent_rho=latent_rho,
        latent_scale=latent_scale,
    )
    return generate_series(spec)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for every channel of the frame."""
    stats = frame.describe().T
    stats["range"] = stats["max"] - stats["min"]
    stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)
    return stats
