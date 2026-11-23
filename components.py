"""Building blocks of the generator: trend, seasonality, noise, anomalies.

Every function takes the number of observations, a :class:`numpy.random.Generator`
and the relevant spec object, and returns a 1-D ``float64`` array.
"""

from __future__ import annotations

import numpy as np

from .config import AnomalySpec, NoiseSpec, SeasonalitySpec, TrendSpec

__all__ = [
    "generate_trend",
    "generate_seasonality",
    "generate_noise",
    "generate_anomalies",
    "generate_latent_factor",
]


def _time_grid(n: int) -> np.ndarray:
    """Normalised time axis in ``[0, 1]``."""
    if n <= 1:
        return np.zeros(n)
    return np.linspace(0.0, 1.0, n)


def generate_trend(n: int, rng: np.random.Generator, spec: TrendSpec) -> np.ndarray:
    """Build the trend component."""
    t = _time_grid(n)
    kind = spec.kind

    if kind == "none":
        return np.zeros(n)
    if kind == "linear":
        return spec.intercept + spec.slope * t
    if kind == "poly":
        return spec.intercept + spec.slope * t**spec.degree
    if kind == "exponential":
        return spec.base * np.exp(spec.slope * t)
    if kind == "logarithmic":
        return spec.intercept + spec.slope * np.log1p(spec.log_scale * t)
    if kind == "sigmoid":
        z = spec.steepness * (t - spec.center)
        # numerically stable logistic
        return spec.amplitude / (1.0 + np.exp(-np.clip(z, -500, 500)))
    raise ValueError(f"unsupported trend kind: {kind!r}")  # pragma: no cover


def generate_seasonality(
    n: int, rng: np.random.Generator, spec: SeasonalitySpec
) -> np.ndarray:
    """Build the seasonal component.

    The result is an *amplitude* signal centred on zero; it is added to the
    series (or used as a multiplicative factor when
    :attr:`SeasonalitySpec.multiplicative` is set).
    """
    if spec.kind == "none" or not len(spec.periods):
        return np.zeros(n)

    periods = np.atleast_1d(np.asarray(spec.periods, dtype=float))
    amplitudes = np.atleast_1d(np.asarray(spec.amplitude, dtype=float))
    if amplitudes.size == 1:
        amplitudes = np.repeat(amplitudes, periods.size)
    if amplitudes.size != periods.size:
        raise ValueError(
            "amplitude must be a scalar or have one value per period "
            f"(got {amplitudes.size} for {periods.size} periods)"
        )

    if spec.phase is None:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=periods.size)
    else:
        phases = np.atleast_1d(np.asarray(spec.phase, dtype=float))
        if phases.size == 1:
            phases = np.repeat(phases, periods.size)
        if phases.size != periods.size:
            raise ValueError("phase must be a scalar or have one value per period")

    t = np.arange(n, dtype=float)
    out = np.zeros(n)

    for period, amplitude, phase in zip(periods, amplitudes, phases):
        if spec.kind == "fourier":
            for h in range(1, spec.harmonics + 1):
                decay = 1.0 / h
                out += (
                    amplitude
                    * decay
                    * np.sin(2.0 * np.pi * h * t / period + phase + (h - 1) * np.pi / 2.0)
                )
        elif spec.kind == "square":
            out += amplitude * np.sign(np.sin(2.0 * np.pi * t / period + phase))
        else:  # sinus
            out += amplitude * np.sin(2.0 * np.pi * t / period + phase)

    return out


def _ar1(n: int, rng: np.random.Generator, rho: float, scale: float) -> np.ndarray:
    """Autoregressive process of order 1 with unit marginal variance."""
    innovations = rng.normal(0.0, scale, size=n)
    out = np.empty(n, dtype=float)
    boost = 1.0 / np.sqrt(max(1.0 - rho * rho, 1e-12))
    out[0] = innovations[0] * boost
    for i in range(1, n):
        out[i] = rho * out[i - 1] + innovations[i]
    return out


def generate_noise(n: int, rng: np.random.Generator, spec: NoiseSpec) -> np.ndarray:
    """Build the random component."""
    kind = spec.kind

    if kind == "none" or spec.scale == 0:
        return np.zeros(n)
    if kind == "gaussian":
        return rng.normal(spec.loc, spec.scale, size=n)
    if kind == "uniform":
        lower = spec.lower if spec.lower is not None else spec.loc - spec.scale
        upper = spec.upper if spec.upper is not None else spec.loc + spec.scale
        return rng.uniform(lower, upper, size=n)
    if kind == "student_t":
        return spec.loc + spec.scale * rng.standard_t(spec.df, size=n)
    if kind == "random_walk":
        steps = rng.normal(spec.loc, spec.scale, size=n)
        return np.cumsum(steps)
    if kind == "ar1":
        return _ar1(n, rng, spec.rho, spec.scale)
    raise ValueError(f"unsupported noise kind: {kind!r}")  # pragma: no cover


def generate_anomalies(
    n: int, rng: np.random.Generator, spec: AnomalySpec
) -> np.ndarray:
    """Inject atypical events (spikes, level shifts, drifts) into the series."""
    out = np.zeros(n)

    if spec.kind == "spikes":
        k = int(round(spec.rate * n))
        if k <= 0:
            return out
        k = min(k, n)
        idx = rng.choice(n, size=k, replace=False)
        signs = rng.choice([-1.0, 1.0], size=k) if spec.symmetric else np.ones(k)
        out[idx] += spec.magnitude * signs
    elif spec.kind == "shift":
        start = int(rng.integers(0, n))
        out[start:] += spec.magnitude
    elif spec.kind == "drift":
        start = int(rng.integers(0, n))
        ramp = np.linspace(0.0, 1.0, n - start)
        out[start:] += spec.magnitude * ramp
    else:  # pragma: no cover - guarded by the dataclass validation
        raise ValueError(f"unsupported anomaly kind: {spec.kind!r}")

    return out


def generate_latent_factor(
    n: int, rng: np.random.Generator, rho: float, scale: float
) -> np.ndarray:
    """Shared AR(1) factor used to correlate the different channels."""
    if scale == 0:
        return np.zeros(n)
    return _ar1(n, rng, rho, scale)
