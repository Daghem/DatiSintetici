"""Declarative configuration objects for the synthetic time-series generator.

Every component of a channel (trend, seasonality, noise, anomalies) is described
by a small dataclass.  The dataclasses are pure data: they contain no logic, so
they can be built programmatically, stored in JSON/YAML, or reused across
experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

TrendKind = Literal["none", "linear", "poly", "exponential", "logarithmic", "sigmoid"]
SeasonKind = Literal["none", "sinus", "fourier", "square"]
NoiseKind = Literal["none", "gaussian", "uniform", "student_t", "random_walk", "ar1"]
AnomalyKind = Literal["spikes", "shift", "drift"]

_TREND_KINDS: frozenset[str] = frozenset(
    {"none", "linear", "poly", "exponential", "logarithmic", "sigmoid"}
)
_SEASON_KINDS: frozenset[str] = frozenset({"none", "sinus", "fourier", "square"})
_NOISE_KINDS: frozenset[str] = frozenset(
    {"none", "gaussian", "uniform", "student_t", "random_walk", "ar1"}
)
_ANOMALY_KINDS: frozenset[str] = frozenset({"spikes", "shift", "drift"})


@dataclass
class TrendSpec:
    """Long-run evolution of the series.

    The time axis is normalised to ``t in [0, 1]`` so that the same
    specification behaves identically whatever the length of the series.

    Attributes:
        kind: Shape of the trend.
        intercept: Constant offset added to the trend.
        slope: Slope for ``linear``/``poly``/``logarithmic``, growth rate for
            ``exponential``.
        degree: Degree of the polynomial (``poly`` only).
        amplitude: Height of the sigmoid (``sigmoid`` only).
        center: Position of the sigmoid inflection point in ``[0, 1]``.
        steepness: Steepness of the sigmoid.
        base: Base of the exponential.
        log_scale: Inner scale of the logarithm, i.e. ``ln(1 + log_scale * t)``.
    """

    kind: TrendKind = "none"
    intercept: float = 0.0
    slope: float = 1.0
    degree: int = 2
    amplitude: float = 1.0
    center: float = 0.5
    steepness: float = 10.0
    base: float = 1.0
    log_scale: float = 10.0

    def __post_init__(self) -> None:
        if self.kind not in _TREND_KINDS:
            raise ValueError(
                f"unknown trend kind {self.kind!r}; expected one of {sorted(_TREND_KINDS)}"
            )
        if self.kind == "poly" and self.degree < 1:
            raise ValueError("poly trend requires degree >= 1")
        if self.kind == "sigmoid" and self.steepness <= 0:
            raise ValueError("sigmoid trend requires steepness > 0")

    @classmethod
    def linear(cls, slope: float = 1.0, intercept: float = 0.0) -> "TrendSpec":
        return cls(kind="linear", slope=slope, intercept=intercept)

    @classmethod
    def poly(cls, slope: float = 1.0, degree: int = 2, intercept: float = 0.0) -> "TrendSpec":
        return cls(kind="poly", slope=slope, degree=degree, intercept=intercept)

    @classmethod
    def exponential(cls, slope: float = 0.5, base: float = 1.0) -> "TrendSpec":
        return cls(kind="exponential", slope=slope, base=base)

    @classmethod
    def logistic(
        cls, amplitude: float = 1.0, center: float = 0.5, steepness: float = 10.0
    ) -> "TrendSpec":
        return cls(kind="sigmoid", amplitude=amplitude, center=center, steepness=steepness)


@dataclass
class SeasonalitySpec:
    """Periodic component of the series.

    ``periods`` are expressed in **number of observations** (7 means "one week"
    for a daily series, 24 means "one day" for an hourly one).  Use
    :func:`dati_sintetici.utils.seasonal_periods` to get sensible candidates
    for a given frequency.

    Attributes:
        kind: Shape of the periodic signal.
        periods: Period lengths in samples.
        amplitude: Amplitude of each period (scalar or one value per period).
        phase: Phase in radians for each period (random if ``None``).
        harmonics: Number of harmonics for the ``fourier`` kind.
        multiplicative: When ``True`` the seasonal component multiplies the
            level of the series instead of being added to it.
    """

    kind: SeasonKind = "none"
    periods: Sequence[float] = ()
    amplitude: float | Sequence[float] = 1.0
    phase: Sequence[float] | None = None
    harmonics: int = 2
    multiplicative: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _SEASON_KINDS:
            raise ValueError(
                f"unknown seasonality kind {self.kind!r}; expected one of {sorted(_SEASON_KINDS)}"
            )
        if any(p <= 0 for p in self.periods):
            raise ValueError("seasonal periods must be strictly positive")
        if self.harmonics < 1:
            raise ValueError("harmonics must be >= 1")

    @classmethod
    def seasonal(
        cls,
        periods: Sequence[float],
        amplitude: float | Sequence[float] = 1.0,
        kind: SeasonKind = "sinus",
    ) -> "SeasonalitySpec":
        return cls(kind=kind, periods=tuple(periods), amplitude=amplitude)


@dataclass
class NoiseSpec:
    """Random component of the series.

    Attributes:
        kind: Distribution of the noise.
        scale: Standard deviation (or half-width for ``uniform``).
        loc: Center of the distribution.
        rho: Autocorrelation coefficient for ``ar1``.
        df: Degrees of freedom for ``student_t``.
        lower: Explicit lower bound for ``uniform``.
        upper: Explicit upper bound for ``uniform``.
    """

    kind: NoiseKind = "none"
    scale: float = 1.0
    loc: float = 0.0
    rho: float = 0.8
    df: float = 5.0
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in _NOISE_KINDS:
            raise ValueError(
                f"unknown noise kind {self.kind!r}; expected one of {sorted(_NOISE_KINDS)}"
            )
        if self.scale < 0:
            raise ValueError("noise scale must be >= 0")
        if self.kind == "ar1" and not -1.0 <= self.rho < 1.0:
            raise ValueError("ar1 rho must be in [-1, 1)")
        if self.kind == "student_t" and self.df <= 0:
            raise ValueError("student_t requires df > 0")

    @classmethod
    def gaussian(cls, scale: float = 1.0, loc: float = 0.0) -> "NoiseSpec":
        return cls(kind="gaussian", scale=scale, loc=loc)

    @classmethod
    def ar1(cls, scale: float = 1.0, rho: float = 0.8) -> "NoiseSpec":
        return cls(kind="ar1", scale=scale, rho=rho)


@dataclass
class AnomalySpec:
    """Rare, atypical events injected into the series.

    Attributes:
        kind: Type of anomaly.
        rate: Fraction of observations affected (for ``spikes``).
        magnitude: Size of the anomaly.
        symmetric: When ``True`` spikes can go up or down, otherwise only up.
    """

    kind: AnomalyKind = "spikes"
    rate: float = 0.01
    magnitude: float = 3.0
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.kind not in _ANOMALY_KINDS:
            raise ValueError(
                f"unknown anomaly kind {self.kind!r}; expected one of {sorted(_ANOMALY_KINDS)}"
            )
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("anomaly rate must be in [0, 1]")


@dataclass
class ChannelSpec:
    """A single (univariate) series of the generated dataset.

    Attributes:
        name: Column name in the resulting DataFrame.
        trend: Trend component.
        seasonality: Seasonal component.
        noise: Random component.
        anomalies: Optional anomaly injection.
        factor_loading: Weight of the shared latent factor; the only source of
            correlation between channels.
        offset: Constant added to the final value.
        scale: Multiplicative factor applied to the final value.
    """

    name: str
    trend: TrendSpec = field(default_factory=TrendSpec)
    seasonality: SeasonalitySpec = field(default_factory=SeasonalitySpec)
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    anomalies: AnomalySpec | None = None
    factor_loading: float = 0.0
    offset: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("channel name cannot be empty")


@dataclass
class SeriesSpec:
    """Complete description of a synthetic dataset.

    Attributes:
        n: Number of observations.
        freq: pandas frequency string (e.g. ``"D"``, ``"h"``, ``"15min"``).
        start: First timestamp of the index.
        seed: Seed of the random generator; ``None`` means non-reproducible.
        channels: The channels to generate.
        latent_rho: Autocorrelation of the shared latent factor.
        latent_scale: Standard deviation of the shared latent factor.
    """

    n: int = 365
    freq: str = "D"
    start: str = "2020-01-01"
    seed: int | None = 42
    channels: list[ChannelSpec] = field(default_factory=list)
    latent_rho: float = 0.8
    latent_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be strictly positive")
        if not self.channels:
            raise ValueError("at least one channel must be provided")
        names = [c.name for c in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("channel names must be unique")
        if not -1.0 <= self.latent_rho < 1.0:
            raise ValueError("latent_rho must be in [-1, 1)")
