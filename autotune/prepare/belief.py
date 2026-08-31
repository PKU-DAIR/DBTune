"""The operator's belief about each workload parameter, used only through its quantile function."""
from __future__ import annotations

import bisect
import math
import random
from abc import ABC, abstractmethod
from typing import Mapping, Sequence

from .workload import ParameterSpace, Workload


class Belief(ABC):
    """A distribution over one parameter's value. Only Q(q) and its inverse are needed."""

    @property
    @abstractmethod
    def mode(self) -> float: ...

    @abstractmethod
    def quantile(self, q: float) -> float:
        """Value below which fraction q of the probability lies."""

    @abstractmethod
    def quantile_of(self, value: float) -> float:
        """Inverse of quantile(): the probability mass below `value`."""


class Uniform(Belief):
    def __init__(self, lo: float, hi: float, mode: float | None = None):
        self.lo, self.hi = float(lo), float(hi)
        self._mode = float(mode) if mode is not None else (self.lo + self.hi) / 2

    @property
    def mode(self) -> float:
        return self._mode

    def quantile(self, q: float) -> float:
        return self.lo + q * (self.hi - self.lo)

    def quantile_of(self, value: float) -> float:
        return _clip01((value - self.lo) / (self.hi - self.lo))


class Triangular(Belief):
    """A range with a most-likely value."""

    def __init__(self, lo: float, mode: float, hi: float):
        assert lo <= mode <= hi, "triangular belief needs lo <= mode <= hi"
        self.lo, self._mode, self.hi = float(lo), float(mode), float(hi)

    @property
    def mode(self) -> float:
        return self._mode

    def quantile(self, q: float) -> float:
        lo, m, hi = self.lo, self._mode, self.hi
        split = (m - lo) / (hi - lo)
        if q < split:
            return lo + math.sqrt(q * (hi - lo) * (m - lo))
        return hi - math.sqrt((1 - q) * (hi - lo) * (hi - m))

    def quantile_of(self, value: float) -> float:
        lo, m, hi = self.lo, self._mode, self.hi
        if value <= lo:
            return 0.0
        if value >= hi:
            return 1.0
        if value < m:
            return (value - lo) ** 2 / ((hi - lo) * (m - lo))
        return 1 - (hi - value) ** 2 / ((hi - lo) * (hi - m))


class Empirical(Belief):
    """Any shape, given as samples (e.g. from production telemetry of similar apps)."""

    def __init__(self, values: Sequence[float], mode: float | None = None):
        self.values = sorted(float(v) for v in values)
        self._mode = float(mode) if mode is not None else self._densest_value()

    @property
    def mode(self) -> float:
        return self._mode

    def quantile(self, q: float) -> float:
        pos = _clip01(q) * (len(self.values) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        return self.values[lo] + (pos - lo) * (self.values[hi] - self.values[lo])

    def quantile_of(self, value: float) -> float:
        """Inverse of quantile(): linear interpolation between neighbouring samples."""
        n = len(self.values)
        if n == 1 or value >= self.values[-1]:
            return 1.0
        if value <= self.values[0]:
            return 0.0
        i = bisect.bisect_right(self.values, value) - 1
        lo, hi = self.values[i], self.values[i + 1]
        frac = (value - lo) / (hi - lo) if hi > lo else 0.0
        return (i + frac) / (n - 1)

    def _densest_value(self, bins: int = 10) -> float:
        lo, hi = self.values[0], self.values[-1]
        if hi == lo:
            return lo
        width = (hi - lo) / bins
        counts = [0] * bins
        for v in self.values:
            counts[min(int((v - lo) / width), bins - 1)] += 1
        best = max(range(bins), key=counts.__getitem__)
        return lo + (best + 0.5) * width


class BeliefSet:
    """Independent beliefs for every parameter of a workload family."""

    def __init__(self, family: str, space: ParameterSpace, beliefs: Mapping[str, Belief]):
        missing = set(space.names) - set(beliefs)
        assert not missing, "no belief for parameters %s" % sorted(missing)
        self.family, self.space, self.beliefs = family, space, dict(beliefs)

    def value_at(self, name: str, q: float) -> float:
        return self.space.coerce(name, self.beliefs[name].quantile(q))

    def quantile_of(self, name: str, value: float) -> float:
        return self.beliefs[name].quantile_of(value)

    def mode_workload(self) -> Workload:
        return Workload.make(self.family, **{n: self.space.coerce(n, b.mode) for n, b in self.beliefs.items()})

    def covered_interval(self, name: str, delta: float) -> tuple[float, float]:
        """The central 1 - delta mass; the delta tail is left to deployment-time tuning."""
        return self.value_at(name, delta / 2), self.value_at(name, 1 - delta / 2)

    def sample(self, n: int, rng: random.Random) -> list[Workload]:
        return [Workload.make(self.family, **{name: self.value_at(name, rng.random()) for name in self.space.names})
                for _ in range(n)]


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)
