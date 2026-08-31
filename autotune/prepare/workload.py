"""Workloads as parameter vectors, and the space they live in."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ParameterSpec:
    """One workload parameter: its plausible range and how to normalize it."""
    name: str
    lo: float
    hi: float
    scale: str = "linear"   # "linear" | "log"
    integer: bool = False

    def coerce(self, value: float) -> float:
        value = min(max(float(value), self.lo), self.hi)
        return float(round(value)) if self.integer else value

    def normalize(self, value: float) -> float:
        """Map a value to [0, 1] over the plausible range (log-spaced if scale == 'log')."""
        if self.scale == "log":
            return (math.log(value) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (value - self.lo) / (self.hi - self.lo)

    def denormalize(self, unit: float) -> float:
        if self.scale == "log":
            return self.coerce(math.exp(math.log(self.lo) + unit * (math.log(self.hi) - math.log(self.lo))))
        return self.coerce(self.lo + unit * (self.hi - self.lo))


@dataclass(frozen=True)
class Workload:
    """A benchmark family plus a fixed parameter assignment. Immutable and hashable."""
    family: str
    params: tuple[tuple[str, float], ...]

    @classmethod
    def make(cls, family: str, **params: float) -> "Workload":
        return cls(family, tuple(sorted((k, float(v)) for k, v in params.items())))

    @classmethod
    def from_dict(cls, d: Mapping) -> "Workload":
        return cls.make(d["family"], **d["params"])

    def as_dict(self) -> dict:
        return {"family": self.family, "params": dict(self.params)}

    def get(self, name: str) -> float:
        return dict(self.params)[name]

    def with_param(self, name: str, value: float) -> "Workload":
        params = dict(self.params)
        params[name] = float(value)
        return Workload.make(self.family, **params)

    def key(self) -> str:
        return hashlib.sha1(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()[:10]

    def label(self) -> str:
        return "%s(%s)" % (self.family, ", ".join("%s=%g" % kv for kv in self.params))


class ParameterSpace:
    """The named parameters of a workload family; defines normalization and distance."""

    def __init__(self, specs: Iterable[ParameterSpec]):
        self.specs = tuple(specs)
        self.by_name = {s.name: s for s in self.specs}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs)

    def coerce(self, name: str, value: float) -> float:
        return self.by_name[name].coerce(value)

    def normalize(self, w: Workload) -> tuple[float, ...]:
        return tuple(self.by_name[n].normalize(w.get(n)) for n in self.names)

    def distance(self, a: Workload, b: Workload) -> float:
        """Euclidean distance between range-normalized parameter vectors."""
        ua, ub = self.normalize(a), self.normalize(b)
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(ua, ub)))

    def same_point(self, a: Workload, b: Workload, tol: float = 1e-6) -> bool:
        return self.distance(a, b) <= tol
