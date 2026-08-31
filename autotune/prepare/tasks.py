"""The two costly operations, as data: tune a workload, or measure configs on a workload."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

from .workload import Workload

Config = Mapping[str, object]          # knob name -> value (JSON-serializable)


def config_key(config: Config) -> str:
    return hashlib.sha1(json.dumps(dict(config), sort_keys=True, default=str).encode()).hexdigest()[:8]


def _key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class TuneBudget:
    """How long Tune(w) may run: stop after `min_success` usable samples or `max_attempts`."""
    min_success: int = 100
    max_attempts: int = 300
    seed: int = 42


@dataclass(frozen=True)
class TuneTask:
    """Tune(w): search configurations on workload w; the expensive primitive."""
    workload: Workload
    budget: TuneBudget = TuneBudget()
    kind: str = field(default="tune", init=False)

    def key(self) -> str:
        return _key(self.kind, self.workload.as_dict(), asdict(self.budget))


@dataclass(frozen=True)
class EvalTask:
    """Perf(c, w) for a list of configs (repeat a config to get repeated measurements)."""
    workload: Workload
    configs: tuple[Config, ...]
    kind: str = field(default="eval", init=False)

    def key(self) -> str:
        return _key(self.kind, self.workload.as_dict(), [dict(c) for c in self.configs])



@dataclass
class TuneResult:
    best_config: dict
    best_tps: float
    n_success: int = 0
    n_attempts: int = 0
    history_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "TuneResult":
        return cls(**d)


@dataclass
class EvalResult:
    """One tps per config in the task's order; None where the evaluation failed."""
    tps: list[float | None]

    def to_dict(self) -> dict:
        return {"tps": self.tps}

    @classmethod
    def from_dict(cls, d: Mapping) -> "EvalResult":
        return cls(list(d["tps"]))


def interleave(configs: Sequence[Config], repeats: int) -> tuple[Config, ...]:
    """[a, b] x 3 -> (a, b, a, b, a, b): repeats are spread over the session so drift hits all equally."""
    return tuple(c for _ in range(repeats) for c in configs)
