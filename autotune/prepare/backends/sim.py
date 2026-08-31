"""A synthetic world for testing methods in seconds: configs remember the workload they
were tuned on, and their throughput on another workload decays once the parameter gap
exceeds a planted per-direction radius. Optional measurement noise."""
from __future__ import annotations

import math
import random
from typing import Mapping

from ..executor import Executor, ResultCache
from ..tasks import EvalResult, EvalTask, TuneResult, TuneTask
from ..workload import ParameterSpace, Workload


class TransferSurface:
    def __init__(self, space: ParameterSpace, radii: Mapping[str, float],
                 loss_slope: float = 0.5, floor: float = 0.1, exponents: Mapping[str, float] | None = None):
        """radii: "param:down"/"param:up" -> radius in raw units (missing/None = insensitive)."""
        self.space = space
        self.radii = {k: (math.inf if v is None else float(v)) for k, v in radii.items()}
        self.loss_slope, self.floor = loss_slope, floor
        self.exponents = dict(exponents or {})

    def base_tps(self, w: Workload) -> float:
        """Each workload's own optimum; varies with the parameters so ratios are meaningful."""
        tps = 10000.0
        for name, unit in zip(self.space.names, self.space.normalize(w)):
            tps *= (0.5 + unit) ** self.exponents.get(name, 0.0)
        return tps

    def transfer(self, origin: Workload, target: Workload) -> float:
        """Fraction of target's optimum kept by a config tuned at origin."""
        kept = 1.0
        for name in self.space.names:
            gap = target.get(name) - origin.get(name)
            radius = self.radii.get("%s:%s" % (name, "down" if gap < 0 else "up"), math.inf)
            kept *= self._loss(abs(gap) / radius if radius > 0 else math.inf)
        return kept

    def _loss(self, x: float) -> float:
        return 1.0 if x <= 1 else max(self.floor, 1 - self.loss_slope * (x - 1))


class SimExecutor(Executor):
    def __init__(self, surface: TransferSurface, noise_sigma: float = 0.0, drop_prob: float = 0.0,
                 seed: int = 0, cache: ResultCache | None = None):
        super().__init__(cache)
        self.surface, self.noise_sigma, self.drop_prob = surface, noise_sigma, drop_prob
        self.rng = random.Random(seed)
        self.tune_calls = 0

    @classmethod
    def from_config(cls, cfg: dict, space: ParameterSpace, cache: ResultCache) -> "SimExecutor":
        surface = TransferSurface(space, cfg.get("radii", {}), cfg.get("loss_slope", 0.5),
                                  cfg.get("floor", 0.1), cfg.get("exponents"))
        return cls(surface, cfg.get("noise_sigma", 0.0), cfg.get("drop_prob", 0.0), cfg.get("seed", 0), cache)

    def _run(self, tasks: list) -> list:
        return [self._tune(t) if isinstance(t, TuneTask) else self._eval(t) for t in tasks]

    def _tune(self, task: TuneTask) -> TuneResult:
        self.tune_calls += 1
        config = {"_origin": task.workload.as_dict()}
        return TuneResult(config, self._noisy(self.surface.base_tps(task.workload)),
                          task.budget.min_success, task.budget.min_success)

    def _eval(self, task: EvalTask) -> EvalResult:
        return EvalResult([self._perf(c, task.workload) for c in task.configs])

    def _perf(self, config, w: Workload) -> float:
        origin = Workload.from_dict(config["_origin"])
        return self._noisy(self.surface.base_tps(w) * self.surface.transfer(origin, w))

    def _noisy(self, tps: float) -> float:
        if self.noise_sigma:
            tps *= math.exp(self.rng.gauss(0.0, self.noise_sigma))
        if self.drop_prob and self.rng.random() < self.drop_prob:
            tps *= self.rng.uniform(0.4, 0.75)
        return tps
