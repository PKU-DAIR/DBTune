"""Build the experiment's objects from one JSON config file (see scripts/lab/prepare/example_*.json)."""
from __future__ import annotations

import json
import os

from .belief import Belief, BeliefSet, Empirical, Triangular, Uniform
from .executor import Executor, ResultCache
from .methods import build_method
from .tasks import TuneBudget
from .workload import ParameterSpace, ParameterSpec

BELIEFS = {"uniform": Uniform, "triangular": Triangular, "empirical": Empirical}


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_budget(cfg: dict) -> TuneBudget:
    return TuneBudget(**cfg.get("budget", {}))


def build_beliefs(cfg: dict) -> BeliefSet:
    specs, beliefs = [], {}
    for p in cfg["parameters"]:
        specs.append(ParameterSpec(p["name"], p["lo"], p["hi"], p.get("scale", "linear"), p.get("integer", False)))
        beliefs[p["name"]] = _build_belief(p["belief"])
    return BeliefSet(cfg["family"], ParameterSpace(specs), beliefs)


def build_methods(cfg: dict, budget: TuneBudget) -> list:
    return [build_method(spec, budget) for spec in cfg["methods"]]


def build_executor(cfg: dict, beliefs: BeliefSet, out_dir: str) -> Executor:
    backend = dict(cfg["backend"])
    kind = backend.pop("type")
    cache = ResultCache(os.path.join(out_dir, "task_cache"))
    if kind == "sim":
        from .backends.sim import SimExecutor
        return SimExecutor.from_config(backend, beliefs.space, cache)
    if kind == "miyabi":
        from .backends.miyabi import MiyabiExecutor
        return MiyabiExecutor.from_config(backend, out_dir, cache)
    raise ValueError("unknown backend %r" % kind)


def _build_belief(spec: dict) -> Belief:
    kwargs = dict(spec)
    return BELIEFS[kwargs.pop("type")](**kwargs)
