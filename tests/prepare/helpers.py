"""Shared fixtures for the prepare tests: a 2-parameter world with planted radii."""
from __future__ import annotations

from autotune.prepare.backends.sim import SimExecutor, TransferSurface
from autotune.prepare.belief import BeliefSet, Triangular, Uniform
from autotune.prepare.workload import ParameterSpace, ParameterSpec

SPACE = ParameterSpace([ParameterSpec("rows", 100, 1000, "linear", integer=True),
                        ParameterSpec("threads", 8, 256, "log", integer=True)])
BELIEFS = BeliefSet("sysbench", SPACE, {"rows": Triangular(100, 500, 1000), "threads": Uniform(8, 256, mode=64)})


def make_sim(radii=None, **kwargs) -> SimExecutor:
    radii = radii if radii is not None else {"rows:down": 300, "rows:up": 150, "threads:down": 20, "threads:up": None}
    return SimExecutor(TransferSurface(SPACE, radii), **kwargs)
