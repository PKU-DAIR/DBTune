"""Registry of preparation methods. Add a method: one file here + one line in METHODS."""
from __future__ import annotations

from ..method import PrepareMethod
from ..tasks import TuneBudget
from .anchor_only import AnchorOnlyPrepare
from .bisection import BisectionPrepare
from .random_sample import RandomPrepare

METHODS = {
    BisectionPrepare.name: BisectionPrepare,
    RandomPrepare.name: RandomPrepare,
    AnchorOnlyPrepare.name: AnchorOnlyPrepare,
}


def build_method(spec: dict, budget: TuneBudget) -> PrepareMethod:
    """spec = {"type": <name>, <constructor kwargs>...}; the Tune budget is shared by all methods."""
    kwargs = dict(spec)
    cls = METHODS[kwargs.pop("type")]
    method = cls(budget=budget, **kwargs)
    if "name" in spec:
        method.name = spec["name"]
    return method
