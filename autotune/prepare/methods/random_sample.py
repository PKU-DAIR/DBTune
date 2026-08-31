"""Baseline: k testing workloads drawn at random from the belief, each tuned once."""
from __future__ import annotations

import random
from typing import Mapping

from ..belief import BeliefSet
from ..common import add_tuned, tune_all
from ..executor import Executor
from ..method import PrepareMethod
from ..repository import Repository
from ..tasks import TuneBudget


class RandomPrepare(PrepareMethod):
    name = "random"

    def __init__(self, k: int | None = None, match: str | None = None, seed: int = 0,
                 budget: TuneBudget = TuneBudget()):
        assert (k is None) != (match is None), "give either k or match=<method name>"
        self.k, self.depends_on, self.seed, self.budget = k, match, seed, budget

    def prepare(self, beliefs: BeliefSet, ex: Executor, context: Mapping[str, Repository]) -> Repository:
        k = self.k if self.k is not None else len(context[self.depends_on])
        workloads = beliefs.sample(k, random.Random(self.seed))
        repo = Repository()
        tuned = tune_all(ex, workloads, self.budget, repo)          # all k in parallel
        for w in workloads:
            add_tuned(repo, w, tuned[w], "random")
        return repo
