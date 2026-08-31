"""Baseline: a single testing workload at the belief mode (k = 1)."""
from __future__ import annotations

from typing import Mapping

from ..belief import BeliefSet
from ..common import add_tuned, tune_all
from ..executor import Executor
from ..method import PrepareMethod
from ..repository import Repository
from ..tasks import TuneBudget


class AnchorOnlyPrepare(PrepareMethod):
    name = "anchor"

    def __init__(self, budget: TuneBudget = TuneBudget()):
        self.budget = budget

    def prepare(self, beliefs: BeliefSet, ex: Executor, context: Mapping[str, Repository]) -> Repository:
        anchor = beliefs.mode_workload()
        repo = Repository()
        add_tuned(repo, anchor, tune_all(ex, [anchor], self.budget, repo)[anchor], "anchor")
        return repo
