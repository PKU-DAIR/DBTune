"""The plugin interface every preparation method implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

from .belief import BeliefSet
from .executor import Executor
from .repository import Entry, Repository
from .workload import ParameterSpace, Workload


class PrepareMethod(ABC):
    name: str = "abstract"
    depends_on: str | None = None   # name of another method whose repository this one must see first

    @abstractmethod
    def prepare(self, beliefs: BeliefSet, ex: Executor, context: Mapping[str, Repository]) -> Repository:
        """Build the repository. Submit independent tasks together before each ex.wait()."""

    def recommend(self, repo: Repository, target: Workload, space: ParameterSpace) -> Entry:
        """Default rule: the repository workload nearest in normalized parameter distance."""
        return repo.nearest(target, space)
