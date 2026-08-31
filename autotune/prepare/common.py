"""Small helpers shared by all methods: batch-tune, batch-measure, add to a repository."""
from __future__ import annotations

import statistics
from typing import Iterable, Sequence

from .executor import Executor
from .repository import Entry, Repository
from .tasks import Config, EvalTask, TuneBudget, TuneResult, TuneTask, config_key, interleave
from .workload import Workload


def tune_all(ex: Executor, workloads: Iterable[Workload], budget: TuneBudget,
             repo: Repository | None = None) -> dict[Workload, TuneResult]:
    """Tune several workloads in one batch (they run in parallel on a parallel backend)."""
    workloads = list(workloads)
    results = ex.run_all([TuneTask(w, budget) for w in workloads])
    if repo is not None:
        repo.ledger.add_tunes(len(workloads))
    return dict(zip(workloads, results))


def measure(ex: Executor, workload: Workload, configs: Sequence[Config], repeats: int,
            repo: Repository | None = None) -> dict[str, list[float | None]]:
    """Measure each config `repeats` times on one workload; returns config_key -> tps list."""
    return measure_many(ex, [(workload, configs)], repeats, repo)[0]


def measure_many(ex: Executor, jobs: Sequence[tuple[Workload, Sequence[Config]]], repeats: int,
                 repo: Repository | None = None) -> list[dict[str, list[float | None]]]:
    """Same as measure(), for several workloads at once (one batch, parallel backend)."""
    tasks = [EvalTask(w, interleave(configs, repeats)) for w, configs in jobs]
    results = ex.run_all(tasks)
    if repo is not None:
        repo.ledger.add_evals(sum(len(t.configs) for t in tasks))
    return [_group_by_config(t, r.tps) for t, r in zip(tasks, results)]


def median_tps(values: Sequence[float | None]) -> float | None:
    """Median of the successful measurements; None if every one failed."""
    ok = [v for v in values if v]
    return statistics.median(ok) if ok else None


def add_tuned(repo: Repository, workload: Workload, result: TuneResult, role: str, note: str = "") -> Entry:
    return repo.add(Entry(workload, dict(result.best_config), result.best_tps, role, note))


def _group_by_config(task: EvalTask, tps: Sequence[float | None]) -> dict[str, list[float | None]]:
    grouped: dict[str, list] = {}
    for config, value in zip(task.configs, tps):
        grouped.setdefault(config_key(config), []).append(value)
    return grouped
