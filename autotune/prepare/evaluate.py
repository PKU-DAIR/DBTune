"""Evaluate preparation methods on sampled target workloads.

For each target: obtain its own tuned best (ground truth), ask every method for a
recommendation, measure the recommended configs and the ground-truth config on the
target under identical conditions, and report the ratio.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence

from .common import measure_many, median_tps
from .executor import Executor, Future
from .method import PrepareMethod
from .repository import Repository
from .tasks import TuneBudget, TuneResult, TuneTask, config_key
from .workload import ParameterSpace, Workload


@dataclass
class EvalRow:
    target: Workload
    method: str
    source: Workload             # the repository workload whose config was recommended
    config_tps: list             # repeated measurements of the recommended config on the target
    truth_tps: list              # repeated measurements of the target's own tuned config
    ratio: float | None          # median(config) / median(truth)

    def to_dict(self) -> dict:
        return {"target": self.target.as_dict(), "method": self.method, "source": self.source.as_dict(),
                "config_tps": self.config_tps, "truth_tps": self.truth_tps, "ratio": self.ratio}


def submit_ground_truth(ex: Executor, targets: Sequence[Workload], budget: TuneBudget) -> list[Future]:
    """Submitted first so the target Tunes share batches with the methods' own Tunes."""
    return [ex.submit(TuneTask(t, budget)) for t in targets]


def evaluate(ex: Executor, targets: Sequence[Workload], truths: Sequence[TuneResult],
             methods: Sequence[PrepareMethod], repos: Mapping[str, Repository],
             space: ParameterSpace, repeats: int) -> list[EvalRow]:
    picks = [{m.name: m.recommend(repos[m.name], t, space) for m in methods} for t in targets]
    jobs = [(t, [truth.best_config] + [e.config for e in pick.values()])
            for t, truth, pick in zip(targets, truths, picks)]
    measured = measure_many(ex, jobs, repeats)                          # all targets in parallel
    rows = []
    for t, truth, pick, m in zip(targets, truths, picks, measured):
        truth_tps = m[config_key(truth.best_config)]
        for name, entry in pick.items():
            config_tps = m[config_key(entry.config)]
            rows.append(EvalRow(t, name, entry.workload, config_tps, truth_tps, _ratio(config_tps, truth_tps)))
    return rows


def summarize(rows: Sequence[EvalRow], repos: Mapping[str, Repository], thresholds=(0.9, 0.8)) -> str:
    lines = ["%-12s %6s %6s %6s %s %8s" % ("method", "mean", "min", "median",
             " ".join("P>=%.1f" % t for t in thresholds), "tunes")]
    for name, repo in repos.items():
        ratios = [r.ratio for r in rows if r.method == name and r.ratio is not None]
        if not ratios:
            continue
        frac = " ".join("%6.2f" % (sum(r >= t for r in ratios) / len(ratios)) for t in thresholds)
        lines.append("%-12s %6.3f %6.3f %6.3f %s %8d" % (name, statistics.mean(ratios), min(ratios),
                                                       statistics.median(ratios), frac, repo.ledger.tune_calls))
    lines.append("")
    lines.append("%-40s %-12s %8s %8s %6s" % ("target", "method", "config", "truth", "ratio"))
    for r in rows:
        lines.append("%-40s %-12s %8s %8s %6s" % (r.target.label(), r.method, _fmt(median_tps(r.config_tps)),
                                                _fmt(median_tps(r.truth_tps)), _fmt(r.ratio, "%.3f")))
    return "\n".join(lines)


def _ratio(config_tps, truth_tps) -> float | None:
    c, t = median_tps(config_tps), median_tps(truth_tps)
    return c / t if c and t else None


def _fmt(value, spec="%.0f") -> str:
    return spec % value if value is not None else "FAIL"
