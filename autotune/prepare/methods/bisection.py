"""Algorithm 1 (PREPARE): tune the belief mode, probe each parameter in both directions
with bisection in quantile space, record per-direction transfer radii, fill the covered
interval at radius spacing. Algorithm 2 (DEPLOY) selection is available as `selection="radius"`.

Every round submits all still-active probes together, so the (parameter, direction)
sides are tuned and measured in parallel on a parallel backend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from ..belief import BeliefSet
from ..common import add_tuned, measure_many, median_tps, tune_all
from ..executor import Executor
from ..method import PrepareMethod
from ..repository import Entry, Repository
from ..tasks import TuneBudget, TuneResult, config_key
from ..workload import ParameterSpace, Workload

NEGLIGIBLE_MASS = 1e-6      # tolerance when comparing quantile levels
MAX_FILL_POINTS = 1000


@dataclass
class Probe:
    """Bisection state of one (parameter, direction) side of the anchor."""
    param: str
    direction: str            # "down" | "up"
    q_edge: float             # quantile of the covered interval's edge on this side
    q_anchor: float           # quantile level of the anchor's value
    q: float                  # quantile of the next probe workload
    failed: int = 0           # probes that fell below theta so far
    radius: float | None = None

    @property
    def done(self) -> bool:
        return self.radius is not None

    @property
    def sign(self) -> int:
        return -1 if self.direction == "down" else 1

    def step_toward_anchor(self) -> None:
        self.q = (self.q + self.q_anchor) / 2     # halve the remaining probability mass


class BisectionPrepare(PrepareMethod):
    name = "bisection"

    def __init__(self, theta: float = 0.9, delta: float = 0.1, max_probes: int = 3, repeats: int = 3,
                 budget: TuneBudget = TuneBudget(), fill: bool = True, fill_widen: float = 1.0,
                 selection: str = "nearest"):
        assert selection in ("nearest", "radius")
        assert 1.0 <= fill_widen <= 2.0, "fill spacing may widen from r toward 2r only"
        self.theta, self.delta, self.max_probes, self.repeats = theta, delta, max_probes, repeats
        self.budget, self.fill, self.fill_widen, self.selection = budget, fill, fill_widen, selection

    # ---- Algorithm 1 -------------------------------------------------------------------
    def prepare(self, beliefs: BeliefSet, ex: Executor, context: Mapping[str, Repository]) -> Repository:
        repo = Repository()
        anchor = self._tune_anchor(beliefs, ex, repo)
        probes = self._init_probes(beliefs, anchor.workload, repo)
        while any(not p.done for p in probes):
            self._probe_round([p for p in probes if not p.done], anchor, beliefs, ex, repo)
        if self.fill:
            self._fill(probes, anchor.workload, beliefs, ex, repo)
        return repo

    def _tune_anchor(self, beliefs: BeliefSet, ex: Executor, repo: Repository) -> Entry:
        anchor = beliefs.mode_workload()
        result = tune_all(ex, [anchor], self.budget, repo)[anchor]
        return add_tuned(repo, anchor, result, "anchor")

    def _init_probes(self, beliefs: BeliefSet, anchor: Workload, repo: Repository) -> list[Probe]:
        probes = []
        for param in beliefs.space.names:
            q_anchor = beliefs.quantile_of(param, anchor.get(param))
            for direction, q_edge in (("down", self.delta / 2), ("up", 1 - self.delta / 2)):
                probe = Probe(param, direction, q_edge, q_anchor, q=q_edge)
                if abs(q_edge - q_anchor) <= self.delta / 2 + NEGLIGIBLE_MASS:   # mode at the boundary
                    self._resolve(probe, math.inf, repo)
                probes.append(probe)
        return probes

    def _probe_round(self, active: list[Probe], anchor: Entry, beliefs: BeliefSet,
                     ex: Executor, repo: Repository) -> None:
        workloads = [self._probe_workload(p, anchor.workload, beliefs) for p in active]
        tuned = tune_all(ex, workloads, self.budget, repo)                                  # parallel
        measurable = [(p, w) for p, w in zip(active, workloads) if tuned[w].best_config]
        measured = measure_many(ex, [(w, [anchor.config, tuned[w].best_config]) for _, w in measurable],
                                self.repeats, repo)                                          # parallel
        for (probe, w), m in zip(measurable, measured):
            self._decide(probe, w, tuned[w], m, anchor, repo)
        for probe, w in zip(active, workloads):
            if not tuned[w].best_config:      # every Tune attempt failed: nothing to measure
                repo.ledger.notes.append("%s:%s probe %s: tune produced no config -> fail"
                                         % (probe.param, probe.direction, w.label()))
                self._register_failure(probe, w, anchor, repo)

    def _probe_workload(self, probe: Probe, anchor: Workload, beliefs: BeliefSet) -> Workload:
        return anchor.with_param(probe.param, beliefs.value_at(probe.param, probe.q))

    def _decide(self, probe: Probe, w: Workload, tuned: TuneResult, measured: Mapping[str, list],
                anchor: Entry, repo: Repository) -> None:
        p = median_tps(measured[config_key(anchor.config)])                 # anchor config on w
        qref = median_tps(measured[config_key(tuned.best_config)]) or tuned.best_tps   # w's own best
        ratio = p / qref if p and qref else 0.0
        gap = abs(w.get(probe.param) - anchor.workload.get(probe.param))
        repo.ledger.notes.append("%s:%s probe %s=%g -> p=%s qref=%s ratio=%.3f" % (
            probe.param, probe.direction, probe.param, w.get(probe.param), p, qref, ratio))
        if ratio > self.theta:
            add_tuned(repo, w, tuned, "probe_pass", "ratio=%.3f" % ratio)
            self._resolve(probe, gap, repo)
            return
        add_tuned(repo, w, tuned, "probe", "ratio=%.3f" % ratio)
        self._register_failure(probe, w, anchor, repo)

    def _register_failure(self, probe: Probe, w: Workload, anchor: Entry, repo: Repository) -> None:
        gap = abs(w.get(probe.param) - anchor.workload.get(probe.param))
        probe.failed += 1
        if probe.failed >= self.max_probes:
            self._resolve(probe, gap / 2, repo)          # conservative: half the innermost failing gap
        else:
            probe.step_toward_anchor()

    def _resolve(self, probe: Probe, radius: float, repo: Repository) -> None:
        probe.radius = radius
        repo.set_radius(probe.param, probe.direction, radius)

    def _fill(self, probes: list[Probe], anchor: Workload, beliefs: BeliefSet,
              ex: Executor, repo: Repository) -> None:
        points: list[Workload] = []
        for probe in probes:
            points += [w for w in self._fill_points(probe, anchor, beliefs, repo) if w not in points]
        tuned = tune_all(ex, points, self.budget, repo)                                         # parallel
        for w in points:
            add_tuned(repo, w, tuned[w], "fill")

    def _fill_points(self, probe: Probe, anchor: Workload, beliefs: BeliefSet,
                     repo: Repository) -> list[Workload]:
        """Workloads at anchor ± m·r along one side, up to the covered edge, skipping tuned ones."""
        if probe.radius is None or math.isinf(probe.radius) or probe.radius <= 0:
            return []
        edge = beliefs.value_at(probe.param, probe.q_edge)
        start, spacing = anchor.get(probe.param), probe.radius * self.fill_widen
        points = []
        for m in range(1, MAX_FILL_POINTS + 1):
            value = beliefs.space.coerce(probe.param, start + probe.sign * m * spacing)
            if probe.sign * (value - edge) >= 0:     # reached the edge: already covered
                break
            w = anchor.with_param(probe.param, value)
            if not repo.contains(w, beliefs.space) and w not in points:
                points.append(w)
        return points

    # ---- Algorithm 2 -------------------------------------------------------------------
    def recommend(self, repo: Repository, target: Workload, space: ParameterSpace) -> Entry:
        if self.selection == "nearest":
            return repo.nearest(target, space)
        return min(repo.entries, key=lambda e: self._radius_scaled_mismatch(repo, e.workload, target, space))

    @staticmethod
    def _radius_scaled_mismatch(repo: Repository, source: Workload, target: Workload,
                                space: ParameterSpace) -> float:
        """max_i |t_i - s_i| / r_i in the direction of the gap (Algorithm 2, line 2)."""
        worst = 0.0
        for param in space.names:
            gap = target.get(param) - source.get(param)
            radius = repo.radius(param, "down" if gap < 0 else "up")
            worst = max(worst, abs(gap) / radius if radius > 0 else (0.0 if gap == 0 else math.inf))
        return worst
