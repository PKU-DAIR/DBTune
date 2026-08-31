"""What a preparation method produces: tuned workloads with their configs, plus bookkeeping."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

from .workload import ParameterSpace, Workload


@dataclass
class Entry:
    workload: Workload
    config: dict
    tps: float                 # the config's measured throughput on its own workload
    role: str                  # anchor | probe | fill | random | ...
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["workload"] = self.workload.as_dict()
        return d

    @classmethod
    def from_dict(cls, d) -> "Entry":
        return cls(Workload.from_dict(d["workload"]), d["config"], d["tps"], d["role"], d.get("note", ""))


@dataclass
class Ledger:
    """Cost accounting: Tune calls are the expensive unit, evaluations the cheap one."""
    tune_calls: int = 0
    eval_runs: int = 0
    notes: list = field(default_factory=list)

    def add_tunes(self, n: int = 1) -> None:
        self.tune_calls += n

    def add_evals(self, n: int) -> None:
        self.eval_runs += n


class Repository:
    def __init__(self):
        self.entries: list[Entry] = []
        self.radii: dict[str, float] = {}      # "param:down" / "param:up" -> radius (inf = insensitive)
        self.ledger = Ledger()

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, entry: Entry) -> Entry:
        self.entries.append(entry)
        return entry

    def workloads(self) -> list[Workload]:
        return [e.workload for e in self.entries]

    def contains(self, w: Workload, space: ParameterSpace) -> bool:
        return any(space.same_point(w, e.workload) for e in self.entries)

    def nearest(self, target: Workload, space: ParameterSpace) -> Entry:
        return min(self.entries, key=lambda e: space.distance(e.workload, target))

    def set_radius(self, param: str, direction: str, radius: float) -> None:
        self.radii["%s:%s" % (param, direction)] = radius

    def radius(self, param: str, direction: str) -> float:
        return self.radii.get("%s:%s" % (param, direction), math.inf)

    def to_dict(self) -> dict:
        return {"entries": [e.to_dict() for e in self.entries],
                "radii": {k: (None if math.isinf(v) else v) for k, v in self.radii.items()},
                "ledger": asdict(self.ledger)}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "Repository":
        with open(path) as f:
            d = json.load(f)
        repo = cls()
        repo.entries = [Entry.from_dict(e) for e in d["entries"]]
        repo.radii = {k: (math.inf if v is None else v) for k, v in d["radii"].items()}
        repo.ledger = Ledger(**d["ledger"])
        return repo
