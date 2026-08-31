#!/usr/bin/env python3
"""End-to-end smoke of the Miyabi backend on the real cluster, cheaply (~25 min, 2 nodes):
one tiny Tune (LlamaTune, 3 successes) and one Eval (a config x2) on the anchor workload,
which already has a snapshot. Exercises ini generation, job script, node_task.sh chunking,
history parsing and the job registry -- everything the full run depends on.

    venv/bin/python scripts/lab/prepare/smoke_miyabi.py scripts/lab/prepare/shopify_pods.json
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from autotune.prepare.backends.miyabi import MiyabiExecutor  # noqa: E402
from autotune.prepare.config import build_beliefs, load_config  # noqa: E402
from autotune.prepare.executor import ResultCache  # noqa: E402
from autotune.prepare.tasks import EvalTask, TuneBudget, TuneTask, interleave  # noqa: E402


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    out_dir = os.path.abspath(os.path.join(cfg.get("out_dir", "prepare_runs/run"), "smoke"))
    backend = dict(cfg["backend"], tune_hours=0.4, eval_minutes_per_config=5, dry_run=False)
    ex = MiyabiExecutor.from_config(backend, out_dir, ResultCache(os.path.join(out_dir, "task_cache")))
    anchor = build_beliefs(cfg).mode_workload()

    tune = ex.submit(TuneTask(anchor, TuneBudget(min_success=3, max_attempts=6, seed=42)))
    result = ex.wait([tune])[0]
    print("TUNE:", result.n_success, "successes /", result.n_attempts, "attempts; best tps %.0f" % result.best_tps)
    assert result.n_success >= 3, "tune smoke did not reach 3 successes"

    ev = ex.submit(EvalTask(anchor, interleave([result.best_config], 2)))
    tps = ex.wait([ev])[0].tps
    print("EVAL:", tps)
    assert all(t for t in tps), "eval smoke has a failed row"
    print("SMOKE OK")


if __name__ == "__main__":
    main(sys.argv[1])
