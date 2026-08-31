#!/usr/bin/env python3
"""The whole evaluation in one program:

    belief over workload parameters
      -> each preparation method builds its repository of tuned testing workloads
      -> N target workloads are sampled from the belief and tuned (ground truth)
      -> every method recommends a config per target; recommended and ground-truth
         configs are measured on the target; ratios and costs are reported.

    usage: prepare_eval.py <config.json>          (see example_sim.json / example_miyabi.json)

Independent work runs together: target Tunes are submitted before the methods start,
methods without dependencies are prepared concurrently, and each method batches its own
independent tasks (e.g. both directions of every parameter) before waiting.
Re-running with the same config resumes: finished tasks are served from the cache.
"""
from __future__ import annotations

import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from autotune.prepare.config import build_beliefs, build_budget, build_executor, build_methods, load_config  # noqa: E402
from autotune.prepare.evaluate import evaluate, submit_ground_truth, summarize  # noqa: E402


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    out_dir = os.path.abspath(cfg.get("out_dir", os.path.splitext(config_path)[0] + "_out"))
    os.makedirs(out_dir, exist_ok=True)

    beliefs = build_beliefs(cfg)
    budget = build_budget(cfg)
    methods = build_methods(cfg, budget)
    ex = build_executor(cfg, beliefs, out_dir)
    targets = beliefs.sample(cfg["n_targets"], random.Random(cfg.get("seed", 42)))

    truth_futures = submit_ground_truth(ex, targets, budget)
    repos = prepare_all(methods, beliefs, ex)
    truths = ex.wait(truth_futures)
    rows = evaluate(ex, targets, truths, methods, repos, beliefs.space, cfg.get("repeats", 3))

    save_outputs(out_dir, repos, rows)
    print(summarize(rows, repos))


def prepare_all(methods, beliefs, ex) -> dict:
    """Prepare methods level by level: independent ones concurrently, dependents afterwards."""
    repos: dict = {}
    remaining = list(methods)
    while remaining:
        ready = [m for m in remaining if m.depends_on is None or m.depends_on in repos]
        assert ready, "unsatisfiable method dependencies: %s" % [m.depends_on for m in remaining]
        with ThreadPoolExecutor(max_workers=len(ready)) as pool:
            built = list(pool.map(lambda m: m.prepare(beliefs, ex, dict(repos)), ready))
        repos.update({m.name: repo for m, repo in zip(ready, built)})
        remaining = [m for m in remaining if m not in ready]
    return repos


def save_outputs(out_dir: str, repos: dict, rows: list) -> None:
    for name, repo in repos.items():
        repo.save(os.path.join(out_dir, "repository_%s.json" % name))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump([r.to_dict() for r in rows], f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
