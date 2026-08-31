# Preparation-first tuning: evaluation program

One program runs the whole experiment described in `prepare_eval.py`:

```
cd DBTune
venv/bin/python scripts/lab/prepare/prepare_eval.py scripts/lab/prepare/example_sim.json      # seconds, simulator
venv/bin/python scripts/lab/prepare/prepare_eval.py scripts/lab/prepare/example_miyabi.json   # cluster (set dry_run=false)
```

Layers (each talks only to the one below):

| layer | where | knows about |
|---|---|---|
| evaluation program | `scripts/lab/prepare/prepare_eval.py` | config file, ordering of methods, report |
| methods + evaluation | `autotune/prepare/{methods/,evaluate.py}` | beliefs, workloads, Tune/Eval tasks |
| executor | `autotune/prepare/executor.py` | batching, caching, waiting |
| backends | `autotune/prepare/backends/{sim,miyabi}.py` | synthetic surface / PBS jobs, nodes, datasets |

Adding a preparation method: one file in `autotune/prepare/methods/`, subclass
`PrepareMethod`, implement `prepare()` (and `recommend()` only if the selection rule
differs), register it in `methods/__init__.py`, name it in the config's `methods` list.

Parallelism: everything submitted to the executor before a `wait()` runs as one batch;
on Miyabi a batch becomes PBS jobs with one node per task. Target ground-truth Tunes are
submitted first, independent methods are prepared concurrently, and Bisection submits
all active (parameter, direction) probes of a round together.

Resume: results are cached under `<out_dir>/task_cache/` by task content; re-running the
same config skips finished work. On Miyabi, submitted job ids are kept in
`<out_dir>/logs/jobs.json`; a restarted program attaches to jobs still in the queue
instead of submitting duplicates. Run the program under `nohup`/`tmux` on the login node.

Tests: `venv/bin/python -m unittest discover -s tests/prepare -t .`

## Cluster prerequisites (not in this branch)

The Miyabi backend drives DBTune's existing tooling: `optimize.py` with the `LlamaTune`
and `Sampler` optimizers, `scripts/lab/{reset_database,start_stack,stop_stack,collect_prep}.sh`,
`mysql_build/` with snapshots `data_150x<rows>k`. These live on the backup branch /
working tree, not on `main`. TPC-C needs a parameterized runner and a dataset loader
(`autotune/prepare/benchmarks.py::TPCC` defines the mapping; the backend refuses TPC-C
tasks until the runner exists).
