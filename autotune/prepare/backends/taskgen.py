"""Turn a Tune/Eval task into the files one Miyabi node needs: config.ini, my.cnf.clean,
a replay list (Eval only) and task.json (what node_task.sh reads)."""
from __future__ import annotations

import configparser
import json
import os

from ..benchmarks import BENCHMARKS, Dataset
from ..tasks import TuneTask

MAX_CONNECTIONS = 1024     # MySQL's default 151 rejects sysbench at >150 threads

# Rows of an Eval history that are not measurements: the default config the Sampler always
# plays first, plus WARMUP_RUNS throwaway replays of the first config (the first benchmark on a
# freshly copied datadir reads cold from Lustre and measures 2-4x low).
WARMUP_RUNS = 1

LLAMATUNE = {"optimize_method": "LlamaTune", "llamatune_low_dim": "16",
             "llamatune_max_num_values": "10000", "llamatune_surrogate": "prf", "initial_runs": "10"}


def write_run(root: str, task, run_dir: str, knob_file: str, online: bool, timeout_s: int) -> dict:
    """Materialize `task` under run_dir; returns the task.json content."""
    benchmark = BENCHMARKS[task.workload.family]
    if not benchmark.supported:
        raise NotImplementedError("%s: no DBTune runner/dataset loader yet (see benchmarks.py)" % benchmark.family)
    os.makedirs(run_dir, exist_ok=True)
    task_id = "prep_%s_%s" % (task.kind, task.key())
    dataset = benchmark.dataset(task.workload)
    ini = _base_ini(root)
    _fill_database(ini["database"], root, run_dir, knob_file, online, benchmark.runner_args(task.workload))
    _fill_tune(ini["tune"], task, task_id, run_dir)
    with open(os.path.join(run_dir, "config.ini"), "w") as f:
        ini.write(f)
    _write_clean_cnf(root, run_dir)
    spec = {"kind": task.kind, "task_id": task_id, "snapshot": dataset.name, "dataset": dataset.__dict__,
            "ini": os.path.join(run_dir, "config.ini"), "timeout_s": timeout_s,
            "history": history_path(root, task_id), "workload": task.workload.as_dict()}
    _write_json(os.path.join(run_dir, "task.json"), spec)
    return spec


def write_prep_run(root: str, dataset: Dataset, run_dir: str, timeout_s: int) -> dict:
    """A dataset-building task (runs the loader once; the snapshot is then reused)."""
    os.makedirs(run_dir, exist_ok=True)
    spec = {"kind": "prep", "task_id": "prep_dataset_%s" % dataset.name, "snapshot": dataset.name,
            "dataset": dataset.__dict__, "timeout_s": timeout_s}
    _write_json(os.path.join(run_dir, "task.json"), spec)
    return spec


def history_path(root: str, task_id: str) -> str:
    return os.path.join(root, "scripts", "DBTune_history", "history_%s.json" % task_id)


def _base_ini(root: str) -> configparser.ConfigParser:
    ini = configparser.ConfigParser()
    ini.optionxform = str
    ini.read(os.path.join(root, "scripts", "lab", "prepare", "base_task.ini"))
    return ini


def _fill_database(db, root: str, run_dir: str, knob_file: str, online: bool, runner_args: dict) -> None:
    for key in ("sysbench_zipfian_exp", "sysbench_rand_type", "sysbench_extra_args"):
        db.pop(key, None)                       # the adapter decides which of these apply
    db.update(runner_args)
    db["cnf"] = os.path.join(run_dir, "my.cnf")
    db["datadir"] = os.path.join(run_dir, "data")
    db["mysqld"] = os.path.join(root, "mysql_build", "bin", "mysqld")
    db["knob_config_file"] = knob_file
    db["online_mode"] = str(bool(online))


def _fill_tune(tune, task, task_id: str, run_dir: str) -> None:
    tune["task_id"] = task_id
    if isinstance(task, TuneTask):
        tune.update(LLAMATUNE)
        tune["max_runs"] = str(task.budget.max_attempts)
        tune["min_success"] = str(task.budget.min_success)
        tune["rand_seed"] = str(task.budget.seed)
    else:
        replay = os.path.join(run_dir, "replay.json")
        played = [dict(c) for c in task.configs]
        warmup = played[:WARMUP_RUNS]     # throwaway: the first run after a fresh datadir is cold
        _write_json(replay, {"source_task_id": task_id, "configs": warmup + played, "warmup_runs": len(warmup)})
        tune.update({"optimize_method": "Sampler", "sampler_method": "replay", "replay_file": replay,
                     "max_runs": str(1 + len(warmup) + len(played)), "initial_runs": "1"})


def _write_clean_cnf(root: str, run_dir: str) -> None:
    """The clean base cnf with datadir/socket/pid/log retargeted to this run (node-local /tmp)."""
    overrides = {"datadir": os.path.join(run_dir, "data"), "socket": "/tmp/dbtune.sock",
                 "pid-file": "/tmp/dbtune.pid", "log-error": "/tmp/dbtune.err"}
    lines = []
    for line in open(os.path.join(root, "mysql_build", "cnf", "my.cnf.clean")):
        key = line.split("=", 1)[0].strip()
        lines.append("%s = %s" % (key, overrides[key]) if key in overrides else line.rstrip("\n"))
    if not any(l.split("=", 1)[0].strip() == "max_connections" for l in lines):
        lines.append("max_connections = %d" % MAX_CONNECTIONS)   # workloads go up to 256 clients
    with open(os.path.join(run_dir, "my.cnf.clean"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
