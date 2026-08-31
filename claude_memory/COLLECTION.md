# Source sample collection: sweep / LHS / random / LlamaTune (S0 + S1)

Scenario: before launching an app we tune on a *testing* workload that imitates the
expected production workload, then warm-start `workload_map` transfer on the real target.
This campaign collects the source datasets and compares 4 sampling strategies under
2 imitation-fidelity levels. Stop condition per cell: **fixed budget N = 451 runs**
(the sweep's natural size; all strategies budget-matched). Knob space:
[`mysql_perf_8.0.json`](../scripts/experiment/gen_knobs/mysql_perf_8.0.json) — 117 curated
perf knobs (see [KNOB_CURATION.md](KNOB_CURATION.md)).

## The 8 cells

| | S0 (oracle) | S1 (concurrency mismatch) |
|---|---|---|
| workload | rw, 150 tbl × 800k rows, **128 threads**, zipf 0.7 | same but **32 threads** |
| sweep / lhs / random / llama | `config_collect_S0_*.ini` | `config_collect_S1_*.ini` |

Target (evaluated later, NOT in this campaign) = S0's workload. task_id convention:
`miyabic_150-800000-<threads>-oltp_read_write-0.7-<strategy>`.

## Components

- **`Sampler_Optimizer`** (`autotune/optimizer/sampler_optimizer.py`): plays back a
  deterministic schedule; `optimize_method = Sampler` + `sampler_method = sweep|lhs|random`
  (+ `sweep_levels`, default 5). All schedules start with the default config (required for
  OtterTune exact-config matching at iteration 0). Sweep levels use **LlamaTune's binning**
  (`unit_to_knob_value` in `llamatune_optimizer.py`, shared with `HesBOProjection.unproject`).
  Resume = `schedule[len(history)]` + a prefix check that refuses diverged history.
  Writes `DBTune_history/schedule_<task_id>.json` manifest.
- **LlamaTune resume fix** (`pipleline.py`): loaded history is replayed into the inner
  low-dim BO via `update()`/`approx_project`, so a resumed llama cell keeps its BO state.
- **`sysbench_zipfian_exp` ini key** (`dbenv.py`): env-prefixes the benchmark cmd —
  needed because pbsdsh strips exported env vars; target skew 0.7 ≠ script default 0.2.
- **Budget**: `scripts/collect_budget.py --knobs ... --levels 5` → N=451 (= max_runs).
- **Offline tests**: `scripts/test_sampler.py` (all passing; also run `test_llamatune.py`).

## How to run (Miyabi)

```bash
# once: build the 150x800k snapshot (mysql_build/data_150x800k, ~30 GB)
qsub scripts/lab/collect_prep.sh
# wave 1 (8 nodes x 48 h; cells pause at 46 h and resume next wave)
w1=$(qsub scripts/lab/collect_launch.sh)
# wave 2 (451 runs x ~4.7 min measured ≈ 35 h -> wave 1 should finish; wave 2 = insurance)
qsub -W depend=afterany:${w1} scripts/lab/collect_launch.sh
# subset: qsub -l select=2 -v 'TASKS_OVERRIDE=S0:sweep;S0:lhs' scripts/lab/collect_launch.sh
```

History accumulates in `scripts/DBTune_history/history_<task_id>.json` (never deleted by
the collect scripts — that's what resume depends on; `par_node_run.sh` deletes its history,
the collect runner deliberately does not).

## Verification status (2026-06-11) — all green, launch blocked on tokens

1. Smoke tests DONE (`scripts/lab/collect_smoke.sh`, debug-c): sweep + llama both pass —
   pause rc=124 → resume, zipfian 0.7 prefix, Sampler prefix check, LlamaTune inner-BO
   replay (`logs/collect_smoke_sweep.log`, `logs/collect_smoke.log`).
2. Snapshot DONE: `mysql_build/data_150x800k`, 57 GB (template already held 150×800k).
3. Pilot DONE (3 h, 1 node, S0-sweep): **~4.7 min/iter** (mean 276 s, range 192–491 s),
   0/32 failed trials; the 32 iterations are banked in the S0-sweep history and the
   waves resume from them. Projection: ~35 h/cell → all 8 cells fit ONE 8-node 48 h
   wave (+ an afterany insurance wave; already-complete cells just load + exit 0).
4. **BLOCKED on tokens**: `show_token` → xg26g002 at 1,417.7/1,440 node-h (98% used).
   Campaign needs ~280 node-h + ~50 for the warm-start eval. Either new tokens
   (~400 node-h covers both) or run cell-by-cell at `sweep_levels = 3` (N≈280,
   ~22 h/cell) as budget allows.
5. Post-collection: per-cell symlink dirs as `data_repo` pools + pre-build
   `_history_cache.pkl` per dir; warm-start eval on the target with ≥3 seeds; budget
   sensitivity by replaying truncated history prefixes (N = 50/100/200/...).

## Risks

- FAILED trials consume budget (recorded at worst-case perf — intended); review per-cell
  failure counts before the warm-start study.
- LHS determinism depends on the venv's skopt version; the Sampler's prefix check turns
  any drift into a loud error instead of silent corruption.
- `innodb_io_capacity` > `innodb_io_capacity_max` and buffer-pool chunk rounding can log
  `[KNOB-VERIFY]` mismatches — benign, mysqld auto-adjusts.
