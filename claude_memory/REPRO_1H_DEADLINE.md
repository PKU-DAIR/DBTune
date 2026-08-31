# REPRO — the 1 h-deadline transfer-method comparison

The experiment behind "automatic tuning is slow, luck-dependent, and surrogate transfer
converts nothing" ([HANDOFF.md §2](HANDOFF.md), findings F4/F7/F7a/F7b in
[exp_notes/similar_workload.md](../exp_notes/similar_workload.md)). Every step is scripted;
this file is the exact command chain, the exact per-arm configuration, where the outputs
land, and the caveats that apply when re-running or citing it.

## 1. What it measures (reference result)

Target workload: sysbench `oltp_read_write`, 150 tables × 800k rows, 128 clients, zipfian
0.7, MySQL 8.0.44, 117-knob space (`scripts/experiment/gen_knobs/mysql_perf_8.0.json`),
offline knob application (my.cnf + restart per iteration), 60 s warmup + 120 s measurement.
Five arms, each a fresh SMAC session killed by `timeout 3600`, three seeds:

```
$ python3 scripts/report_eval2.py
best tps within 3600 s, per arm and seed (n = iterations started within the deadline; t = time of best)

arm                         seed 42               seed 43               seed 44     mean   median
opadviser     19508 (n= 9, t= 408s) 19413 (n=11, t=1829s) 19110 (n=10, t=1768s)    19344    19413
opadviser_ns   6700 (n=12, t=2771s) 21640 (n=10, t=2404s)  6800 (n=10, t=2765s)    11713     6800
ottertune      1933 (n= 9, t=1818s) 13716 (n=11, t=2511s)  3096 (n=11, t= 448s)     6249     3096
rgpe           1925 (n= 9, t=1826s)  4266 (n= 8, t= 438s)  3547 (n=11, t= 440s)     3246     3547
cold           3183 (n=11, t=3324s)  5146 (n=12, t=2415s)  1568 (n= 7, t= 441s)     3299     3183
```

Reading: ~9–12 evaluations/hour in every arm (speed is physics: ~5–6 min per restart +
benchmark, optimizer overhead 2–17 s); best-after-60-min varies 2–3× across seeds within an
arm; OtterTune mapping and RGPE ≈ cold even with a 15-source repository containing the
identical workload; the only arm that reliably wins (OpAdviser, 19.1–19.5k on every seed,
reached by ~minute 7–30) does so by **replaying source-best configs in its init**, an
undocumented behaviour of the released code — remove it (`opadviser_ns`) and the median
drops to 6.8k.

Runs: main wave 2026-07-07 12:56–15:59 JST (nodes mc024 / mc184 / mc185 / mc186, one arm
per node), ablation 2026-07-08 15:31–18:34 (mc074). Cost ≈ 4 × 3.1 + 3.1 ≈ 15.5 node-h.

## 2. Code state

Everything is on branch `feature/prepare-method` at commit `2c622d0` (2026-08-27, "Track the
experiment tooling that was untracked on the backup branch"). The July runs used the
then-uncommitted working tree of `backup/dml-ottertune-wip`; between the runs and the
2026-08-27 commits that captured that tree (`e5d3a15`, `2c622d0`) no other commit touched
`autotune/` or the transfer tooling. Relevant code:

| what | where |
|---|---|
| seeded pipeline (`rand_seed` ini → `random_state`) | `autotune/tuner.py` |
| source repo loading + `_history_cache.pkl` (only for `workload_map`/`rgpe` or `space_transfer`) | `autotune/tuner.py:68-114, 133-134, 176-177` |
| OpAdviser replay init + the `space_transfer_replay=False` switch | `autotune/pipleline/pipleline.py:172, 557-561` |
| per-iteration wall-clock (`update_time` = seconds since optimize.py start) | `autotune/utils/history_container.py:134, 294` |
| OtterTune metric-binning source selection | `autotune/transfer/tlbo/workload_map.py` |

## 3. Scripts (all under `scripts/`, run `qsub` from the repo root)

| step | script | notes |
|---|---|---|
| per-run ini + isolated my.cnf | `lab/transfer_gen_task.py <arm> <seed> <root>` | derives from `config_collect_S0_sweep.ini` (target workload base); see §5 |
| one arm, 3 seeds sequentially, on one node | `lab/transfer_node_run.sh <root> <arm>` | fresh history + fresh datadir + default-config reset + metrics stack per seed, then `timeout 3600 optimize.py` |
| PBS wave: one arm per vnode | `lab/transfer_launch.sh` | `regular-c`, `select=4`, walltime 4:30; `ARMS_OVERRIDE='a;b'` picks arms |
| smoke one arm for 3 iterations (debug-c) | `lab/transfer_smoke.sh` (`-v MODE=`) | exercises rgpe / space_transfer code paths before a wave |
| per-cell SUCCESS-only pools + caches | `build_eval_pools.py` | from `DBTune_history/history_miyabic_*.json` |
| multi-source repo `pool_ALL` | `build_pool_all.py` | 15 contexts (S0–S4 × lhs/random/llama), 1,517 rows, pre-built cache |
| result table | `report_eval2.py [--deadline S] [--trace]` | reproduces §1 from the histories |
| helpers used per seed | `lab/reset_database.sh`, `lab/start_stack.sh`, `lab/stop_stack.sh` | default cnf from the knob file; Prometheus + mysqld/node exporters |

## 4. Exact command chain

```bash
cd /work/xg26g002/x10563/DBTune

# 0. one-time environment (already present in this checkout)
qsub scripts/lab/setup_build.sh        # MySQL 8.0.44 build, sysbench, venv  (short-c, 2 h)
qsub scripts/lab/collect_prep.sh       # snapshot mysql_build/data_150x800k, 57 GB (regular-c, ≤6 h)
#    tools/{mysqld_exporter_dir,node_exporter_dir,prometheus} must be unpacked (they are)

# 1. source data — REUSE the existing collections, do not recollect:
#    scripts/DBTune_history/history_miyabic_150-{800000,80000}-{128,32}-oltp_read_write[_ps30]-{0.7,uniform}-{lhs,random,llama}.json
#    (recollection = the S0–S4 campaign in COLLECTION.md, 10–28 node-h per llama cell)

# 2. pools (offline, login node is fine; single-threaded on purpose — see §7)
cd scripts
../venv/bin/python build_eval_pools.py      # DBTune_history/pool_<cell>_<strategy>/
../venv/bin/python build_pool_all.py        # DBTune_history/pool_ALL/ (+ _history_cache.pkl)
cd ..

# 3. optional smoke, 3 iterations at full workload size (debug-c, 30 min each)
qsub -v MODE=rgpe      scripts/lab/transfer_smoke.sh
qsub -v MODE=opadviser scripts/lab/transfer_smoke.sh

# 4. main wave: ottertune / rgpe / opadviser / cold × seeds 42 43 44  (~3 h 05 min, 4 nodes)
qsub scripts/lab/transfer_launch.sh

# 5. ablation arm (as run on 2026-07-08; reconstructed from logs/transfer_launch_ns.log —
#    the shell history was not retained)
qsub -l select=1 -o logs/transfer_launch_ns.log -v 'ARMS_OVERRIDE=opadviser_ns' scripts/lab/transfer_launch.sh

# 6. summarize
python3 scripts/report_eval2.py --trace
```

`build_pool_all.py` was written after the fact (the July `pool_ALL` was assembled by hand);
it was verified to rebuild all 15 JSONs byte-identically and an equivalent cache
(same labels, order, configs and perfs).

## 5. Exact per-arm configuration

Common `[tune]` (from `transfer_gen_task.py`): `optimize_method = SMAC`, `max_runs = 200`
(never reached — the timeout is the stop), `selector_type = shap` but `incremental = none`
and `initial_tunable_knob_num = -1` (no knob pruning), `rand_seed = <seed>`,
`internal_metrics_source = prometheus`. Common `[database]`: the target workload above,
`knob_num = -1`, `online_mode = False`, `remote_mode = False`, per-run `cnf`/`datadir`/`sock`
under `parallel/eval2_<arm>_s<seed>/`.

| arm | `transfer_framework` | `space_transfer` | `space_transfer_replay` | `initial_runs` | `data_repo` | other |
|---|---|---|---|---|---|---|
| `ottertune` | `workload_map` | None | – | 1 | `./DBTune_history/pool_ALL` | `mapping_method = ottertune`, `mapping_prune_metrics = false` |
| `rgpe` | `rgpe` | None | – | 1 | `./DBTune_history/pool_ALL` | |
| `opadviser` | none | True | (default True) | 5 | `./DBTune_history/pool_ALL` | first 5 evals = source-best configs (replay init) |
| `opadviser_ns` | none | True | **False** | 1 | `./DBTune_history/pool_ALL` | paper's compact-space mechanism only |
| `cold` | none | None | – | 1 | `./DBTune_history/empty_repo` | dir does not exist; never read (§2 loader guard) |

The generated files are kept: `parallel/eval2_<arm>_s<seed>/config.ini` is the literal ini
each session ran with (`my.cnf.clean` = base cnf, `my.cnf.default` = default knobs written
by `reset_database.sh`, `my.cnf` = last applied config).

Per seed, `transfer_node_run.sh` does: delete `history_eval2_<arm>_s<seed>.json` (timed
sessions never resume) → `cp -a mysql_build/data_150x800k parallel/<task>/data` (~1–2 min)
→ `reset_database.sh` (default cnf, purge binlogs) → `start_stack.sh` → `timeout
--signal=TERM --kill-after=120 3600 python optimize.py --config=<ini>` → expect rc=124 →
stop stack, shutdown mysqld, delete datadir. `update_time` counts from optimize.py start,
so the snapshot copy / reset are outside the hour.

## 6. Outputs

| artefact | path |
|---|---|
| per-session history (config, metrics, `trial_state`, `update_time` per row) | `scripts/DBTune_history/history_eval2_<arm>_s<seed>.json` |
| per-arm node log (incl. source selected by OT, iteration timings) | `logs/parallel/eval2_<arm>.log` (first line = hostname) |
| PBS job log | `logs/transfer_launch.log`, `logs/transfer_launch_ns.log` (PBS `-o` appends) |
| per-run ini / cnf | `parallel/eval2_<arm>_s<seed>/` |
| arm exit code | `parallel/eval2_<arm>/.rc` |
| smoke runs | `history_eval2smoke_<arm>.json`, `logs/transfer_smoke_<arm>.log`, `scripts/eval2smoke_<arm>.png` |

None of these are git-tracked (`*.log`, `*.pkl`, `DBTune_history/`, `parallel/` are ignored).

## 7. Gotchas and caveats

Re-running:
- **Re-running overwrites the July data**: `transfer_node_run.sh` deletes
  `history_eval2_<arm>_s<seed>.json` at the start of each seed. Back the 15 histories up (or
  change the `eval2_` prefix in `transfer_gen_task.py`) before submitting.
- `pool_ALL/_history_cache.pkl` must exist and be newer than its JSONs before the wave —
  otherwise four `pbsdsh` arms race to build it on Lustre and clobber each other
  ([MIYABI.md §5](MIYABI.md)). `build_pool_all.py` writes it; re-run it if you touch the pool.
- The deadline is `DEADLINE_S` in `transfer_node_run.sh` (env `TRANSFER_DEADLINE_S`) — but
  `pbsdsh` strips exported env vars, so `qsub -v` does not reach it; edit the default (and
  the PBS walltime: 3 seeds × (deadline + ~5 min setup) + margin).
- `qstat <finished job>` prints "No unfinished job found" but exits 0 — grep the message.
- Budget: check `show_token` first; one wave ≈ 12.5 node-h, the ablation ≈ 3 node-h.

Interpreting (details in [RQ_CRITIQUE.md](RQ_CRITIQUE.md)):
- **Seeds are shared across arms, so the three seeds are not independent between arms**:
  seed 43 made cold / ottertune / rgpe / opadviser_ns evaluate the same early configs; OT's
  13,716 on seed 43 is that shared config, not a transfer effect
  ([RQ_CRITIQUE.md:50](RQ_CRITIQUE.md#L50)).
- OtterTune's source selection failed the oracle test (never picked an S0 pool; ranks
  {5,9,10}, {4,9,10}, {6,7,9} in the three sessions) because the binned distance is dominated
  by node-level nuisance dims (TCP retransmits, MemTotal, disk IOPS…); dropping ~20
  node_exporter dims is a cheap untested control ([RQ_CRITIQUE.md:64](RQ_CRITIQUE.md#L64)).
  Per F4, even a correct pick would not have been exploited by the EI route.
- The iteration in flight at 3600 s is killed and never recorded; `n` in the table is
  completed iterations (cold s44: 7).
- Every number is a single measurement; high-TPS configs re-measure within ~4 % but with a
  13–18 % tail of >25 % drops, mid-range configs up to 2.3×
  ([RQ_CRITIQUE.md:94](RQ_CRITIQUE.md#L94)).
- The `opadviser` arm is the *released implementation* (replay init from commit `8ca5a99`
  upstream); label it so in any writeup (F7a).

## 8. Reproduction runs

**Run 2 — 2026-08-27 22:25 → 08-28 01:40 JST**, job `3088190.opbs`, `ARMS_OVERRIDE=ottertune;rgpe;opadviser_ns;opadviser`
(cold omitted), nodes mc057–mc060, ~13 node-h. Inis byte-identical to July's; July artefacts
backed up to `scripts/DBTune_history/eval2_run1_2026-07/`, `logs/parallel/eval2_run1_2026-07/`,
`parallel/eval2_run1_2026-07/` (read with `report_eval2.py --hist-dir`).

```
arm                         seed 42               seed 43               seed 44     mean   median
ottertune      1676 (n= 8, t=2317s)  9523 (n=10, t=2617s)  1596 (n= 9, t=2815s)     4265     1676
rgpe           1686 (n= 8, t=2593s)  7633 (n=10, t=2892s)  1587 (n= 9, t=2575s)     3635     1686
opadviser_ns   1631 (n= 7, t=1827s)  1693 (n=10, t=2029s)  1844 (n= 9, t=2319s)     1723     1693
opadviser     14840 (n=10, t=1934s) 16218 (n=10, t=2002s) 15649 (n= 9, t=1757s)    15569    15649
```

Reproduced: surrogate arms ≈ 1.7k median vs replay 15.6k; seed decides; OT picks S4-lhs again;
first 3 iterations byte-identical across the three surrogate arms (shared seed). New caveat:
the cluster measured **22–60 % slower than in July for identical configs** (default 138–245 →
76–112 tps; OpAdviser's identical replayed configs 19.5k → 10.8–14.8k) on all four nodes — so
compare within a run, never absolute numbers across runs. Full write-up: exp_notes F7c.

## 9. Online (dynamic-knob) variant — "does the failure reproduce without restarts?"

Same experiment with the **95 dynamic knobs** (`mysql_perf_8.0_online.json`; 94 ⊂ the 117-knob
perf file + `innodb_redo_log_capacity`, 23 restart-only knobs dropped) applied via
`SET GLOBAL` (`online_mode = True`, no restart per iteration → ~17 evaluations/h instead of ~10),
and a **single, online-collected source**: S0 × LlamaTune with the same knob file and online
application. Everything is parameterised on top of the offline chain (offline behaviour unchanged;
`transfer_gen_task.py ... offline` output stays byte-identical).

```bash
cd /work/xg26g002/x10563/DBTune

# 1. source: S0 x LlamaTune, online knobs, stop at 100 SUCCESS / 300 attempts (1 node, ~6-8 h)
#    ini scripts/config_collect_S0_llama_online.ini; chunked runner (fresh datadir every 4 h);
#    resubmit the same line to resume a PAUSED cell
qsub -o logs/collect_S0_llama_online.log scripts/lab/collect_online_launch.sh
#    -> scripts/DBTune_history/history_miyabic_150-800000-128-oltp_read_write-0.7-llama_online.json
#       log logs/parallel/collect_S0_llama_online.log

# 2. pool (SUCCESS-only + cache, built in the 95-knob space)
cd scripts
../venv/bin/python build_pool_all.py --knob-file ./experiment/gen_knobs/mysql_perf_8.0_online.json \
    --from-history DBTune_history/history_miyabic_150-800000-128-oltp_read_write-0.7-llama_online.json \
    --out DBTune_history/pool_S0_llama_online
cd ..

# 3. 1 h-deadline wave, online mode, task prefix eval3_ (does not touch eval2_ data)
qsub -o logs/transfer_launch_online.log -v 'VARIANT=online,ARMS_OVERRIDE=ottertune;rgpe;opadviser_ns;opadviser' \
    scripts/lab/transfer_launch.sh

# 4. table
python3 scripts/report_eval2.py --prefix eval3 --arms ottertune rgpe opadviser_ns opadviser
```

Per-arm inis are the §5 table with `knob_config_file` / `online_mode = True` /
`data_repo = ./DBTune_history/pool_S0_llama_online` swapped in (`transfer_gen_task.py <arm> <seed>
<root> online`). Outputs: `history_eval3_<arm>_s<seed>.json`, `logs/parallel/eval3_<arm>.log`,
`parallel/eval3_<arm>_s<seed>/`. Caveats specific to this variant: OT's source selection is
trivial (one context); the earlier 1 h online probe (`history_probe_online_llama.json`, before the
`6dd7e26` online-apply fix) reached only 143 tps in 16 iterations — check the collection's best
tps before reading the wave.

### 9a. Online-collection log (2026-08-28/29) — resize stalls, fix, history edit

- Job `3132872.opbs` (mc170, 22:45): online apply works (iteration 1 = 1,353 tps, iteration 11 =
  11,983 tps), but **11 of the first 30 iterations stalled** — sysbench froze mid-run (0 tps for
  the rest of the window, one transaction per thread), DBTune's 210 s benchmark timeout fired
  (`dbenv.py`: `benchmark_timeout = True` is commented out upstream, so the row is stored as
  SUCCESS with tps = −1, objective +1) and LlamaTune learned "worst possible" for configs that
  were running at 5.9k tps before the freeze. Cause: `innodb_buffer_pool_size` /
  `innodb_redo_log_capacity` resize **asynchronously**; the apply step only waited 30 s for the
  variable to change (`Timeout waiting for innodb_buffer_pool_size to apply` precedes every
  stall) and the benchmark started while the resize was still withdrawing/allocating pages.
- Fix (`autotune/database/mysqldb.py`, `_wait_for_async_resizes`, called at the end of
  `apply_knobs_online`): poll `Innodb_buffer_pool_resize_status` (until "Completed…"/empty) and
  `Innodb_redo_log_resize_status` (until `OK`), cap 900 s, log `[resize-wait] … settled after N s`.
  Offline mode is untouched. `build_pool_all.py --from-history` now also drops tps ≤ 0 rows.
- History edit: the 30-row history was backed up to
  `history_…-llama_online.json.bak_with_stalls_2026-08-29` and the 11 stall rows (10 × −1 and one
  7-tps run with the same 128-transaction signature) were removed before resuming (LlamaTune's
  resume replays observations by content, so row deletion is safe). Job `3143348.opbs` (mc189,
  00:58) resumed from the 19 clean rows with the fix active.
- Interpretation caveat for the online variant: the two biggest knobs are "dynamic" only
  nominally — an online resize costs minutes and blocks the server, so per-iteration cost does
  not vanish without restarts. Report the `[resize-wait]` distribution alongside the results.

### 9b. Outcome of the 95-knob attempt (job 3143348, 00:58–09:05) and the 93-knob restart

- Resize waits: 28 measured since the fix, **median 901 s, 21 of 28 hit the 900 s cap** ("buffer
  pool 7 : withdrawing blocks (5872/73725)" on an idle server — ~100 MB per 15 min; the flush needed
  by a shrink is throttled by whatever `innodb_io_capacity` / dirty-page knobs the current config
  set). A tunable buffer pool / redo capacity therefore costs 15 min+ per iteration online.
- Then a second harness bug: `DBEnv.get_states` created a new `multiprocessing.Manager()` per
  iteration and never shut it down (67 live manager processes), and node-local `/tmp` (14 GB) filled
  up (`OSError: [Errno 28] No space left on device: /tmp/<job>/pymp-…`) → 88 consecutive FAILED
  iterations, then mysqld could not start at the next chunk (socket/err-log are in `/tmp`) → job
  ended rc=1. Net: 132 attempts, 41 usable rows, best 11,983. History archived as
  `history_…-llama_online.json.95knob_attempt_2026-08-29`.
- Fixes: one Manager per `DBEnv` (`dbenv.py`); `stop_stack.sh` deletes the Prometheus TSDB it
  started; `collect2_node_run.sh` logs a `/tmp` probe (df + biggest entries) at chunk start and
  hourly — the actual `/tmp` consumer is not yet identified (first occurrence ever; only online
  sessions keep one mysqld alive for hours).
- **Design change: the two async-resize knobs are pinned.** New space
  `mysql_perf_8.0_online_noresize.json` (93 knobs); `innodb_buffer_pool_size = 20615843020`
  (19.2 GB) and `innodb_redo_log_capacity = 104857600` (100 MB, the knob-file defaults) are written
  as fixed lines into the cell's `my.cnf.clean` (`[database] cnf_extra` in the collect ini; the
  `online` variant of `transfer_gen_task.py` does the same). The online variant is therefore
  "cheaply-applicable dynamic knobs", not "all dynamic knobs"; the buffer pool and redo size stay at
  the study's default-config values. Fresh collection: job `3192570.opbs` (mc158, 11:17).

## 10. Where the offline dead time goes — restart-policy micro-benchmark (2026-08-29)

Offline iterations cost a median **125 s of dead time** (interval − 180 s benchmark; run 2,
n=124; p90 ≈ 400 s). `MysqlDB._kill_mysqld` shuts the server down with **`kill -9`** (the
graceful path is commented out), so every startup performs crash recovery of the redo written
during the previous 120 s benchmark. `scripts/lab/test_restart_policy.sh` (job 3195221, mc017,
short-c, ~45 min): fresh snapshot, three real top-tps S0 configs (26 GB/5 GB, 23.7 GB/2.6 GB,
15 GB/5.5 GB pool/redo), 9 cycles of start → sysbench RW 60+120 s → shutdown, policy rotated
(K9 = kill -9 | G1 = `innodb_fast_shutdown=1` + `mysqladmin shutdown` | G2 = `fast_shutdown=2`),
configs Latin-square ordered, buffer-pool dump/load pinned OFF. Report:
`python3 scripts/lab/test_restart_policy_report.py`; raw: `parallel/restart_test/`.

| shutdown policy | shutdown | next startup | dead time / iteration |
|---|---|---|---|
| K9 (DBTune today) | 1–3 s | 118 / 120 / 282 s | **175 s** |
| G2 (`fast_shutdown=2`, log flush only) | 5–7 s | 158 / 185 s | 178 s |
| **G1 (`fast_shutdown=1`, clean)** | 14 / 14 / 21 s | 6 / 8 / 10 s | **25 s** |

- Recovery time scales with the checkpoint age at the kill (0.8–0.9 GB → ~120 s, 3 GB → 282 s);
  the snapshot (clean) starts a 26 GB-pool config in 7 s. The startup itself is not the cost —
  redo re-application on Lustre is.
- Clean shutdown flushes 500–880 k dirty pages (8–13.5 GB) in 14–21 s; the flush is bounded by
  dirty-page volume, not redo volume. No fallback fired.
- tps per config is unaffected by the policy (c2: 16.2k/20.3k/20.2k; c3: 13.8k/13.7k/13.5k).
- Projected offline iteration: 180 s benchmark + ~25 s restart + ~10 s bookkeeping ≈ 215 s →
  ~16.7 evaluations/h (vs 11.7 today), i.e. the same speed-up the pinned-online variant gave,
  with **all 117 knobs (buffer pool included) tunable**.

Implementation under test (uncommitted): `[database] graceful_shutdown = True` (default False =
unchanged kill -9) → `_kill_mysqld` sets `innodb_fast_shutdown=1` and
`innodb_buffer_pool_dump_at_shutdown=OFF`, runs `mysqladmin shutdown` with a 180 s budget, and
falls back to kill -9. End-to-end check: `qsub -v MODE=cold,GRACEFUL=1 scripts/lab/transfer_smoke.sh`
(task `eval2smoke_cold_graceful`).

End-to-end check (job 3198981, mc001, `eval2smoke_cold_graceful`, 3 SMAC iterations, 117 knobs):
shutdown 2 / 29 / 109 s, startup 3 / 7 / 6 s, no fallback; iteration intervals 226 s and 304 s
(dead time 46 s and 124 s) vs run 2's 308 s median. The 109 s shutdown followed a 1.2k-tps config
with doublewrite ON, `innodb_flush_neighbors=1`, 20 GB pool (io_capacity was 1.57 M, so not
throttling) — clean-shutdown cost is bounded by dirty-page flush volume on Lustre, and varies by
config. Same-load baseline from run 2 (kill -9, previous config at 500–2 000 tps): dead time
median 120 s, mean 184 s, max 575 s. Nothing committed; `graceful_shutdown` defaults to False.

## 11. Clean shutdown is now the default (2026-08-29) — `fast` variant, `eval4_` prefix

`MysqlDB.graceful_shutdown` defaults to **True**; `[database] graceful_shutdown = False` restores
kill -9. `transfer_gen_task.py … offline` (the eval2 recipe) pins `False` so §4 stays faithful; the
new variant **`fast`** = identical arms/knobs/pool but with the clean shutdown, task prefix `eval4_`:

```bash
qsub -o logs/transfer_launch_fast.log -v 'VARIANT=fast,ARMS_OVERRIDE=ottertune;rgpe;opadviser_ns;opadviser' \
    scripts/lab/transfer_launch.sh
python3 scripts/report_eval2.py --prefix eval4 --arms ottertune rgpe opadviser_ns opadviser
```
Expected: ~16 evaluations/h instead of ~11 (more BO iterations inside the same 1 h). Results are a
new baseline (starting state per trial differs from kill -9 runs); compare arms within the wave.
Collection inis (`config_collect_*`) also pick up the default from now on.

### 11a. First `fast` wave aborted; doublewrite decides the flush cost (2026-08-29 13:24–13:44)

Job 3202623 (eval4, 4 arms): with the plain clean shutdown every **doublewrite=ON** config SMAC
explored took ≥168 s to flush (168, 178, 4 × >180 s → kill -9 fallback), doublewrite=OFF configs
11–33 s. SMAC picks doublewrite=ON in 52 % of configs, so the naive policy was slower than kill -9
for half of all iterations; aborted after 20 min. Follow-up micro-test (job 3204312, the two slow
configs A/B, ~300–390 k dirty pages, `test_restart_policy.sh` with `RT_POL/RT_CFG/RT_HIST`):

| config (dw=ON) | clean (fast_shutdown=1) | clean + `innodb_doublewrite=DETECT_ONLY` | kill -9 recovery |
|---|---|---|---|
| A (20 GB pool, io_cap 1.57M) | 111 s | 132 s, 144 s | 271 s (smoke) |
| B (16 GB pool, flush_neighbors=0) | 214 s | 225 s | 169 s |

DETECT_ONLY does not help (the per-batch fsync pattern remains; OFF cannot be set at runtime).
Historical kill -9 dead time after dw=ON configs: median 119–153 s, mean 178–204 s, max ~595 s.
**Rule now in `_kill_mysqld`:** estimate flush = dirty pages ÷ (40 k/s if dw=OFF, 1.4 k/s if ON);
clean shutdown when ≤450 s (600 s budget), else kill -9. Validated in the loop (job 3206958):
`dirty_pages=42625 → clean, 26 s`; `dirty_pages=295425 → kill -9`. Expected per-iteration dead time
≈25 s for dw=OFF configs, 110–225 s (bounded) for dw=ON, vs 175–200 s mean before.

### 11b. eval4 wave result (job 3208151, 2026-08-29 15:29–18:35)

```
arm                         seed 42               seed 43               seed 44     mean   median
ottertune      2264 (n=13, t=1577s) 11157 (n=13, t=2103s)  3451 (n=11, t=2074s)     5624     3451
rgpe           5629 (n=14, t=3572s) 12358 (n=13, t=2133s)  3289 (n=11, t=1944s)     7092     5629
opadviser_ns   1723 (n=13, t=3341s)  3943 (n=14, t= 415s) 13942 (n=12, t=3275s)     6536     3943
opadviser     20004 (n=13, t=1246s) 20464 (n=15, t=1271s) 20065 (n=13, t= 834s)    20178    20065
```
Dead time per iteration 43 s median / 92 s mean (run 2: 140 / 196); 158 clean shutdowns
(median 15 s, p90 205 s, max 405 s), 4 kill -9 by rule, 0 budget fallbacks, 2 FAILED trials.
Write-up: exp_notes F7d.
