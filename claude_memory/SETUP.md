# DBTune — DML vs OtterTune Experiment Setup

This document describes the transfer-tuning comparison experiment run on the
**Miyabi-C** supercomputer (JCAHPC), the goal, the one-time environment setup
(MySQL, sysbench, oltpbench, the Prometheus stack, the Python venv), and the
scripts that orchestrate and report the experiment.

> **Cluster/environment specifics** — node hardware, storage layout, PBS, build constraints,
> and Lustre gotchas (including why disk-IOPS reads ~0) — live in **[MIYABI.md](MIYABI.md)**.

---

## 1. Goal of the experiment

Compare two **one-shot transfer-tuning** methods — **DML** (Deep Metric
Learning) and **OtterTune** (workload mapping) — on three sysbench OLTP
workloads, and measure **how much each method improves throughput (TPS) using
prior tuning history**, with no new search budget on the target.

Each experiment is a 2-iteration session:

| iter | config applied | purpose |
|------|----------------|---------|
| 0 | **default** (the 12 `DML_12.json` defaults: 1 GB buffer pool, etc.) | baseline TPS |
| 1 | **transfer-learned** config (from prior history) | improved TPS |

Improvement = `tps(iter1) / tps(iter0)`. The comparison runs both methods over
3 workloads (read / readwrite-50 / write-only) → **6 experiments**.

> The deeper question driving the current work is **where the DML↔OtterTune
> discrepancy comes from — the source-selection step or the downstream
> performance model**. That ablation is documented in **§8**.

- **DML** ([autotune/optimizer/dml_optimizer.py](autotune/optimizer/dml_optimizer.py)):
  embeds the target's iter-0 internal metrics, finds the nearest *source
  context*, then (as of this work) trains a **GP** on that context's
  `(config → TPS)` data and recommends the predicted-best config.
- **OtterTune** ([autotune/transfer/tlbo/workload_map.py](autotune/transfer/tlbo/workload_map.py)):
  `optimize_method=SMAC` + `transfer_framework=workload_map` (`mapping_method=ottertune`).
  Maps the target to the most similar source workload by binned internal
  metrics, builds a surrogate on source+target data, and proposes the next
  config via Bayesian optimization.

---

## 2. Environment

The cluster, node hardware, storage layout, available modules, and the Docker-unavailable
constraint are in **[MIYABI.md](MIYABI.md)** (§1–§2). In brief: Miyabi-C login node
`miyabi-c1`, PBS compute, project `xg26g002`, its own venv; **no Docker**.

---

## 3. One-time setup

All paths are relative to the repo root `/work/xg26g002/x10563/DBTune`.

### 3.1 Submodules

```bash
git submodule update --init third_party/mysql-server third_party/sysbench third_party/oltpbench
```
Pinned: `mysql-server` = **mysql-8.0.44**, `sysbench` = **1.1.0** (3ceba0b),
`oltpbench` = Apache OLTPBench (ant-based).

### 3.2 Build MySQL, sysbench, oltpbench, venv — `scripts/lab/setup_build.sh`

Run once (heavy build; do it from the login node or a compute job):

```bash
BUILD_JOBS=48 bash scripts/lab/setup_build.sh    # ~30-60 min
```

What it does, and the gotchas it works around:

1. **MySQL 8.0.44** — `cmake` + `ninja` into `mysql_build/`, then
   `cmake --install --prefix mysql_build` (FHS layout: `bin/`, `lib/`, `include/`).
   Key flags (the build fails on Miyabi without them):
   - `-DWITH_BOOST=./boost` (downloads **boost 1.77.0**, required by 8.0.44)
   - `-DWITH_TIRPC=bundled` (no system `libtirpc-dev`)
   - `-DWITHOUT_GROUP_REPLICATION=1` (no `rpcgen` available)
   - `-DCMAKE_BUILD_TYPE=Release`, storage-engine excludes
   - Note: the `Server` install component errors on the X-plugin test driver;
     the script installs `Client`/`Common`/`Development` components which is
     enough (`mysqld`, `mysql`, `mysql_config`, headers, `libmysqlclient.so.21`).
2. **sysbench 1.1** — `./autogen.sh && ./configure --with-mysql=mysql_build --without-pgsql && make`,
   installed to `sysbench_install/`. Links against the just-built
   `libmysqlclient.so.21` — runtime needs `LD_LIBRARY_PATH=mysql_build/lib`.
3. **oltpbench** — installs **Apache Ant 1.10.15** locally under `tools/`, then
   `ant bootstrap && ant resolve && ant` (Ivy fetches deps). Build lands in
   `third_party/oltpbench/build/`.
4. **Python venv** — `python3 -m venv venv`, installs `requirements.txt` plus
   runtime extras the requirements miss: `dill cma terminaltables hiplot
   tensorboardX` and canonical `mysql-connector-python`, then `pip install
   --no-deps -e .` so `autotune` is importable.

### 3.3 Prometheus stack (native binaries, not Docker)

DBTune collects 11 internal metrics from Prometheus (`internal_metrics_source =
prometheus`; the source CSVs have 114 Prometheus metrics, not the 65 from
`information_schema`), so a Prometheus + exporters stack is required.

**Why native binaries, not containers:** Docker is unavailable on Miyabi.
Apptainer/Singularity work and SIFs were pulled to `containers/`, **but**
[autotune/database/mysqldb.py](autotune/database/mysqldb.py) asserts
`pgrep -x mysqld_exporter` — it requires a real host process literally named
`mysqld_exporter`, which Apptainer's wrapper process breaks. So we run the
native release binaries, downloaded under `tools/`:

- `tools/prometheus/prometheus`            (Prometheus 3.0.1)
- `tools/mysqld_exporter_dir/mysqld_exporter` (0.15.1)
- `tools/node_exporter_dir/node_exporter`     (1.8.2)

Config files: [scripts/lab/prometheus.yml](scripts/lab/prometheus.yml) (scrapes
`localhost:9104`/`:9100`, relabels instances to `mysqld-exporter:9104` /
`node-exporter:9100` to match the source-data labels) and
[scripts/lab/exporter_my.cnf](scripts/lab/exporter_my.cnf) (mysqld_exporter
client credentials).

---

## 4. Experiment configuration files

Six ini files in `scripts/`, one per (method × workload):

```
config_sysbench_dml_read.ini    config_sysbench_ot_read.ini
config_sysbench_dml_rw50.ini    config_sysbench_ot_rw50.ini
config_sysbench_dml_write.ini   config_sysbench_ot_write.ini
```

Shared `[database]` settings (current run):
- `knob_config_file = ./experiment/gen_knobs/DML_12.json` — **12 InnoDB knobs**
  (buffer_pool_size/instances, read/write_io_threads, flush_log_at_trx_commit,
  adaptive_hash_index, sync_binlog, lru_scan_depth, change_buffer_max_size,
  io_capacity, log_file_size, table_open_cache).
- `workload = sysbench`, `workload_type = read | readwrite | write`
  (→ sysbench `oltp_read_only | oltp_read_write | oltp_write_only`).
- `thread_num` = **clients** (sysbench `--threads`).
- `sysbench_tables = 64`, `sysbench_table_size` = rows/table.
- `workload_warmup_time`, `workload_time` (seconds).
- `online_mode = False` → DBTune **restarts mysqld between trials**.

Method-specific `[tune]` settings:
- DML: `optimize_method = DML`, `transfer_framework = dml`, `dml_model_path`,
  `dml_context_metrics_path`, `dml_result_data_dir`.
- OtterTune: `optimize_method = SMAC`, `transfer_framework = workload_map`,
  `mapping_method = ottertune`, `data_repo = ./DBTune_history/csv_source_no_<wl>`.

**To match the source-data collection conditions** (recommended for a fair
comparison): set `thread_num = 4` and `sysbench_table_size = 1000000` — the
source contexts were collected at **4 clients, 64×1M-row tables** (the label
`64-1000000-4-oltp_...` encodes `tables-tablesize-clients-workload`).

---

## 5. Running the experiment

Submitted as a single PBS job on `short-c`:

```bash
cd /work/xg26g002/x10563/DBTune
qsub scripts/lab/run_tuning.sh
```

[scripts/lab/run_tuning.sh](scripts/lab/run_tuning.sh) orchestrates everything:

1. `source venv/bin/activate`; export `LD_LIBRARY_PATH`, `SYSBENCH_BIN`, `MYSQL_SOCK`.
2. **`reset_database.sh config_sysbench_dml_read.ini`** — write the default cnf
   from the ini's `knob_config_file`, restart mysqld on it, purge binlogs.
3. **`init_sbtest.sh`** — ensure `sbtest` is loaded at `SBTEST_TABLES`×`SBTEST_TABLE_SIZE`
   (drops & reloads if the row count changed).
4. **`start_stack.sh`** — launch mysqld_exporter / node_exporter / prometheus.
5. Run each of the 6 configs: `cd scripts && python optimize.py --config=<ini>`,
   logging to `logs/runs/<config>.log`. Continues on per-config failure.
6. `trap`-driven `stop_stack.sh` on exit.

### Supporting scripts (`scripts/lab/`)

| script | role |
|--------|------|
| [setup_build.sh](scripts/lab/setup_build.sh) | one-time build of MySQL + sysbench + oltpbench + venv |
| [reset_database.sh](scripts/lab/reset_database.sh) | **reusable**: ini → write `my.cnf.default` (bare base + the 12 knob defaults) → restart mysqld on default → purge binlogs. Owns mysqld lifecycle. |
| [init_sbtest.sh](scripts/lab/init_sbtest.sh) | **data only** (assumes mysqld running): create `sbtest`, `sysbench prepare`; reloads if `--table-size` changed |
| [start_stack.sh](scripts/lab/start_stack.sh) / [stop_stack.sh](scripts/lab/stop_stack.sh) | start/stop the native Prometheus + exporters |
| [run_tuning.sh](scripts/lab/run_tuning.sh) | PBS job; runs all 6 experiments end-to-end |
| [report.py](scripts/lab/report.py) | print per-workload tables: `selected_source` + `tps` + all 12 knobs, one column per method×iteration |
| [diag_default.sh](scripts/lab/diag_default.sh) | verify default config is applied (SHOW GLOBAL VARIABLES) + rw50 determinism |
| [diag_warmup.sh](scripts/lab/diag_warmup.sh) | cold-start TPS ramp to test warmup adequacy / buffer-pool warming |
| [prometheus.yml](scripts/lab/prometheus.yml), [exporter_my.cnf](scripts/lab/exporter_my.cnf) | Prometheus + exporter config |

### Running the configs in parallel (one PBS job, N nodes)

`run_tuning.sh` runs configs **sequentially**. To run them **simultaneously** — the Miyabi
per-project cap is **RUN=2** so separate jobs don't help (see the
[miyabi-submit-job skill](.claude/skills/miyabi-submit-job.md)) — pack them into ONE multi-node
job and dispatch one task per node with `pbsdsh`:

| script | role |
|--------|------|
| [par_launch.sh](scripts/lab/par_launch.sh) | `select=N` PBS job; `pbsdsh -n i` runs one `method:wl:seed` task per node. Tasks via the script's `TASKS` array or `TASKS_OVERRIDE` (`;`-separated). |
| [par_node_run.sh](scripts/lab/par_node_run.sh) | per-node isolated runner: **own mysqld** on a unique `/work` Lustre datadir (copied from the template) + node-local socket/stack/cnf; exports `WM_SOURCE_ONLY=1`, `REC_ARGMAX_MEAN=1`, `DBTUNE_SEED`. Writes `history_<method>_<wl>_s<seed>.json`. |
| [par_gen_task.py](scripts/lab/par_gen_task.py) | per-task `config.ini` (retargets `[database]` paths + `task_id` from the base `config_sysbench_{ot,dmlmap}_{wl}.ini`). |
| [build_loo_embedding.py](scripts/lab/build_loo_embedding.py) | cached per-target leave-one-out DML embedding (filters the target out of the triplet+context CSVs, then trains). |

```bash
qsub scripts/lab/par_launch.sh   # select=6 + the 6 tasks (ot/dml × read/rw50/write @ seed 42) are baked in
```
Pre-build the `data_repo` `_history_cache.pkl` once (load all source JSONs → pickle) to avoid a
parallel cache-build race; the `exclude_contexts` filter is applied in-memory so the cache stays
the full pool.

### Reporting

```bash
python3 scripts/lab/report.py [read rw50 write]
```
Reads the per-task histories `scripts/DBTune_history/history_{dml,ot}_sysbench_<wl>.json`
and the matched-context log lines, and prints a table per workload.

---

## 6. PBS notes (Miyabi-C)

In **[MIYABI.md](MIYABI.md)** §3: queues, the **RUN=2** per-project concurrent cap, HELD-state
recovery (`qrls`), node-exclusive ports, and the Lustre log-append caveat.

---

## 7. Gotchas / lessons (important for interpreting results)

> **Environment/Lustre gotchas** — disk-IOPS reading ~0, slow restart-recovery & buffer-pool
> warmup on Lustre, the low CPU% metric, and the parallel cache race — are in
> **[MIYABI.md](MIYABI.md)** §5. Below are the experiment/DBMS-level gotchas.

- **Buffer-pool dump/restore confounds baselines.** MySQL restores the previous
  shutdown's pool on startup, giving small configs a warm head start. Disable
  with `innodb_buffer_pool_dump_at_shutdown=OFF` / `..._load_at_startup=OFF` in
  `my.cnf.clean` for an equal cold start.
- **`innodb_log_file_size` is deprecated (8.0.30) but still functional** — it
  computes `innodb_redo_log_capacity` *only if the latter is unset*.
- **Durability knobs dominate write-heavy workloads.** `flush_log_at_trx_commit`
  and `sync_binlog` = 0/2 (relaxed) avoid per-commit fsync and yield large TPS
  gains on rw50/write — but sacrifice durability. Hold them constant if you want
  an apples-to-apples speed comparison.
- **TPS vs QPS across workloads.** sysbench `write_only` transactions are lighter
  (4 point writes) than `read_only` (14 SELECTs incl. range scans), so write_only
  can show higher *TPS* while read_only does more *QPS*. Compare QPS across
  workload types.
- **Source-data labels** `64-1000000-4-oltp_<wl>-<ratio>` =
  `tables-tablesize-clients-workload-ratio`. The target should match
  (`thread_num`, `sysbench_table_size`). A task_id-derivation bug that truncated
  the `<ratio>` (`f.split('.')[0]`) was fixed to `os.path.splitext(f)[0]` in
  [autotune/tuner.py](autotune/tuner.py) so all source contexts load distinctly.

---

## 8. Ablation: is the DML↔OtterTune gap source selection or the downstream model?

### 8.1 What we're asking

DML and OtterTune are both **two-stage** one-shot transfer tuners:

1. **Source selection** — pick the most similar historical *context* from the
   source data, using the target's iter-0 (default-config) internal metrics.
2. **Downstream performance model** — build a model of `config → performance`
   from the selected source (plus the target's one observed point) and emit a
   recommended config.

| | Source selection | Downstream model |
|---|---|---|
| **DML** | triplet **embedding** (neural net) on 11 metrics → nearest context (Euclidean in embedding space) | originally: **GP** trained on the source context's `(config→TPS)` rows, recommend **argmax of predicted mean** (exploitation only) |
| **OtterTune** | **decile-binned** internal metrics → Euclidean distance → nearest source (SIGMOD'17 §6.1) | **Random Forest** surrogate (`prf` — *probabilistic random forest*, an ensemble of regression trees; **not** a single decision tree) on **target+source** data, recommend via **Bayesian optimization with EI** acquisition |

> **Note on OtterTune's model:** the original OtterTune paper used a Gaussian
> Process. In *this* codebase the OT configs use `optimize_method = SMAC`, whose
> surrogate is a **random forest** (`RandomForestWithInstances`, sklearn
> `RandomForestRegressor` fallback). With `optimize_method = MBO` it would be a
> GP instead. The acquisition is Expected Improvement (`acq_type=auto`→`ei`).

Because the two methods differ in **both** stages, a head-to-head TPS gap
doesn't tell us *which* stage is responsible. The ablation separates them.

### 8.2 How we isolate the two stages

We compare **three** configurations, all on the same target (4 clients, 64×1M
rows, zipfian skew 0.2 — matching the source-data collection):

| method | source selection | downstream model | history file |
|--------|------------------|------------------|--------------|
| **DML** | embedding | GP argmax-mean (DML's own) | `history_dml_sysbench_<wl>.json` |
| **dmlmap** | **embedding** | **OtterTune's** RF + EI BO | `history_dmlmap_sysbench_<wl>.json` |
| **OtterTune** | binning | OtterTune's RF + EI BO | `history_ot_sysbench_<wl>.json` |

- **dmlmap vs OtterTune** → *same downstream*, differ only in **source
  selection** ⇒ isolates the source-selection effect.
- **DML vs dmlmap** → *same source selection*, differ only in **downstream** ⇒
  isolates the downstream-model effect.

`dmlmap` is implemented as a new `WorkloadMapping` mapping method
(`mapping_method = dml`) so it reuses OtterTune's **exact** downstream
(`build_single_surrogate` PRF + `BO_Optimizer` EI), with only the source
selection swapped to the DML embedding:
- [autotune/transfer/tlbo/workload_map.py](autotune/transfer/tlbo/workload_map.py):
  `_load_dml_embedding`, `_dml_select_source`, `_train_dml` (selects the source
  via the embedding, maps it to a `source_dict` key, then calls the shared
  `_finalize_match` downstream).
- Wired through [tuner.py](autotune/tuner.py) (`mapping_method=dml` →
  `surrogate_type=tlbo_dmlmap_<method>`),
  [surrogate/core.py](autotune/optimizer/surrogate/core.py) (`dmlmap` dispatch),
  and [pipleline.py](autotune/pipleline/pipleline.py) (bridges the DML model /
  Prometheus params to `WorkloadMapping` via `DMLMAP_*` env vars).
- Configs: `scripts/config_sysbench_dmlmap_{read,rw50,write}.ini` (OT-style:
  `optimize_method=SMAC`, `transfer_framework=workload_map`, `mapping_method=dml`,
  `data_repo=csv_source_no_<wl>`, plus `dml_model_path` /
  `dml_context_metrics_path`).

Two extra controls in [dml_optimizer.py](autotune/optimizer/dml_optimizer.py)
support finer ablations (used in intermediate steps, not required for the main
result):
- `DML_DOWNSTREAM=gp|ottertune` — switch DML's own recommender between
  GP-argmax and a self-contained GP+EI re-implementation of OT's downstream.
  (Caveat: that re-implementation uses a **GP**, so it is *not* identical to
  OT's **RF** — which is exactly why the faithful comparison uses `dmlmap`.)
- `FORCE_SOURCE_CONTEXT=<hw>_<workload>-<ratio>` — pin DML's source context
  (compact label, e.g. `88c190g_read_write_80-0.2`) to test a chosen source.

### 8.3 How to run it

```bash
# dmlmap (DML embedding source + OtterTune's real RF+EI downstream) for all 3 workloads.
# CONFIGS_OVERRIDE is ':'-separated (PBS -v treats ',' as a variable separator).
qsub -v "CONFIGS_OVERRIDE=config_sysbench_dmlmap_read.ini:config_sysbench_dmlmap_rw50.ini:config_sysbench_dmlmap_write.ini" \
     scripts/lab/run_tuning.sh

# OtterTune baseline (if not already present): the ot_* configs (default CONFIGS list).
# Optional finer ablations on the dml_* configs:
qsub -v "DML_DOWNSTREAM=ottertune" scripts/lab/run_tuning.sh                 # DML + GP+EI downstream
qsub -v "FORCE_SOURCES=1,DML_DOWNSTREAM=ottertune" scripts/lab/run_tuning.sh # DML forced to a pinned source
```

The run-time conditions (clients, table size, skew, warmup) are set in the inis
and in [run_tuning.sh](scripts/lab/run_tuning.sh) (`SYSBENCH_ZIPFIAN_EXP=0.2`).

### 8.4 Reporting

```bash
REPORT_METHODS=dmlmap:ot  python3 scripts/lab/report.py    # dmlmap vs OtterTune (the clean isolation)
REPORT_METHODS=dml:dmlmap:ot python3 scripts/lab/report.py # all three
```
`report.py` prints, per workload, a `selected_source` row (read from each run's
history `context.matched_context`), a `tps` row, and all 12 knob values — one
column per method×iteration.

### 8.6 Refinement (4 clients, 64×1M rows, zipfian 0.2)

**(a) The downstream is provably identical**
Once a source is chosen, `_train_ottertune` and `_train_dml` both call the **same**
`_finalize_match` → same `build_single_surrogate` (PRF/GP) + same `BO_Optimizer` EI. Offline
tests ([scripts/tmp/test_downstream_identity.py](scripts/tmp/test_downstream_identity.py),
[test_same_seed_suggestion.py](scripts/tmp/test_same_seed_suggestion.py)) confirm: with the
**same source + same seed + same iter-0 target**, ot and dmlmap emit a **byte-identical** config.
But DBTune ran **unseeded**, and with `max_runs=2` (one BO step) EI's random candidates fall far
from the ~1300 source rows where the surrogate is ~flat, so the single recommendation is set by
the **acquisition RNG, not the source**. A seeded natural-selection run produced **one identical
config across all selected sources** — i.e. **source selection is washed out** by this downstream.

**(b) Reproducibility fix.** `tuner.py` now seeds the pipeline via `rand_seed` / `DBTUNE_SEED`
(default 42). Set it for reproducible comparisons.

**(c) Source-driven downstream.** To actually measure source-selection quality, switch the
recommender to **argmax of the surrogate's predicted mean** on a **source-only** surrogate
(`REC_ARGMAX_MEAN=1` + `WM_SOURCE_ONLY=1`, GP via `optimize_method=MBO`). Then a different source
→ a different config (verified). The DML embedding training data was also expanded from **1M-only
to 1M + 100k** (293 contexts; `dml_models_all/`, see [DML.md](DML.md)).

iter-1 TPS (seed 42, GP + argmax-mean + source-only; **bold = winner**):

| setup | read OT/dml | rw50 OT/dml | write OT/dml | takeaway |
|---|---|---|---|---|
| leave-one-**workload**-out, 1M-only embedding | **3533**/3199 | **962**/957 | **1370**/623 | OT wins all; DML picks small-HW / cross-family sources (embedding is **hardware-blind**) |
| **all-source** embedding + full `csv_source` pool | **3467**/3078 | 382/**886** | 1167/**1271** | DML wins rw50 (+132%) & write (+9%); reverses the row above |
| **strict leave-one-target-out** (remove `1M/<wl>/0.2` from pool **and** embedding) | **3425**/3212 | 344/**435** | **1180**/1097 | roughly even; DML wins rw50 (+26%), OT narrowly read/write (~7%, within noise) |

Latest run in full (the **strict leave-one-target-out** row above; PBS job `1987134`,
`report.py`-style — histories `history_{ot,dml}_<wl>_s42.json`). iter-0 = default config, iter-1 =
transferred config; `selected_source` is the compact label, `src_table_size` the source's
rows/table:

```
==================================== sysbench read (LOO: 1M/read/0.2 removed) ====================================
row                                  OT i0                   OT i1                  DML i0                  DML i1
------------------------------------------------------------------------------------------------------------------
selected_source                  (default)    32c64g_read_only-0.6               (default)      4c6g_read_only-0.6
src_table_size                   (default)                  100000               (default)                 1000000
tps                                  335.2                  3424.7                  3149.5                  3211.6
bufpool_size                    1073741824             19327352832              1073741824              4294967296
read_io_thr                              2                      15                       2                       2
write_io_thr                             2                       2                       2                       2
flush_log_commit                         1                       1                       1                       1
adaptive_hash                            1                       1                       1                       1
sync_binlog                              1                       1                       1                       1
lru_scan_depth                        1024                    1024                    1024                    5000
bufpool_inst                             1                       1                       1                       1
chg_buf_max                             25                      25                      25                      25
io_capacity                            100                     100                     100                     100
log_file_size                     50331648                50331648                50331648                50331648
table_open_cache                      4000                    4000                    4000                    4000

==================================== sysbench rw50 (LOO: 1M/rw50/0.2 removed) ====================================
row                                  OT i0                   OT i1                  DML i0                  DML i1
------------------------------------------------------------------------------------------------------------------
selected_source                  (default)32c64g_read_write_50-0.2               (default)16c24g_read_write_80-0.6
src_table_size                   (default)                  100000               (default)                  100000
tps                                  335.2                   344.2                   334.5                   435.0
bufpool_size                    1073741824              3221225472              1073741824              4294967296
read_io_thr                              2                       2                       2                       2
write_io_thr                             2                       2                       2                       2
flush_log_commit                         1                       1                       1                       0
adaptive_hash                            1                       1                       1                       1
sync_binlog                              1                       1                       1                       1
lru_scan_depth                        1024                    1024                    1024                    1024
bufpool_inst                             1                       1                       1                       1
chg_buf_max                             25                      25                      25                      25
io_capacity                            100                     100                     100                     100
log_file_size                     50331648              5242880000                50331648                50331648
table_open_cache                      4000                    4000                    4000                    4000

=================================== sysbench write (LOO: 1M/write/0.2 removed) ===================================
row                                  OT i0                   OT i1                  DML i0                  DML i1
------------------------------------------------------------------------------------------------------------------
selected_source                  (default)  88c190g_write_only-0.6               (default) 24c32g_read_write_5-1.0
src_table_size                   (default)                 1000000               (default)                 1000000
tps                                  457.7                  1180.0                   457.3                  1096.7
bufpool_size                    1073741824              8589934592              1073741824              6442450944
read_io_thr                              2                       2                       2                       2
write_io_thr                             2                       2                       2                       2
flush_log_commit                         1                       1                       1                       1
adaptive_hash                            1                       1                       1                       1
sync_binlog                              1                       1                       1                       1
lru_scan_depth                        1024                    1024                    1024                    1024
bufpool_inst                             1                       1                       1                       1
chg_buf_max                             25                      25                      25                      25
io_capacity                            100                     100                     100                     100
log_file_size                     50331648              5242880000                50331648              5242880000
table_open_cache                      4000                    4000                    4000                    4000
```

Notes from the detail: OT's rw50 fallback is the **100k** `read_write_50-0.2` twin (the 1M one was
removed) — a *near-identical-workload* substitute, which is why this narrow LOO is "easy"; DML's
rw50 pick is a different ratio (`read_write_80`) yet scores higher (435 vs 344). The transferred
configs mostly move `bufpool_size` (+ `log_file_size` for OT) and leave the other knobs at
default — consistent with the argmax-mean downstream essentially recommending "the selected
source's best config".

**Force control:** pinning DML to a good big-machine source (`88c190g` 1M) closes the gap entirely
(DML ≈ OT on all 3) — confirming DML's deficit is the **selection** step, not the (identical)
downstream.

**Mechanisms / code (this refinement):**
- Source-only surrogate + argmax-mean: `WM_SOURCE_ONLY` / `REC_ARGMAX_MEAN` envs (see CLAUDE.md →
  "Transfer downstream toggles"); `FORCE_SOURCE_CONTEXT` pins a source (both ot & dml paths).
- Leave-one-out: `[tune] exclude_contexts` (drops the target's 7 hardware variants from the pool
  in-memory) + a per-target LOO embedding via
  [build_loo_embedding.py](scripts/lab/build_loo_embedding.py) → `dml_models_loo_<wl>/`.
- Pool: `data_repo = ./DBTune_history/csv_source` (full 293-context pool; both methods select
  from it).
- **Caveat:** single-seed runs (~7% gaps are within benchmark noise; rw50's +26%/+132% are
  decisive). Removing only `1M/<wl>/0.2` is a *narrow* LOO — the `100k/<wl>/0.2` twin remains, so
  a method can still fall back to a near-identical-workload source (OT did so for rw50).
