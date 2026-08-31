# MIYABI.md — Miyabi-C (JCAHPC) environment: facts, gotchas & findings

Everything specific to running this project on the **Miyabi-C** supercomputer (JCAHPC).
This is the home for cluster/environment specifics so the other docs stay portable:
- **[CLAUDE.md](CLAUDE.md)** — DBTune code/architecture (environment-independent).
- **[SETUP.md](SETUP.md)** — the DML-vs-OtterTune experiment design, setup recipe, and workflow.
- **[DML.md](DML.md)** — DML embedding model files & retraining.

---

## 1. Cluster & access

- **Host:** login node `miyabi-c1`; compute via PBS (no Docker — see below).
- **Project code:** `xg26g002` (passed as `-W group_list=xg26g002`).
- **Python:** Intel oneAPI Python 3.9 (system); the project uses its own venv at
  `/work/xg26g002/x10563/DBTune/venv`.
- **Modules available:** `cmake`, `ninja`, `gcc 11.4`, `autoconf/automake/libtool`, Java 11,
  `apptainer`/`singularity`.
- **Docker is NOT available** (HPC security model). Use native binaries or apptainer.

## 2. Compute-node hardware & storage

- **Node-exclusive** (one job owns the whole node) → the hard-coded ports
  (3306 / 9090 / 9104 / 9100) never collide on compute nodes.
- Observed node (`mc001`): **~124 GB RAM** (≈118 GB free at idle), many-core.
- **`/work` is Lustre** — a parallel/network filesystem over InfiniBand
  (`172.16.20.201@o2ib200:…:/lustre/work`, ~9.9 PB). **The repo and the mysqld datadir
  (`mysql_build/data`) live here**, so all DB I/O is Lustre RPCs over the network.
- **Node-local `/tmp` = `/dev/sda8`, xfs, ~14 GB** (1% used; ~385 MB/s sequential write).
  Too small for the ~16 GB datadir, which is why the datadir stays on Lustre.
- **No `/local`** on Miyabi-C (that exists only on Miyabi-G). RAM-backed tmpfs is an option
  for a datadir but does **not** fix the disk-IOPS measurement (see §5).
- Probe script: [scripts/lab/probe_tmp.sh](scripts/lab/probe_tmp.sh) /
  [scripts/lab/probe_shm.sh](scripts/lab/probe_shm.sh) characterize node-local storage.

## 3. PBS

- Queues: `short-c` (1–2 nodes, 8 h), `debug-c` (short). `prepost` rejects batch jobs
  (interactive only).
- **Per-project concurrent-run cap = RUN=2.** Separate jobs don't parallelize past 2 — to run
  more configs at once, pack them into ONE multi-node job and dispatch with `pbsdsh`
  (see SETUP.md §5 "Running the configs in parallel" and the
  [miyabi-submit-job skill](.claude/skills/miyabi-submit-job.md)).
- Jobs sometimes land in **HELD** state after a force-killed prior run → release with
  `qrls <jobid>` (or just submit fresh; a new job id usually avoids it).
- Job output goes to a Lustre log (`logs/run_tuning.log`); **delete it before re-submitting**
  to avoid Lustre append issues.

## 4. Build / runtime constraints (these drive the setup recipe)

MySQL 8.0.44 won't build on Miyabi without these flags (full recipe in SETUP.md §3.2):
- `-DWITH_BOOST=./boost` (downloads boost 1.77.0, required by 8.0.44)
- `-DWITH_TIRPC=bundled` (no system `libtirpc-dev`)
- `-DWITHOUT_GROUP_REPLICATION=1` (no `rpcgen` available)
- The `Server` install component errors on the X-plugin test driver → install
  `Client`/`Common`/`Development` only (enough for `mysqld`, client, headers, libs).

Prometheus stack runs as **native binaries, not containers**: Docker is unavailable, and
[autotune/database/mysqldb.py](autotune/database/mysqldb.py) asserts `pgrep -x mysqld_exporter`
— a real host process literally named `mysqld_exporter`, which the apptainer wrapper process
breaks. So the native release binaries live under `tools/` (see SETUP.md §3.3).

## 5. Storage gotchas & findings (affect how to interpret results)

- **Disk-IOPS metrics read ~0 on Miyabi — a measurement artifact, not a workload effect.**
  The two disk metrics (`Average Disk IOPS Read/Write`) are collected as
  `max(rate(node_disk_{reads,writes}_completed_total[60s]))`
  ([autotune/optimizer/dml_metrics.py](autotune/optimizer/dml_metrics.py)), which node_exporter
  sources from `/proc/diskstats`. But the mysqld datadir is on **Lustre** (§2), so DB I/O is
  Lustre RPCs that **never appear in `/proc/diskstats`**. The only local disk is `/dev/sda8`
  (the 14 GB `/tmp`), and node_exporter's default
  `--collector.diskstats.device-exclude` regex `^(z?ram|loop|fd|(h|s|v|xv)d[a-z]|nvme\d+n\d+p)\d+$`
  **excludes `sda8`** anyway. Net: both read & write IOPS read ~0–1 ops/s for every workload.
  This is a mismatch with the source `*-result.csv` (collected where mysqld sat on a real local
  block device → hundreds–thousands of IOPS), which is why those two features are
  "dead"/out-of-distribution in the DML embedding (see DML.md "Hardware-blindness" and the
  source-selection analysis). **tmpfs would NOT fix it** (RAM → no `/proc/diskstats` activity).
  For a real I/O signal you'd need Lustre client stats (`/proc/fs/lustre/llite/*/stats` or a
  `lustre_exporter`); otherwise drop the 2 disk-IOPS features and retrain on 9.
- **`Max CPU Usage` reads anomalously low for write/rw50.** The metric is
  `(1 - min_over_cpus(idle_rate)) * 100` (busiest single core). With only 4 client threads on a
  many-core node, no core saturates, so it lands *below the source min* for write-heavy targets
  — another out-of-distribution input feature for the embedding.
- **Restart between trials needs a clean datadir.** A force-killed mysqld leaves a large dirty
  redo log; the next default-config startup must shrink + recover it, which is **slow on Lustre**.
  [scripts/lab/reset_database.sh](scripts/lab/reset_database.sh) allows up to
  `RESET_START_TIMEOUT` (default 1200 s). When changing table size you reload anyway, so wiping
  `mysql_build/data` gives a guaranteed clean start.
- **Warmup is slow on Lustre.** The buffer pool starts cold; a large (10–25 GB) pool warms
  slowly on Lustre. Too-short warmup (5 s) measures large-pool configs cold and *understates*
  them; TPS plateaus by ~30–60 s — set `workload_warmup_time` accordingly.
- **Parallel runs race on the Lustre cache.** `tuner.load_history` builds
  `<data_repo>/_history_cache.pkl` on first use; concurrent `pbsdsh` tasks racing to build it on
  shared Lustre clobber each other. **Pre-build it once** before a parallel job (`exclude_contexts`
  filters in-memory, so the cache stays the full pool).

## 6. Performance note

- Tuning iterations with sysbench RW can take **~10 minutes**. The `innodb_log_file_size`
  (5 GB → 48 MB) restart hypothesis was **disproven** — a standalone test
  (`scripts/test_restart_slowdown.sh`) showed that restart adds only ~33 s (30 s shutdown + 3 s
  startup). The real bottleneck is elsewhere (likely Lustre I/O latency during warmup/recovery).
- **Throughput differs by tens of percent between days/node sets for identical configs.**
  The 1 h-deadline wave re-run on 2026-08-27 (mc057–060) measured the default config at
  76–112 tps vs 138–245 on 2026-07-07 (mc024/184/185/186), and OpAdviser's identical replayed
  source-best configs at 10.8–14.8k vs 18.9–19.5k (−22…−60 %), consistently on all four
  nodes. Never compare absolute tps across runs/days; compare arms within one wave or
  normalise by a within-run reference config (exp_notes F7c).
