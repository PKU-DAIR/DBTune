---
name: miyabi-submit-job
description: Submit and manage PBS batch/interactive jobs on the Miyabi-C (or Miyabi-G) supercomputer at JCAHPC
user_invocable: true
---

# Submit a Job to Miyabi

## When to use
- User wants to run something on Miyabi compute nodes instead of the login node.
- User asks to write a job script, submit `qsub`, schedule a batch run, or get an interactive compute session.
- A long-running task (benchmark, training, DBTune `optimize.py`) is about to be started on a login node — stop and submit it as a job instead.

## Environment detection
First verify the environment, then pick the right system:

```bash
hostname                      # miyabi-c1 → Miyabi-C, miyabi-g1 → Miyabi-G
which qsub qstat qdel         # should be /opt/pbs/bin/...
id                            # groups list contains the project codes (e.g. xg26g002)
```

The **project code** (passed to `-W group_list=`) is one of the user's group names — typically the prefix of the path under `/work/` (e.g. `/work/xg26g002/...` → project `xg26g002`).

## Miyabi-C queues (Table 5-7)

| Queue (routing) | Real queue | Nodes (min–max) | Wall time | Mem/node |
|---|---|---|---|---|
| `debug-c` | — | 1–4 | 30 min | 118 GiB |
| `short-c` | — | 1–2 | 8 h | 118 GiB |
| `regular-c` → | `small-c` | 1–16 | 48 h | 118 GiB |
|  | `medium-c` | 17–32 | 48 h | 118 GiB |
|  | `large-c` | 33–64 | 48 h | 118 GiB |
| `interact-c` → | `intract-c_n1` | 1 | 2 h | 118 GiB |
|  | `intract-c_n2` | 2 | 10 min | 118 GiB |
| `prepost` | — | 1 | 6 h | 240 GiB (login-equivalent env, no tokens) |

Routing queues (`regular-c`, `interact-c`) auto-select the real queue from the `select=` node count.

## Miyabi-G queues (Tables 5-5, 5-6)

**Node-exclusive:** `debug-g` (1–16, 30 min), `short-g` (1–8, 8 h), `regular-g` → `small-g`/`medium-g`/`large-g`/`x-large-g` (1–256, up to 48 h), `interact-g`.
**MIG (1/4 GPU):** `debug-mig`, `short-mig`, `regular-mig`, `interact-mig` (specify `select=1|2|4` MIG instances, 25 GiB each).

## qsub options (must-know)

Basic (use as `#PBS -...` directives in scripts, or CLI flags):
- **`-q <queue>`** — queue name (required).
- **`-W group_list=<project>`** — project that pays tokens (required).
- `-o file` / `-e file` — redirect stdout/stderr (appends if file exists).
- `-j oe` — merge stderr into stdout.
- `-N <name>` — job name.
- `-I [-X]` — interactive (with X11 forwarding).
- `-J <start>-<end>[:step]` — array job (requires `-r y`).
- `-V` — inherit login env vars into the job.
- `-r y|n` — whether the job may be re-run.
- `-m abe` / `-m n` — mail on (a)bort/(b)egin/(e)nd, or none.

Resources (passed as `-l name=value`, colon-joined inside one `-l`):
- **`select=<n>`** — node count (or MIG instance count on Miyabi-G MIG queues) (required).
- **`walltime=[[hh:]mm:]ss`** — runtime cap (required; tighter values get backfilled sooner).
- `mpiprocs=<n>` — MPI procs per node.
- `ompthreads=<n>` — OpenMP threads per process.
- `mem=<limit>` — per-node memory cap.
- `filesystem=<name>` — hold job if that FS is unhealthy.

**Job ID format:** `<number>.opbs` (e.g. `123456.opbs`).
**No multi-byte chars** in env vars, paths, or job names — submission will fail.

## Job script template (Miyabi-C, sequential)

```sh
#!/bin/sh
#PBS -q debug-c
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=<PROJECT>
#PBS -j oe
#PBS -N <jobname>

cd ${PBS_O_WORKDIR}
./a.out
```

If using `module` from a bash script, the shebang **must** be `#!/bin/bash -l` (or `#!/bin/sh -l` for sh) so the login profile loads.

## Parallelism patterns

| Pattern | `-l select=...` | Launcher (Miyabi-C, Intel MPI) | Launcher (Miyabi-G, HPC-X) |
|---|---|---|---|
| Sequential | `select=1` | (none) | (none) |
| OpenMP only | `select=1:ompthreads=T` | (none) | (none) |
| MPI single-node | `select=1:mpiprocs=P` | `mpiexec.hydra ./a.out` | `mpirun ./a.out` |
| MPI multi-node | `select=N:mpiprocs=P` | `mpiexec.hydra ./a.out` | `mpirun ./a.out` |
| Hybrid MPI+OpenMP | `select=N:mpiprocs=P:ompthreads=T` | `mpiexec.hydra ./a.out` | `mpirun ./a.out` |

Total MPI ranks = `N × P`. For GCC+OpenMPI on either system, `module purge && module load gcc ompi` first and use `mpirun`.

## Running many independent tasks in parallel (embarrassingly parallel, non-MPI)

For K independent single-node tasks you want to run **at the same time** (a seed/parameter sweep, K separate `optimize.py` runs, etc.):

**Do NOT submit K separate jobs.** The per-project cap is **RUN = 2** (Table 5-9): only 2 jobs run concurrently (grows only by buying more "sets", max 8). **Array jobs don't help** — each subjob counts against RUN. So K separate jobs would run ~2 at a time.

**Pack them into ONE multi-node job** and dispatch one task per node. `select=K` (K ≤ 64) consumes **1 RUN slot** but grabs K nodes (project NODE cap = 64). Choose a queue whose node range fits K: `debug-c` ≤4, `short-c` ≤2, `regular-c`→`small-c` ≤16, `medium-c` ≤32, `large-c` ≤64. (e.g. K=12 → `-q regular-c`.)

**Dispatch with `pbsdsh`, not ssh** — passwordless ssh between compute nodes is **not** available (`Permission denied (publickey...)`). `pbsdsh -n <i> -- <prog> <args>` runs `<prog>` on the i-th allocated node (i = 0..K-1, one per node when `select=K`). Launch all in the background and `wait`:

```bash
#!/bin/bash -l
#PBS -q regular-c
#PBS -l select=12
#PBS -l walltime=01:30:00
#PBS -W group_list=<PROJECT>
#PBS -j oe
cd "${PBS_O_WORKDIR}"; ROOT="$(pwd)"
TASKS=(taskA taskB ... )                 # K entries, one per node
for i in "${!TASKS[@]}"; do
    pbsdsh -n "$i" -- /bin/bash "${ROOT}/per_node.sh" "${ROOT}" "${TASKS[$i]}" &
done
wait                                     # blocks until all per-node tasks finish
```

**`pbsdsh` launches a MINIMAL environment** — these bit us, fix them in the per-node script:
- No login profile, **no `$USER`/`$HOME`**, truncated `$PATH` (missing `/usr/sbin`,`/sbin` → tools like `ss` fail), no auto-loaded modules. Set defensively (or launch via `/bin/bash -l`):
  ```bash
  export USER="${USER:-$(id -un)}"; export HOME="${HOME:-/home/$USER}"
  export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:<tool dirs>"
  ```
- `pbsdsh`'s own exit code is unreliable; have each task write a success marker (e.g. a `.rc` file) and check those after `wait`.

**Per-task isolation on shared Lustre** — every node sees the same `/work` and `/home`, so any path a task **writes** must be **unique per task** (datadirs, sockets, pidfiles, generated configs, logs); otherwise concurrent tasks clobber each other (and two servers on one datadir corrupt it). **Ports can be reused** across tasks — Miyabi-C is node-allocated, so each node is private.

**Node-local scratch is small/absent on Miyabi-C**: `$TMPDIR=/tmp/<jobid>.<n>` is a real local xfs disk but **only ~14 GB**; the fast NVMe `/local` (~900 GB) is **Miyabi-G only** (Miyabi-C's spec lists no internal disk). For large per-task working sets, use **unique `/work` Lustre paths**. Avoid RAM-backed `/dev/shm` if it changes what you measure (it makes `fsync` ~free, distorting DB durability/throughput).

**Worked example in this repo:** [`scripts/lab/par_launch.sh`](../../scripts/lab/par_launch.sh) (`select=N` launcher) + `par_node_run.sh` (per-node isolated runner) + `par_gen_task.py` (per-task config/cnf generator) run 12 DBTune `optimize.py` tasks in one wave (~12 min vs ~1.5 h sequential), each with its own mysqld on a unique Lustre datadir + node-local socket/stack.

## Auto-loaded environment
- **Miyabi-C batch jobs:** Intel oneAPI (`intel`, `impi`) is auto-loaded — no `module load` needed.
- **Miyabi-G batch jobs:** NVIDIA HPC SDK (`nvidia`, `nv-hpcx`) is auto-loaded.
- Login-shell env vars are **not** inherited unless you pass `-V`.

## Submit / monitor / cancel

```bash
qsub run.sh                        # submit → prints 123456.opbs
qsub -W depend=afterok:123456.opbs step2.sh   # chain job
qsub -r y -J 1-8 array.sh          # array job (uses ${PBS_ARRAY_INDEX})

qstat                              # list your jobs
qstat -f 123456.opbs               # detailed status
qstat --limit                      # show project/queue limits

qhold 123456.opbs                  # hold
qrls  123456.opbs                  # release
qdel  123456.opbs                  # cancel
```

Output files default to `<jobname>.o<seq>` and `<jobname>.e<seq>` in the submit directory.

## Useful in-script env vars
- `PBS_O_WORKDIR` — directory where `qsub` was run (always `cd` here first).
- `PBS_JOBID`, `PBS_JOBNAME`, `PBS_ENVIRONMENT` (`BATCH`|`INTERACT`).
- `PBS_NODEFILE` — path to the host file (one line per assigned rank).
- `PBS_ARRAY_INDEX` — array subjob index.
- `MPI_PROC`, `MPI_PROC_PER_NODE` — total / per-node MPI rank counts (only after `module load` of an MPI library).

## Interactive session (Miyabi-C example)

```bash
qsub -I -l select=1 -W group_list=<PROJECT> \
     -q interact-c -l walltime=01:00:00
# → drops into a shell on a compute node; `exit` ends the job.
```

## Chain & array job dependency types (`-W depend=<type>:<jobid>`)
`after`, `afterok`, `afternotok`, `afterany`, `before`, `beforeok`, `beforenotok`, `beforeany`.

## Group / token limits (Miyabi-C, Table 5-9)
- Concurrent submissions: `running_cap × 2`.
- Concurrent running: 2 jobs at 16 sets, +1 per additional 16 sets, max 8.
- Concurrent node use: 64. Max nodes per job: 64.

## Common mistakes to avoid
- Forgetting `-W group_list=` → job is rejected.
- Setting `walltime` to the queue maximum "just in case" → defeats backfill, much longer queue wait.
- Using `mpirun` on Miyabi-C with Intel MPI → use `mpiexec.hydra`.
- Heavy work on `miyabi-c1` / `miyabi-g1` login nodes → use `prepost` queue or an interactive job.
- Multi-byte chars in env, job name, or cwd → submission fails silently with a cryptic error.
- Bash script that calls `module` without `#!/bin/bash -l` shebang.
- Submitting many separate single-node jobs to run them in parallel → only RUN=2 run at once; pack into one `select=N` job + `pbsdsh` (see "Running many independent tasks in parallel").
- Using `ssh <node>` to launch per-node tasks → no passwordless ssh between compute nodes; use `pbsdsh -n <i>`.
- Relying on `$USER`/`$HOME`/full `$PATH` inside a `pbsdsh`-launched script → its env is minimal; set them defensively.

## Quick reference: minimal submit
```bash
PROJECT=xg26g002  # set to your project code

cat > run.sh <<'EOF'
#!/bin/sh
#PBS -q debug-c
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=PROJECT_PLACEHOLDER
#PBS -j oe

cd ${PBS_O_WORKDIR}
./a.out
EOF
sed -i "s/PROJECT_PLACEHOLDER/${PROJECT}/" run.sh
qsub run.sh
```

## Source
Section 5 of `miyabi利用手引き.pdf` (Fujitsu, v1.6, 2025-09-29). The full guide also covers MPI tuning (5.6), advanced job topics like step jobs and node placement (5.7), and detailed `qstat` output formats (5.8).
