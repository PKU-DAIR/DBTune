#!/bin/bash -l
#PBS -q short-c
#PBS -l select=1
#PBS -l walltime=03:00:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -N dbtune-smoke
#PBS -o logs/run_tuning.log

# Sequentially runs the 6 smoke-test configs:
#   DML vs OtterTune for sysbench read / readwrite (rw50) / write.
# Each config takes ~minutes (max_runs=1..2 + initial_runs=1..10 with
# workload_time=15s; main cost is mysqld restarts between trials).

set -uo pipefail
if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
    cd "${PBS_O_WORKDIR}"
fi
ROOT="$(pwd)"
echo "[INFO] ROOT=${ROOT}  node=$(hostname)  date=$(date -Iseconds)"

# Surface fatal setup errors instead of silently continuing into tuning.
fatal() {
    echo "[FATAL] $*" >&2
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    exit 1
}

# Put our tools on PATH so DBTune can find sysbench.
export PATH="${ROOT}/sysbench_install/bin:${ROOT}/mysql_build/bin:${PATH}"
# sysbench is linked against the freshly built libmysqlclient.so.21 — make sure
# the loader can find it.
export LD_LIBRARY_PATH="${ROOT}/mysql_build/lib:${LD_LIBRARY_PATH:-}"
# DBTune uses SYSBENCH_BIN and MYSQL_SOCK in its CLI scripts
export SYSBENCH_BIN="${ROOT}/sysbench_install/bin/sysbench"
export MYSQL_SOCK="${ROOT}/mysql_build/mysql.sock"
# Match the source-data access distribution: zipfian with skew exponent 0.2
# (source collected via benchmark_native.py with --rand-type=zipfian).
export SYSBENCH_ZIPFIAN_EXP=0.2
# DML downstream optimizer: 'gp' (argmax-mean, default) or 'ottertune'
# (GP on target+source + EI). Passed through from qsub -v.
export DML_DOWNSTREAM="${DML_DOWNSTREAM:-gp}"
# Source-only surrogate + argmax-of-predicted-mean recommendation (makes source
# selection drive the result). Passed through from qsub -v.
export WM_SOURCE_ONLY="${WM_SOURCE_ONLY:-}"
export REC_ARGMAX_MEAN="${REC_ARGMAX_MEAN:-}"
export REC_N_RANDOM="${REC_N_RANDOM:-2000}"
# Optional fixed seed override (DBTUNE_SEED) for repeat/seed-sweep runs.
export DBTUNE_SEED="${DBTUNE_SEED:-}"

# 1) Activate venv
source "${ROOT}/venv/bin/activate"

# 2) Reset MySQL to the DEFAULT config (writes my.cnf.default from the ini's
#    knob_config_file, restarts mysqld, purges binlogs). All 6 configs share the
#    same knob_config_file / mysql_build, so any one ini establishes the baseline.
bash "${ROOT}/scripts/lab/reset_database.sh" config_sysbench_dml_read.ini \
    || fatal "reset_database.sh failed"

# 3) Ensure sbtest schema/data exists (data only; mysqld already running).
#    Must match the experiments' sysbench_tables / sysbench_table_size so the
#    loaded dataset and the benchmark --table-size agree.
export SBTEST_TABLES=64
export SBTEST_TABLE_SIZE=1000000
bash "${ROOT}/scripts/lab/init_sbtest.sh" || fatal "init_sbtest.sh failed"

# 4) Start Prometheus stack (mysqld_exporter / node_exporter / prometheus).
#    DBTune manages mysqld lifecycle per-trial from here on.
bash "${ROOT}/scripts/lab/start_stack.sh" || fatal "start_stack.sh failed"

# Ensure cleanup on exit (success, failure, or signal)
cleanup() {
    bash "${ROOT}/scripts/lab/stop_stack.sh" || true
}
trap cleanup EXIT INT TERM

# 3) Run each config sequentially. Don't abort the whole batch if one fails;
#    log the result and continue so we get partial data even if a config
#    explodes.
if [[ -n "${CONFIGS_OVERRIDE:-}" ]]; then
    # ':'-separated (PBS -v uses comma as a variable separator).
    IFS=':' read -ra CONFIGS <<< "${CONFIGS_OVERRIDE}"
else
    CONFIGS=(
        config_sysbench_dml_read.ini
        config_sysbench_dml_rw50.ini
        config_sysbench_dml_write.ini
        config_sysbench_ot_read.ini
        config_sysbench_ot_rw50.ini
        config_sysbench_ot_write.ini
    )
fi

cd "${ROOT}/scripts"
# mysqldb.py writes generated cnf to hardcoded relative path ./tmp/mysqld.cnf
mkdir -p "${ROOT}/scripts/tmp" "${ROOT}/logs/runs"
overall_rc=0
for cfg in "${CONFIGS[@]}"; do
    name="${cfg%.ini}"
    # Optional source-choice sensitivity test: when FORCE_SOURCES is set, pin the
    # DML source context per workload (DML honors FORCE_SOURCE_CONTEXT; OT ignores it).
    unset FORCE_SOURCE_CONTEXT
    if [[ -n "${FORCE_SOURCES:-}" ]]; then
        # Pin the transfer source to OtterTune's exact auto-picked source (read
        # from each OT run's history matched_context) so dmlmap uses the SAME
        # source as OT and the run isolates the downstream model. dmlmap needs
        # the FULL task_id, not a compact label: the source repo holds both the
        # 64-100000 and 64-1000000 variants of the same compact label, and OT
        # picked the 100000 variant for read — only the full id reproduces it.
        # (The dml_* compact entries are kept for the older DML-optimizer path;
        # this experiment runs the dmlmap_* configs.) Order matters: the more
        # specific *dmlmap_* patterns must precede the generic *_<wl> ones.
        # Force ONLY dml (dmlmap) to the chosen 88c190g 1M sources; OtterTune is
        # left to its own natural (binning) pick (ot_* configs match nothing here,
        # so FORCE_SOURCE_CONTEXT stays unset for them).
        case "${name}" in
            *dmlmap_read)  export FORCE_SOURCE_CONTEXT="history_88c190g_64-1000000-4-oltp_read_write_95-0.2" ;;
            *dmlmap_rw50)  export FORCE_SOURCE_CONTEXT="history_88c190g_64-1000000-4-oltp_read_write_80-0.6" ;;
            *dmlmap_write) export FORCE_SOURCE_CONTEXT="history_88c190g_64-1000000-4-oltp_read_write_5-1.0" ;;
        esac
    fi
    echo "============================================================"
    echo "[RUN] ${cfg}   start=$(date -Iseconds)  FORCE_SOURCE_CONTEXT=${FORCE_SOURCE_CONTEXT:-<auto>}"
    echo "============================================================"
    if python optimize.py --config="${cfg}" \
            >"${ROOT}/logs/runs/${name}.log" 2>&1; then
        echo "[OK]  ${cfg}   end=$(date -Iseconds)"
    else
        rc=$?
        echo "[FAIL ${rc}] ${cfg} — see logs/runs/${name}.log"
        overall_rc=$((overall_rc | rc))
    fi
done

echo "[DONE] all configs attempted. overall_rc=${overall_rc}"
exit "${overall_rc}"
