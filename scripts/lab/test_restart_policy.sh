#!/bin/bash -l
#PBS -q short-c
#PBS -l select=1
#PBS -l walltime=01:45:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -o logs/test_restart_policy.log
#PBS -N dbtune-restart-test
# Micro-benchmark: how much of DBTune's offline per-iteration dead time (~125 s median,
# REPRO §9) is caused by the shutdown policy?  DBTune's _kill_mysqld does `kill -9`, so
# every startup runs crash recovery over up to 120 s of heavy writes.
# 9 cycles of  [start mysqld with config c] -> [sysbench RW 60 s warmup + 120 s] -> [shutdown v]
#   v (shutdown policy): K9 = kill -9 (DBTune today) | G1 = SET innodb_fast_shutdown=1 + mysqladmin
#   shutdown (clean; 600 s timeout then kill -9) | G2 = innodb_fast_shutdown=2 (flush logs only)
#   c (config): 3 real top-tps S0 configs (26/5 GB, 23.7/2.6 GB, 15.1/5.5 GB pool/redo)
#   order: v = K9 G1 G2 K9 G1 G2 K9 G1 G2 ; c = 1 2 3 2 3 1 3 1 2  (each startup config follows
#   each shutdown policy once).  Buffer-pool dump/load pinned OFF for all variants.
# Overrides (qsub -v, ';'-separated): RT_POL='G1;D1;K9' RT_CFG='cA;cA;cB' RT_HIST=<history json> RT_ROWS='1;2'
#   -> configs cA,cB,... are rows of RT_HIST; policy D1 = SET innodb_doublewrite=DETECT_ONLY, then G1
#   (doublewrite page-content writes are what makes the clean flush slow on Lustre for dw=ON configs).
# Outputs: parallel/restart_test/results.csv, errlog_<cycle>_{start,stop}.txt, sb_<cycle>.log
set -uo pipefail
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "${PBS_O_WORKDIR}"
ROOT="$(pwd)"; T="${ROOT}/parallel/restart_test"; SNAP="${ROOT}/mysql_build/data_150x800k"
export PATH="${ROOT}/sysbench_install/bin:${ROOT}/mysql_build/bin:${PATH}"
export RT_HIST RT_ROWS 2>/dev/null || true
export LD_LIBRARY_PATH="${ROOT}/mysql_build/lib:${LD_LIBRARY_PATH:-}"
export SYSBENCH_BIN="${ROOT}/sysbench_install/bin/sysbench" MYSQL_SOCK="/tmp/dbtune.sock" SYSBENCH_ZIPFIAN_EXP=0.7
MYSQLD="${ROOT}/mysql_build/bin/mysqld"; MYSQL="${ROOT}/mysql_build/bin/mysql -uroot -S /tmp/dbtune.sock"
MYSQLADMIN="${ROOT}/mysql_build/bin/mysqladmin -uroot -S /tmp/dbtune.sock"
ERR="${T}/mysql.err"; RES="${T}/results.csv"
mkdir -p "${T}"; rm -f "${T}"/errlog_* "${T}"/sb_*.log "${RES}" "${ERR}"
echo "[RT] node=$(hostname) start=$(date -Iseconds)"
cleanup() { pkill -9 -x mysqld 2>/dev/null; rm -rf "${T}/data"; echo "[RT] end=$(date -Iseconds)"; }
trap cleanup EXIT

# --- 3 full cnfs from real configs (DBTune format: base lines + every knob of the 117-knob file)
python3 - "${ROOT}" "${T}" <<'PY'
import json, os, sys
root, T = sys.argv[1:]
base = ['[mysqld]', 'basedir = %s/mysql_build' % root, 'datadir = %s/data' % T, 'socket = /tmp/dbtune.sock',
        'pid-file = /tmp/dbtune.pid', 'log-error = %s/mysql.err' % T,
        'innodb_buffer_pool_dump_at_shutdown = OFF', 'innodb_buffer_pool_load_at_startup = OFF']
K = json.load(open(os.path.join(root, 'scripts/experiment/gen_knobs/mysql_perf_8.0.json')))
lhs = json.load(open(os.path.join(root, 'scripts/eval/replay_S0_lhs.json')))
rnd = json.load(open(os.path.join(root, 'scripts/eval/replay_S0_random.json')))
cfgs = [('c1', lhs['configs'][0], lhs['source_tps'][0]), ('c2', rnd['configs'][0], rnd['source_tps'][0]),
        ('c3', rnd['configs'][2], rnd['source_tps'][2])]
if os.environ.get('RT_HIST'):
    rows = json.load(open(os.path.join(root, os.environ['RT_HIST'])))['data']
    cfgs = [('c' + chr(ord('A') + i), rows[int(r)]['configuration'], rows[int(r)]['external_metrics']['tps'])
            for i, r in enumerate(os.environ['RT_ROWS'].split(';'))]
for name, c, tps in cfgs:
    lines = base + ['%s = %s' % (k, ("'%s'" % v if isinstance(v, str) and ' ' in v else v)) for k, v in c.items() if k in K]
    open(os.path.join(T, name + '.cnf'), 'w').write('\n'.join(lines) + '\n')
    print('[RT] %s: source tps %.0f, pool %.1f GB, redo %.1f GB, flush_trx %s, dirty_pct %s, io_cap %s' % (
        name, tps, c['innodb_buffer_pool_size'] / 2**30, c['innodb_log_file_size'] * c.get('innodb_log_files_in_group', 1) / 2**30,
        c['innodb_flush_log_at_trx_commit'], c.get('innodb_max_dirty_pages_pct'), c.get('innodb_io_capacity')), 'dw=%s' % c.get('innodb_doublewrite'))
PY

echo "[RT] copying snapshot ..."; t0=$(date +%s); rm -rf "${T}/data"; cp -a "${SNAP}" "${T}/data" || { echo "[RT] copy failed"; exit 1; }
echo "[RT] copied in $(( $(date +%s) - t0 ))s"

errmark() { wc -l < "${ERR}" 2>/dev/null || echo 0; }
errsince() { tail -n +"$(( $1 + 1 ))" "${ERR}" 2>/dev/null; }
start_mysqld() {  # $1 = cnf ; prints seconds to first successful ping
    local m; m=$(errmark); local t0; t0=$(date +%s.%N)
    "${MYSQLD}" --defaults-file="$1" >/dev/null 2>&1 &
    local n=0
    while ! ${MYSQLADMIN} ping >/dev/null 2>&1; do sleep 1; n=$((n+1)); if (( n > 900 )); then echo "TIMEOUT"; return 1; fi; done
    errsince "$m" > "${T}/errlog_${CYC}_start.txt"
    python3 -c "import time,sys; print('%.1f' % (time.time()-float(sys.argv[1])))" "$t0"
}
innodb_state() {  # dirty pages, checkpoint age (bytes of redo a crash recovery must apply), pool size
    ${MYSQL} -N -e "SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_pages_dirty'" 2>/dev/null | awk '{printf "%s,", $2}'
    ${MYSQL} -N -e "SHOW ENGINE INNODB STATUS\G" 2>/dev/null | awk '/^Log sequence number/{l=$4} /^Last checkpoint at/{c=$4} END{printf "%d,", l-c}'
    ${MYSQL} -N -e "SELECT @@innodb_buffer_pool_size DIV 1048576" 2>/dev/null | tr -d '\n'
}
stop_mysqld() {  # $1 = policy ; prints seconds until no mysqld process, and whether the kill -9 fallback fired
    local m; m=$(errmark); local t0; t0=$(date +%s.%N); local fallback=0
    case "$1" in
        K9) pkill -9 -x mysqld ;;
        D1) ${MYSQL} -e "SET GLOBAL innodb_doublewrite='DETECT_ONLY'" 2>&1 | sed 's/^/[RT]   dw->DETECT_ONLY: /' >&2
            ${MYSQL} -N -e "SELECT @@innodb_doublewrite" 2>&1 | sed 's/^/[RT]   dw now: /' >&2
            ${MYSQL} -e "SET GLOBAL innodb_fast_shutdown=1" 2>/dev/null
            timeout 600 ${MYSQLADMIN} shutdown 2>/dev/null || { fallback=1; pkill -9 -x mysqld; } ;;
        G1|G2) ${MYSQL} -e "SET GLOBAL innodb_fast_shutdown=${1#G}" 2>/dev/null
               timeout 600 ${MYSQLADMIN} shutdown 2>/dev/null || { fallback=1; pkill -9 -x mysqld; } ;;
    esac
    while pgrep -x mysqld >/dev/null; do sleep 0.5; done
    rm -f /tmp/dbtune.sock /tmp/dbtune.sock.lock /tmp/dbtune.pid
    errsince "$m" > "${T}/errlog_${CYC}_stop.txt"
    python3 -c "import time,sys; print('%.1f,%s' % (time.time()-float(sys.argv[1]), sys.argv[2]))" "$t0" "$fallback"
}

echo "cycle,start_cnf,prev_policy,startup_s,tps,dirty_pages,checkpoint_age_bytes,pool_mb,policy,shutdown_s,fallback_kill9" > "${RES}"
IFS=";" read -ra POL <<< "${RT_POL:-K9;G1;G2;K9;G1;G2;K9;G1;G2}"; IFS=";" read -ra CFG <<< "${RT_CFG:-c1;c2;c3;c2;c3;c1;c3;c1;c2}"; prev="snapshot"
echo "[RT] plan: policies=${POL[*]} configs=${CFG[*]}"
for i in "${!POL[@]}"; do
    CYC=$((i+1)); v="${POL[$i]}"; c="${CFG[$i]}"
    echo "[RT] cycle ${CYC}: start ${c} (after ${prev}) $(date -Iseconds)"
    up=$(start_mysqld "${T}/${c}.cnf") || { echo "[RT] startup failed/timeout"; tail -5 "${ERR}"; exit 1; }
    echo "[RT]   up in ${up}s"
    SYSBENCH_ZIPFIAN_EXP=0.7 bash "${ROOT}/autotune/cli/run_sysbench.sh" readwrite 127.0.0.1 3306 root "" 150 800000 60 128 120 "${T}/sb_${CYC}.log" sbtest
    tps=$(grep -oE 'transactions:\s+[0-9]+\s+\(([0-9.]+) per sec' "${T}/sb_${CYC}.log" | grep -oE '\([0-9.]+' | tr -d '(')
    st=$(innodb_state)
    echo "[RT]   tps=${tps} dirty,ckpt_age,pool_mb=${st}; shutdown ${v} $(date -Iseconds)"
    down=$(stop_mysqld "${v}")
    echo "[RT]   down in ${down}"
    echo "${CYC},${c},${prev},${up},${tps},${st},${v},${down}" >> "${RES}"
    prev="${v}"
done
echo "[RT] done $(date -Iseconds)"; cat "${RES}"
