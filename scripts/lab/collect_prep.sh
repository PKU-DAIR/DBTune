#!/bin/bash -l
#PBS -q regular-c
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -o logs/collect_prep.log
#PBS -N dbtune-collect-prep
# Build the 150 tables x 800k rows sbtest snapshot (mysql_build/data_150x800k)
# that every collection cell copies at wave start. Run ONCE before the first
# collect_launch.sh wave. Strategy: copy the clean 64x1M template datadir, start
# mysqld on it, let init_sbtest.sh detect the size mismatch and drop & reload at
# 150x800k, purge binlogs, shut down cleanly.
set -uo pipefail
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "${PBS_O_WORKDIR}"
ROOT="$(pwd)"
# override via qsub -v SBTEST_TABLE_SIZE=80000 for the S3 (scale-mismatch) snapshot
SBTEST_TABLE_SIZE="${SBTEST_TABLE_SIZE:-800000}"
SIZE_LABEL="$(( SBTEST_TABLE_SIZE / 1000 ))k"
SNAP="${ROOT}/mysql_build/data_150x${SIZE_LABEL}"
SOCK="/tmp/dbtune.sock"
CNF="/tmp/dbtune_prep.cnf"
MYSQL="${ROOT}/mysql_build/bin/mysql"
MYSQLADMIN="${ROOT}/mysql_build/bin/mysqladmin"

export USER="${USER:-$(id -un)}"
export HOME="${HOME:-/home/${USER}}"
export PATH="${ROOT}/sysbench_install/bin:${ROOT}/mysql_build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="${ROOT}/mysql_build/lib:${LD_LIBRARY_PATH:-}"

echo "[PREP] node=$(hostname) start=$(date -Iseconds)"
pkill -x mysqld 2>/dev/null && sleep 5 || true

echo "[PREP] copying template datadir -> ${SNAP} ..."; t0=$(date +%s)
rm -rf "${SNAP}"
cp -a "${ROOT}/mysql_build/data" "${SNAP}"
echo "[PREP] template copied in $(( $(date +%s) - t0 ))s"

# cnf = clean base with datadir/socket/pid/log retargeted (same sed as gen_task)
python3 - "${ROOT}" "${SNAP}" "${SOCK}" "${CNF}" <<'PY'
import sys
root, snap, sock, cnf = sys.argv[1:]
out = []
for line in open(root + '/mysql_build/cnf/my.cnf.clean'):
    k = line.split('=', 1)[0].strip()
    if k == 'datadir':     out.append('datadir = ' + snap)
    elif k == 'socket':    out.append('socket = ' + sock)
    elif k == 'pid-file':  out.append('pid-file = /tmp/dbtune_prep.pid')
    elif k == 'log-error': out.append('log-error = /tmp/dbtune_prep.err')
    else:                  out.append(line.rstrip('\n'))
open(cnf, 'w').write('\n'.join(out) + '\n')
PY

echo "[PREP] starting mysqld ..."
"${ROOT}/mysql_build/bin/mysqld" --defaults-file="${CNF}" --user="${USER}" &
for _ in $(seq 1 120); do
    "${MYSQLADMIN}" -uroot -S "${SOCK}" ping >/dev/null 2>&1 && break
    sleep 5
done
"${MYSQLADMIN}" -uroot -S "${SOCK}" ping >/dev/null 2>&1 \
    || { echo "[PREP][FATAL] mysqld did not come up (see /tmp/dbtune_prep.err)"; exit 1; }

echo "[PREP] sysbench prepare 150 x ${SBTEST_TABLE_SIZE} (drop & reload via init_sbtest.sh) ..."
SOCK="${SOCK}" SBTEST_TABLES=150 SBTEST_TABLE_SIZE="${SBTEST_TABLE_SIZE}" \
    bash "${ROOT}/scripts/lab/init_sbtest.sh"

echo "[PREP] purging binlogs + clean shutdown ..."
"${MYSQL}" -uroot -S "${SOCK}" -e "RESET MASTER;" || true
"${MYSQLADMIN}" -uroot -S "${SOCK}" shutdown
sleep 5
rm -f "${SNAP}/dbtune.pid" 2>/dev/null || true

du -sh "${SNAP}"
echo "[PREP] snapshot ready: ${SNAP}  end=$(date -Iseconds)"
