#!/bin/bash -l
# Reusable: reset MySQL to the DEFAULT configuration defined by an ini file and
# bring it up on a lean data dir.
#
#   usage: reset_database.sh <config.ini>
#
# Steps:
#   1. Parse the [database] section of <config.ini> for knob_config_file, cnf,
#      mysqld, sock (paths are resolved relative to scripts/, like optimize.py).
#   2. Write <cnf>.default = the cnf's non-knob base lines + every knob's
#      "default" value from knob_config_file (e.g. DML_12.json). Apply it (copy
#      to <cnf>). The .default file is the auditable canonical baseline.
#   3. Stop any running mysqld, initialize the data dir if empty, start mysqld
#      with the default cnf, and wait until it accepts connections.
#   4. Purge accumulated binary logs so the data dir stays lean across runs.
#
# This script OWNS mysqld start/stop. init_sbtest.sh assumes a running server.

set -uo pipefail
INI="${1:?usage: reset_database.sh <config.ini>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="${ROOT}/scripts"
export LD_LIBRARY_PATH="${ROOT}/mysql_build/lib:${LD_LIBRARY_PATH:-}"

# Allow the ini to be given relative to scripts/ or as an absolute/cwd path.
[[ -f "${INI}" ]] || INI="${SCRIPTS}/${INI}"
[[ -f "${INI}" ]] || { echo "[ERROR] ini not found: ${1}" >&2; exit 1; }

# Parse ini + write <cnf>.default, then emit the paths bash needs.
PATHS="$(python3 - "${INI}" "${SCRIPTS}" <<'PY'
import configparser, json, os, sys, shutil
ini, scripts = sys.argv[1], sys.argv[2]
c = configparser.ConfigParser()
c.read(ini)
db = c['database']
def absify(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(scripts, p))
knob_json = absify(db['knob_config_file'])
cnf       = absify(db['cnf'])
mysqld    = absify(db['mysqld'])
sock      = absify(db['sock'])

defaults = json.load(open(knob_json))
knob_names = set(defaults.keys())

# Base = the clean cnf if present (canonical bare base, incl. any non-knob
# settings like buffer-pool dump/restore = OFF), else the current cnf with
# tuning-knob lines stripped. Keeps [mysqld] header + path/server settings.
base_src = cnf + '.clean' if os.path.exists(cnf + '.clean') else cnf
base = []
with open(base_src) as f:
    for line in f:
        key = line.split('=', 1)[0].strip()
        if key in knob_names:
            continue
        base.append(line.rstrip('\n'))
while base and base[-1].strip() == '':
    base.pop()

default_cnf = cnf + '.default'
with open(default_cnf, 'w') as f:
    f.write('\n'.join(base) + '\n')
    for k, v in defaults.items():
        dv = v.get('default') if isinstance(v, dict) else v
        f.write(f'{k}\t\t= {dv}\n')
shutil.copyfile(default_cnf, cnf)

datadir = absify(c['database'].get('datadir', os.path.join(os.path.dirname(os.path.dirname(cnf)), 'data')))
print(mysqld); print(sock); print(cnf); print(datadir); print(default_cnf)
PY
)" || { echo "[ERROR] failed to parse ini / write default cnf" >&2; exit 1; }

MYSQLD="$(sed -n 1p <<<"${PATHS}")"
SOCK="$(sed -n 2p <<<"${PATHS}")"
CNF="$(sed -n 3p <<<"${PATHS}")"
DATADIR="$(sed -n 4p <<<"${PATHS}")"
DEFAULT_CNF="$(sed -n 5p <<<"${PATHS}")"
MYSQLADMIN="$(dirname "${MYSQLD}")/mysqladmin"
PIDFILE="$(dirname "${SOCK}")/$(basename "${SOCK%.sock}").pid"

echo "[INFO] wrote default cnf: ${DEFAULT_CNF}"
echo "[INFO] mysqld=${MYSQLD}"
echo "[INFO] sock=${SOCK}  datadir=${DATADIR}"

# 1) Stop any running server (graceful, then clean stale runtime files).
"${MYSQLADMIN}" -uroot -S "${SOCK}" shutdown 2>/dev/null || true
pkill -x mysqld 2>/dev/null || true
sleep 1
rm -f "${SOCK}" "${SOCK}.lock" "${DATADIR%/}/../mysql.pid" "$(dirname "${SOCK}")/mysql.pid"

# 2) Initialize the data dir if it has no system tables.
if [[ ! -d "${DATADIR}/mysql" ]]; then
    echo "[STEP] mysqld --initialize-insecure ..."
    "${MYSQLD}" --defaults-file="${CNF}" --initialize-insecure
fi

# 3) Start mysqld with the default cnf and wait (crash recovery / redo resize
#    after a force-killed prior run can take a while).
echo "[STEP] starting mysqld on default config ..."
"${MYSQLD}" --defaults-file="${CNF}" >"${ROOT}/logs/mysqld.default.log" 2>&1 &
MPID=$!
MYSQL="$(dirname "${MYSQLD}")/mysql"
# Generous timeout: a force-killed prior run can leave a large dirty redo log
# whose shrink + crash recovery on first startup takes many minutes.
START_TIMEOUT="${RESET_START_TIMEOUT:-1200}"
for _ in $(seq 1 "${START_TIMEOUT}"); do
    if [[ -S "${SOCK}" ]] && "${MYSQLADMIN}" -uroot -S "${SOCK}" ping >/dev/null 2>&1; then
        echo "[OK] mysqld up on default config (pid=${MPID})"
        # Purge accumulated binary logs so the data dir doesn't grow unbounded
        # across runs (each tuning trial writes binlog; sync_binlog is a knob so
        # binlog stays ON). RESET MASTER drops all binlogs + resets the index;
        # fall back to the 8.4+ spelling if RESET MASTER is unavailable.
        if "${MYSQL}" -uroot -S "${SOCK}" -e "RESET MASTER;" 2>/dev/null; then
            echo "[OK] purged binary logs (RESET MASTER)"
        elif "${MYSQL}" -uroot -S "${SOCK}" -e "RESET BINARY LOGS AND GTIDS;" 2>/dev/null; then
            echo "[OK] purged binary logs (RESET BINARY LOGS AND GTIDS)"
        else
            echo "[WARN] binlog purge failed (continuing)"
        fi
        exit 0
    fi
    kill -0 "${MPID}" 2>/dev/null || break
    sleep 1
done
echo "[ERROR] mysqld did not come up on default config. Last log:" >&2
tail -30 "${ROOT}/logs/mysqld.default.log" >&2
exit 1
