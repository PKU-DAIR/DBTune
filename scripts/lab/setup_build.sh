#!/bin/bash -l
#PBS -q short-c
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -N dbtune-build
#PBS -o logs/setup_build.log

set -euo pipefail

# When run via qsub, PBS_O_WORKDIR points at the submit dir. When run directly
# from the login node for testing, fall back to the script's own grandparent.
if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
    cd "${PBS_O_WORKDIR}"
fi
ROOT="$(pwd)"
echo "[INFO] ROOT=${ROOT}"
echo "[INFO] node=$(hostname)  date=$(date -Iseconds)"

# Use a writable, large location for any wget-style caches
export TMPDIR="${TMPDIR:-/tmp}"

###############################################################################
# 1) Build MySQL 8.0.44 from source.
#    mysql_download.sh handles cmake + ninja (downloads boost 1.77 first run).
#    Then cmake --install lays out bin/, lib/, include/ FHS-style so
#    sysbench's configure can find mysql_config.
###############################################################################
# Be a polite neighbor on the login node: cap parallel jobs.
JOBS="${BUILD_JOBS:-32}"

if [[ -x "${ROOT}/mysql_build/bin/mysql_config" && -x "${ROOT}/mysql_build/bin/mysqld" ]]; then
    echo "[SKIP] mysql_build already fully built and installed"
else
    if [[ ! -f "${ROOT}/mysql_build/build.ninja" ]]; then
        echo "[STEP] Configuring + building MySQL via mysql_download.sh (JOBS=${JOBS}) ..."
        # mysql_download.sh ends with `ninja $1 -j $(nproc)`; pass an empty
        # target so ninja builds 'all', and override -j via NINJA_STATUS-free run.
        # Easier: just run cmake + ninja ourselves with the same flags.
        mkdir -p "${ROOT}/mysql_build/data" "${ROOT}/mysql_build/cnf"
        cat > "${ROOT}/mysql_build/cnf/my.cnf" <<EOF
[mysqld]
basedir = ${ROOT}/mysql_build
datadir = ${ROOT}/mysql_build/data
socket = ${ROOT}/mysql_build/mysql.sock
pid-file = ${ROOT}/mysql_build/mysql.pid
log-error = ${ROOT}/mysql_build/mysql.err
EOF
        if [[ ! -d "${ROOT}/mysql_build/boost" ]]; then
            mkdir -p "${ROOT}/mysql_build/boost"
            pushd "${ROOT}/mysql_build" >/dev/null
            wget -q https://archives.boost.io/release/1.77.0/source/boost_1_77_0.tar.bz2
            tar --bzip2 -xf boost_1_77_0.tar.bz2 -C boost --strip-components=1
            rm boost_1_77_0.tar.bz2
            popd >/dev/null
        fi
        pushd "${ROOT}/mysql_build" >/dev/null
        cmake "${ROOT}/third_party/mysql-server" \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
            -DWITH_BUILD_ID=0 \
            -DWITH_ASAN=0 \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="${ROOT}/mysql_build" \
            -DWITH_BOOST=./boost \
            -DWITH_TIRPC=bundled \
            -DWITHOUT_GROUP_REPLICATION=1 \
            -DWITHOUT_EXAMPLE_STORAGE_ENGINE=1 \
            -DWITHOUT_FEDERATED_STORAGE_ENGINE=1 \
            -DWITHOUT_ARCHIVE_STORAGE_ENGINE=1 \
            -DWITHOUT_BLACKHOLE_STORAGE_ENGINE=1 \
            -DWITHOUT_NDB_STORAGE_ENGINE=1 \
            -DWITHOUT_NDBCLUSTER_STORAGE_ENGINE=1 \
            -DWITHOUT_PARTITION_STORAGE_ENGINE=1 \
            -G Ninja
        ninja -j"${JOBS}"
        popd >/dev/null
    else
        echo "[INFO] cmake config exists; resuming ninja build with JOBS=${JOBS}"
        (cd "${ROOT}/mysql_build" && ninja -j"${JOBS}")
    fi
    echo "[STEP] Installing MySQL in place (prefix=${ROOT}/mysql_build) ..."
    cmake --install "${ROOT}/mysql_build" --prefix "${ROOT}/mysql_build"
fi

if [[ ! -x "${ROOT}/mysql_build/bin/mysqld" ]]; then
    echo "[ERROR] MySQL build did not produce mysql_build/bin/mysqld" >&2
    exit 1
fi
echo "[OK] mysqld: $(${ROOT}/mysql_build/bin/mysqld --version)"
echo "[OK] mysql_config: $(${ROOT}/mysql_build/bin/mysql_config --version 2>/dev/null || echo missing)"

###############################################################################
# 2) Build sysbench, linked against the just-built MySQL client lib
###############################################################################
SYSBENCH_PREFIX="${ROOT}/sysbench_install"
if [[ -x "${SYSBENCH_PREFIX}/bin/sysbench" ]]; then
    echo "[SKIP] sysbench already installed at ${SYSBENCH_PREFIX}"
else
    echo "[STEP] Building sysbench ..."
    # sysbench locates MySQL via --with-mysql=<root> which must contain bin/mysql_config
    pushd "${ROOT}/third_party/sysbench" >/dev/null
    ./autogen.sh
    ./configure --prefix="${SYSBENCH_PREFIX}" \
                --with-mysql="${ROOT}/mysql_build" \
                --without-pgsql
    make -j"$(nproc)"
    make install
    popd >/dev/null
fi

if [[ ! -x "${SYSBENCH_PREFIX}/bin/sysbench" ]]; then
    echo "[ERROR] sysbench build did not produce ${SYSBENCH_PREFIX}/bin/sysbench" >&2
    exit 1
fi
echo "[OK] sysbench: $(${SYSBENCH_PREFIX}/bin/sysbench --version)"

###############################################################################
# 3) Install Apache Ant locally and build oltpbench
###############################################################################
ANT_VERSION="1.10.15"
ANT_HOME="${ROOT}/tools/apache-ant-${ANT_VERSION}"
if [[ ! -x "${ANT_HOME}/bin/ant" ]]; then
    echo "[STEP] Installing Apache Ant ${ANT_VERSION} ..."
    mkdir -p "${ROOT}/tools"
    pushd "${ROOT}/tools" >/dev/null
    wget -q "https://archive.apache.org/dist/ant/binaries/apache-ant-${ANT_VERSION}-bin.tar.gz"
    tar xf "apache-ant-${ANT_VERSION}-bin.tar.gz"
    rm "apache-ant-${ANT_VERSION}-bin.tar.gz"
    popd >/dev/null
fi
export PATH="${ANT_HOME}/bin:${PATH}"
echo "[OK] ant: $(ant -version)"

if [[ -d "${ROOT}/third_party/oltpbench/build" && -f "${ROOT}/third_party/oltpbench/oltpbenchmark" ]]; then
    echo "[SKIP] oltpbench already built"
else
    echo "[STEP] Building oltpbench ..."
    pushd "${ROOT}/third_party/oltpbench" >/dev/null
    # Resolve Ivy deps then build. 'bootstrap' downloads ivy itself if needed.
    ant bootstrap || true
    ant resolve
    ant
    popd >/dev/null
fi
echo "[OK] oltpbench built at ${ROOT}/third_party/oltpbench"

###############################################################################
# 4) Python venv + DBTune requirements
###############################################################################
VENV="${ROOT}/venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[STEP] Creating Python venv ..."
    python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade pip wheel setuptools
# Best-effort install of the project's requirements. Some pinned/legacy
# packages may fail (e.g. mysql-connector-python-rf is no longer on PyPI);
# we install canonical replacements below regardless.
pip install --no-cache-dir -r "${ROOT}/requirements.txt" || \
    echo "[WARN] some requirements failed to install; see log above"

# Ensure runtime-critical packages are present.
pip install --no-cache-dir \
    mysql-connector-python \
    "paramiko<3" \
    requests

# Install the project itself in editable mode so 'autotune' is importable.
# Use --no-deps because setup.py pins legacy mysql-connector-python-rf which
# may not resolve on PyPI; runtime deps are already installed above.
pip install --no-cache-dir --no-deps -e "${ROOT}"

# Smoke-import test
python - <<'PY'
import importlib
for mod in ("autotune", "openbox", "torch", "lightgbm", "shap",
            "mysql.connector", "ConfigSpace", "smac", "xgboost"):
    importlib.import_module(mod)
print("[OK] All required Python modules import cleanly")
PY

echo "[DONE] Build complete at $(date -Iseconds)"
