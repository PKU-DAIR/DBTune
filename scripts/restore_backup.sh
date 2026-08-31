#!/bin/bash
# Restore DBTune environment from backup
#
# Prerequisites:
#   1. Clone the repo and checkout the backup branch:
#      git clone git@github.com:Tatzhiro/DBTune.git
#      cd DBTune
#      git checkout backup/dml-ottertune-wip
#
#   2. Place dbtune_data_backup.tar.gz in the repo root:
#      scp user@server:~/dbtune_data_backup.tar.gz .
#
#   3. Run this script:
#      bash scripts/restore_backup.sh
#
# What this script does:
#   - Extracts data backup (DBTune_history + DBMSTransferLearning)
#   - Creates Python venv and installs dependencies
#   - Downloads and builds MySQL 8.0 from source (uses mysql_download.sh + mysql_init.sh)
#   - Prints next steps for starting MySQL and running experiments

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Step 1: Extract data backup ==="
if [ -f dbtune_data_backup.tar.gz ]; then
    tar xzf dbtune_data_backup.tar.gz
    echo "Extracted scripts/DBTune_history/ and DBMSTransferLearning/"
else
    echo "WARNING: dbtune_data_backup.tar.gz not found in repo root."
    echo "Place it here and re-run, or skip if you don't need historical data."
fi

echo ""
echo "=== Step 2: Create Python venv ==="
if [ ! -d venv ]; then
    python3.10 -m venv venv
    echo "Created venv"
else
    echo "venv already exists, skipping"
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed"

echo ""
echo "=== Step 3: MySQL setup ==="
if [ ! -d mysql_build ]; then
    echo "Running mysql_download.sh to download and build MySQL..."
    bash mysql_download.sh
    echo "Running mysql_init.sh to initialize MySQL data directory..."
    bash mysql_init.sh
else
    echo "mysql_build/ already exists, skipping"
fi

echo ""
echo "=== Step 4: Submodules ==="
git submodule update --init --recursive

echo ""
echo "============================================"
echo "Restore complete. Next steps:"
echo "============================================"
echo ""
echo "1. Start MySQL:"
echo "   ../mysql_build/bin/mysqld --defaults-file=../mysql_build/cnf/my.cnf &"
echo ""
echo "2. Set environment variables:"
echo "   export SYSBENCH_BIN=/usr/local/bin/sysbench"
echo "   export MYSQL_SOCK=../mysql_build/mysql.sock"
echo "   source venv/bin/activate"
echo ""
echo "3. (Optional) Start Prometheus stack:"
echo "   cd scripts/prometheus && sudo docker compose up -d && cd ../.."
echo ""
echo "4. Load benchmark data (sysbench example):"
echo "   \$SYSBENCH_BIN --db-driver=mysql --mysql-host=127.0.0.1 --mysql-user=root \\"
echo "     --tables=150 --table-size=800000 --threads=32 oltp_common prepare"
echo ""
echo "5. Run an experiment:"
echo "   cd scripts && python optimize.py --config=config_sysbench_ot_rw50.ini"
echo ""
