---
name: run-experiment
description: Run a DBTune tuning experiment (DML, OtterTune, SMAC, etc.)
user_invocable: true
---

# Run DBTune Experiment

## Prerequisites
- MySQL must be running: `../mysql_build/bin/mysqld --defaults-file=../mysql_build/cnf/my.cnf &`
- Prometheus stack must be running (if using Prometheus metrics): `cd scripts/prometheus && sudo docker compose up -d`
- The target database must be loaded with the benchmark data (OLTPBench `--create=true --load=true` or sysbench `--prepare`)
- venv must be activated: `source venv/bin/activate`

## Required Environment Variables

Set these **before** running experiments. The benchmark scripts reference them and will fail with exit code 127 if unset.

```bash
# Sysbench workloads
export SYSBENCH_BIN=/usr/local/bin/sysbench
export MYSQL_SOCK=../mysql_build/mysql.sock

# OLTPBench workloads (adjust paths as needed)
export OLTPBENCH_HOME=/work/dpl-sfc/users/tatsu/tmp/DBTune/third_party/oltpbench
```

The benchmark shell scripts (`autotune/cli/run_sysbench.sh`, `autotune/cli/run_oltpbench.sh`) use `$SYSBENCH_BIN`, `$MYSQL_SOCK`, etc. directly — if they are not exported, the benchmark command is empty and the shell returns error 127 (command not found).

## Experiment Workflow

### 1. Prepare the config file
Copy `scripts/config_dml_test.ini` or `scripts/config_performance.ini` as a base. Key fields to set:

**[database] section:**
- `knob_config_file`: Path to knob JSON (e.g., `./experiment/gen_knobs/DML_12.json`)
- `knob_num`: Number of knobs (must match the JSON file)
- `dbname`: Database name matching the OLTPBench DB (e.g., `tpcc`, `ycsb`, `twitter`)
- `workload`: Workload type (e.g., `oltpbench_tpcc`, `oltpbench_ycsb`, `oltpbench_twitter`)
- `oltpbench_config_xml`: Absolute path to OLTPBench XML config
- `workload_time`: Benchmark duration in seconds per iteration (15s for quick tests)

**[tune] section:**
- `task_id`: Unique ID for this experiment (history saved as `DBTune_history/history_{task_id}.json`)
- `max_runs`: Number of iterations (2 for quick comparison tests)
- `optimize_method`: `DML` | `SMAC` | `MBO` | `TPE` | `GA` | `DDPG` | `TurBO`
- `transfer_framework`: `dml` | `workload_map` | `rgpe` | `none`

**DML-specific fields:**
- `dml_model_path`: Path to trained model (e.g., `../autotune/optimizer/dml_models/context_model.pth`)
- `dml_context_metrics_path`: Path to context metrics CSV
- `dml_result_data_dir`: Path to result CSVs directory
- `prometheus_url`: `http://localhost:9090` (or empty to use DBTune collectors)

**OtterTune (workload_map) specific:**
- `transfer_framework = workload_map`
- `optimize_method = SMAC`
- `data_repo`: Path to directory with source JSON history files

### 2. Load benchmark data (if needed)
```bash
cd third_party/oltpbench
java -jar oltpbench-0.1-SNAPSHOT.jar -b {benchmark} \
  -c config/{xml_config} --create=true --load=true -s 5
```
Benchmarks: `tpcc`, `ycsb`, `twitter`, `wikipedia`, `seats`, `smallbank`, `tatp`

### 3. Run the experiment
```bash
cd scripts && python optimize.py --config={config_file}.ini
```

### 4. Check results
History JSON saved to: `scripts/DBTune_history/history_{task_id}.json`

```python
import json
with open('DBTune_history/history_{task_id}.json') as f:
    h = json.load(f)
for i, d in enumerate(h['data']):
    tps = d['external_metrics'].get('tps', 'N/A')
    ctx = d.get('context')
    print(f'Iteration {i+1}: TPS={tps}, context={ctx}')
```

## Excluding Target Workload from Source Data

When testing transfer to a specific workload (e.g., TPC-C), you must exclude that workload from all source data so the model/algorithm has never seen it. This applies to **both DML and OtterTune**.

### For DML:
1. **Retrain the model** excluding target workload data:
   - Filter `context_default_metrics_all.csv` to remove rows with target workload in `context_id`
   - Regenerate triplet data without target workload contexts (run `DBMSTransferLearning/dataset/cross_validation.py` or filter `full_triplet_data_concordance.csv`)
   - Retrain: `cd scripts && python train_dml_model.py --context_csv <filtered_csv> --triplet_csv <filtered_triplets> --output_dir ../autotune/optimizer/dml_models_no_{workload}/`
2. **Update config** to point to the new model directory:
   - `dml_model_path = ../autotune/optimizer/dml_models_no_{workload}/context_model.pth`
   - `dml_context_metrics_path = ../autotune/optimizer/dml_models_no_{workload}/context_default_metrics_all.csv`
3. **Filter result CSVs**: The `dml_result_data_dir` result CSVs contain all workloads per hardware. The DML optimizer reads `workload_label` from these CSVs to build context IDs. Either:
   - Filter the CSVs to exclude target workload rows, OR
   - The model won't match a target workload context anyway (since it's not in context_default_metrics_all.csv), so the result CSVs can stay as-is — only contexts in the context CSV are matchable.

### For OtterTune (workload_map):
1. **Filter source history JSONs**: Create a filtered directory excluding target workload files:
   ```bash
   mkdir -p scripts/DBTune_history/csv_source_no_{workload}
   cd scripts/DBTune_history/csv_source
   for f in *.json; do
       echo "$f" | grep -q "{workload}" || ln -s "$(pwd)/$f" ../csv_source_no_{workload}/"$f"
   done
   ```
2. **Update config**: `data_repo = ./DBTune_history/csv_source_no_{workload}`

### Naming convention for excluded-workload directories:
- `dml_models_no_tpcc/` — DML model trained without TPC-C
- `csv_source_no_tpcc/` — OtterTune source JSONs without TPC-C
- Pattern: `{type}_no_{workload}/`

## IMPORTANT RULES
- **NEVER change knob config files** (e.g., `DML_12.json`, `OLTP_8.0.json`) without asking the user first
- **NEVER change OLTPBench XML configs** without asking the user first
- When comparing methods, use the **same knob set, workload, and benchmark settings**
- **Always exclude target workload from source data** when testing transfer learning approaches
- DML iteration 0 always runs default config to collect baseline metrics; iteration 1 applies transferred config
- For "unseen workload" tests, exclude that workload from training data and use a separate model directory (e.g., `dml_models_no_tpcc/`)

## Current Environment
- MySQL binary build at `../mysql_build/`
- MySQL socket: `../mysql_build/mysql.sock`
- MySQL cnf: `../mysql_build/cnf/my.cnf`
- OLTPBench at `third_party/oltpbench/`
- Prometheus: `http://localhost:9090`, MySQL exporter: `mysqld-exporter:9104`, Node exporter: `node-exporter:9100`
- DML knob set (12 knobs): `scripts/experiment/gen_knobs/DML_12.json`
- DML models: `autotune/optimizer/dml_models/` (full), `autotune/optimizer/dml_models_no_tpcc/` (TPC-C excluded)
- Source history JSONs: `scripts/DBTune_history/csv_source/` (all), `scripts/DBTune_history/csv_source_no_tpcc/` (TPC-C excluded)
- Python venv: `source venv/bin/activate`
