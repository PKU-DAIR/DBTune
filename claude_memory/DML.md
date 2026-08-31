# DML.md — DML Embedding Model: files, code paths, and how to (re)train

The **DML (Deep Metric Learning)** model is a triplet-embedding network that maps a context's
default-config runtime metrics to a 16-D vector, used for **source-context selection** in the
one-shot transfer experiment (the `dmlmap` method = DML embedding selection + OtterTune
downstream; see [SETUP.md](SETUP.md) §8). This doc records everything needed to retrain it.

> TL;DR retrain: `cd DBMSTransferLearning/dataset && python generate_concordance.py && python triplets.py`, then `cd ../../scripts && python train_dml_model.py --output_dir ../autotune/optimizer/dml_models_<tag>/`, then repoint the `dmlmap` configs.

---

## 1. The model

- Architecture `EmbeddingNet`: `Linear(11→64) → ReLU → Linear(64→32) → ReLU → Linear(32→16)`,
  output L2-normalized to the unit sphere. **input_dim = 11**, embedding_dim = 16.
  Defined identically in [scripts/train_dml_model.py](scripts/train_dml_model.py),
  [autotune/optimizer/dml_optimizer.py](autotune/optimizer/dml_optimizer.py),
  and [autotune/transfer/tlbo/workload_map.py](autotune/transfer/tlbo/workload_map.py) (`_load_dml_embedding`).
- Loss: `TripletMarginLoss(margin=1.0, p=2)`, Adam lr=0.001, 64 batch, 50 epochs (CPU ~1 min).
- **Inputs scaled by a `MinMaxScaler`** fit on the context default metrics (`scaler.pkl`).

### The 11 metrics (model input — order matters)
Defined as `METRIC_NAMES` in [autotune/optimizer/dml_metrics.py](autotune/optimizer/dml_metrics.py)
and **must match** `FEATURE_COLS` in [scripts/train_dml_model.py](scripts/train_dml_model.py):
```
Average Memory Usage Percentage, InnoDB Buffer Pool Cache Hit Rate, InnoDB Dirty Buffer Pages,
Current QPS (Queries Per Second), Max CPU Usage (100 - Idle), InnoDB Rows Deleted (60s Rate),
InnoDB Rows Inserted (60s Rate), InnoDB Rows Read (60s Rate), InnoDB Rows Updated (60s Rate),
Average Disk IOPS (Read), Average Disk IOPS (Write)
```
The 12 tuned knobs (`PARAMS`/`CONFIG_PARAMS`): innodb_buffer_pool_size, innodb_read_io_threads,
innodb_write_io_threads, innodb_flush_log_at_trx_commit, innodb_adaptive_hash_index, sync_binlog,
innodb_lru_scan_depth, innodb_buffer_pool_instances, innodb_change_buffer_max_size,
innodb_io_capacity, innodb_log_file_size, table_open_cache.

---

## 2. Files & code paths

All training assets live under `DBMSTransferLearning/` (its own repo/submodule, see
[DBMSTransferLearning/CLAUDE.md](DBMSTransferLearning/CLAUDE.md)).

### Data
| path | what |
|---|---|
| `DBMSTransferLearning/dataset/[hw]-result.csv` | **Raw context data, processed 27-col schema** (7 hardware: 4c6g, 8c12g, 12c16g, 16c24g, 24c32g, 32c64g, 88c190g). Columns = `id` + 12 knobs + `tps` + 11 metrics + `workload_label` + `parameter_importance`. One row per config-run. **`triplets.py` reads these** (relative names, so it runs from `dataset/`). Now span 1M + 100k tables (was 1M-only); 88c190g is 1M-only. See `[[embedding-training-data-now-1m-plus-100k]]`. |
| `DBMSTransferLearning/dataset/backup/[hw]-result_{1M,100k}.csv` | **Raw 138-col exports** (messy Prometheus metric names + num_table/table_size/num_client/workload/skew). Pre-processing input. (Suffixes were historically off by 10×; corrected — `_1M` = 1M, `_100k` = 100k. No 10k data exists.) |
| `DBMSTransferLearning/dataset/full_data/[hw]-result.csv` | A separate, fuller copy (= dataset 1M + 100k). Not read by triplets.py. |
| `DBMSTransferLearning/dataset/context_default_metrics_all.csv` | **Per-context default-config metrics** (one row per context, the 11 metrics) — the embedding's reference/candidate set + what the scaler is fit on. Produced by `triplets.py`. |
| `DBMSTransferLearning/dataset/concordant_pair_ranking.csv` | Context-similarity ranking (concordance of config→tps orderings). Produced by `generate_concordance.py`; consumed by `triplets.py`. |
| `DBMSTransferLearning/dataset/full_triplet_data_concordance.csv` | **Triplets** (anchor/pos/neg × 11 metrics) — the actual training data. Produced by `triplets.py`. (`full_triplet_data_transfer.csv` is the alt "transfer ranking" variant.) |

### Scripts (the pipeline)
| path | role |
|---|---|
| [DBMSTransferLearning/neural_network/data_preprocess.py](DBMSTransferLearning/neural_network/data_preprocess.py) | Raw `dataset/backup/*.csv` (138-col) → cleaned. `unify_metrics` + `rename_columns` map messy Prometheus names → the 11 clean names; computes `parameter_importance` per workload. **Outputs 130-col `*_full_metrics.csv` to cwd** (the 27-col selection at the bottom is commented out — select `["id"]+PARAMS+columns` to match the dataset schema). Needs `ipython` in the venv + `PYTHONPATH=.` (imports `regression.*`). |
| [DBMSTransferLearning/dataset/generate_concordance.py](DBMSTransferLearning/dataset/generate_concordance.py) | per-hardware result CSVs → `concordant_pair_ranking.csv`. |
| [DBMSTransferLearning/dataset/triplets.py](DBMSTransferLearning/dataset/triplets.py) | `dataset/[hw]-result.csv` + `concordant_pair_ranking.csv` → `full_triplet_data_concordance.csv` + `context_default_metrics_all.csv`. The `FILES` list (line 7) selects which hardware; active triplet source = "Choice B: concordance" (`generate_triplets_concordance_csv`). |
| [scripts/train_dml_model.py](scripts/train_dml_model.py) | **The deployment trainer.** `--triplet_csv` + `--context_csv` → trains `EmbeddingNet(11,16)`, fits `MinMaxScaler` on the 11 cols, writes `context_model.pth` + `scaler.pkl` + a copy of the context CSV to `--output_dir`. |
| `DBMSTransferLearning/dataset/{cross_validation,concordance_test,topk}.py` | leave-one-out model gen / eval (produce `context_model_exclude_<id>.pth`, top-k retrieval tests). |
| `DBMSTransferLearning/dataset/train.py` | **Stale/older** trainer — uses `input_dim=12`. Do **not** use; the real pipeline is `scripts/train_dml_model.py` (11-D). |

### Deployed artifacts (what the experiment loads)
Each model dir holds `context_model.pth` (11-D embedding) + `scaler.pkl` (MinMaxScaler, loaded
from the model's dir) + a reference-contexts CSV. The `dmlmap` configs
(`dml_model_path` / `dml_context_metrics_path`) point at one of these:

| dir | contexts | what |
|---|---|---|
| `autotune/optimizer/dml_models_all/` | **293** (1M + 100k + tpcc) | **all-source** embedding (current default). Reference = `context_default_metrics_all.csv`. |
| `autotune/optimizer/dml_models_loo_<wl>/` | 286 | **per-target leave-one-out** embeddings (`<wl>` = read/rw50/write); exclude the target's `1M/<wl>/0.2` signature (7 HW variants) from triplets + reference. Built by [build_loo_embedding.py](scripts/lab/build_loo_embedding.py) (cached). |
| `autotune/optimizer/dml_models_sysbench/` | 147→140 | **older 1M-only** model + leave-one-**workload**-out reference CSVs `context_default_metrics_no_<wl>.csv` (drop all 21 of a workload type). Used by SETUP.md §8.5. |
| `autotune/optimizer/dml_models/` | — | `train_dml_model.py`'s default output dir (non-sysbench). |

### Inference / usage code paths
- `dmlmap` (embedding selection + OT downstream): [autotune/transfer/tlbo/workload_map.py](autotune/transfer/tlbo/workload_map.py) `_load_dml_embedding` (loads `.pth` + sibling `scaler.pkl` + the context CSV), `_dml_select_source` (embeds target iter-0 metrics, nearest context → source). Wired via [autotune/tuner.py](autotune/tuner.py) (`mapping_method=dml` → `surrogate_type=tlbo_dmlmap_*`) and [autotune/pipleline/pipleline.py](autotune/pipleline/pipleline.py) (bridges `DMLMAP_*` env vars).
- Original `DML` optimizer (GP-argmax downstream): [autotune/optimizer/dml_optimizer.py](autotune/optimizer/dml_optimizer.py).
- Metric collection at inference: [autotune/optimizer/dml_metrics.py](autotune/optimizer/dml_metrics.py) (`collect_metrics_from_prometheus`, `extract_metrics_from_observation`).

---

## 3. How to retrain (end to end)

Env: repo venv at `/work/xg26g002/x10563/DBTune/venv`; `DBMSTransferLearning` needs `PYTHONPATH`
pointing at itself (it imports `regression.*`), plus `ipython` (pip-installed) for `data_preprocess.py`.

```bash
source /work/xg26g002/x10563/DBTune/venv/bin/activate
cd /work/xg26g002/x10563/DBTune/DBMSTransferLearning

# (0) ONLY if raw backups changed: rebuild the 27-col dataset CSVs from dataset/backup/*.csv.
#     data_preprocess.py emits 130-col *_full_metrics.csv to cwd; select the 27-col schema
#     (id + PARAMS + [tps, 11 metrics, workload_label, parameter_importance]) and append/replace
#     dataset/[hw]-result.csv. (This is how the 100k data was merged — see the memory note.)
PYTHONPATH=. python neural_network/data_preprocess.py

# (1) similarity ranking + triplets + context metrics  (run from dataset/; no PYTHONPATH needed)
cd dataset
python generate_concordance.py        # reads the 7 [hw]-result.csv -> concordant_pair_ranking.csv
python triplets.py                    # -> full_triplet_data_concordance.csv + context_default_metrics_all.csv
cd ..

# (2) train the embedding + scaler
cd ../scripts
python train_dml_model.py \
    --triplet_csv ../DBMSTransferLearning/dataset/full_triplet_data_concordance.csv \
    --context_csv ../DBMSTransferLearning/dataset/context_default_metrics_all.csv \
    --output_dir  ../autotune/optimizer/dml_models_all/      # new dir to keep the old one
# -> dml_models_all/{context_model.pth, scaler.pkl, context_default_metrics_all.csv}

# (3) point the dmlmap configs at the new artifacts
#     in scripts/config_sysbench_dmlmap_{read,rw50,write}.ini:
#       dml_model_path           = ../autotune/optimizer/dml_models_all/context_model.pth
#       dml_context_metrics_path = ../autotune/optimizer/dml_models_all/context_default_metrics_all.csv
```

### Leave-one-out (per-target) embeddings

For a strict leave-one-target-out experiment, remove the target's exact signature from **both**
the embedding and the selection pool:

```bash
# cached: skips if the output dir already has context_model.pth
python scripts/lab/build_loo_embedding.py 64-1000000-4-oltp_read_only-0.2 \
       autotune/optimizer/dml_models_loo_read        # 286 contexts (293 − 7 HW variants)
```
`build_loo_embedding.py <exclude_pattern> <outdir>` filters every triplet/context referencing a
matching context out of the all-source CSVs, then runs `train_dml_model.py`. Then in the
`dmlmap` config: point `dml_model_path` / `dml_context_metrics_path` at `dml_models_loo_<wl>/`,
keep `data_repo = ./DBTune_history/csv_source` (full pool), and add
`[tune] exclude_contexts = <pattern>` so `tuner.load_history` drops the same contexts from the
pool in-memory. **Keep the embedding candidate set and the pool consistent** (both exclude the
same contexts).

(Older approach = leave-one-**workload**-out: `context_default_metrics_no_<wl>.csv` +
`data_repo=csv_source_no_<wl>` — drops the entire workload type; used by SETUP.md §8.5.)

---

## 4. Gotchas

- **11 vs 12 dims:** use `scripts/train_dml_model.py` (11-D, matches `dml_metrics.METRIC_NAMES`). `dataset/train.py` is a stale 12-D version — don't use it.
- **`triplets.py` reads `dataset/*.csv`** (bare relative `FILES`), *not* `dataset/full_data/*.csv`. Run it from `dataset/`. Whatever table sizes are in those CSVs define the trainable contexts.
- **`data_preprocess.py` 27-col selection is commented out** (bottom of `__main__`); its raw output is 130 cols. Apply `metric_df[["id"]+MySQLConfiguration().get_param_names()+columns]` to match the dataset schema before appending.
- **Candidate set vs downstream pool must agree:** the embedding's `context_default_metrics_*.csv` (what the embedding can pick) must match the downstream's `data_repo` pool. Current default = the **full** pool `scripts/DBTune_history/csv_source/` (293 contexts) + the `dml_models_all/` reference. For leave-one-out, exclude the **same** contexts from both: `[tune] exclude_contexts` (pool) + a `dml_models_loo_<wl>/` reference (embedding). The old `csv_source_no_<wl>/` dirs are the leave-one-workload-out pools.
- **Pre-build the `data_repo` cache** (`_history_cache.pkl`) before a parallel run; `tuner.load_history` builds it on first use and `exclude_contexts` filters in-memory, so the cache stays the full pool. (Why pre-build: concurrent tasks racing to build it on shared Lustre clobber each other — see [MIYABI.md](MIYABI.md) §5.)
- **Hardware-blindness:** the 11 features are all runtime stats — no explicit hardware-size feature. This is why the embedding over-selects small machines; more data of the same features won't fix it. See `[[why-dml-embedding-selects-poor-sources]]`.
- **`scaler.pkl` is loaded from the model file's directory**, so always keep `context_model.pth` + `scaler.pkl` together.
- `MySQLConfiguration` (param encoding, `get_param_names`, `preprocess_param_values`) lives in `DBMSTransferLearning/regression/system_configuration.py`.
