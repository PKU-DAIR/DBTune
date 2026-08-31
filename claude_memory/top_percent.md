# top_percent.md — Top-K% Source Selection: signal, RF predictor, experiments

The **top-K%-overlap** family of methods uses each context's set of near-optimal
configurations (top K% by TPS) to score source-target similarity. It's the strongest
signal we've found for one-shot transfer source selection — better than OtterTune's
decile binning, better than the DML embedding (see [SETUP.md](SETUP.md) §8 for those).

This doc records: what we measured, what was deployed, where the code lives, and how
to reproduce.

> TL;DR — to rerun on the existing Miyabi target hardware:
> 1. `python scripts/lab/train_rf_model.py --target_context 112c125g_64-1000000-4-oltp_read_only-0.2 --exclude '64-100000-4-oltp_' --output_dir autotune/optimizer/rf_models_loo_read/` (and the rw50/write equivalents).
> 2. `qsub scripts/lab/par_launch.sh` (default TASKS = OT + RF, seed 42, 6 nodes).
> 3. Results land in `scripts/DBTune_history/history_{ot,rf}_<wl>_s<seed>.json` and `parallel/<tag>/.rc`.

---

## 1. The signal: top-K%-Jaccard overlap

For a (target, source) pair, compute:

```
Top_K%(c) = the top K% of c's tested configs by TPS in c's own sweep
Jaccard(target, source) = |Top_K%(target) ∩ Top_K%(source)| / |Top_K%(target) ∪ Top_K%(source)|
```

This measures **agreement on the near-optimal region** — not whole-distribution rank
agreement (which is what concordance / Kendall-τ measures, and which the DML embedding
is trained to predict). For one-shot transfer we only consume the source's *argmax*
config, so the metric should focus on the top, not the bulk.

### Offline effectiveness — direct top-K% picker (pseudo-target oracle study)

Pseudo-targets = source contexts treated as targets. Score = transferred-config TPS ÷
target's own best (full sweep, not shared grid). Mean across 5 Miyabi targets:

| K | mean transferred/best | vs OT (0.845) | vs embedding (0.690) |
|---|---:|---:|---:|
| **1%** | **0.923** | **+0.08** | **+0.23** |
| 2% | 0.894 | +0.05 | +0.20 |
| 5% | 0.871 | +0.03 | +0.18 |
| 10% | 0.825 | −0.02 | +0.14 |
| 20% | 0.867 | +0.02 | +0.18 |

Tighter K is better. The ceiling for any model trained on this label is ~0.92 on
Miyabi; OT only reaches 0.845; the DML embedding 0.690.

### Live measurement — top-1% pinned via FORCE_SOURCE_CONTEXT

Job `2098599.opbs` (seed 42, workload_time=90, 100k sources excluded), source
forced to each target's offline-computed true-top-1% pick (the source with highest
actual Jaccard against the target):

| target | top-1% pick | top1 ratio | live × default | OT × default |
|---|---|---:|---:|---:|
| read | `24c32g read_only-1.0` (1M) | 0.214 | 1.21× | 1.18× |
| **rw50** | **`24c32g 100-wh TPCC`** | **0.889** | **3.06×** | 1.12× |
| **write** | `12c16g rw5-1.0` (1M) | 0.500 | **3.60×** | 1.20× |
| **mean** | | | **2.63×** | 1.17× |

So when given the *true* top-1% pick, the live downstream (GP-on-source + argmax-mean)
realises a ~2.2× advantage over OT.

> Note: for rw50, the true top-1% best is a **TPC-C** source, not sysbench-rw50. This
> reflects the durability/log-file structure the workloads share at the top of the
> config-TPS surface, not "workload type" similarity.

---

## 2. RF predictor for top-1%-overlap

We trained a **RandomForestRegressor** to predict pairwise top-1% Jaccard from a pair
of default-config metric vectors. Use case: at inference (live tuning iter-1), only
the target's iter-0 metric vector is available — not its full sweep — so we need a
*predictor* of overlap, not the direct metric.

### Architecture

- Input: two 114-D Prometheus default-config metric arrays (`internal_metrics[0]`).
- Pair features: `[|m_t − m_s|, (m_t + m_s)/2]` over the **usable** subset of the 114
  dims (non-NaN and non-constant across the training pool). 79 dims usable from
  csv_source JSONs → 158-D pair feature.
- Output: predicted Jaccard ∈ [0, 1].
- Estimator: `RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)`.

### Training data

- 293 source contexts in [scripts/DBTune_history/csv_source/](../scripts/DBTune_history/csv_source/).
- Per context, the default-config 114-vector is the mean of `internal_metrics` over
  observations whose `configuration` dict matches the DML_12 defaults
  (`bp=1073741824 bytes, log=50331648 bytes, flush=1, adaptive=1, ...`).
- **Leave-one-out per target**: by default `--target_context` triggers a set-UNION
  exclusion — any source whose context_id shares either (a) the target's workload
  signature (everything after the hw prefix: `64-1000000-4-oltp_read_only-0.2`) or
  (b) the target's hardware prefix (`112c125g_`).
- **100k pool exclusion**: add `--exclude '64-100000-4-oltp_'` to drop the 126 small-
  table-size sysbench contexts — they have the same workload type as their 1M twins
  but the size mismatch consistently degrades transfer.
- After both exclusions for the Miyabi targets: **160 sources** retained
  (293 − 126 (100k) − 7 (target workload sig) − 0 (Miyabi hw not in pool)).
- Labels: pairwise Jaccard over each context's top-1% set (~13 configs per context).
- ~25,440 training pairs (160 × 159) per model.

### Findings on the RF predictor

1. **Magnitude does NOT matter for the RF** (we tested this — tree splits are
   threshold-based, not distance-based). Standardizing features changes nothing.
2. **`MemTotal_bytes` (idx 10) is 98% of the target's L2 magnitude**, but the RF
   ignores raw magnitudes — it learns per-feature thresholds. So this isn't the
   "bias source" we initially thought.
3. **The real bias is in the label distribution / training pool**. TPC-C contexts
   have low per-second metric values because each transaction is heavy
   (NewOrder = ~10 SQL statements). Miyabi sysbench targets *also* have low
   per-second rates because Lustre slows the system. The RF can't distinguish
   "low rates because TPC-C" from "low rates because slow storage" — so it picks
   TPC-C for every sysbench target on Miyabi.
4. **Local vs live picks agree** — when we feed the offline 112c125g default-config
   metrics to the RF instead of live iter-0, the picks are the same (or swap within
   top 3). So the bias is in the model, not in live/offline drift.
5. **On live measurement, RF still tracks the top-1% picker on average** (mean
   tps1 = 1855 vs OT 1369 across 3 seeds × 3 workloads), but with **very high
   variance** (RF tps1 std ≥ OT std on every workload). The picks land in
   reasonable workload regions when averaged but individual seeds can give bad picks.

---

## 3. Files and code paths

### Trainer

- [`scripts/lab/train_rf_model.py`](../scripts/lab/train_rf_model.py): per-target trainer.
  - `--target_context <ctx>`: auto-derives set-UNION exclusion (workload sig OR hw).
  - `--no_workload_excl` / `--no_hw_excl`: disable each axis.
  - `--exclude <patterns>`: extra substring patterns (e.g. `'64-100000-4-oltp_'` for
    the 100k drop).
  - Outputs to `--output_dir`:
    - `rf_model.joblib` (the RandomForestRegressor, ~100–300 MB).
    - `source_default_metrics.csv` (rows = `context_id, m_0 ... m_113` — the source
      default-config 114-arrays in csv_source emission order, which equals live
      `internal_metrics[0]` order).
    - `rf_meta.json` (`{'usable_idx': [...], 'n_sources': N, 'exclude_patterns': [...]}`).

### Live runtime (mapping_method=rf)

Mirrors the `mapping_method=dml` wiring exactly:

| layer | file | what it does |
|---|---|---|
| **`mapping_method='rf'` branch** | [autotune/transfer/tlbo/workload_map.py](../autotune/transfer/tlbo/workload_map.py) | `_load_rf_model()` loads model + source vectors + usable_idx from `RFMAP_MODEL_PATH` / `RFMAP_CONTEXT_METRICS_PATH` / `RFMAP_META_PATH` env vars. `_train_rf(target_hpo_data)` reads `target_hpo_data.internal_metrics[0]`, slices to usable dims, builds pair features against each source, predicts overlap, picks argmax, calls shared `_finalize_match` for the OT downstream. Honors `FORCE_SOURCE_CONTEXT` for ablations. |
| **dispatch** | [autotune/optimizer/surrogate/core.py](../autotune/optimizer/surrogate/core.py) | `'rfmap' in func_str` → `WorkloadMapping(..., mapping_method='rf')`. |
| **surrogate_type** | [autotune/tuner.py](../autotune/tuner.py) | `mapping_method=rf` → `surrogate_type=tlbo_rfmap_<method>`. Passes `rf_model_path`, `rf_context_metrics_path`, `rf_meta_path` config keys through to `PipleLine`. |
| **env bridge** | [autotune/pipleline/pipleline.py](../autotune/pipleline/pipleline.py) | When `'rfmap' in surrogate_type`, sets `RFMAP_MODEL_PATH`, `RFMAP_CONTEXT_METRICS_PATH`, `RFMAP_META_PATH` env vars from the kwargs. |

### Per-target deployed artifacts

| dir | target | source pool (after exclusions) |
|---|---|---:|
| `autotune/optimizer/rf_models_loo_read/` | `112c125g_64-1000000-4-oltp_read_only-0.2` | 160 |
| `autotune/optimizer/rf_models_loo_rw50/` | `112c125g_64-1000000-4-oltp_read_write_50-0.2` | 160 |
| `autotune/optimizer/rf_models_loo_write/` | `112c125g_64-1000000-4-oltp_write_only-0.2` | 160 |

### Experiment configs

| config | mapping_method | exclude_contexts |
|---|---|---|
| `scripts/config_sysbench_rf_{read,rw50,write}.ini` | `rf` (live) | target workload sig + 100k |
| `scripts/config_sysbench_ot_{read,rw50,write}.ini` | `ottertune` (OT binning live) | target workload sig + 100k |

Each rf config points `rf_model_path` / `rf_context_metrics_path` / `rf_meta_path` at
the matching `rf_models_loo_<wl>/` artifacts above.

### Other top-K% picks (ablation)

- [`scripts/lab/top1_picks.json`](../scripts/lab/top1_picks.json) — true top-1% Jaccard pick per target (computed offline from full sweeps). Used by `method='top1'` in the parallel runner (FORCE_SOURCE_CONTEXT-pinned).
- [`scripts/lab/method_picks.json`](../scripts/lab/method_picks.json) — `ot`+`rf` offline picks (older RF, before the 100k drop). Used by an earlier ablation that pinned both selectors to their offline picks.
- [`scripts/lab/compute_rf_picks.py`](../scripts/lab/compute_rf_picks.py) — offline-only RF that produces `rf_picks.json`. Superseded by the live `mapping_method=rf` flow; kept for the offline-pinned ablation.

---

## 4. How to run experiments

All experiments use the existing Miyabi parallel infrastructure (see
[MIYABI.md](MIYABI.md) §3 for PBS / `pbsdsh` background).

### One-shot: rerun the canonical 6-task comparison

```bash
cd /work/xg26g002/x10563/DBTune
# (assumes the 3 rf_models_loo_<wl>/ artifacts are present; if not, see §5 below)
qsub scripts/lab/par_launch.sh         # default TASKS = ot/rf × read/rw50/write at seed 42
```

Default `par_launch.sh` tasks:
```
ot:read:42 ot:rw50:42 ot:write:42 rf:read:42 rf:rw50:42 rf:write:42
```
(`rf` here means **live runtime selection** via `mapping_method=rf`; the model picks
the source from target's actual iter-0 Prometheus metrics.)

### Override the task list (multi-seed, different methods)

```bash
# 3-seed OT vs RF, 18 nodes:
TASKS="ot:read:43;ot:read:44;ot:read:45;ot:rw50:43;ot:rw50:44;ot:rw50:45;\
ot:write:43;ot:write:44;ot:write:45;rf:read:43;rf:read:44;rf:read:45;\
rf:rw50:43;rf:rw50:44;rf:rw50:45;rf:write:43;rf:write:44;rf:write:45"
qsub -l select=18 -v "TASKS_OVERRIDE=${TASKS}" scripts/lab/par_launch.sh

# OT vs true top-1% (FORCE_SOURCE_CONTEXT-pinned per top1_picks.json):
TASKS="ot:read:42;ot:rw50:42;ot:write:42;top1:read:42;top1:rw50:42;top1:write:42"
qsub -v "TASKS_OVERRIDE=${TASKS}" scripts/lab/par_launch.sh
```

Supported methods in [`par_gen_task.py`](../scripts/lab/par_gen_task.py) /
[`par_node_run.sh`](../scripts/lab/par_node_run.sh):
- `ot` → `config_sysbench_ot_<wl>.ini`, mapping_method=ottertune (native binning live).
- `rf` → `config_sysbench_rf_<wl>.ini`, mapping_method=rf (the live RF described above).
- `top1` → `config_sysbench_ot_<wl>.ini` + `FORCE_SOURCE_CONTEXT=history_<top1_picks[wl]>`.
- `dml` → `config_sysbench_dmlmap_<wl>.ini` (DML embedding selector).

### Results

```bash
# verify job done
qstat <jobid>.opbs                                         # should be "No unfinished job found"
for t in ot_read_s42 rf_read_s42 ...; do cat parallel/$t/.rc; done   # 0 = success

# per-task TPS + picked source + transferred config
python3 - <<'PY'
import json
for t in ('ot_read_s42','rf_read_s42','ot_rw50_s42','rf_rw50_s42','ot_write_s42','rf_write_s42'):
    d=json.load(open(f'scripts/DBTune_history/history_{t}.json'))
    o=d['data']; tps=[round(x['external_metrics'].get('tps',0),1) for x in o]
    src=(o[-1].get('context') or {}).get('matched_context','-')
    c=o[1]['configuration']
    bp=c['innodb_buffer_pool_size']/(1024**3); lf=c['innodb_log_file_size']/(1024**2)
    print(f"{t:18s} tps0={tps[0]:>7.1f} tps1={tps[1]:>7.1f} improv={tps[1]/tps[0]:.2f}x "
          f"bp={bp:.0f}G flush={c['innodb_flush_log_at_trx_commit']} log={lf:.0f}M src={src}")
PY
```

`workload_time` is **90 s** in the current configs (set during the RF rollout; was
15 s in §8 of SETUP.md). Each task takes ~5 min: 30 s warmup + 90 s iter-0, then
30 s warmup + 90 s iter-1, plus ~2 min mysqld restart + datadir setup. Full 18-task
job runs in ~10 min wall time on `medium-c`.

---

## 5. How to retrain the RF (different target hardware / different pool)

```bash
cd /work/xg26g002/x10563/DBTune
for wl in read rw50 write; do
    case $wl in
        read)  tgt='112c125g_64-1000000-4-oltp_read_only-0.2' ;;
        rw50)  tgt='112c125g_64-1000000-4-oltp_read_write_50-0.2' ;;
        write) tgt='112c125g_64-1000000-4-oltp_write_only-0.2' ;;
    esac
    venv/bin/python scripts/lab/train_rf_model.py \
        --target_context "$tgt" \
        --exclude '64-100000-4-oltp_' \
        --output_dir "autotune/optimizer/rf_models_loo_${wl}/"
done
```

Per model takes ~60 s on CPU. Produces ~100 MB joblib each. To target different
hardware, change the `tgt` hardware prefix (e.g. `88c190g_…`); the `--target_context`
auto-exclusion will then keep all `112c125g_…` and `64-1000000-4-oltp_*` sources
that don't match the new target's workload signature.

To recompute true top-1% picks for a different target set, see
[`scripts/lab/top5_overlap_miyabi.py`](../scripts/lab/top5_overlap_miyabi.py)
(pairwise overlap CSV builder; the top-1% sister script lives in the same file by
changing the `frac` arg).

---

## 6. Known issues / open questions

- **High live variance for RF.** Across 3 seeds × 3 workloads, RF tps1 std was
  ≥ OT's on every workload, and one rw50 seed got `tps1=265` while another got `2614`.
  Partly attributable to per-compute-node iter-0 variance (some nodes had cold Lustre
  cache producing tps0 ≈ 370 for read where the median is ~3000). Need multi-seed
  with restricted node placement to separate model variance from infrastructure.
- **All-TPC-C bias for sysbench targets on Miyabi.** With 100k excluded, the RF
  picks TPC-C variants for every sysbench target (read, rw50, write). Root cause:
  TPC-C and Miyabi-sysbench share the "low per-second metric values" signature
  (heavy transactions vs slow Lustre, respectively). The training set has no
  "OOD-low-but-actually-sysbench" examples to learn the distinction.
- **Top-1% label has a hard ceiling per target.** For read on Miyabi, even the
  *best* available source has Jaccard only 0.21 (3 of 9 configs shared). No model
  trained on top-1% can beat that. For rw50 the best is 0.89 (8 of 9); top-1% is
  very informative there.

---

## 7. Document conventions

- All paths in this doc are relative to `/work/xg26g002/x10563/DBTune/` unless noted.
- `<wl>` is one of `read`, `rw50`, `write` — the three sysbench target workloads.
- "Miyabi target" = the 112-core / 125 GB compute node (see [MIYABI.md](MIYABI.md)
  for filesystem/PBS specifics).
- Source pool is `scripts/DBTune_history/csv_source/` (293 contexts: 7 hardware
  × { 7 sysbench workloads × {1M, 100k} table sizes × 3 skews + 3 tpcc warehouse
  counts }).
- `csv_source/history_<context_id>.json` and `dataset/full_data/<hw>-result.csv`
  contain the same underlying data; the JSON internal_metrics array order matches
  the CSV column order 19:133 (the 114-Prometheus-metric block), and both match the
  live runtime emission order at `internal_metrics[0]`.
