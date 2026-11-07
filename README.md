# FogPG-Tuner

**Autonomous PostgreSQL Configuration Tuning for Fog/Edge Kubernetes via Deep Reinforcement Learning**

> This repository is a fork of [DBTune (PKU-DAIR Lab)](https://github.com/PKU-DAIR/DBTune) extended with FogPG-Tuner, a DDPG-based agent for autonomous PostgreSQL knob tuning on resource-constrained fog/edge Kubernetes nodes.

---

## Overview

FogPG-Tuner trains a Deep Deterministic Policy Gradient (DDPG) agent to tune PostgreSQL configuration knobs on CloudNativePG clusters running on heterogeneous fog/edge hardware. The agent observes a 31-dimensional state vector, selects 5 knobs, and receives a reward that penalises swap usage and CPU pressure alongside throughput gains.

**Specialists** are trained per hardware tier (rpi_2gb, rpi_3gb, rpi_4gb, rpi_6gb). A **Generalista** is trained across all four tiers using cyclic domain randomisation (boustrophedon order) and evaluated zero-shot on an unseen 5 GB tier.

### Key design decisions

| Component | Detail |
|-----------|--------|
| State | 31 dimensions; dims 28–30 encode hardware context (RAM, CPU, tier) |
| Action | 5 knobs: `shared_buffers` (Postmaster), `effective_cache_size`, `work_mem`, `maintenance_work_mem`, `checkpoint_completion_target` (SIGHUP) |
| Reward | `FogAwareRewardV2` = throughput reward + swap penalty + CPU penalty |
| Hardware constants | `wal_buffers=16MB`, `random_page_cost=1.1`, `effective_io_concurrency=32`, `wal_compression=on`, `max_wal_size=4GB`, `min_wal_size=1GB` |
| Knob application | CloudNativePG CNPG patch via `cnpg_applier.py`; Postmaster knobs trigger a pod restart |
| Workload | TPC-C (HammerDB TPROC-C), 80 warehouses, run as a Kubernetes Job |

---

## Repository structure

```
autotune/
  optimizer/        # DDPG agent (EdgeDDPG, PrioritizedReplayMemory, OUProcess)
  reward/           # FogAwareRewardV2
  constraints/      # tier_profiles (hardware limits per tier)
  knob_management/  # cnpg_applier, pg_knob_classifier
  workload/         # k8s_tpcc_runner, k8s_tpch_runner
  database/         # PostgreSQL connector
scripts/
  launch/           # Specialist and Generalista training launchers
  eval/             # Formal evaluation scripts (N=10 protocol)
  config_*.ini      # Training configs per tier/agent (set passwd and node IP)
  rebuild_benchmark_db.sh
  run_baseline.py / run_inference.py
k8s/
  cnpg-1.27.1.yaml            # CNPG operator
  postgres_cluster.tpl.yaml   # Cluster template
  postgresql-training.yaml    # Training cluster manifest
  postgresql-eval.yaml        # Evaluation cluster manifest
  setup_benchmark_env.yaml    # Namespace + ConfigMap (set PGSUPERPASS)
  pg-bench-secret.example.yaml
tests/
```

Trained models and evaluation data are not included in this repository.

---

## Requirements

- Python 3.10+
- Kubernetes cluster with [CloudNativePG 1.27.1](https://cloudnative-pg.io/) (`k8s/cnpg-1.27.1.yaml`)
- kube-prometheus-stack (Prometheus + node-exporter)
- `postgres-exporter` sidecar (`k8s/postgres-exporter-final.yaml`)
- HammerDB (deployed as a K8s Job via `k8s/tprocc_rebuild.yaml`)

```bash
pip install -r requirements.txt
```

---

## Quick start

### 1. Deploy infrastructure

```bash
export KUBECONFIG=<path-to-your-kubeconfig>

kubectl apply -f k8s/cnpg-1.27.1.yaml

# Create the benchmark secret (copy example and fill in your password)
cp k8s/pg-bench-secret.example.yaml k8s/pg-bench-secret.yaml
# edit k8s/pg-bench-secret.yaml and set PGSUPERPASS
kubectl apply -f k8s/pg-bench-secret.yaml

kubectl apply -f k8s/setup_benchmark_env.yaml
kubectl apply -f k8s/postgresql-training.yaml
```

### 2. Configure

Edit the relevant `scripts/config_*.ini`:

```ini
[database]
passwd = <your-cnpg-app-user-password>  # or set PG_APP_PASSWORD env var
k8s_pg_primary_node_instance =          # leave empty for auto-discovery
```

### 3. Build the TPC-C database

```bash
export FSTRIM_SSH_USER=<ssh-user-on-storage-node>
bash scripts/rebuild_benchmark_db.sh
```

### 4. Train a specialist

```bash
# 2 GB tier specialist
nohup bash scripts/launch/specialist_launch_2gb_v12.sh \
    > logs/specialist_2gb_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# All tier specialists (3/4/6 GB)
nohup bash scripts/launch/specialist_launch_all_v14.sh \
    > logs/specialists_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 5. Train the Generalista

```bash
nohup bash scripts/launch/generalist_launch_v9.sh \
    > logs/generalista_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 6. Evaluate (N=10 formal protocol)

```bash
# 2 GB specialist
nohup bash scripts/eval/eval_2gb_v12_fullrebuild.sh \
    > logs/eval_2gb_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Statistical significance
python scripts/analyze_significance.py --tier rpi_2gb
```

The evaluation protocol: rebuild TPC-C → 20 neutral warm-up runs → random config order → N=10 measured runs per config → Mann-Whitney U + Holm-Bonferroni correction.

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `KUBECONFIG` | Path to kubeconfig (default: `$HOME/.kube/config`) |
| `FSTRIM_SSH_USER` | SSH user on the storage node for `fstrim` (required) |
| `FOGPG_BASELINE_TPS` | Override baseline TPS (auto-measured if unset) |
| `TRAINING_CYCLE` | Current training cycle number (injected by launchers) |
| `TASK_ID` | Agent task identifier for checkpoint resume |

---

## Citation

If you use this code, please cite:

```bibtex
@article{fogpg2026,
  title   = {FogPG-Tuner: Autonomous PostgreSQL Configuration Tuning for Fog/Edge
             Kubernetes via Deep Reinforcement Learning},
  author  = {Gomez Gonzalez, Alberto},
  journal = {Future Generation Computer Systems},
  year    = {2026},
  publisher = {Elsevier}
}
```

This work builds on [DBTune](https://github.com/PKU-DAIR/DBTune) (PKU-DAIR Lab, MIT License).

---

## License

MIT License — see [LICENSE](LICENSE).
