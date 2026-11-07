#!/bin/bash
# DBTune Specialist 2GB — v12
# Igual que v11 pero con fstrim correcto: SSH al nodo K8s (worker02, /storage)
# en vez de fstrim / local (que trimeaba el portátil orquestador, no el SSD de Postgres).
# Fix detectado en v14 — aplicado aquí para que los resultados 2gb sean comparables.

set -e
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
source venv/bin/activate
cd "$(dirname "$0")/../.."

NAMESPACE="pg"; CLUSTER_NAME="pg-cluster"
TASK_ID="dbtune_specialist_2gb_v12"
CONFIG_FILE="scripts/config_specialist_2gb_v11.ini"
TIER_ID="rpi_2gb"; LIMIT_MEM="2Gi"; LIMIT_CPU="1000m"; STEPS=200
RECAL_INTERVAL=25

mkdir -p logs
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $*"; }

trim_pg_node() {
    local POD NODE_NAME NODE_IP
    POD=$(kubectl get pods -n pg -l cnpg.io/cluster=$CLUSTER_NAME -o name | head -n 1)
    if [ -z "$POD" ]; then
        log "WARN: no se encontró pod de postgres, skip trim."
        return
    fi
    NODE_NAME=$(kubectl get "$POD" -n pg -o jsonpath='{.spec.nodeName}' 2>/dev/null)
    NODE_IP=$(kubectl get node "$NODE_NAME" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null)
    if [ -z "$NODE_IP" ]; then
        log "WARN: no se pudo resolver IP de $NODE_NAME, skip trim."
        return
    fi
    log "Trimming /storage en $NODE_NAME ($NODE_IP)..."
    ssh -o ConnectTimeout=10 -o BatchMode=yes ${FSTRIM_SSH_USER:?set FSTRIM_SSH_USER}@"$NODE_IP" \
        "sync && sudo fstrim -v /storage" 2>&1 || log "WARN: fstrim remoto falló en $NODE_NAME."
}

log "=== DBTune Specialist 2gb v12 | fstrim correcto | $STEPS steps ==="

kubectl patch cluster $CLUSTER_NAME -n $NAMESPACE --type merge --patch-file postgresql-training.yaml
kubectl patch cluster $CLUSTER_NAME -n $NAMESPACE --type merge \
    --patch "{\"spec\":{\"instances\":1,\"resources\":{\"limits\":{\"memory\":\"$LIMIT_MEM\",\"cpu\":\"$LIMIT_CPU\"},\"requests\":{\"memory\":\"$LIMIT_MEM\",\"cpu\":\"$LIMIT_CPU\"}}}}"
kubectl delete pod -l cnpg.io/cluster=$CLUSTER_NAME -n $NAMESPACE --wait=true --grace-period=5 2>/dev/null || true
sleep 15
kubectl wait --for=condition=Ready pod -l cnpg.io/cluster=$CLUSTER_NAME -n $NAMESPACE --timeout=600s
sleep 20

log "Rebuilding TPC-C schema..."
./scripts/rebuild_benchmark_db.sh

log "Midiendo baseline_tps (Default, N=3, skip-first=1)..."
_BL_JSON=$(mktemp /tmp/baseline_2gb_XXXX.json)
python3 scripts/run_baseline.py --config "$CONFIG_FILE" --mode default --tier 2gb --steps 3 --skip-first 1 --output "$_BL_JSON" || true
if [ -f "$_BL_JSON" ]; then
    _BL=$(python3 -c "import json; d=json.load(open('$_BL_JSON')); print(round(d['avg_tps'],1))" 2>/dev/null || echo "")
    rm -f "$_BL_JSON"
fi
if [ -n "${_BL:-}" ] && [ "$_BL" != "0.0" ]; then
    export FOGPG_BASELINE_TPS=$_BL; log "baseline_tps: $FOGPG_BASELINE_TPS TPS"
else
    export FOGPG_BASELINE_TPS=55.0; log "WARN: usando fallback $FOGPG_BASELINE_TPS TPS"
fi

log ">>> Training $STEPS steps (recalibración cada $RECAL_INTERVAL steps)"
TRAINING_CYCLE=1 python3 scripts/optimize.py \
    --config "$CONFIG_FILE" --task_id "$TASK_ID" --max_runs $STEPS \
    --fog_tier "$TIER_ID" --global_t_base 0 --steps_per_phase $STEPS \
    --recalibration_interval $RECAL_INTERVAL

log "=== SSD recovery (10 min) — fstrim correcto en nodo K8s ==="
trim_pg_node
sleep 600

log "=== Eval rápida ==="
./run_formal_eval.sh 2gb v12 2>&1 | tee logs/formal_eval_${TASK_ID}_$(date +%Y%m%d_%H%M%S).log

log "=== Eval interleaved N=10 (paper) ==="
./run_formal_eval_interleaved.sh 2gb v12 2>&1 | tee logs/formal_eval_interleaved_${TASK_ID}_$(date +%Y%m%d_%H%M%S).log

log "=== DONE — 2gb v12 completado ==="
