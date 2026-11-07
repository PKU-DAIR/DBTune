#!/bin/bash
# ==============================================================================
# specialist_launch_all_v14.sh
# Orquestador: lanza 4GB → 3GB → 6GB en secuencia automática
#
# Reemplaza a specialist_launch_all_v13.sh. El run 4GB v13 (step 88/200) se
# mató manualmente tras detectar que el fstrim de scripts/rebuild_benchmark_db.sh
# trimeaba el disco equivocado: corría `fstrim /` en ESTE host (el portátil
# orquestador), no en el nodo K8s que aloja el PVC de postgres (worker02,
# /storage, local-path). Es decir, el SSD real de Postgres nunca se trimeó en
# NINGÚN run anterior (2GB, 3GB, 4GB v11/v12/v13). El baseline_tps medido cayó
# un 22% entre v12 (245.7 TPS) y v13 (191.0 TPS) con BD recién reconstruida y
# mismo HW — consistente con fragmentación SSD acumulada sin resolver.
#
# Fix aplicado:
#   - rebuild_benchmark_db.sh ahora resuelve dinámicamente el nodo que aloja
#     el pod y hace `ssh doctorado@<nodo> sudo fstrim -v /storage` antes Y
#     después del rebuild, más VACUUM ANALYZE + CHECKPOINT tras la carga.
#   - worker02 se reinició por completo (reboot) antes de este relanzamiento
#     para descartar cualquier estado de kernel/page cache residual.
#   - task_id nuevo (v14) para 4GB, sin checkpoints previos.
#
# Metodología igual que runs anteriores: rebuild BD + baseline dinámico + 200
# steps con recalibración rolling cada 25 steps + eval rápida + eval interleaved N=10
#
# Uso:
#   nohup ./specialist_launch_all_v14.sh \
#       > logs/all_specialists_v14_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# ETA total: ~6-7 días (200 steps × ~15 min/step × 3 tiers + evals)
# ==============================================================================

set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
source venv/bin/activate
cd "$(dirname "$0")/../.."

NAMESPACE="pg"
CLUSTER_NAME="pg-cluster"
STEPS=200
RECAL_INTERVAL=25
ERRORS=0

mkdir -p logs

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $*"; }

# ── Trim del disco real del PVC (nodo K8s, no este host) ────────────────────
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

# ── Función: ejecutar un tier completo ───────────────────────────────────────
run_tier() {
    local LABEL="$1"       # "4gb", "3gb", "6gb"
    local TIER_ID="$2"     # "rpi_4gb", "rpi_3gb", "rpi_6gb"
    local MEM="$3"         # "4Gi", "3Gi", "6Gi"
    local CPU="$4"         # "3000m", "2000m", "3500m"
    local CONFIG="$5"
    local TASK_ID="$6"
    local FALLBACK_BL="$7"
    local EVAL_ARGS="${8:-$LABEL}"  # args para los scripts de eval (e.g. "4gb v14")

    log "======================================================================"
    log "INICIO TIER: $LABEL | mem=$MEM cpu=$CPU"
    log "task_id=$TASK_ID | config=$CONFIG"
    log "======================================================================"

    # ── Patch K8s ─────────────────────────────────────────────────────────────
    kubectl patch cluster $CLUSTER_NAME -n $NAMESPACE \
        --type merge --patch-file postgresql-training.yaml
    kubectl patch cluster $CLUSTER_NAME -n $NAMESPACE --type merge \
        --patch "{\"spec\":{\"instances\":1,\"resources\":{\"limits\":{\"memory\":\"$MEM\",\"cpu\":\"$CPU\"},\"requests\":{\"memory\":\"$MEM\",\"cpu\":\"$CPU\"}}}}"
    kubectl delete pod -l cnpg.io/cluster=$CLUSTER_NAME -n $NAMESPACE \
        --wait=true --grace-period=5 2>/dev/null || true
    sleep 15
    kubectl wait --for=condition=Ready pod \
        -l cnpg.io/cluster=$CLUSTER_NAME -n $NAMESPACE --timeout=600s
    sleep 20

    # ── Rebuild BD (incluye trim antes/después + VACUUM ANALYZE + CHECKPOINT) ──
    log "Rebuilding TPC-C schema..."
    ./scripts/rebuild_benchmark_db.sh

    # ── Medir baseline dinámico ───────────────────────────────────────────────
    log "Midiendo baseline_tps (Default, N=3, skip-first=1)..."
    _BL_JSON=$(mktemp /tmp/bl_${LABEL}_XXXX.json)
    python3 scripts/run_baseline.py \
        --config "$CONFIG" \
        --mode default \
        --tier "$LABEL" \
        --steps 3 \
        --skip-first 1 \
        --output "$_BL_JSON" || true

    _BL=""
    if [ -f "$_BL_JSON" ]; then
        _BL=$(python3 -c \
            "import json; d=json.load(open('$_BL_JSON')); print(round(d['avg_tps'],1))" \
            2>/dev/null || echo "")
        rm -f "$_BL_JSON"
    fi

    if [ -n "$_BL" ] && [ "$_BL" != "0.0" ]; then
        export FOGPG_BASELINE_TPS=$_BL
        log "baseline_tps medido: $FOGPG_BASELINE_TPS TPS"
    else
        export FOGPG_BASELINE_TPS=$FALLBACK_BL
        log "WARN: medición fallida — usando fallback $FOGPG_BASELINE_TPS TPS"
    fi

    # ── Training ──────────────────────────────────────────────────────────────
    log "Training: $STEPS steps, recalibración cada $RECAL_INTERVAL steps"
    TRAINING_CYCLE=1 python3 scripts/optimize.py \
        --config "$CONFIG" \
        --task_id "$TASK_ID" \
        --max_runs $STEPS \
        --fog_tier "$TIER_ID" \
        --global_t_base 0 \
        --steps_per_phase $STEPS \
        --recalibration_interval $RECAL_INTERVAL

    # ── SSD recovery (trim del disco real, no de este host) ─────────────────────
    log "Training completado — SSD recovery (10 min)..."
    trim_pg_node
    sleep 600

    # ── Eval rápida ───────────────────────────────────────────────────────────
    local TS_FAST=$(date +%Y%m%d_%H%M%S)
    log "Eval rápida $LABEL..."
    ./run_formal_eval.sh $EVAL_ARGS 2>&1 \
        | tee logs/formal_eval_${TASK_ID}_${TS_FAST}.log

    # ── Eval interleaved N=10 ─────────────────────────────────────────────────
    local TS_INT=$(date +%Y%m%d_%H%M%S)
    log "Eval interleaved N=10 $LABEL..."
    ./run_formal_eval_interleaved.sh $EVAL_ARGS 2>&1 \
        | tee logs/formal_eval_interleaved_${TASK_ID}_${TS_INT}.log

    log "FIN TIER: $LABEL ($(date))"
    log "======================================================================"
}

# ════════════════════════════════════════════════════════════════════════════════
log "SECUENCIA INICIADA: 4GB v14 → 3GB v12 → 6GB v12"
log "ETA: ~6-7 días (200 steps × ~15 min × 3 tiers + evals N=10)"
log "======================================================================"

# ── 4GB (v14: task_id nuevo, trim corregido al disco real, nodo reiniciado) ──
run_tier "4gb" "rpi_4gb" "4Gi" "3000m" \
    "scripts/config_specialist_4gb_v14.ini" \
    "dbtune_specialist_4gb_v14" \
    "150.0" \
    "4gb v14" \
|| {
    log "!!! ERROR en 4GB v14 — abortando secuencia"
    ERRORS=$((ERRORS+1))
    exit 1
}

# ── 3GB (v12: task_id nuevo, recalibración activa) ───────────────────────────
run_tier "3gb" "rpi_3gb" "3Gi" "2000m" \
    "scripts/config_specialist_3gb_v12.ini" \
    "dbtune_specialist_3gb_v12" \
    "100.0" \
    "3gb v12" \
|| {
    log "!!! ERROR en 3GB v12 — abortando secuencia"
    ERRORS=$((ERRORS+1))
    exit 1
}

# ── 6GB (v12: fresh model, recalibración activa, extended action space) ─────
# Preserva los checkpoints dbtune_specialist_6gb_v11_ext existentes (paper-ready).
run_tier "6gb" "rpi_6gb" "6Gi" "3500m" \
    "scripts/config_specialist_6gb_v12.ini" \
    "dbtune_specialist_6gb_v12" \
    "134.5" \
    "6gb v12" \
|| {
    log "!!! ERROR en 6GB v12 — abortando"
    ERRORS=$((ERRORS+1))
    exit 1
}

# ════════════════════════════════════════════════════════════════════════════════
log "======================================================================"
log "SECUENCIA COMPLETADA: 4GB v14 → 3GB v12 → 6GB v12"
log "Errores: $ERRORS"
log "Resultados en: results/formal_eval/"
log "  v14_4gb_n10/  v12_3gb_n10/  v12_6gb_n10/"
log "======================================================================"
