#!/bin/bash
# ==============================================================================
# FogPG-Tuner Generalista — Ablación (sin dims 28-30 de contexto hardware)
#
# Mismo protocolo boustrophedon que el generalista completo, pero con las
# dimensiones 28-30 del estado (contexto hardware: RAM, CPU, tier) desactivadas.
# Permite medir el impacto del hardware context en la generalización cross-tier.
#
# Totales:
#   C1: 75+60+60+60 = 255 · C2: 45+45+45+50 = 185 · C3: 35+35+35+40 = 145
#   C4: 15+15+15+15 = 60 · TOTAL: 645 steps
#
# Auto-resume: relanzar con el mismo comando detecta el último checkpoint.
#
# Requiere: KUBECONFIG, FSTRIM_SSH_USER (usuario SSH para fstrim remoto)
# ==============================================================================

set -e
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
cd "$(dirname "$0")/../.."
source venv/bin/activate

NAMESPACE="pg"
CLUSTER_NAME="pg-cluster"
TASK_ID="dbtune_generalist_ablation"
CONFIG_FILE="scripts/config_generalist_ablation.ini"
MAX_CYCLES=4       # 3 ciclos + 1 ronda de consolidación
RECAL_INTERVAL=25

declare -A TIER_PROFILE=([6gb]="rpi_6gb" [4gb]="rpi_4gb" [3gb]="rpi_3gb" [2gb]="rpi_2gb")
declare -A TIER_MEM=(   [6gb]="6Gi"      [4gb]="4Gi"      [3gb]="3Gi"      [2gb]="2Gi")
declare -A TIER_CPU=(   [6gb]="3500m"    [4gb]="3000m"    [3gb]="2000m"    [2gb]="1000m")
declare -A TIER_FALLBACK_BL=([6gb]="134.5" [4gb]="150.0" [3gb]="100.0" [2gb]="60.0")

# Orden boustrophedon: ciclos impares (1,3) = forward; pares (2,4) = backward.
# C4 (par) termina en 6GB → checkpoint final fresco en vecino del zero-shot 5GB.
phase_order_for_cycle() {
    local cycle=$1
    if (( cycle % 2 == 1 )); then
        echo "6gb 4gb 3gb 2gb"   # C1, C3: forward
    else
        echo "2gb 3gb 4gb 6gb"   # C2, C4: backward
    fi
}

# Último tier del ciclo (para detectar transiciones entre ciclos)
last_tier_of_cycle() {
    local cycle=$1
    if (( cycle % 2 == 1 )); then echo "2gb"; else echo "6gb"; fi
}

steps_for_phase() {
    local cycle=$1 tier=$2
    case $cycle in
        1) case $tier in 6gb) echo 75;; 4gb) echo 60;; 3gb) echo 60;; 2gb) echo 60;; esac ;;
        2) case $tier in 6gb) echo 50;; 4gb) echo 45;; 3gb) echo 45;; 2gb) echo 45;; esac ;;
        3) case $tier in 6gb) echo 40;; 4gb) echo 35;; 3gb) echo 35;; 2gb) echo 35;; esac ;;
        4) echo 15 ;;  # consolidación: uniform 15 steps/tier
    esac
}

mkdir -p logs
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $*"; }

trim_pg_node() {
    local POD NODE_NAME NODE_IP
    POD=$(kubectl get pods -n pg -l cnpg.io/cluster=$CLUSTER_NAME -o name | head -n 1)
    [ -z "$POD" ] && { log "WARN: no pod, skip trim."; return; }
    NODE_NAME=$(kubectl get "$POD" -n pg -o jsonpath='{.spec.nodeName}' 2>/dev/null)
    NODE_IP=$(kubectl get node "$NODE_NAME" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null)
    [ -z "$NODE_IP" ] && { log "WARN: no IP de $NODE_NAME, skip trim."; return; }
    log "fstrim /storage en $NODE_NAME ($NODE_IP)..."
    ssh -o ConnectTimeout=10 -o BatchMode=yes ${FSTRIM_SSH_USER:?set FSTRIM_SSH_USER}@"$NODE_IP" \
        "sync && sudo fstrim -v /storage" 2>&1 || log "WARN: fstrim remoto falló."
}

# ── Auto-resume ───────────────────────────────────────────────────────────────
EXISTING_STEPS=$(python3 scripts/get_max_checkpoint.py "$TASK_ID" 2>/dev/null || echo 0)
log "Checkpoints existentes: $EXISTING_STEPS steps acumulados"

GLOBAL_BASE=0

log "======================================================================"
log "ABLACIÓN — Boustrophedon (sin context dims) | MAX_CYCLES=$MAX_CYCLES | RECAL=$RECAL_INTERVAL"
log "Orden: C1(fwd) C2(rev) C3(fwd) C4-consolidación(rev, 15 steps/tier)"
log "ETA: ~645 steps × ~20 min ≈ 8-9 días"
log "======================================================================"

# ── Bucle principal ciclo × tier (boustrophedon) ──────────────────────────────
for (( CYCLE=1; CYCLE<=MAX_CYCLES; CYCLE++ )); do
    read -ra PHASE_ORDER <<< "$(phase_order_for_cycle $CYCLE)"
    CYCLE_LAST_TIER=$(last_tier_of_cycle $CYCLE)

    [ $CYCLE -eq 4 ] && log "=== INICIO RONDA DE CONSOLIDACIÓN (C4) ===" \
                     || log "=== INICIO CICLO $CYCLE / $((MAX_CYCLES-1)) ==="

    for TIER in "${PHASE_ORDER[@]}"; do
        STEPS=$(steps_for_phase $CYCLE $TIER)
        PHASE_END=$(( GLOBAL_BASE + STEPS ))

        # Auto-resume: saltar fases ya completadas
        if [ "$EXISTING_STEPS" -ge "$PHASE_END" ]; then
            log "Saltando C${CYCLE}-${TIER} (completado: base=$GLOBAL_BASE → end=$PHASE_END)"
            GLOBAL_BASE=$PHASE_END
            continue
        fi

        PROFILE=${TIER_PROFILE[$TIER]}
        MEM=${TIER_MEM[$TIER]}
        CPU=${TIER_CPU[$TIER]}
        FALLBACK=${TIER_FALLBACK_BL[$TIER]}

        log "======================================================================"
        log "INICIO C${CYCLE}-${TIER} | $PROFILE | mem=$MEM cpu=$CPU | $STEPS steps"
        log "global_t_base=$GLOBAL_BASE | TRAINING_CYCLE=$CYCLE"
        [ $CYCLE -eq 4 ] && log "(Consolidación: alpha rampa rápida 0.1→1.0 en 15 steps)"
        log "======================================================================"

        # ── Patch K8s ─────────────────────────────────────────────────────────
        kubectl patch cluster $CLUSTER_NAME -n $NAMESPACE --type merge \
            --patch-file k8s/postgresql-training.yaml
        kubectl patch cluster $CLUSTER_NAME -n $NAMESPACE --type merge \
            --patch "{\"spec\":{\"instances\":1,\"resources\":{\"limits\":{\"memory\":\"$MEM\",\"cpu\":\"$CPU\"},\"requests\":{\"memory\":\"$MEM\",\"cpu\":\"$CPU\"}}}}"
        kubectl delete pod -l cnpg.io/cluster=$CLUSTER_NAME -n $NAMESPACE \
            --wait=true --grace-period=5 2>/dev/null || true
        # Poll hasta que el operador CNPG recree el pod (evita "no matching resources" bajo set -e)
        for _i in $(seq 1 60); do
            kubectl get pod -l cnpg.io/cluster=$CLUSTER_NAME -n $NAMESPACE \
                -o name 2>/dev/null | grep -q pod/ && break
            sleep 5
        done
        kubectl wait --for=condition=Ready pod \
            -l cnpg.io/cluster=$CLUSTER_NAME -n $NAMESPACE --timeout=600s
        sleep 20

        # ── Rebuild BD ────────────────────────────────────────────────────────
        log "Rebuilding TPC-C schema..."
        ./scripts/rebuild_benchmark_db.sh

        # ── Baseline dinámico (N=5 skip=1 → efectivo N=4) ────────────────────
        log "Midiendo baseline_tps (Default, N=5, skip-first=1)..."
        _BL_JSON=$(mktemp /tmp/bl_gen9_${TIER}_XXXX.json)
        python3 scripts/run_baseline.py \
            --config "$CONFIG_FILE" \
            --mode default \
            --tier "$TIER" \
            --steps 5 \
            --skip-first 1 \
            --output "$_BL_JSON" || true

        _BL=""
        [ -f "$_BL_JSON" ] && \
            _BL=$(python3 -c \
                "import json; d=json.load(open('$_BL_JSON')); print(round(d['avg_tps'],1))" \
                2>/dev/null || echo "") && rm -f "$_BL_JSON"

        # Sanidad: rechazar mediciones < 50% del fallback (BD corrupta / pod no listo)
        if [ -n "$_BL" ] && awk "BEGIN{exit !($_BL > 0.5*$FALLBACK)}"; then
            export FOGPG_BASELINE_TPS=$_BL
            log "baseline_tps medido: $FOGPG_BASELINE_TPS TPS"
        else
            export FOGPG_BASELINE_TPS=$FALLBACK
            log "WARN: baseline '${_BL:-vacío}' fuera de rango — usando fallback $FOGPG_BASELINE_TPS TPS"
        fi

        # ── Training ──────────────────────────────────────────────────────────
        log "Training C${CYCLE}-${TIER}: $STEPS steps, recal cada $RECAL_INTERVAL"
        TRAINING_CYCLE=$CYCLE python3 scripts/optimize.py \
            --config "$CONFIG_FILE" \
            --task_id "$TASK_ID" \
            --max_runs $STEPS \
            --fog_tier "$PROFILE" \
            --global_t_base $GLOBAL_BASE \
            --steps_per_phase $STEPS \
            --recalibration_interval $RECAL_INTERVAL

        # Sincronizar GLOBAL_BASE al checkpoint real (evita desincronía bash vs Python por init_runs)
        GLOBAL_BASE=$(python3 scripts/get_max_checkpoint.py "$TASK_ID" 2>/dev/null || echo $PHASE_END)
        log "FIN C${CYCLE}-${TIER} — acumulado real: $GLOBAL_BASE steps (teórico: $PHASE_END)"

        # ── SSD recovery entre fases / ciclos ─────────────────────────────────
        # Última fase del último ciclo (C4): sin sleep, va directo al recovery de eval
        IS_VERY_LAST=0
        [ $CYCLE -eq $MAX_CYCLES ] && [ "$TIER" = "$CYCLE_LAST_TIER" ] && IS_VERY_LAST=1

        if [ $IS_VERY_LAST -eq 0 ]; then
            trim_pg_node
            if [ "$TIER" = "$CYCLE_LAST_TIER" ]; then
                # Entre ciclos: recovery más largo
                log "SSD recovery entre C${CYCLE}→C$((CYCLE+1)) (10 min)..."
                sleep 600
            else
                log "SSD recovery entre fases (5 min)..."
                sleep 300
            fi
        fi
    done
done

# ── SSD recovery largo antes de eval ─────────────────────────────────────────
log "======================================================================"
log "TRAINING COMPLETADO ($GLOBAL_BASE steps totales) — SSD recovery (10 min)"
log "Checkpoint final: tier 6GB (boustrophedon C4 termina en 6GB → zero-shot 5GB ✓)"
log "======================================================================"
trim_pg_node
sleep 600

# ── Eval inline (RQ1 + RQ3) ──────────────────────────────────────────────────
# set +e: permite que fallos puntuales de eval no abortan el script completo.
set +e

BEST_STEP=$(python3 scripts/get_max_checkpoint.py "$TASK_ID" 2>/dev/null || echo 0)
log "Checkpoint para eval: step $BEST_STEP"
log "Normalizer: per-tier (generalist_ablation_mean_var_rpi_{tier}.pkl); 5GB usa proxy 6GB"

EVAL_TIERS=(5gb 6gb 4gb 3gb 2gb)   # 5GB primero: zero-shot RQ3
declare -A EVAL_PROFILE=(
    [5gb]="rpi_5gb"
    [6gb]="rpi_6gb"
    [4gb]="rpi_4gb"
    [3gb]="rpi_3gb"
    [2gb]="rpi_2gb"
)
declare -A EVAL_MEM=([5gb]="5Gi" [6gb]="6Gi" [4gb]="4Gi" [3gb]="3Gi" [2gb]="2Gi")
declare -A EVAL_CPU=([5gb]="3200m" [6gb]="3500m" [4gb]="3000m" [3gb]="2000m" [2gb]="1000m")
# Normalizer per-tier: el global al final del run estaría calibrado para el
# último tier de C4 = 6GB, pero la red los distingue por context dims 28-30.
# rpi_5gb nunca visto → usa rpi_6gb como vecino más cercano (4600MB ≈ 5400MB).
declare -A EVAL_NORM_TIER=(
    [5gb]="rpi_6gb"
    [6gb]="rpi_6gb"
    [4gb]="rpi_4gb"
    [3gb]="rpi_3gb"
    [2gb]="rpi_2gb"
)

RESULTS_DIR="results/formal_eval/generalist_ablation"
mkdir -p "$RESULTS_DIR"

# Protocolo fullrebuild: rebuild + 20 warmup antes de CADA CONFIG individualmente.
# Idéntico al usado en specialists (eval_*_fullrebuild2_n10.sh) → comparable en el paper.
eval_one_config() {
    local mode="$1" label="$2" outfile="$3" tier="$4" norm_file="$5"
    local total=12  # skip 2 + 10 medidos

    log "  ── REBUILD + WARMUP → $label ($tier) ──"
    kubectl delete jobs -n benchmark --all --ignore-not-found=true 2>/dev/null || true
    ./scripts/rebuild_benchmark_db.sh || { log "ERROR: rebuild falló para $label $tier"; return 1; }

    python3 scripts/run_baseline.py \
        --config "$CONFIG_FILE" --mode default --tier "$tier" \
        --steps 20 --skip-first 20 --output /dev/null 2>/dev/null || true

    log "  Midiendo $label ($total runs)..."
    if [ "$mode" = "agent" ]; then
        python3 scripts/run_inference.py \
            --config "$CONFIG_FILE" --tier "$tier" \
            --checkpoint "model_params/${TASK_ID}_${BEST_STEP}" \
            --mean-var-file "$norm_file" \
            --candidates 1 --n-runs $total --skip-first 2 \
            --output "$outfile" \
            2>&1 | tee logs/generalist_ablation_agent_${tier}_$(date +%H%M%S).log || true
    else
        python3 scripts/run_baseline.py \
            --config "$CONFIG_FILE" --mode "$mode" --tier "$tier" \
            --steps $total --skip-first 2 \
            --output "$outfile" \
            2>&1 | tee logs/generalist_ablation_${mode}_${tier}_$(date +%H%M%S).log || true
    fi

    [ -f "$outfile" ] && python3 -c "
import json, numpy as np
d = json.load(open('$outfile'))
lst = d.get('measured_tps', d.get('effective_tps', []))
arr = [t for t in lst if t > 0]
print('    >>> $label ($tier): %.1f ± %.1f TPS (N=%d)' % (np.mean(arr), np.std(arr), len(arr)))
" 2>/dev/null || true
}

for TIER in "${EVAL_TIERS[@]}"; do
    MEM=${EVAL_MEM[$TIER]}
    CPU=${EVAL_CPU[$TIER]}
    PROFILE=${EVAL_PROFILE[$TIER]}
    NORM_PROFILE=${EVAL_NORM_TIER[$TIER]}
    NORM_FILE="generalist_ablation_mean_var_${NORM_PROFILE}.pkl"
    TS=$(date +%Y%m%d_%H%M%S)

    log "======================================================================"
    log "EVAL $TIER ($PROFILE) — fullrebuild por config (protocolo paper)"
    log "======================================================================"

    # Patch K8s al tier de eval
    kubectl patch cluster $CLUSTER_NAME -n $NAMESPACE --type merge \
        --patch "{\"spec\":{\"instances\":1,\"resources\":{\"limits\":{\"memory\":\"$MEM\",\"cpu\":\"$CPU\"},\"requests\":{\"memory\":\"$MEM\",\"cpu\":\"$CPU\"}}}}"
    kubectl delete pod -l cnpg.io/cluster=$CLUSTER_NAME -n $NAMESPACE \
        --wait=true --grace-period=5 2>/dev/null || true
    for _i in $(seq 1 60); do
        kubectl get pod -l cnpg.io/cluster=$CLUSTER_NAME -n $NAMESPACE \
            -o name 2>/dev/null | grep -q pod/ && break
        sleep 5
    done
    kubectl wait --for=condition=Ready pod \
        -l cnpg.io/cluster=$CLUSTER_NAME -n $NAMESPACE --timeout=600s
    sleep 20

    # Hard-fail si falta el normalizer
    if [ ! -f "$NORM_FILE" ]; then
        log "FATAL: normalizer $NORM_FILE no encontrado — saltando eval $TIER"
        continue
    fi
    log "Normalizer: $NORM_FILE"

    # Solo agent — Default y PGTune se reutilizan del generalista v9 en el paper
    # para evitar dos conjuntos de baselines con posibles diferencias por desgaste SSD
    log "Orden configs: agent (solo)"
    eval_one_config "agent" "DBTune-G-Ablation" "${RESULTS_DIR}/agent_${TIER}_${TS}.json" "$TIER" "$NORM_FILE"

    # Resumen tier
    python3 - << PYEOF 2>/dev/null || true
import json, numpy as np, os
def tps(f):
    if not os.path.exists(f): return []
    d = json.load(open(f))
    lst = d.get('measured_tps', d.get('effective_tps', []))
    return [t for t in lst if t > 0]
a  = tps('${RESULTS_DIR}/agent_${TIER}_${TS}.json')
pg = tps('${RESULTS_DIR}/pgtune_${TIER}_${TS}.json')
df = tps('${RESULTS_DIR}/default_${TIER}_${TS}.json')
if a and df:
    print('  RESUMEN ${TIER}: Agent=%.1f±%.1f  Default=%.1f  PGTune=%.1f  vs_def=%+.1f%%  vs_pgt=%+.1f%%' % (
        np.mean(a), np.std(a), np.mean(df), np.mean(pg) if pg else 0,
        (np.mean(a)/np.mean(df)-1)*100,
        (np.mean(a)/np.mean(pg)-1)*100 if pg else 0))
PYEOF

    trim_pg_node
    log "SSD recovery entre tiers de eval (5 min)..."
    sleep 300
done

log "======================================================================"
log "GENERALISTA ABLACIÓN COMPLETADO"
log "Resultados: $RESULTS_DIR"
log "======================================================================"
