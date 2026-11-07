#!/bin/bash
# ==============================================================================
# eval_4gb_v14_reeval_n10.sh
#
# Re-eval de 4GB v14 con candidatos 5, 15, 195.
# Motivo: la eval anterior (fullrebuild2) usó 5, 10, 15 (auto-detect).
# Añadimos checkpoint 195 (modelo final) que nunca se evaluó formalmente.
#
# Protocolo idéntico al fullrebuild2: rebuild + 20 warmup + N=10 medidos.
# ==============================================================================

set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
source venv/bin/activate
cd "$(dirname "$0")/../.."

TASK_ID="dbtune_specialist_4gb_v14"
CONFIG="scripts/config_specialist_4gb_v14.ini"
CKPT_PREFIX="model_params/${TASK_ID}"
RESULTS_DIR="results/formal_eval/v14_4gb_reeval_n10"
N_SKIP=2
N_MEASURE=10

mkdir -p "$RESULTS_DIR" logs

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $*"; }

trim_pg_node() {
    local POD NODE_NAME NODE_IP
    POD=$(kubectl get pods -n pg -l cnpg.io/cluster=pg-cluster -o name | head -n 1)
    [ -z "$POD" ] && { log "WARN: no pod, skip trim."; return; }
    NODE_NAME=$(kubectl get "$POD" -n pg -o jsonpath='{.spec.nodeName}' 2>/dev/null)
    NODE_IP=$(kubectl get node "$NODE_NAME" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null)
    [ -z "$NODE_IP" ] && { log "WARN: no IP de $NODE_NAME, skip trim."; return; }
    log "fstrim /storage en $NODE_NAME ($NODE_IP)..."
    ssh -o ConnectTimeout=10 -o BatchMode=yes ${FSTRIM_SSH_USER:?set FSTRIM_SSH_USER}@"$NODE_IP" \
        "sync && sudo fstrim -v /storage" 2>&1 || log "WARN: fstrim falló."
}

TS=$(date +%Y%m%d_%H%M%S)

log "======================================================================"
log "RE-EVAL 4GB v14 — candidatos: step 5, 15, 195"
log "Results: $RESULTS_DIR"
log "======================================================================"

# ── Aplicar recursos K8s tier 4GB ────────────────────────────────────────────
kubectl patch cluster pg-cluster -n pg --type merge \
    -p '{"spec":{"resources":{"limits":{"memory":"4Gi","cpu":"3000m"},"requests":{"memory":"4Gi","cpu":"3000m"}}}}' \
    || { log "ERROR: no se pudo parchear cluster"; exit 1; }
kubectl delete pod -l cnpg.io/cluster=pg-cluster -n pg \
    --wait=true --grace-period=5 2>/dev/null || true
sleep 15
kubectl wait pod -n pg -l cnpg.io/instanceRole=primary \
    --for=condition=Ready --timeout=300s

# ── Normalizer v14 ────────────────────────────────────────────────────────────
for pkl in "specialist_4gb_v14_mean_var.pkl" \
           "specialist_4gb_v14_mean_var_rpi_4gb.pkl"; do
    [ -f "$pkl" ] && log "Normalizer en raíz: $pkl" && continue
    src="runs/${TASK_ID}/${pkl}"
    [ -f "$src" ] && cp "$src" . && log "Normalizer copiado: $pkl"
done

# ── Verificar checkpoints ─────────────────────────────────────────────────────
CANDIDATES=""
for step in 5 15 195; do
    f="${CKPT_PREFIX}_${step}_actor.pth"
    if [ -f "$f" ]; then
        CANDIDATES="${CANDIDATES}${CKPT_PREFIX}_${step},"
        log "Checkpoint step $step: OK"
    else
        log "WARN: step $step no encontrado — omitido"
    fi
done
CANDIDATES="${CANDIDATES%,}"
[ -z "$CANDIDATES" ] && { log "FATAL: no hay checkpoints disponibles"; exit 1; }
log "Candidatos: $CANDIDATES"

# ── Orden aleatorio de las 3 configs ─────────────────────────────────────────
ORDER=$(python3 -c "import random; c=['agent','pgtune','default']; random.shuffle(c); print(' '.join(c))")
log "Orden: $ORDER"

# ── Función: rebuild completo + warmup + medición ─────────────────────────────
run_config() {
    local mode="$1" label="$2" outfile="$3"
    local total=$((N_SKIP + N_MEASURE))

    log "── REBUILD + WARMUP → $label ──"
    kubectl delete jobs -n benchmark --all --ignore-not-found=true 2>/dev/null || true
    ./scripts/rebuild_benchmark_db.sh || { log "FATAL: rebuild falló"; exit 1; }

    log "Warm-up neutral (20 runs Default)..."
    python3 scripts/run_baseline.py \
        --config "$CONFIG" --mode default --tier "4gb" \
        --steps 20 --skip-first 20 --output /dev/null 2>/dev/null || true

    log "Midiendo $label ($total runs: $N_SKIP skip + $N_MEASURE medidos)..."
    if [ "$mode" = "agent" ]; then
        python3 scripts/run_inference.py \
            --config "$CONFIG" --tier "4gb" \
            --checkpoint "$CANDIDATES" \
            --candidates 1 --n-runs "$total" --skip-first "$N_SKIP" \
            --output "$outfile" || log "ERROR en inferencia"
    else
        python3 scripts/run_baseline.py \
            --config "$CONFIG" --mode "$mode" --tier "4gb" \
            --steps "$total" --skip-first "$N_SKIP" \
            --output "$outfile" || log "ERROR en baseline $mode"
    fi

    [ -f "$outfile" ] && python3 -c "
import json, numpy as np
d = json.load(open('$outfile'))
lst = d.get('measured_tps', d.get('effective_tps', []))
arr = [t for t in lst if t > 0]
print('  >>> $label: %.1f ± %.1f TPS (N=%d)' % (np.mean(arr), np.std(arr), len(arr)))
" 2>/dev/null || true
}

for mode in $ORDER; do
    case "$mode" in
        agent)   run_config "agent"   "DBTune-S" "${RESULTS_DIR}/agent_${TS}.json" ;;
        pgtune)  run_config "pgtune"  "PGTune"   "${RESULTS_DIR}/pgtune_${TS}.json" ;;
        default) run_config "default" "Default"  "${RESULTS_DIR}/default_${TS}.json" ;;
    esac
done

# ── Análisis estadístico ──────────────────────────────────────────────────────
log "=== ANÁLISIS 4GB v14 re-eval ==="
python3 << PYEOF
import json, numpy as np, os, sys
from scipy import stats

def load_tps(f):
    if not os.path.exists(f): return []
    d = json.load(open(f))
    lst = d.get('measured_tps', d.get('effective_tps', []))
    return [t for t in lst if t > 0]

def cohen_d(x, y):
    pool = np.sqrt((np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2) / 2)
    return (np.mean(x) - np.mean(y)) / pool if pool > 0 else 0

a  = load_tps('${RESULTS_DIR}/agent_${TS}.json')
pg = load_tps('${RESULTS_DIR}/pgtune_${TS}.json')
df = load_tps('${RESULTS_DIR}/default_${TS}.json')
n  = min(len(a), len(pg), len(df))

if n < 5:
    print('  N insuficiente (%d)' % n); sys.exit(0)

a, pg, df = a[:n], pg[:n], df[:n]
w_ad = stats.wilcoxon(a, df, alternative='greater')
w_ap = stats.wilcoxon(a, pg, alternative='greater')
ci   = stats.t.interval(0.95, n-1, np.mean(a), stats.sem(a))

print('  Agent:   %.1f ± %.1f  CI95=[%.1f, %.1f]' % (np.mean(a), np.std(a), ci[0], ci[1]))
print('  Default: %.1f ± %.1f' % (np.mean(df), np.std(df)))
print('  PGTune:  %.1f ± %.1f' % (np.mean(pg), np.std(pg)))
print('  vs Default: %+.1f%%  p=%.4f  d=%.2f' % ((np.mean(a)/np.mean(df)-1)*100, w_ad.pvalue, cohen_d(a, df)))
print('  vs PGTune:  %+.1f%%  p=%.4f  d=%.2f' % ((np.mean(a)/np.mean(pg)-1)*100, w_ap.pvalue, cohen_d(a, pg)))

# Comparación con eval anterior (fullrebuild2)
prev_agent = 324.9
print('  vs anterior (v14 step 15): %+.1f TPS (%+.1f%%)' % (np.mean(a)-prev_agent, (np.mean(a)/prev_agent-1)*100))

summary = {
    'tier': '4gb', 'task_id': '${TASK_ID}', 'n': n,
    'checkpoint_candidates': '5 15 195',
    'agent':   {'values': a,  'mean': float(np.mean(a)),  'std': float(np.std(a)),  'ci95': list(ci)},
    'pgtune':  {'values': pg, 'mean': float(np.mean(pg)), 'std': float(np.std(pg))},
    'default': {'values': df, 'mean': float(np.mean(df)), 'std': float(np.std(df))},
    'delta_vs_pgtune_pct':  float((np.mean(a)/np.mean(pg)-1)*100),
    'delta_vs_default_pct': float((np.mean(a)/np.mean(df)-1)*100),
    'wilcoxon_ap': {'p': float(w_ap.pvalue)},
    'wilcoxon_ad': {'p': float(w_ad.pvalue)},
    'protocol': 'rebuild+warmup por config, N=10, skip=2, candidates=5,15,195'
}
json.dump(summary, open('${RESULTS_DIR}/summary_${TS}.json', 'w'), indent=2)
print('  Guardado: ${RESULTS_DIR}/summary_${TS}.json')
PYEOF

trim_pg_node
log "======================================================================"
log "RE-EVAL 4GB v14 COMPLETADA — ver $RESULTS_DIR"
log "======================================================================"
