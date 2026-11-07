#!/bin/bash
# ==============================================================================
# eval_3gb_v12_fullrebuild2_n10.sh
#
# Segunda eval formal 3GB v12 — metodología idéntica al resto de tiers:
#   fullrebuild TPC-C (80WH) + 20-run warmup neutral ANTES de cada config
#   N=10 por config, orden aleatorio, skip=2
#
# Objetivo: replicar con datos frescos para confirmar/refutar 3GB vs PGTune
# Resultados en: results/formal_eval/v12_3gb_fullrebuild2_n10/
#
# Lanzar:
#   nohup ./scripts/eval/eval_3gb_v12_fullrebuild2_n10.sh \
#       > logs/eval_3gb_fullrebuild2_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
#
# Duración estimada: ~5h  (rebuild+warmup+12runs × 3 configs)
# ==============================================================================

set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
source venv/bin/activate
cd "$(dirname "$0")/../.."

TIER="3gb"
TASK_ID="dbtune_specialist_3gb_v12"
CONFIG="scripts/config_specialist_3gb_v12.ini"
CKPT_PREFIX="model_params/${TASK_ID}"
JSONL="runs/${TASK_ID}/training_steps.jsonl"
RESULTS_DIR="results/formal_eval/v12_3gb_fullrebuild2_n10"
N_SKIP=2
N_MEASURE=10
TS=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RESULTS_DIR" logs

# ── Copiar normalizer a raíz (run_inference.py lo busca ahí) ─────────────
for pkl in "specialist_${TIER}_v12_mean_var.pkl" \
           "specialist_${TIER}_v12_mean_var_rpi_${TIER}.pkl"; do
    src="runs/${TASK_ID}/${pkl}"
    [ -f "$src" ] && cp "$src" . && echo "  Normalizer copiado: ${pkl}" \
                  || echo "  AVISO: normalizer no encontrado: ${src}"
done

echo "======================================================================"
echo "  Eval formal 3GB v12 — fullrebuild2 (réplica metodología paper)"
echo "  Task:    $TASK_ID"
echo "  Config:  $CONFIG"
echo "  Results: $RESULTS_DIR"
echo "  N: $N_MEASURE medidos + $N_SKIP skip por config"
echo "======================================================================"

# ── Auto-detectar checkpoint (zona peak reward, no TPS) ──────────────────
PEAK_STEPS=$(python3 -c "
import json, glob, re
try:
    steps = [json.loads(l) for l in open('$JSONL')]
except Exception as e:
    print('10 15 25'); exit(0)
ckpts = glob.glob('${CKPT_PREFIX}_*_actor.pth')
avail = sorted(set(int(re.search(r'_(\d+)_actor', c).group(1)) for c in ckpts))
scored = []
for s in avail:
    w = [x.get('reward', x.get('tps', 0)) for x in steps if s-20 <= x.get('global_t', x.get('step', 0)) <= s]
    if len(w) >= 3:
        scored.append((sum(w)/len(w), s))
scored.sort(reverse=True)
top3 = sorted([s for _,s in scored[:3]])
print(' '.join(map(str, top3)))
" 2>/dev/null || echo "10 15 25")

CANDIDATES=$(echo $PEAK_STEPS | tr ' ' '\n' | \
    awk -v pre="${CKPT_PREFIX}" '{printf "%s_%s,", pre, $1}' | sed 's/,$//')

echo "  Checkpoint candidatos: steps $PEAK_STEPS"

# ── Orden aleatorio de configs ────────────────────────────────────────────
ORDER=$(python3 -c "
import random
configs = ['agent', 'pgtune', 'default']
random.shuffle(configs)
print(' '.join(configs))
")
echo "  Orden (aleatorio): $ORDER"
echo ""

# ── Función: rebuild + warmup + medición por config ──────────────────────
run_config() {
    local mode="$1"
    local label="$2"
    local outfile="$3"
    local total=$((N_SKIP + N_MEASURE))

    echo ""
    echo "──────────────────────────────────────────────────────────────────"
    echo "  REBUILD + WARMUP previo a: $label"
    echo "──────────────────────────────────────────────────────────────────"

    ./scripts/rebuild_benchmark_db.sh

    echo ""
    echo "  Warm-up neutral: 20 runs con Default (steady state)..."
    python3 scripts/run_baseline.py \
        --config "$CONFIG" \
        --mode default \
        --tier "$TIER" \
        --steps 20 \
        --skip-first 20 \
        --output /dev/null 2>/dev/null || true

    echo "  BD en steady state — midiendo $label ($total runs: $N_SKIP skip + $N_MEASURE medidos)..."
    echo ""

    if [ "$mode" = "agent" ]; then
        python3 scripts/run_inference.py \
            --config "$CONFIG" \
            --tier "$TIER" \
            --checkpoint "$CANDIDATES" \
            --candidates 1 \
            --n-runs "$total" \
            --skip-first "$N_SKIP" \
            --output "$outfile" || echo "  ERROR en inferencia del agente"
    else
        python3 scripts/run_baseline.py \
            --config "$CONFIG" \
            --mode "$mode" \
            --tier "$TIER" \
            --steps "$total" \
            --skip-first "$N_SKIP" \
            --output "$outfile" || echo "  ERROR en baseline $mode"
    fi

    if [ -f "$outfile" ]; then
        python3 -c "
import json, numpy as np
d = json.load(open('$outfile'))
tps_list = d.get('measured_tps', d.get('effective_tps', []))
if not tps_list and 'all_tps' in d:
    tps_list = d['all_tps'][$N_SKIP:]
arr = [t for t in tps_list if t > 0]
print('  >>> $label: %.1f ± %.1f TPS (N=%d)' % (np.mean(arr), np.std(arr), len(arr)))
print('  runs: %s' % [round(t,1) for t in arr])
" 2>/dev/null || echo "  (no se pudo leer resultado)"
    fi
}

# ── Evaluar cada config ───────────────────────────────────────────────────
for mode in $ORDER; do
    case "$mode" in
        agent)   run_config "agent"   "DBTune-S" "${RESULTS_DIR}/agent_${TS}.json" ;;
        pgtune)  run_config "pgtune"  "PGTune"   "${RESULTS_DIR}/pgtune_${TS}.json" ;;
        default) run_config "default" "Default"  "${RESULTS_DIR}/default_${TS}.json" ;;
    esac
done

# ── Análisis estadístico ──────────────────────────────────────────────────
echo ""
echo "=== ANÁLISIS ESTADÍSTICO — 3GB v12 fullrebuild2 ==="
python3 -c "
import json, numpy as np, os
from scipy import stats

def load_tps(f):
    if not os.path.exists(f): return []
    d = json.load(open(f))
    lst = d.get('measured_tps', d.get('effective_tps', []))
    if not lst and 'all_tps' in d:
        lst = d['all_tps']
    return [t for t in lst if t > 0]

def cohen_d(x, y):
    pooled = np.sqrt((np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2) / 2)
    return (np.mean(x) - np.mean(y)) / pooled if pooled > 0 else 0

agent   = load_tps('${RESULTS_DIR}/agent_${TS}.json')
pgtune  = load_tps('${RESULTS_DIR}/pgtune_${TS}.json')
default = load_tps('${RESULTS_DIR}/default_${TS}.json')

print('  Config      N=10 mean±std')
for name, arr in [('DBTune-S', agent), ('PGTune', pgtune), ('Default', default)]:
    if arr:
        print('  %-10s  %.1f ± %.1f    runs: %s' % (
            name, np.mean(arr), np.std(arr), [round(t,1) for t in arr]))

n = min(len(agent), len(pgtune), len(default))
if n >= 5:
    a, p, d = agent[:n], pgtune[:n], default[:n]

    w_ap = stats.wilcoxon(a, p, alternative='greater')
    w_ad = stats.wilcoxon(a, d, alternative='greater')
    ci_a = stats.t.interval(0.95, n-1, np.mean(a), stats.sem(a))

    print()
    print('  CI95 DBTune-S: [%.1f, %.1f]' % ci_a)
    print('  vs PGTune  N=10: %+.1f%%  p=%.4f  d=%.2f  %s' % (
        (np.mean(a)/np.mean(p)-1)*100, w_ap.pvalue, cohen_d(a,p),
        'OK' if w_ap.pvalue < 0.05 else 'ns'))
    print('  vs Default N=10: %+.1f%%  p=%.4f  d=%.2f  %s' % (
        (np.mean(a)/np.mean(d)-1)*100, w_ad.pvalue, cohen_d(a,d),
        'OK' if w_ad.pvalue < 0.05 else 'ns'))

    summary = {
        'tier': '3gb', 'task_id': '$TASK_ID', 'n': n,
        'checkpoint_candidates': '$PEAK_STEPS',
        'agent':   {'values': a, 'mean': float(np.mean(a)), 'std': float(np.std(a)), 'ci95': list(ci_a)},
        'pgtune':  {'values': p, 'mean': float(np.mean(p)), 'std': float(np.std(p))},
        'default': {'values': d, 'mean': float(np.mean(d)), 'std': float(np.std(d))},
        'delta_vs_pgtune_pct':  float((np.mean(a)/np.mean(p)-1)*100),
        'delta_vs_default_pct': float((np.mean(a)/np.mean(d)-1)*100),
        'wilcoxon_ap': {'p': float(w_ap.pvalue)},
        'wilcoxon_ad': {'p': float(w_ad.pvalue)},
        'protocol': 'rebuild+warmup por config, N=10, skip=2, Wilcoxon one-sided'
    }
    with open('${RESULTS_DIR}/summary_${TS}.json', 'w') as f:
        import json as j2; j2.dump(summary, f, indent=2)
    print()
    print('  Guardado: ${RESULTS_DIR}/summary_${TS}.json')
else:
    print('  N insuficiente (minimo 5)')
" 2>/dev/null || echo "  Error en análisis estadístico"

echo ""
echo "======================================================================"
echo "  COMPLETADO — resultados en: $RESULTS_DIR/"
echo "======================================================================"
