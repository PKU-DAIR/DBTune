#!/bin/bash
# ==============================================================================
# Evaluación formal con N=10 — cualquier tier specialist v11
#
# Protocolo N=10 intercalado:
#   1. Rebuild BD (estado limpio y reproducible)
#   2. Warm-up neutral: 20 runs con Default (sin sesgar buffer pool)
#   3. Orden aleatorio de configs para controlar drift de SSD
#   4. Por cada config: N_SKIP skip + N_MEASURE runs medidos
#   5. Wilcoxon pareado + CI95
#
# Usage:
#   nohup ./run_formal_eval_interleaved.sh <tier> [suffix] > logs/formal_eval_interleaved_<tier>_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Ejemplos:
#   ./run_formal_eval_interleaved.sh 6gb ext   → dbtune_specialist_6gb_v11_ext
#   ./run_formal_eval_interleaved.sh 4gb        → dbtune_specialist_4gb_v11
#   ./run_formal_eval_interleaved.sh 3gb        → dbtune_specialist_3gb_v11
# ==============================================================================

set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
source venv/bin/activate
cd "$(dirname "$0")/../.."

TIER="${1:-6gb}"
SUFFIX="${2:-}"

# Derivar TASK_ID y CONFIG según suffix/versión
# Si SUFFIX empieza por 'v[0-9]' (e.g. "v12"): versión explícita completa
# Si SUFFIX es texto (e.g. "ext"): sufijo legacy concatenado a v11
if [[ "${SUFFIX}" =~ ^v[0-9] ]]; then
    TASK_ID="dbtune_specialist_${TIER}_${SUFFIX}"
    CONFIG="scripts/config_specialist_${TIER}_${SUFFIX}.ini"
    RESULTS_DIR="results/formal_eval/${SUFFIX}_${TIER}_n10"
elif [ -n "$SUFFIX" ]; then
    TASK_ID="dbtune_specialist_${TIER}_v11_${SUFFIX}"
    CONFIG="scripts/config_specialist_${TIER}_v11_${SUFFIX}.ini"
    RESULTS_DIR="results/formal_eval/v11_${TIER}_${SUFFIX}_n10"
else
    TASK_ID="dbtune_specialist_${TIER}_v11"
    CONFIG="scripts/config_specialist_${TIER}_v11.ini"
    RESULTS_DIR="results/formal_eval/v11_${TIER}_n10"
fi
CKPT_PREFIX="model_params/${TASK_ID}"
JSONL="training_steps_${TASK_ID}.jsonl"
TS=$(date +%Y%m%d_%H%M%S)
N_SKIP=2
N_MEASURE=10

mkdir -p "$RESULTS_DIR" logs

echo "======================================================================"
echo "  Evaluación formal N=10 — ${TASK_ID} | config: $CONFIG"
echo "  N por config: $N_MEASURE | skip: $N_SKIP"
echo "  Total runs por config: $((N_SKIP + N_MEASURE))"
echo "======================================================================"

# ── Auto-detectar candidatos: 3 checkpoints de la zona peak del training ───
# La zona peak es donde el TPS promedio (ventana 20 steps) es más alto,
# considerando que el training degrada con el crecimiento de la BD.
# Se buscan los 3 mejores checkpoints disponibles por avg TPS en su ventana.
PEAK_STEPS=$(python3 -c "
import json, glob, re
try:
    steps = [json.loads(l) for l in open('$JSONL')]
except:
    print('5 10 15'); exit(0)
ckpts = glob.glob('${CKPT_PREFIX}_*_actor.pth')
avail = sorted(set(int(re.search(r'_(\d+)_actor', c).group(1)) for c in ckpts))
scored = []
for s in avail:
    w = [x['tps'] for x in steps if s-20 <= x['global_t'] <= s]
    if len(w) >= 3:
        scored.append((sum(w)/len(w), s))
scored.sort(reverse=True)
top3 = sorted([s for _,s in scored[:3]])
print(' '.join(map(str, top3)))
" 2>/dev/null || echo "5 10 15")
CANDIDATES=$(echo $PEAK_STEPS | tr ' ' '\n' | awk -v pre="${CKPT_PREFIX}" '{printf "%s_%s,", pre, $1}' | sed 's/,$//')
echo "  Checkpoint candidatos: steps $PEAK_STEPS (auto-detectados, zona peak training)"
echo "  Selección: 1 run por candidato sobre BD calentada → mejor TPS"

# ── Orden aleatorio de configs (controla drift SSD) ───────────────────────
ORDER=$(python3 -c "
import random
configs = ['agent', 'pgtune', 'default']
random.shuffle(configs)
print(' '.join(configs))
")
echo "  Orden de evaluación (aleatorio): $ORDER"
echo ""

# ── Rebuild BD ─────────────────────────────────────────────────────────────
echo "=== [1/5] REBUILD BD ==="
./scripts/rebuild_benchmark_db.sh
echo "  BD reconstruida"

# ── Warm-up neutral: 20 runs con Default ──────────────────────────────────
echo ""
echo "=== [2/5] WARM-UP NEUTRAL: 20 runs con Default ==="
python3 scripts/run_baseline.py \
    --config "$CONFIG" \
    --mode default \
    --tier "$TIER" \
    --steps 20 \
    --skip-first 20 \
    --output /dev/null 2>/dev/null || true
echo "  Warm-up completado — BD en steady state"

# ── Función: medir N runs de una config ───────────────────────────────────
run_config() {
    local mode="$1"   # "agent", "pgtune", "default"
    local label="$2"
    local outfile="$3"
    local total=$((N_SKIP + N_MEASURE))

    echo ""
    echo "=== Midiendo $label ($total runs: $N_SKIP skip + $N_MEASURE medidos) ==="

    if [ "$mode" = "agent" ]; then
        # Agente: run_inference.py evalúa los candidatos auto-detectados con 1 run
        # cada uno sobre la BD ya calentada y elige el mejor antes de los N runs formales
        python3 scripts/run_inference.py \
            --config "$CONFIG" \
            --tier "$TIER" \
            --checkpoint "$CANDIDATES" \
            --candidates 1 \
            --n-runs $total \
            --skip-first $N_SKIP \
            --output "$outfile" || {
                echo "  ERROR en inferencia del agente"
            }
    else
        # PGTune o Default: usa run_baseline.py
        python3 scripts/run_baseline.py \
            --config "$CONFIG" \
            --mode "$mode" \
            --tier "$TIER" \
            --steps $total \
            --skip-first $N_SKIP \
            --output "$outfile" || {
                echo "  ERROR en baseline $mode"
            }
    fi

    # Mostrar resultado
    if [ -f "$outfile" ]; then
        python3 -c "
import json
d = json.load(open('$outfile'))
tps_list = d.get('measured_tps', d.get('effective_tps', []))
if not tps_list and 'all_tps' in d:
    tps_list = d['all_tps'][$N_SKIP:]
import numpy as np
arr = [t for t in tps_list if t > 0]
print('  >>> $label: %.1f ± %.1f TPS (N=%d)' % (np.mean(arr), np.std(arr), len(arr)))
print('  runs: %s' % [round(t,1) for t in arr])
" 2>/dev/null || echo "  (no se pudo leer resultado)"
    fi
}

# ── Ejecutar configs en orden aleatorio ───────────────────────────────────
echo ""
echo "=== [3/5] EVALUACIÓN EN ORDEN: $ORDER ==="
STEP=3
for mode in $ORDER; do
    STEP=$((STEP+1))
    case "$mode" in
        agent)   run_config "agent"   "DBTune-S"  "${RESULTS_DIR}/agent_${TS}.json" ;;
        pgtune)  run_config "pgtune"  "PGTune"    "${RESULTS_DIR}/pgtune_${TS}.json" ;;
        default) run_config "default" "Default"   "${RESULTS_DIR}/default_${TS}.json" ;;
    esac
done

# ── Análisis estadístico ───────────────────────────────────────────────────
echo ""
echo "=== [5/5] ANÁLISIS ESTADÍSTICO ==="
python3 -c "
import json, numpy as np, os
from scipy import stats

def load_tps(f, n_skip):
    if not os.path.exists(f):
        return []
    d = json.load(open(f))
    lst = d.get('measured_tps', d.get('effective_tps', []))
    if not lst and 'all_tps' in d:
        lst = d['all_tps'][n_skip:]
    return [t for t in lst if t > 0]

agent   = load_tps('${RESULTS_DIR}/agent_${TS}.json',   $N_SKIP)
pgtune  = load_tps('${RESULTS_DIR}/pgtune_${TS}.json',  $N_SKIP)
default = load_tps('${RESULTS_DIR}/default_${TS}.json', $N_SKIP)

print('  Agente  : N=%d  mean=%.1f  std=%.1f' % (len(agent),   np.mean(agent)   if agent   else 0, np.std(agent)   if agent   else 0))
print('  PGTune  : N=%d  mean=%.1f  std=%.1f' % (len(pgtune),  np.mean(pgtune)  if pgtune  else 0, np.std(pgtune)  if pgtune  else 0))
print('  Default : N=%d  mean=%.1f  std=%.1f' % (len(default), np.mean(default) if default else 0, np.std(default) if default else 0))

n = min(len(agent), len(pgtune), len(default))
if n >= 5:
    a, p, d = agent[:n], pgtune[:n], default[:n]
    w_ap = stats.wilcoxon(a, p)
    w_ad = stats.wilcoxon(a, d)
    t_ap = stats.ttest_rel(a, p)
    t_ad = stats.ttest_rel(a, d)
    ci_a = stats.t.interval(0.95, n-1, np.mean(a), stats.sem(a))

    print()
    print('  DBTune-S CI95: [%.1f, %.1f]' % ci_a)
    print('  Delta vs PGTune : %+.1f%%  (Wilcoxon p=%.4f, t-test p=%.4f)' % (
        (np.mean(a)/np.mean(p)-1)*100, w_ap.pvalue, t_ap.pvalue))
    print('  Delta vs Default: %+.1f%%  (Wilcoxon p=%.4f, t-test p=%.4f)' % (
        (np.mean(a)/np.mean(d)-1)*100, w_ad.pvalue, t_ad.pvalue))

    result = {
        'agent':   {'values': a, 'mean': np.mean(a), 'std': np.std(a), 'ci95': list(ci_a)},
        'pgtune':  {'values': p, 'mean': np.mean(p), 'std': np.std(p)},
        'default': {'values': d, 'mean': np.mean(d), 'std': np.std(d)},
        'delta_vs_pgtune_pct':  (np.mean(a)/np.mean(p)-1)*100,
        'delta_vs_default_pct': (np.mean(a)/np.mean(d)-1)*100,
        'wilcoxon_ap': {'stat': w_ap.statistic, 'p': w_ap.pvalue},
        'wilcoxon_ad': {'stat': w_ad.statistic, 'p': w_ad.pvalue},
        'ttest_ap': {'stat': t_ap.statistic, 'p': t_ap.pvalue},
        'ttest_ad': {'stat': t_ad.statistic, 'p': t_ad.pvalue},
        'n': n, 'checkpoint_candidates': '$PEAK_STEPS',
        'order': '$ORDER',
        'protocol': 'rebuild + 20-run neutral warmup + ${N_SKIP} skip + ${N_MEASURE} measured + randomized order'
    }
    with open('${RESULTS_DIR}/summary_${TS}.json', 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print()
    print('  Guardado: ${RESULTS_DIR}/summary_${TS}.json')
else:
    print('  N insuficiente para análisis estadístico (mínimo 5)')
" 2>/dev/null || echo "  Error en análisis estadístico"

echo ""
echo "======================================================================"
echo "  COMPLETADO — resultados en: $RESULTS_DIR/"
echo "======================================================================"
