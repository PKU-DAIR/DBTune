#!/bin/bash
# ==============================================================================
# eval_2gb_v12_fullrebuild.sh
#
# Eval formal 2GB v12 con rebuild + 20-run warmup ANTES de cada config.
# Metodología consistente con 4GB v14 y 6GB v12 (run_formal_eval_full_rebuild.sh).
#
# Lanzar DESPUÉS de que termine specialist_launch_2gb_v12.sh:
#   nohup ./eval_2gb_v12_fullrebuild.sh \
#       > logs/eval_2gb_v12_fullrebuild_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   disown
#
# Duración estimada: ~5h
#   rebuild (~40min) + warmup (~40min) + 12 runs (~24min) = ~104min/config × 3 configs
# ==============================================================================

set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
source venv/bin/activate
cd "$(dirname "$0")/../.."

TIER="2gb"
TASK_ID="dbtune_specialist_2gb_v12"
CONFIG="scripts/config_specialist_2gb_v11.ini"  # v12 usa el mismo config que v11
CKPT_PREFIX="model_params/${TASK_ID}"
JSONL="training_steps_${TASK_ID}.jsonl"
RESULTS_DIR="results/formal_eval/v12_2gb_fullrebuild_n10"
N_SKIP=2
N_MEASURE=10
TS=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RESULTS_DIR" logs

echo "======================================================================"
echo "  Eval formal 2GB v12 — rebuild + warmup por config"
echo "  Task:    $TASK_ID"
echo "  Config:  $CONFIG"
echo "  Results: $RESULTS_DIR"
echo "  N: $N_MEASURE medidos + $N_SKIP skip por config"
echo "======================================================================"

# ── Auto-detectar checkpoint (zona peak reward normalizado, ventana 20 steps) ──
PEAK_STEPS=$(python3 -c "
import json, glob, re
try:
    steps = [json.loads(l) for l in open('$JSONL')]
except:
    print('5 10 15'); exit(0)
ckpts = glob.glob('${CKPT_PREFIX}_*_actor.pth')
avail = sorted(set(int(re.search(r'_(\d+)_actor', c).group(1)) for c in ckpts))
# Usar reward normalizado (no TPS absoluto) para evitar sesgo por crecimiento de BD
scored = []
for s in avail:
    w = [x.get('reward', x.get('tps', 0)) for x in steps if s-20 <= x.get('global_t', x.get('step', 0)) <= s]
    if len(w) >= 3:
        scored.append((sum(w)/len(w), s))
scored.sort(reverse=True)
top3 = sorted([s for _,s in scored[:3]])
print(' '.join(map(str, top3)))
" 2>/dev/null || echo "5 10 15")
CANDIDATES=$(echo $PEAK_STEPS | tr ' ' '\n' | \
    awk -v pre="${CKPT_PREFIX}" '{printf "%s_%s,", pre, $1}' | sed 's/,$//')

echo "  Checkpoint candidatos: steps $PEAK_STEPS"

# ── Orden aleatorio de configs ─────────────────────────────────────────────
ORDER=$(python3 -c "
import random
configs = ['agent', 'pgtune', 'default']
random.shuffle(configs)
print(' '.join(configs))
")
echo "  Orden (aleatorio): $ORDER"
echo ""

# ── Función: rebuild + warmup + medición por config ───────────────────────
run_config() {
    local mode="$1"
    local label="$2"
    local outfile="$3"
    local total=$((N_SKIP + N_MEASURE))

    echo ""
    echo "──────────────────────────────────────────────────────────────────────"
    echo "  REBUILD + WARMUP previo a: $label"
    echo "──────────────────────────────────────────────────────────────────────"

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

# ── Evaluar cada config ────────────────────────────────────────────────────
for mode in $ORDER; do
    case "$mode" in
        agent)   run_config "agent"   "DBTune-S"  "${RESULTS_DIR}/agent_${TS}.json" ;;
        pgtune)  run_config "pgtune"  "PGTune"    "${RESULTS_DIR}/pgtune_${TS}.json" ;;
        default) run_config "default" "Default"   "${RESULTS_DIR}/default_${TS}.json" ;;
    esac
done

# ── Análisis estadístico ───────────────────────────────────────────────────
echo ""
echo "=== ANÁLISIS ESTADÍSTICO — 2GB v12 ==="
python3 -c "
import json, numpy as np, os
from scipy import stats

def load_tps(f, n_skip):
    if not os.path.exists(f): return []
    d = json.load(open(f))
    lst = d.get('measured_tps', d.get('effective_tps', []))
    if not lst and 'all_tps' in d:
        lst = d['all_tps'][n_skip:]
    return [t for t in lst if t > 0]

def trim(arr): return sorted(arr)[1:-1]

agent   = load_tps('${RESULTS_DIR}/agent_${TS}.json',   $N_SKIP)
pgtune  = load_tps('${RESULTS_DIR}/pgtune_${TS}.json',  $N_SKIP)
default = load_tps('${RESULTS_DIR}/default_${TS}.json', $N_SKIP)

print('  Config      N=10 mean±std         N=8 trim mean±std     elim (min/max)')
for name, raw in [('DBTune-S', agent), ('PGTune', pgtune), ('Default', default)]:
    if raw:
        t = trim(raw)
        print('  %-10s  %.1f ± %-7.1f      %.1f ± %-7.1f      %.1f / %.1f' % (
            name, np.mean(raw), np.std(raw), np.mean(t), np.std(t), min(raw), max(raw)))

n = min(len(agent), len(pgtune), len(default))
if n >= 5:
    a, p, d = agent[:n], pgtune[:n], default[:n]
    a8, p8, d8 = trim(a), trim(p), trim(d)

    w_ap  = stats.wilcoxon(a, p)
    w_ad  = stats.wilcoxon(a, d)
    w_ap8 = stats.wilcoxon(a8[:8], p8[:8])
    w_ad8 = stats.wilcoxon(a8[:8], d8[:8])
    ci_a  = stats.t.interval(0.95, n-1, np.mean(a), stats.sem(a))

    print()
    print('  DBTune-S CI95: [%.1f, %.1f]' % ci_a)
    print()
    print('  vs PGTune  N=10:      %+.1f%%  Wilcoxon p=%.4f  %s' % (
        (np.mean(a)/np.mean(p)-1)*100, w_ap.pvalue,
        'OK' if w_ap.pvalue < 0.05 else 'ns'))
    print('  vs Default N=10:      %+.1f%%  Wilcoxon p=%.4f  %s' % (
        (np.mean(a)/np.mean(d)-1)*100, w_ad.pvalue,
        'OK' if w_ad.pvalue < 0.05 else 'ns'))
    print()
    print('  vs PGTune  N=8 trim:  %+.1f%%  Wilcoxon p=%.4f  %s' % (
        (np.mean(a8)/np.mean(p8)-1)*100, w_ap8.pvalue,
        'OK' if w_ap8.pvalue < 0.05 else 'ns'))
    print('  vs Default N=8 trim:  %+.1f%%  Wilcoxon p=%.4f  %s' % (
        (np.mean(a8)/np.mean(d8)-1)*100, w_ad8.pvalue,
        'OK' if w_ad8.pvalue < 0.05 else 'ns'))

    result = {
        'tier': '2gb', 'task_id': '$TASK_ID',
        'checkpoint_candidates': '$PEAK_STEPS',
        'agent':   {'values': a,  'mean': float(np.mean(a)),  'std': float(np.std(a)),  'ci95': list(ci_a)},
        'pgtune':  {'values': p,  'mean': float(np.mean(p)),  'std': float(np.std(p))},
        'default': {'values': d,  'mean': float(np.mean(d)),  'std': float(np.std(d))},
        'delta_vs_pgtune_pct':  float((np.mean(a)/np.mean(p)-1)*100),
        'delta_vs_default_pct': float((np.mean(a)/np.mean(d)-1)*100),
        'wilcoxon_ap': {'p': float(w_ap.pvalue)},
        'wilcoxon_ad': {'p': float(w_ad.pvalue)},
        'n': n,
        'protocol': 'rebuild+warmup por config, N=${N_MEASURE}, skip=${N_SKIP}, checkpoint auto-detect por reward'
    }
    with open('${RESULTS_DIR}/summary_${TS}.json', 'w') as f:
        import json as j2; j2.dump(result, f, indent=2)
    print()
    print('  Guardado: ${RESULTS_DIR}/summary_${TS}.json')
else:
    print('  N insuficiente para análisis (mínimo 5)')
" 2>/dev/null || echo "  Error en análisis estadístico"

echo ""
echo "======================================================================"
echo "  COMPLETADO — resultados en: $RESULTS_DIR/"
echo "======================================================================"
