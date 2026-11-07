#!/bin/bash
# =============================================================================
# run_formal_eval_full_rebuild.sh
#
# Como run_formal_eval_interleaved.sh pero con rebuild + 20-run warmup ANTES
# de CADA config individual. Elimina cualquier efecto de estado residual o
# calentamiento diferencial entre configuraciones.
#
# Ejecuta secuencialmente: 4GB v14 → 6GB v12
#
# Usage:
#   nohup ./run_formal_eval_full_rebuild.sh \
#       > logs/formal_eval_full_rebuild_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   disown
#
# Duración estimada: ~7-8h
#   (rebuild ~15min + warmup ~40min + 12 runs ~24min = ~79min/config × 6 configs)
# =============================================================================

set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
source venv/bin/activate
cd "$(dirname "$0")/../.."

N_SKIP=2
N_MEASURE=10

# Tiers a evaluar en orden
TIERS=(
    "4gb v14"
    "6gb v12"
    "3gb v12"
)

# =============================================================================
# run_config: rebuild + warmup + medición para una sola config
# =============================================================================
run_config() {
    local mode="$1"
    local label="$2"
    local outfile="$3"
    local config="$4"
    local tier="$5"
    local candidates="$6"
    local total=$((N_SKIP + N_MEASURE))

    echo ""
    echo "──────────────────────────────────────────────────────────────────────"
    echo "  REBUILD + WARMUP previo a: $label"
    echo "──────────────────────────────────────────────────────────────────────"

    ./scripts/rebuild_benchmark_db.sh

    echo ""
    echo "  Warm-up neutral: 20 runs con Default (steady state)..."
    python3 scripts/run_baseline.py \
        --config "$config" \
        --mode default \
        --tier "$tier" \
        --steps 20 \
        --skip-first 20 \
        --output /dev/null 2>/dev/null || true
    echo "  BD en steady state — midiendo $label ($total runs: $N_SKIP skip + $N_MEASURE medidos)..."
    echo ""

    if [ "$mode" = "agent" ]; then
        python3 scripts/run_inference.py \
            --config "$config" \
            --tier "$tier" \
            --checkpoint "$candidates" \
            --candidates 1 \
            --n-runs "$total" \
            --skip-first "$N_SKIP" \
            --output "$outfile" || echo "  ERROR en inferencia del agente"
    else
        python3 scripts/run_baseline.py \
            --config "$config" \
            --mode "$mode" \
            --tier "$tier" \
            --steps "$total" \
            --skip-first "$N_SKIP" \
            --output "$outfile" || echo "  ERROR en baseline $mode"
    fi

    if [ -f "$outfile" ]; then
        python3 -c "
import json
try:
    d = json.load(open('$outfile'))
    tps_list = d.get('measured_tps', d.get('effective_tps', []))
    if not tps_list and 'all_tps' in d:
        tps_list = d['all_tps'][$N_SKIP:]
    import numpy as np
    arr = [t for t in tps_list if t > 0]
    print('  >>> $label: %.1f ± %.1f TPS (N=%d)' % (np.mean(arr), np.std(arr), len(arr)))
    print('  runs: %s' % [round(t,1) for t in arr])
except Exception as e:
    print('  (no se pudo leer resultado: %s)' % e)
" 2>/dev/null || echo "  (no se pudo leer resultado)"
    fi
}

# =============================================================================
# Loop principal: un tier a la vez
# =============================================================================

for TIER_ENTRY in "${TIERS[@]}"; do
    TIER=$(echo "$TIER_ENTRY" | awk '{print $1}')
    SUFFIX=$(echo "$TIER_ENTRY" | awk '{print $2}')

    # Derivar TASK_ID, CONFIG y RESULTS_DIR (misma lógica que run_formal_eval_interleaved.sh)
    if [[ "${SUFFIX}" =~ ^v[0-9] ]]; then
        TASK_ID="dbtune_specialist_${TIER}_${SUFFIX}"
        CONFIG="scripts/config_specialist_${TIER}_${SUFFIX}.ini"
        RESULTS_DIR="results/formal_eval/${SUFFIX}_${TIER}_fullrebuild_n10"
    elif [ -n "$SUFFIX" ]; then
        TASK_ID="dbtune_specialist_${TIER}_v11_${SUFFIX}"
        CONFIG="scripts/config_specialist_${TIER}_v11_${SUFFIX}.ini"
        RESULTS_DIR="results/formal_eval/v11_${TIER}_${SUFFIX}_fullrebuild_n10"
    else
        TASK_ID="dbtune_specialist_${TIER}_v11"
        CONFIG="scripts/config_specialist_${TIER}_v11.ini"
        RESULTS_DIR="results/formal_eval/v11_${TIER}_fullrebuild_n10"
    fi
    CKPT_PREFIX="model_params/${TASK_ID}"
    JSONL="runs/${TASK_ID}/training_steps.jsonl"
    # Fallback: JSONL en raíz (evals pre-refactoring)
    [ ! -f "$JSONL" ] && JSONL="training_steps_${TASK_ID}.jsonl"
    TS=$(date +%Y%m%d_%H%M%S)
    mkdir -p "$RESULTS_DIR" logs

    # Copiar normalizer a raíz (run_inference.py lo busca ahí)
    for pkl in "specialist_${TIER}_${SUFFIX}_mean_var.pkl" \
               "specialist_${TIER}_${SUFFIX}_mean_var_rpi_${TIER}.pkl"; do
        src="runs/${TASK_ID}/${pkl}"
        [ -f "$src" ] && cp "$src" . && echo "  Normalizer copiado: ${pkl}" \
                      || true
    done

    echo ""
    echo "======================================================================"
    echo "  TIER: ${TIER} (${TASK_ID})"
    echo "  Protocolo: rebuild + 20-run warmup ANTES de cada config"
    echo "  N por config: $N_MEASURE | skip: $N_SKIP"
    echo "  Resultados: $RESULTS_DIR"
    echo "======================================================================"

    # ── Auto-detectar candidatos de checkpoint (zona peak, ventana 20 steps) ──
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
    CANDIDATES=$(echo "$PEAK_STEPS" | tr ' ' '\n' | \
        awk -v pre="${CKPT_PREFIX}" '{printf "%s_%s,", pre, $1}' | sed 's/,$//')
    echo "  Checkpoint candidatos: steps $PEAK_STEPS"

    # ── Orden aleatorio de configs ─────────────────────────────────────────
    ORDER=$(python3 -c "
import random
configs = ['agent', 'pgtune', 'default']
random.shuffle(configs)
print(' '.join(configs))
")
    echo "  Orden (aleatorio): $ORDER"
    echo ""

    # ── Evaluar cada config con su propio rebuild + warmup ─────────────────
    for mode in $ORDER; do
        case "$mode" in
            agent)
                run_config "agent" "DBTune-S" \
                    "${RESULTS_DIR}/agent_${TS}.json" \
                    "$CONFIG" "$TIER" "$CANDIDATES"
                ;;
            pgtune)
                run_config "pgtune" "PGTune" \
                    "${RESULTS_DIR}/pgtune_${TS}.json" \
                    "$CONFIG" "$TIER" "$CANDIDATES"
                ;;
            default)
                run_config "default" "Default" \
                    "${RESULTS_DIR}/default_${TS}.json" \
                    "$CONFIG" "$TIER" "$CANDIDATES"
                ;;
        esac
    done

    # ── Análisis estadístico del tier ─────────────────────────────────────
    echo ""
    echo "=== ANÁLISIS ESTADÍSTICO — ${TIER} ==="
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
        'tier': '$TIER', 'suffix': '$SUFFIX', 'task_id': '$TASK_ID',
        'agent':   {'values': a, 'mean': float(np.mean(a)), 'std': float(np.std(a)), 'ci95': list(ci_a)},
        'pgtune':  {'values': p, 'mean': float(np.mean(p)), 'std': float(np.std(p))},
        'default': {'values': d, 'mean': float(np.mean(d)), 'std': float(np.std(d))},
        'delta_vs_pgtune_pct':  float((np.mean(a)/np.mean(p)-1)*100),
        'delta_vs_default_pct': float((np.mean(a)/np.mean(d)-1)*100),
        'wilcoxon_ap': {'stat': float(w_ap.statistic), 'p': float(w_ap.pvalue)},
        'wilcoxon_ad': {'stat': float(w_ad.statistic), 'p': float(w_ad.pvalue)},
        'ttest_ap': {'stat': float(t_ap.statistic), 'p': float(t_ap.pvalue)},
        'ttest_ad': {'stat': float(t_ad.statistic), 'p': float(t_ad.pvalue)},
        'n': n, 'checkpoint_candidates': '$PEAK_STEPS', 'order': '$ORDER',
        'protocol': 'rebuild+warmup por config, N=$N_MEASURE, skip=$N_SKIP'
    }
    with open('${RESULTS_DIR}/summary_${TS}.json', 'w') as f:
        import json as j2; j2.dump(result, f, indent=2)
    print()
    print('  Guardado: ${RESULTS_DIR}/summary_${TS}.json')
else:
    print('  N insuficiente para análisis estadístico (mínimo 5)')
" 2>/dev/null || echo "  Error en análisis estadístico"

    echo ""
    echo "======================================================================"
    echo "  COMPLETADO: ${TIER} — resultados en $RESULTS_DIR/"
    echo "======================================================================"

done

echo ""
echo "======================================================================"
echo "  TODO COMPLETADO: 4GB v14 + 6GB v12 (rebuild+warmup por config)"
echo "======================================================================"
