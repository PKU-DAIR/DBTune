#!/usr/bin/env python3
"""
DBTune 2.0 — Statistical Significance Analysis

Loads benchmark results from run_significance_test.sh and computes:
  - Mean ± Std TPS per method and tier
  - Mann-Whitney U test (independent samples, non-parametric) — suitable for N=10
  - Cohen's d effect size
  - % improvement over Default and PGTune baselines
  - LaTeX table ready for paper

Usage:
    python3 scripts/analyze_significance.py \
        --results_dir results/significance \
        --tiers "6gb 4gb 3gb 2gb"
"""

import argparse
import json
import os
import numpy as np
from scipy import stats

# Tier-specific baselines (from tier_profiles.py, measured N=5)
TIER_BASELINES = {
    '6gb': 272,
    '5gb': 320,
    '4gb': 231,
    '3gb': 167,
    '2gb': 109,
}

METHODS = ['default', 'pgtune', 'dbtune_g']
METHOD_LABELS = {
    'default':  'Default PG',
    'pgtune':   'PGTune',
    'dbtune_g': 'DBTune-G (ours)',
}


def run_metric_from_data(data):
    """
    Métrica por run: avg de steps 2-N (descarta step 1 = warmup).
    Usa all_tps si está disponible, si no avg_tps o peak_tps como fallback.
    """
    tps_list = data.get('all_tps', [])
    if len(tps_list) > 1:
        return float(np.mean(tps_list[1:]))  # skip step 1 (warmup)
    elif len(tps_list) == 1:
        return float(tps_list[0])
    # Fallback a métricas agregadas
    return float(data.get('avg_tps') or data.get('peak_tps') or data.get('tps') or 0)


def load_results(results_dir, tier, method, n_runs):
    """
    Carga TPS por run desde los JSONs generados por run_formal_eval.sh.
    Convención de nombres:
      - default/pgtune: baseline_{method}_rpi_{tier}_run{N}.json
      - dbtune_g:       inference_rpi_{tier}_run{N}.json
    Fallback legacy: {results_dir}/{tier}/{method}_run{N}.json
    """
    tps_list = []
    for run in range(1, n_runs + 1):
        # Formato run_formal_eval.sh
        if method == 'dbtune_g':
            fpath = os.path.join(results_dir, f"inference_rpi_{tier}_run{run}.json")
        else:
            fpath = os.path.join(results_dir, f"baseline_{method}_rpi_{tier}_run{run}.json")

        # Fallback formato legacy
        if not os.path.exists(fpath):
            fpath = os.path.join(results_dir, tier, f"{method}_run{run}.json")

        if not os.path.exists(fpath):
            continue

        with open(fpath) as f:
            data = json.load(f)
        tps = run_metric_from_data(data)
        if tps > 0:
            tps_list.append(tps)
    return tps_list


def cohens_d(a, b):
    """Cohen's d effect size between two samples."""
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def interpret_d(d):
    d = abs(d)
    if d < 0.2:  return 'negligible'
    if d < 0.5:  return 'small'
    if d < 0.8:  return 'medium'
    return 'large'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/formal_eval')
    parser.add_argument('--tiers', type=str, default='6gb 4gb 3gb 2gb')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    args = parser.parse_args()

    tiers = args.tiers.split()

    print('\n' + '='*80)
    print('DBTune 2.0 — Statistical Significance Results')
    print('='*80)

    # Store per-tier results for LaTeX table
    latex_rows = []

    for tier in tiers:
        print(f'\n📊 Tier: rpi_{tier}  (baseline={TIER_BASELINES.get(tier, "?")} TPS)')
        print('-'*60)

        data = {}
        for method in METHODS:
            tps_list = load_results(args.results_dir, tier, method, args.runs)
            if not tps_list:
                print(f'  ⚠️  No data for {method} @ {tier}')
                data[method] = None
            else:
                data[method] = np.array(tps_list)
                print(f'  {METHOD_LABELS[method]:<22}: '
                      f'mean={np.mean(tps_list):.1f}  '
                      f'std={np.std(tps_list, ddof=1):.1f}  '
                      f'median={np.median(tps_list):.1f}  '
                      f'N={len(tps_list)}')

        # Statistical tests: DBTune-G vs Default, DBTune-G vs PGTune
        if data['dbtune_g'] is not None:
            print()
            for baseline in ['default', 'pgtune']:
                if data[baseline] is None:
                    continue
                a = data['dbtune_g']
                b = data[baseline]

                # Mann-Whitney U test (independent samples — configs run in sequential blocks)
                stat, p = stats.mannwhitneyu(a, b, alternative='greater')
                test_name = 'Mann-Whitney'

                d = cohens_d(a, b)
                pct_improv = (np.mean(a) - np.mean(b)) / np.mean(b) * 100
                sig = '✅ p<0.05' if p < args.alpha else '❌ not sig'

                print(f'  DBTune-G vs {METHOD_LABELS[baseline]:<14}: '
                      f'+{pct_improv:.1f}%  '
                      f'{test_name} p={p:.4f} {sig}  '
                      f"Cohen's d={d:.2f} ({interpret_d(d)})")

        # Collect for LaTeX
        if all(data[m] is not None for m in METHODS):
            latex_rows.append((tier, data))

    # LaTeX table
    print('\n' + '='*80)
    print('LaTeX Table (copy-paste into paper)')
    print('='*80)
    print(r'\begin{table}[h]')
    print(r'\centering')
    print(r'\caption{TPS comparison: DBTune-G vs baselines (mean ± std, N=10 runs)}')
    print(r'\begin{tabular}{lrrrrr}')
    print(r'\hline')
    print(r'Tier & Default PG & PGTune & \textbf{DBTune-G} & vs Default & vs PGTune \\')
    print(r'\hline')
    for tier, data in latex_rows:
        d_mean  = np.mean(data['default'])
        d_std   = np.std(data['default'],  ddof=1)
        pg_mean = np.mean(data['pgtune'])
        pg_std  = np.std(data['pgtune'],   ddof=1)
        g_mean  = np.mean(data['dbtune_g'])
        g_std   = np.std(data['dbtune_g'], ddof=1)
        vs_d  = (g_mean - d_mean)  / d_mean  * 100
        vs_pg = (g_mean - pg_mean) / pg_mean * 100
        print(f'rpi-{tier} & '
              f'${d_mean:.0f} \\pm {d_std:.0f}$ & '
              f'${pg_mean:.0f} \\pm {pg_std:.0f}$ & '
              f'$\\mathbf{{{g_mean:.0f} \\pm {g_std:.0f}}}$ & '
              f'+{vs_d:.1f}\\% & '
              f'+{vs_pg:.1f}\\% \\\\')
    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{table}')

    # Save summary JSON
    summary = {}
    for tier, data in latex_rows:
        summary[tier] = {
            m: {'mean': float(np.mean(data[m])), 'std': float(np.std(data[m], ddof=1)),
                'n': len(data[m]), 'all': data[m].tolist()}
            for m in METHODS
        }
    out_path = os.path.join(args.results_dir, 'significance_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n📁 Summary saved to {out_path}')


if __name__ == '__main__':
    main()