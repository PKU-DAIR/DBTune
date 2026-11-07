#!/usr/bin/env python3
"""
FogPG-Tuner: Inference Script v2
Loads a trained Edge-DDPG Actor checkpoint and runs formal evaluation.

Evaluation protocol (mirrors run_baseline.py):
  1. [Checkpoint selection] If multiple checkpoints given, run quick benchmark per
     checkpoint (1 run) and select the one with highest TPS.
  2. [Candidate selection] Run actor K times with tiny noise → K candidate configs.
     Benchmark each once. Select best.
  3. [State warm-up] Apply selected config once with the real system. Collect the
     warm state. Re-infer deterministically so actor sees its own steady-state.
  4. [Formal runs] Apply final config N times. Discard first run (cold cache).
     Report avg ± std of runs 2..N — identical to baseline protocol.

Usage:
  # Single checkpoint, deterministic (K=1), N=9 formal runs
  python3 scripts/run_inference.py \\
      --tier 4gb \\
      --checkpoint model_params/dbtune_generalist_v5_200

  # Checkpoint selection from training steps 100,150,200 + best-of-5 candidates
  python3 scripts/run_inference.py \\
      --tier 4gb \\
      --checkpoint model_params/dbtune_generalist_v5_100,model_params/dbtune_generalist_v5_150,model_params/dbtune_generalist_v5_200 \\
      --candidates 5 \\
      --output results/formal_eval/v5/inference_4gb.json
"""

import argparse
import sys
import logging
import os
import time
import json
import pickle
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotune.utils.config import parse_args
from autotune.database.postgresqldb import PostgresqlDB
from autotune.dbenv import DBEnv
from autotune.knobs import initialize_knobs
from autotune.constraints.tier_profiles import get_profile

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s'
)
logger = logging.getLogger('InferenceRunner')

N_STATES  = 31   # State vector dims (7 buffer + 9 db + 8 WAL + 4 system + 3 context)
N_ACTIONS = None  # Auto-detected from checkpoint (supports 5-knob v11 and 6-knob v8)


# ── Actor architecture (must match training checkpoint) ───────────────────────
class EdgeActor(torch.nn.Module):
    """Edge-DDPG Actor: 31 → 128 → 64 → 32 → N (Sigmoid). N auto-detected from checkpoint."""
    def __init__(self, n_states=N_STATES, n_actions=5):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_states, 128),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.3),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.3),
            torch.nn.LayerNorm(32),
            torch.nn.Linear(32, n_actions),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.layers(x))


class MockConfig:
    """Satisfies DBEnv.step() interface."""
    def __init__(self, knobs):
        self.knobs = knobs
    def get_dictionary(self):
        return self.knobs


# ── Helper functions ──────────────────────────────────────────────────────────

def load_actor(checkpoint_prefix: str) -> EdgeActor:
    """Load Actor weights from checkpoint. Auto-detects n_actions from state dict."""
    path = f"{checkpoint_prefix}_actor.pth"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Actor checkpoint not found: {path}")
    sd = torch.load(path, map_location='cpu', weights_only=True)
    # Infer n_actions from last Linear layer output (key ends with 'weight', largest index)
    linear_keys = [k for k in sd if k.endswith('.weight')]
    n_actions = sd[linear_keys[-1]].shape[0]
    actor = EdgeActor(n_states=N_STATES, n_actions=n_actions)
    actor.load_state_dict(sd)
    actor.eval()
    logger.info(f"  Loaded actor: {path} (n_actions={n_actions})")
    return actor


def load_normalizer(mean_var_file: str):
    """Load EMA normalizer stats. Returns (mean, std) arrays of shape (N_STATES,)."""
    if not os.path.exists(mean_var_file):
        logger.warning(f"  Normalizer not found ({mean_var_file}) — using identity")
        return np.zeros(N_STATES), np.ones(N_STATES)
    with open(mean_var_file, 'rb') as f:
        stats = pickle.load(f)
    mean = np.array(stats['M'], dtype=np.float64)
    # RunningStat uses EMA: _S stores variance directly (not Welford sum-of-squares).
    # Do NOT divide by (n-1) — that would produce std ~10x too small.
    var  = np.array(stats['S'], dtype=np.float64)
    std  = np.sqrt(var + 1e-5)
    # Adapt if stored with wrong dims (safety guard)
    if len(mean) != N_STATES:
        logger.warning(f"  Normalizer has {len(mean)} dims, expected {N_STATES} — truncating/padding")
        mean = np.pad(mean[:N_STATES], (0, max(0, N_STATES - len(mean))))
        std  = np.pad(std[:N_STATES],  (0, max(0, N_STATES - len(std))), constant_values=1.0)
    logger.info(f"  Normalizer loaded from {mean_var_file} (N={stats['n']} samples)")
    return mean, std


def action_to_knobs(action: np.ndarray, knobs_detail: dict) -> dict:
    """
    Denormalize Actor output [0,1]^8 → actual knob values.
    Uses sorted key order (alphabetical), matching ConfigSpace default ordering.
    """
    keys = sorted(knobs_detail.keys())
    result = {}
    for i, key in enumerate(keys):
        detail = knobs_detail[key]
        lo, hi = detail['min'], detail['max']
        val = lo + float(action[i]) * (hi - lo)
        if detail['type'] == 'integer':
            val = int(round(val))
        result[key] = val
    return result


def normalize_state(state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    norm = (state - mean) / std
    return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)


def infer_knobs(actor: EdgeActor, state: np.ndarray,
                mean: np.ndarray, std: np.ndarray,
                knobs_detail: dict, noise: float = 0.0) -> tuple:
    """Run actor inference. Returns (knobs_dict, raw_action_array)."""
    norm_state = normalize_state(state, mean, std)
    with torch.no_grad():
        t = torch.FloatTensor(norm_state).unsqueeze(0)
        action = actor(t).squeeze(0).numpy()
    if noise > 0.0:
        action = np.clip(action + np.random.normal(0, noise, action.shape), 0.0, 1.0)
    knobs = action_to_knobs(action, knobs_detail)
    return knobs, action


def collect_state(env: DBEnv, knobs: dict) -> tuple:
    """
    Run one benchmark step with given knobs. Returns (tps, state_vector).
    Used for warm-up and candidate evaluation.
    """
    objs, _, ext, _, internal, _, trial_state = env.step(MockConfig(knobs))
    tps = ext.get('tps', 0.0) if isinstance(ext, dict) else 0.0
    if trial_state != 0:
        tps = 0.0
    state = np.array(internal[:N_STATES], dtype=np.float64) \
            if internal is not None and len(internal) >= N_STATES \
            else np.zeros(N_STATES)
    return tps, state


def run_formal_benchmark(env: DBEnv, knobs: dict, n_runs: int, skip_first: int) -> dict:
    """
    Run N formal benchmark steps with fixed knobs.
    Discards first `skip_first` runs (cold-cache warm-up).
    Returns stats dict.
    """
    all_tps = []
    for i in range(1, n_runs + 1):
        logger.info(f"  Formal run {i}/{n_runs}...")
        try:
            _, _, ext, _, _, _, trial_state = env.step(MockConfig(knobs))
            tps = ext.get('tps', 0.0) if isinstance(ext, dict) else 0.0
            if trial_state != 0:
                logger.warning(f"  Run {i}: FAILED (trial_state={trial_state}), TPS=0")
                tps = 0.0
            else:
                logger.info(f"  Run {i}: TPS={tps:.1f}")
            all_tps.append(tps)
        except Exception as e:
            logger.error(f"  Run {i} exception: {e}")
            all_tps.append(0.0)

    effective = all_tps[skip_first:]
    effective_nonzero = [t for t in effective if t > 0]
    return {
        'all_tps':     all_tps,
        'effective_tps': effective,
        'avg_tps':     float(np.mean(effective_nonzero)) if effective_nonzero else 0.0,
        'std_tps':     float(np.std(effective_nonzero))  if len(effective_nonzero) > 1 else 0.0,
        'max_tps':     float(max(effective_nonzero)) if effective_nonzero else 0.0,
        'n_valid':     len(effective_nonzero),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FogPG-Tuner: Formal Inference Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--config', default='scripts/config_edge_ddpg_k8s_v7.ini',
                        help='Path to config .ini (default: config_edge_ddpg_k8s_v2.ini)')
    parser.add_argument('--tier', required=True,
                        choices=['8gb', '6gb', '5gb', '4gb', '3gb', '2gb'],
                        help='Hardware tier to evaluate on')
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint prefix(es), comma-separated if multiple. '
                             'E.g.: model_params/v5_200  or  model_params/v5_100,model_params/v5_150,model_params/v5_200')
    parser.add_argument('--candidates', type=int, default=1,
                        help='K: number of actor samples for best-of-K config selection. '
                             'K=1 = pure deterministic (default). K>1 uses tiny noise.')
    parser.add_argument('--candidate-noise', type=float, default=0.05,
                        help='Exploration noise for candidate sampling (default: 0.05). '
                             'Ignored when --candidates=1.')
    parser.add_argument('--n-runs', type=int, default=9,
                        help='Number of formal benchmark runs (default: 9, matching baseline protocol)')
    parser.add_argument('--skip-first', type=int, default=1,
                        help='Discard first N runs as cold-cache warm-up (default: 1)')
    parser.add_argument('--output', default=None,
                        help='Output JSON file path. Auto-generated if not specified.')
    parser.add_argument('--mean-var-file', default=None,
                        help='Welford normalizer .pkl file. Defaults to value in config.')
    args = parser.parse_args()

    checkpoints = [c.strip() for c in args.checkpoint.split(',') if c.strip()]
    fog_tier = f"rpi_{args.tier}"
    noise = args.candidate_noise if args.candidates > 1 else 0.0

    logger.info("=" * 65)
    logger.info(f"FogPG-Tuner Inference — tier={fog_tier}")
    logger.info(f"  Checkpoints : {checkpoints}")
    logger.info(f"  Candidates  : {args.candidates} (noise={noise:.3f})")
    logger.info(f"  Formal runs : {args.n_runs} (skip first {args.skip_first})")
    logger.info("=" * 65)

    # Validate tier
    try:
        profile = get_profile(fog_tier)
        logger.info(f"  Tier profile: max_memory={profile['max_memory_mb']} MB, "
                    f"max_cpu={profile['max_cpu_percent']}%")
    except ValueError as e:
        logger.error(str(e)); sys.exit(1)

    # Load config and init environment
    args_db, args_tune = parse_args(args.config)
    args_tune['use_fog_reward'] = 'True'
    args_tune['fog_tier'] = fog_tier

    db  = PostgresqlDB(args_db)
    env = DBEnv(args_db, args_tune, db)
    # Disable anti-restart heuristic so the actor's knobs are always applied
    env.restart_interval = 1

    # Load knob definitions
    knobs_detail = initialize_knobs(args_db['knob_config_file'], int(args_db['knob_num']))
    logger.info(f"  Knob space   : {len(knobs_detail)} knobs from {args_db['knob_config_file']}")

    # Load Welford normalizer
    mvf = args.mean_var_file or args_tune.get('mean_var_file', 'edge_ddpg_mean_var.pkl')
    norm_mean, norm_std = load_normalizer(mvf)

    # ── Stage 1: Checkpoint selection ─────────────────────────────────────────
    if len(checkpoints) == 1:
        best_checkpoint = checkpoints[0]
        logger.info(f"\n[Stage 1] Single checkpoint: {best_checkpoint}")
    else:
        logger.info(f"\n[Stage 1] Selecting best checkpoint from {len(checkpoints)}...")
        default_knobs = {k: v['default'] for k, v in knobs_detail.items()}
        # Collect a baseline state first (one run with default config)
        logger.info("  Warm-up with default config to get initial state...")
        warmup_tps, warmup_state = collect_state(env, default_knobs)
        logger.info(f"  Warm-up TPS (default): {warmup_tps:.1f}")

        best_checkpoint, best_ckpt_tps = checkpoints[0], -1.0
        ckpt_results = []
        for ckpt in checkpoints:
            try:
                actor = load_actor(ckpt)
            except FileNotFoundError as e:
                logger.warning(f"  Skipping {ckpt}: {e}"); continue

            knobs, _ = infer_knobs(actor, warmup_state, norm_mean, norm_std, knobs_detail, noise=0.0)
            tps, _ = collect_state(env, knobs)
            logger.info(f"  Checkpoint {ckpt}: TPS={tps:.1f}")
            ckpt_results.append({'checkpoint': ckpt, 'tps': tps})
            if tps > best_ckpt_tps:
                best_ckpt_tps, best_checkpoint = tps, ckpt

        logger.info(f"  → Selected: {best_checkpoint} (TPS={best_ckpt_tps:.1f})")

    # Load selected actor
    actor = load_actor(best_checkpoint)

    # ── Stage 2: Initial state collection (warm-up with default config) ───────
    logger.info(f"\n[Stage 2] Collecting initial system state (default config warm-up)...")
    default_knobs = {k: v['default'] for k, v in knobs_detail.items()}
    warmup_tps, current_state = collect_state(env, default_knobs)
    logger.info(f"  Warm-up TPS (default): {warmup_tps:.1f}")

    # ── Stage 3: Candidate config selection ───────────────────────────────────
    logger.info(f"\n[Stage 3] Generating {args.candidates} candidate config(s) "
                f"(noise={noise:.3f})...")

    if args.candidates == 1:
        best_knobs, best_action = infer_knobs(
            actor, current_state, norm_mean, norm_std, knobs_detail, noise=0.0)
        logger.info(f"  Deterministic inference → {best_knobs}")
    else:
        best_knobs, best_action, best_cand_tps = None, None, -1.0
        for k in range(args.candidates):
            knobs, action = infer_knobs(
                actor, current_state, norm_mean, norm_std, knobs_detail, noise=noise)
            tps, _ = collect_state(env, knobs)
            logger.info(f"  Candidate {k+1}/{args.candidates}: TPS={tps:.1f} | "
                        f"sb={knobs.get('shared_buffers',0)*8//1024}MB "
                        f"ecs={knobs.get('effective_cache_size',0)*8//1024}MB")
            if tps > best_cand_tps:
                best_cand_tps, best_knobs, best_action = tps, knobs, action
        logger.info(f"  → Best candidate TPS={best_cand_tps:.1f}")

    # ── Stage 4: State warm-up with selected config → re-infer ───────────────
    logger.info(f"\n[Stage 4] Applying selected config to collect warm steady-state...")
    _, warm_state = collect_state(env, best_knobs)
    final_knobs, final_action = infer_knobs(
        actor, warm_state, norm_mean, norm_std, knobs_detail, noise=0.0)
    logger.info(f"  Re-inferred (warm state) → sb={final_knobs.get('shared_buffers',0)*8//1024}MB "
                f"ecs={final_knobs.get('effective_cache_size',0)*8//1024}MB "
                f"wm={final_knobs.get('work_mem',0)/1024:.1f}MB")
    logger.info(f"  Final knobs: {final_knobs}")

    # ── Stage 5: Formal N runs ────────────────────────────────────────────────
    logger.info(f"\n[Stage 5] Formal evaluation: {args.n_runs} runs "
                f"(discarding first {args.skip_first})...")
    stats = run_formal_benchmark(env, final_knobs, args.n_runs, args.skip_first)

    # ── Results ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info(f"INFERENCE RESULTS — {fog_tier}")
    logger.info("=" * 65)
    logger.info(f"  Checkpoint  : {best_checkpoint}")
    logger.info(f"  Final knobs : {final_knobs}")
    logger.info(f"  All TPS     : {[f'{t:.1f}' for t in stats['all_tps']]}")
    logger.info(f"  Effective   : runs {args.skip_first+1}..{args.n_runs} "
                f"(N={stats['n_valid']} valid)")
    logger.info(f"  Avg TPS     : {stats['avg_tps']:.1f} ± {stats['std_tps']:.1f}")
    logger.info(f"  Max TPS     : {stats['max_tps']:.1f}")
    logger.info("=" * 65)

    # ── Save JSON output ──────────────────────────────────────────────────────
    output_file = args.output or f"inference_{fog_tier}_result.json"
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    result = {
        'tier':             fog_tier,
        'checkpoint':       best_checkpoint,
        'final_knobs':      final_knobs,
        'final_action':     final_action.tolist() if final_action is not None else [],
        'protocol': {
            'n_runs':        args.n_runs,
            'skip_first':    args.skip_first,
            'candidates':    args.candidates,
            'candidate_noise': noise,
            'n_states':      N_STATES,
        },
        **stats,
    }
    if len(checkpoints) > 1:
        result['checkpoint_selection'] = ckpt_results

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {output_file}")
