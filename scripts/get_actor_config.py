#!/usr/bin/env python3
"""
Collects a real Prometheus state via one default-config benchmark step,
runs the trained Actor deterministically, and saves the suggested knobs
to a JSON file.

This is called ONCE per tier before running N=10 benchmark runs.
The resulting knobs file is then passed to run_baseline.py --mode dbtune.

Usage:
  python3 scripts/get_actor_config.py \
      --tier 4gb \
      --checkpoint model_params/dbtune_generalist_v2_650 \
      --output results/formal_eval/knobs_rpi_4gb.json
"""

import argparse
import sys
import logging
import os
import json
import pickle
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotune.utils.config import parse_args
from autotune.database.postgresqldb import PostgresqlDB
from autotune.dbenv import DBEnv
from autotune.knobs import initialize_knobs
from autotune.constraints.tier_profiles import get_profile

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s'
)
logger = logging.getLogger('GetActorConfig')


class EdgeActor(torch.nn.Module):
    def __init__(self, n_states, n_actions):
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
            torch.nn.Linear(32, n_actions)
        )
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return self.act(self.layers(x))


class MockConfig:
    def __init__(self, knobs):
        self.knobs = knobs
    def get_dictionary(self):
        return self.knobs


def action_to_knobs(action, knobs_detail):
    """Convert Actor sigmoid output [0,1]^n → physical knob values via linear interpolation."""
    keys = list(knobs_detail.keys())
    result = {}
    for i, key in enumerate(keys):
        detail = knobs_detail[key]
        lo, hi = detail['min'], detail['max']
        val = lo + action[i] * (hi - lo)
        if detail['type'] == 'integer':
            val = int(round(val))
        result[key] = float(val)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='scripts/config_edge_ddpg_k8s_v2.ini')
    parser.add_argument('--tier', required=True,
                        choices=['2gb', '3gb', '4gb', '5gb', '6gb', '8gb'])
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint prefix (without _actor.pth)')
    parser.add_argument('--output', required=True,
                        help='Output JSON file for knobs')
    args = parser.parse_args()

    actor_path = f"{args.checkpoint}_actor.pth"
    if not os.path.exists(actor_path):
        logger.error(f"Checkpoint not found: {actor_path}")
        sys.exit(1)

    args_db, args_tune = parse_args(args.config)
    fog_tier = f"rpi_{args.tier}"
    args_tune['use_fog_reward'] = 'True'
    args_tune['fog_tier'] = fog_tier

    profile = get_profile(fog_tier)
    logger.info(f"Tier: {fog_tier} | Max memory: {profile['max_memory_mb']} MB")

    # Load knob definitions
    knobs_detail = initialize_knobs(args_db['knob_config_file'], int(args_db['knob_num']))
    n_states = 35
    n_actions = len(knobs_detail)

    # Load Actor
    actor = EdgeActor(n_states, n_actions)
    state_dict = torch.load(actor_path, map_location='cpu', weights_only=True)
    actor.load_state_dict(state_dict)
    actor.eval()
    logger.info(f"Actor loaded from {actor_path}")

    # Load Welford normalizer
    mean_var_file = args_tune.get('mean_var_file', 'edge_ddpg_mean_var.pkl')
    if os.path.exists(mean_var_file):
        with open(mean_var_file, 'rb') as f:
            stats = pickle.load(f)
        norm_mean = np.array(stats['M'])
        norm_std = np.sqrt(np.array(stats['S']) / max(stats['n'] - 1, 1) + 1e-5)
        logger.info(f"Normalizer loaded (N={stats['n']} samples)")
    else:
        logger.warning(f"Normalizer {mean_var_file} not found — using identity")
        norm_mean = np.zeros(n_states)
        norm_std = np.ones(n_states)

    # One default-config benchmark step to collect real Prometheus state
    logger.info("Running one DEFAULT benchmark step to collect system state...")
    db = PostgresqlDB(args_db)
    env = DBEnv(args_db, args_tune, db)
    env.restart_interval = 1

    default_knobs = {k: v['default'] for k, v in knobs_detail.items()}
    try:
        objs, constraints, ext, resource, internal_metrics, info, trial_state = \
            env.step(MockConfig(default_knobs))
        warmup_tps = ext.get('tps', 0)
        logger.info(f"Warmup TPS (default config): {warmup_tps:.1f}")
        state = np.array(internal_metrics[:n_states], dtype=np.float64)
    except Exception as e:
        logger.warning(f"Warmup step failed ({e}), using normalizer mean as state")
        state = norm_mean.copy()

    # Normalize state
    norm_state = (state - norm_mean) / norm_std
    norm_state = np.nan_to_num(norm_state, nan=0.0, posinf=0.0, neginf=0.0)

    # Inference: Actor → knobs
    with torch.no_grad():
        action = actor(torch.FloatTensor(norm_state).unsqueeze(0)).squeeze(0).numpy()

    knobs = action_to_knobs(action, knobs_detail)

    logger.info(f"Actor raw output: {action.round(3).tolist()}")
    logger.info(f"Suggested knobs: {knobs}")

    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    output = {
        'tier': fog_tier,
        'checkpoint': actor_path,
        'warmup_tps_default': float(warmup_tps) if 'warmup_tps' in dir() else 0.0,
        'actor_output': action.tolist(),
        'knobs': knobs
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Knobs saved to {args.output}")
