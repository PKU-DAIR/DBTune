
import argparse
import sys
import logging
import os
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotune.utils.config import parse_args
from autotune.database.postgresqldb import PostgresqlDB

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s'
)
logger = logging.getLogger('BaselineRunner')


def get_pgtune_config(tier='8gb'):
    """
    Returns PGTune-inspired configuration for different RAM tiers (Raspberry Pi/Fog nodes).
    Uses pod-available memory (from tier_profiles) not chip total RAM.
    Clamped to the search space MAX values defined in postgres_fog_v3.json.

    Workload: TPC-C, 80 warehouses, 20 virtual users → 20 concurrent connections.
    Hardware constants (rpc=1.1, eio=32, checkpoint_timeout=10min) are injected
    automatically by cnpg_applier.py — do not include them here.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from autotune.constraints.tier_profiles import get_profile

    tier_key = f'rpi_{tier.lower()}'
    try:
        profile = get_profile(tier_key)
        pod_ram_mb = profile['max_memory_mb']
    except ValueError:
        pod_ram_mb = {'8gb': 7680, '6gb': 5600, '5gb': 4600,
                      '4gb': 3200, '3gb': 2300, '2gb': 1536}.get(tier.lower(), 3200)

    total_ram_kb = pod_ram_mb * 1024
    CONCURRENT_CONNECTIONS = 20

    shared_buffers_kb = int(total_ram_kb * 0.25)
    shared_buffers_pages = shared_buffers_kb // 8
    shared_buffers_pages = min(shared_buffers_pages, 196608)  # postgres_fog_v6.json max: 1536MB
    shared_buffers_pages = max(shared_buffers_pages, 65536)   # json min: 512MB

    effective_cache_kb = int(total_ram_kb * 0.75)
    effective_cache_pages = effective_cache_kb // 8
    effective_cache_pages = min(effective_cache_pages, 1048576)

    work_mem_kb = int(total_ram_kb / CONCURRENT_CONNECTIONS)
    work_mem_kb = min(work_mem_kb, 16384)
    work_mem_kb = max(work_mem_kb, 1024)

    maint_work_mem_kb = int(total_ram_kb * 0.05)
    maint_work_mem_kb = min(maint_work_mem_kb, 524288)  # postgres_fog_v6.json max: 512MB
    maint_work_mem_kb = max(maint_work_mem_kb, 16384)

    return {
        "shared_buffers": shared_buffers_pages,
        "effective_cache_size": effective_cache_pages,
        "work_mem": work_mem_kb,
        "maintenance_work_mem": maint_work_mem_kb,
        "checkpoint_completion_target": 0.9,
        "wal_buffers": 2048,  # 16MB
    }


def get_default_config():
    """
    Returns the SSD-aware PostgreSQL default configuration used as the 'Default' baseline.
    Hardcoded — NOT read from the knob JSON, because the JSON min/default values
    are tuned for the agent's action space (min=512MB), not for PostgreSQL defaults.
    Matches the old formal eval baselines (shared_buffers=128MB, cct=0.9).
    Hardware constants (rpc=1.1, eio=32, checkpoint_timeout=10min) injected by cnpg_applier.
    """
    return {
        "shared_buffers": 16384,           # 128MB — PostgreSQL default
        "effective_cache_size": 524288,    # 4GB — PostgreSQL default
        "work_mem": 4096,                  # 4MB — PostgreSQL default
        "maintenance_work_mem": 65536,     # 64MB — PostgreSQL default
        "checkpoint_completion_target": 0.9,
        "wal_buffers": 512,                # 4MB
    }


def run_one_benchmark(db, knobs, step_i, is_first_run, tier=None, hw_override=None, pure=False):
    """
    Apply knobs directly via cnpg_applier (bypassing normalize_knobs entirely)
    and run one benchmark. Returns TPS.

    On the first run: applies knobs (may restart) + warmup benchmark.
    On subsequent runs: applies knobs (diff=0, no restart) + benchmark.

    hw_override: dict applied AFTER hardware constants — overrides them selectively.
                 Use e.g. {"synchronous_commit":"on"} to test without sc=off.
    pure: if True, bypasses ALL injected constants (HARDWARE_CONSTANTS + tier_constants).
          Applies only the knobs dict as-is — true PostgreSQL default comparison.
    """
    from autotune.knob_management.cnpg_applier import CnpgKnobApplier

    # --pure: no tier → _tier_constants stays empty (no shared_buffers override)
    effective_tier = None if pure else tier
    applier = CnpgKnobApplier(
        namespace=db.k8s_pg_namespace,
        cluster_name=db.k8s_pg_cluster_name,
        tier=effective_tier
    )

    if pure:
        # Bypass HARDWARE_CONSTANTS (sc=off, fpw=off, rpc=1.1, eio=32, etc.)
        applier.HARDWARE_CONSTANTS = {}
        logger.info("  [PURE MODE] All hardware constants bypassed (rpc, eio, wal_comp, wal_buffers, max/min_wal_size) — applying knobs as-is")

    knobs_to_apply = dict(knobs)
    if hw_override:
        # Applied after HARDWARE_CONSTANTS inside apply_knobs — patch via post-update
        # We extend apply_knobs by temporarily appending hw_override to _tier_constants
        original_tier_constants = dict(applier._tier_constants)
        applier._tier_constants.update(hw_override)

    logger.info(f"  Applying knobs (absolute values, no normalization): {knobs_to_apply}")
    if hw_override:
        logger.info(f"  HW override (post-constants): {hw_override}")
    result = applier.apply_knobs(knobs_to_apply)  # injects HARDWARE_CONSTANTS automatically

    if hw_override:
        applier._tier_constants = original_tier_constants

    restart_happened = result.get('restart_required', False) if isinstance(result, dict) else False

    if restart_happened:
        logger.info("  🔥 Restart detected. Stabilizing (30s)...")
        time.sleep(30)
        # Warmup TPC-C REMOVED (v11): training also does not run warmup post-restart
        # (dbenv.py:578). Both baseline and training use only pg_prewarm + cold buffers.
        # Symmetric protocol — matches OtterTune/LlamaTune.

    tps = db.get_benchmark_result()
    return tps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Baseline Experiments')
    parser.add_argument('--config', type=str, default='scripts/config_edge_ddpg_k8s_v7.ini')
    parser.add_argument('--mode', type=str, choices=['default', 'pgtune', 'dbtune'], required=True)
    parser.add_argument('--tier', type=str, choices=['8gb', '6gb', '5gb', '4gb', '3gb', '2gb'], default='4gb')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--steps', type=int, default=6, help='Total runs (run1=warmup, runs 2-N measured)')
    parser.add_argument('--skip-first', type=int, default=1, help='Discard first N runs as warmup (default: 1)')
    parser.add_argument('--knobs-file', type=str, default=None)
    parser.add_argument('--hw-override', type=str, default=None,
                        help='JSON dict of knobs applied AFTER hardware constants (overrides them). '
                             'E.g. \'{"synchronous_commit":"on","full_page_writes":"on"}\'')
    parser.add_argument('--no-prewarm', action='store_true', default=False,
                        help='Skip pg_prewarm before each benchmark run (cold-cache measurement)')
    parser.add_argument('--pure', action='store_true', default=False,
                        help='Pure baseline: bypass ALL hardware constants and tier_constants. '
                             'Applies only the mode knobs as-is (true PostgreSQL default comparison).')
    args = parser.parse_args()

    args_db, args_tune = parse_args(args.config)

    logger.info(f"Initializing DB for Mode: {args.mode.upper()}, Tier: {args.tier}")
    db = PostgresqlDB(args_db)

    # Prepare knobs (absolute values — NOT passed through normalize_knobs)
    if args.mode == 'default':
        knobs = get_default_config()
    elif args.mode == 'pgtune':
        knobs = get_pgtune_config(args.tier)
    elif args.mode == 'dbtune':
        import json as _json
        if not args.knobs_file or not os.path.exists(args.knobs_file):
            logger.error(f"--mode dbtune requires --knobs-file. Got: {args.knobs_file}")
            sys.exit(1)
        with open(args.knobs_file) as f:
            knobs_data = _json.load(f)
        knobs = {k: int(v) if isinstance(v, float) and v == int(v) else v
                 for k, v in knobs_data['knobs'].items()}
        # v9 action space has no shared_buffers (POSTMASTER, not learned).
        # If absent, inject tier-specific value (25% RAM) so the DB runs with a known SB.
        if 'shared_buffers' not in knobs:
            from autotune.constraints.tier_profiles import get_profile
            try:
                tier_key = args.tier if args.tier.startswith('rpi_') else f'rpi_{args.tier}'
                _sb_mb = get_profile(tier_key)['shared_buffers_mb']
                knobs['shared_buffers'] = _sb_mb * 128  # MB → 8KB blocks
                logger.info(f"dbtune mode: injected shared_buffers={_sb_mb}MB from tier profile")
            except Exception as e:
                logger.warning(f"Could not inject shared_buffers: {e}")

    logger.info(f"Target knobs (before hardware constant injection): {knobs}")
    logger.info(f"Hardware constants will be injected by cnpg_applier: rpc=1.1, eio=32, checkpoint_timeout=10min")

    # --hw-override: parse JSON string into dict
    hw_override = None
    if args.hw_override:
        import json as _json2
        hw_override = _json2.loads(args.hw_override)
        logger.info(f"HW override active: {hw_override}")

    # --no-prewarm: set env var before DB object uses it
    if args.no_prewarm:
        import os as _os
        _os.environ['SKIP_PREWARM'] = '1'
        logger.info("pg_prewarm DISABLED for this run (--no-prewarm)")

    if args.pure:
        logger.info("PURE MODE: all hardware constants and tier_constants bypassed — true PostgreSQL default")

    import json as _json
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    all_tps = []
    best_tps = 0.0

    skip_n = args.skip_first
    for step_i in range(1, args.steps + 1):
        label = f'[WARMUP — discarded]' if step_i <= skip_n else f'[MEASURED run {step_i-skip_n}/{args.steps-skip_n}]'
        logger.info(f"--- Step {step_i}/{args.steps} {label} ---")

        try:
            tps = run_one_benchmark(db, knobs, step_i, is_first_run=(step_i == 1), tier=args.tier,
                                    hw_override=hw_override, pure=args.pure)
            logger.info(f"  TPS = {tps:.1f}")
            all_tps.append(tps)
            if tps > best_tps:
                best_tps = tps

            if args.output_dir:
                step_file = os.path.join(
                    args.output_dir,
                    f"baseline_{args.mode}_rpi_{args.tier}_run{step_i}.json"
                )
                with open(step_file, 'w') as f:
                    _json.dump({
                        'tps': tps,
                        'mode': args.mode,
                        'tier': args.tier,
                        'run': step_i,
                        'warmup': step_i == 1,
                        'knobs': knobs,
                    }, f, indent=2)

        except Exception as e:
            logger.error(f"  Step {step_i} failed: {e}")
            import traceback; traceback.print_exc()

    measured = all_tps[skip_n:] if len(all_tps) > skip_n else all_tps
    avg_tps = float(np.mean(measured)) if measured else 0.0
    std_tps = float(np.std(measured)) if measured else 0.0

    logger.info("=" * 50)
    logger.info(f"BASELINE RESULTS — {args.mode.upper()} / {args.tier}")
    logger.info("=" * 50)
    logger.info(f"N={len(measured)} measured runs (first {skip_n} discarded as warmup)")
    logger.info(f"Avg TPS : {avg_tps:.1f}  ±  {std_tps:.1f}")
    logger.info(f"Peak TPS: {best_tps:.1f}")
    logger.info(f"All TPS : {[round(t,1) for t in all_tps]}")
    logger.info("=" * 50)

    output_file = args.output or (
        os.path.join(args.output_dir, f"summary_{args.mode}_rpi_{args.tier}.json")
        if args.output_dir else f"baseline_{args.mode}_result.txt"
    )
    if output_file.endswith('.json'):
        with open(output_file, 'w') as f:
            _json.dump({
                'avg_tps': avg_tps, 'std_tps': std_tps,
                'peak_tps': best_tps,
                'all_tps': all_tps, 'measured_tps': measured,
                'n_measured': len(measured),
                'mode': args.mode, 'tier': args.tier, 'knobs': knobs,
            }, f, indent=2)
    else:
        with open(output_file, 'w') as f:
            f.write(f"Avg TPS (N={len(measured)}): {avg_tps:.1f}\n")
            f.write(f"Std TPS: {std_tps:.1f}\n")
            f.write(f"Peak TPS: {best_tps:.1f}\n")
            f.write(f"Knobs: {knobs}\n")
    logger.info(f"📁 Saved to {output_file}")
