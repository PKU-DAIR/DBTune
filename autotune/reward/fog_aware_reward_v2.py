"""
FogPG-Tuner V2: Improved Fog-Aware Reward Function
Designed for Deep Deterministic Policy Gradient (DDPG) stability.

CHANGELOG (V2 Improvements):
1. Continuous Memory Penalties: Replaced step-functions (if >85% logic) with a continuous 
   exponential/polynomial curve. This allows the DDPG Critic network to calculate 
   smooth gradients and learn boundaries much faster.
2. Latency Awareness: Added a soft penalty for P95 Latency spikes, punishing configs 
   that achieve high TPS but cause extreme lag for the end-user.
3. Continuous CPU Penalty: Replaced the flat -50 penalty with a linear scale based 
   on how much the CPU threshold is exceeded.
"""

import logging

logger = logging.getLogger(__name__)

class FogAwareRewardV2:
    """
    Reward function V2 for resource-constrained fog computing nodes.
    Math is optimized for continuous Reinforcement Learning action spaces.
    """
    
    def __init__(self, constraints, baseline_tps=1000.0):
        self.constraints = constraints
        self.baseline_tps = baseline_tps
        
        # Hard Penalties
        self.SWAP_PENALTY = -1000      # Swapping is entirely unacceptable in Fog
        self.MAX_MEM_PENALTY = -500    # Maximum mathematical penalty for OOM limits
        
        logger.info(f"[FogAwareRewardV2] Initialized with smooth gradients. Constraints: {constraints}")
    
    def calculate(self, tps, memory_used_mb, swap_used_mb, cpu_percent=0.0, io_wait=0.0, latency_p95=0.0, penalty_alpha=1.0, dynamic_max_memory_mb=None):
        
        # 1. Base reward: Normalized TPS (0-100 scale)
        base_reward = (tps / self.baseline_tps) * 100
        
        # 2. SWAP PENALTY (CRITICAL - hard limit)
        swap_penalty = 0
        max_swap = self.constraints.get('max_swap_mb', 0)
        if swap_used_mb > max_swap:
            swap_penalty = self.SWAP_PENALTY
            logger.warning(
                f"[FogAwareRewardV2] SWAP VIOLATION: {swap_used_mb:.1f}MB "
                f"(limit: {max_swap}MB) → penalty={swap_penalty}"
            )

        # 3. MEMORY PENALTY — FOG-AWARE (v11 reactivated, 2026-05-30)
        # Penalizes configurations that exceed the tier's memory limit. This is the core
        # fog-aware contribution: a config optimal for 8GB can OOM a 2GB/4GB node.
        # The agent learns to respect per-tier memory constraints via this penalty.
        # Formula: polynomial curve from 80% → 100% load (smooth gradients for Critic).
        # NOTE: shared_buffers + work_mem*N_conn are the main controllable contributors.
        # OS page cache is included in memory_used_mb but grows with TPS (not with knobs
        # directly). Alpha scaling ensures the penalty is strict only in later training steps.
        memory_penalty = 0
        max_memory = dynamic_max_memory_mb if dynamic_max_memory_mb else self.constraints['max_memory_mb']
        memory_ratio = memory_used_mb / max_memory
        if memory_ratio > 0.80:
            memory_penalty = -500.0 * ((memory_ratio - 0.80) / 0.20) ** 2 * penalty_alpha
            logger.info(
                f"[FogAwareRewardV2] MEMORY PENALTY: {memory_ratio*100:.1f}% "
                f"({memory_used_mb:.1f}MB / {max_memory:.1f}MB) → penalty={memory_penalty:.1f}"
            )

        # 4. CPU PENALTY (linear proportional above training threshold)
        # TPCC/20VUs keeps CPU at 96-99% regardless of knobs — the benchmark saturates CPU
        # by design. Using the tier's 95% operational threshold caused the penalty to fire
        # every step, adding constant noise rather than gradient signal.
        # Training threshold = 99%: only fires for genuinely pathological configs (e.g., a
        # knob combination that causes excessive context switching or sort spills).
        # In production fog scenarios, the tier threshold (95%) remains the correct limit
        # since the node runs co-located services competing for the same CPU.
        cpu_penalty = 0
        max_cpu = self.constraints.get('max_cpu_percent', 100)
        training_cpu_threshold = max(max_cpu, 99.0)
        if cpu_percent > training_cpu_threshold:
            cpu_penalty = -((cpu_percent - training_cpu_threshold) * 2.0) * penalty_alpha
            logger.info(
                f"[FogAwareRewardV2] CPU CRITICAL: {cpu_percent:.1f}% "
                f"(training limit: {training_cpu_threshold:.0f}%) → penalty={cpu_penalty:.1f}"
            )
        elif cpu_percent > max_cpu:
            logger.info(
                f"[FogAwareRewardV2] CPU HIGH (info only): {cpu_percent:.1f}% "
                f"(tier limit: {max_cpu}%, training threshold: {training_cpu_threshold:.0f}%)"
            )

        # 5. I/O WAIT PENALTY — DISABLED (v5)
        # IO wait is ~80% on this hardware regardless of config chosen.
        # It is not discriminative and dominated the reward in 4GB/6GB tiers,
        # making all configs appear equally bad. TPS already captures IO wait
        # implicitly: high IO wait → more disk reads → fewer transactions → lower TPS.
        io_penalty = 0
        if io_wait > 50.0:
            logger.info(
                f"[FogAwareRewardV2] IO WAIT (info only, no penalty): {io_wait:.1f}%"
            )

        # 6. LATENCY P95 PENALTY (dead code in k8s mode where lat=0, harmless)
        latency_penalty = 0
        if latency_p95 > 50.0:
            latency_penalty = -((latency_p95 - 50.0) * 1.5) * penalty_alpha
            latency_penalty = max(latency_penalty, -200.0)
            logger.info(
                f"[FogAwareRewardV2] LATENCY P95: {latency_p95:.1f}ms → penalty={latency_penalty:.1f}"
            )

        penalty_sum = swap_penalty + memory_penalty + cpu_penalty + io_penalty + latency_penalty
        total_reward = base_reward + penalty_sum

        logger.info(
            f"[FogAwareRewardV2] TPS={tps:.1f} base={base_reward:.1f} "
            f"swap={swap_penalty:.1f} mem={memory_penalty:.1f} cpu={cpu_penalty:.1f} "
            f"io={io_penalty:.1f} lat={latency_penalty:.1f} → total={total_reward:.1f}"
        )

        # Expose raw components for replay buffer re-labeling (PopArt-style).
        # Storing tps_raw + penalty_sum allows recalculating reward with any future
        # baseline without re-running benchmarks.
        self.last_tps_raw = tps
        self.last_penalty_sum = penalty_sum

        self.last_breakdown = {
            'base_tps':  round(base_reward, 3),
            'swap':      round(swap_penalty, 3),
            'memory':    round(memory_penalty, 3),
            'cpu':       round(cpu_penalty, 3),
            'io':        round(io_penalty, 3),
            'latency':   round(latency_penalty, 3),
            'total':     round(total_reward, 3),
        }

        return total_reward
    
    def is_valid_config(self, memory_used_mb, swap_used_mb):
        """ Hard rejection logic identical to V1 """
        max_memory = self.constraints['max_memory_mb']
        max_swap = self.constraints.get('max_swap_mb', 0)
        
        if swap_used_mb > max_swap: return False
        if memory_used_mb > max_memory * 0.98: return False
        return True
    
    def get_reward_from_metrics(self, res_dict, step_count=0, max_steps=100):
        tps = res_dict.get('tps', 0)
        memory_used_mb = res_dict.get('physical', 0)
        swap_used_mb = res_dict.get('swap', 0) 
        cpu_percent = res_dict.get('cpu', 0)
        io_wait = res_dict.get('io_wait', 0)
        total_mem = res_dict.get('total_mem', 0)
        # Assuming external metrics pass latency in ms
        latency_p95 = res_dict.get('lat', 0.0) 
        
        current_progress = min(1.0, step_count / max(1, max_steps))
        alpha = 0.1 + (0.9 * current_progress)
        
        # Use the stricter of the live container limit and the tier's safe ceiling.
        # total_mem is the K8s memory limit (e.g. 6144MB for 6Gi), while max_memory_mb
        # already reserves OS/sidecar headroom (e.g. 5400MB). Using container limit alone
        # would delay the penalty past the safe threshold and risk OOM without penalisation.
        profile_max = self.constraints['max_memory_mb']
        if total_mem > 0:
            dynamic_max = min(total_mem, profile_max)
        else:
            dynamic_max = profile_max
        reward_value = self.calculate(
            tps=tps, memory_used_mb=memory_used_mb, swap_used_mb=swap_used_mb,
            cpu_percent=cpu_percent, io_wait=io_wait, latency_p95=latency_p95,
            penalty_alpha=alpha, dynamic_max_memory_mb=dynamic_max
        )
        return reward_value
