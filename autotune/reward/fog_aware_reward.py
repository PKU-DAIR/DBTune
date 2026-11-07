"""
FogPG-Tuner: Fog-Aware Reward Function
Penalizes configurations that violate resource constraints in fog nodes
"""

import logging

logger = logging.getLogger(__name__)


class FogAwareReward:
    """
    Reward function for resource-constrained fog computing nodes
    
    Design principles:
    1. Avoid swapping at all costs (critical for edge devices)
    2. Stay within memory limits (prevent OOM kills)
    3. Maximize throughput (TPS) within constraints
    
    Example:
        >>> constraints = {'max_memory_mb': 3200, 'max_swap_mb': 0}
        >>> reward = FogAwareReward(constraints, baseline_tps=2000)
        >>> score = reward.calculate(tps=2400, memory_used_mb=2000, swap_used_mb=0)
        >>> print(f"Reward: {score}")
    """
    
    def __init__(self, constraints, baseline_tps=1000.0):
        """
        Initialize fog-aware reward function
        
        Args:
            constraints (dict): Resource limits for this node
                {
                    'max_memory_mb': 3200,    # Maximum memory allowed
                    'max_swap_mb': 0,         # Maximum swap allowed (usually 0)
                    'max_cpu_percent': 95     # Maximum CPU usage (optional)
                }
            baseline_tps (float): Reference TPS for normalization
        """
        self.constraints = constraints
        self.baseline_tps = baseline_tps
        
        # Penalty weights
        self.SWAP_PENALTY = -1000      # Swapping is unacceptable
        self.MEMORY_CRITICAL = -500    # >95% memory is dangerous
        self.MEMORY_HIGH = -100        # >85% memory is concerning
        
        logger.info(f"[FogAwareReward] Initialized with constraints: {constraints}")
        logger.info(f"[FogAwareReward] Baseline TPS: {baseline_tps}")
    
    def calculate(self, tps, memory_used_mb, swap_used_mb, cpu_percent=0.0, io_wait=0.0, penalty_alpha=1.0, dynamic_max_memory_mb=None):
        """
        Calculate fog-aware reward for a given configuration
        
        Args:
            tps (float): Transactions per second achieved
            memory_used_mb (float): Memory used in MB
            swap_used_mb (float): Swap used in MB
            cpu_percent (float): CPU usage percentage (0-100)
            io_wait (float): I/O Wait percentage (0-100)
            penalty_alpha (float): Scaling factor for penalties (0.0 to 1.0)
            dynamic_max_memory_mb (float): Context-aware memory limit (overrides static constraints)
        
        Returns:
            float: Reward value
        """
        
        # Base reward: Normalized TPS (0-100 scale for typical configs)
        base_reward = (tps / self.baseline_tps) * 100
        
        # PENALTY 1: Swapping (CRITICAL)
        swap_penalty = 0
        max_swap = self.constraints.get('max_swap_mb', 0)
        if swap_used_mb > max_swap:
            swap_penalty = self.SWAP_PENALTY
            logger.warning(
                f"[FogAwareReward] SWAP VIOLATION: {swap_used_mb:.1f}MB "
                f"(limit: {max_swap}MB) → penalty={swap_penalty}"
            )
        
        # PENALTY 2: High memory usage (Soft-ish)
        memory_penalty = 0
        if dynamic_max_memory_mb and dynamic_max_memory_mb > 0:
             max_memory = dynamic_max_memory_mb
        else:
             max_memory = self.constraints['max_memory_mb']
             
        memory_ratio = memory_used_mb / max_memory
        
        if memory_ratio > 0.95:
            memory_penalty = self.MEMORY_CRITICAL * penalty_alpha
            logger.warning(
                f"[FogAwareReward] MEMORY CRITICAL: {memory_ratio*100:.1f}% "
                f"({memory_used_mb:.1f}MB / {max_memory}MB) → penalty={memory_penalty:.1f}"
            )
        elif memory_ratio > 0.85:
            memory_penalty = self.MEMORY_HIGH * penalty_alpha
            logger.info(
                f"[FogAwareReward] MEMORY HIGH: {memory_ratio*100:.1f}% "
                f"({memory_used_mb:.1f}MB / {max_memory}MB) → penalty={memory_penalty:.1f}"
            )
        
        # PENALTY 3: Sustained high CPU (Soft)
        cpu_penalty = 0
        max_cpu = self.constraints.get('max_cpu_percent', 100)
        if cpu_percent > max_cpu:
            cpu_penalty = -50 * penalty_alpha
        
        # PENALTY 4: High I/O Wait (New for SSD/Edge)
        # Relaxed values for SSD: We accept high I/O if TPS is good.
        # PENALTY 4: High I/O Wait (Soft Penalty for SSD)
        # Linear scaling starting at 60% (Sweet spot is ~30-50%)
        # Formula: (io_wait - 60) * 2.5. Examples: 70%->25, 80%->50, 100%->100
        io_penalty = 0
        if io_wait > 50.0:
            scale_factor = 2.5
            raw_penalty = (io_wait - 50.0) * scale_factor
            io_penalty = -min(raw_penalty, 100.0) * penalty_alpha
            
            logger.warning(
                f"[FogAwareReward] I/O WAIT CRITICAL: {io_wait:.1f}% "
                f"(>50%) → penalty={io_penalty:.1f}"
            )
        elif io_wait > 30.0:
            # Just log for monitoring, no penalty in the "sweet spot"
            logger.info(
                f"[FogAwareReward] I/O WAIT ELEVATED: {io_wait:.1f}% (OK for SSD)"
            )
        
        # Calculate total reward
        total_reward = base_reward + swap_penalty + memory_penalty + cpu_penalty + io_penalty
        
        logger.debug(
            f"[FogAwareReward] TPS={tps:.1f} Mem={memory_used_mb:.1f}MB "
            f"Swap={swap_used_mb:.1f}MB CPU={cpu_percent:.1f}% I/O={io_wait:.1f}% "
            f"Alpha={penalty_alpha:.2f} → Reward={total_reward:.1f}"
        )
        
        return total_reward
    
    def is_valid_config(self, memory_used_mb, swap_used_mb):
        """
        Check if configuration violates hard constraints
        
        Args:
            memory_used_mb (float): Memory used in MB
            swap_used_mb (float): Swap used in MB
        
        Returns:
            bool: True if configuration is acceptable, False if must be rejected
        """
        max_memory = self.constraints['max_memory_mb']
        max_swap = self.constraints.get('max_swap_mb', 0)
        
        # Hard constraint 1: No swapping allowed in fog
        if swap_used_mb > max_swap:
            logger.error(
                f"[FogAwareReward] Config REJECTED: swap={swap_used_mb:.1f}MB > {max_swap}MB"
            )
            return False
        
        # Hard constraint 2: Memory must not exceed 98% (too dangerous)
        if memory_used_mb > max_memory * 0.98:
            logger.error(
                f"[FogAwareReward] Config REJECTED: memory={memory_used_mb:.1f}MB "
                f"> {max_memory * 0.98:.1f}MB (98% of {max_memory}MB)"
            )
            return False
        
        return True
    
    def get_reward_from_metrics(self, res_dict, step_count=0, max_steps=100):
        """
        Orchestrate reward calculation from raw metrics dictionary
        Handles metric extraction, alpha scaling, and logging
        """
        # Extract metrics
        tps = res_dict.get('tps', 0)
        memory_used_mb = res_dict.get('physical', 0)
        swap_used_mb = res_dict.get('swap', 0) # Assumes swap is in res_dict, default 0
        cpu_percent = res_dict.get('cpu', 0)
        io_wait = res_dict.get('io_wait', 0)
        total_mem = res_dict.get('total_mem', 0)
        
        # Calculate dynamic alpha (progress-based penalty scaling)
        # Linear growth from 0.1 (lenient) to 1.0 (strict)
        current_progress = min(1.0, step_count / max(1, max_steps))
        alpha = 0.1 + (0.9 * current_progress)
        
        # Calculate Reward
        reward_value = self.calculate(
            tps=tps,
            memory_used_mb=memory_used_mb,
            swap_used_mb=swap_used_mb,
            cpu_percent=cpu_percent,
            io_wait=io_wait,
            penalty_alpha=alpha,
            dynamic_max_memory_mb=total_mem
        )
        
        logger.info(f"[FogPG-Tuner] TPS={tps:.1f}, Mem={memory_used_mb:.1f}MB, Limit={total_mem:.1f}MB, "
                   f"CPU={cpu_percent:.1f}% → Reward={reward_value:.1f}")
        
        return reward_value
    
    def __repr__(self):
        return (
            f"FogAwareReward(max_memory={self.constraints['max_memory_mb']}MB, "
            f"baseline_tps={self.baseline_tps})"
        )
