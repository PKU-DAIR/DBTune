"""
FogPG-Tuner: Tier Profiles
Defines resource constraints and knob spaces for different hardware tiers
"""
import os

# Tier profiles with hardware constraints
TIER_PROFILES = {
    'rpi_4gb': {
        # Hardware limits
        'max_memory_mb': 3200,           # Leave 800MB for OS + other services
        'max_swap_mb': 0,                # Swapping is unacceptable in fog
        'max_cpu_percent': 95,
        'baseline_tps': 257.8,           # Default PostgreSQL (SSD-aware) @ 4GB (N=5 baselines_v7). reward=100 = match Default, >100 = beat Default.
        
        # PostgreSQL-specific limits
        'shared_buffers_mb': 768,        # ~20% of available RAM (conservative)
        'max_work_mem_mb': 8,            # 8MB per operation
        'max_maintenance_work_mem_mb': 256,  # For VACUUM, CREATE INDEX
        'max_connections': 50,
        
        # Knobs to optimize (memory-critical only)
        'knobs': [
            'shared_buffers',
            'work_mem',
            'effective_cache_size',
            'checkpoint_completion_target',
            'random_page_cost',
        ],
        
        # Description
        'description': 'Raspberry Pi 4GB - Severely resource constrained',
        'storage': 'SD Card / USB',
    },
    
    'rpi_8gb': {
        # Hardware limits
        'max_memory_mb': 7680,           # 7.5GB (Maximized for Pi 8GB)
        'max_swap_mb': 0,                # Still avoid swapping
        'max_cpu_percent': 95,
        'baseline_tps': 320,             # Estimated (no direct measurement; extrapolated from 6GB=272)
        
        # PostgreSQL-specific limits
        'shared_buffers_mb': 1920,       # ~25% of 7.5GB
        'max_work_mem_mb': 24,           # Increased
        'max_maintenance_work_mem_mb': 768,
        'max_connections': 100,
        
        # Knobs to optimize (memory + I/O)
        'knobs': [
            'shared_buffers',
            'work_mem',
            'maintenance_work_mem',
            'effective_cache_size',
            'checkpoint_completion_target',
            'checkpoint_timeout',
            'wal_buffers',
            'random_page_cost',
        ],
        
        # Description
        'description': 'Raspberry Pi 8GB - Moderately resource constrained',
        'storage': 'SSD via USB',
    },

    'rpi_6gb': {
        # Hardware limits (Interpolated 4GB <-> 8GB)
        'max_memory_mb': 5400,           # K8s limit=6Gi=6144MB, leaving 11.5% headroom for OS/sidecar
        'max_swap_mb': 0,
        'max_cpu_percent': 95,
        'baseline_tps': 134.5,           # Default V2 (sc=on, fpw=on, prewarm, HW constants) @ 6GB (N=5, 2026-05-30). reward=100=match Default, >100=beat Default.
        
        # PostgreSQL-specific limits
        'shared_buffers_mb': 1536,       # ~26%
        'max_work_mem_mb': 16,           # 16MB
        'max_maintenance_work_mem_mb': 512,
        'max_connections': 80,
        
        # Knobs to optimize
        'knobs': [
            'shared_buffers',
            'work_mem',
            'effective_cache_size',
            'checkpoint_completion_target',
            'random_page_cost',
            'wal_buffers',
        ],
        
        # Description
        'description': 'Raspberry Pi 6GB - High constraint',
        'storage': 'SSD',
    },

    'rpi_5gb': {
        # Hardware limits (Zero-shot validation: unseen during training)
        'max_memory_mb': 4600,           # ~5GB (leave ~400MB for OS)
        'max_swap_mb': 0,
        'max_cpu_percent': 95,
        'baseline_tps': 120.0,           # Estimated (sc=on): proportional scale from 6GB=134.5. Override via FOGPG_BASELINE_TPS env var.
        
        # PostgreSQL-specific limits (interpolated 4GB <-> 6GB)
        'shared_buffers_mb': 1152,       # ~25%
        'max_work_mem_mb': 12,           # 12MB
        'max_maintenance_work_mem_mb': 384,
        'max_connections': 60,
        
        # Knobs to optimize
        'knobs': [
            'shared_buffers',
            'work_mem',
            'effective_cache_size',
            'checkpoint_completion_target',
            'random_page_cost',
            'wal_buffers',
        ],
        
        # Description
        'description': 'Raspberry Pi 5GB - Zero-shot validation (unseen during training)',
        'storage': 'SSD',
    },

    'rpi_3gb': {
        # Hardware limits (Interpolated 2GB <-> 4GB)
        'max_memory_mb': 2300,           # leaving ~700MB for OS
        'max_swap_mb': 0,
        'max_cpu_percent': 95,
        'baseline_tps': 80.0,            # Estimated (sc=on). Override via FOGPG_BASELINE_TPS env var at training start.
        
        # PostgreSQL-specific limits
        'shared_buffers_mb': 512,        # ~22%
        'max_work_mem_mb': 6,            # 6MB
        'max_maintenance_work_mem_mb': 128,
        'max_connections': 40,
        
        # Knobs to optimize
        'knobs': [
            'shared_buffers',
            'work_mem',
            'effective_cache_size',
            'checkpoint_completion_target',
            'random_page_cost',
        ],
        
        # Description
        'description': 'Raspberry Pi 3GB - Medium-Low constraint',
        'storage': 'SD Card',
    },

    'rpi_2gb': {
        # Hardware limits
        'max_memory_mb': 1536,           # Leave ~500MB for OS/K8s
        'max_swap_mb': 0,                # Critical strictness
        'max_cpu_percent': 95,
        'baseline_tps': 55.0,            # Estimated (sc=on). Override via FOGPG_BASELINE_TPS env var at training start.
        
        # PostgreSQL-specific limits
        'shared_buffers_mb': 160,        # Capped: 256MB causaba OOM bajo TPC-C; 160MB (~1600MB total) bajo umbral OOM con margen 446MB
        'max_work_mem_mb': 4,            # Minimal
        'max_maintenance_work_mem_mb': 64,
        'max_connections': 25,           # Reduced connections
        
        # Knobs to optimize (Pure survival mode)
        'knobs': [
            'shared_buffers',
            'work_mem',
            'effective_cache_size',
            'random_page_cost',
        ],
        
        # Description
        'description': 'Raspberry Pi 2GB - Extreme constraints (Survival Mode)',
        'storage': 'SD Card',
    },
    
    'i7_training': {
        # Hardware limits (for training on powerful machine)
        'max_memory_mb': 36864,          # 36GB (leave 4GB for OS)
        'max_swap_mb': 2048,             # Allow some swap for training
        'max_cpu_percent': 95,
        'baseline_tps': 1500,            # Estimated (no measurement; high-end hardware)
        
        # PostgreSQL-specific limits
        'shared_buffers_mb': 10240,      # 10GB
        'max_work_mem_mb': 128,
        'max_maintenance_work_mem_mb': 2048,
        'max_connections': 200,
        
        # Knobs to optimize (full set for training)
        'knobs': [
            'shared_buffers',
            'work_mem',
            'maintenance_work_mem',
            'effective_cache_size',
            'checkpoint_completion_target',
            'checkpoint_timeout',
            'wal_buffers',
            'random_page_cost',
            'max_parallel_workers_per_gather',
            'max_worker_processes',
        ],
        
        # Description
        'description': 'Intel i7 Gen 12 40GB - Training environment',
        'storage': 'NVMe SSD',
    }
}


def get_profile(tier):
    """
    Get constraint profile for a specific hardware tier
    
    Args:
        tier (str): Tier name ('rpi_4gb', 'rpi_8gb', 'i7_training')
    
    Returns:
        dict: Profile with constraints and knob list
    
    Raises:
        ValueError: If tier is unknown
    
    Example:
        >>> profile = get_profile('rpi_4gb')
        >>> print(profile['max_memory_mb'])
        3200
        >>> print(profile['knobs'])
        ['shared_buffers', 'work_mem', ...]
    """
    if tier not in TIER_PROFILES:
        available = ', '.join(TIER_PROFILES.keys())
        raise ValueError(
            f"Unknown tier: '{tier}'. Available tiers: {available}"
        )

    profile = dict(TIER_PROFILES[tier])
    env_baseline = os.environ.get('FOGPG_BASELINE_TPS')
    if env_baseline:
        try:
            profile['baseline_tps'] = float(env_baseline)
        except ValueError:
            pass
    return profile


def list_tiers():
    """
    List all available hardware tiers
    
    Returns:
        list: List of tier names
    """
    return list(TIER_PROFILES.keys())


def print_profile(tier):
    """
    Pretty-print a tier profile
    
    Args:
        tier (str): Tier name
    """
    profile = get_profile(tier)
    
    print(f"\n{'='*60}")
    print(f"TIER PROFILE: {tier}")
    print(f"{'='*60}")
    print(f"Description: {profile['description']}")
    print(f"Storage: {profile['storage']}")
    print(f"\nHARDWARE LIMITS:")
    print(f"  Max Memory: {profile['max_memory_mb']}MB")
    print(f"  Max Swap: {profile['max_swap_mb']}MB")
    print(f"  Max CPU: {profile['max_cpu_percent']}%")
    print(f"\nPOSTGRESQL LIMITS:")
    print(f"  shared_buffers: ≤{profile['shared_buffers_mb']}MB")
    print(f"  work_mem: ≤{profile['max_work_mem_mb']}MB")
    print(f"  maintenance_work_mem: ≤{profile['max_maintenance_work_mem_mb']}MB")
    print(f"  max_connections: ≤{profile['max_connections']}")
    print(f"\nOPTIMIZABLE KNOBS ({len(profile['knobs'])}):")
    for knob in profile['knobs']:
        print(f"  - {knob}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Demo: Print all profiles
    print("\nFogPG-Tuner Tier Profiles")
    print("=" * 60)
    
    for tier in list_tiers():
        print_profile(tier)
