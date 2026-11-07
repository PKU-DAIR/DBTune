#!/usr/bin/env python3
"""
Unit tests for FogAwareReward

Run with: python tests/test_fog_reward.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autotune.reward.fog_aware_reward import FogAwareReward


def test_normal_operation():
    """Test 1: Normal operation without constraint violations"""
    print("\n" + "="*60)
    print("TEST 1: Normal operation")
    print("="*60)
    
    reward = FogAwareReward(
        constraints={'max_memory_mb': 3200, 'max_swap_mb': 0},
        baseline_tps=2000
    )
    
    result = reward.calculate(
        tps=2400,           # 20% better than baseline
        memory_used_mb=2000, # 62.5% of max memory (safe)
        swap_used_mb=0       # No swap (good)
    )
    
    print(f"TPS: 2400 (baseline: 2000)")
    print(f"Memory: 2000MB / 3200MB (62.5%)")
    print(f"Swap: 0MB")
    print(f"→ Reward: {result:.2f}")
    
    # Assertions
    assert result > 100, f"Expected positive reward, got {result}"
    assert reward.is_valid_config(2000, 0), "Config should be valid"
    
    print("✅ PASSED")
    return True


def test_swap_penalty():
    """Test 2: Configuration with swapping should be heavily penalized"""
    print("\n" + "="*60)
    print("TEST 2: Swap penalty")
    print("="*60)
    
    reward = FogAwareReward(
        constraints={'max_memory_mb': 3200, 'max_swap_mb': 0},
        baseline_tps=2000
    )
    
    result = reward.calculate(
        tps=2500,            # Good TPS
        memory_used_mb=3100, # High memory (96.8%)
        swap_used_mb=50      # ❌ SWAP DETECTED!
    )
    
    print(f"TPS: 2500")
    print(f"Memory: 3100MB / 3200MB (96.8%)")
    print(f"Swap: 50MB (VIOLATION!)")
    print(f"→ Reward: {result:.2f}")
    
    # Assertions
    assert result < -900, f"Swap should give severe penalty, got {result}"
    assert not reward.is_valid_config(3100, 50), "Config with swap should be invalid"
    
    print("✅ PASSED")
    return True


def test_memory_critical():
    """Test 3: Memory >95% should be heavily penalized"""
    print("\n" + "="*60)
    print("TEST 3: Memory critical penalty")
    print("="*60)
    
    reward = FogAwareReward(
        constraints={'max_memory_mb': 3200, 'max_swap_mb': 0},
        baseline_tps=2000
    )
    
    result = reward.calculate(
        tps=2600,
        memory_used_mb=3100,  # 96.8% memory (CRITICAL)
        swap_used_mb=0
    )
    
    print(f"TPS: 2600")
    print(f"Memory: 3100MB / 3200MB (96.8%)")
    print(f"Swap: 0MB")
    print(f"→ Reward: {result:.2f}")
    
    # Assertions
    assert result < 0, f"High memory should give negative reward, got {result}"
    
    print("✅ PASSED")
    return True


def test_memory_high():
    """Test 4: Memory >85% but <95% should be moderately penalized"""
    print("\n" + "="*60)
    print("TEST 4: Memory high penalty")
    print("="*60)
    
    reward = FogAwareReward(
        constraints={'max_memory_mb': 3200, 'max_swap_mb': 0},
        baseline_tps=2000
    )
    
    result = reward.calculate(
        tps=2500,
        memory_used_mb=2900,  # 90.6% memory (HIGH)
        swap_used_mb=0
    )
    
    print(f"TPS: 2500")
    print(f"Memory: 2900MB / 3200MB (90.6%)")
    print(f"Swap: 0MB")
    print(f"→ Reward: {result:.2f}")
    
    # Assertions
    assert result < 125, f"High memory should reduce reward, got {result}"
    assert result > 0, f"Should still be positive (good TPS), got {result}"
    assert reward.is_valid_config(2900, 0), "Config should still be valid"
    
    print("✅ PASSED")
    return True


def test_excellent_config():
    """Test 5: Excellent configuration should get high reward"""
    print("\n" + "="*60)
    print("TEST 5: Excellent configuration")
    print("="*60)
    
    reward = FogAwareReward(
        constraints={'max_memory_mb': 3200, 'max_swap_mb': 0},
        baseline_tps=2000
    )
    
    result = reward.calculate(
        tps=3000,           # 50% better than baseline!
        memory_used_mb=1800, # Only 56% memory
        swap_used_mb=0
    )
    
    print(f"TPS: 3000 (baseline: 2000)")
    print(f"Memory: 1800MB / 3200MB (56.2%)")
    print(f"Swap: 0MB")
    print(f"→ Reward: {result:.2f}")
    
    # Assertions
    assert result > 140, f"Excellent config should get high reward, got {result}"
    assert reward.is_valid_config(1800, 0), "Config should be valid"
    
    print("✅ PASSED")
    return True


def main():
    """Run all tests"""
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + " "*10 + "FogAwareReward - Unit Tests" + " "*21 + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    tests = [
        test_normal_operation,
        test_swap_penalty,
        test_memory_critical,
        test_memory_high,
        test_excellent_config,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
