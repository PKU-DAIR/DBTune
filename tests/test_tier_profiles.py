#!/usr/bin/env python3
"""
Unit tests for Tier Profiles

Run with: python tests/test_tier_profiles.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autotune.constraints.tier_profiles import get_profile, list_tiers, TIER_PROFILES


def test_get_profile_rpi_4gb():
    """Test 1: Get RPi 4GB profile"""
    print("\n" + "="*60)
    print("TEST 1: Get RPi 4GB profile")
    print("="*60)
    
    profile = get_profile('rpi_4gb')
    
    print(f"Max memory: {profile['max_memory_mb']}MB")
    print(f"Max swap: {profile['max_swap_mb']}MB")
    print(f"Knobs: {len(profile['knobs'])}")
    
    # Assertions
    assert profile['max_memory_mb'] == 3200, "Expected 3200MB"
    assert profile['max_swap_mb'] == 0, "Expected no swap"
    assert len(profile['knobs']) == 5, "Expected 5 knobs"
    assert 'shared_buffers' in profile['knobs'], "Expected shared_buffers"
    
    print("✅ PASSED")
    return True


def test_get_profile_rpi_8gb():
    """Test 2: Get RPi 8GB profile"""
    print("\n" + "="*60)
    print("TEST 2: Get RPi 8GB profile")
    print("="*60)
    
    profile = get_profile('rpi_8gb')
    
    print(f"Max memory: {profile['max_memory_mb']}MB")
    print(f"Max swap: {profile['max_swap_mb']}MB")
    print(f"Knobs: {len(profile['knobs'])}")
    
    # Assertions
    assert profile['max_memory_mb'] == 6656, "Expected 6656MB"
    assert profile['max_swap_mb'] == 0, "Expected no swap"
    assert len(profile['knobs']) == 8, "Expected 8 knobs"
    assert 'maintenance_work_mem' in profile['knobs'], "Expected maintenance_work_mem"
    
    print("✅ PASSED")
    return True


def test_get_profile_i7():
    """Test 3: Get i7 training profile"""
    print("\n" + "="*60)
    print("TEST 3: Get i7 training profile")
    print("="*60)
    
    profile = get_profile('i7_training')
    
    print(f"Max memory: {profile['max_memory_mb']}MB")
    print(f"Max swap: {profile['max_swap_mb']}MB")
    print(f"Knobs: {len(profile['knobs'])}")
    
    # Assertions
    assert profile['max_memory_mb'] == 36864, "Expected 36864MB"
    assert profile['max_swap_mb'] == 2048, "Expected 2048MB swap allowed"
    assert len(profile['knobs']) == 10, "Expected 10 knobs"
    
    print("✅ PASSED")
    return True


def test_invalid_tier():
    """Test 4: Invalid tier should raise ValueError"""
    print("\n" + "="*60)
    print("TEST 4: Invalid tier handling")
    print("="*60)
    
    try:
        profile = get_profile('invalid_tier')
        print("❌ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
        print("✅ PASSED")
        return True


def test_list_tiers():
    """Test 5: List all available tiers"""
    print("\n" + "="*60)
    print("TEST 5: List all tiers")
    print("="*60)
    
    tiers = list_tiers()
    
    print(f"Available tiers: {tiers}")
    
    # Assertions
    assert 'rpi_4gb' in tiers, "Expected rpi_4gb"
    assert 'rpi_8gb' in tiers, "Expected rpi_8gb"
    assert 'i7_training' in tiers, "Expected i7_training"
    assert len(tiers) == 3, "Expected 3 tiers"
    
    print("✅ PASSED")
    return True


def test_profile_constraints_logic():
    """Test 6: Verify constraint logic makes sense"""
    print("\n" + "="*60)
    print("TEST 6: Constraint logic validation")
    print("="*60)
    
    rpi_4gb = get_profile('rpi_4gb')
    rpi_8gb = get_profile('rpi_8gb')
    
    # RPi 8GB should have more resources than 4GB
    assert rpi_8gb['max_memory_mb'] > rpi_4gb['max_memory_mb'], "8GB should have more memory"
    assert rpi_8gb['shared_buffers_mb'] > rpi_4gb['shared_buffers_mb'], "8GB should allow more shared_buffers"
    assert len(rpi_8gb['knobs']) > len(rpi_4gb['knobs']), "8GB should optimize more knobs"
    
    # shared_buffers should be ~20% of max_memory
    for tier in ['rpi_4gb', 'rpi_8gb']:
        profile = get_profile(tier)
        ratio = profile['shared_buffers_mb'] / profile['max_memory_mb']
        print(f"{tier}: shared_buffers ratio = {ratio:.1%}")
        assert 0.15 < ratio < 0.25, f"shared_buffers should be ~20% of RAM, got {ratio:.1%}"
    
    print("✅ PASSED")
    return True


def main():
    """Run all tests"""
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + " "*12 + "Tier Profiles - Unit Tests" + " "*20 + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    tests = [
        test_get_profile_rpi_4gb,
        test_get_profile_rpi_8gb,
        test_get_profile_i7,
        test_invalid_tier,
        test_list_tiers,
        test_profile_constraints_logic,
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
