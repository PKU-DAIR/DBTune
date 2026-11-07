#!/usr/bin/env python3
"""
Integration test: Verify DBTune can load FogAwareReward
Run with: python tests/test_integration.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autotune.reward.fog_aware_reward import FogAwareReward
from autotune.constraints.tier_profiles import get_profile

def test_reward_integration():
    """Test that reward can be created from tier profile"""
    print("\n" + "="*60)
    print("TEST: Create FogAwareReward from tier profile")
    print("="*60)
    
    # Get profile
    profile = get_profile('rpi_4gb')
    print(f"Loaded profile: {profile['description']}")
    
    # Create reward
    reward = FogAwareReward(
        constraints=profile,
        baseline_tps=2000.0
    )
    
    print(f"Created reward: {reward}")
    
    # Test calculation
    result = reward.calculate(
        tps=2400,
        memory_used_mb=2000,
        swap_used_mb=0
    )
    
    print(f"Test calculation: TPS=2400, Mem=2000MB → Reward={result:.2f}")
    
    assert result > 0, "Should give positive reward for good config"
    
    print("✅ PASSED")
    return True

def test_all_tiers():
    """Test that all tiers can create rewards"""
    print("\n" + "="*60)
    print("TEST: Create rewards for all tiers")
    print("="*60)
    
    from autotune.constraints.tier_profiles import list_tiers
    
    for tier in list_tiers():
        profile = get_profile(tier)
        reward = FogAwareReward(constraints=profile, baseline_tps=2000)
        print(f"  ✓ {tier}: {reward}")
    
    print("✅ PASSED")
    return True

def main():
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + " "*15 + "Integration Tests" + " "*26 + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    tests = [
        test_reward_integration,
        test_all_tiers,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
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
