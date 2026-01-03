#!/usr/bin/env python3
"""
Quick test script to verify F1Metrics simulator installation and functionality.
Run this after installation to ensure everything works correctly.
"""

import sys

print("="*70)
print("F1METRICS SIMULATOR - QUICK TEST")
print("="*70)

# Test 1: Import modules
print("\n[1/5] Testing imports...")
try:
    from src.simulation import (
        F1MetricsRaceSimulator,
        MonteCarloRunner,
        Driver,
        PitStop,
        TyreCompound,
        RaceConfig,
        DriverCalibrator,
        SessionData
    )
    print("✅ All modules imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Create basic driver
print("\n[2/5] Creating test driver...")
try:
    driver = Driver(
        name="TestDriver",
        driver_id=1,
        qualifying_position=1,
        start_bonus=0.0,
        maximum_speed=340.0,
        base_race_pace=85.0,
        lap_time_variability=0.25,
        pit_strategy=[PitStop(lap=15, compound=TyreCompound.MEDIUM)],
        dnf_probability=0.001,
        tyre_deg_multiplier=1.0
    )
    print(f"✅ Driver created: {driver.name}")
except Exception as e:
    print(f"❌ Driver creation failed: {e}")
    sys.exit(1)

# Test 3: Create race config
print("\n[3/5] Creating race configuration...")
try:
    config = RaceConfig(
        total_laps=10,  # Short race for quick test
        track_name="Test Track"
    )
    print(f"✅ Race config created: {config.track_name}, {config.total_laps} laps")
except Exception as e:
    print(f"❌ Config creation failed: {e}")
    sys.exit(1)

# Test 4: Run single simulation
print("\n[4/5] Running single race simulation (10 laps)...")
try:
    simulator = F1MetricsRaceSimulator(config, [driver], seed=42)
    result = simulator.run_simulation()
    
    print(f"✅ Simulation complete")
    print(f"   Winner: {result['winner']}")
    print(f"   Total time: {result['finishing_order'][0]['total_time']:.2f}s")
    print(f"   Pit stops: {result['finishing_order'][0]['pit_stops']}")
except Exception as e:
    print(f"❌ Simulation failed: {e}")
    sys.exit(1)

# Test 5: Run mini Monte Carlo
print("\n[5/5] Running Monte Carlo analysis (10 simulations)...")
try:
    # Create two drivers for comparison
    driver2 = Driver(
        name="Rival",
        driver_id=2,
        qualifying_position=2,
        start_bonus=0.1,
        maximum_speed=338.0,
        base_race_pace=85.3,
        lap_time_variability=0.30,
        pit_strategy=[PitStop(lap=12, compound=TyreCompound.HARD)],
        dnf_probability=0.002,
        tyre_deg_multiplier=1.1
    )
    
    mc_runner = MonteCarloRunner(
        config,
        [driver, driver2],
        n_simulations=10,  # Small number for quick test
        base_seed=42
    )
    
    results = mc_runner.run()
    
    print(f"✅ Monte Carlo complete")
    print(f"   Simulations: {results['n_simulations']}")
    
    for driver_name, stats in results['driver_statistics'].items():
        win_pct = stats['win_probability'] * 100
        print(f"   {driver_name}: {win_pct:.1f}% win probability")
    
except Exception as e:
    print(f"❌ Monte Carlo failed: {e}")
    sys.exit(1)

# Test 6: Verify calibrator
print("\n[6/6] Testing driver calibrator...")
try:
    session_data = SessionData(
        driver_name="TestDriver",
        fp2_best_time=84.5,
        qualifying_time=83.8,
        qualifying_position=1,
        race_pace_samples=[85.0, 85.1, 85.2],
        top_speed_kmh=340.0
    )
    
    calibrator = DriverCalibrator()
    calibrated_driver = calibrator.calibrate_driver(
        session_data,
        field_fp2_median=85.0,
        field_qualifying_median=84.5,
        field_avg_degradation=3.5,
        driver_id=1,
        pit_strategy=[PitStop(lap=15, compound=TyreCompound.MEDIUM)]
    )
    
    print(f"✅ Calibration successful")
    print(f"   Base pace: {calibrated_driver.base_race_pace:.3f}s")
    print(f"   Variability: {calibrated_driver.lap_time_variability:.3f}s")
    
except Exception as e:
    print(f"❌ Calibration failed: {e}")
    sys.exit(1)

# All tests passed!
print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nF1Metrics simulator is working correctly.")
print("\nNext steps:")
print("  - Run full demo: python run_f1metrics_demo.py")
print("  - Read guide: F1METRICS_IMPLEMENTATION.md")
print("  - Check examples in run_f1metrics_demo.py")
print()
