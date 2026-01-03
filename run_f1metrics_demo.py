#!/usr/bin/env python3
"""
Demo script for F1Metrics Monte Carlo Race Simulator.

This script demonstrates how to:
1. Set up drivers with the 8 F1Metrics parameters
2. Run a single race simulation
3. Execute Monte Carlo analysis (1000+ simulations)
4. Analyze strategy effectiveness
5. Generate probability distributions
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from src.simulation.f1metrics_simulator import (
    F1MetricsRaceSimulator,
    MonteCarloRunner,
    RaceConfig,
    Driver,
    PitStop,
    TyreCompound
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_realistic_race_scenario():
    """
    Create a realistic race scenario with 10 drivers.
    
    Demonstrates proper parameter calibration based on F1Metrics model:
    - Base pace calculated from practice/qualifying data
    - Realistic variability (consistent drivers = 0.2s, erratic = 0.5s+)
    - Strategy diversity (1-stop, 2-stop, 3-stop)
    - Different tyre management abilities
    """
    
    config = RaceConfig(
        total_laps=53,  # Monaco GP length
        track_name="Monaco",
        fuel_penalty_per_lap=0.037,
        # Monaco: tight circuit, harder overtaking
        overtake_base_probability=0.08,  # Lower than average
        dirty_air_threshold=1.2,
        pit_in_lap_penalty=7.0,  # Monaco has longer pit lane
        pit_out_lap_penalty=6.5
    )
    
    drivers = [
        # P1: Verstappen - Dominant pace, consistent, 1-stop strategy
        Driver(
            name="Verstappen",
            driver_id=1,
            qualifying_position=1,
            start_bonus=0.0,
            maximum_speed=310.0,  # Monaco top speed lower
            base_race_pace=73.5,  # Monaco ~73s lap
            lap_time_variability=0.20,  # Very consistent
            pit_strategy=[
                PitStop(lap=28, compound=TyreCompound.HARD)
            ],
            dnf_probability=0.0008,
            tyre_deg_multiplier=0.85  # Excellent tyre management
        ),
        
        # P2: Hamilton - Similar pace, good starter
        Driver(
            name="Hamilton",
            driver_id=2,
            qualifying_position=2,
            start_bonus=0.15,  # Good at race starts
            maximum_speed=309.0,
            base_race_pace=73.6,
            lap_time_variability=0.22,
            pit_strategy=[
                PitStop(lap=26, compound=TyreCompound.HARD)
            ],
            dnf_probability=0.0010,
            tyre_deg_multiplier=0.90
        ),
        
        # P3: Leclerc - Fast but inconsistent, aggressive 2-stop
        Driver(
            name="Leclerc",
            driver_id=3,
            qualifying_position=3,
            start_bonus=-0.1,  # Slightly slower start
            maximum_speed=311.0,
            base_race_pace=73.7,
            lap_time_variability=0.35,  # More variable
            pit_strategy=[
                PitStop(lap=18, compound=TyreCompound.MEDIUM),
                PitStop(lap=38, compound=TyreCompound.HARD)
            ],
            dnf_probability=0.0020,  # Higher risk
            tyre_deg_multiplier=1.05  # Harder on tyres
        ),
        
        # P4: Perez - Solid midfield, good tyre management
        Driver(
            name="Perez",
            driver_id=4,
            qualifying_position=4,
            start_bonus=0.05,
            maximum_speed=308.0,
            base_race_pace=73.9,
            lap_time_variability=0.28,
            pit_strategy=[
                PitStop(lap=30, compound=TyreCompound.HARD)
            ],
            dnf_probability=0.0012,
            tyre_deg_multiplier=0.88
        ),
        
        # P5: Sainz - Trying 3-stop aggressive strategy
        Driver(
            name="Sainz",
            driver_id=5,
            qualifying_position=5,
            start_bonus=0.0,
            maximum_speed=310.0,
            base_race_pace=74.0,
            lap_time_variability=0.30,
            pit_strategy=[
                PitStop(lap=15, compound=TyreCompound.SOFT),
                PitStop(lap=30, compound=TyreCompound.SOFT),
                PitStop(lap=45, compound=TyreCompound.HARD)
            ],
            dnf_probability=0.0015,
            tyre_deg_multiplier=1.00
        ),
        
        # P6-10: Midfield runners with varying strategies
        Driver(
            name="Norris",
            driver_id=6,
            qualifying_position=6,
            start_bonus=0.1,
            maximum_speed=307.0,
            base_race_pace=74.2,
            lap_time_variability=0.25,
            pit_strategy=[
                PitStop(lap=22, compound=TyreCompound.MEDIUM),
                PitStop(lap=42, compound=TyreCompound.HARD)
            ],
            dnf_probability=0.0010,
            tyre_deg_multiplier=0.95
        ),
        
        Driver(
            name="Alonso",
            driver_id=7,
            qualifying_position=7,
            start_bonus=0.2,  # Excellent starter
            maximum_speed=306.0,
            base_race_pace=74.3,
            lap_time_variability=0.24,
            pit_strategy=[
                PitStop(lap=35, compound=TyreCompound.HARD)  # Late 1-stop
            ],
            dnf_probability=0.0008,
            tyre_deg_multiplier=0.82  # Best tyre management on grid
        ),
        
        Driver(
            name="Russell",
            driver_id=8,
            qualifying_position=8,
            start_bonus=-0.05,
            maximum_speed=308.0,
            base_race_pace=74.1,
            lap_time_variability=0.27,
            pit_strategy=[
                PitStop(lap=25, compound=TyreCompound.HARD)
            ],
            dnf_probability=0.0012,
            tyre_deg_multiplier=0.92
        ),
        
        Driver(
            name="Piastri",
            driver_id=9,
            qualifying_position=9,
            start_bonus=0.0,
            maximum_speed=307.0,
            base_race_pace=74.4,
            lap_time_variability=0.29,
            pit_strategy=[
                PitStop(lap=27, compound=TyreCompound.MEDIUM)
            ],
            dnf_probability=0.0011,
            tyre_deg_multiplier=0.98
        ),
        
        Driver(
            name="Gasly",
            driver_id=10,
            qualifying_position=10,
            start_bonus=0.05,
            maximum_speed=305.0,
            base_race_pace=74.5,
            lap_time_variability=0.32,
            pit_strategy=[
                PitStop(lap=20, compound=TyreCompound.SOFT),
                PitStop(lap=40, compound=TyreCompound.HARD)
            ],
            dnf_probability=0.0016,
            tyre_deg_multiplier=1.02
        ),
    ]
    
    return config, drivers


def run_single_simulation(config, drivers):
    """Run and display a single race simulation."""
    print("\n" + "="*80)
    print("SINGLE RACE SIMULATION")
    print("="*80)
    print(f"Track: {config.track_name} | Laps: {config.total_laps}\n")
    
    simulator = F1MetricsRaceSimulator(config, drivers, seed=42)
    result = simulator.run_simulation()
    
    print("\n📊 FINAL CLASSIFICATION:")
    print("-" * 80)
    print(f"{'Pos':<5} {'Driver':<15} {'Total Time':<12} {'Stops':<7} {'Final Tyre':<12}")
    print("-" * 80)
    
    for finish in result['finishing_order']:
        pos = finish['position']
        driver = finish['driver']
        time = finish['total_time']
        stops = finish['pit_stops']
        compound = finish['final_compound']
        
        # Calculate gap to leader
        if pos == 1:
            gap_str = "Winner"
        else:
            gap = time - result['finishing_order'][0]['total_time']
            gap_str = f"+{gap:.2f}s"
        
        print(f"P{pos:<4} {driver:<15} {gap_str:<12} {stops:<7} {compound:<12}")
    
    if result['dnfs']:
        print(f"\n❌ DNFs: {', '.join(result['dnfs'])}")
    
    return result


def run_monte_carlo_analysis(config, drivers, n_simulations=1000):
    """Run Monte Carlo simulation and display results."""
    print("\n" + "="*80)
    print(f"MONTE CARLO ANALYSIS ({n_simulations} simulations)")
    print("="*80)
    print("This may take a few minutes...\n")
    
    mc_runner = MonteCarloRunner(config, drivers, n_simulations=n_simulations, base_seed=42)
    results = mc_runner.run()
    
    # Display summary
    mc_runner.print_summary(results)
    
    return results


def plot_monte_carlo_results(mc_results, output_file='outputs/f1metrics_analysis.png'):
    """
    Create visualization of Monte Carlo results.
    
    Generates:
    1. Win probability bar chart
    2. Position distribution box plots
    3. Race time distribution
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('F1Metrics Monte Carlo Race Analysis', fontsize=16, fontweight='bold')
        
        stats = mc_results['driver_statistics']
        drivers = list(stats.keys())
        
        # 1. Win Probability
        win_probs = [stats[d]['win_probability'] * 100 for d in drivers]
        axes[0, 0].barh(drivers, win_probs, color='#00D2BE')
        axes[0, 0].set_xlabel('Win Probability (%)')
        axes[0, 0].set_title('Win Probability by Driver')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. Average Position
        avg_positions = [stats[d]['average_position'] for d in drivers]
        colors = plt.cm.RdYlGn_r([p/max(avg_positions) for p in avg_positions])
        axes[0, 1].barh(drivers, avg_positions, color=colors)
        axes[0, 1].set_xlabel('Average Finishing Position')
        axes[0, 1].set_title('Average Position (Lower is Better)')
        axes[0, 1].invert_xaxis()
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Race Time Distribution (top 5 drivers by win prob)
        top_5_drivers = sorted(drivers, key=lambda d: stats[d]['win_probability'], reverse=True)[:5]
        race_times = []
        labels = []
        
        for driver in top_5_drivers:
            # Get race times from raw results
            times = [
                r['finishing_order'][next(i for i, f in enumerate(r['finishing_order']) if f['driver'] == driver)]['total_time']
                for r in mc_results['raw_results']
                if driver in [f['driver'] for f in r['finishing_order']]
            ]
            race_times.append(times)
            labels.append(driver)
        
        axes[1, 0].boxplot(race_times, labels=labels, vert=True)
        axes[1, 0].set_ylabel('Total Race Time (seconds)')
        axes[1, 0].set_title('Race Time Distribution (Top 5 Contenders)')
        axes[1, 0].grid(axis='y', alpha=0.3)
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. DNF Rate
        dnf_rates = [stats[d]['dnf_rate'] * 100 for d in drivers]
        axes[1, 1].barh(drivers, dnf_rates, color='#E10600')
        axes[1, 1].set_xlabel('DNF Rate (%)')
        axes[1, 1].set_title('Reliability (DNF Rate)')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n📈 Visualization saved to: {output_file}")
        
    except ImportError:
        print("\n⚠️  matplotlib not available. Skipping visualization.")
    except Exception as e:
        print(f"\n⚠️  Could not create visualization: {e}")


def compare_strategies(mc_results):
    """
    Analyze strategy effectiveness from Monte Carlo results.
    """
    print("\n" + "="*80)
    print("STRATEGY ANALYSIS")
    print("="*80)
    
    # Group drivers by number of stops
    strategy_performance = defaultdict(list)
    
    for result in mc_results['raw_results']:
        for finish in result['finishing_order']:
            driver = finish['driver']
            stops = finish['pit_stops']
            position = finish['position']
            strategy_performance[stops].append(position)
    
    print("\nAverage Position by Strategy:")
    print("-" * 40)
    
    from collections import defaultdict
    for stops in sorted(strategy_performance.keys()):
        positions = strategy_performance[stops]
        avg_pos = np.mean(positions)
        std_pos = np.std(positions)
        print(f"{stops}-stop strategy: Avg P{avg_pos:.2f} (±{std_pos:.2f})")


def main():
    """Main demonstration script."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    F1METRICS MONTE CARLO RACE SIMULATOR                      ║
║                                                                              ║
║  Realistic lap-by-lap simulation with:                                      ║
║  ✓ 8-parameter driver model (F1Metrics methodology)                         ║
║  ✓ Quadratic tyre degradation (soft degrades 2x faster)                     ║
║  ✓ Traffic simulation with dirty air effects                                ║
║  ✓ Probabilistic overtaking based on pace, mistakes, and top speed          ║
║  ✓ Stochastic pit stops with fat-tail distributions                         ║
║  ✓ Monte Carlo analysis for strategy optimization                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Create race scenario
    config, drivers = create_realistic_race_scenario()
    
    # Step 1: Single simulation
    single_result = run_single_simulation(config, drivers)
    
    # Step 2: Monte Carlo analysis (default: 1000 simulations)
    # Change n_simulations to 100 for faster demo, 1000+ for production
    n_sims = 1000
    mc_results = run_monte_carlo_analysis(config, drivers, n_simulations=n_sims)
    
    # Step 3: Visualize results
    plot_monte_carlo_results(mc_results)
    
    # Step 4: Strategy comparison
    compare_strategies(mc_results)
    
    print("\n✅ F1Metrics simulation complete!")
    print("\nNext steps:")
    print("  1. Adjust driver parameters based on real data (FP2, qualifying)")
    print("  2. Test different strategies by modifying pit_strategy")
    print("  3. Increase n_simulations to 5000+ for production analysis")
    print("  4. Integrate with your RL agent for strategy optimization\n")


if __name__ == "__main__":
    main()
