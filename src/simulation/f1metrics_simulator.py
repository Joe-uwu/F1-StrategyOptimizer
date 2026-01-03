# src/simulation/f1metrics_simulator.py
"""
F1Metrics-based Monte Carlo Race Simulator
Implements highly realistic lap-by-lap race simulation with:
- 8-parameter driver model
- Quadratic tyre degradation
- Traffic and overtaking logic
- Stochastic pit stop mechanics
- Full Monte Carlo analysis
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class TyreCompound(Enum):
    """Tyre compound types."""
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class PitStop:
    """Planned pit stop."""
    lap: int
    compound: TyreCompound


@dataclass
class Driver:
    """
    Driver class with 8 F1Metrics parameters.
    
    Parameters based on F1Metrics race simulator model:
    - Qualifying position affects Lap 1 time (~0.25s per grid slot)
    - Start bonus/penalty for positions gained/lost at start
    - Maximum speed for overtaking probability
    - Base race pace (theoretical clean air lap time on fresh Prime tyres)
    - Lap-time variability (sigma) for driver consistency
    - Pit strategy (planned stops and compounds)
    - DNF probability per lap
    - Tyre degradation multiplier (driver-specific wear factor)
    """
    
    # Core identification
    name: str
    driver_id: int
    
    # F1Metrics Model Parameters (8 parameters)
    qualifying_position: int  # Grid position (1-20)
    start_bonus: float  # Time adjustment for Lap 1 (seconds, can be negative)
    maximum_speed: float  # Top speed in km/h (for overtaking)
    base_race_pace: float  # Clean air lap time on fresh Prime tyres (seconds)
    lap_time_variability: float  # Standard deviation (sigma) in seconds (0.2-0.7s)
    pit_strategy: List[PitStop]  # Planned pit stops
    dnf_probability: float  # Per-lap failure probability (0.0-1.0)
    tyre_deg_multiplier: float  # Driver-specific wear factor (0.8-1.2)
    
    # Runtime state (not parameters)
    current_lap: int = 0
    current_compound: TyreCompound = TyreCompound.MEDIUM
    tyre_age: int = 0
    total_time: float = 0.0
    pit_stops_completed: int = 0
    is_dnf: bool = False
    position: int = 1
    
    # Traffic state
    time_behind_leader: float = 0.0
    car_ahead_id: Optional[int] = None
    
    def __post_init__(self):
        """Set initial compound from first pit strategy if available."""
        if self.pit_strategy and len(self.pit_strategy) > 0:
            # Start on compound before first stop
            # Default: start on medium unless specified otherwise
            self.current_compound = TyreCompound.MEDIUM
        self.tyre_age = 0


@dataclass
class RaceConfig:
    """Race configuration parameters."""
    total_laps: int = 50
    track_name: str = "Generic"
    
    # Fuel penalty
    fuel_penalty_per_lap: float = 0.037  # seconds per lap of fuel remaining
    
    # Tyre degradation parameters (quadratic model)
    # Degradation = base_rate * (tyre_age ** 2) * driver_multiplier
    tyre_deg_soft_base: float = 0.0020  # Soft: faster initially, degrades 2x faster
    tyre_deg_prime_base: float = 0.0010  # Prime (Medium/Hard): slower but durable
    soft_pace_advantage: float = 0.7  # Soft is 0.7s faster on fresh tyres
    
    # Pit stop mechanics
    pit_in_lap_penalty: float = 6.0  # Time lost entering pit lane
    pit_stop_mean_duration: float = 3.0  # Mean pit stop duration
    pit_stop_std_duration: float = 0.5  # Std dev (with fat tail for errors)
    pit_out_lap_penalty: float = 5.5  # Time lost exiting and warming tyres
    
    # Traffic and overtaking
    dirty_air_threshold: float = 1.0  # Within 1.0s = dirty air effect
    dirty_air_penalty: float = 0.2  # Stuck behind car: add 0.2s to pace
    overtake_base_probability: float = 0.15  # Base overtaking chance per lap
    pace_delta_coefficient: float = 0.10  # Bonus per 0.1s pace advantage
    mistake_factor_boost: float = 2.0  # Multiplier if car ahead makes mistake
    top_speed_coefficient: float = 0.005  # Bonus per 1 km/h speed advantage
    
    # Lap 1 grid penalty
    lap1_grid_penalty: float = 0.25  # Cumulative 0.25s per grid slot on Lap 1


class F1MetricsRaceSimulator:
    """
    Lap-by-lap Monte Carlo race simulator using F1Metrics model.
    
    This simulator operates on realistic physical and statistical models:
    - Lap time = Base + Random + Tyre Deg + Fuel + Traffic
    - Quadratic tyre degradation (soft degrades 2x faster)
    - Traffic simulation with dirty air and overtaking probability
    - Stochastic pit stops with realistic time distributions
    - Full Monte Carlo support for probabilistic strategy analysis
    """
    
    def __init__(self, config: RaceConfig, drivers: List[Driver], seed: Optional[int] = None):
        """
        Initialize race simulator.
        
        Args:
            config: Race configuration
            drivers: List of drivers in the race
            seed: Random seed for reproducibility
        """
        self.config = config
        self.drivers = {d.driver_id: d for d in drivers}
        self.rng = np.random.RandomState(seed)
        
        # Race state tracking
        self.current_lap = 0
        self.race_order: List[int] = []  # Driver IDs in current race order
        self.lap_history: List[Dict] = []  # History of each lap
        
        # Initialize race order by qualifying position
        self._initialize_race_order()
    
    def _initialize_race_order(self):
        """Set initial race order based on qualifying positions."""
        sorted_drivers = sorted(
            self.drivers.values(),
            key=lambda d: d.qualifying_position
        )
        self.race_order = [d.driver_id for d in sorted_drivers]
        
        # Set initial positions
        for position, driver_id in enumerate(self.race_order, start=1):
            self.drivers[driver_id].position = position
    
    def _calculate_random_variance(self, driver: Driver) -> float:
        """
        Calculate random lap time variance from Normal distribution.
        
        Args:
            driver: Driver with lap_time_variability parameter
        
        Returns:
            Random time delta (can be positive or negative)
        """
        return self.rng.normal(0, driver.lap_time_variability)
    
    def _calculate_fuel_adjustment(self, driver: Driver) -> float:
        """
        Calculate fuel weight penalty based on remaining laps.
        
        Formula: FuelAdj = fuel_penalty_per_lap * remaining_laps
        
        Args:
            driver: Driver instance
        
        Returns:
            Time penalty in seconds
        """
        remaining_laps = self.config.total_laps - driver.current_lap
        return self.config.fuel_penalty_per_lap * remaining_laps
    
    def _calculate_tyre_degradation(self, driver: Driver) -> float:
        """
        Calculate quadratic tyre degradation.
        
        Formula: TyreDeg = base_rate * (age^2) * driver_multiplier
        - Soft tyres: Start 0.7s faster, degrade 2x faster
        - Prime tyres: Slower initial pace but more durable
        
        Args:
            driver: Driver with tyre_age and current_compound
        
        Returns:
            Time loss in seconds due to tyre wear
        """
        age_squared = driver.tyre_age ** 2
        
        if driver.current_compound == TyreCompound.SOFT:
            # Soft: faster initially but degrades faster
            base_deg = self.config.tyre_deg_soft_base * age_squared
            # Initial advantage diminishes with age
            soft_advantage = max(0, self.config.soft_pace_advantage - (driver.tyre_age * 0.05))
            degradation = base_deg - soft_advantage
        else:
            # Medium/Hard (Prime tyres)
            base_deg = self.config.tyre_deg_prime_base * age_squared
            degradation = base_deg
        
        # Apply driver-specific tyre management multiplier
        return degradation * driver.tyre_deg_multiplier
    
    def _calculate_base_lap_time(self, driver: Driver) -> float:
        """
        Calculate theoretical lap time in clean air.
        
        LapTime = BasePace + RandomVar + TyreDeg + FuelAdj
        
        Args:
            driver: Driver instance
        
        Returns:
            Theoretical lap time in seconds (before traffic)
        """
        base_pace = driver.base_race_pace
        random_var = self._calculate_random_variance(driver)
        tyre_deg = self._calculate_tyre_degradation(driver)
        fuel_adj = self._calculate_fuel_adjustment(driver)
        
        # Lap 1 grid penalty
        lap1_penalty = 0.0
        if driver.current_lap == 1:
            grid_slots_behind = driver.qualifying_position - 1
            lap1_penalty = grid_slots_behind * self.config.lap1_grid_penalty
            lap1_penalty += driver.start_bonus  # Can be negative (good start)
        
        lap_time = base_pace + random_var + tyre_deg + fuel_adj + lap1_penalty
        
        return max(lap_time, 0.0), random_var  # Return both time and random var
    
    def _check_overtaking(
        self,
        driver_behind: Driver,
        driver_ahead: Driver,
        pace_delta: float,
        ahead_random_var: float
    ) -> bool:
        """
        Calculate probability of overtaking and determine if it happens.
        
        Factors:
        1. Pace delta (faster car has better chance)
        2. Mistake factor (if car ahead has bad lap)
        3. Top speed advantage
        
        Args:
            driver_behind: Attacking driver
            driver_ahead: Defending driver
            pace_delta: Time advantage of attacking driver (positive = faster)
            ahead_random_var: Random variance of car ahead (positive = mistake)
        
        Returns:
            True if overtake succeeds
        """
        base_prob = self.config.overtake_base_probability
        
        # Factor 1: Pace advantage (each 0.1s = 10% bonus)
        pace_bonus = (pace_delta / 0.1) * self.config.pace_delta_coefficient
        
        # Factor 2: Mistake factor (car ahead makes error)
        mistake_bonus = 0.0
        if ahead_random_var > 0.3:  # Car ahead had a bad lap
            mistake_bonus = self.config.mistake_factor_boost * base_prob
        
        # Factor 3: Top speed advantage
        speed_diff = driver_behind.maximum_speed - driver_ahead.maximum_speed
        speed_bonus = speed_diff * self.config.top_speed_coefficient
        
        # Total overtaking probability
        overtake_prob = base_prob + pace_bonus + mistake_bonus + speed_bonus
        overtake_prob = np.clip(overtake_prob, 0.0, 0.95)  # Max 95% per lap
        
        return self.rng.random() < overtake_prob
    
    def _simulate_traffic_and_overtaking(self, lap_times: Dict[int, Tuple[float, float]]):
        """
        Simulate dirty air effects and overtaking for this lap.
        
        Cars cannot simply drive at theoretical pace if stuck in traffic.
        Updates lap times based on traffic and resolves overtaking attempts.
        
        Args:
            lap_times: Dict of {driver_id: (theoretical_time, random_var)}
        """
        # Create list of (driver_id, lap_time, random_var, position)
        results = []
        for driver_id in self.race_order:
            driver = self.drivers[driver_id]
            if driver.is_dnf:
                continue
            
            lap_time, random_var = lap_times[driver_id]
            results.append({
                'driver_id': driver_id,
                'lap_time': lap_time,
                'random_var': random_var,
                'position': driver.position,
                'dirty_air_applied': False
            })
        
        # Sort by position to process order correctly
        results.sort(key=lambda x: x['position'])
        
        # Apply dirty air effects
        for i in range(1, len(results)):
            driver_behind = self.drivers[results[i]['driver_id']]
            driver_ahead = self.drivers[results[i-1]['driver_id']]
            
            # Calculate time gap (accumulated total time difference)
            time_gap = abs(driver_behind.total_time - driver_ahead.total_time)
            
            # Dirty air check: if within threshold
            if time_gap < self.config.dirty_air_threshold:
                # Car behind cannot drive faster than car ahead + penalty
                car_ahead_pace = results[i-1]['lap_time']
                own_pace = results[i]['lap_time']
                
                # Apply dirty air: must match car ahead pace + penalty
                if own_pace < car_ahead_pace:
                    results[i]['lap_time'] = car_ahead_pace + self.config.dirty_air_penalty
                    results[i]['dirty_air_applied'] = True
                    
                    # Attempt overtaking
                    pace_delta = car_ahead_pace - own_pace  # How much faster we are
                    ahead_random_var = results[i-1]['random_var']
                    
                    if self._check_overtaking(
                        driver_behind, driver_ahead, pace_delta, ahead_random_var
                    ):
                        # Overtake successful - swap positions
                        results[i]['position'], results[i-1]['position'] = \
                            results[i-1]['position'], results[i]['position']
        
        # Update lap times and positions
        for result in results:
            driver_id = result['driver_id']
            self.drivers[driver_id].position = result['position']
            lap_times[driver_id] = (result['lap_time'], result['random_var'])
    
    def _execute_pit_stop(self, driver: Driver, target_compound: TyreCompound) -> float:
        """
        Execute a pit stop with realistic time penalties.
        
        Time breakdown:
        - In-lap penalty (entering pit lane)
        - Pit stop duration (stochastic with fat tail)
        - Out-lap penalty (exit + tyre warm-up)
        
        Args:
            driver: Driver pitting
            target_compound: New tyre compound
        
        Returns:
            Total pit stop time loss
        """
        in_lap = self.config.pit_in_lap_penalty
        
        # Pit stop duration with fat tail (occasional errors)
        # Use exponential distribution for fat tail
        if self.rng.random() < 0.05:  # 5% chance of slow stop
            pit_duration = self.config.pit_stop_mean_duration + self.rng.exponential(3.0)
        else:
            pit_duration = self.rng.normal(
                self.config.pit_stop_mean_duration,
                self.config.pit_stop_std_duration
            )
        pit_duration = max(pit_duration, 2.0)  # Minimum 2.0s
        
        out_lap = self.config.pit_out_lap_penalty
        
        # Update driver state
        driver.current_compound = target_compound
        driver.tyre_age = 0
        driver.pit_stops_completed += 1
        
        total_time_loss = in_lap + pit_duration + out_lap
        
        logger.debug(
            f"Lap {self.current_lap}: {driver.name} pits "
            f"(in={in_lap:.1f}, stop={pit_duration:.1f}, out={out_lap:.1f}) "
            f"Total: {total_time_loss:.1f}s, New compound: {target_compound.value}"
        )
        
        return total_time_loss
    
    def _check_dnf(self, driver: Driver) -> bool:
        """
        Check if driver has a DNF (crash or mechanical failure).
        
        Args:
            driver: Driver to check
        
        Returns:
            True if DNF occurs this lap
        """
        if self.rng.random() < driver.dnf_probability:
            driver.is_dnf = True
            logger.info(f"Lap {self.current_lap}: {driver.name} DNF!")
            return True
        return False
    
    def _simulate_lap(self) -> Dict:
        """
        Simulate one lap of the race for all drivers.
        
        Returns:
            Dictionary with lap data
        """
        self.current_lap += 1
        lap_data = {
            'lap': self.current_lap,
            'driver_times': {},
            'positions': {},
            'pit_stops': [],
            'dnfs': []
        }
        
        # Calculate theoretical lap times for all drivers
        lap_times = {}  # {driver_id: (lap_time, random_var)}
        
        for driver_id, driver in self.drivers.items():
            if driver.is_dnf:
                continue
            
            driver.current_lap = self.current_lap
            
            # Check for DNF
            if self._check_dnf(driver):
                lap_data['dnfs'].append(driver.name)
                continue
            
            # Check if driver should pit
            should_pit = False
            target_compound = driver.current_compound
            
            for pit_stop in driver.pit_strategy:
                if pit_stop.lap == self.current_lap:
                    should_pit = True
                    target_compound = pit_stop.compound
                    break
            
            # Calculate base lap time
            lap_time, random_var = self._calculate_base_lap_time(driver)
            
            # Execute pit stop if planned
            if should_pit:
                pit_time = self._execute_pit_stop(driver, target_compound)
                lap_time += pit_time
                lap_data['pit_stops'].append({
                    'driver': driver.name,
                    'compound': target_compound.value,
                    'time_loss': pit_time
                })
            else:
                # Age tyres if not pitting
                driver.tyre_age += 1
            
            lap_times[driver_id] = (lap_time, random_var)
        
        # Simulate traffic and overtaking
        self._simulate_traffic_and_overtaking(lap_times)
        
        # Update total times and record data
        for driver_id, (lap_time, _) in lap_times.items():
            driver = self.drivers[driver_id]
            driver.total_time += lap_time
            
            lap_data['driver_times'][driver.name] = {
                'lap_time': lap_time,
                'total_time': driver.total_time,
                'tyre_age': driver.tyre_age,
                'compound': driver.current_compound.value,
                'position': driver.position
            }
        
        # Update race order by total time
        active_drivers = [d for d in self.drivers.values() if not d.is_dnf]
        active_drivers.sort(key=lambda d: d.total_time)
        
        for position, driver in enumerate(active_drivers, start=1):
            driver.position = position
            lap_data['positions'][driver.name] = position
        
        self.race_order = [d.driver_id for d in active_drivers]
        
        return lap_data
    
    def run_simulation(self) -> Dict:
        """
        Run a complete race simulation (all laps).
        
        Returns:
            Complete race results with lap-by-lap data
        """
        logger.info(f"Starting race simulation: {self.config.track_name}, {self.config.total_laps} laps")
        
        # Reset all drivers
        for driver in self.drivers.values():
            driver.current_lap = 0
            driver.total_time = 0.0
            driver.tyre_age = 0
            driver.pit_stops_completed = 0
            driver.is_dnf = False
        
        self._initialize_race_order()
        self.current_lap = 0
        self.lap_history = []
        
        # Simulate each lap
        for lap in range(self.config.total_laps):
            lap_data = self._simulate_lap()
            self.lap_history.append(lap_data)
        
        # Compile final results
        final_results = self._compile_results()
        
        logger.info(f"Race complete. Winner: {final_results['winner']}")
        
        return final_results
    
    def _compile_results(self) -> Dict:
        """Compile final race results."""
        active_drivers = [d for d in self.drivers.values() if not d.is_dnf]
        active_drivers.sort(key=lambda d: d.total_time)
        
        results = {
            'track': self.config.track_name,
            'total_laps': self.config.total_laps,
            'winner': active_drivers[0].name if active_drivers else "No finishers",
            'finishing_order': [
                {
                    'position': i + 1,
                    'driver': d.name,
                    'total_time': d.total_time,
                    'pit_stops': d.pit_stops_completed,
                    'tyre_age': d.tyre_age,
                    'final_compound': d.current_compound.value
                }
                for i, d in enumerate(active_drivers)
            ],
            'dnfs': [d.name for d in self.drivers.values() if d.is_dnf],
            'lap_history': self.lap_history
        }
        
        return results


class MonteCarloRunner:
    """
    Run multiple Monte Carlo simulations for probabilistic strategy analysis.
    
    This runner executes the race simulation N times with different random seeds
    to generate distributions of outcomes and probabilities.
    """
    
    def __init__(
        self,
        config: RaceConfig,
        drivers: List[Driver],
        n_simulations: int = 1000,
        base_seed: int = 42
    ):
        """
        Initialize Monte Carlo runner.
        
        Args:
            config: Race configuration
            drivers: List of drivers (will be deep-copied for each sim)
            n_simulations: Number of simulations to run
            base_seed: Base random seed
        """
        self.config = config
        self.base_drivers = drivers
        self.n_simulations = n_simulations
        self.base_seed = base_seed
        
        self.simulation_results: List[Dict] = []
    
    def run(self) -> Dict:
        """
        Execute Monte Carlo simulations.
        
        Returns:
            Aggregated results with probability distributions
        """
        logger.info(f"Starting Monte Carlo simulation: {self.n_simulations} runs")
        
        self.simulation_results = []
        
        for sim_idx in range(self.n_simulations):
            # Create fresh driver copies
            drivers = self._copy_drivers()
            
            # Run simulation with unique seed
            seed = self.base_seed + sim_idx
            simulator = F1MetricsRaceSimulator(self.config, drivers, seed=seed)
            result = simulator.run_simulation()
            
            self.simulation_results.append(result)
            
            if (sim_idx + 1) % 100 == 0:
                logger.info(f"Completed {sim_idx + 1}/{self.n_simulations} simulations")
        
        # Aggregate results
        aggregated = self._aggregate_results()
        
        logger.info("Monte Carlo simulation complete")
        
        return aggregated
    
    def _copy_drivers(self) -> List[Driver]:
        """Create fresh copies of drivers for new simulation."""
        return [
            Driver(
                name=d.name,
                driver_id=d.driver_id,
                qualifying_position=d.qualifying_position,
                start_bonus=d.start_bonus,
                maximum_speed=d.maximum_speed,
                base_race_pace=d.base_race_pace,
                lap_time_variability=d.lap_time_variability,
                pit_strategy=[PitStop(ps.lap, ps.compound) for ps in d.pit_strategy],
                dnf_probability=d.dnf_probability,
                tyre_deg_multiplier=d.tyre_deg_multiplier
            )
            for d in self.base_drivers
        ]
    
    def _aggregate_results(self) -> Dict:
        """
        Aggregate Monte Carlo results into probability distributions.
        
        Returns:
            Dictionary containing:
            - Win probabilities for each driver
            - Position distributions
            - Average race times
            - Strategy success rates
        """
        # Position frequency counter
        position_counts = defaultdict(lambda: defaultdict(int))
        total_times = defaultdict(list)
        dnf_counts = defaultdict(int)
        
        for result in self.simulation_results:
            # Count finishing positions
            for finish in result['finishing_order']:
                driver = finish['driver']
                position = finish['position']
                position_counts[driver][position] += 1
                total_times[driver].append(finish['total_time'])
            
            # Count DNFs
            for driver in result['dnfs']:
                dnf_counts[driver] += 1
        
        # Calculate probabilities
        driver_stats = {}
        
        for driver_name in position_counts.keys():
            positions = position_counts[driver_name]
            total_finishes = sum(positions.values())
            
            # Win probability
            win_prob = positions.get(1, 0) / self.n_simulations
            
            # Average position (excluding DNFs)
            avg_position = sum(pos * count for pos, count in positions.items()) / total_finishes if total_finishes > 0 else 0
            
            # Average race time
            avg_time = np.mean(total_times[driver_name]) if total_times[driver_name] else 0
            std_time = np.std(total_times[driver_name]) if len(total_times[driver_name]) > 1 else 0
            
            # DNF rate
            dnf_rate = dnf_counts.get(driver_name, 0) / self.n_simulations
            
            # Position distribution
            position_distribution = {
                pos: count / self.n_simulations
                for pos, count in positions.items()
            }
            
            driver_stats[driver_name] = {
                'win_probability': win_prob,
                'average_position': avg_position,
                'average_race_time': avg_time,
                'race_time_std': std_time,
                'dnf_rate': dnf_rate,
                'position_distribution': position_distribution
            }
        
        # Sort by win probability
        sorted_drivers = sorted(
            driver_stats.items(),
            key=lambda x: x[1]['win_probability'],
            reverse=True
        )
        
        aggregated = {
            'n_simulations': self.n_simulations,
            'driver_statistics': dict(sorted_drivers),
            'raw_results': self.simulation_results  # For detailed analysis
        }
        
        return aggregated
    
    def print_summary(self, aggregated_results: Dict):
        """
        Print human-readable summary of Monte Carlo results.
        
        Args:
            aggregated_results: Output from run()
        """
        print(f"\n{'='*70}")
        print(f"MONTE CARLO SIMULATION RESULTS ({aggregated_results['n_simulations']} runs)")
        print(f"{'='*70}\n")
        
        print(f"{'Driver':<20} {'Win %':<10} {'Avg Pos':<10} {'Avg Time':<12} {'DNF %':<10}")
        print("-" * 70)
        
        for driver_name, stats in aggregated_results['driver_statistics'].items():
            win_pct = stats['win_probability'] * 100
            avg_pos = stats['average_position']
            avg_time = stats['average_race_time']
            dnf_pct = stats['dnf_rate'] * 100
            
            print(f"{driver_name:<20} {win_pct:>6.2f}%   {avg_pos:>6.2f}    "
                  f"{avg_time:>8.1f}s    {dnf_pct:>6.2f}%")
        
        print(f"\n{'='*70}\n")


def create_example_race():
    """
    Create an example race with 5 drivers for demonstration.
    
    Returns:
        Tuple of (config, drivers)
    """
    config = RaceConfig(
        total_laps=50,
        track_name="Monza",
        fuel_penalty_per_lap=0.037
    )
    
    # Create 5 example drivers with different strategies
    drivers = [
        Driver(
            name="Hamilton",
            driver_id=1,
            qualifying_position=1,
            start_bonus=0.0,
            maximum_speed=340.0,
            base_race_pace=85.0,
            lap_time_variability=0.25,
            pit_strategy=[
                PitStop(lap=15, compound=TyreCompound.MEDIUM),
                PitStop(lap=35, compound=TyreCompound.HARD)
            ],
            dnf_probability=0.001,
            tyre_deg_multiplier=0.9
        ),
        Driver(
            name="Verstappen",
            driver_id=2,
            qualifying_position=2,
            start_bonus=0.1,  # Good start
            maximum_speed=342.0,
            base_race_pace=84.8,
            lap_time_variability=0.20,
            pit_strategy=[
                PitStop(lap=18, compound=TyreCompound.HARD),
                PitStop(lap=38, compound=TyreCompound.MEDIUM)
            ],
            dnf_probability=0.001,
            tyre_deg_multiplier=0.85
        ),
        Driver(
            name="Leclerc",
            driver_id=3,
            qualifying_position=3,
            start_bonus=-0.2,  # Bad start
            maximum_speed=338.0,
            base_race_pace=85.2,
            lap_time_variability=0.30,
            pit_strategy=[
                PitStop(lap=12, compound=TyreCompound.SOFT),
                PitStop(lap=28, compound=TyreCompound.MEDIUM),
                PitStop(lap=42, compound=TyreCompound.SOFT)
            ],
            dnf_probability=0.002,
            tyre_deg_multiplier=1.1
        ),
        Driver(
            name="Norris",
            driver_id=4,
            qualifying_position=4,
            start_bonus=0.0,
            maximum_speed=336.0,
            base_race_pace=85.5,
            lap_time_variability=0.28,
            pit_strategy=[
                PitStop(lap=20, compound=TyreCompound.HARD),
            ],
            dnf_probability=0.0015,
            tyre_deg_multiplier=1.0
        ),
        Driver(
            name="Sainz",
            driver_id=5,
            qualifying_position=5,
            start_bonus=0.15,
            maximum_speed=337.0,
            base_race_pace=85.3,
            lap_time_variability=0.32,
            pit_strategy=[
                PitStop(lap=16, compound=TyreCompound.MEDIUM),
                PitStop(lap=34, compound=TyreCompound.SOFT)
            ],
            dnf_probability=0.0018,
            tyre_deg_multiplier=0.95
        ),
    ]
    
    return config, drivers


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create example race
    config, drivers = create_example_race()
    
    # Run single simulation
    print("\n" + "="*70)
    print("SINGLE RACE SIMULATION")
    print("="*70)
    
    simulator = F1MetricsRaceSimulator(config, drivers, seed=42)
    result = simulator.run_simulation()
    
    print("\nFinal Results:")
    for finish in result['finishing_order'][:5]:
        print(f"  P{finish['position']}: {finish['driver']} - "
              f"{finish['total_time']:.1f}s ({finish['pit_stops']} stops)")
    
    # Run Monte Carlo simulation
    print("\n" + "="*70)
    print("MONTE CARLO SIMULATION (100 runs for demo)")
    print("="*70)
    
    mc_runner = MonteCarloRunner(config, drivers, n_simulations=100, base_seed=42)
    mc_results = mc_runner.run()
    mc_runner.print_summary(mc_results)
