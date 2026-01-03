# src/simulation/driver_calibration.py
"""
Utility functions for calibrating F1Metrics driver parameters from real data.

This module helps convert real F1 data (practice times, qualifying, race results)
into the 8 F1Metrics driver parameters for accurate race simulation.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from .f1metrics_simulator import Driver, PitStop, TyreCompound


@dataclass
class SessionData:
    """Real session data for a driver."""
    driver_name: str
    fp2_best_time: float  # Best lap in FP2 (seconds)
    qualifying_time: float  # Q3 best lap (seconds)
    qualifying_position: int  # Grid position
    race_pace_samples: List[float]  # Clean air race lap times
    top_speed_kmh: float  # Maximum speed recorded


class DriverCalibrator:
    """
    Calibrate F1Metrics driver parameters from real session data.
    
    Uses the F1Metrics methodology:
    - Base Race Pace = FP2_Time + 0.5 * (Qualifying_Delta - FP2_Delta)
    - Lap-Time Variability = StdDev of clean air laps
    - Tyre Degradation Multiplier = from stint analysis
    """
    
    @staticmethod
    def calculate_base_race_pace(
        fp2_time: float,
        qualifying_time: float,
        field_fp2_median: float,
        field_qualifying_median: float
    ) -> float:
        """
        Calculate base race pace using F1Metrics formula.
        
        Formula: BasePace = FP2_Time + 0.5 * (Qualifying_Delta - FP2_Delta)
        
        This accounts for:
        - Practice pace as baseline
        - Adjustment for qualifying performance relative to field
        
        Args:
            fp2_time: Driver's best FP2 lap time
            qualifying_time: Driver's best qualifying lap
            field_fp2_median: Median FP2 time across field
            field_qualifying_median: Median qualifying time across field
        
        Returns:
            Predicted base race pace (seconds)
        """
        fp2_delta = fp2_time - field_fp2_median
        qualifying_delta = qualifying_time - field_qualifying_median
        
        base_pace = fp2_time + 0.5 * (qualifying_delta - fp2_delta)
        
        return base_pace
    
    @staticmethod
    def calculate_lap_time_variability(race_pace_samples: List[float]) -> float:
        """
        Calculate driver consistency (lap-time variability).
        
        Uses standard deviation of clean air laps (no traffic).
        Typical values:
        - Elite consistent drivers: 0.20-0.25s
        - Average drivers: 0.25-0.35s
        - Inconsistent drivers: 0.35-0.50s+
        
        Args:
            race_pace_samples: List of clean air lap times
        
        Returns:
            Standard deviation (sigma) in seconds
        """
        if len(race_pace_samples) < 3:
            # Not enough data, use default
            return 0.30
        
        # Remove outliers (likely traffic or mistakes)
        samples = np.array(race_pace_samples)
        q1, q3 = np.percentile(samples, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filtered = samples[(samples >= lower_bound) & (samples <= upper_bound)]
        
        if len(filtered) < 3:
            filtered = samples
        
        return float(np.std(filtered))
    
    @staticmethod
    def estimate_tyre_deg_multiplier(
        stint_degradation_observed: float,
        field_average_degradation: float
    ) -> float:
        """
        Estimate driver-specific tyre degradation multiplier.
        
        Compares driver's observed tyre wear vs field average.
        
        Args:
            stint_degradation_observed: Driver's lap time increase over stint (s)
            field_average_degradation: Average for the field (s)
        
        Returns:
            Multiplier (1.0 = average, <1.0 = good tyre mgmt, >1.0 = harsh)
        """
        if field_average_degradation == 0:
            return 1.0
        
        multiplier = stint_degradation_observed / field_average_degradation
        
        # Clamp to reasonable range
        return np.clip(multiplier, 0.7, 1.3)
    
    @staticmethod
    def estimate_start_bonus(
        grid_position: int,
        turn1_position: int
    ) -> float:
        """
        Calculate Lap 1 start bonus/penalty.
        
        Positive = gained positions (good start)
        Negative = lost positions (bad start)
        
        Rule of thumb: ±0.1s per position changed
        
        Args:
            grid_position: Starting grid slot
            turn1_position: Position after Turn 1
        
        Returns:
            Time adjustment for Lap 1 (seconds)
        """
        positions_gained = grid_position - turn1_position
        return positions_gained * 0.1
    
    @staticmethod
    def estimate_dnf_probability(
        dnf_history: List[bool],
        total_races: int
    ) -> float:
        """
        Estimate per-lap DNF probability from historical data.
        
        Args:
            dnf_history: Boolean list of DNFs (True = DNF)
            total_races: Total races in sample
        
        Returns:
            Per-lap DNF probability
        """
        if total_races == 0:
            return 0.001  # Default
        
        dnf_count = sum(dnf_history)
        race_dnf_rate = dnf_count / total_races
        
        # Convert to per-lap (assuming ~50 lap average)
        per_lap_dnf = race_dnf_rate / 50.0
        
        # Reasonable bounds
        return np.clip(per_lap_dnf, 0.0005, 0.005)
    
    @classmethod
    def calibrate_driver(
        cls,
        session_data: SessionData,
        field_fp2_median: float,
        field_qualifying_median: float,
        field_avg_degradation: float,
        driver_id: int,
        pit_strategy: List[PitStop],
        stint_degradation_observed: float = None,
        dnf_history: List[bool] = None,
        start_positions: Tuple[int, int] = None
    ) -> Driver:
        """
        Full calibration: convert session data to Driver instance.
        
        Args:
            session_data: Real session data for driver
            field_fp2_median: Median FP2 time across field
            field_qualifying_median: Median qualifying time
            field_avg_degradation: Average tyre deg in field
            driver_id: Unique driver ID
            pit_strategy: Planned pit stops
            stint_degradation_observed: Driver's tyre wear (optional)
            dnf_history: Historical DNF data (optional)
            start_positions: (grid_pos, turn1_pos) for start bonus
        
        Returns:
            Calibrated Driver instance ready for simulation
        """
        # Calculate base race pace
        base_pace = cls.calculate_base_race_pace(
            session_data.fp2_best_time,
            session_data.qualifying_time,
            field_fp2_median,
            field_qualifying_median
        )
        
        # Calculate variability
        variability = cls.calculate_lap_time_variability(
            session_data.race_pace_samples
        )
        
        # Estimate tyre deg multiplier
        if stint_degradation_observed is not None:
            tyre_mult = cls.estimate_tyre_deg_multiplier(
                stint_degradation_observed,
                field_avg_degradation
            )
        else:
            tyre_mult = 1.0  # Default
        
        # Estimate start bonus
        if start_positions is not None:
            start_bonus = cls.estimate_start_bonus(
                start_positions[0],
                start_positions[1]
            )
        else:
            start_bonus = 0.0
        
        # Estimate DNF probability
        if dnf_history is not None:
            dnf_prob = cls.estimate_dnf_probability(
                dnf_history,
                len(dnf_history)
            )
        else:
            dnf_prob = 0.001  # Default
        
        return Driver(
            name=session_data.driver_name,
            driver_id=driver_id,
            qualifying_position=session_data.qualifying_position,
            start_bonus=start_bonus,
            maximum_speed=session_data.top_speed_kmh,
            base_race_pace=base_pace,
            lap_time_variability=variability,
            pit_strategy=pit_strategy,
            dnf_probability=dnf_prob,
            tyre_deg_multiplier=tyre_mult
        )


def example_calibration():
    """
    Example: Calibrate drivers from mock session data.
    """
    print("="*70)
    print("DRIVER CALIBRATION EXAMPLE")
    print("="*70)
    
    # Mock session data for 3 drivers
    verstappen_data = SessionData(
        driver_name="Verstappen",
        fp2_best_time=84.5,
        qualifying_time=83.8,
        qualifying_position=1,
        race_pace_samples=[85.1, 85.3, 85.0, 85.2, 85.4, 85.1, 85.0],
        top_speed_kmh=342.0
    )
    
    hamilton_data = SessionData(
        driver_name="Hamilton",
        fp2_best_time=84.7,
        qualifying_time=84.0,
        qualifying_position=2,
        race_pace_samples=[85.4, 85.6, 85.3, 85.7, 85.5, 85.4, 85.6],
        top_speed_kmh=340.0
    )
    
    leclerc_data = SessionData(
        driver_name="Leclerc",
        fp2_best_time=84.9,
        qualifying_time=84.2,
        qualifying_position=3,
        race_pace_samples=[85.6, 86.1, 85.4, 86.3, 85.7, 85.9, 86.0],
        top_speed_kmh=341.0
    )
    
    # Field statistics
    field_fp2_median = 85.5
    field_qualifying_median = 84.5
    field_avg_degradation = 3.5  # 3.5s degradation over 20-lap stint
    
    # Calibrate drivers
    calibrator = DriverCalibrator()
    
    verstappen = calibrator.calibrate_driver(
        verstappen_data,
        field_fp2_median,
        field_qualifying_median,
        field_avg_degradation,
        driver_id=1,
        pit_strategy=[PitStop(lap=25, compound=TyreCompound.HARD)],
        stint_degradation_observed=2.8,  # Good tyre management
        dnf_history=[False, False, False, False, False],  # Reliable
        start_positions=(1, 1)  # Maintained P1
    )
    
    hamilton = calibrator.calibrate_driver(
        hamilton_data,
        field_fp2_median,
        field_qualifying_median,
        field_avg_degradation,
        driver_id=2,
        pit_strategy=[PitStop(lap=23, compound=TyreCompound.HARD)],
        stint_degradation_observed=3.2,
        dnf_history=[False, False, False, True, False],
        start_positions=(2, 1)  # Good start, gained P1
    )
    
    leclerc = calibrator.calibrate_driver(
        leclerc_data,
        field_fp2_median,
        field_qualifying_median,
        field_avg_degradation,
        driver_id=3,
        pit_strategy=[
            PitStop(lap=18, compound=TyreCompound.MEDIUM),
            PitStop(lap=36, compound=TyreCompound.HARD)
        ],
        stint_degradation_observed=4.2,  # Harder on tyres
        dnf_history=[False, True, False, False, True],
        start_positions=(3, 3)  # Maintained position
    )
    
    # Print calibrated parameters
    print("\nCalibrated Driver Parameters:")
    print("-" * 70)
    
    for driver in [verstappen, hamilton, leclerc]:
        print(f"\n{driver.name}:")
        print(f"  Base Race Pace: {driver.base_race_pace:.3f}s")
        print(f"  Lap-Time Variability (σ): {driver.lap_time_variability:.3f}s")
        print(f"  Tyre Deg Multiplier: {driver.tyre_deg_multiplier:.2f}x")
        print(f"  Start Bonus: {driver.start_bonus:+.2f}s")
        print(f"  DNF Probability (per lap): {driver.dnf_probability:.5f}")
        print(f"  Maximum Speed: {driver.maximum_speed:.0f} km/h")
        print(f"  Pit Strategy: {len(driver.pit_strategy)}-stop")
    
    print("\n" + "="*70)
    print("✅ Calibration complete. Drivers ready for simulation.")
    print("="*70)
    
    return [verstappen, hamilton, leclerc]


if __name__ == "__main__":
    example_calibration()
