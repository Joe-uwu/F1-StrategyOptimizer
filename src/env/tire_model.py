# src/env/tire_model.py
"""
Non-linear tire degradation model for F1 racing.
Models how tire compound affects lap time based on age and fuel load.
"""
import numpy as np
from typing import Literal


class TireModel:
    """Physics-based tire degradation model."""
    
    # Degradation curves per compound (lap age vs. time loss, seconds)
    # Steep non-linear degradation: tires must be changed regularly
    DEGRADATION_CURVES = {
        "soft": {
            "peak_performance_laps": 3,
            "usable_laps": 12,
            "degradation_rate": 0.35,  # Very steep degradation
            "cliff_laps": 18,
        },
        "medium": {
            "peak_performance_laps": 8,
            "usable_laps": 22,
            "degradation_rate": 0.22,
            "cliff_laps": 28,
        },
        "hard": {
            "peak_performance_laps": 12,
            "usable_laps": 32,
            "degradation_rate": 0.15,
            "cliff_laps": 40,
        },
    }
    
    @staticmethod
    def calculate_lap_time_delta(
        lap_age: int,
        compound: Literal["soft", "medium", "hard"],
        degradation_multiplier: float = 1.0,
    ) -> float:
        """
        Calculate lap time increase due to tire wear.
        
        Model: L_t = L_base + tire_deg(age)
        
        Args:
            lap_age: Number of laps on current tires (0 = fresh)
            compound: Tire compound name
            degradation_multiplier: Track-specific multiplier (e.g., Monza=0.9, Monaco=1.2)
        
        Returns:
            Lap time delta in seconds (positive = slower)
        """
        if compound not in TireModel.DEGRADATION_CURVES:
            raise ValueError(f"Unknown compound: {compound}")
        
        params = TireModel.DEGRADATION_CURVES[compound]
        peak_laps = params["peak_performance_laps"]
        rate = params["degradation_rate"]
        
        # Tire is fastest (no penalty) for first N laps
        if lap_age < peak_laps:
            return 0.0
        
        # After peak: steep non-linear degradation
        age_past_peak = lap_age - peak_laps
        delta = (age_past_peak ** 2.0) * rate * degradation_multiplier
        
        # Beyond cliff: extreme penalty (risk of blowout/significant grip loss)
        cliff_laps = params.get("cliff_laps", params["usable_laps"] + 10)
        if lap_age > cliff_laps:
            delta += (lap_age - cliff_laps) * 0.5  # Additional 0.5s per lap
        
        return delta
    
    @staticmethod
    def estimate_remaining_life(
        lap_age: int,
        compound: Literal["soft", "medium", "hard"],
        max_acceptable_delta: float = 1.5,  # s/lap
        degradation_multiplier: float = 1.0,
    ) -> int:
        """
        Estimate remaining useful laps before tire becomes too slow.
        
        Args:
            lap_age: Current tire age
            compound: Tire compound
            max_acceptable_delta: Max acceptable time loss per lap (s)
            degradation_multiplier: Track multiplier
        
        Returns:
            Remaining usable laps (0 if past limit)
        """
        usable = TireModel.DEGRADATION_CURVES[compound]["usable_laps"]
        remaining = max(0, usable - lap_age)
        
        # Check if current delta exceeds threshold
        current_delta = TireModel.calculate_lap_time_delta(
            lap_age, compound, degradation_multiplier
        )
        if current_delta > max_acceptable_delta:
            remaining = 0
        
        return remaining
