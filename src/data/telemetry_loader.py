# src/data/telemetry_loader.py
"""
Telemetry data loader using FastF1.
Fetches real race data and caches it for environment calibration.
"""
import logging
from typing import Optional, Dict, Tuple
import numpy as np

try:
    import fastf1
except ImportError:
    fastf1 = None

logger = logging.getLogger(__name__)


class FastF1DataCache:
    """Cache for FastF1 session data."""
    
    def __init__(self):
        self._cache: Dict[str, any] = {}
    
    def get_session(self, year: int, circuit: str) -> Optional[object]:
        """Fetch and cache a FastF1 session."""
        key = f"{year}_{circuit}"
        if key not in self._cache:
            try:
                if fastf1 is None:
                    logger.warning("FastF1 not installed. Using synthetic data.")
                    return None
                session = fastf1.get_session(year, circuit, "R")
                session.load()
                self._cache[key] = session
            except Exception as e:
                logger.error(f"Failed to load FastF1 session {year} {circuit}: {e}")
                return None
        return self._cache.get(key)


class TelemetryLoader:
    """Load and process telemetry data from FastF1 or synthetic sources."""
    
    def __init__(self, use_cache: bool = True):
        self.cache = FastF1DataCache() if use_cache else None
    
    def get_baseline_lap_time(
        self, 
        year: int, 
        circuit: str, 
        compound: str = "soft"
    ) -> float:
        """
        Fetch real baseline lap time from FastF1.
        Falls back to synthetic data if unavailable.
        
        Args:
            year: Season year
            circuit: Circuit name (e.g., 'monza')
            compound: Tire compound ('soft', 'medium', 'hard')
        
        Returns:
            Lap time in seconds
        """
        if self.cache is None:
            return self._synthetic_baseline(circuit, compound)
        
        session = self.cache.get_session(year, circuit)
        if session is None:
            return self._synthetic_baseline(circuit, compound)
        
        try:
            laps = session.laps
            # Filter for qualifying-like lap times (fastest lap per driver)
            fastest_laps = laps.groupby("Driver")["LapTime"].min()
            baseline = fastest_laps.median().total_seconds()
            return baseline
        except Exception as e:
            logger.error(f"Error processing FastF1 data: {e}")
            return self._synthetic_baseline(circuit, compound)
    
    @staticmethod
    def _synthetic_baseline(circuit: str, compound: str) -> float:
        """
        Synthetic baseline lap times (fallback).
        Compound modifier: soft is fastest, hard is ~2-3s slower.
        """
        baselines = {
            "monza": 133.0,
            "monaco": 80.0,
            "silverstone": 128.0,
            "suzuka": 131.0,
        }
        compound_penalty = {"soft": 0.0, "medium": 0.8, "hard": 2.0}
        
        base = baselines.get(circuit.lower(), 130.0)
        penalty = compound_penalty.get(compound.lower(), 1.0)
        return base + penalty
    
    @staticmethod
    def get_fuel_effect(fuel_load_kg: float, baseline_lap_time: float) -> float:
        """
        Model fuel load impact on lap time.
        Approximately 0.035s per kg of fuel.
        
        Args:
            fuel_load_kg: Current fuel load in kg
            baseline_lap_time: Baseline lap time at zero fuel
        
        Returns:
            Lap time delta (seconds)
        """
        # Conservative estimate: 0.035s per kg
        return fuel_load_kg * 0.035
