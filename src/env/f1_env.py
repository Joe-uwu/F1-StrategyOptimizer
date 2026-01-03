# src/env/f1_env.py
"""
F1 Race Strategy Gymnasium Environment.
State space: (current_lap, tire_age, compound, fuel_load, weather, position, track_state)
Action space: (pit=0/1, target_compound if pit)
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Any, Literal
import logging

from src.data.track_config import TrackConfig, TRACKS_DB

# Use a unified observation space upper bound across all tracks to allow a single
# trained policy to run on any circuit without space mismatches.
MAX_LAPS = max(cfg.lap_count for cfg in TRACKS_DB.values())
from src.data.telemetry_loader import TelemetryLoader
from .tire_model import TireModel

logger = logging.getLogger(__name__)


class F1RaceEnv(gym.Env):
    """Gymnasium environment for F1 race strategy optimization."""
    
    metadata = {"render_modes": []}
    
    # Tire compounds available
    COMPOUNDS = ["soft", "medium", "hard"]
    WEATHER_STATES = ["dry", "intermediate", "wet"]
    
    def __init__(
        self,
        track_name: str = "Monza",
        initial_fuel_load_kg: float = 130.0,
        fuel_consumption_rate: float = 2.0,  # kg/lap
        max_pit_stops: int = 3,
        min_stint_laps: int = 5,
        seed: int = None,
    ):
        """
        Initialize F1 race environment.
        
        Args:
            track_name: Track name (key in TRACKS_DB)
            initial_fuel_load_kg: Starting fuel load
            fuel_consumption_rate: Fuel consumed per lap (kg)
            max_pit_stops: Maximum allowed pit stops
            min_stint_laps: Minimum laps between pit stops (stint length)
            seed: Random seed
        """
        super().__init__()
        
        if track_name not in TRACKS_DB:
            raise ValueError(f"Unknown track: {track_name}")
        
        self.track: TrackConfig = TRACKS_DB[track_name]
        self.telemetry_loader = TelemetryLoader(use_cache=False)
        
        self.initial_fuel = initial_fuel_load_kg
        self.fuel_consumption = fuel_consumption_rate
        self.max_pit_stops = max_pit_stops
        self.min_stint_laps = min_stint_laps
        self.last_pit_lap = -999
        
        # Get baseline lap time from telemetry or synthetic
        self.baseline_lap_time = self.telemetry_loader.get_baseline_lap_time(
            2023, track_name, "soft"
        )
        
        # State variables
        self.current_lap = 0
        self.fuel_load = initial_fuel_load_kg
        self.tire_age = 0
        self.compound = "soft"
        self.weather = "dry"
        self.position = 1  # Starting position (1 = P1)
        self.pit_stops = 0
        self.total_time = 0.0
        
        # State space: [lap, tire_age, fuel_load, position, weather_id, compound_id]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 1, 0, 0], dtype=np.float32),
            high=np.array([
                MAX_LAPS,  # unified upper bound for all tracks
                50,        # Max tire age
                initial_fuel_load_kg * 1.2,
                20,        # Max position
                len(self.WEATHER_STATES) - 1,
                len(self.COMPOUNDS) - 1,
            ], dtype=np.float32),
        )
        
        # Action space: (pit_now, target_compound if pit)
        # pit_now: 0 = stay out, 1 = pit
        # target_compound: 0 = soft, 1 = medium, 2 = hard
        self.action_space = spaces.MultiDiscrete([2, 3])
        
        self.np_random = np.random.RandomState(seed)
        self.seed(seed)
    
    def seed(self, seed: int = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to race start."""
        if seed is not None:
            self.seed(seed)
        
        self.current_lap = 0
        self.fuel_load = self.initial_fuel
        self.tire_age = 0
        self.compound = "soft"
        self.weather = "dry"
        self.position = 1
        self.pit_stops = 0
        self.total_time = 0.0
        self.last_pit_lap = -999
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Return current state observation."""
        weather_id = self.WEATHER_STATES.index(self.weather)
        compound_id = self.COMPOUNDS.index(self.compound)
        
        return np.array([
            self.current_lap,
            self.tire_age,
            self.fuel_load,
            self.position,
            weather_id,
            compound_id,
        ], dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one racing lap.
        
        Args:
            action: [pit_flag (0/1), target_compound (0/1/2)]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        pit_flag, target_compound_id = action[0], action[1]
        target_compound = self.COMPOUNDS[int(target_compound_id)]
        
        pit_stop_this_lap = bool(pit_flag)
        lap_time = 0.0
        illegal_pit = False
        
        # === PIT LOGIC ===
        if pit_stop_this_lap:
            illegal_pit = False
            if self.pit_stops >= self.max_pit_stops or (self.current_lap - self.last_pit_lap) < self.min_stint_laps:
                # Illegal or premature pit stop: apply heavy penalty but continue
                lap_time += self.track.pit_loss_time + 10.0
                illegal_pit = True
            else:
                # Complete pit stop
                pit_time = self._simulate_pit_stop(target_compound)
                lap_time += pit_time
                self.tire_age = 0
                self.compound = target_compound
                self.pit_stops += 1
                self.last_pit_lap = self.current_lap
        
        # === RACING LAP ===
        # Calculate lap time
        lap_time += self._calculate_lap_time()
        
        # Fuel consumption
        self.fuel_load = max(0, self.fuel_load - self.fuel_consumption)
        
        # Tire aging
        if not pit_stop_this_lap:
            self.tire_age += 1
        
        # Weather transition (stochastic)
        self._update_weather()
        
        # Position update (simplified: reward based on fuel/tire efficiency)
        # In a full system, this would model multi-agent interactions
        
        self.current_lap += 1
        self.total_time += lap_time
        
        # === TERMINATION CHECK ===
        done = self.current_lap >= self.track.lap_count
        
        # === REWARD ===
        reward = self._calculate_reward(lap_time, pit_stop_this_lap)
        
        # Extra penalty for illegal or premature pit
        if illegal_pit:
            reward -= 100.0
        
        # Out of fuel penalty
        if self.fuel_load <= 0 and self.current_lap < self.track.lap_count:
            reward -= 500.0
            done = True
        
        info = {
            "lap": self.current_lap,
            "lap_time": lap_time,
            "fuel_load": self.fuel_load,
            "tire_age": self.tire_age,
            "compound": self.compound,
            "pit_stop": pit_stop_this_lap,
            "total_time": self.total_time,
            "pit_stops": self.pit_stops,
            "illegal_pit": illegal_pit,
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _calculate_lap_time(self) -> float:
        """
        Calculate lap time for current state.
        L_t = L_base + fuel_effect + tire_degradation + weather_penalty
        """
        # Baseline
        lap_time = self.baseline_lap_time
        
        # Fuel effect
        fuel_delta = TelemetryLoader.get_fuel_effect(self.fuel_load, self.baseline_lap_time)
        lap_time += fuel_delta
        
        # Tire degradation
        tire_delta = TireModel.calculate_lap_time_delta(
            self.tire_age,
            self.compound,
            self.track.degradation_factor,
        )
        lap_time += tire_delta
        
        # Weather penalty
        weather_delta = self._get_weather_penalty()
        lap_time += weather_delta
        
        # Safety margin (random variation ~±0.3s)
        variation = self.np_random.normal(0, 0.3)
        lap_time += variation
        
        return max(10.0, lap_time)  # Minimum lap time floor
    
    def _simulate_pit_stop(self, target_compound: str) -> float:
        """
        Simulate pit stop duration (modern F1: no refueling).
        Time = pit_loss_time + tire_change_time
        Note: In modern F1 (since 2010), refueling is banned. Cars start with enough fuel for the race.
        """
        # Pit stop time (fixed per track, includes all pit lane overhead)
        pit_time = self.track.pit_loss_time
        
        # Tire change (typical ~2.5s for modern F1)
        tire_change_time = 2.5  # Seconds
        
        # No refueling in modern F1 rules
        total_pit_time = pit_time + tire_change_time
        return total_pit_time
    
    def _update_weather(self) -> None:
        """Update weather stochastically."""
        weather_transitions = {
            "dry": {"dry": 0.85, "intermediate": 0.10, "wet": 0.05},
            "intermediate": {"dry": 0.15, "intermediate": 0.70, "wet": 0.15},
            "wet": {"dry": 0.05, "intermediate": 0.20, "wet": 0.75},
        }
        
        current_probs = weather_transitions.get(self.weather, {"dry": 1.0})
        self.weather = self.np_random.choice(
            list(current_probs.keys()),
            p=list(current_probs.values()),
        )
    
    def _get_weather_penalty(self) -> float:
        """Time loss due to weather (relative to dry baseline)."""
        penalties = {
            "dry": 0.0,
            "intermediate": 2.5,  # Wet tires on dry = slow; inter on wet = okay
            "wet": 3.0,
        }
        return penalties.get(self.weather, 0.0)
    
    def _calculate_reward(self, lap_time: float, pit_stop: bool) -> float:
        """
        Reward function incentivizes:
        - Fast lap times
        - Strategic pit stop timing (necessary due to steep tire wear)
        - Tire management
        """
        # Base reward: negative of lap time (minimize)
        reward = -lap_time
        
        # Pit stop cost (but usually necessary)
        if pit_stop:
            reward -= 2.0  # Small penalty; pitting is strategically necessary
        
        # STRONG tire wear penalties (main incentive to pit)
        tire_delta = TireModel.calculate_lap_time_delta(
            self.tire_age,
            self.compound,
            degradation_multiplier=self.track.degradation_factor,
        )
        
        # Severe penalty for worn tires
        if tire_delta > 3.0:
            reward -= 50.0  # Extreme penalty - must pit soon
        elif tire_delta > 2.0:
            reward -= 30.0  # Very severe penalty
        elif tire_delta > 1.0:
            reward -= 15.0  # Substantial penalty
        elif tire_delta > 0.5:
            reward -= 5.0  # Moderate penalty
        
        # Bonus for fresh tires
        if self.tire_age < 2:
            reward += 5.0
        
        # Critical fuel penalty (shouldn't happen)
        if self.fuel_load < self.initial_fuel * 0.05:
            reward -= 100.0
        
        return reward
    
    def render(self, mode: str = "human") -> None:
        """Render environment state (not implemented for simplicity)."""
        pass
    
    def close(self) -> None:
        """Close environment."""
        pass
