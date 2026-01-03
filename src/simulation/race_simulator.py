# src/simulation/race_simulator.py
"""
Race simulator that orchestrates environment, agent, and race progression.
Outputs strategy decisions and race telemetry.
"""
import logging
from typing import Dict, List, Tuple
import numpy as np
from src.env import F1RaceEnv
from src.agent import F1StrategyAgent

logger = logging.getLogger(__name__)


class RaceSimulator:
    """Orchestrate a single race or multiple race scenarios."""
    
    def __init__(self, track_name: str = "Monza"):
        self.track_name = track_name
        self.env = F1RaceEnv(track_name=track_name)
        self.agent = None
        self.race_history = []
    
    def set_agent(self, agent: F1StrategyAgent) -> None:
        """Set the trained agent for simulation."""
        self.agent = agent
    
    def simulate_race(
        self,
        deterministic: bool = True,
        render: bool = False,
    ) -> Dict:
        """
        Run a single race simulation.
        
        Args:
            deterministic: Use greedy policy if True
            render: Render environment (not implemented)
        
        Returns:
            Dictionary containing race telemetry and decisions
        """
        if self.agent is None:
            raise RuntimeError("Agent not set. Call set_agent() first.")
        
        obs, _ = self.env.reset()
        done = False
        
        race_data = {
            "track": self.track_name,
            "laps": [],
            "strategy_decisions": [],
            "total_race_time": 0.0,
            "pit_stops": 0,
        }
        
        while not done:
            # Get agent decision
            action, _ = self.agent.predict(obs, deterministic=deterministic)
            
            # Execute action
            obs, reward, done, truncated, info = self.env.step(action)
            
            # Log race data
            pit_flag, target_compound_id = action[0], action[1]
            
            lap_data = {
                "lap": info["lap"],
                "lap_time": info["lap_time"],
                "fuel_load": info["fuel_load"],
                "tire_age": info["tire_age"],
                "compound": info["compound"],
                "pit_stop": bool(pit_flag),
                "target_compound": self.env.COMPOUNDS[int(target_compound_id)],
                "weather": self.env.weather,
                "reward": reward,
            }
            race_data["laps"].append(lap_data)
            race_data["total_race_time"] = info["total_time"]
            race_data["pit_stops"] = info["pit_stops"]
        
        self.race_history.append(race_data)
        return race_data
    
    def simulate_scenario(
        self,
        n_races: int = 5,
        deterministic: bool = True,
    ) -> List[Dict]:
        """
        Run multiple races and aggregate statistics.
        
        Args:
            n_races: Number of races to simulate
            deterministic: Use deterministic policy
        
        Returns:
            List of race results
        """
        results = []
        for i in range(n_races):
            logger.info(f"Simulating race {i+1}/{n_races}...")
            race_result = self.simulate_race(deterministic=deterministic)
            results.append(race_result)
        
        return results
    
    def get_strategy_summary(self, race_data: Dict) -> Dict:
        """Extract strategy recommendations from a race."""
        pit_laps = [
            lap["lap"] for lap in race_data["laps"] if lap["pit_stop"]
        ]
        avg_lap_time = np.mean([lap["lap_time"] for lap in race_data["laps"]])
        
        return {
            "track": race_data["track"],
            "pit_laps": pit_laps,
            "pit_count": race_data["pit_stops"],
            "total_race_time": race_data["total_race_time"],
            "avg_lap_time": avg_lap_time,
        }
