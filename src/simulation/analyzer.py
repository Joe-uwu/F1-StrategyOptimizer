# src/simulation/analyzer.py
"""
Strategy analysis and visualization.
Generates plots for tire strategy, fuel management, and lap time progression.
"""
import logging
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class StrategyAnalyzer:
    """Analyze and visualize F1 race strategies."""
    
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = output_dir
        sns.set_style("darkgrid")
    
    def plot_race_telemetry(self, race_data: Dict, save_path: str = None) -> None:
        """
        Plot comprehensive race telemetry.
        
        Subplots:
        - Lap time progression
        - Tire age over race
        - Fuel load over race
        - Compound usage
        """
        laps = race_data["laps"]
        lap_nums = [lap["lap"] for lap in laps]
        lap_times = [lap["lap_time"] for lap in laps]
        tire_ages = [lap["tire_age"] for lap in laps]
        fuel_loads = [lap["fuel_load"] for lap in laps]
        compounds = [lap["compound"] for lap in laps]
        pit_stops = [i for i, lap in enumerate(laps) if lap["pit_stop"]]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Race Telemetry: {race_data['track']}", fontsize=16, fontweight="bold")
        
        # Plot 1: Lap times
        ax = axes[0, 0]
        ax.plot(lap_nums, lap_times, linewidth=2, color="blue", label="Lap Time")
        if pit_stops:
            for pit_lap in pit_stops:
                ax.axvline(pit_lap, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Lap")
        ax.set_ylabel("Lap Time (s)")
        ax.set_title("Lap Time Progression")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Tire age
        ax = axes[0, 1]
        colors = {"soft": "red", "medium": "yellow", "hard": "gray"}
        for i, (lap_num, age, compound) in enumerate(zip(lap_nums, tire_ages, compounds)):
            ax.scatter(lap_num, age, color=colors.get(compound, "blue"), s=50, alpha=0.7)
        ax.set_xlabel("Lap")
        ax.set_ylabel("Tire Age (laps)")
        ax.set_title("Tire Age & Compound (Red=Soft, Yellow=Medium, Gray=Hard)")
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Fuel load
        ax = axes[1, 0]
        ax.plot(lap_nums, fuel_loads, linewidth=2, color="green", label="Fuel Load")
        if pit_stops:
            for pit_lap in pit_stops:
                ax.axvline(pit_lap, color="red", linestyle="--", alpha=0.5, label="Pit Stop")
        ax.set_xlabel("Lap")
        ax.set_ylabel("Fuel Load (kg)")
        ax.set_title("Fuel Management")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Summary stats
        ax = axes[1, 1]
        ax.axis("off")
        stats_text = f"""
        Total Race Time: {race_data['total_race_time']:.1f}s
        Pit Stops: {race_data['pit_stops']}
        Avg Lap Time: {np.mean(lap_times):.2f}s
        Min Lap Time: {np.min(lap_times):.2f}s
        Max Lap Time: {np.max(lap_times):.2f}s
        """
        ax.text(0.1, 0.5, stats_text, fontsize=12, family="monospace",
                verticalalignment="center")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_strategy_comparison(
        self,
        race_data_list: List[Dict],
        save_path: str = None,
    ) -> None:
        """
        Compare strategies across multiple race simulations.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Strategy Comparison Across Scenarios", fontsize=16, fontweight="bold")
        
        # Extract metrics
        race_times = [race["total_race_time"] for race in race_data_list]
        pit_counts = [race["pit_stops"] for race in race_data_list]
        scenario_names = [f"Scenario {i+1}" for i in range(len(race_data_list))]
        
        # Plot race times
        ax = axes[0]
        ax.bar(scenario_names, race_times, color="steelblue", alpha=0.8)
        ax.set_ylabel("Total Race Time (s)")
        ax.set_title("Race Time Comparison")
        ax.grid(True, alpha=0.3, axis="y")
        for i, (name, time) in enumerate(zip(scenario_names, race_times)):
            ax.text(i, time + 5, f"{time:.0f}s", ha="center", fontsize=10)
        
        # Plot pit stop counts
        ax = axes[1]
        ax.bar(scenario_names, pit_counts, color="coral", alpha=0.8)
        ax.set_ylabel("Pit Stops")
        ax.set_title("Pit Stop Strategy")
        ax.set_ylim(0, max(pit_counts) + 1)
        ax.grid(True, alpha=0.3, axis="y")
        for i, (name, pit) in enumerate(zip(scenario_names, pit_counts)):
            ax.text(i, pit + 0.1, f"{pit}", ha="center", fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Comparison plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_tire_degradation_model(self, save_path: str = None) -> None:
        """
        Visualize the tire degradation model.
        """
        from src.env.tire_model import TireModel
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Tire Degradation Models", fontsize=16, fontweight="bold")
        
        compounds = ["soft", "medium", "hard"]
        colors = {"soft": "red", "medium": "yellow", "hard": "gray"}
        
        # Plot 1: Absolute degradation curves
        ax = axes[0]
        for compound in compounds:
            lap_ages = range(0, 46)
            deltas = [
                TireModel.calculate_lap_time_delta(age, compound, degradation_multiplier=1.0)
                for age in lap_ages
            ]
            ax.plot(lap_ages, deltas, linewidth=2.5, label=compound.capitalize(),
                   color=colors[compound])
        ax.set_xlabel("Tire Age (laps)")
        ax.set_ylabel("Lap Time Delta (s)")
        ax.set_title("Tire Wear Impact (Neutral Track)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Remaining useful life
        ax = axes[1]
        for compound in compounds:
            lap_ages = range(0, 46)
            remaining = [
                TireModel.estimate_remaining_life(age, compound, degradation_multiplier=1.0)
                for age in lap_ages
            ]
            ax.plot(lap_ages, remaining, linewidth=2.5, label=compound.capitalize(),
                   color=colors[compound])
        ax.set_xlabel("Tire Age (laps)")
        ax.set_ylabel("Remaining Useful Life (laps)")
        ax.set_title("Remaining Usable Laps")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Tire degradation plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
