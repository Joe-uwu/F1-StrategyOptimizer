# src/data/track_config.py
"""
Track-specific configurations for F1 race simulation.
Includes pit loss times, tire degradation factors, and weather transition probabilities.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class TrackConfig:
    """Configuration for a specific F1 circuit."""
    name: str
    pit_loss_time: float  # seconds
    track_length_km: float
    lap_count: int
    degradation_factor: float  # multiplier on tire wear
    weather_dry_prob: float  # P(Dry -> Dry)
    weather_inter_prob: float  # P(Inter -> Inter)
    baseline_lap_time: float  # seconds (dry, fresh tires, nominal fuel)


# Historical baseline lap times (median qualifying times, race trim)
TRACKS_DB: Dict[str, TrackConfig] = {
    "Monza": TrackConfig(
        name="Monza",
        pit_loss_time=21.5,  # Very fast pit stop lane
        track_length_km=5.793,
        lap_count=53,
        degradation_factor=0.9,  # Lower tire wear: high downforce, smooth
        weather_dry_prob=0.85,
        weather_inter_prob=0.70,
        baseline_lap_time=133.0,
    ),
    "Monaco": TrackConfig(
        name="Monaco",
        pit_loss_time=27.3,  # Long pit lane
        track_length_km=3.337,
        lap_count=78,
        degradation_factor=1.2,  # Higher tire wear: high brake usage
        weather_dry_prob=0.80,
        weather_inter_prob=0.65,
        baseline_lap_time=80.0,
    ),
    "Silverstone": TrackConfig(
        name="Silverstone",
        pit_loss_time=23.0,
        track_length_km=5.891,
        lap_count=52,
        degradation_factor=1.0,
        weather_dry_prob=0.70,  # UK weather: volatile
        weather_inter_prob=0.75,
        baseline_lap_time=128.0,
    ),
    "Suzuka": TrackConfig(
        name="Suzuka",
        pit_loss_time=25.0,
        track_length_km=5.807,
        lap_count=53,
        degradation_factor=1.15,  # High speed → high tire wear
        weather_dry_prob=0.75,
        weather_inter_prob=0.70,
        baseline_lap_time=131.0,
    ),
}
