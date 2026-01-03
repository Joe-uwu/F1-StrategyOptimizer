# configs/training_config.py
"""
Training configuration file.
Modify these settings to customize training behavior.
"""

# ============================================================================
# ENVIRONMENT SETTINGS
# ============================================================================

# Track selection
TRACK_NAME = "Monza"

# Fuel settings
INITIAL_FUEL_KG = 110.0
FUEL_CONSUMPTION_RATE = 2.0  # kg per lap

# Pit stop settings
MAX_PIT_STOPS = 3

# Random seed for reproducibility
SEED = 42

# ============================================================================
# TRAINING SETTINGS
# ============================================================================

# Total training timesteps
TRAINING_TIMESTEPS = 50_000

# PPO Hyperparameters
LEARNING_RATE = 3e-4
N_STEPS = 2048  # Steps per rollout
BATCH_SIZE = 64
N_EPOCHS = 10
CLIP_RANGE = 0.2

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Tracks to evaluate on
EVAL_TRACKS = ["Monza", "Monaco", "Silverstone"]

# Number of evaluation episodes per track
EVAL_EPISODES = 3

# Model path
MODEL_PATH = "./outputs/models/ppo_f1_final.zip"

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Output directory for models and visualizations
OUTPUT_DIR = "./outputs"

# Enable logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# ============================================================================
# ENVIRONMENT PARAMETERS (Advanced)
# ============================================================================

# Reward function weights (modify to change agent incentives)
REWARD_WEIGHTS = {
    "lap_time": 1.0,  # Primary: minimize lap time
    "pit_stop_penalty": -5.0,  # Penalize frequent stops
    "tire_freshness_bonus": 2.0,  # Reward fresh tires
    "tire_bald_penalty": -10.0,  # Penalize bald tires
    "fuel_excess_penalty": -1.0,  # Penalize carrying excess fuel
    "fuel_critical_penalty": -5.0,  # Penalize critical fuel levels
}

# Weather transition probabilities
WEATHER_TRANSITIONS = {
    "dry": {"dry": 0.85, "intermediate": 0.10, "wet": 0.05},
    "intermediate": {"dry": 0.15, "intermediate": 0.70, "wet": 0.15},
    "wet": {"dry": 0.05, "intermediate": 0.20, "wet": 0.75},
}

# ============================================================================
# NOTES
# ============================================================================
"""
Tuning Tips:

1. LEARNING_RATE:
   - Increase (e.g., 1e-3) for faster learning (may be unstable)
   - Decrease (e.g., 1e-4) for more stable, slower learning

2. N_STEPS:
   - Larger (4096) = less frequent updates, more stability
   - Smaller (1024) = more frequent updates, faster learning

3. BATCH_SIZE:
   - Must divide evenly into N_STEPS
   - Larger (128) = fewer updates per epoch, less compute
   - Smaller (32) = more updates per epoch, more compute

4. N_EPOCHS:
   - Higher = more gradient updates per rollout
   - Useful if experience is precious (small environment)
   - Lower = less compute per step

5. For faster initial convergence:
   - Increase LEARNING_RATE to 1e-3
   - Increase BATCH_SIZE to 128
   - Decrease N_EPOCHS to 5

6. For stable long-term learning:
   - Keep LEARNING_RATE at 3e-4
   - Keep BATCH_SIZE at 64
   - Keep N_EPOCHS at 10
"""
