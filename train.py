# train.py
"""
Main training script for F1 Strategy Optimizer.
Trains PPO agent on F1 race environment.
"""
import logging
import sys
from src.env import F1RaceEnv
from src.agent import F1StrategyAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Train the F1 strategy PPO agent."""
    logger.info("=" * 60)
    logger.info("F1 RACE STRATEGY OPTIMIZER - TRAINING")
    logger.info("=" * 60)
    
    # === CONFIGURATION ===
    TRACK = "Monza"
    INITIAL_FUEL = 110.0
    TRAINING_STEPS = 50_000  # Can increase to 100k for production
    
    logger.info(f"Track: {TRACK}")
    logger.info(f"Initial Fuel: {INITIAL_FUEL} kg")
    logger.info(f"Training Steps: {TRAINING_STEPS:,}")
    
    # === CREATE ENVIRONMENT ===
    logger.info("Initializing F1 race environment...")
    env = F1RaceEnv(
        track_name=TRACK,
        initial_fuel_load_kg=INITIAL_FUEL,
        fuel_consumption_rate=2.0,
        max_pit_stops=3,
        seed=42,
    )
    logger.info(f"Environment created: {TRACK} ({env.track.lap_count} laps)")
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    
    # === CREATE AGENT ===
    logger.info("Initializing PPO agent...")
    agent = F1StrategyAgent(
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
        model_dir="./outputs/models",
    )
    logger.info("PPO agent initialized with tuned hyperparameters")
    
    # === TRAINING ===
    logger.info("Starting training...")
    try:
        agent.train(total_timesteps=TRAINING_STEPS, save_interval=10_000)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
    
    # === EVALUATION ===
    logger.info("Evaluating trained agent...")
    eval_metrics = agent.evaluate(n_episodes=5)
    logger.info("Evaluation Results:")
    logger.info(f"  Mean Reward: {eval_metrics['mean_reward']:.2f}")
    logger.info(f"  Std Reward: {eval_metrics['std_reward']:.2f}")
    logger.info(f"  Mean Total Time: {eval_metrics['mean_total_time']:.1f}s")
    logger.info(f"  Mean Pit Stops: {eval_metrics['mean_pit_stops']:.1f}")
    
    logger.info("=" * 60)
    logger.info("Training and evaluation complete!")
    logger.info("Next step: Run evaluate.py for detailed strategy analysis")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
