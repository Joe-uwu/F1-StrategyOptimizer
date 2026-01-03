# src/agent/ppo_agent.py
"""
PPO (Proximal Policy Optimization) agent for F1 strategy.
Wraps Stable Baselines3 PPO with race-specific utilities.
"""
import os
import logging
from typing import Optional, Tuple
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import Env

logger = logging.getLogger(__name__)


class RaceMetricsCallback(BaseCallback):
    """Custom callback to track race-specific metrics during training."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.lap_times = []
        self.pit_stop_counts = []
        self.avg_rewards = []
    
    def _on_step(self) -> bool:
        """Called after every step."""
        return True


class F1StrategyAgent:
    """Wrapper for PPO agent optimizing F1 race strategy."""
    
    def __init__(
        self,
        env: Env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        clip_range: float = 0.2,
        model_dir: str = "./outputs/models",
    ):
        """
        Initialize F1 strategy PPO agent.
        
        Args:
            env: Gymnasium environment
            learning_rate: Learning rate for optimizer
            n_steps: Steps per rollout
            batch_size: Batch size for SGD
            n_epochs: Number of epochs for training
            clip_range: PPO clip range
            model_dir: Directory to save models
        """
        self.env = env
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # PPO hyperparameters tuned for F1 domain
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            clip_range=clip_range,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=1,
        )
        
        self.callback = RaceMetricsCallback()
    
    def train(
        self,
        total_timesteps: int = 100_000,
        save_interval: int = 10_000,
    ) -> None:
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total training steps
            save_interval: Save checkpoint every N steps
        """
        logger.info(f"Starting training for {total_timesteps} steps...")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            log_interval=10,
        )
        
        # Save final model
        model_path = os.path.join(self.model_dir, "ppo_f1_final")
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get action from policy.
        
        Args:
            observation: Current observation
            deterministic: Use deterministic policy (greedy) if True
        
        Returns:
            action, state
        """
        action, state = self.model.predict(observation, deterministic=deterministic)
        return action, state
    
    def evaluate(self, n_episodes: int = 10) -> dict:
        """
        Evaluate agent over N episodes.
        
        Args:
            n_episodes: Number of evaluation episodes
        
        Returns:
            Dictionary with evaluation metrics
        """
        episode_rewards = []
        episode_lap_times = []
        episode_pit_stops = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            total_lap_time = 0.0
            pit_stops = 0
            
            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                total_lap_time += info.get("lap_time", 0.0)
                pit_stops = info.get("pit_stops", 0)
            
            episode_rewards.append(episode_reward)
            episode_lap_times.append(total_lap_time)
            episode_pit_stops.append(pit_stops)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_total_time": np.mean(episode_lap_times),
            "mean_pit_stops": np.mean(episode_pit_stops),
        }
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        self.model = PPO.load(path, env=self.env)
        logger.info(f"Model loaded from {path}")
