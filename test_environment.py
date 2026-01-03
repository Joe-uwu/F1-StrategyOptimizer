# test_environment.py
"""
Quick test to verify environment and agent work correctly.
"""
import logging
import numpy as np
from src.env import F1RaceEnv
from src.agent import F1StrategyAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_environment():
    """Test F1 environment basic functionality."""
    logger.info("Testing F1RaceEnv...")
    
    env = F1RaceEnv(track_name="Monza", seed=42)
    obs, info = env.reset()
    
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
    logger.info(f"✓ Initial observation shape: {obs.shape}")
    
    # Take a few random steps
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        
        logger.info(f"  Step {step+1}: Lap {info['lap']}, "
                   f"Lap time: {info['lap_time']:.2f}s, "
                   f"Fuel: {info['fuel_load']:.1f}kg")
        
        if done:
            break
    
    logger.info("✓ Environment test passed!")
    env.close()


def test_agent():
    """Test PPO agent creation and prediction."""
    logger.info("\nTesting F1StrategyAgent...")
    
    env = F1RaceEnv(track_name="Monza", seed=42)
    agent = F1StrategyAgent(env=env, learning_rate=3e-4)
    
    obs, _ = env.reset()
    action, state = agent.predict(obs, deterministic=True)
    
    assert action.shape == env.action_space.shape
    logger.info(f"✓ Agent prediction shape: {action.shape}")
    logger.info(f"✓ Predicted action: {action} (pit: {action[0]}, compound: {action[1]})")
    
    logger.info("✓ Agent test passed!")
    env.close()


def test_full_integration():
    """Test environment + agent together."""
    logger.info("\nTesting full integration (Env + Agent)...")
    
    env = F1RaceEnv(track_name="Monza", seed=42)
    agent = F1StrategyAgent(env=env)
    
    obs, _ = env.reset()
    total_reward = 0.0
    
    for step in range(10):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    logger.info(f"✓ Completed {step+1} steps")
    logger.info(f"✓ Total reward: {total_reward:.2f}")
    logger.info("✓ Integration test passed!")
    env.close()


if __name__ == "__main__":
    try:
        test_environment()
        test_agent()
        test_full_integration()
        logger.info("\n" + "=" * 50)
        logger.info("ALL TESTS PASSED!")
        logger.info("Environment and agent are working correctly.")
        logger.info("=" * 50)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        exit(1)
