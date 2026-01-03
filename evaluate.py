# evaluate.py
"""
Evaluate trained agent and generate strategy visualizations.
"""
import logging
import sys
import os
from src.env import F1RaceEnv
from src.agent import F1StrategyAgent
from src.simulation import RaceSimulator, StrategyAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Evaluate agent on multiple tracks and generate visualizations."""
    logger.info("=" * 60)
    logger.info("F1 RACE STRATEGY OPTIMIZER - EVALUATION")
    logger.info("=" * 60)
    
    # === CONFIGURATION ===
    TRACKS = ["Monza", "Monaco", "Silverstone"]
    MODEL_PATH = "./outputs/models/ppo_f1_final.zip"
    OUTPUT_DIR = "./outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        logger.error("Please run train.py first.")
        sys.exit(1)
    
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Evaluation tracks: {TRACKS}")
    
    # === TIRE DEGRADATION VISUALIZATION ===
    logger.info("Generating tire degradation model visualization...")
    analyzer = StrategyAnalyzer(output_dir=OUTPUT_DIR)
    analyzer.plot_tire_degradation_model(
        save_path=os.path.join(OUTPUT_DIR, "tire_degradation_model.png")
    )
    
    # === PER-TRACK EVALUATION ===
    for track in TRACKS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating on {track}...")
        logger.info("=" * 50)
        
        try:
            # Create environment
            env = F1RaceEnv(track_name=track, seed=42)
            
            # Load trained agent
            agent = F1StrategyAgent(env=env, model_dir=OUTPUT_DIR)
            agent.load(MODEL_PATH)
            logger.info(f"Loaded model for {track}")
            
            # Run race simulations
            simulator = RaceSimulator(track_name=track)
            simulator.set_agent(agent)
            
            logger.info(f"Simulating 3 races on {track}...")
            race_results = simulator.simulate_scenario(n_races=3, deterministic=True)
            
            # Generate visualizations
            for i, race_data in enumerate(race_results):
                plot_path = os.path.join(
                    OUTPUT_DIR,
                    f"race_telemetry_{track}_scenario{i+1}.png"
                )
                analyzer.plot_race_telemetry(race_data, save_path=plot_path)
                
                # Print strategy summary
                summary = simulator.get_strategy_summary(race_data)
                logger.info(f"\nScenario {i+1} Summary:")
                logger.info(f"  Pit Laps: {summary['pit_laps']}")
                logger.info(f"  Pit Stops: {summary['pit_count']}")
                logger.info(f"  Total Race Time: {summary['total_race_time']:.1f}s")
                logger.info(f"  Avg Lap Time: {summary['avg_lap_time']:.2f}s")
            
            # Comparison plot
            comparison_path = os.path.join(OUTPUT_DIR, f"strategy_comparison_{track}.png")
            analyzer.plot_strategy_comparison(race_results, save_path=comparison_path)
        
        except Exception as e:
            logger.error(f"Evaluation failed for {track}: {e}", exc_info=True)
            continue
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation and visualization complete!")
    logger.info(f"Results saved to {OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
