# F1 Strategy Optimizer

> **Elite Formula 1 Race Strategy Optimizer** using Deep Reinforcement Learning (PPO) and F1Metrics Monte Carlo Simulation  
> _A Virtual Race Strategist for optimal pit-stop and tire compound decisions_

---

## 🚀 Quick Start

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify setup
python test_environment.py
```

### Train RL Agent

```bash
# 3. Train and evaluate
python train.py
python evaluate.py
```

### Run F1Metrics Monte Carlo Simulator

```bash
# 4. Run realistic race simulation (NEW!)
python run_f1metrics_demo.py
```

---

## 🎯 What It Does

This project provides **two powerful approaches** for F1 strategy optimization:

### 1. Deep RL Agent (PPO)

- **Input**: Current race state (lap, tires, fuel, weather)
- **Agent**: PPO (Reinforcement Learning)
- **Output**: Real-time pit decisions + compound selection

### 2. F1Metrics Monte Carlo Simulator (NEW!)

- **Realistic lap-by-lap race simulation**
- **8-parameter driver model** based on F1Metrics methodology
- **Monte Carlo analysis** with 1000+ simulations for probability distributions
- **Strategy optimization** via probabilistic outcome analysis

---

## 🏎️ Key Features

### RL Environment

✅ Non-linear tire degradation (soft, medium, hard)  
✅ Fuel consumption dynamics (0.037s per lap)  
✅ Track-specific pit times (Monza 21.5s → Monaco 27.3s)  
✅ Stochastic weather transitions  
✅ 4 pre-configured tracks

### F1Metrics Simulator (NEW!)

✅ **8-parameter driver model**: qualifying position, start bonus, max speed, base pace, variability, strategy, DNF probability, tyre degradation multiplier  
✅ **Realistic lap time physics**: LapTime = BasePace + RandomVar + TyreDeg + FuelAdj + TrafficPenalty  
✅ **Quadratic tyre degradation**: Soft tyres 0.7s faster but degrade 2x faster  
✅ **Traffic simulation**: Dirty air effects and probabilistic overtaking  
✅ **Stochastic pit stops**: In-lap, stop duration (with fat-tail), out-lap penalties  
✅ **Monte Carlo analysis**: Win probabilities, position distributions, strategy effectiveness  
✅ Comprehensive documentation

---

## 📁 Quick Structure

```
src/
├── data/          # Tracks & telemetry
├── env/           # Gymnasium environment + tire physics
├── agent/         # PPO agent
└── simulation/    # Race sim & visualization

train.py          # Full training (50k steps)
evaluate.py       # Multi-track evaluation
test_environment.py # Unit tests
```

---

## 🛠️ Getting Started (3 Steps)

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Verify

```bash
python test_environment.py
# Expected: ALL TESTS PASSED!
```

### 3. Run Demo

```bash
python demo_quick_train.py
# Expected: ~8 seconds
```

---

## 📚 Documentation

| Doc                                                        | Purpose                  |
| ---------------------------------------------------------- | ------------------------ |
| **[QUICKSTART.md](QUICKSTART.md)**                         | 5-minute quick reference |
| **[INSTALLATION.md](INSTALLATION.md)**                     | Detailed setup guide     |
| **[README_FULL.md](README_FULL.md)**                       | Comprehensive guide      |
| **[TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md)** | Deep technical dive      |
| **[SYSTEM_VERIFICATION.md](SYSTEM_VERIFICATION.md)**       | Build status & tests     |
| **[BUILD_SUMMARY.md](BUILD_SUMMARY.md)**                   | What's included          |

---

## 🔥 Expected Results

After 50k step training on Monza:

```
Learned Strategy:
├─ Laps 1-15: Soft tires
├─ Lap 16: PIT → Medium
├─ Laps 16-35: Stable pace
├─ Lap 36: PIT → Hard
└─ Laps 36-53: Final stint

Performance:
├─ Race time: ~2050s
├─ Pit stops: 2 (optimal)
└─ Improvement: ~3%
```

---

## 💻 Core Commands

### RL Agent

```bash
python test_environment.py     # Verify setup
python train.py                # Full 50k training
python evaluate.py             # Evaluate on tracks
```

### F1Metrics Simulator (NEW!)

```bash
python run_f1metrics_demo.py   # Run Monte Carlo simulation
```

---

## 📖 F1Metrics Simulator Guide

### Quick Example

```python
from src.simulation import (
    F1MetricsRaceSimulator, MonteCarloRunner,
    Driver, PitStop, TyreCompound, RaceConfig
)

# Configure race
config = RaceConfig(total_laps=50, track_name="Monaco")

# Create driver with F1Metrics parameters
driver = Driver(
    name="Verstappen",
    driver_id=1,
    qualifying_position=1,
    start_bonus=0.0,
    maximum_speed=342.0,
    base_race_pace=85.0,           # Clean air lap time (seconds)
    lap_time_variability=0.20,     # Consistency (sigma)
    pit_strategy=[
        PitStop(lap=25, compound=TyreCompound.HARD)
    ],
    dnf_probability=0.001,         # Per-lap failure rate
    tyre_deg_multiplier=0.85       # Tyre management (1.0 = avg)
)

# Run single simulation
simulator = F1MetricsRaceSimulator(config, [driver], seed=42)
result = simulator.run_simulation()

# Run Monte Carlo (1000 simulations)
mc_runner = MonteCarloRunner(config, [driver], n_simulations=1000)
results = mc_runner.run()
mc_runner.print_summary(results)
```

### Driver Parameter Calibration

Convert real F1 data to driver parameters:

```python
from src.simulation import DriverCalibrator, SessionData

# Real session data
session = SessionData(
    driver_name="Hamilton",
    fp2_best_time=84.5,
    qualifying_time=83.8,
    qualifying_position=2,
    race_pace_samples=[85.1, 85.3, 85.0, 85.2],
    top_speed_kmh=340.0
)

# Auto-calibrate parameters
calibrator = DriverCalibrator()
driver = calibrator.calibrate_driver(
    session,
    field_fp2_median=85.0,
    field_qualifying_median=84.5,
    field_avg_degradation=3.5,
    driver_id=1,
    pit_strategy=[PitStop(lap=25, compound=TyreCompound.HARD)]
)
```

### Key Features

**8-Parameter Driver Model:**

1. **Qualifying Position** - Grid slot (affects Lap 1 penalty)
2. **Start Bonus** - Time gained/lost at start (±0.1s per position)
3. **Maximum Speed** - Top speed for overtaking probability
4. **Base Race Pace** - Clean air lap time on Prime tyres
5. **Lap-Time Variability** - Consistency (σ = 0.2-0.5s)
6. **Pit Strategy** - Planned stops and tyre compounds
7. **DNF Probability** - Per-lap failure rate (0.0005-0.005)
8. **Tyre Deg Multiplier** - Wear factor (0.8-1.2x)

**Lap Time Physics:**

```
LapTime = BasePace + RandomVar + TyreDeg + FuelAdj + TrafficPenalty

where:
  RandomVar ~ N(0, σ)                           # Driver variability
  TyreDeg = base_rate × (age²) × multiplier     # Quadratic degradation
  FuelAdj = 0.037s × remaining_laps             # Fuel weight
  TrafficPenalty = dirty_air or overtaking      # Race dynamics
```

**Tyre Degradation:**

- **Soft**: 0.7s faster initially, degrades 2x faster (quadratic)
- **Medium/Hard (Prime)**: Slower but durable

**Traffic & Overtaking:**

- **Dirty Air**: Within 1.0s → pace limited to car ahead + 0.2s
- **Overtaking Probability**: f(pace_delta, mistakes, top_speed)

**Monte Carlo Output:**

- Win probability for each driver
- Position distribution matrices
- Average race time ± std dev
- DNF rates
- Strategy effectiveness comparison

---

## 📊 Performance

| Metric                      | Value              |
| --------------------------- | ------------------ |
| RL Training (50k steps)     | ~100 seconds (CPU) |
| RL Evaluation (3 tracks)    | ~2-3 minutes       |
| Single F1Metrics Simulation | ~0.1-0.5 seconds   |
| Monte Carlo (1000 runs)     | ~2-5 minutes       |
| Model size                  | ~2-3 MB            |
| Memory                      | ~150 MB            |

---

## 📁 Project Structure

```
F1-StrategyOptimizer/
├── src/
│   ├── agent/                      # PPO RL agent
│   │   └── ppo_agent.py
│   ├── env/                        # Gymnasium environment
│   │   ├── f1_env.py
│   │   └── tire_model.py
│   ├── data/                       # Track configs & telemetry
│   │   ├── track_config.py
│   │   └── telemetry_loader.py
│   └── simulation/                 # Race simulators
│       ├── race_simulator.py       # RL-based simulator
│       ├── f1metrics_simulator.py  # Monte Carlo (NEW!)
│       ├── driver_calibration.py   # Parameter calibration (NEW!)
│       └── analyzer.py
├── configs/
│   └── training_config.py
├── outputs/
│   └── models/                     # Saved models
├── train.py                        # Train RL agent
├── evaluate.py                     # Evaluate RL agent
├── run_f1metrics_demo.py           # F1Metrics demo (NEW!)
├── test_environment.py             # Unit tests
├── requirements.txt
└── README.md
```

---

## 🎯 Use Cases

### 1. Strategy Optimization

Compare different pit strategies using Monte Carlo analysis:

```python
# Test 1-stop vs 2-stop
strategy_1 = [PitStop(lap=30, compound=TyreCompound.HARD)]
strategy_2 = [
    PitStop(lap=20, compound=TyreCompound.MEDIUM),
    PitStop(lap=40, compound=TyreCompound.SOFT)
]
# Run Monte Carlo for each, compare win probabilities
```

### 2. Risk Analysis

Quantify uncertainty and risk:

```python
# Monte Carlo gives probability distributions
# - Win probability: 45% ± 5%
# - Average position: P2.3 ± 1.1
# - DNF risk: 2.5%
```

### 3. Real-Time Decision Making

During a race, evaluate live options:

```python
# Current: Lap 25, mediums 10 laps old
# Option A: Continue 10 more laps
# Option B: Pit now for hards
# Run quick MC (100 sims) to compare
```

### 4. RL Agent Training

Use F1Metrics as reward function:

```python
# Hybrid approach:
# 1. RL agent proposes strategy
# 2. F1Metrics evaluates via Monte Carlo
# 3. Use expected position as reward
```

---

## 🔬 Scientific Basis

The F1Metrics simulator implements:

- **Empirical models**: Based on real F1 lap times and telemetry
- **Statistical methods**: Normal distributions, Monte Carlo sampling
- **Physical models**: Quadratic tyre degradation, fuel weight effects
- **Probabilistic overtaking**: From historical F1 data analysis
- **Validated against**: Historical race results and position distributions

---

## 🚀 Next Steps

### Quick Start

1. `pip install -r requirements.txt`
2. `python test_environment.py`
3. `python run_f1metrics_demo.py`

### Deep Dive

1. Read the code: [src/simulation/f1metrics_simulator.py](src/simulation/f1metrics_simulator.py)
2. Try calibration: [src/simulation/driver_calibration.py](src/simulation/driver_calibration.py)
3. Run demo: [run_f1metrics_demo.py](run_f1metrics_demo.py)
4. Experiment with strategies and parameters

---

**Built with ❤️ for F1 strategy optimization**

_Version 2.0.0 | F1Metrics Monte Carlo Integration ✅_
