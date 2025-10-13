# UNO Reinforcement Learning Project

A reinforcement learning project implementing intelligent agents to play UNO using deep learning techniques.

## Authors
- Praneet Sai Madhu Surabhi (psurabhi@tamu.edu)
- Akash Pillai (akash.pillai.0810@tamu.edu)

## Project Overview

This project implements reinforcement learning agents to play UNO using the RLCard environment. The project includes:

- **DQN Agent**: Basic Deep Q-Network implementation with sparse rewards
- **DMC Agent**: Deep Monte Carlo agent with MCTS reward reshaping
- **Training Infrastructure**: Comprehensive training and evaluation framework
- **Baseline Comparisons**: Testing against random and rule-based agents

## Project Structure

```
UNO/
├── src/
│   ├── agents/          # RL agent implementations
│   ├── environments/    # Environment wrappers and utilities
│   ├── training/        # Training loops and infrastructure
│   ├── evaluation/      # Evaluation and metrics
│   └── utils/          # Utility functions and helpers
├── models/             # Saved model checkpoints
├── logs/              # Training logs and tensorboard files
├── experiments/       # Experiment configurations and results
└── requirements.txt   # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify RLCard UNO environment:
```bash
python -c "import rlcard; print('RLCard version:', rlcard.__version__)"
```

## Quick Start

### Basic Usage

```bash
# Run the main demo
python main.py

# Test DQN agent
python test_dqn.py

# Quick training run (1000 episodes)
python quick_train.py

# Full training (uses config.yaml)
python src/training/train_dqn.py
```

### Analyze Environment

```bash
# Comprehensive environment analysis
python src/environments/analyze_env.py
```

### Training Results

Recent training achieved:
- **Peak win rate: 59%** (very close to 60% target)
- Training time: ~30 seconds for 1000 episodes
- Clear learning progression demonstrated

Logs and plots saved in `./logs/` directory.

## Evaluation Metrics

- Win rate against different opponent types
- Average game length
- Strategic behavior analysis
- Training convergence speed and stability

## Roadmap

1. ✅ Project setup and environment configuration
2. ✅ Basic DQN agent implementation  
3. ✅ Training infrastructure and logging
4. ✅ Successful training (59% win rate achieved!)
5. ⏳ MCTS reward reshaping
6. ⏳ Deep Monte Carlo (DMC) implementation
7. ⏳ Comprehensive evaluation framework
8. ⏳ Stretch goals (opponent modeling, multi-agent training)

## License

This project is for educational purposes as part of the Reinforcement Learning course.