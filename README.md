# UNO Reinforcement Learning Project

A reinforcement learning project implementing intelligent agents to play UNO using deep learning techniques.

## Authors
- Praneet Sai Madhu Surabhi (psurabhi@tamu.edu)
- Akash Pillai (akash.pillai.0810@tamu.edu)

## Project Overview

This project implements and evaluates multiple reinforcement learning algorithms for the UNO card game, with a focus on addressing the sparse reward problem through Monte Carlo Tree Search (MCTS) reward shaping and opponent modeling. We compare five distinct approaches:

1. **DQN (Deep Q-Network)**: Baseline deep RL with sparse rewards only
2. **DQN+MCTS**: DQN enhanced with MCTS-based intermediate reward shaping
3. **DMC (Deep Monte Carlo)**: Episode-based Monte Carlo method with MCTS reward shaping
4. **DMC with Opponent Modeling**: DMC enhanced with explicit neural network opponent modeling
5. **RLCard DMC**: Built-in DMC implementation from RLCard toolkit

## Project Structure

```
UNO/
 src/
    agents/          # RL agent implementations
       dqn_agent_new.py    # Canonical DQN agent ([256,128] architecture)
       dmc_agent.py        # DMC agent with three-headed architecture
       dmc_agent_with_opponent.py  # DMC with opponent modeling
       opponent_modeling.py        # Opponent modeling network
       random_agent.py     # Random baseline agent
    environments/    # Environment wrappers and utilities
       uno_env.py          # RLCard UNO environment wrapper
       analyze_env.py      # Environment analysis tools
    training/        # Training loops and infrastructure
       train_dqn.py        # DQN training (sparse rewards)
       train_dqn_new.py    # DQN+MCTS training
       train_dmc.py        # DMC+MCTS training
       train_dmc_with_opponent.py  # DMC with opponent modeling training
       rlcard_dmc_trainer.py        # RLCard DMC training
       rlcard_dmc_mcts_trainer.py   # RLCard DMC+MCTS training
    features/           # Opponent feature extraction
       opponent_features.py
    evaluation/      # Evaluation and metrics
       evaluator.py        # Evaluation framework
       comprehensive_eval.py
       statistical_analysis.py  # Statistical analysis utilities
       generate_statistical_report.py  # Generate statistical reports
    mcts/           # MCTS reward shaping
       proper_mcts.py      # MCTS implementation for reward shaping
       mcts_tree.py        # MCTS tree structure
    utils/          # Utility functions and helpers
        replay_buffer.py    # Experience replay buffer
 models/             # Saved model checkpoints
    custom/         # Custom model checkpoints
    rlcard/         # RLCard model checkpoints
 experiments/        # Experiment outputs
    rlcard_dmc_uno_100M/     # RLCard DMC (100M frames)
    rlcard_dmc_mcts_uno_100M/# RLCard DMC+MCTS (100M frames)
    archived/       # Archived incomplete experiments
 logs/              # Training logs and metrics
 results/           # Evaluation results
 config.yaml        # Configuration file
 requirements.txt   # Project dependencies
 MODELS.md          # Model registry and documentation
 OPPONENT_MODELING.md # Opponent modeling research and implementation plan
 LITERATURE_SURVEY.md # Comprehensive literature survey
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

### Training Models

```bash
# Train DQN (sparse rewards)
python src/training/train_dqn.py

# Train DQN+MCTS (with MCTS reward shaping)
python src/training/train_dqn_new.py

# Train DMC+MCTS (custom implementation)
python src/training/train_dmc.py

# Train DMC with Opponent Modeling
python src/training/train_dmc_with_opponent.py

# Train RLCard DMC (parallel training)
python src/training/rlcard_dmc_trainer.py

# Train RLCard DMC+MCTS (experimental)
python src/training/rlcard_dmc_mcts_trainer.py
```

### Evaluating Models

```bash
# Evaluate all custom models
python evaluate_all_models.py

# Evaluate RLCard DMC model
python evaluate_rlcard_dmc.py experiments/rlcard_dmc_uno_100M/uno_rlcard_dmc/model.tar

# Evaluate all models (custom + RLCard) with statistical analysis
python evaluate_all_models_comprehensive.py

# Evaluate opponent modeling against diverse opponents
python evaluate_opponent_modeling.py
```

### Analyze Environment

```bash
# Comprehensive environment analysis
python src/environments/analyze_env.py
```

## Model Performance

### Current Results (vs Random Baseline)

| Model | Win Rate | 95% CI | Meets 55%? | Status |
|-------|----------|--------|------------|--------|
| RLCard DMC+MCTS (100M) | 61.3% | [59.6%, 63.0%] |  Yes | Best overall |
| RLCard DMC (100M) | 60.4% | [59.3%, 61.5%] |  Yes | Best overall |
| DMC with Opponent Modeling | TBD* | TBD | TBD | Stretch goal |
| DQN+MCTS | 51.2% | [49.9%, 52.4%] |  No | Best custom |
| DMC Ep10k | 50.0% | [48.7%, 51.4%] |  No | Good performance |
| DQN Original | 49.8% | [48.2%, 51.4%] |  No | Baseline |
| DMC Old | 47.4% | [46.3%, 48.6%] |  No | Baseline |

*TBD: To be determined after training and evaluation
**Results based on 10 evaluation runs with 500 games each (5,000 total games per model). See `results/statistical_analysis_report.md` for detailed analysis.

### Key Insights

1. **RLCard DMC is Superior**: Parallel actor-learner architecture achieves 60.4-61.3% win rate vs 47.4-51.2% for custom sequential training. Performance gap is 10.2% over best custom model with large effect sizes (p<0.001).

2. **DQN+MCTS is Best Custom Model**: DQN+MCTS (51.2%) is the best performing custom model, significantly outperforming DMC Old (p<0.001) but not significantly different from DMC Ep10k (p=0.19).

3. **MCTS Reward Shaping**: Provides modest improvements in custom models (+1.4% for DQN), but RLCard DMC+MCTS (61.3%) vs RLCard DMC (60.4%) difference is not statistically significant (p=0.30).

4. **Checkpoint Selection Matters**: DMC Ep10k (50.0%) performs significantly better than DMC Old (47.4%) with p=0.004, suggesting potential overfitting in later training stages.

5. **Statistical Robustness**: All comparisons use 10 evaluation runs with 500 games each (5,000 total games per model), providing robust 95% confidence intervals.

For detailed model information, see [MODELS.md](MODELS.md).

## Evaluation Metrics

- Win rate against random baseline
- Win rate in head-to-head comparisons
- Average game length
- Training convergence metrics
- Strategic behavior analysis

## Documentation

- **[MODELS.md](MODELS.md)**: Comprehensive model registry with architectures, performance, and checkpoint locations
- **[docs/](docs/)**: Additional documentation including research reports, tournament results, and analysis
  - **[LITERATURE_SURVEY.md](docs/LITERATURE_SURVEY.md)**: Comprehensive literature survey with 20 references
  - **[OPPONENT_MODELING.md](docs/OPPONENT_MODELING.md)**: Opponent modeling research and implementation plan
  - **[SCALING_ANALYSIS.md](docs/SCALING_ANALYSIS.md)**: Complexity scaling analysis across different player counts
  - **[FINAL_TOURNAMENT_LEADERBOARD.md](docs/FINAL_TOURNAMENT_LEADERBOARD.md)**: Tournament results and rankings
  - **[FINAL_RESEARCH_REPORT.md](docs/FINAL_RESEARCH_REPORT.md)**: Final research report and findings

## Roadmap

1.  Project setup and environment configuration
2.  Basic DQN agent implementation
3.  Training infrastructure and logging
4.  MCTS reward reshaping implementation
5.  Deep Monte Carlo (DMC) implementation
6.  Comprehensive evaluation framework
7.  RLCard DMC integration
8.  Opponent modeling implementation (see OPPONENT_MODELING.md)
9. ⏳ Opponent modeling training and evaluation
10. ⏳ Multi-agent training and self-play

## Key Contributions

- Custom DMC agent with three-headed architecture (value, policy, Monte Carlo heads)
- MCTS reward shaping integration for sparse reward problem
- Comprehensive evaluation framework comparing multiple algorithms
- Achievement of 52.6% win rate (custom DMC) and 60% win rate (RLCard DMC) vs random baseline
- Opponent modeling implementation with explicit neural network approach (DRON-inspired)
- Statistical analysis framework with confidence intervals and significance testing

## License

This project is for educational purposes as part of the Reinforcement Learning course.
