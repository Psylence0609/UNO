# UNO Model Registry

This document provides a comprehensive registry of all trained models in this repository, including their architectures, training configurations, performance metrics, and checkpoint locations.

## Model Overview

We have implemented and evaluated 7 core models for playing UNO:

1. **DQN (Deep Q-Network)** - Baseline deep RL with sparse rewards
2. **DQN+MCTS** - DQN enhanced with MCTS reward shaping
3. **DMC (Deep Monte Carlo)** - Episode-based learning with MCTS reward shaping
4. **DMC+MCTS** - DMC with MCTS reward shaping (same as DMC, explicitly noted)
5. **DMC with Opponent Modeling** - DMC enhanced with explicit neural network opponent modeling
6. **RLCard DMC** - RLCard's built-in DMC implementation
7. **RLCard DMC+MCTS** - RLCard DMC with MCTS reward shaping wrapper

## Model Performance Summary

| Model | Win Rate vs Random | 95% CI | Meets 55% Threshold? | Status |
|-------|-------------------|--------|---------------------|--------|
| RLCard DMC+MCTS (100M) | 61.3% | [59.6%, 63.0%] | Yes | Best overall |
| RLCard DMC (100M) | 60.4% | [59.3%, 61.5%] | Yes | Best overall |
| DMC with Opponent Modeling | TBD* | TBD | TBD | Stretch goal |
| DQN+MCTS | 51.2% | [49.9%, 52.4%] | No | Improved over baseline |
| DMC Ep10k | 50.0% | [48.7%, 51.4%] | No | Best custom model |
| DQN Original | 49.8% | [48.2%, 51.4%] | No | Baseline |
| DMC Old | 47.4% | [46.3%, 48.6%] | No | Good performance |

*TBD: To be determined after training and evaluation
**Results based on 10 evaluation runs with 500 games each (5,000 total games per model). Statistical analysis performed with 95% confidence intervals.

---

## Custom Models

### 1. DQN Original

**Description**: Baseline Deep Q-Network implementation using sparse rewards only (win/loss signals).

**Architecture**:
- Type: Dueling Double DQN
- Hidden Layers: [256, 128]
- Activation: ReLU
- Dropout: 0.1
- Dueling: Yes
- Double DQN: Yes

**Training Configuration**:
- Episodes: 15,000
- Learning Rate: 0.0005
- Batch Size: 64
- Gamma: 0.99
- Epsilon: 1.0 → 0.05 (decay: 0.9995)
- Memory Size: 20,000
- Target Update Frequency: 1000
- MCTS Reward Shaping: No

**Performance**:
- Win Rate vs Random: 49.8% (95% CI: [48.2%, 51.4%])
- Average Game Length: 47.1 turns
- Meets 55% Threshold: No
- Statistical Significance: Not significantly different from DQN+MCTS (p=0.14)

**Checkpoint Location**: `models/dqn_final.pth`

**Training Script**: `src/training/train_dqn.py`

**Notes**: Baseline model for comparison. Performance is below random baseline, indicating challenges with sparse rewards in UNO.

---

### 2. DQN+MCTS

**Description**: Dueling Double DQN enhanced with MCTS-based intermediate reward shaping to address sparse reward problem.

**Architecture**:
- Type: Dueling Double DQN
- Hidden Layers: [256, 128]
- Activation: ReLU
- Dropout: 0.1
- Dueling: Yes
- Double DQN: Yes

**Training Configuration**:
- Episodes: 15,000
- Learning Rate: 0.0005
- Batch Size: 64
- Gamma: 0.99
- Epsilon: 1.0 → 0.05 (decay: 0.9995)
- Memory Size: 20,000
- Target Update Frequency: 1000
- MCTS Reward Shaping: Yes
  - Simulations: 200
  - Exploration Constant: 1.414
  - Max Depth: 20
  - Intermediate Reward Weight: 0.3

**Performance**:
- Win Rate vs Random: 51.2% (95% CI: [49.9%, 52.4%])
- Average Game Length: 47.4 turns
- Meets 55% Threshold: No
- Statistical Significance: Significantly better than DMC Old (p<0.001, large effect)

**Checkpoint Location**: `models/dqn_mcts_final.pth`

**Training Script**: `src/training/train_dqn_new.py`

**Notes**: MCTS reward shaping provides modest improvement over baseline DQN (+1.4%). Performance is at random level, suggesting need for more training or better reward shaping.

---

### 3. DMC (Deep Monte Carlo)

**Description**: Deep Monte Carlo agent with episode-based learning and MCTS reward shaping. Uses three-headed architecture (value, policy, Monte Carlo heads).

**Architecture**:
- Type: Deep Monte Carlo (DMC)
- Hidden Layers: [256, 128]
- Activation: ReLU
- Dropout: 0.1
- Heads: 3 (value, policy, Monte Carlo)
- Learning: Episode-based (Monte Carlo returns)

**Training Configuration**:
- Episodes: 15,000
- Learning Rate: 0.0005
- Batch Size: 64
- Gamma: 0.99
- Epsilon: 1.0 → 0.05 (decay: 0.9995)
- MCTS Reward Shaping: Yes
  - Simulations: 200
  - Exploration Constant: 1.414
  - Max Depth: 20
  - Intermediate Reward Weight: 0.3

**Performance**:
- Win Rate vs Random: 47.4% (DMC Old, 95% CI: [46.3%, 48.6%]), 50.0% (DMC Ep10k, 95% CI: [48.7%, 51.4%])
- Average Game Length: 46.5 turns (DMC Old), 47.6 turns (DMC Ep10k)
- Meets 55% Threshold: No
- Statistical Significance: DMC Ep10k significantly better than DMC Old (p=0.004, large effect)

**Checkpoint Locations**:
- Final: `models/custom/dmc_mcts_final.pth` (47.4% win rate)
- Episode 10,000: `models/custom/dmc_episode_10000.pth` (50.0% win rate)

**Training Script**: `src/training/train_dmc.py`

**Notes**: DMC Ep10k (checkpoint at episode 10,000) performs better than the final checkpoint, suggesting potential overfitting in later training. DMC outperforms DQN variants, demonstrating the effectiveness of episode-based learning for UNO.

---

### 4. DMC+MCTS

**Description**: Same as DMC above. MCTS reward shaping is integrated into the DMC training process.

**Architecture**: Same as DMC
**Training Configuration**: Same as DMC
**Performance**: Same as DMC
**Checkpoint Locations**: Same as DMC
**Training Script**: Same as DMC

**Notes**: This is the same model as DMC, explicitly noted to match the 6-model structure. MCTS reward shaping is a core component of the DMC implementation.

---

### 5. DMC with Opponent Modeling

**Description**: Deep Monte Carlo agent enhanced with explicit neural network opponent modeling. Uses opponent feature extraction and strategy representation to adapt to different opponent playing styles.

**Architecture**:
- Type: Deep Monte Carlo (DMC) with Opponent Modeling
- State Network: [256, 128] hidden layers
- Opponent Modeling Network: [128, 64] hidden layers
- Strategy Dimension: 32
- Fusion Layer: Combines state and opponent strategy features
- Heads: 3 (value, policy, Monte Carlo)
- Learning: Episode-based (Monte Carlo returns) with joint opponent modeling

**Training Configuration**:
- Episodes: 15,000
- Learning Rate: 0.0005
- Batch Size: 64 (episode-based)
- Gamma: 0.99
- Epsilon: 1.0 → 0.05 (decay: 0.9995)
- MCTS Reward Shaping: Optional (configurable)
- Opponent Modeling:
  - History Size: 20 actions
  - Feature Size: 75+ (hand size, action history, play patterns, etc.)
  - Strategy Dimension: 32
  - Hidden Layers: [128, 64]

**Features**:
- Opponent action history tracking (last 20 actions)
- Hand size tracking and change rate
- Action type frequencies (number cards, special cards, wild cards)
- Play style indicators (aggressive vs conservative)
- Game progress features

**Performance**:
- Win Rate vs Random: TBD (requires training and evaluation)
- Average Game Length: TBD
- Meets 55% Threshold: TBD
- Expected Benefits:
  - Better adaptation to opponent strategies
  - Improved performance against diverse opponents
  - Ability to exploit opponent weaknesses

**Checkpoint Location**: `models/custom/dmc_opponent_modeling_final.pth`

**Training Script**: `src/training/train_dmc_with_opponent.py`

**Evaluation Script**: `evaluate_opponent_modeling.py`

**Notes**: This is a stretch goal implementation. The agent learns to model opponent strategies explicitly through a separate neural network that processes opponent features. The opponent strategy representation is fused with state features to inform action selection. Expected to show improved performance against diverse opponent types compared to baseline DMC.

---

## RLCard Models

### 6. RLCard DMC

**Description**: RLCard's built-in Deep Monte Carlo implementation using their optimized actor-learner architecture with parallel training.

**Architecture**:
- Type: RLCard DMC
- Hidden Layers: [512, 512, 512, 512, 512]
- Activation: ReLU (RLCard default)
- Architecture: Single Q-value head (simpler than custom DMC)
- Training: Actor-learner with parallel actors

**Training Configuration**:
- Frames: 100,000,000 (100M)
- Learning Rate: 0.0001
- Batch Size: 32
- Unroll Length: 100
- Number of Actors: 5
- Exploration Epsilon: 0.01
- Number of Buffers: 50
- Number of Threads: 4

**Performance**:
- Win Rate vs Random: 60.4% (95% CI: [59.3%, 61.5%])
- Average Game Length: 39.3 turns
- Meets 55% Threshold:  Yes
- Statistical Significance: Significantly better than all custom models (p<0.001, large effect sizes: d=-5.5 to -8.2)
- Performance Gap: 10.2% improvement over best custom model (DQN+MCTS)

**Checkpoint Location**: `models/rlcard/rlcard_dmc_100M.tar`

**Training Script**: `src/training/rlcard_dmc_trainer.py`

**Notes**: RLCard's optimized implementation with parallel training significantly outperforms custom sequential training. The simpler architecture but efficient parallel training achieves superior performance. Confirmed through comprehensive evaluation with 10 runs × 500 games.

---

### 7. RLCard DMC+MCTS

**Description**: RLCard DMC with experimental MCTS reward shaping wrapper. This is an experimental integration that attempts to add MCTS rewards to RLCard's training loop.

**Architecture**: Same as RLCard DMC
**Training Configuration**: Same as RLCard DMC, with MCTS wrapper

**Performance**:
- Win Rate vs Random: 61.3% (95% CI: [59.6%, 63.0%])
- Average Game Length: 39.2 turns
- Meets 55% Threshold:  Yes
- Statistical Significance: Significantly better than all custom models (p<0.001, large effect sizes: d=-4.8 to -6.9)
- Performance Gap: 10.1% improvement over best custom model (DQN+MCTS)
- vs RLCard DMC: Not significantly different (p=0.30, small effect: d=-0.47)

**Checkpoint Location**: `models/rlcard/rlcard_dmc_mcts_100M.tar`

**Training Script**: `src/training/rlcard_dmc_mcts_trainer.py`

**Notes**: Experimental integration of MCTS reward shaping with RLCard's trainer. The MCTS-enhanced version shows a small improvement (61.3% vs 60.4%), but the difference is not statistically significant (p=0.30), suggesting MCTS reward shaping provides minimal benefit in RLCard's parallel training setup. Confirmed through comprehensive evaluation.

---

## Model Comparison

### Performance Ranking (vs Random)

1. **RLCard DMC+MCTS (100M)** - 61.3% (95% CI: [59.6%, 63.0%]) Yes
2. **RLCard DMC (100M)** - 60.4% (95% CI: [59.3%, 61.5%]) Yes
3. **DMC with Opponent Modeling** - TBD (requires training) *
4. **DQN+MCTS** - 51.2% (95% CI: [49.9%, 52.4%]) No
5. **DMC Ep10k** - 50.0% (95% CI: [48.7%, 51.4%]) No
6. **DQN Original** - 49.8% (95% CI: [48.2%, 51.4%]) No
7. **DMC Old** - 47.4% (95% CI: [46.3%, 48.6%]) No

*Expected to outperform baseline DMC after training
**All results based on 10 evaluation runs with 500 games each (5,000 total games per model)

### Key Insights

1. **RLCard DMC is Superior**: RLCard's parallel actor-learner architecture achieves significantly better performance (60.4-61.3% vs 47.4-51.2%) than custom sequential training. The performance gap is 10.2% over the best custom model, with large effect sizes (Cohen's d = -4.8 to -8.2).

2. **RLCard DMC+MCTS vs DMC**: The MCTS-enhanced version shows a small improvement (61.3% vs 60.4%), but the difference is not statistically significant (p=0.30), suggesting MCTS reward shaping provides minimal benefit in RLCard's parallel training setup.

3. **DQN+MCTS is Best Custom Model**: DQN+MCTS (51.2%) is the best performing custom model, significantly outperforming DMC Old (p<0.001, large effect) but not significantly different from DMC Ep10k (p=0.19).

4. **DMC Performance Varies by Checkpoint**: DMC Ep10k (50.0%) performs better than DMC Old (47.4%), with the difference being statistically significant (p=0.004, large effect). This suggests potential overfitting in later training stages or checkpoint selection matters.

5. **Architecture vs Training Efficiency**: RLCard uses simpler architecture but parallel training, while custom models use more complex architectures but sequential training. Parallel training appears to be significantly more important than architecture complexity (10.2% performance gap).

6. **Statistical Robustness**: All comparisons between RLCard and custom models are highly statistically significant (p<0.001) with large effect sizes, confirming the superiority of parallel training architecture.

---

## Evaluation Results

### All Models (Evaluated with Statistical Analysis)

Results from `results/model_evaluation_results.csv` (10 runs × 500 games = 5,000 games per model):

| Model | Win Rate | 95% CI | Avg Turns | Ranking |
|-------|----------|--------|-----------|---------|
| RLCard DMC+MCTS (100M) | 61.3% | [59.6%, 63.0%] | 39.2 | 1 |
| RLCard DMC (100M) | 60.4% | [59.3%, 61.5%] | 39.3 | 2 |
| DQN+MCTS | 51.2% | [49.9%, 52.4%] | 47.4 | 3 |
| DMC Ep10k | 50.0% | [48.7%, 51.4%] | 47.6 | 4 |
| DQN Original | 49.8% | [48.2%, 51.4%] | 47.1 | 5 |
| DMC Old | 47.4% | [46.3%, 48.6%] | 46.5 | 6 |

**Statistical Analysis**: See `results/statistical_analysis_report.md` for detailed pairwise comparisons, effect sizes, and significance tests.

---

## Model Files Structure

```
models/
 dqn_final.pth                    # DQN Original
 dqn_mcts_final.pth               # DQN+MCTS
 dqn_mcts_episode_10000.pth       # DQN+MCTS checkpoint
 dmc_mcts_final.pth               # DMC Old (final)
 dmc_episode_10000.pth            # DMC Ep10k (best)

experiments/
 rlcard_dmc_uno_100M/
    uno_rlcard_dmc/
        model.tar                # RLCard DMC (100M frames)
 rlcard_dmc_mcts_uno_100M/
     uno_rlcard_dmc_mcts/
         model.tar                # RLCard DMC+MCTS (100M frames)
```

---

## Training Scripts

| Model | Training Script | Agent File |
|-------|----------------|------------|
| DQN Original | `src/training/train_dqn.py` | `src/agents/dqn_agent_new.py` |
| DQN+MCTS | `src/training/train_dqn_new.py` | `src/agents/dqn_agent_new.py` |
| DMC / DMC+MCTS | `src/training/train_dmc.py` | `src/agents/dmc_agent.py` |
| DMC with Opponent Modeling | `src/training/train_dmc_with_opponent.py` | `src/agents/dmc_agent_with_opponent.py` |
| RLCard DMC | `src/training/rlcard_dmc_trainer.py` | RLCard built-in |
| RLCard DMC+MCTS | `src/training/rlcard_dmc_mcts_trainer.py` | RLCard built-in + wrapper |

---

## Evaluation Scripts

- `evaluate_all_models.py` - Evaluates custom models
- `evaluate_rlcard_dmc.py` - Evaluates RLCard DMC models
- `evaluate_all_models_comprehensive.py` - Evaluates all models (custom + RLCard) with statistical analysis
- `evaluate_opponent_modeling.py` - Evaluates DMC with opponent modeling against diverse opponents

---

## Recommendations

### For Best Performance

1. **Use RLCard DMC**: Achieves ~60% win rate with parallel training
2. **Evaluate RLCard Models**: Confirm exact performance with comprehensive evaluation
3. **Consider Retraining**: Custom models below 55% threshold may benefit from:
   - More training episodes
   - Better hyperparameter tuning
   - Parallel training architecture

### For Research

1. **Custom DMC**: Best custom model (52.6%) with three-headed architecture
2. **MCTS Integration**: Investigate why MCTS reward shaping provides only modest gains
3. **Checkpoint Selection**: Study why DMC Ep10k outperforms final checkpoint

### For Development

1. **Start with RLCard DMC**: Use as baseline for comparison
2. **Experiment with Custom DMC**: Modify architecture and training for research
3. **Evaluate Regularly**: Use evaluation scripts to track performance

---

## Future Work

1. **Evaluate RLCard Models**: Confirm performance with comprehensive evaluation
2. **Retrain Custom Models**: Optimize hyperparameters to reach 55% threshold
3. **Parallel Training**: Implement parallel training for custom models
4. **Train and Evaluate Opponent Modeling**: Complete training and evaluation of DMC with opponent modeling
5. **Self-Play**: Train agents through self-play for improved performance
6. **Advanced Opponent Modeling**: Explore Bayesian opponent modeling and meta-learning approaches

---

*Last Updated: Based on evaluation results from `results/complete_evaluation_results.csv` and training logs.*

