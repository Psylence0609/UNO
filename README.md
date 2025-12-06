# UNO Reinforcement Learning Project

A comprehensive survey and evaluation of Deep Reinforcement Learning architectures for UNO, featuring advanced opponent modeling and multi-scale tournament evaluation across 2, 3, and 4-player games.

## Authors
- Praneet Sai Madhu Surabhi (psurabhi@tamu.edu)
- Akash Pillai (akash.pillai.0810@tamu.edu)

## Project Overview

This project implements and evaluates five distinct Deep RL approaches for the UNO card game, addressing key challenges in imperfect information multi-agent environments:

1. **DRON** (Deep Reinforcement Opponent Network): Advanced DMC with transformer-based opponent modeling
2. **DMC + MCTS**: Deep Monte Carlo with MCTS reward shaping
3. **DMC**: Custom Deep Monte Carlo with three-headed architecture
4. **DQN + MCTS**: Deep Q-Network with MCTS reward shaping
5. **RLCard DMC**: Official RLCard DMC implementation baseline

### Key Research Focus
- **Sparse Reward Problem**: Addressed through MCTS-based reward shaping
- **Opponent Modeling**: Explicit neural network approach with attention mechanisms
- **Scalability Analysis**: Systematic evaluation across different game complexities

## Project Structure

```
UNO/
├── src/
│   ├── agents/              # RL agent implementations
│   ├── environments/        # Environment wrappers
│   ├── training/            # Training infrastructure
│   ├── evaluation/          # Tournament and evaluation
│   ├── mcts/                # MCTS reward shaping
│   └── utils/               # Utilities
├── models/                  # Trained model checkpoints
├── docs/                    # Documentation and reports
├── tournament_*player_results.json  # Tournament data
├── config.yaml              # Configuration
└── requirements.txt         # Dependencies
```

## Quick Start (Automated Setup)

### One-Click Tournament Execution

We provide automated setup scripts that handle **everything automatically** - including downloading trained models from Google Drive:

**macOS/Linux:**
```bash
# Run with default settings (3 players, 1000 games)
./setup_and_run.sh

# Custom configuration
./setup_and_run.sh 4 500  # 4 players, 500 games per matchup
```

**Windows(Experimental and not tested):**
```cmd
REM Run with default settings (3 players, 1000 games)
setup_and_run.bat

REM Custom configuration
setup_and_run.bat 4 500  REM 4 players, 500 games per matchup
```

**The script automatically:**
1. Checks Python installation
2. Creates virtual environment
3. Installs dependencies (including `gdown`)
4. **Downloads trained models from Google Drive** (~2GB)
5. Verifies project structure
6. Runs tournament and saves results

**First-time setup takes ~5-10 minutes** (includes model download). Subsequent runs are instant.

**IMPORTANT NOTE**: If you want to rerun a particular tournament you have to MOVE or DELETE the corresponding json file like 'tournament_3player_results.json'. The script resumes the tournament if an old file is found.

See [SETUP_SCRIPTS_README.md](SETUP_SCRIPTS_README.md) for detailed script documentation.

### Model Information

**Models are automatically downloaded from:** [Google Drive - UNO RL Models](https://drive.google.com/drive/folders/1A7MnHhTU2ZQ188I1O3_hURcqP8bCQsjg?usp=sharing)

**Included Models:**
- `models/custom/` - Custom trained agents (DRON, DMC, DQN variants) - 5 models
- `models/rlcard/` - RLCard DMC baseline implementation - 1 model

**Manual Download (if automated download fails):**
1. Visit the [Google Drive link](https://drive.google.com/drive/folders/1A7MnHhTU2ZQ188I1O3_hURcqP8bCQsjg?usp=sharing)
2. Download the `custom` and `rlcard` folders
3. Place them in the `models/` directory in your project root

## Manual Setup

If you prefer manual setup:

### Installation

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download models** from the [Google Drive link](https://drive.google.com/drive/folders/1A7MnHhTU2ZQ188I1O3_hURcqP8bCQsjg?usp=sharing) and place in `models/`

4. **Verify installation:**
```bash
python -c "import rlcard; print('RLCard version:', rlcard.__version__)"
```

### Running Tournaments

Evaluate agents across different complexity levels:

```bash
# 2-player tournament (head-to-head)
python src/evaluation/tournament_multiplayer.py --players 2

# 3-player tournament (balanced complexity)
python src/evaluation/tournament_multiplayer.py --players 3

# 4-player tournament (maximum complexity)
python src/evaluation/tournament_multiplayer.py --players 4

# Custom number of games per matchup
python src/evaluation/tournament_multiplayer.py --players 4 --games 500
```

**Tournament Features:**
- Automatic result saving to `tournament_{n}player_results.json`
- Smart skip logic for resuming interrupted tournaments
- Statistical analysis with win rates and average placements

## Tournament Results

### 4-Player Tournament (Maximum Complexity)

Our comprehensive 4-player tournament reveals how algorithms scale with multi-agent complexity:

| Rank | Agent | Win Rate | Avg Placement | Scaling (2→4) |
|------|-------|----------|---------------|---------------|
| 1 | **DRON** | **54.2%** | 1.46 | **+0.8%** ✓ |
| 2 | **DMC + MCTS** | **47.4%** | 1.53 | **+1.5%** ✓ |
| 3 | DMC | 41.6% | 1.51 | -6.3% |
| 4 | DQN + MCTS | 30.8% | 1.51 | -17.6% |
| 5 | DQN | 17.6% | 1.53 | -29.5% |
| 6 | RLCard DMC | 8.3% | 1.42 | -49.4% |
| 7 | Heuristic | 0.0% | - | -52.0% |
| 8 | Random | 0.0% | - | -47.6% |

**Total Games**: 70,000 (8,750 per agent across 70 unique matchups)

### Key Findings

#### 1. DRON Demonstrates Exceptional Scalability
- **Only agent that improves with complexity** (+0.8% from 2 to 4 players)
- Maintains 54.2% win rate in maximum complexity scenarios
- Advanced opponent modeling successfully captures strategic patterns
- **Beats all opponents in head-to-head matchups**

#### 2. MCTS Provides Algorithmic Resilience
- **DMC + MCTS shows positive scaling** (+1.5% from 2 to 4 players)
- MCTS reward shaping becomes more valuable with increased complexity
- Significantly outperforms vanilla DMC in 4-player games (47.4% vs 41.6%)

#### 3. Traditional RL Algorithms Fail to Scale
- **DQN variants collapse**: 29.5% performance loss for vanilla DQN
- Deep Q-Learning fundamentally ill-suited for complex multi-agent environments
- Temporal difference learning struggles with increased stochasticity

#### 4. RLCard DMC Catastrophic Failure
- **Worst performance drop**: 57.7% (2-player) → 8.3% (4-player)
- Optimized implementation overfits to simpler scenarios
- Demonstrates importance of architectural choices over computational efficiency

#### 5. Rule-Based Methods Cannot Handle Complexity
- **Heuristic agent**: 52.0% (2-player) → 0.0% (4-player)
- Hand-crafted strategies fail spectacularly in complex settings
- Learning-based approaches essential for multi-agent domains

### Performance Categories

**Elite Scalers (Maintain/Improve):**
- DRON: +0.8% scaling
- DMC+MCTS: +1.5% scaling

**Poor Scalers (Major Degradation):**
- DQN: -29.5% scaling
- RLCard DMC: -49.4% scaling

**Complete Failures:**
- Heuristic: -52.0% scaling
- Random: -47.6% scaling

### Scaling Analysis Summary

Our **504,000-game evaluation** across three complexity levels reveals:

1. **Opponent Modeling is Complexity-Robust**: DRON's positive scaling validates that modeling hidden information is crucial for multi-agent RL success

2. **MCTS Enables Scaling**: Intermediate value estimates help agents navigate increased stochasticity

3. **Evaluation Methodology Matters**: Performance in 2-player games does not predict 4-player performance

4. **Complexity Reveals Algorithm Quality**: 4-player games effectively filter robust algorithms from brittle ones

## Research Contributions

1. **DRON Architecture**: Novel opponent modeling with transformer attention and gated fusion, achieving 54.2% win rate in 4-player tournaments

2. **First Systematic Scaling Study**: Comprehensive analysis showing that sophisticated opponent modeling scales positively with complexity

3. **MCTS Integration with DMC**: Demonstration that reward shaping provides algorithmic resilience across complexity levels

4. **Empirical Rigor**: 504,000 games across 154 unique matchups with statistical significance testing

5. **Open-Source Implementation**: Complete reproducible codebase with trained models

## Documentation

- **[docs/FINAL_REPORT.md](docs/FINAL_REPORT.md)**: Comprehensive survey paper with detailed architectures and analysis
- **[docs/SCALING_ANALYSIS.md](docs/SCALING_ANALYSIS.md)**: Complexity scaling analysis across player counts
- **[docs/TOURNAMENT_4PLAYER_LEADERBOARD.md](docs/TOURNAMENT_4PLAYER_LEADERBOARD.md)**: 4-player tournament results
- **[docs/FINAL_TOURNAMENT_LEADERBOARD.md](docs/FINAL_TOURNAMENT_LEADERBOARD.md)**: 3-player tournament results
- **[docs/OPPONENT_MODELING.md](docs/OPPONENT_MODELING.md)**: Opponent modeling implementation details
- **[docs/LITERATURE_SURVEY.md](docs/LITERATURE_SURVEY.md)**: Comprehensive literature survey (20 references)
- **[MODELS.md](MODELS.md)**: Model registry with architectures and checkpoints

## Citation

If you use this work, please cite:

```
P. S. M. Surabhi and A. Pillai, "A Survey of Deep Reinforcement Learning 
Architectures for UNO," Texas A&M University, 2025.
```

## Key Insights

**For Researchers:**
- Opponent modeling is the key to multi-agent RL success
- Test algorithms across multiple complexity levels
- MCTS reward shaping provides resilience in sparse reward domains
- Architectural choices matter more than computational efficiency

**For Practitioners:**
- Avoid DQN for complex multi-agent scenarios
- Prioritize opponent modeling + MCTS for imperfect information games
- Evaluate across different player counts to ensure robustness
- Official implementations may not generalize to complex scenarios

## License

This project is for educational purposes as part of the Deep Reinforcement Learning course at Texas A&M University.

---

**GitHub Repository**: [https://github.com/Psylence0609/UNO](https://github.com/Psylence0609/UNO)