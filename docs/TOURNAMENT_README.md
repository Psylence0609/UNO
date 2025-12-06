# Multi-Player UNO Tournament

Run comprehensive tournaments with 2, 3, or 4 players.

## Usage

```bash
# 2-player tournament (default 1000 games per matchup)
python src/evaluation/tournament_multiplayer.py --players 2

# 3-player tournament with custom game count
python src/evaluation/tournament_multiplayer.py --players 3 --games 500

# 4-player tournament
python src/evaluation/tournament_multiplayer.py --players 4
```

## Features

- **Multi-player support**: 2, 3, or 4 players per game
- **Resume capability**: Automatically skips completed matchups
- **Result merging**: Combines new results with existing data
- **Comprehensive stats**: Win rates, placements, and rankings
- **File organization**: Saves to `tournament_{n}player_results.json`

## Available Agents

- DRON (57.5%) - Advanced opponent modeling DMC
- DMC + MCTS - DMC with MCTS reward shaping
- DMC - Standard Deep Monte Carlo
- DQN + MCTS - DQN with MCTS reward shaping  
- DQN - Standard Deep Q-Network
- RLCard DMC - Official RLCard implementation
- Heuristic - Rule-based agent
- Random - Random action baseline

## Output Files

- `tournament_2player_results.json` - 2-player tournament results
- `tournament_3player_results.json` - 3-player tournament results  
- `tournament_4player_results.json` - 4-player tournament results
- `FINAL_TOURNAMENT_LEADERBOARD.md` - Comprehensive analysis (for 3-player)


