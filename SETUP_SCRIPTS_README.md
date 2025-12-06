# Setup and Execution Scripts

This directory contains automated setup and execution scripts for running UNO RL tournaments with **automatic model downloads from Google Drive**.

## Quick Start

### macOS/Linux
```bash
# Default: 3 players, 1000 games
./setup_and_run.sh

# Custom: 4 players, 500 games
./setup_and_run.sh 4 500
```

### Windows
```cmd
REM Default: 3 players, 1000 games
setup_and_run.bat

REM Custom: 4 players, 500 games
setup_and_run.bat 4 500
```

## What the Scripts Do

1. **Check Python Installation** - Verifies Python 3.8+ is installed
2. **Create Virtual Environment** - Sets up isolated Python environment
3. **Install Dependencies** - Installs all required packages from `requirements.txt` (including `gdown`)
4. **Download Models Automatically** - Downloads trained models from Google Drive (~2GB, first-time only)
5. **Verify Structure** - Ensures project files are in correct locations
6. **Run Tournament** - Executes multi-player tournament evaluation

## Automatic Model Downloads

The scripts automatically download models using `gdown`:

**What happens:**
- Script checks if `models/custom/` and `models/rlcard/` exist
- If missing, automatically installs `gdown` package
- Downloads models from Google Drive folder
- Verifies successful download

**Google Drive Source:** [UNO RL Models](https://drive.google.com/drive/folders/1A7MnHhTU2ZQ188I1O3_hURcqP8bCQsjg?usp=sharing)

**Download Size:** ~2GB (one-time download)
**Download Time:** 3-5 minutes (depends on connection speed)

## Configuration

### Command Line Arguments

```bash
./setup_and_run.sh [NUM_PLAYERS] [NUM_GAMES]
```

**Parameters:**
- `NUM_PLAYERS` - Number of players per game (2, 3, or 4). Default: 3
- `NUM_GAMES` - Games per matchup. Default: 1000

### Examples

```bash
# 2-player tournament, 500 games per matchup
./setup_and_run.sh 2 500

# 4-player tournament, 2000 games per matchup
./setup_and_run.sh 4 2000

# 3-player tournament (default), custom game count
./setup_and_run.sh 3 1500
```

## Model Downloads

If models are not found, the script will **automatically download them**:

**Automatic Process:**
1. Script detects missing models
2. Installs `gdown` package (if not present)
3. Downloads from [Google Drive](https://drive.google.com/drive/folders/1A7MnHhTU2ZQ188I1O3_hURcqP8bCQsjg?usp=sharing)
4. Extracts to `models/custom/` and `models/rlcard/`
5. Verifies download success

**If Automatic Download Fails:**

Manual download option is provided:
1. Visit the [Google Drive link](https://drive.google.com/drive/folders/1A7MnHhTU2ZQ188I1O3_hURcqP8bCQsjg?usp=sharing)
2. Download `custom` and `rlcard` folders
3. Place in `models/` directory
4. Run script again

**Downloaded Models:**
- **Custom models** (5 files): DRON, DMC+MCTS, DMC, DQN+MCTS, DQN
- **RLCard models** (1 file): Official RLCard DMC baseline

## Output

Results are saved to `tournament_{NUM_PLAYERS}player_results.json` in the project root.

**Example:** Running a 4-player tournament creates `tournament_4player_results.json`

## Troubleshooting

**Python not found:**
- Install Python 3.8 or higher from [python.org](https://www.python.org/)

**Permission denied (macOS/Linux):**
```bash
chmod +x setup_and_run.sh
```

**Models not found:**
- Download models from the Google Drive link
- Place in `models/custom/` and `models/rlcard/`
- Run script again

**Import errors:**
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

## Requirements

- Python 3.8 or higher
- ~2GB disk space for models
- ~4GB RAM for tournament execution
- Internet connection (for initial setup only)

## Estimated Runtime

Tournament duration depends on configuration:

- **2-player, 1000 games**: ~15-30 minutes
- **3-player, 1000 games**: ~30-60 minutes
- **4-player, 1000 games**: ~60-120 minutes

Times vary based on hardware (CPU, RAM).

