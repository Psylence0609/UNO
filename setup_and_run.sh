#!/bin/bash
# UNO RL Tournament Setup and Execution Script
# This script sets up the environment, downloads models, and runs tournaments

set -e  # Exit on error

# Configuration variables (can be overridden via command line)
NUM_PLAYERS=${1:-3}  # Default: 3 players
NUM_GAMES=${2:-1000}  # Default: 1000 games per matchup
MODELS_URL="https://drive.google.com/drive/folders/1A7MnHhTU2ZQ188I1O3_hURcqP8bCQsjg?usp=sharing"

echo "=================================================="
echo "UNO RL Tournament Setup and Execution"
echo "=================================================="
echo "Configuration:"
echo "  Number of Players: $NUM_PLAYERS"
echo "  Games per Matchup: $NUM_GAMES"
echo "=================================================="
echo ""

# Step 1: Check Python installation
echo "[1/6] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "  ✓ Found Python $PYTHON_VERSION"
echo ""

# Step 2: Create virtual environment if it doesn't exist
echo "[2/6] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    echo "  Creating new virtual environment..."
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment already exists"
fi
echo ""

# Step 3: Activate virtual environment and install dependencies
echo "[3/6] Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
echo "  Installing project requirements..."
pip install -r requirements.txt > /dev/null 2>&1
echo "  ✓ Dependencies installed"
echo ""

# Step 4: Check if models exist, if not download automatically
echo "[4/6] Checking for trained models..."
if [ ! -d "models/custom" ] || [ ! -d "models/rlcard" ]; then
    echo "  Models not found. Downloading from Google Drive..."
    echo ""
    
    # Check if gdown is installed
    if ! python -c "import gdown" &> /dev/null; then
        echo "  Installing gdown for Google Drive downloads..."
        pip install gdown > /dev/null 2>&1
        echo "  ✓ gdown installed"
    fi
    
    # Extract folder ID from URL
    FOLDER_ID="1A7MnHhTU2ZQ188I1O3_hURcqP8bCQsjg"
    
    echo "  Downloading models (this may take a few minutes)..."
    echo "  Source: Google Drive folder"
    echo ""
    
    # Download the folder directly to current directory
    # Google Drive folder already contains models/custom/ and models/rlcard/
    python -m gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O ./ --remaining-ok
    
    if [ $? -eq 0 ]; then
        echo ""
        # Verify models are in correct location
        if [ -d "models/custom" ] && [ -d "models/rlcard" ]; then
            echo "  ✓ Models downloaded successfully"
            echo "    - Custom models: $(ls models/custom/*.pth 2>/dev/null | wc -l) files"
            echo "    - RLCard models: $(ls models/rlcard/*.tar 2>/dev/null | wc -l) files"
        else
            echo "  ✗ Models not found in expected location. Please download manually:"
            echo "    1. Visit: $MODELS_URL"
            echo "    2. Download the 'models' folder"
            echo "    3. Place it in the project root directory"
            echo ""
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Exiting. Please download models and run again."
                exit 1
            fi
        fi
    else
        echo ""
        echo "  ✗ Download failed. Please download manually:"
        echo "    1. Visit: $MODELS_URL"
        echo "    2. Download the 'models' folder"
        echo "    3. Place it in the project root directory"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Exiting. Please download models and run again."
            exit 1
        fi
    fi
else
    echo "  ✓ Models already present"
    echo "    - Custom models: $(ls models/custom/*.pth 2>/dev/null | wc -l) files"
    echo "    - RLCard models: $(ls models/rlcard/*.tar 2>/dev/null | wc -l) files"
fi
echo ""

# Step 5: Verify project structure
echo "[5/6] Verifying project structure..."
if [ ! -f "src/evaluation/tournament_multiplayer.py" ]; then
    echo "  ERROR: Tournament script not found. Please ensure you're in the project root directory."
    exit 1
fi
echo "  ✓ Project structure verified"
echo ""

# Step 6: Run tournament
echo "[6/6] Starting tournament..."
echo "=================================================="
echo "Running ${NUM_PLAYERS}-player tournament with ${NUM_GAMES} games per matchup"
echo "This may take a while depending on your configuration..."
echo "Results will be saved to: tournament_${NUM_PLAYERS}player_results.json"
echo "=================================================="
echo ""

python src/evaluation/tournament_multiplayer.py --players "$NUM_PLAYERS" --games "$NUM_GAMES"

echo ""
echo "=================================================="
echo "Tournament complete!"
echo "Results saved to: tournament_${NUM_PLAYERS}player_results.json"
echo "=================================================="

