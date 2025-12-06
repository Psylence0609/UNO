@echo off
REM UNO RL Tournament Setup and Execution Script (Windows)
REM This script sets up the environment, downloads models, and runs tournaments

setlocal enabledelayedexpansion

REM Configuration variables (can be overridden via command line)
set NUM_PLAYERS=%1
set NUM_GAMES=%2
if "%NUM_PLAYERS%"=="" set NUM_PLAYERS=3
if "%NUM_GAMES%"=="" set NUM_GAMES=1000
set MODELS_URL=https://drive.google.com/drive/folders/1A7MnHhTU2ZQ188I1O3_hURcqP8bCQsjg?usp=sharing

echo ==================================================
echo UNO RL Tournament Setup and Execution
echo ==================================================
echo Configuration:
echo   Number of Players: %NUM_PLAYERS%
echo   Games per Matchup: %NUM_GAMES%
echo ==================================================
echo.

REM Step 1: Check Python installation
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3 is not installed. Please install Python 3.8 or higher.
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   [OK] Found Python %PYTHON_VERSION%
echo.

REM Step 2: Create virtual environment if it doesn't exist
echo [2/6] Setting up virtual environment...
if not exist "venv\" (
    echo   Creating new virtual environment...
    python -m venv venv
    echo   [OK] Virtual environment created
) else (
    echo   [OK] Virtual environment already exists
)
echo.

REM Step 3: Activate virtual environment and install dependencies
echo [3/6] Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
echo   Installing project requirements...
pip install -r requirements.txt >nul 2>&1
echo   [OK] Dependencies installed
echo.

REM Step 4: Check if models exist, if not download automatically
echo [4/6] Checking for trained models...
if not exist "models\custom\" (
    set MODELS_MISSING=1
) else if not exist "models\rlcard\" (
    set MODELS_MISSING=1
) else (
    set MODELS_MISSING=0
)

if !MODELS_MISSING!==1 (
    echo   Models not found. Downloading from Google Drive...
    echo.
    
    REM Check if gdown is installed
    python -c "import gdown" >nul 2>&1
    if errorlevel 1 (
        echo   Installing gdown for Google Drive downloads...
        pip install gdown >nul 2>&1
        echo   [OK] gdown installed
    )
    
    REM Extract folder ID from URL
    set FOLDER_ID=1A7MnHhTU2ZQ188I1O3_hURcqP8bCQsjg
    
    echo   Downloading models ^(this may take a few minutes^)...
    echo   Source: Google Drive folder
    echo.
    
    REM Create models directory if it doesn't exist
    if not exist "models\" mkdir models
    
    REM Download the folder using gdown
    python -m gdown --folder "https://drive.google.com/drive/folders/!FOLDER_ID!" -O models/ --remaining-ok
    
    if errorlevel 1 (
        echo.
        echo   [ERROR] Download failed. Please download manually:
        echo     1. Visit: %MODELS_URL%
        echo     2. Download the 'custom' and 'rlcard' folders
        echo     3. Place them in the 'models\' directory
        echo.
        set /p CONTINUE="Continue anyway? (y/N): "
        if /i not "!CONTINUE!"=="y" (
            echo Exiting. Please download models and run again.
            exit /b 1
        )
    ) else (
        echo.
        echo   [OK] Models downloaded successfully
    )
) else (
    echo   [OK] Models already present
)
echo.

REM Step 5: Verify project structure
echo [5/6] Verifying project structure...
if not exist "src\evaluation\tournament_multiplayer.py" (
    echo   ERROR: Tournament script not found. Please ensure you're in the project root directory.
    exit /b 1
)
echo   [OK] Project structure verified
echo.

REM Step 6: Run tournament
echo [6/6] Starting tournament...
echo ==================================================
echo Running %NUM_PLAYERS%-player tournament with %NUM_GAMES% games per matchup
echo This may take a while depending on your configuration...
echo Results will be saved to: tournament_%NUM_PLAYERS%player_results.json
echo ==================================================
echo.

python src\evaluation\tournament_multiplayer.py --players %NUM_PLAYERS% --games %NUM_GAMES%

echo.
echo ==================================================
echo Tournament complete!
echo Results saved to: tournament_%NUM_PLAYERS%player_results.json
echo ==================================================

endlocal

