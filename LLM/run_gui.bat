@echo off
REM Setup and run the LLM Fine-tuning GUI
REM This script will check for Python, create a virtual environment, install dependencies, and run Streamlit

cd /d "%~dp0"

echo ================================================
echo   LLM Fine-tuning Studio - Setup and Launch
echo ================================================
echo.

REM Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found in PATH!
    echo.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/4] Python found: 
python --version
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo [2/4] Creating virtual environment...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo [2/4] Virtual environment already exists.
)
echo.

REM Activate virtual environment
echo [3/4] Activating virtual environment...
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo.

REM Install/upgrade dependencies
echo [4/4] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)
echo.

REM Check GPU availability
echo ================================================
echo   Checking GPU...
echo ================================================
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>nul
echo.

REM Launch Streamlit
echo ================================================
echo   Starting GUI...
echo ================================================
echo.
echo The GUI will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server.
echo.

if exist ".venv\Scripts\streamlit.exe" (
    echo Starting GUI...
    .venv\Scripts\streamlit.exe run gui.py
) else (
    echo ERROR: Streamlit not found in virtual environment!
    echo Try running manually: .venv\Scripts\streamlit.exe run gui.py
    pause
)

pause

