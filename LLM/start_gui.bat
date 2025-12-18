@echo off
REM Simple GUI Launcher - assumes virtual environment is already set up
REM Run this to start the Streamlit GUI

echo ================================================
echo   Starting LLM Fine-tuning GUI
echo ================================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo ERROR: Virtual environment not found!
    echo Run install_python.bat first to set up the environment.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Launching GUI
echo ================================================
echo.
echo The GUI will open in your browser at: http://localhost:8501
echo Press Ctrl+C to stop the server.
echo.

REM Use full path to streamlit in virtual environment
if exist ".venv\Scripts\streamlit.exe" (
    echo Starting Streamlit...
    .venv\Scripts\streamlit.exe run gui.py
) else (
    echo ERROR: Streamlit not found in virtual environment!
    echo Try reinstalling: install_python.bat
    pause
)
