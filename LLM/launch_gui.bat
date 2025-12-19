@echo off
REM Auto-launch LLM Fine-Tuning Studio
cd /d "%~dp0"

echo.
echo ========================================
echo   LLM Fine-Tuning Studio Launcher
echo ========================================
echo.
echo Starting GUI... Your browser will open automatically.
echo.

REM Start Streamlit in background and open browser
start "" http://localhost:8501
.\.venv\Scripts\streamlit.exe run gui.py --server.port 8501

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ERROR: Failed to start the GUI
    pause
)

