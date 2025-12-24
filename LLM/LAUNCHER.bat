@echo off
chcp 65001 >nul
title 🚀 LLM Fine-tuning Studio Launcher
color 0D

REM Change to script directory
cd /d "%~dp0"

REM Check if first-time setup has been completed
if not exist ".setup_complete" (
    echo.
    echo ═══════════════════════════════════════════════════════
    echo    🚀 LLM Fine-tuning Studio - First Run Setup 🚀
    echo ═══════════════════════════════════════════════════════
    echo.
    echo ⚙️  First-time setup required...
    echo This will detect your hardware and install all dependencies.
    echo Please wait, this may take 5-15 minutes.
    echo.
    
    REM Create virtual environment if it doesn't exist
    if not exist .venv (
        echo Creating virtual environment...
        python -m venv .venv
        if errorlevel 1 (
            echo ❌ Failed to create virtual environment!
            echo Make sure Python 3.8+ is installed and in PATH.
            pause
            exit /b 1
        )
        echo ✓ Virtual environment created
        echo.
    )
    
    REM Activate virtual environment
    if exist .venv\Scripts\activate.bat (
        call .venv\Scripts\activate.bat
        echo ✓ Virtual environment activated
        echo.
    )
    
    REM Install PySide6 for the setup wizard GUI
    echo Installing PySide6 for setup wizard...
    python -m pip install --quiet PySide6
    if errorlevel 1 (
        echo ⚠️ Warning: Failed to install PySide6
    )
    echo.
    
    REM Run first-time setup
    python first_run_setup.py
    
    if errorlevel 1 (
        echo.
        echo ❌ Setup failed! Please check the logs above.
        echo.
        echo Common issues:
        echo  - No internet connection
        echo  - Python not in PATH
        echo  - Antivirus blocking downloads
        echo.
        echo You can retry by running this launcher again.
        pause
        exit /b 1
    )
    
    echo.
    echo ✅ Setup completed successfully!
    echo.
)

REM Normal app launch
echo.
echo ═══════════════════════════════════════════════════════
echo    🚀 LLM Fine-tuning Studio Launcher 🚀
echo ═══════════════════════════════════════════════════════
echo.
echo Starting application...
echo.

if exist .venv\Scripts\activate.bat (
  echo ✓ Activating virtual environment...
  call .venv\Scripts\activate.bat
)

REM Launch with pythonw.exe (no console) if available, else use python.exe
if exist .venv\Scripts\pythonw.exe (
    start "" .venv\Scripts\pythonw.exe -m desktop_app.main
) else (
    start "" python -m desktop_app.main
)

REM Exit immediately (don't wait for the app)
exit
