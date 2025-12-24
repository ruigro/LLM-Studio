@echo off
chcp 65001 >nul
title 🚀 LLM Fine-tuning Studio Launcher
color 0D

REM Change to script directory
cd /d "%~dp0"

REM Find Python executable
set PYTHON_EXE=python

REM Check if python is in PATH
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found in PATH. Searching common install locations...
    
    REM Check common Python install locations
    for %%P in (
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python38\python.exe"
        "C:\Python312\python.exe"
        "C:\Python311\python.exe"
        "C:\Python310\python.exe"
        "C:\Python39\python.exe"
        "C:\Python38\python.exe"
        "%ProgramFiles%\Python312\python.exe"
        "%ProgramFiles%\Python311\python.exe"
        "%ProgramFiles%\Python310\python.exe"
        "%ProgramFiles%\Python39\python.exe"
        "%ProgramFiles%\Python38\python.exe"
    ) do (
        if exist %%P (
            set PYTHON_EXE=%%P
            echo Found Python at: %%P
            goto :python_found
        )
    )
    
    REM Python not found anywhere
    echo.
    echo ❌ Python not found!
    echo.
    echo Please install Python 3.8 or later from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: During installation, check the box:
    echo   "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

:python_found

REM Check if first-time setup has been completed
if not exist ".setup_complete" (
    REM Launch bootstrap GUI (uses tkinter, no deps needed)
    "%PYTHON_EXE%" bootstrap_setup.py
    
    if errorlevel 1 (
        echo.
        echo ❌ Setup failed! Please check the error message.
        pause
        exit /b 1
    )
    
    REM Exit after bootstrap (bootstrap will launch main setup wizard)
    exit /b 0
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
