@echo off
chcp 65001 >nul

REM Change to script directory
cd /d "%~dp0"

REM Find Python executable
set PYTHON_EXE=python

REM Check if python is in PATH
python --version >nul 2>&1
if errorlevel 1 (
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
            goto :python_found
        )
    )
    
    REM Python not found anywhere
    msg * "Python not found! Please install Python 3.8+ from: https://www.python.org/downloads/ and check 'Add Python to PATH' during installation."
    exit /b 1
)

:python_found

REM Check if first-time setup has been completed
if not exist ".setup_complete" (
    REM Create venv if needed (silently)
    if not exist .venv (
        "%PYTHON_EXE%" -m venv .venv >nul 2>&1
    )
    
    REM Activate venv and install PySide6 (silently)
    if exist .venv\Scripts\activate.bat (
        call .venv\Scripts\activate.bat >nul 2>&1
        python -m pip install --quiet --upgrade pip >nul 2>&1
        python -m pip install --quiet PySide6 >nul 2>&1
    )
    
    REM Launch setup wizard GUI (no console)
    start "" pythonw.exe first_run_setup.py
    exit /b 0
)

REM Normal app launch (no console)
if exist .venv\Scripts\pythonw.exe (
    start "" .venv\Scripts\pythonw.exe -m desktop_app.main
) else (
    start "" python -m desktop_app.main
)

exit
