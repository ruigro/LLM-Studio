@echo off
REM ================================================================
REM   LLM FINE-TUNING STUDIO - ONE-CLICK LAUNCHER
REM   Double-click this file to launch the app
REM ================================================================

REM Try to find Python - check multiple locations
set PYTHON_EXE=

REM Check if python is in PATH
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_EXE=python
    goto :found_python
)

REM Check common Python install locations
for %%P in (
    "C:\Program Files\Python312\python.exe"
    "C:\Program Files\Python311\python.exe"
    "C:\Program Files\Python310\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
) do (
    if exist %%P (
        set PYTHON_EXE=%%P
        goto :found_python
    )
)

REM Python not found
echo ================================================================
echo ERROR: Python Not Found
echo ================================================================
echo.
echo Python 3.8 or higher is required but was not found.
echo.
echo Please install Python from:
echo https://www.python.org/downloads/
echo.
echo Make sure to check 'Add Python to PATH' during installation!
echo.
pause
exit /b 1

:found_python

REM Change to LLM directory
cd /d "%~dp0\LLM"

REM Check if venv exists
if not exist ".venv\Scripts\pythonw.exe" (
    echo ================================================================
    echo FIRST TIME SETUP
    echo ================================================================
    echo.
    echo Creating virtual environment and installing GUI library...
    echo This will take 2-3 minutes...
    echo.
    
    REM Create venv
    "%PYTHON_EXE%" -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    
    REM Install PySide6
    echo Installing GUI library...
    .venv\Scripts\python.exe -m pip install --quiet --upgrade pip
    .venv\Scripts\python.exe -m pip install PySide6==6.8.1
    if errorlevel 1 (
        echo ERROR: Failed to install GUI library!
        pause
        exit /b 1
    )
    
    echo.
    echo Setup complete!
    echo.
)

REM Ensure setup marker exists (skip the broken wizard)
if not exist ".setup_complete" (
    echo. > .setup_complete
)

REM Launch the app
start "" .venv\Scripts\pythonw.exe -m desktop_app.main

exit /b 0

