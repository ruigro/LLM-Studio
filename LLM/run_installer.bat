@echo off
REM Bootstrap launcher for LLM Fine-tuning Studio Installer
REM ALWAYS runs installer from bootstrap\.venv, never from LLM\.venv

setlocal enabledelayedexpansion

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Locate project root (parent of LLM if we're in LLM, otherwise current dir)
cd /d "%SCRIPT_DIR%"
if exist "..\bootstrap" if exist "installer_gui.py" (
    REM We're in LLM folder, parent is project root
    set "PROJECT_ROOT=%SCRIPT_DIR%..\"
    set "LLM_DIR=%SCRIPT_DIR%"
) else (
    REM We're in project root
    set "PROJECT_ROOT=%SCRIPT_DIR%"
    set "LLM_DIR=%SCRIPT_DIR%LLM\"
)

set "BOOTSTRAP_VENV=%PROJECT_ROOT%\bootstrap\.venv"
set "BOOTSTRAP_PYTHON=%BOOTSTRAP_VENV%\Scripts\python.exe"
set "INSTALLER_SCRIPT=%LLM_DIR%\installer_gui.py"

REM Check if bootstrap venv exists
if not exist "%BOOTSTRAP_PYTHON%" (
    echo Creating bootstrap venv...
    
    REM Find system Python
    set "SYSTEM_PYTHON="
    for %%P in (py -3.10 py -3 python3.10 python3 python) do (
        %%P --version >nul 2>&1
        if !ERRORLEVEL! EQU 0 (
            for /f "delims=" %%E in ('%%P -c "import sys; print(sys.executable)" 2^>nul') do set "SYSTEM_PYTHON=%%E"
            if defined SYSTEM_PYTHON goto :found_python
        )
    )
    
    :found_python
    if not defined SYSTEM_PYTHON (
        echo ERROR: Cannot find system Python!
        echo Please install Python 3.10+ and try again.
        pause
        exit /b 1
    )
    
    REM Create bootstrap venv
    "%SYSTEM_PYTHON%" -m venv "%BOOTSTRAP_VENV%"
    if !ERRORLEVEL! NEQ 0 (
        echo ERROR: Failed to create bootstrap venv!
        pause
        exit /b 1
    )
    
    REM Install installer dependencies
    echo Installing installer dependencies...
    "%BOOTSTRAP_PYTHON%" -m pip install -U pip --quiet
    if exist "%LLM_DIR%\installer_requirements.txt" (
        "%BOOTSTRAP_PYTHON%" -m pip install -r "%LLM_DIR%\installer_requirements.txt" --quiet
    )
)

REM Run installer from bootstrap
echo Starting installer from bootstrap...
"%BOOTSTRAP_PYTHON%" "%INSTALLER_SCRIPT%"

endlocal

