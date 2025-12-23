@echo off
REM Auto-launch LLM Fine-Tuning Studio
cd /d "%~dp0"

echo.
echo ========================================
echo   LLM Fine-Tuning Studio Launcher
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo ERROR: Virtual environment not found!
    echo Please run install_python.bat or run_gui.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

REM Check if streamlit.exe exists
if not exist ".venv\Scripts\streamlit.exe" (
    echo ERROR: Streamlit not found in virtual environment!
    echo Please install dependencies: pip install streamlit
    echo.
    pause
    exit /b 1
)

REM Check if gui.py exists
if not exist "gui.py" (
    echo ERROR: gui.py not found!
    echo Current directory: %CD%
    echo.
    pause
    exit /b 1
)

echo Starting GUI... Your browser will open automatically.
echo.

REM Kill any existing Streamlit processes on port 8501
echo Checking for existing Streamlit processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do (
    echo Killing process %%a on port 8501...
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

REM Activate virtual environment and run Streamlit
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)

REM Run Streamlit and capture exit code
echo Starting Streamlit server...
echo Please wait 5-10 seconds for the server to start...
echo.
echo The browser will open automatically once Streamlit is ready.
echo If you see "This site can't be reached", wait a few more seconds and refresh.
echo.
echo Press Ctrl+C to stop the server.
echo.

REM Use Python launcher (has better error handling and port cleanup)
echo Using Python launcher...
echo.

REM Run launcher.py with venv Python
.venv\Scripts\python.exe launcher.py
set STREAMLIT_EXIT=%ERRORLEVEL%

REM Keep window open to show any errors
if %STREAMLIT_EXIT% NEQ 0 (
    echo.
    echo ========================================
    echo ERROR: Streamlit exited with code %STREAMLIT_EXIT%
    echo ========================================
    echo.
    echo Common issues:
    echo 1. Port 8501 may be in use
    echo    Check with: netstat -ano ^| findstr :8501
    echo 2. Check if gui.py has syntax errors
    echo 3. Make sure all dependencies are installed: pip install -r requirements.txt
    echo 4. Try running manually: .venv\Scripts\python.exe -m streamlit run gui.py
    echo.
    pause
)

