# Simple GUI Launcher - assumes virtual environment is already set up
# Run this to start the Streamlit GUI

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Starting LLM Fine-tuning GUI" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (!(Test-Path ".venv")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Run .\install_python.ps1 first to set up the environment." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to activate virtual environment!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Launching GUI" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The GUI will open in your browser at: http://localhost:8501" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow
Write-Host ""

# Use full path to streamlit in virtual environment
$streamlitPath = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"
if (Test-Path $streamlitPath) {
    Write-Host "Starting Streamlit..." -ForegroundColor Green
    & $streamlitPath run gui.py
} else {
    Write-Host "❌ Streamlit not found in virtual environment!" -ForegroundColor Red
    Write-Host "Try reinstalling: .\install_python.ps1" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
}
