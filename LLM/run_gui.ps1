# Setup and run the LLM Fine-tuning GUI
# This script will check for Python, create a virtual environment, install dependencies, and run Streamlit

# Navigate to script directory
Set-Location -Path $PSScriptRoot

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  LLM Fine-tuning Studio - Setup and Launch" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
$pythonCmd = $null
$pythonCommands = @("python", "python3", "py")

foreach ($cmd in $pythonCommands) {
    try {
        $result = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $cmd
            break
        }
    } catch {
        continue
    }
}

if ($null -eq $pythonCmd) {
    Write-Host "ERROR: Python not found in PATH!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[1/4] Python found:" -ForegroundColor Green
& $pythonCmd --version
Write-Host ""

# Check if virtual environment exists
if (!(Test-Path ".venv")) {
    Write-Host "[2/4] Creating virtual environment..." -ForegroundColor Yellow
    & $pythonCmd -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "[2/4] Virtual environment already exists." -ForegroundColor Green
}
Write-Host ""

# Activate virtual environment
Write-Host "[3/4] Activating virtual environment..." -ForegroundColor Yellow
$activateScript = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
} else {
    Write-Host "ERROR: Failed to find activation script!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# Install/upgrade dependencies
Write-Host "[4/4] Installing dependencies..." -ForegroundColor Yellow
& python -m pip install --upgrade pip --quiet
& pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Dependencies installed successfully." -ForegroundColor Green
Write-Host ""

# Check GPU availability
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Checking GPU..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
& python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>$null
Write-Host ""

# Launch Streamlit
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Starting GUI..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The GUI will open in your browser at http://localhost:8501" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow
Write-Host ""

# Use full path to streamlit in virtual environment
$streamlitPath = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"
if (Test-Path $streamlitPath) {
    Write-Host "Launching GUI..." -ForegroundColor Green
    & $streamlitPath run gui.py
} else {
    Write-Host "❌ Streamlit not found in virtual environment!" -ForegroundColor Red
    Write-Host "Try running manually: .\.venv\Scripts\streamlit.exe run gui.py" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
}

