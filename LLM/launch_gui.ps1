# Auto-launch LLM Fine-Tuning Studio
cd (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host ""
Write-Host "========================================"
Write-Host "  LLM Fine-Tuning Studio Launcher"
Write-Host "========================================"
Write-Host ""
Write-Host "Starting GUI... Your browser will open automatically."
Write-Host ""

# Start Streamlit and open browser
Start-Process "http://localhost:8501"
& ".\.venv\Scripts\streamlit.exe" run gui.py --server.port 8501

# Keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to start the GUI"
    pause
}

