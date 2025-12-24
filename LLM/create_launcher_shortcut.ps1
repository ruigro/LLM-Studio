# Create shortcut with rocket icon
$WScriptShell = New-Object -ComObject WScript.Shell
$ShortcutPath = "$PSScriptRoot\Launch LLM Studio.lnk"

# Verify icon file exists
$IconPath = Join-Path $PSScriptRoot "rocket.ico"
if (-not (Test-Path $IconPath)) {
    Write-Host ""
    Write-Host "ERROR: rocket.ico not found!" -ForegroundColor Red
    Write-Host "The icon file should be in the same directory as this script." -ForegroundColor Yellow
    Write-Host "Path checked: $IconPath" -ForegroundColor Yellow
    Write-Host ""
    pause
    exit 1
}

# Delete old shortcut if exists
if (Test-Path $ShortcutPath) {
    Remove-Item $ShortcutPath -Force
}

$Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "$PSScriptRoot\LAUNCHER.bat"
$Shortcut.WorkingDirectory = "$PSScriptRoot"
$Shortcut.Description = "Launch LLM Fine-tuning Studio"

# Use absolute path for icon
$AbsoluteIconPath = Resolve-Path $IconPath
$Shortcut.IconLocation = $AbsoluteIconPath.Path

$Shortcut.Save()

Write-Host ""
Write-Host "SUCCESS: Launcher shortcut created with rocket icon!" -ForegroundColor Green
Write-Host "Icon path: $($AbsoluteIconPath.Path)" -ForegroundColor Cyan
Write-Host ""

# Refresh Windows icon cache
Write-Host "Refreshing icon cache..." -ForegroundColor Yellow
ie4uinit.exe -show
Start-Sleep -Seconds 1

Write-Host ""
Write-Host "Done! If icon doesn't appear, restart Explorer:" -ForegroundColor Cyan
Write-Host "  taskkill /f /im explorer.exe & start explorer.exe" -ForegroundColor Gray
Write-Host ""

