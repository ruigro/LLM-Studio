# Create shortcut with custom rocket icon
$WScriptShell = New-Object -ComObject WScript.Shell
$ShortcutPath = "$PSScriptRoot\Launch LLM Studio.lnk"

# Delete old shortcut if exists
if (Test-Path $ShortcutPath) {
    Remove-Item $ShortcutPath -Force
}

# Check if rocket.ico exists, if not create it
$IconPath = Join-Path $PSScriptRoot "rocket.ico"
if (-not (Test-Path $IconPath)) {
    Write-Host "Creating rocket.ico..." -ForegroundColor Yellow
    python (Join-Path $PSScriptRoot "create_rocket_ico.py")
    if (-not (Test-Path $IconPath)) {
        Write-Host "ERROR: Failed to create rocket.ico" -ForegroundColor Red
        pause
        exit 1
    }
}

$Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "$PSScriptRoot\LAUNCHER.bat"
$Shortcut.WorkingDirectory = "$PSScriptRoot"
$Shortcut.Description = "Launch LLM Fine-tuning Studio"

# Use absolute path for custom rocket icon
$AbsoluteIconPath = (Resolve-Path $IconPath).Path
$Shortcut.IconLocation = $AbsoluteIconPath

$Shortcut.Save()

Write-Host ""
Write-Host "SUCCESS: Launcher shortcut created with custom rocket icon!" -ForegroundColor Green
Write-Host "Icon: $AbsoluteIconPath" -ForegroundColor Cyan
Write-Host ""

