# Create shortcut with custom rocket icon
$WScriptShell = New-Object -ComObject WScript.Shell
$ShortcutPath = "$PSScriptRoot\Launch LLM Studio.lnk"

# Delete old shortcut if exists
if (Test-Path $ShortcutPath) {
    Remove-Item $ShortcutPath -Force
}

# Check if rocket.ico exists
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

# CRITICAL: Use icon with explicit index 0 and quotes for path with spaces
$AbsoluteIconPath = (Resolve-Path $IconPath).Path
$Shortcut.IconLocation = "`"$AbsoluteIconPath`",0"

$Shortcut.Save()

# Force icon refresh by touching the shortcut file
$shortcutFile = Get-Item $ShortcutPath
$shortcutFile.LastWriteTime = Get-Date

Write-Host ""
Write-Host "SUCCESS: Launcher shortcut created!" -ForegroundColor Green
Write-Host "Icon: $AbsoluteIconPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "Refreshing icon cache..." -ForegroundColor Yellow
ie4uinit.exe -show
Write-Host ""
Write-Host "If icon still doesn't show, right-click the shortcut > Properties > Change Icon > OK" -ForegroundColor Cyan
Write-Host ""

