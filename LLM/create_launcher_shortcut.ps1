# Create shortcut pointing to launcher.exe (with embedded rocket icon)
$WScriptShell = New-Object -ComObject WScript.Shell
$ShortcutPath = "$PSScriptRoot\Launch LLM Studio.lnk"

# Delete old shortcut if exists
if (Test-Path $ShortcutPath) {
    Remove-Item $ShortcutPath -Force
}

# Check if launcher.exe exists
$LauncherPath = Join-Path $PSScriptRoot "launcher.exe"
if (-not (Test-Path $LauncherPath)) {
    Write-Host "ERROR: launcher.exe not found!" -ForegroundColor Red
    Write-Host "Please compile it first by running: build_launcher.bat" -ForegroundColor Yellow
    pause
    exit 1
}

# Create the shortcut
$Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "$PSScriptRoot\launcher.exe"
$Shortcut.WorkingDirectory = "$PSScriptRoot"
$Shortcut.Description = "Launch LLM Fine-tuning Studio"

# No need to set IconLocation - the .exe already has the icon embedded
# But we can explicitly point to it for clarity
$Shortcut.IconLocation = "$PSScriptRoot\launcher.exe,0"

$Shortcut.Save()

# Force icon refresh by touching the shortcut file
$shortcutFile = Get-Item $ShortcutPath
$shortcutFile.LastWriteTime = Get-Date

Write-Host ""
Write-Host "SUCCESS: Launcher shortcut created!" -ForegroundColor Green
Write-Host "Target: $LauncherPath" -ForegroundColor Cyan
Write-Host "Icon: Embedded in launcher.exe (rocket icon)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Refreshing icon cache..." -ForegroundColor Yellow
ie4uinit.exe -show
Write-Host ""
Write-Host "You can now use 'Launch LLM Studio.lnk' to start the application." -ForegroundColor Green
Write-Host "The console window will close automatically after launch." -ForegroundColor Cyan
Write-Host ""
