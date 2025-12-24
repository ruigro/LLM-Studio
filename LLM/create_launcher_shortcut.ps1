# Create shortcut with rocket icon
$WScriptShell = New-Object -ComObject WScript.Shell
$ShortcutPath = "$PSScriptRoot\Launch LLM Studio.lnk"

# Delete old shortcut if exists
if (Test-Path $ShortcutPath) {
    Remove-Item $ShortcutPath -Force
}

$Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "$PSScriptRoot\LAUNCHER.bat"
$Shortcut.WorkingDirectory = "$PSScriptRoot"
$Shortcut.Description = "Launch LLM Fine-tuning Studio"

# Use a RELIABLE Windows system icon instead of custom .ico file
# Icon 137 from imageres.dll is a star/launch icon that works on all Windows systems
$Shortcut.IconLocation = "%SystemRoot%\System32\imageres.dll,137"

$Shortcut.Save()

Write-Host ""
Write-Host "SUCCESS: Launcher shortcut created!" -ForegroundColor Green
Write-Host "Using Windows system icon (star/launch icon)" -ForegroundColor Cyan
Write-Host ""

