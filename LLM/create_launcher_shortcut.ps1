# Create shortcut with rocket icon
$WScriptShell = New-Object -ComObject WScript.Shell
$ShortcutPath = "$PSScriptRoot\🚀 Launch LLM Studio.lnk"

# Delete old shortcut if exists
if (Test-Path $ShortcutPath) {
    Remove-Item $ShortcutPath -Force
}

$Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "$PSScriptRoot\LAUNCHER.bat"
$Shortcut.WorkingDirectory = "$PSScriptRoot"
$Shortcut.Description = "Launch LLM Fine-tuning Studio"

# Use the custom rocket.ico file
$Shortcut.IconLocation = "$PSScriptRoot\rocket.ico"

$Shortcut.Save()

Write-Host ""
Write-Host "SUCCESS: Launcher shortcut created with rocket icon!" -ForegroundColor Green
Write-Host ""

