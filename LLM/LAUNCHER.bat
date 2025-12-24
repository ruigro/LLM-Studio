@echo off
chcp 65001 >nul
title 🚀 LLM Fine-tuning Studio Launcher
color 0D
echo.
echo ═══════════════════════════════════════════════════════
echo    🚀 LLM Fine-tuning Studio Launcher 🚀
echo ═══════════════════════════════════════════════════════
echo.
echo Starting application...
echo.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat (
  echo ✓ Activating virtual environment...
  call .venv\Scripts\activate.bat
)
python -m desktop_app.main
pause

