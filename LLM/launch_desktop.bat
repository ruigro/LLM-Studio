@echo off
REM Launch LLM Fine-tuning Studio (Qt Desktop Version)
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
)
python -m desktop_app.main
pause

