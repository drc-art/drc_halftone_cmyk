@echo off
setlocal
cd /d "%~dp0"

REM One-click launcher: install deps if missing, then run app.
py -3 -c "import PySide6" 1>nul 2>nul
if %errorlevel% neq 0 (
  echo First run setup: installing dependencies...
  call setup_env.bat
)

call run_app.bat
