@echo off
setlocal
cd /d "%~dp0"

echo Installing/upgrading pip...
py -3 -m pip install --upgrade pip
if %errorlevel% neq 0 python -m pip install --upgrade pip

echo Installing requirements...
py -3 -m pip install -r requirements.txt
if %errorlevel% neq 0 python -m pip install -r requirements.txt

echo.
echo Setup complete.
echo Use run_app.bat to start the program.
pause
