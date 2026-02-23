@echo off
setlocal
cd /d "%~dp0"

REM Launch app with Python launcher if available
py -3 "drc_halftone_cmyk.py"
if %errorlevel%==0 goto :eof

REM Fallback to python on PATH
python "drc_halftone_cmyk.py"
if %errorlevel%==0 goto :eof

echo.
echo Failed to start the app.
echo Make sure Python 3 is installed and dependencies are installed.
echo Run: setup_env.bat
pause
