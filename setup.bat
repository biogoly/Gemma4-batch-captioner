@echo off
setlocal
cd /d "%~dp0"

echo Creating Python environment...
py -3.11 -m venv .venv >nul 2>&1
if errorlevel 1 py -3.12 -m venv .venv >nul 2>&1
if errorlevel 1 python -m venv .venv
if errorlevel 1 (
    echo ERROR: Could not create .venv. Install Python 3.11 or newer.
    pause
    exit /b 1
)

echo Installing requirements...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

if not exist server_config.toml copy /Y server_config.example.toml server_config.toml >nul
if not exist input mkdir input
if not exist output mkdir output
if not exist logs mkdir logs

echo.
echo Setup complete.
echo Edit server_config.toml with your llama.cpp and model paths before using run_all.bat.
pause
exit /b 0

:failed
echo.
echo ERROR: Setup failed.
pause
exit /b 1
