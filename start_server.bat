@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python environment not found. Run setup.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" run_all.py --server-only %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXIT_CODE%
