@echo off
title Maske // Auto-Launcher
color 0b

echo =======================================================
echo          MASKE SECURITY SYSTEMS // INITIALIZING
echo =======================================================
echo.

:: Check if Python is accessible
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! 
    echo Please install Python 3.10 or 3.11 and add it to your PATH.
    echo.
    pause
    exit
)

echo [+] Checking Verification Protocols...
echo [+] Updating Neural Dependencies...
pip install -r requirements.txt

echo.
echo [+] Dependencies Synchronized.
echo [+] Launching Cloak Engine...
echo.

python maske_app.py

if %errorlevel% neq 0 (
    echo.
    echo [CRITICAL FAILURE] Application crashed. See error above.
)

pause
