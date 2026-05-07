@echo off
title EDUDIN Pro Labeller
echo ======================================================
echo           EDUDIN Pro Labeller - Starting
echo ======================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please run install.bat first or install Python.
    pause
    exit /b
)

echo [INFO] Starting Flask server...
start "" /b python labeller_app.py

echo [WAIT] Waiting for server (3s)...
timeout /t 3 /nobreak >nul

echo [URL] Opening browser at http://localhost:5000
start http://localhost:5000

echo.
echo ======================================================
echo    Labeller is running. DO NOT CLOSE THIS WINDOW.
echo ======================================================
pause
