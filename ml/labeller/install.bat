@echo off
title EDUDIN Pro Labeller - Installation
echo ======================================================
echo       EDUDIN Pro Labeller - Installing Dependencies
echo ======================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10 or higher.
    echo Visit https://www.python.org/downloads/
    pause
    exit /b
)

echo [INFO] Installing requirements...
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo ======================================================
    echo          Installation completed successfully!
    echo ======================================================
    echo [INFO] You can now use Labeller.bat to start the app.
) else (
    echo.
    echo [ERROR] Installation failed. Please check the errors above.
)

echo.
pause
