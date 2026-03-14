@echo off
echo ======================================================
echo   PREPARANDO HERRAMIENTA DE REVISION EDUDIN
echo ======================================================

:: Verificar Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no esta instalado en el sistema.
    pause
    exit /b
)

echo [1/2] Instalando dependencias necesarias (Flask)...
pip install flask flask-cors

echo [2/2] Lanzando servidor...
start http://127.0.0.1:5000
python labeller_revision.py

pause
