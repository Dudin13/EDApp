@echo off
title EDApp - Entrenamiento Incremental
set ROOT_DIR=%~dp0
cd /d %ROOT_DIR%

echo ======================================================
echo           EDApp - INICIANDO ENTRENAMIENTO
echo ======================================================
echo.

:: Verificar si el venv existe
if not exist "venv_cuda\Scripts\python.exe" (
    echo [ERROR] No se encontro el entorno virtual venv_cuda.
    echo Asegurate de que venv_cuda este instalado en la raiz.
    pause
    exit /b
)

:: Ejecutar el script de entrenamiento
"venv_cuda\Scripts\python.exe" "scripts\incremental_train.py"

echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
