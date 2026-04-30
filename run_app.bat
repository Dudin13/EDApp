@echo off
echo ====================================================
echo   EDApp v2 - Iniciando Plataforma de Analisis
echo ====================================================
echo.

if not exist "venv" (
    echo [ERROR] No se encontro el entorno virtual. Ejecuta 'install.bat' primero.
    pause
    exit /b
)

call .\venv\Scripts\activate
streamlit run app/app.py

pause
