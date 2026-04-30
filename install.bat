@echo off
echo ====================================================
echo   EDApp v2 - Instalar Entorno de Analisis
echo ====================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no esta instalado. Por favor instala Python 3.10 o superior.
    pause
    exit /b
)

echo [1/3] Creando entorno virtual (venv)...
python -m venv venv

echo [2/3] Activando entorno e instalando librerias...
call .\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo [3/3] Verificando modelos de IA...
if exist "C:\D\New folder\detect_players.pt" (
    echo [OK] Modelos encontrados en C:\D\New folder
) else (
    echo [AVISO] No se encontraron modelos en C:\D\New folder. 
    echo Asegurese de copiar los archivos .pt antes de procesar videos.
)

echo.
echo ====================================================
echo   INSTALACION COMPLETADA
echo   Usa 'run_app.bat' para iniciar la plataforma.
echo ====================================================
pause
