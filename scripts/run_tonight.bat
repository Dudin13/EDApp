@echo off
cd /d %~dp0

:: Si se llama con el flag --internal, ejecuta la secuencia real.
if "%~1"=="--internal" goto :run_internal

:: ── Wrapper para guardar logs y mostrar por pantalla simultáneamente ──
echo Iniciando script maestro nocturno...
if not exist "..\output" mkdir "..\output"
powershell -Command "cmd.exe /c '%~f0' --internal 2>&1 | Tee-Object -FilePath '..\output\tonight_log.txt'"
exit /b

:run_internal
echo ============================================================
echo [1/3] Ejecutando Train_v7...
echo ============================================================
call ..\Train_v7.bat device=0 fraction=0.7
if %errorlevel% neq 0 (
    echo [ERROR] Train_v7 falló con codigo %errorlevel%. Deteniendo secuencia.
    goto :shutdown
)

echo.
echo ============================================================
echo [2/3] Ejecutando extraccion de frames tacticos...
echo ============================================================
call python extract_tactical_frames.py
if %errorlevel% neq 0 (
    echo [ERROR] Extraccion de frames falló con codigo %errorlevel%. Deteniendo secuencia.
    goto :shutdown
)

echo.
echo ============================================================
echo [3/3] Probando LocateAnything...
echo ============================================================
call python test_locate_anything.py
if %errorlevel% neq 0 (
    echo [ERROR] LocateAnything falló con codigo %errorlevel%. Deteniendo secuencia.
    goto :shutdown
)

echo.
echo ============================================================
echo [EXITO] Todas las tareas completadas sin errores.
echo ============================================================

:shutdown
echo.
echo Apagando el equipo en 60 segundos... (Escribe "shutdown /a" en otra terminal para cancelar)
shutdown /s /t 60
