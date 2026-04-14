@echo off
echo Abriendo el Roadmap Unificado de EDApp...
:: Buscamos la ruta absoluta del archivo relativa al script
set "ROADMAP_PATH=%~dp0docs\roadmap_unified.html"
if exist "%ROADMAP_PATH%" (
    start "" "%ROADMAP_PATH%"
) else (
    echo [ERROR] No se pudo encontrar el archivo en: %ROADMAP_PATH%
    pause
)
exit
