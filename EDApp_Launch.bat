@echo off
title EDApp Launcher
cd /d C:\apped
echo [INFO] Activando venv_cuda...
call venv_cuda\Scripts\activate
echo [INFO] Lanzando Streamlit en puerto 8501...
start http://localhost:8501
streamlit run app/app.py
pause
