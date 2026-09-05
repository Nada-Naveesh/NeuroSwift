@echo off
set PYTHONIOENCODING=utf-8
chcp 65001 >nul
echo ======================================================
echo Starting NeuroSwift Streamlit Web App...
echo ======================================================
.\.venv\Scripts\streamlit.exe run demo/app.py
pause
