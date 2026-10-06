@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set "PYTHONPATH=%~dp0..\..	mp\deps78;%PYTHONPATH%"
cd /d "%~dp0"
python -m streamlit run app.py --server.port 8508 --server.address 127.0.0.1 --browser.gatherUsageStats false
pause
