@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "taskPython=python"
if exist "%~dp0..\..\..\tmp\venv1112\Scripts\python.exe" set "taskPython=%~dp0..\..\..\tmp\venv1112\Scripts\python.exe"
"%taskPython%" "研发\train.py" train
pause
