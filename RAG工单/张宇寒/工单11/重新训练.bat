@echo off
cd /d "%~dp0"
set PYTHON=C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe
"%PYTHON%" prepare.py
if errorlevel 1 goto :error
"%PYTHON%" train.py --epochs 60 --dimension 96
if errorlevel 1 goto :error
echo.
echo 训练完成，请运行“启动项目.bat”。
pause
exit /b 0
:error
echo.
echo 训练失败，请查看上方错误信息。
pause
exit /b 1

