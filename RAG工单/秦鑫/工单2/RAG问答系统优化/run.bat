@echo off
setlocal
set PYTHONUTF8=1
if "%RAG_PORT%"=="" set RAG_PORT=4174
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 -c "import pypdf" >nul 2>nul
  if errorlevel 1 (
    echo Installing the PDF parser dependency...
    py -3 -m pip install -r requirements.txt
    if errorlevel 1 goto :error
  )
  py -3 server.py
  goto :end
)
python -c "import pypdf" >nul 2>nul
if errorlevel 1 (
  echo Python 3 and pypdf are required. Install Python 3, then run: python -m pip install -r requirements.txt
  pause
  exit /b 1
)
python server.py
goto :end
:error
echo Dependency installation failed. See README.md for manual setup.
pause
:end
endlocal
