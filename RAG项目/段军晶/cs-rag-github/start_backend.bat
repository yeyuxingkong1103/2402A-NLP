@echo off
rem ============================================================================
rem  cs-rag  FastAPI backend one-click launcher
rem ----------------------------------------------------------------------------
rem  This file is deliberately ASCII-only.
rem  cmd.exe reads .bat files using the OEM code page (936 on zh-CN Windows),
rem  so non-ASCII characters here would be garbled. All Chinese messages are
rem  printed by scripts/preflight_check.py instead, which we control.
rem
rem  Steps: check venv -> preflight dependency check -> start backend
rem  Usage: double-click, or  start_backend.bat --yes
rem ============================================================================

chcp 65001 >nul
setlocal
title cs-rag Backend

rem Force Python stdio to UTF-8.
rem Without this, Python falls back to the locale encoding (GBK on zh-CN Windows)
rem whenever stdout is not a real console (e.g. output redirected to a pipe or
rem file), which garbles the Chinese log lines from backend/logging_config.py.
set "PYTHONIOENCODING=utf-8"

rem Switch to the directory containing this script, so double-clicking works
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo.
    echo [ERROR] Python virtual environment not found:
    echo         %PY%
    echo.
    echo   Create it first:
    echo     python -m venv --system-site-packages .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

rem ---- Preflight: dependency reachability + access URLs ----
"%PY%" -m scripts.preflight_check %*
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)

rem ---- Start the backend in the foreground ----
echo Starting backend... (press Ctrl+C to stop)
echo.

"%PY%" -m backend.main

echo.
echo Backend stopped.
pause
