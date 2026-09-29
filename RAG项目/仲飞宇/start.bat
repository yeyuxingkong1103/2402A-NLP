@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title RAG roleplay system - launcher

rem ===========================================================================
rem  Windows double-click entry point. The real work is done by
rem  scripts/start_all.sh (dependency self-check -> auto-start what can be
rem  auto-started -> start the app -> verify /health). This file only:
rem    1) locates WSL and converts the current Windows dir to a WSL path
rem       via wslpath, so it works from any drive
rem    2) forwards arguments to start_all.sh and translates its exit code
rem    3) opens the browser once the app is up
rem
rem  Usage:
rem    double-click          start        (== bash scripts/start_all.sh)
rem    start.bat --no-start  self-check only, do not start the app
rem    start.bat --strict    non-zero exit if any component is degraded
rem    start.bat stop        stop the service (== bash scripts/shutdown.sh)
rem
rem  WHY THIS FILE IS ASCII-ONLY: cmd.exe tokenizes a .bat by the *console code
rem  page*, not by the file's encoding. With a UTF-8 file containing Chinese,
rem  lines get split at arbitrary bytes -- reproduced on this machine: an echo
rem  line holding ASCII square brackets was cut in half and executed as a bogus
rem  command, and a rem line with ASCII parentheses sent the script into a loop.
rem  Chinese output is fine though: chcp 65001 above makes the UTF-8 text coming
rem  from the WSL side (start_all.sh, shutdown.sh) render correctly, and those
rem  scripts do all the user-facing talking. Keep new lines in here ASCII-only.
rem
rem  Why not `wsl --cd`: that needs a recent WSL build and fails silently on
rem  older ones; `cd $(wslpath -a '...')` works everywhere and needs no extra
rem  quoting for paths with spaces.
rem ===========================================================================

where wsl >nul 2>nul
if errorlevel 1 (
    echo.
    echo   [ERROR] wsl.exe not found - this project runs inside WSL2.
    echo           Install it from an admin PowerShell:  wsl --install
    echo.
    set "RC=1"
    goto :end
)

rem Deliberately `goto` instead of an if(...) block: inside a block, %ERRORLEVEL%
rem is expanded when the whole block is parsed, so it would hold the value from
rem *before* wsl ran and the exit code would always be wrong.
if /i "%~1"=="stop" goto :stop
rem --no-start only runs the dependency self-check: exit code 0 then means "checks
rem passed", NOT "the app is up". Must not be reported as a successful start
rem (doing so printed "Done" and opened a browser at a port nobody was serving).
rem Checking the two realistic positions beats `for %%a in (%*)`, which is a
rem syntax error when there are no arguments at all.
set "NO_START=0"
if /i "%~1"=="--no-start" set "NO_START=1"
if /i "%~2"=="--no-start" set "NO_START=1"
if /i "%~1"=="-h" goto :help
if /i "%~1"=="--help" goto :help
if /i "%~1"=="/?" goto :help

echo.
echo   Starting the RAG roleplay system - first/cold start can take 1-3 minutes
echo   (the reranker service has to load its weights). Progress is printed below
echo   in Chinese by scripts/start_all.sh.
echo.
wsl bash -lc "cd $(wslpath -a '%CD%') && exec bash scripts/start_all.sh %*"
set "RC=%ERRORLEVEL%"
goto :report

:stop
echo.
wsl bash -lc "cd $(wslpath -a '%CD%') && exec bash scripts/shutdown.sh"
set "RC=%ERRORLEVEL%"
set "MODE=stop"
goto :report

:report
echo.
if "%MODE%"=="stop" goto :report_stop
if not "%RC%"=="0" goto :report_fail
if "%NO_START%"=="1" goto :report_checked
echo   ---------------------------------------------------------------
echo   Done. The service keeps running in the background - closing this
echo   window does NOT stop it.
if not defined RAG_NO_BROWSER (
    echo   Opening http://localhost:8000/
    start "" http://localhost:8000/
) else (
    echo   Browser skipped - open http://localhost:8000/ manually.
)
echo   Port follows PORT in .env - 8000 by default.
echo   Stop it with:  start.bat stop
echo   ---------------------------------------------------------------
goto :end

:report_checked
echo   ---------------------------------------------------------------
echo   Self-check finished; the service was NOT started (--no-start).
echo   Run this file again with no arguments to start the service.
echo   ---------------------------------------------------------------
goto :end

:report_stop
if "%RC%"=="0" (
    echo   Stopped.
) else (
    echo   [ERROR] shutdown failed, exit code %RC% - see the output above.
)
goto :end

:report_fail
if "%RC%"=="1" (
    echo   [ERROR] the app did not come up. Last 20 lines of logs/uvicorn.out:
    echo.
    wsl bash -lc "cd $(wslpath -a '%CD%') && tail -20 logs/uvicorn.out 2>/dev/null"
    echo.
    echo   Scroll up: start_all.sh prints the failing step with its reason.
    goto :end
)
if "%RC%"=="2" (
    echo   [ERROR] bad argument. Use:  --no-start / --strict / stop
    goto :end
)
if "%RC%"=="3" (
    echo   [DEGRADED] some component was unavailable - see the degraded list
    echo   printed above by start_all.sh. The service still runs.
    goto :end
)
echo   [ERROR] exit code %RC% - scroll up for the details.

:end
echo.
echo   Press any key to close this window...
pause >nul
exit /b %RC%

:help
set "RC=0"
echo.
echo   Usage:  start.bat [--no-start ^| --strict ^| stop]
echo.
echo     (no args)    start: self-check deps, auto-start what can be, then the app
echo     --no-start   self-check only
echo     --strict     non-zero exit if any component is degraded
echo     stop         stop the service
echo.
echo   Set RAG_NO_BROWSER=1 to skip opening the browser after a start.
echo.
goto :end
