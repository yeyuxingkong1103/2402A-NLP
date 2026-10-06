@echo off
chcp 65001 >nul
cd /d "%~dp0"
docker network inspect zhaozhen-finance-net >nul 2>&1 || docker network create zhaozhen-finance-net
docker volume inspect zhaozhen-finance-data >nul 2>&1 || docker volume create zhaozhen-finance-data
docker compose up -d --build
if errorlevel 1 (pause & exit /b 1)
start "" http://127.0.0.1:8509
pause
