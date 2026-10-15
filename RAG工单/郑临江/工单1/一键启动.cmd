@echo off
chcp 65001 >nul
title PDF问答系统 - 工单1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0部署\启动本地大模型.ps1"
pause
