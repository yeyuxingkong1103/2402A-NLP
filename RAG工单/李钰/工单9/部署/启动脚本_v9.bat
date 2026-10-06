@echo off
REM ============================================================
REM V9 Graph RAG 优化 - Windows 启动脚本
REM 工单编号: 人工智能 NLP-RAG-Graph RAG 优化任务
REM ============================================================

chcp 65001 >nul
cd /d "%~dp0\..\研发"

echo ============================================================
echo   Graph RAG V9 优化版 启动中...
echo   工单编号: 人工智能 NLP-RAG-Graph RAG 优化任务
echo   端口: 5008
echo   验收: Context Precision≥0.80, Context Recall≥0.90
echo ============================================================

python app_v9.py

pause
