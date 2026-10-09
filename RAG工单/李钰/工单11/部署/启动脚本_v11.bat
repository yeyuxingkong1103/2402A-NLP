@echo off
REM ============================================================
REM V11 Embedding 微调 - Windows 启动脚本
REM 工单编号: 人工智能 NLP-RAG 项目-Embedding 模型微调任务
REM ============================================================

chcp 65001 >nul
cd /d "%~dp0\..\研发"

echo ============================================================
echo   Embedding 模型微调 启动中...
echo   工单编号: 人工智能 NLP-RAG 项目-Embedding 模型微调任务
echo ============================================================
echo [注意] 微调需要 GPU 和 sentence-transformers
echo         pip install sentence-transformers torch

python fine_tune.py --loss triplet

pause
