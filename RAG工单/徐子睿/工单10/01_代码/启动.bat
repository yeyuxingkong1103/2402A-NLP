@echo off
chcp 65001 >nul
title 招股说明书智能问答系统（RAG）
cd /d %~dp0
echo ============================================
echo   招股说明书智能问答系统（RAG）
echo   工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
echo ============================================
echo.
echo [1] 检查 Ollama 模型 ...
ollama list | findstr /i "bge-m3" >nul || echo   警告：未找到 bge-m3，请先 ollama pull bge-m3
ollama list | findstr /i "qwen2:7b" >nul || echo   警告：未找到 qwen2:7b，请先 ollama pull qwen2:7b
echo.
echo [2] 启动服务： http://localhost:8100
echo     （按 Ctrl+C 停止）
echo.
set PYTHONWARNINGS=ignore
python -m uvicorn app.server:app --host 0.0.0.0 --port 8100
pause
