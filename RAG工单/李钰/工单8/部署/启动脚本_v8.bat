@echo off
REM ============================================================
REM V8 Graph RAG 金融问答 - Windows 启动脚本
REM 工单编号: 人工智能 NLP-RAG-基于 Graph RAG 实现金融问答
REM ============================================================

chcp 65001 >nul
cd /d "%~dp0\..\研发"

echo ============================================================
echo   Graph RAG 金融问答 V8 启动中...
echo   工单编号: 人工智能 NLP-RAG-基于 Graph RAG 实现金融问答
echo   端口: 5007
echo ============================================================
echo [可选] SET NEO4J_URI=bolt://localhost:7687

python app_v8.py

pause
