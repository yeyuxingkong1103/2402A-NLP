@echo off
REM ============================================================
REM V6 混合检索任务 - Windows 启动脚本
REM 工单编号: 人工智能 NLP-RAG-混合检索任务
REM ============================================================

chcp 65001 >nul
cd /d "%~dp0\..\研发"

echo ============================================================
echo   招股说明书问答系统 V6 混合检索版 启动中...
echo   工单编号: 人工智能 NLP-RAG-混合检索任务
echo   端口: 5005
echo ============================================================

echo [可选] 环境变量:
echo   SET RETRIEVAL_STRATEGY=hybrid     ^&^& REM vector/fulltext/hybrid
echo   SET EMBEDDING_MODEL=tfidf         ^&^& REM tfidf/bge/m3e
echo   SET RERANKER=tfidf                ^&^& REM tfidf/llm/feedback
echo   SET FUSION_METHOD=weighted        ^&^& REM weighted/rrf/vote

python app_v6.py

pause
