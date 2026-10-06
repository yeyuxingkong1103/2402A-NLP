@echo off
REM ============================================================
REM 基于 PDF 文档的问答系统优化版 V2 - Windows 启动脚本
REM 工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
REM ============================================================

chcp 65001 >nul
cd /d "%~dp0\..\研发"

echo ============================================================
echo   招股说明书问答系统 V2 优化版 启动中...
echo   工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
echo   端口: 5001 (避免与 V1 冲突)
echo ============================================================

if "%LLM_API_KEY%"=="" (
    echo [提示] 未配置 LLM_API_KEY, 将使用降级模式 (本地检索片段)
    echo        如需启用完整 RAG:
    echo        set LLM_API_KEY=sk-xxxx
    echo        set LLM_BASE_URL=https://api.openai.com/v1
    echo        set LLM_MODEL=gpt-3.5-turbo
)

python app_v2.py

pause
