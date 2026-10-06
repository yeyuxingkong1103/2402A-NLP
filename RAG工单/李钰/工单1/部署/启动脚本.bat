@echo off
REM ============================================================
REM 基于 PDF 文档的问答系统 - Windows 启动脚本
REM 工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统
REM ============================================================

chcp 65001 >nul
cd /d "%~dp0\..\研发"

echo ============================================================
echo   招股说明书问答系统 启动中...
echo   工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统
echo ============================================================

REM 设置 LLM 配置 (按需修改或通过环境变量传入)
if "%LLM_API_KEY%"=="" (
    echo [提示] 未配置 LLM_API_KEY, 将使用降级模式 (本地检索片段)
    echo        如需启用完整 RAG 与对比基线, 请设置:
    echo        set LLM_API_KEY=sk-xxxx
    echo        set LLM_BASE_URL=https://api.openai.com/v1
    echo        set LLM_MODEL=gpt-3.5-turbo
)

REM 启动 Flask 应用
python app.py

pause
