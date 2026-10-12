@echo off
chcp 65001 >nul
REM 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
REM 一键启动脚本：检查依赖服务 -> 启动 Milvus -> 启动 FastAPI -> 打开浏览器

setlocal
cd /d "%~dp0"

echo ============================================================
echo   基于PDF文档的问答系统  启动中
echo ============================================================

REM ---- 1. 检查 Python ----
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 python，请先安装 Python 并加入 PATH
    pause & exit /b 1
)
python --version

REM ---- 2. 注入 MinerU 中文路径补丁 ----
REM 本项目路径含中文，MinerU 依赖的 fasttext 无法打开非 ASCII 路径；
REM 该补丁通过 sitecustomize 在解释器启动时改写模型路径（详见部署文档）。
if exist "C:\mineru_models\langfix\sitecustomize.py" (
    set "PYTHONPATH=C:\mineru_models\langfix;%PYTHONPATH%"
    echo [OK] 已注入 MinerU 中文路径补丁
) else (
    echo [警告] 未找到 C:\mineru_models\langfix，MinerU 解析大概率失败，将自动降级为 pymupdf
)
set "MINERU_MODEL_SOURCE=modelscope"

REM ---- 3. 启动 Redis（本机为 Windows 原生服务，已装则跳过）----
sc query Redis >nul 2>&1
if not errorlevel 1 (
    sc query Redis | find "RUNNING" >nul
    if errorlevel 1 (
        echo [..] 启动 Redis 服务...
        net start Redis >nul 2>&1
    )
    echo [OK] Redis 已就绪
) else (
    echo [警告] 未检测到 Redis 服务，系统会自动切换为无缓存模式
)

REM ---- 4. 启动 Milvus（Docker）----
docker info >nul 2>&1
if errorlevel 1 (
    echo [警告] Docker 未启动，跳过 Milvus；系统将自动降级为 milvus-lite 本地库
) else (
    echo [..] 启动 Milvus 容器（etcd + minio + milvus）...
    docker compose -f deploy\docker-compose.yml up -d >nul 2>&1
    echo [OK] Milvus 容器已提交启动，首次冷启动约需 90 秒
)

REM ---- 5. 启动 FastAPI ----
echo [..] 启动 FastAPI 服务（http://127.0.0.1:8000）...
start "" http://127.0.0.1:8000
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

endlocal
