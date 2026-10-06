@echo off
REM ============================================================
REM 一键 Docker 启动 - Windows
REM 工单编号: 人工智能 NLP-RAG-金融问答系统部署
REM ============================================================

chcp 65001 >nul
cd /d "%~dp0\.."

echo ============================================================
echo   Financial RAG QA System - Docker 部署
echo   工单编号: 人工智能 NLP-RAG-金融问答系统部署
echo ============================================================

where docker >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Docker, 请先安装 Docker Desktop
    echo 下载: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo.
echo [1/3] 构建镜像...
docker build -f 工单10\部署\Dockerfile -t financial-rag:v10.0 .
if errorlevel 1 (
    echo [错误] 镜像构建失败
    pause
    exit /b 1
)

echo.
echo [2/3] 启动容器...
docker run -d --name financial-rag-qa ^
    -p 5008:5008 ^
    -v rag_data:/app/data ^
    -v rag_cache:/app/cache ^
    -v rag_logs:/app/logs ^
    -v rag_shared:/app/shared ^
    -e LLM_API_KEY=%LLM_API_KEY% ^
    --restart unless-stopped ^
    financial-rag:v10.0

echo.
echo [3/3] 等待健康检查...
timeout /t 5 /nobreak >nul

docker ps --filter "name=financial-rag-qa"
echo.
echo ============================================================
echo   ✅ 启动完成!
echo   访问: http://localhost:5008
echo   健康检查: http://localhost:5008/api/health
echo ============================================================
echo.
echo   常用命令:
echo     docker logs -f financial-rag-qa
echo     docker stop financial-rag-qa
echo     docker start financial-rag-qa
echo     docker rm -f financial-rag-qa
echo ============================================================
pause
