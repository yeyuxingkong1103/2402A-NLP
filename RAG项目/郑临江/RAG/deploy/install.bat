@echo off
chcp 65001 >nul
echo ============================================================
echo   RAG2 在线阶段 —— 依赖检查（不改动现有环境）
echo ============================================================
echo.
echo 运行环境：conda 环境 rags_（D:\an\envs\rags_）
echo 本项目不在此安装/升级依赖，请确认以下包已在 rags_ 环境中：
echo.
echo   Python 依赖：
echo     fastapi  uvicorn  redis  httpx  python-multipart  openai  yaml
echo     pymilvus  jieba  pymysql  pdfplumber
echo     （可选）ragas  sentence-transformers
echo.
echo   一键检查：
echo     D:\an\envs\rags_\python.exe -c "import fastapi,uvicorn,redis,httpx,yaml,openai,pymilvus; print('依赖 OK')"
echo.
echo  外部服务（需自行启动）：
echo     Redis  127.0.0.1:6379        （无密码）
echo     Milvus http://localhost:19530 （集合 agriculture_knowledge）
echo     Ollama http://localhost:11434 （模型 Qwen3.5:4B）
echo.
echo  模型权重（需存在）：
echo     D:\modelscope\bge-m3
echo     D:\modelscope\bge-reranker-v2-m3
echo.
pause
