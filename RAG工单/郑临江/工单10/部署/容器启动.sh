#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-金融问答系统部署
set -e
cd "$(dirname "$0")/.."
docker build -t rag-workorder10:local -f 部署/Dockerfile .
if docker inspect rag-workorder10 >/dev/null 2>&1; then
  docker start rag-workorder10
else
  docker run -d --name rag-workorder10 --restart unless-stopped --network host -e LLM_BASE_URL=http://127.0.0.1:11434 -v "$PWD/研发/data:/app/data" rag-workorder10:local
fi
docker exec rag-workorder10 python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8774/api/health').read().decode())"
