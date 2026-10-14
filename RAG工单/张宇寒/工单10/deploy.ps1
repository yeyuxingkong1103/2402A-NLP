$ErrorActionPreference = "Stop"

Write-Host "[1/4] 检查Docker引擎..."
docker info | Out-Null

Write-Host "[2/4] 构建并启动金融GraphRAG容器..."
docker compose up -d --build

Write-Host "[3/4] 等待健康并运行问答评测..."
python container_check.py --base-url http://127.0.0.1:8021

Write-Host "[4/4] 容器状态："
docker compose ps
Write-Host "部署完成：http://127.0.0.1:8021/"
