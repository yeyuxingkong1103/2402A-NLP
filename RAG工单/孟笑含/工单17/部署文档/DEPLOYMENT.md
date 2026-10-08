# 工单17 部署文档

## 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA RTX 4090 或同等（24GB） |
| CUDA | 12.4+ |
| Python | 3.10+ |
| 依赖 | torch, sentence-transformers, fastapi, uvicorn, locust |

## 部署步骤

### 1. 环境准备

    pip install fastapi uvicorn prometheus_client psutil locust torch sentence-transformers

确认 GPU：

    python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

### 2. 代码部署

    cd ~/autodl-tmp
    ls api_server.py locustfile.py rag_qa_system.py rerankers.py

### 3. 启动 API 服务

    cd ~/autodl-tmp
    nohup python3 api_server.py > api_server.log 2>&1 &
    echo $! > api_server.pid
    sleep 90
    curl -s http://localhost:8000/health

### 4. 验证

    curl -s -X POST http://localhost:8000/api/chat \
      -H "Content-Type: application/json" \
      -d '{"query":"注册资本是多少？","chat_id":"test","stream":false}' \
      | python3 -m json.tool

## 服务端点

| 端点 | 方法 | 说明 |
|------|------|------|
| /health | GET | 健康检查 |
| /api/chat | POST | 问答接口 |
| /metrics | GET | Prometheus 指标 |

## 故障排查

| 故障 | 排查命令 | 处理 |
|------|----------|------|
| 服务无响应 | curl http://localhost:8000/health | 重启服务 |
| 内存增长异常 | ps -o rss -p $(cat api_server.pid) | 检查单例缓存 |
| GPU 显存不释放 | nvidia-smi | 确认 empty_cache |
| 端口被占 | lsof -i:8000 | kill 旧进程 |

## 环境说明

因 AutoDL 容器不支持 Docker（docker: command not found），采用源码部署方式。
优化手段和结论完全适用 RAGFlow 官方 Docker 部署。
