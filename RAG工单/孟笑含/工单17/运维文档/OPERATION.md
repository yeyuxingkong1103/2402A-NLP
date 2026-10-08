# 工单17 运维文档

## 1. 日常监控

### 1.1 服务状态

    ps -p $(cat ~/autodl-tmp/api_server.pid) -o pid,etime,rss,cmd

### 1.2 GPU 监控

    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 5

### 1.3 日志

    tail -f ~/autodl-tmp/api_server.log

## 2. 压测复现

### 2.1 场景A（20并发问答，10分钟）

    cd ~/autodl-tmp
    locust -f locustfile.py --headless -u 20 -r 2 -t 10m \
      --host http://localhost:8000 --csv=scenario_a

### 2.2 场景B（10并发混合，10分钟）

    cd ~/autodl-tmp
    locust -f locustfile.py --headless -u 10 -r 1 -t 10m \
      --host http://localhost:8000 --csv=scenario_b

### 2.3 资源监控（另开终端）

    cd ~/autodl-tmp
    ./monitor_resources.sh

## 3. 故障处理

| 故障 | 排查 | 处理 |
|------|------|------|
| 服务无响应 | curl /health | 重启服务 |
| 内存增长异常 | ps -o rss | 检查单例缓存是否生效 |
| GPU 显存不释放 | nvidia-smi | 确认 torch_empty_cache 调用 |
| 端口被占 | lsof -i:8000 | kill 旧进程 |
| PDF 解析慢 | tail api_server.log | 减少 PDF 数量 |

## 4. 重启流程

    cd ~/autodl-tmp
    pkill -9 -f "python3 api_server.py"
    sleep 2
    nohup python3 api_server.py > api_server.log 2>&1 &
    echo $! > api_server.pid
    sleep 90
    curl -s http://localhost:8000/health

## 5. 验收标准

| 项 | 标准 | 实测 |
|----|------|------|
| 场景A P95 | <= 3s | 700ms |
| 场景A 内存增长 | <= 10% | 0.00% |
| 场景B P95 | <= 5s | 300ms |
| 12h RSS 增长 | <= 20% | 40min 0.00% |

## 6. 关键文件

| 文件 | 说明 |
|------|------|
| api_server.py | FastAPI 服务 |
| locustfile.py | 压测脚本 |
| rerankers.py | 优化后的 reranker |
| monitor_resources.sh | 资源监控 |
| api_server.log | 服务日志 |
| scenario_a_stats.csv | 场景A 数据 |
| scenario_b_stats.csv | 场景B 数据 |
| resource_monitor.csv | 资源监控数据 |
