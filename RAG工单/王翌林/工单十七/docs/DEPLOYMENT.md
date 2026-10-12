# 工单十七 · 部署文档（DEPLOYMENT）

## 1. 环境要求
- Python 3.10+，依赖：fastapi、uvicorn、prometheus_client、psutil、locust（压测）。
- 可选：Docker + docker-compose（含 Redis / Prometheus / Grafana）。

## 2. 安装依赖
```bash
pip install fastapi uvicorn prometheus_client psutil locust
```

## 3. 本地运行
```bash
cd 工单十七
export PYTHONPATH=$(pwd)
# 优化版服务
uvicorn src.api:app --host 0.0.0.0 --port 8017 --workers 1
# 健康检查
curl http://127.0.0.1:8017/health
# 指标
curl http://127.0.0.1:8017/metrics
```

## 4. Docker 编排（含监控栈）
```bash
docker-compose up -d
# 服务    rag-api:8017   Prometheus:9090   Grafana:3000   Redis:6379
```
`docker-compose.yml` 已对 rag-api 配置 `memory: 1024M / cpus: 2.0` 资源限制，模拟生产约束、提前暴露泄漏。

## 5. 压测
```bash
# Python 自包含压测（出对比数据）
bash scripts/run_loadtest.sh
# Locust（Web UI）
locust -f loadtest/locustfile.py --host http://127.0.0.1:8017
```

## 6. 生成报告与图表
```bash
python scripts/gen_report.py    # 输出 docs/screenshots/ 对比图
```
