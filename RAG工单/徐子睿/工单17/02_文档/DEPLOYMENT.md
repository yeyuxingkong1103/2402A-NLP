# DEPLOYMENT.md（工单 17 · 部署文档）

> 面向运维新人，指导在 Linux（Ubuntu 22.04）/ WSL2 上部署 RAGFlow 全栈（API Server + Task Executor + 向量库 + Redis + MinIO）。

## 1. 前置条件

| 项 | 要求 |
|---|---|
| 系统 | Ubuntu 22.04+ / WSL2（Windows 需 Docker Desktop + WSL2 后端） |
| Docker | ≥ 24，Docker Compose v2 |
| 资源 | ≥ 16GB RAM，≥ 50GB 磁盘；GPU 可选（跑本地 embedding/rerank/VLM 时必需） |
| 内核参数 | `vm.max_map_count >= 262144`（Elasticsearch 需要） |

```bash
sudo sysctl -w vm.max_map_count=262144
echo 'vm.max_map_count=262144' | sudo tee -a /etc/sysctl.conf
```

## 2. 部署步骤

```bash
# 2.1 获取源码
cd /root && git clone https://github.com/infiniflow/ragflow.git   # 或解压 ragflow-main.zip
cd ragflow/docker

# 2.2 配置（模型、端口、镜像）
cp .env .env.bak
vi .env            # 设置 SVR_HTTP_PORT / RAGFLOW_IMAGE / 模型相关；国内可设镜像源

# 2.3 启动（先依赖后主服务）
docker compose -f docker-compose.yml up -d

# 2.4 校验
docker compose ps                 # 所有容器 Up (healthy)
docker logs -f ragflow-server     # 观察启动日志
curl -s -o /dev/null -w "%{http_code}\n" http://localhost    # 期望 200
```

## 3. 资源限制（模拟生产约束，提前暴露问题）

```yaml
# docker-compose.override.yml
services:
  ragflow:
    deploy:
      resources:
        limits: { cpus: "4", memory: 8g }   # 触发内存上限，验证 OOM 前是否有缓慢增长
```

## 4. 初始化

1. 浏览器打开 `http://<host>` → 注册管理员；
2. 配置模型：**Embedding / Chat / Rerank / IMAGE2TEXT(VLM)**（本地 Ollama 或云端 API）；
3. 创建知识库（解析方法 **DeepDOC**）→ 上传测试文档 → 点击解析。

## 5. 压测与监控

```bash
# 压测
jmeter -n -t ragflow_stress.jmx -Jconcurrency=20 -Jduration=600 -l result_A.jtl
# 监控
docker stats ; nvidia-smi dmon ; redis-cli INFO stats
```

## 6. 常见故障

| 现象 | 处置 |
|---|---|
| ES 起不来 | 检查 `vm.max_map_count` |
| 解析任务不消费 | 查 `redis-cli XINFO GROUPS rag_flow_svr_queue` 的 lag；确认 task_executor 容器存活 |
| 端口占用 | 改 `.env` 的 `SVR_HTTP_PORT` |
| 容器 OOM | 提高 `memory` 限制或按《排查与优化》修泄漏 |
