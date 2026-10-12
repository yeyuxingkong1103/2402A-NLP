# 运维文档

**工单编号**：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏

面向日常运维：怎么看不健康、怎么看指标、出问题先查什么。

---

## 一、日常巡检（5 分钟）

```bash
# 1. 服务活着吗
curl -o /dev/null -w "API %{http_code}\n" http://localhost/v1/system/version   # 200

# 2. 三个容器健康吗
docker ps --format "{{.Names}}\t{{.Status}}"
#   ragflow-cpu  Up          mysql  Up (healthy)
#   es01         Up (healthy) redis  Up (healthy)

# 3. 本机模型服务在吗
curl -s http://127.0.0.1:9997/health

# 4. 任务队列有没有堆积
docker exec docker-redis-1 redis-cli -a infini_rag_flow --no-auth-warning XLEN rag_flow_svr_queue
#   正常：0 或个位数。持续几百说明 Task Executor 跟不上
```

---

## 二、看监控指标

**Prometheus 端点**：`http://localhost:9390/metrics`（RAGFlow 容器内独立端口，
**不经过 nginx** —— 业务被打满时它照样能采）。

| 指标 | 看什么 | 正常范围 |
|---|---|---|
| `ragflow_http_requests_total` | 按路由/状态码的请求量 | `status!="200"` 应长期为 0 |
| `ragflow_http_request_duration_seconds` | 延迟直方图，算 P95 | 单请求 P95 应 < 3s |
| `ragflow_http_in_flight_requests` | 在途请求数 | 稳态应贴着并发用户数；**持续上涨 = 有堆积** |
| `ragflow_process_resident_memory_bytes` | 进程 RSS | **长期走平**；持续上扬才是泄漏 |
| `ragflow_process_cpu_percent` | CPU 占用 | 并发高但 CPU 低 = 在等 I/O，不是算不过来 |
| `ragflow_task_queue_depth` | 解析队列长度 | 涨起来后能被消费掉就正常 |

### Grafana 面板

`http://localhost:3001` → 文件夹 **RAGFlow** → 面板「**RAGFlow API 并发与资源**」

七块图：延迟分位、请求速率与错误、在途请求、进程 RSS、队列深度、CPU、各路由 P95。

> **面板没数据先查这两处**：
> 1. Prometheus → Status → Targets，`ragflow` 是不是 `up`
> 2. 是 `down` 的话，多半是 `install_extras.sh` 没跑（prometheus-client 没装，
>    `/metrics` 就起不来）

---

## 三、常见故障与处置

### 现象：接口大量 502

**先看 nginx 日志**：

```bash
docker exec docker-ragflow-cpu-1 tail -20 /var/log/nginx/error.log
```

| 日志内容 | 含义 | 处置 |
|---|---|---|
| `no live upstreams` | nginx 把后端判死了 | 见下 |
| `upstream timed out` | 后端处理太慢 | 看是不是并发太高；查 `/metrics` 的在途请求数 |

> **`no live upstreams` 的历史坑**：nginx 原来配的是
> `proxy_pass http://localhost:9380`，而容器里 `localhost` 同时解析到
> `127.0.0.1` 和 `::1`，RAGFlow 只监听 IPv4 —— 一半请求打到死地址上，
> 高并发时整组被判死。**已改成 `127.0.0.1`**（`nginx/ragflow.conf`）。
> 如果谁把配置改回 `localhost`，这个故障会复现。

### 现象：响应突然变得很慢（10 秒以上）

按这个顺序查：

```bash
# ① 容器内存有没有涨（泄漏？）
docker stats --no-stream docker-ragflow-cpu-1

# ② GPU 显存有没有异常
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv

# ③ 队列有没有堆
docker exec docker-redis-1 redis-cli -a infini_rag_flow --no-auth-warning XLEN rag_flow_svr_queue

# ④ 本机模型服务还在吗（它挂了检索会退化成只剩 BM25）
curl -s http://127.0.0.1:9997/health
```

**最常见的两个原因**：

1. **本机模型服务没起** —— 检索失去嵌入和重排，回复质量会明显下降（不一定变慢）
2. **有人把 `RERANK_MAX_CANDIDATES` 调大或设成 0** —— 重排是 GPU 硬瓶颈
   （约 60 对/秒），候选数翻倍，并发延迟就翻倍

### 现象：解析任务一直卡在「排队中」

```bash
# Task Executor 还活着吗
docker exec docker-ragflow-cpu-1 ps -eo pid,args | grep task_executor

# 它最近在干什么
docker logs docker-ragflow-cpu-1 --tail 50 | grep -i task
```

注意 RAGFlow 的解析是**串行**的：`MAX_CONCURRENT_CHUNK_BUILDERS` 默认 1，
即**同一时刻只有一个任务在真正分块**（OCR/版面分析很吃资源，是有意为之）。
队列堆积不一定是故障，可能只是文档多。

---

## 四、性能剖析（需要深挖时）

```bash
# 0. 先装工具（容器重建后必须重跑）
cd 部署 && bash install_extras.sh

# 1. 找 API Server 的 PID（容器重建后会变！）
docker exec docker-ragflow-cpu-1 sh -c "ps -eo pid,args | grep api/ragflow_server | grep -v grep"

# 2. 采样 60 秒，出火焰图
docker exec docker-ragflow-cpu-1 /ragflow/.venv/bin/py-spy record \
    --pid <PID> --duration 60 --rate 50 --format flamegraph -o /tmp/flame.svg
docker cp docker-ragflow-cpu-1:/tmp/flame.svg 测试/results/火焰图.svg

# 3. 想要能算 self time 的原始数据，用 --format raw
docker exec docker-ragflow-cpu-1 /ragflow/.venv/bin/py-spy record \
    --pid <PID> --duration 60 --rate 50 --format raw -o /tmp/raw.txt
```

> **两个坑**：
> 1. **PID 会变**。容器每次重建，PID 都重新分配（实测在 18~27 之间跳）。
>    照着旧 PID 采会报 `Failed to find python version from target process` ——
>    这个报错具有误导性，实际原因是**采错进程了**（当时那个 PID 是 nginx）。
> 2. py-spy 需要 `SYS_PTRACE`。这一条已经写进 compose 的 `ragflow-cpu` 服务，
>    缺了会报 `Failed to copy Py_Version symbol: Permission denied`。

### 看火焰图的要点

**父帧是累计值，只有叶子帧代表真正耗时**。所以：

- 火焰图上看**最底下那一层**宽不宽
- 或者用 `--format raw` 导出后自己按最后一个分号后面的函数统计

本工单就是靠这个发现 `_worker (concurrent/futures/thread.py)` 占 12.7%
（线程在等），从而把方向从「算得慢」扭到「在等」。

---

## 五、容器重建后的必做项

```bash
cd 研发/ragflow/docker && docker compose up -d --force-recreate ragflow-cpu
cd 部署 && bash install_extras.sh      # ← 别忘
```

重建会丢：`py-spy`、`objgraph`、`prometheus-client`（装在容器可写层）。
业务数据（MySQL/ES/Redis 的卷）不受影响。

---

## 六、日常维护

| 周期 | 做什么 |
|---|---|
| 每天 | 看 Grafana 面板：错误率是否 0、RSS 是否走平 |
| 每周 | 查队列深度趋势；看容器日志有无 ERROR |
| 磁盘紧张时 | `docker image prune`（别用 `-a`，会删掉在用的镜像）；必要时压缩 WSL 的 vhdx |
| 改完配置/代码 | `--force-recreate` + `install_extras.sh`，然后**跑一轮 `jmeter_run.sh` 对比**，别裸奔上线 |
