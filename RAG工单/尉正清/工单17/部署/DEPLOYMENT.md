# 部署文档

**工单编号**：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏

本文面向**新接手的运维人员**，从零把这套 RAGFlow 服务跑起来。

---

## 一、这套部署里有什么

```
docker/                     RAGFlow 那套（compose 项目名 `docker`）
├── ragflow-cpu             API Server + Task Executor（一个容器里跑两个进程）
├── mysql                   元数据
├── es01                    向量库（Elasticsearch）
└── redis                   valkey，任务队列

monitoring/                 监控那套（compose 项目名 `wo17-monitoring`，与上面互不影响）
├── prometheus              抓 RAGFlow 的 /metrics
└── grafana                 面板
```

**两套分开起**，监控挂了不影响业务。

另外还有一个**不在容器里**的服务：

```
本机模型服务（测试/embedding_server.py）
  · /v1/embeddings  bge-m3            —— RAGFlow 的嵌入模型指向它
  · /rerank         bge-reranker-v2-m3 —— RAGFlow 的重排模型指向它
```

RAGFlow 通过 `host.docker.internal:9997` 访问它。

---

## 二、前置条件

| 项 | 要求 |
|---|---|
| Docker | Docker Desktop（本机数据盘在 `D:\dockerimage`） |
| GPU | NVIDIA，**6GB 显存**（嵌入 + 重排两个模型都放 GPU 上，实测占 3.4GB） |
| conda | `rag_gd` 环境（跑模型服务和测试脚本） |
| 模型文件 | `D:/Pycharm/yzq/models/` 下的 `bge-m3`、`bge-reranker-v2-m3` |
| 镜像 | RAGFlow 相关镜像用 `测试/pull_images.sh` 拉（本机 Docker Hub 直连不通，走加速器） |

---

## 三、部署步骤

### 1. 起 RAGFlow

```bash
cd 研发/ragflow/docker
docker compose up -d ragflow-cpu
```

> **必须带服务名 `ragflow-cpu`**。不带的话 compose 会去准备整个项目里
> 声明但用不到的服务（minio / opensearch / infinity 等），拉镜像会失败。

等它就绪：

```bash
curl -o /dev/null -w "%{http_code}\n" http://localhost/v1/system/version   # 期望 200
```

> 首次启动要 1~2 分钟（要等 MySQL 健康检查通过）。

### 2. 起本机模型服务

```bash
cd 测试
D:/Anaconda/envs/rag_gd/python.exe embedding_server.py --port 9997 \
    --embed-device cuda --rerank-device cuda
```

**两个模型都放 GPU**：嵌入放 CPU 的话，20 并发下它自己就要 2.3 秒，
和重排抢资源。都放 GPU 后实测 20 并发 P50 从 4.1s 降到 3.4s。

验证：

```bash
curl -s http://127.0.0.1:9997/health
# {"status":"ok","embed":"bge-m3","dim":1024,"rerank":"bge-reranker-v2-m3"}
```

### 3. 装剖析与监控依赖

```bash
cd 部署
bash install_extras.sh          # py-spy / objgraph / prometheus-client
bash install_extras.sh --check  # 只检查
```

> ⚠️ **容器每次重建（`--force-recreate`）之后都要重跑一次** ——
> 这些包装在容器的可写层里，重建就没了。这是有意的：不动镜像，保持镜像干净。

### 4. 起监控（可选）

```bash
cd 部署/monitoring
docker compose up -d
```

| | 地址 | 账号 |
|---|---|---|
| Grafana | http://localhost:3001 | admin / ragflow17 |
| Prometheus | http://localhost:9090 | 无 |

> **宿主 3000 端口已被占用**（本机 Milvus 套件的 Web 界面），所以 Grafana 用 **3001**。

---

## 四、配置文件在哪、改什么

### `研发/ragflow/docker/.env`

| 变量 | 默认 | 说明 |
|---|---|---|
| `RERANK_MAX_CANDIDATES` | 4 | 每次问答最多重排多少条候选。**重排是 GPU 硬瓶颈（约 60 对/秒）**，这个值直接决定并发下的延迟。设 0 = 不限（原行为） |
| `RAGFLOW_REASONING_EFFORT` | low | deepseek 思维链档位：`none`/`low`/`high`/`max` |
| `RAGFLOW_MAX_TOKENS` | 8192 | 思维链与正文共用的额度下限（RAGFlow 默认只给 512） |
| `DEEPSEEK_API_KEY` | — | 对话模型的 key |

### `研发/conf/visual_rag.json`

跨模态检索的融合权重（工单15 引入），与并发无关，一般不用动。

### 补丁是怎么挂进去的

RAGFlow 的代码在**镜像里**，本地改的文件通过 `docker-compose.yml` 的
`volumes` 一行行挂进去覆盖。要改哪一行、为什么，见 compose 里带「工单」注释的挂载项。

> **附件源码不能直接挂**：附件那份比镜像里的新，挂了容器起不来
> （`ImportError: thread_pool_exec`）。正确姿势是
> `docker cp 容器:/ragflow/xxx.py` 把镜像自带的那份导出来再改。

---

## 五、目录速查

| 想做什么 | 去哪 |
|---|---|
| 改 RAGFlow 的代码 | `研发/patches/`，改完 `docker compose up -d --force-recreate ragflow-cpu` |
| 改部署参数 | `研发/ragflow/docker/.env`，改完同样要重建容器（新增变量必须重建） |
| 跑压测 | `测试/jmeter_run.sh` |
| 看资源曲线 | `测试/monitor.py` |
| 画对比图 | `测试/画图.py` |
| 日常运维、排障 | `部署/OPERATION.md` |
