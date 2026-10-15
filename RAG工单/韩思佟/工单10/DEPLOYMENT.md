# 工单10 · 金融问答系统部署文档（DEPLOYMENT）

> 📌 **端口提醒**：本机 **8000 端口已被另一个项目占用**
> （`curl http://127.0.0.1:8000/` 会返回 `{"service":"知愈医疗RAG",...}`），
> 所以工单10 部署时请把宿主机端口换成 **8010**：
> `docker run -p 8010:8000 ...`，或 compose 里写成 `"8010:8000"`，
> 对应的访问地址也换成 `http://127.0.0.1:8010/`。


> 工单编号：人工智能 NLP-RAG-金融问答系统部署
> 目标：把金融问答系统以 Docker 容器形式部署并运行成功。

---

## 一、前置条件

| 项 | 要求 |
|---|---|
| Docker | ≥ 24（实测 29.3.1 可用） |
| Docker Compose | ≥ v2（实测 v5.1.1 可用） |
| 磁盘 | ≥ 10GB（镜像含 torch CPU 版） |
| 网络 | 能访问 LLM API（DeepSeek） |

## 二、目录准备

```bash
# 交付目录结构（已备好）
RAG工单交付/
├── Dockerfile
├── docker-compose.yml      # 本工单的 compose 文件在 工单10/ 下
├── requirements.txt
├── src/                    # 代码
├── scripts/                # 运行脚本
├── data/                   # 源文档（招股书 / 年报）
└── cache/                  # 知识库与模型缓存
```

把 `工单10/Dockerfile`、`工单10/docker-compose.yml`、`工单10/requirements.txt`
复制到项目根目录，或直接在 `工单10/` 下执行（此时 `docker build` 的上下文要指向项目根目录）：

```bash
cd D:\网页下载\RAG工单交付
docker build -f 工单10/Dockerfile -t rag-fin-qa:1.0 .
```

## 三、构建镜像

```bash
# 方式一：直接 build
docker build -f 工单10/Dockerfile -t rag-fin-qa:1.0 .

# 方式二：compose 构建
cd 工单10
docker compose build
```

**构建慢的优化**：把 `requirements.txt` 单独 COPY 并先安装（Dockerfile 已这么做），
改代码时不会重复下载依赖。torch CPU 版约 124MB，是整个镜像的大头。

## 四、配置环境变量

在项目根目录建 `.env`（与 `docker-compose.yml` 同级）：

```env
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
HOST_DATA_DIR=./data
```

> `.env` 已在 `.dockerignore` 里排除，不会打进镜像，Key 不会泄露。

## 五、启动

```bash
cd 工单10
docker compose up -d
docker compose ps                 # 确认 rag-fin-qa 状态为 Up (healthy)
docker compose logs -f rag-api    # 看启动日志
```

首次启动会预加载知识库（约 10-30 秒），日志出现 `服务启动：http://0.0.0.0:8000/` 即就绪。

## 六、验收检查（对应工单"容器管理验收标准"）

### 1. 容器启动与运行

```bash
docker run --rm -p 8000:8000 -e LLM_API_KEY=$LLM_API_KEY rag-fin-qa:1.0   # 单容器验证
curl http://127.0.0.1:8000/healthz          # 期望 {"status":"ok",...}
curl http://127.0.0.1:8000/                 # 打开问答界面
```

- `docker compose ps` 状态应为 `Up (healthy)`（HEALTHCHECK 每 30s 探活 `/healthz`）
- 日志无异常堆栈

### 2. 容器数据管理

```bash
docker volume inspect rag-kb rag-outputs rag-logs
docker compose down && docker compose up -d   # 重启后知识库/输出仍在
```

- 卷持久化：`rag-models`（模型缓存）、`rag-kb`（知识库）、`rag-outputs`（产出）、`rag-logs`（日志）
- 数据共享：源文档以 `ro` 方式挂载进容器，多个容器可共享同一份数据目录

### 3. 网络配置

```bash
docker network inspect 工单10_rag-net        # 172.28.0.0/16
docker exec -it rag-fin-qa ping rag-redis    # 容器间互通（compose 自带 DNS）
docker exec -it rag-fin-qa redis-cli -h rag-redis ping   # 期望 PONG
```

### 4. 服务功能

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"招商银行2019年归属于本行股东的净利润是多少？","kb":"ccf","strategy":"hybrid"}'
```

返回含 `answer / contexts / timings / total_time` 即为成功。

## 七、常见问题（详见 `部署问题记录.md`）

| 现象 | 原因 | 处理 |
|---|---|---|
| 构建时 pip 超时 | 默认 PyPI 在境外 | 已换清华源 + pytorch CPU 源；仍慢可改用国内镜像 |
| 容器起不来，日志报 `未配置 LLM_API_KEY` | 没传 `LLM_API_KEY` | 在 `.env` 里配置，或 `-e LLM_API_KEY=xxx` |
| 首次请求很慢 | 懒加载 embedding 模型 | 加 `--kb ccf` 预加载；或保持常驻让首次请求预热 |
| 内存被 OOM | 默认无限制 | compose 里已设 `memory: 4g`；低配机器可再下调 |
| 端口占用 | 8000 被占 | 改 `ports: - "8001:8000"` |

## 八、停止与清理

```bash
docker compose down            # 停容器（保留卷）
docker compose down -v         # 停容器并删卷（数据会丢，慎用）
docker system prune -a         # 清理无用镜像
```
