# 阶段五 · 部署说明

> 工单编号：人工智能NLP-RAG-混合检索任务

---

## 1. 本机现状（部署前必读）

部署前先确认本机实际状态，**不要盲目执行 `docker-compose up`**：

| 组件 | 本机实测状态 | 处置 |
|---|---|---|
| Milvus | 已在容器 `rag-milvus` 运行，19530 已映射 | **直接复用，不要再起 compose** |
| Redis | 已在 `127.0.0.1:6379` 运行（服务端版本 **5.0.14.1**） | 直接复用；客户端必须 `protocol=2` |
| 端口 8000 | 已被本机工单五服务占用 | 本工单固定用 **8010** |
| GPU | RTX 4060 Laptop，8188 MiB，其中约 4.8 GB 已被其它进程占用 | bge-m3(fp16)+bge-reranker(fp16) 共存实测仅需 **2174 MiB**，余量够用 |
| Python | 唯一齐备的解释器：`D:/myconda1/python.exe`（Python 3.12.7，CUDA 12.6 torch） | 全部命令都用它 |

> **工单要求 Python 3.11，本机实际为 3.12.7。** 本机没有装 MinerU + CUDA torch + ragas 齐全的 3.11 解释器，
> 因此沿用 3.12.7。全部依赖在 3.12 下实测通过；若必须在 3.11 上运行，按 `requirements.txt` 装即可，代码未使用 3.12 专有语法。

---

## 2. 依赖安装

```bash
cd "d:/Users/雨子/Desktop/专高六/liu-专高六/RAG/工单六"
D:/myconda1/python.exe -m pip install -r requirements.txt
```

## 3. 配置

```bash
cp .env.example .env
# 编辑 .env，至少填入 DEEPSEEK_API_KEY
```

`.env` 关键项（**陷阱写在配置项旁边**）：

```ini
DEEPSEEK_API_KEY=sk-xxxxxxxx                 # 必填，也可用系统环境变量覆盖
EMBED_MODEL_PATH=D:/模型/bge-m3              # 必须正斜杠！反斜杠的 \b 会被解析成退格符
RERANK_MODEL_PATH=D:/模型/bge-reranker-v2-m3
MILVUS_URI=http://localhost:19530
REDIS_URL=redis://localhost:6379/0
REDIS_PROTOCOL=2                             # 本机 Redis 5.x 不支持 RESP3，必须 2
API_PORT=8010                                # 8000 已被占用
MINERU_LANGFIX_DIR=C:/mineru_models/langfix  # 中文用户名下缺此项 MinerU 必失败
```

> `.env` 中已存在的**真实环境变量优先**（`load_dotenv(override=False)`），部署时可用环境变量覆盖而不改文件。

## 4. (可选) docker-compose 拉起 Milvus 与 Redis

`docker-compose.yml` 已提供 etcd + minio + milvus + redis 的完整编排。

```bash
docker-compose up -d          # 仅在"没有现成 Milvus/Redis"的机器上执行
docker-compose ps
```

**本机不要执行**：19530 与 6379 已被占用，重复拉起会端口冲突。这份编排是给新机器用的。

## 5. 启动服务

```bash
# 方式一：模块方式（推荐，等价于 python -m app.main）
D:/myconda1/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010

# 方式二：便捷脚本
scripts\start.bat            # Windows
bash scripts/start.sh        # Git Bash
```

启动时 `lifespan` 会自动：载入 BM25 索引 → 预热 bge-m3。**预热失败不会阻断启动**，服务会以降级模式继续运行并在日志里说明原因。

前端：`frontend/index.html` 由 FastAPI 挂在根路径（`StaticFiles`），直接访问 <http://127.0.0.1:8010/> 即可。
Vue 已本地内置（`frontend/vue.global.prod.js`），**不依赖 CDN**，断网可用。

## 6. 首次入库

```bash
# 通过接口
curl -X POST http://127.0.0.1:8010/ingest

# 或直接跑模块（带进度输出）
D:/myconda1/python.exe -m app.ingest
```

入库流程：解析 `data/raw/*.pdf` → 分块 → 去噪 → 建 BM25（落 `data/bm25.pkl`）→ 写 Milvus（集合 `gd6_chunks`）。

> **关于解析缓存**：MinerU 解析一份 548 页招股说明书耗时数分钟。`app/ingest.py` 会优先读取
> `data/parsed/<文档名>/<文档名>/auto/<文档名>_content_list.json` 缓存。本项目的 `data/parsed/` 是
> **同一批 PDF 的 MinerU 真实产物**（自本机前序工单沿用，非模拟数据）。若要验证从头解析，删除 `data/parsed/`
> 后重新 `POST /ingest`，或执行 `tests/test_mineru_smoke.py` 在小切片上实跑 MinerU。

## 7. 验收自测

```bash
# 接口冒烟测试（覆盖 3 模式 / 3 融合 / 3 重排 / 布尔短语模糊 / 中英文问答 / 多轮 / 缓存 / 反馈）
D:/myconda1/python.exe tests/test_api.py

# RAGAS 评估（22 题，输出准确率/召回率/响应时间，结果落 results/）
D:/myconda1/python.exe -m app.eval_ragas
curl -X POST http://127.0.0.1:8010/eval
```

## 8. 停止服务

```bash
# 前台启动时 Ctrl+C；后台启动时按端口找进程
netstat -ano | findstr :8010
taskkill /PID <PID> /F
```

## 9. 常见故障对照表

| 现象 | 根因 | 处置 |
|---|---|---|
| 启动报 `No module named 'app'` | 工作目录不对 | 必须在项目根目录执行 |
| 检索全空且日志有 `[检索降级] Milvus 检索失败` | Milvus 未启动 | `docker ps` 确认 `rag-milvus` 运行 |
| 日志 `unknown command 'HELLO'` | Redis 5.x 不支持 RESP3 | 确认 `.env` 的 `REDIS_PROTOCOL=2` |
| 大模型返回空内容 | 推理型模型未关思考 | 确认 `LLM_REASONING_EFFORT`（名字含 flash/reason 时必须 `none`） |
| MinerU 报 fasttext 打不开模型 | 中文用户名 | 确认 `MINERU_LANGFIX_DIR=C:/mineru_models/langfix` |
| 路径报不存在，路径里出现乱码字符 | `.env` 里写了反斜杠 | 路径改用正斜杠 |
| 评估指标大量为 `null` | 被限流或余额不足（402） | 调小 `EVAL_MAX_WORKERS`；看 `results/*.json` 的 `valid_samples` |
| 端口被占用 | 8000 是本机既有服务 | 用 8010 |

## 10. 资源与并发说明

- 单进程 uvicorn。`/chat`、`/search` 是同步 `def` 路由，FastAPI 会自动放到**线程池**执行，因此多个请求可以并发等待 GPU/网络，不会互相阻塞事件循环。
- 向量模型与重排模型是**进程内单例**，只加载一次；并发的检索请求共用同一份权重，这是本方案能在 8 GB 显存下支撑并发的前提。
- 缓存层显著降低重复提问的负载：相同问句+相同参数第二次命中 Redis，**不再调用大模型**。
- 如需横向扩展，多副本共享同一 Milvus 与 Redis 即可；模型权重按副本各存一份，需相应增加显存。
