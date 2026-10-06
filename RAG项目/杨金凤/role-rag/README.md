# 高血压医生 RAG 系统

基于 RAG（检索增强生成）的高血压医生角色扮演系统：用《国家基层高血压防治管理指南》PDF 构建本地知识库，以「高血压专科医生」人设调用 DeepSeek 回答患者问题，回答附带检索来源（页码 + 相似度）。

## 技术栈

- Python 3.10+
- 向量模型：`BAAI/bge-m3`（本地路径加载，1024 维）
- 重排序：`BAAI/bge-reranker-v2-m3`（CrossEncoder）
- 向量库：Milvus（默认，`USE_MILVUS=true`，`MILVUS_URI` 指向服务端）；Chroma 为降级（`USE_MILVUS=false` 时本地持久化到 `data/chroma/`）
- 大模型：DeepSeek API（`deepseek-flash`，OpenAI 兼容接口）
- Web：FastAPI

检索流程：向量召回 `RECALL_K`（默认 20）条 + BM25 关键词召回 `RECALL_K` 条 → RRF 融合取 top-`RECALL_K` → BGE-rerank 精排取 `RERANK_TOP_K`（默认 4）条。
后处理：回答经 `postprocess` 正则清洗（去代码块标记、压缩空行、规范引用格式）。

## 目录结构

```
role-rag/
├── data/raw/guide.pdf     # 数据源（勿动）
├── milvus_store.py        # Milvus 客户端 + collection 管理
├── parser.py              # PDF 解析（PyMuPDF 正文 + pdfplumber 表格）
├── ingest.py              # PDF 解析 + 清洗 + 分块 + 向量化 + 入库（Milvus/Chroma 双路径）
├── admin.py               # 知识库增删改接口（上传/删除/列出）
├── rag.py                 # 检索 + 医生人设 + 调用 DeepSeek
├── api.py                 # FastAPI 接口
├── .env                   # 密钥/配置（不提交）
├── .env.example           # 配置模板
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.10+（本机已用 3.12 验证）
- CPU 即可运行（本地加载 bge-m3）；GPU 非必需
- 首次运行需联网下载 bge-m3 模型（约 2GB）

## 安装

```bash
cd role-rag
python3 -m venv .venv
source .venv/bin/activate

# CPU 环境：先单独装 CPU 版 torch（避免默认拉 ~2GB 无用的 CUDA 包）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 再装其余依赖
pip install -r requirements.txt
```

> 有 NVIDIA GPU 的环境可跳过上面「CPU 版 torch」这步，直接 `pip install -r requirements.txt`。

## 配置

复制模板并填写：

```bash
cp .env.example .env
```

`.env` 关键项：

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY1` | 必填，DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | 默认 `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | `deepseek-flash`（推荐）或 `deepseek-v4-pro` |
| `HF_ENDPOINT` | HuggingFace 镜像；国内访问 `huggingface.co` 不可达时填 `https://hf-mirror.com` |
| `REDIS_HOST` | Redis 地址，默认 `127.0.0.1` |
| `REDIS_PORT` | Redis 端口，默认 `6379` |
| `REDIS_DB` | Redis 库号，默认 `0` |
| `REDIS_MAX_ROUNDS` | 每会话保留最近 N 轮对话，默认 `10` |
| `REDIS_TTL` | 会话记忆过期秒数，默认 `86400`（1 天） |
| `RERANK_MODEL` | 重排序模型，可填 HF 模型名（如 `BAAI/bge-reranker-v2-m3`）或本地路径，默认 `BAAI/bge-reranker-v2-m3` |
| `RECALL_K` | 向量/BM25 各召回条数，默认 `20` |
| `RRF_K` | RRF 融合平滑常数 k，默认 `60` |
| `RERANK_TOP_K` | 精排后取 top 条数，默认 `4` |
| `USE_MILVUS` | 向量库开关，`true` 用 Milvus（默认）、`false` 降级 Chroma |
| `MILVUS_URI` | Milvus 服务地址，默认 `http://127.0.0.1:19530` |
| `MILVUS_COLLECTION` | Milvus collection 名，默认 `hypertension_guide` |

Redis 配置（可选，不配则用默认值）：

```ini
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
REDIS_MAX_ROUNDS=10
REDIS_TTL=86400
```

重排序配置（可选，不配则用默认值）：

```ini
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RECALL_K=20
RRF_K=60
RERANK_TOP_K=4
```

> Redis 不可用时自动降级为无记忆（只记录 warning 日志），服务不中断，仍正常返回回答。

## 使用

### 1. 构建知识库

```bash
.venv/bin/python ingest.py
```

输出示例（日志）：`抽取到 15 页正文、0 个表格` → `分块后共 X 个 chunk` → `清洗前 X 个 chunk → 清洗后 Y 个` → `已写入 Y 条到 Milvus`。

入库前会做数据清洗（`clean_chunks()`）：按内容 MD5 去重，丢弃过短（<20 字）、纯符号、乱码的 chunk，清洗前后数量会打印在日志里。

### 2. 启动服务

```bash
.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
```

### 3. 调用接口

```bash
# 健康检查
curl http://127.0.0.1:8000/health
# => {"status":"ok"}

# 提问
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"血压多少算高？"}'
```

`POST /chat` 返回：

```json
{
  "answer": "…（医生回答）…",
  "sources": [
    {"content": "…", "page": 4, "similarity": 0.7505}
  ]
}
```

### 4. 运行单元测试

项目含 pytest 单元测试（`tests/` 目录，覆盖 `rag.py` 核心函数），运行：

```bash
.venv/bin/pytest tests/ -v
```

端到端测试：手动起 uvicorn 后跑 `.venv/bin/pytest -m slow -v`，依赖真实 DeepSeek + Redis + Milvus。

## 部署（Ubuntu 服务器一键部署）

三个脚本负责 Python 侧安装与启动/停止；Redis、MySQL、Milvus 是系统服务，需自行部署（脚本会检查并提示，缺失不阻断）。

### 环境要求

- Ubuntu 20.04+
- Python 3.10+
- Redis（会话记忆）
- MySQL（用户/角色/会话元数据）
- Docker（跑 Milvus；不装可设 `USE_MILVUS=false` 降级 Chroma）
- 内存 ≥ 8G（本地加载 bge-m3 与 reranker）

### 三步启动

```bash
# 第一步：安装依赖（检查系统/Python，建 venv，装 requirements）
bash scripts/install.sh

# 第二步：初始化数据
cp .env.example .env        # 填 DEEPSEEK_API_KEY1、MYSQL_URL 等
.venv/bin/python init_db.py # 建库建表 + 默认角色
# 把 PDF 放进 data/raw/ 后构建知识库
.venv/bin/python ingest.py

# 第三步：启动
bash scripts/run.sh
curl http://0.0.0.0:8000/health   # => {"status":"ok"}
```

停止：

```bash
bash scripts/shutdown.sh
```

`run.sh` 参数：`--host`（默认 `0.0.0.0`）、`--port`（默认 `8000`）、`--foreground`（前台运行，Ctrl-C 停止，便于调试）。后台运行日志在 `logs/uvicorn.log`。

### 部署常见问题

- **启动报 `EMBED_MODEL` 路径不存在**：`.env.example` 默认值 `/mnt/d/models/...` 是开发机（WSL）路径，服务器上不存在。改成 HF 模型名 `BAAI/bge-m3`（首次启动自动下载），或手动下载到服务器某路径后把 `EMBED_MODEL` 指向它。
- **MySQL 连不上（root 无法密码登录）**：Ubuntu 新装 MySQL 的 root 默认是 `auth_socket` 插件，见 `init_db.py` 头注释，建议新建 `rag` 用户并授权（`GRANT ALL ON role_rag.*` + `GRANT CREATE ON *.*`），再把 `.env` 的 `MYSQL_URL` 改成对应账号。
- **Milvus 连不上**：确认 Milvus 容器已起（`docker ps`），或设 `USE_MILVUS=false` 降级 Chroma。
- **端口 8000 被占用**：`bash scripts/run.sh --port 8001` 换个端口。
- **服务无输出**：后台日志在 `logs/uvicorn.log`，`tail -f logs/uvicorn.log` 实时查看。
- **装完 torch 很大**：`install.sh` 默认走 CPU 版 torch（`--index-url .../cpu`）；有 GPU 的机器可跳过这步直接 `pip install -r requirements.txt`。

## API

- `GET /health`：健康检查
- `POST /chat`：多轮短期记忆（Redis）
  - 请求：`{"message": "...", "session_id": "..."}`
    - `message` 必填，空串返回 422
    - `session_id` 可选，默认 `"default"`；多用户场景必须传唯一值，否则所有请求共享 `default` 会话会串记忆
  - 响应：`{"answer": "...", "sources": [{"content","page","similarity"}]}`
  - `similarity`：rerank 成功时为 sigmoid 归一化分数；降级时为余弦相似度（= 1 − 余弦距离）

## 常见问题

- **下载 bge 模型报 `Network is unreachable`**：在 `.env` 里把 `HF_ENDPOINT` 设为 `https://hf-mirror.com`（代码已支持，`ingest.py` 运行前会 `load_dotenv()`）。
- **torch 装了 2GB 多**：说明装到了默认的 CUDA 版；纯 CPU 机器按上面「安装」一节先装 CPU 版 torch 即可。
- **改人设**：编辑 `rag.py` 顶部的 `DOCTOR_PERSONA` 常量。
- **改分块参数**：`ingest.py` 里的 `CHUNK_SIZE`、`CHUNK_OVERLAP`。
- **改检索参数**：`.env` 里的 `RECALL_K`（向量/BM25 各召回条数，默认 20）、`RRF_K`（RRF 融合常数 k，默认 60）、`RERANK_TOP_K`（精排后条数，默认 4）。

## 已知限制

- 表格经 `pdfplumber` 抽取为独立 chunk（行内以制表符连接），行列关系以文本形式保留，但未做合并单元格等结构化还原。
- 本阶段无持久会话（Redis 记忆为短期，默认 1 天过期）。
- reranker 首次加载约 20s（服务启动时已预热，首请求不冷启动）。
