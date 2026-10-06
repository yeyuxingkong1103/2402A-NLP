# RAG 知识库问答系统

基于 **LangChain + Milvus + BGE-M3 + BGE-Reranker + DeepSeek** 的中文知识库问答（RAG）项目。

文档入库 → 向量召回 → BGE 重排 → DeepSeek 生成，支持流式输出与来源引用标注。

## 架构流程

```
                 ┌─ 入库链路（ingest）──────────────────────────┐
 docs/ 文档 ──► Loader 加载（pdf/txt/md/docx/csv）
        ──► RecursiveCharacterTextSplitter 切分（512 字符，重叠 50）
        ──► BGE-M3 向量化 ──► Milvus（HNSW + COSINE）

                 ┌─ 问答链路（ask / chat）──────────────────────┐
 用户问题 ──► BGE-M3 向量化 ──► Milvus 召回 Top-10
        ──► BGE-Reranker 精排取 Top-3
        ──► 拼装提示词（带编号与来源）
        ──► DeepSeek 生成答案（流式输出，末尾标注 [编号] 引用）
```

## 项目结构

```
rag/
├── main.py           # CLI 入口：ingest / ask / chat / search / reset / serve
├── config.py         # 全局配置（系统环境变量读取，均有默认值）
├── loaders.py        # 文档加载与切分
├── embeddings.py     # BGE-M3 向量化（HuggingFaceEmbeddings，1024 维 dense）
├── vector_store.py   # Milvus 连接 / 建集合 / 写入 / 检索（支持按领域选集合）
├── reranker.py       # BGE 重排器（BaseDocumentCompressor）
├── rag_chain.py      # RAG 主链（LCEL 编排：召回 → 重排 → 生成，支持历史注入）
├── chat_store.py     # Redis 对话历史存储（按用户+领域分 key，TTL 自动过期）
├── llm.py            # DeepSeek LLM（OpenAI 兼容接口）
├── database.py       # 数据库连接与用户模型（SQLAlchemy + PostgreSQL）
├── auth.py           # 用户认证：注册 / 登录 / JWT / 角色权限校验
├── api.py            # FastAPI 路由：注册 / 登录 / 问答 / 历史管理
├── requirements.txt  # 依赖清单
```

## 环境要求

- Python 3.10+
- Milvus 服务（本地 Docker 或 Zilliz Cloud）
- Redis 服务（本地或远程，用于多轮对话历史存储）
- PostgreSQL 服务（用于用户表，本地或远程）
- DeepSeek API Key（[申请地址](https://platform.deepseek.com/)）
- Embedding / Reranker 模型（本地已放置于 `D:\models\`，或首次运行自动从 HuggingFace 下载）
- 可选：NVIDIA GPU + CUDA（未配置则自动用 CPU）

## 安装步骤

### 1. 安装 Python 依赖

```bash
conda create -n rag python=3.10 -y
conda activate rag
pip install -r requirements.txt
```

### 2. 启动 Milvus（Windows 用 Docker Desktop）

```bash
# Milvus standalone 单机版
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

验证：浏览器访问 `http://localhost:19530` 端口有响应即启动成功。

### 3. 启动 Redis（多轮对话历史需要）

```bash
# Windows：Docker 一行启动
docker run -d --name redis -p 6379:6379 redis

# 或用 Windows 本地版：下载后直接运行 redis-server.exe
```

验证：`redis-cli ping` 返回 `PONG` 即启动成功。Redis 为必需依赖（ask/chat 的多轮对话历史存储）。

### 4. 启动 PostgreSQL（多用户模式需要）

```bash
# Windows：Docker 一行启动
docker run -d --name pg -e POSTGRES_PASSWORD=123456 -e POSTGRES_DB=rag -p 5432:5432 postgres

# 或安装本地 PostgreSQL 后创建数据库
createdb -U postgres rag
```

验证：`psql -U postgres -d rag` 能连上即成功。不使用多用户模式（只用命令行）时可不启动 PostgreSQL。

### 5. 配置环境变量（系统级）

项目不使用 `.env` 文件，直接在 Windows 系统环境变量中配置。必填项只有 DeepSeek API 密钥：

```powershell
# PowerShell（永久生效，需重开终端）
setx DEEPSEEK_API_KEY1 "sk-你的密钥"
```

或在「系统属性 → 高级 → 环境变量」中添加。变量名以 `config.py` 中 `os.getenv` 实际读取的为准（当前为 `DEEPSEEK_API_KEY1`，另支持 `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` / `DOCS_DIR` / `USE_FP16` / `PG_HOST` / `PG_PORT` / `PG_USER` / `PG_PASSWORD` / `PG_DB` / `JWT_SECRET` / `JWT_EXPIRE_HOURS`）。其余参数（模型路径、设备、检索参数等）已直接写在 `config.py` 中，按需修改该文件。

### 6. 放置模型

当前 `config.py` 指向本地模型：

| 配置项 | 路径 |
|---|---|
| `BGE_M3_MODEL` | `D:\models\bge-m3` |
| `BGE_RERANKER_MODEL` | `D:\models\bge-reranker-large` |

若无本地模型，可改为 HuggingFace ID（如 `BAAI/bge-m3`），首次运行自动下载；国内加速可设置环境变量 `HF_ENDPOINT=https://hf-mirror.com`。

## 快速开始

所有命令均需指定 `--domain`（medical / education），系统按领域隔离知识库与对话历史，历史保留最近 10 轮，Redis TTL 1 天自动过期。`--domain` 只接受 `config.domain_collections` 的键，传集合名（如 `rag_medical`）会被拒绝。

```bash
# 1. 文档入库（--dir 指定目录、--domain 指定领域，两者必填）
python main.py ingest --dir ./docs/medical --domain medical
python main.py ingest --dir ./docs/education --domain education

# 2. 单次问答（自动读写该领域 Redis 历史）
python main.py ask "什么是过敏" --domain medical

# 3. 交互式问答（进入时显示最近历史，多轮记得上下文）
python main.py chat --domain medical

# 4. 只看检索效果（召回 + 重排结果，不生成答案）—— 调参利器
python main.py search "什么是RAG" --domain education

# 5. 清空向量集合（重灌数据前使用）
python main.py reset --domain medical
```

> 已无「不分领域」模式：所有 ask/chat/search/reset 必须带 `--domain`，Redis 为必需依赖（用于多轮对话历史）。

### 多用户多角色模式（FastAPI）

支持用户注册/登录（JWT 鉴权）、角色权限控制、对话历史按用户隔离。

```bash
# 1. 启动 API 服务（首次自动建表，第一个注册的用户自动成为 admin）
python main.py serve
# 或指定 host/port
python main.py serve --host 0.0.0.0 --port 8000

# 2. 注册用户（doctor 角色只能查 medical，teacher 只能查 education）
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"doctor1\",\"password\":\"123456\",\"role\":\"doctor\"}"

# 3. 登录拿 token
curl -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"doctor1\",\"password\":\"123456\"}"
# 返回 {"token":"eyJ...","role":"doctor"}

# 4. 提问（带 token，只能问 medical 领域）
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"什么是过敏\",\"domain\":\"medical\"}"

# 5. 越权测试（doctor 访问 education 应返回 403）
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"课程安排\",\"domain\":\"education\"}"
# 预期 403 Forbidden

# 6. 查看当前角色可访问的领域
curl http://localhost:8000/domains -H "Authorization: Bearer eyJ..."

# 7. 查看历史对话
curl "http://localhost:8000/history?domain=medical" -H "Authorization: Bearer eyJ..."

# 8. 清空历史
curl -X POST http://localhost:8000/clear-history \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d "{\"domain\":\"medical\"}"
```

> API 文档（Swagger）：启动后访问 `http://localhost:8000/docs`
> 角色权限映射：`doctor` → medical，`teacher` → education，`admin` → 全部领域

## 配置参数

配置全部集中在 `config.py`。其中**以下几项**通过系统环境变量读取（`os.getenv`）：`DEEPSEEK_API_KEY1`（必填）、`REDIS_HOST` / `REDIS_PORT` / `REDIS_DB`、`DOCS_DIR`、`USE_FP16`、`PG_HOST` / `PG_PORT` / `PG_USER` / `PG_PASSWORD` / `PG_DB`、`JWT_SECRET` / `JWT_EXPIRE_HOURS`；其余参数直接修改 `config.py`。

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `deepseek_api_key` | 无（必填，环境变量 `DEEPSEEK_API_KEY1`） | DeepSeek API 密钥 |
| `deepseek_base_url` | `https://api.deepseek.com` | API 地址 |
| `bge_m3_model` | `D:\models\bge-m3` | 向量模型（1024 维 dense） |
| `bge_reranker_model` | `D:\models\bge-reranker-large` | 重排模型 |
| `device` | `cuda` | `cpu` / `cuda`（无 GPU 自动降级） |
| `use_fp16` | `true`（环境变量 `USE_FP16`） | 半精度推理（CPU 自动降级 fp32，仅重排生效） |
| `milvus_uri` | `http://localhost:19530` | Milvus 地址 |
| `domain_collections` | `{"medical": "rag_medical", "education": "rag_education"}` | 领域 → 集合映射表（唯一集合来源，无默认集合） |
| `retrieve_k` | `10` | 向量召回条数 |
| `rerank_top_n` | `3` | 重排后送入 LLM 的条数 |
| `chunk_size` | `512` | 切分片段最大字符数 |
| `chunk_overlap` | `50` | 相邻片段重叠字符数 |
| `docs_dir` | `docs`（环境变量 `DOCS_DIR` 可覆盖） | 默认文档目录 |
| `redis_host` | `localhost`（环境变量 `REDIS_HOST`） | Redis 地址 |
| `redis_port` | `6379`（环境变量 `REDIS_PORT`） | Redis 端口 |
| `redis_db` | `0`（环境变量 `REDIS_DB`） | Redis 数据库编号 |
| `history_ttl` | `86400`（1 天） | 对话历史过期时间（秒） |
| `history_turns` | `10` | 每用户每领域保留最近 N 轮对话 |
| `pg_host` | `localhost`（环境变量 `PG_HOST`） | PostgreSQL 地址 |
| `pg_port` | `5432`（环境变量 `PG_PORT`） | PostgreSQL 端口 |
| `pg_user` | `postgres`（环境变量 `PG_USER`） | PostgreSQL 用户名 |
| `pg_password` | 无（环境变量 `PG_PASSWORD`） | PostgreSQL 密码 |
| `pg_db` | `rag`（环境变量 `PG_DB`） | PostgreSQL 数据库名 |
| `jwt_secret` | `change-me-in-production`（环境变量 `JWT_SECRET`） | JWT 签名密钥（生产环境务必修改） |
| `jwt_expire_hours` | `24`（环境变量 `JWT_EXPIRE_HOURS`） | JWT 令牌过期时间（小时） |
| `role_domains` | `{"doctor":["medical"], "teacher":["education"], "admin":["medical","education"]}` | 角色 → 可访问领域映射（无 guest 角色） |

## 常见问题

**Q: 中文文档读取乱码 / 编码报错？**
`TextLoader` / `CSVLoader` 已开启 `autodetect_encoding`，兼容 GBK / UTF-8；仍失败可将文件另存为 UTF-8。

**Q: 连接 Milvus 失败？**
确认 Docker 容器在运行（`docker ps | findstr milvus`），且 `milvus_uri` 与实际端口一致。

**Q: Redis 连接失败 / 多轮对话不保存历史？**
确认 Redis 服务已启动（`redis-cli ping` 返回 `PONG`），且 `redis_host` / `redis_port` 配置正确。Redis 为必需依赖（所有 ask/chat 均需 `--domain`，均会读写历史）。

**Q: 对话历史保留多久？**
默认保留 1 天（`history_ttl`），每用户每领域只保留最近 10 轮（`history_turns`），可在 `config.py` 中调整。

**Q: API 服务启动报 "连接 PostgreSQL 失败"？**
确认 PostgreSQL 服务已启动，且 `pg_host` / `pg_port` / `pg_user` / `pg_password` / `pg_db` 配置正确。可设环境变量 `PG_PASSWORD` 为你的实际密码。

**Q: 登录后 token 多久过期？**
默认 24 小时（`jwt_expire_hours`），可设环境变量 `JWT_EXPIRE_HOURS` 调整。过期后需重新登录。

**Q: 第一个注册的用户是什么角色？**
自动成为 `admin`（管理员），可访问所有领域。之后注册的用户按请求中指定的 `role` 分配。

**Q: 如何限制某用户只能访问特定领域？**
通过 `config.py` 的 `role_domains` 映射控制。例如给 `doctor` 角色加 `education` 权限：

```python
role_domains = {
    "doctor": ["medical", "education"],
    ...
}
```

**Q: 改了 Embedding 模型后检索结果异常？**
不同模型向量空间不同，需 `python main.py reset` 后重新 `ingest`。

**Q: 如何调优检索质量？**
用 `python main.py search "问题"` 直接观察召回与相关度得分，再调整 `retrieve_k` / `rerank_top_n` / `chunk_size`。
