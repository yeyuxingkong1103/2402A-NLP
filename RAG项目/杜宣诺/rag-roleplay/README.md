# RAG 角色扮演系统

一个基于 **RAG** 的多用户角色扮演陪聊后端：通过混合检索（Milvus dense + BM25 sparse + RRF + BGE-rerank）增强长期记忆与角色设定，支持预设角色 + 用户自建角色卡，角色间记忆隔离。

- 后端：FastAPI + SQLAlchemy 2.0（async）+ ARQ
- 存储：MySQL（关系数据）、Redis（短期记忆 / ARQ 队列）、Milvus（长期记忆 / 设定向量库）
- 模型：DeepSeek（LLM，OpenAI 兼容）、BGE-m3（embedding）、BGE-rerank（重排）
- 前端：`frontend/index.html` 单页聊天页（对接流式接口）

---

## 1. 目录结构

```
├── backend/                 # FastAPI 后端
│   ├── app/
│   │   ├── api/             # 路由（auth / characters / sessions / chat / memories）
│   │   ├── services/        # RAG 管线、对话、记忆抽取、Prompt 组装
│   │   ├── store/           # MySQL / Redis / Milvus 存储
│   │   ├── core/            # 提供方抽象协议 + 假实现 + RRF + 安全
│   │   ├── worker/          # ARQ 记忆抽取 worker
│   │   ├── config.py        # Settings（pydantic-settings，读环境变量 / .env）
│   │   ├── deps.py          # 依赖装配（按配置选择 provider 实现）
│   │   └── main.py          # create_app 应用工厂
│   └── tests/               # pytest（fake provider，不碰真实外部服务）
├── services/rerank/         # BGE-rerank 封装服务（自封装 FastAPI）
├── frontend/index.html      # 最简聊天前端
└── docker-compose.yml       # 一键编排
```

---

## 2. 配置 `.env`

在仓库根目录创建 `.env`（也可放 `backend/.env`），参考 `backend/.env.example`：

```dotenv
# 数据库 / 缓存 / 向量库
DATABASE_URL=mysql+asyncmy://root:password@localhost:3306/roleplay
REDIS_URL=redis://localhost:6379/0
MILVUS_HOST=localhost
MILVUS_PORT=19530

# JWT —— 生产环境务必替换为 32 字节以上随机串（openssl rand -hex 32）
JWT_SECRET=change-me-to-a-long-random-secret-at-least-32-bytes

# LLM（DeepSeek，OpenAI 兼容）
LLM_PROVIDER=openai_compat
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=sk-xxxxxxxx
LLM_MODEL=deepseek-chat

# Embedding / Rerank 服务地址
EMBEDDING_BASE_URL=http://localhost:8080
RERANK_BASE_URL=http://localhost:8081
```

`docker compose up` 只依赖根目录 `.env` 里的 `DEEPSEEK_API_KEY`（其余容器内已配好默认值）。如需自定义模型，可选加：

```dotenv
DEEPSEEK_API_KEY=sk-xxxxxxxx
RERANK_MODEL=BAAI/bge-reranker-base
HF_ENDPOINT=https://hf-mirror.com
```

---

## 3. 一键启动（Docker Compose）

前置：安装 Docker Desktop 并启动引擎；准备 `DEEPSEEK_API_KEY`。

```bash
# 根目录（先设好 .env）
docker compose up -d
```

会拉起：`mysql` `redis` `milvus(+etcd+minio)` `bge-m3` `bge-rerank` `api` `worker`。

验证各服务健康：

```bash
docker compose ps
curl http://localhost:8000/api/v1/health
# -> {"code":0,"data":{"status":"ok"},"message":"ok"}
```

> bge-m3 / bge-rerank 使用**本地模型**：请把模型文件分别放到 `./models/bge-m3` 与 `./models/bge-reranker-base`（`docker-compose.yml` 中已改为相对路径挂载，也可改回你自己的绝对路径）。模型文件体积大、已加进 `.gitignore`，不会随仓库上传。

---

## 4. 接口文档

启动后访问 **Swagger UI**：

```
http://localhost:8000/docs
```

统一约定：前缀 `/api/v1`；JWT Bearer 认证；响应 `{ "code": 0, "data": ..., "message": "ok" }`；流式接口用 SSE。

### 已实现接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/auth/register` | 注册 `{username, password}` |
| POST | `/api/v1/auth/login` | 登录，返回 `data.token` |
| GET | `/api/v1/users/me` | 当前用户 |
| POST | `/api/v1/characters` | 创建角色卡 |
| GET | `/api/v1/characters/{id}` | 角色详情（`hidden_setting` 仅 owner 可见） |
| GET | `/api/v1/sessions` | 会话列表 |
| POST | `/api/v1/sessions` | 建会话 `{character_id}`（有 greeting 则插入开场白） |
| GET | `/api/v1/sessions/{id}/messages` | 历史消息 |
| DELETE | `/api/v1/sessions/{id}` | 删除会话 |
| POST | `/api/v1/chat` | 非流式对话 `{session_id, content}` |
| POST | `/api/v1/chat/stream` | 流式对话（SSE，推荐） |
| GET | `/api/v1/sessions/{id}/memories` | 查看长期记忆 |
| DELETE | `/api/v1/memories/{id}` | 删除单条记忆 |
| POST | `/api/v1/sessions/{id}/extract` | 手动触发记忆抽取（入 ARQ） |

### 调用示例（curl）

```bash
# 1. 注册 + 登录
curl -s -X POST http://localhost:8000/api/v1/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"alice","password":"secret"}'

TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"alice","password":"secret"}' | python -c 'import sys,json;print(json.load(sys.stdin)["data"]["token"])')

# 2. 建角色卡
CID=$(curl -s -X POST http://localhost:8000/api/v1/characters \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"name":"小白","persona":"温柔的图书馆管理员","greeting":"你好呀"}' \
  | python -c 'import sys,json;print(json.load(sys.stdin)["data"]["id"])')

# 3. 建会话
SID=$(curl -s -X POST http://localhost:8000/api/v1/sessions \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d "{\"character_id\":$CID}" \
  | python -c 'import sys,json;print(json.load(sys.stdin)["data"]["id"])')

# 4. 非流式对话
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d "{\"session_id\":$SID,\"content\":\"你好\"}"

# 5. 流式对话（SSE）
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d "{\"session_id\":$SID,\"content\":\"你好\"}"
```

SSE 事件：

```
event: delta   data: {"token": "……"}
event: done    data: {"message_id": 0}
event: error   data: {"code": 3001, "message": "..."}
```

---

## 5. 本地开发（不经 Docker）

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate          # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -e ".[dev]"

# 跑测试
pytest -q

# 起服务（需自备 MySQL/Redis/Milvus 或改 .env 指向已有实例）
uvicorn app.main:app --reload --port 8000
```

> 单测/接口测全部用 fake provider + SQLite 内存库，不依赖任何外部服务；集成测试（`test_integration.py`）用 testcontainers，需本机 Docker 引擎运行，未运行时自动 skip。

---

## 6. 关键设计

- **混合检索**：查询 → BGE-m3 dense + sparse 双路 → Milvus 各自 top-K → RRF 融合 → BGE-rerank 精排 top-M。
- **记忆隔离**：`long_term_memory` 检索按 `user_id + character_id` 过滤。
- **降级矩阵**：Milvus/Embedding/Rerank/Redis 任一不可用时对话继续（仅 LLM 失败才报错），见设计文档 §8.1。
- **hidden_setting**：仅角色 owner 可见，预设角色不下发；`greeting` 仅作开场白、不入 Milvus。

完整设计见 `docs/superpowers/specs/2026-09-18-rag-roleplay-system-design.md`。
