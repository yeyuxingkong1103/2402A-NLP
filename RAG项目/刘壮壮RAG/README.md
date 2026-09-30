# RAG Assistant

多用户、多角色的 RAG 智能聊天助手。支持本地（Ollama/vLLM）与云端大模型、知识库文件（txt/pdf/图片）、多轮流式对话、Redis 短期记忆、Milvus 长期记忆。

## 功能特性

- **多用户**：注册/登录（JWT），数据按用户完全隔离
- **角色管理**：创建/编辑/删除角色，内置 6 个模板（NPC 朋友、虚拟朋友、中医、心理医生、律师、金融理财师），也支持完全自定义
- **知识库**：上传 txt/md/PDF/图片，图片走 OCR 文字检索（视觉向量入库备用）
- **多轮流式对话**：SSE 流式输出，上下文由 Redis 短期记忆承载
- **双层记忆**：Redis 短期记忆（含摘要压缩）+ Milvus 长期记忆（对话事实异步沉淀）
- **模型灵活接入**：统一 OpenAI 兼容协议，本地 Ollama/vLLM 与云端 API 通用

## 技术栈

| 层 | 选型 |
|----|------|
| 后端 | FastAPI（Python 3.11）+ SQLAlchemy 2.0 (async) |
| 存储 | MySQL（元数据）、Redis（短期记忆 + 任务队列）、Milvus（向量） |
| 异步任务 | arq（复用 Redis） |
| 前端 | 纯 HTML/CSS/JS（FastAPI 静态托管） |
| 部署 | Docker Compose 一键启动 |

---

## 快速开始

### 方式一：Docker 一键启动（推荐）

**前置**：已安装 Docker Desktop 并启动。

```powershell
# 1. 复制配置模板
Copy-Item .env.example .env

# 2. 编辑 .env，填入真实模型 API Key（至少 LLM + Embedding）
#    LLM_API_KEY=sk-xxx
#    EMBEDDING_API_KEY=sk-xxx

# 3. 一键启动（MySQL/Redis/Milvus/App/Worker）
docker compose up -d --build

# 4. 访问
# 前端：http://localhost:8000/
# 接口文档：http://localhost:8000/docs
```

### 方式二：本地开发

```powershell
# 1. 启动中间件（Docker 只起中间件）
docker compose up -d mysql redis etcd minio milvus

# 2. 创建虚拟环境并安装依赖
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env

# 3. 启动应用（终端 1）
uvicorn app.main:app --reload

# 4. 启动 worker（终端 2，处理文件入库与记忆沉淀）
python -m app.worker.main
```

---

## 配置说明（.env）

| 变量 | 说明 |
|------|------|
| `LLM_BASE_URL` / `LLM_API_KEY` | 对话模型（OpenAI 兼容）。本地 Ollama 填 `http://localhost:11434/v1`，key 填 `ollama` |
| `EMBEDDING_MODEL` / `EMBEDDING_API_KEY` | 文本向量模型（默认 `text-embedding-3-small`） |
| `VISION_EMBEDDING_*` | 图片视觉向量多模态 API（v1 仅入库备用，不检索） |
| `TEXT_EMBEDDING_DIM` / `IMAGE_EMBEDDING_DIM` | 向量维度，须与所选 embedding 模型一致（换模型时同步改） |
| `MYSQL_*` / `REDIS_*` / `MILVUS_*` | 中间件连接信息（Docker 下会被 compose 覆盖为服务名） |
| `JWT_SECRET` | 生产环境务必修改 |

---

## 使用流程

1. 打开 http://localhost:8000/login.html，注册并登录
2. 点左侧「＋」创建角色：可选模板（如「中医」）或自定义系统提示词与模型
3. 选中角色，在底部「知识库」面板上传文件（txt/md/PDF/图片）
4. 点「新对话」开始聊天，回答以流式逐字渲染
5. 每个角色可建多个会话，会话间上下文独立；不同角色/用户的资料完全隔离

---

## 测试

```powershell
pytest -v
```

测试**无需启动任何外部中间件**（用 SQLite 内存库、fakeredis、monkeypatch 隔离）。当前 **42 个测试全部通过**。

---

## 验证记录（2026-09-22）

已通过实测验证（应用以 SQLite 覆盖 + 无中间件模式启动）：

| 项目 | 结果 |
|------|------|
| 健康检查 `/api/health` | ✅ 200 `{"status":"ok"}` |
| 静态页面 `/`、`/login.html` | ✅ 200 |
| 注册 / 登录 / `me` | ✅ 201 / 200 token / 200 |
| 角色模板列表 | ✅ 6 个模板 |
| 创建角色（模板）、建会话 | ✅ 201 |
| 上传 txt（`status=pending`） | ✅ 201 |
| 拒绝 .exe | ✅ 400「不支持的文件类型」 |
| 删除文件 | ✅ 204 |
| SSE 对话（模型不可用时降级） | ✅ 200，返回 `error` + `[DONE]` 事件，不中断连接 |

> 完整端到端（对话引用知识库、长期记忆）需 Docker + 真实模型 API Key，见「快速开始」。

---

## 目录结构

```text
app/
├── main.py              # FastAPI 入口 + 路由注册 + 静态托管
├── config.py            # 配置（pydantic-settings）
├── database.py          # async engine / session
├── models/              # ORM：User/Character/Conversation/KnowledgeFile
├── schemas/             # Pydantic DTO
├── core/                # security/llm/embeddings/vision/vector_store/memory/retriever/file_processing/...
├── services/            # 业务逻辑
├── worker/              # arq 任务（process_file/extract_memory）
├── api/routes/          # health/auth/characters/knowledge/conversations/chat
└── static/              # 前端（login.html/index.html/app.js/style.css）
tests/                   # 42 个测试
specs/001-rag-assistant/ # speckit 设计文档（spec/plan/tasks/contracts/...）
docs/superpowers/        # superpowers 计划文档
```
