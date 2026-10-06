# RAG 角色扮演系统 — 设计文档

- 日期：2026-09-18
- 状态：已确认（待进入实现计划）
- 类型：多用户自部署 Web 应用

## 1. 项目概述

一个**基于 RAG 的角色扮演陪聊系统**，多用户自部署、前后端分离 Web、Python 后端、国内模型驱动。

核心能力：通过 RAG 检索「长期记忆」与「角色/世界观设定」来增强对话。支持预设角色 + 用户自建角色卡，每个角色拥有独立记忆与设定，角色间记忆隔离。

### 1.1 关键决策（已确认）

| 决策点 | 结论 |
|--------|------|
| 角色形态 | 多角色（预设 + 用户自建，角色间记忆隔离） |
| RAG 检索内容 | 长期记忆 + 角色/世界观设定 |
| LLM | DeepSeek（接口抽象，后续可换通义/本地模型） |
| Embedding / Rerank | BGE-m3（dense + sparse 双向量）、BGE-rerank，CPU 版独立推理服务 |
| 向量库 | Milvus ≥ 2.4（原生 BM25），混合检索 |
| 关系数据 | MySQL |
| 短期记忆 | Redis |
| 交互形态 | 前后端分离 Web（FastAPI + 前端 SPA） |
| 规模 | 多用户自部署 |
| 角色来源 | 预设 + 自定义 |
| 异步任务 | ARQ（与 FastAPI 同源 asyncio） |

## 2. 整体架构

```
                          ┌─────────────────────────────────────────────┐
                          │              浏览器(React/Vue SPA)          │
                          └──────────────────────┬──────────────────────┘
                                                 │ HTTPS / WebSocket(SSE 流式)
                          ┌──────────────────────▼──────────────────────┐
                          │            Nginx(反向代理 / 静态资源)         │
                          └──────────────────────┬──────────────────────┘
                                                 │
                          ┌──────────────────────▼──────────────────────┐
                          │              FastAPI 后端(核心)              │
                          │                                             │
                          │  · 认证/用户模块      · 角色卡管理模块        │
                          │  · 会话管理模块        · 对话编排 Orchestrator │
                          │  · 记忆抽取服务(异步)  · Embedding 客户端     │
                          └───┬───────┬───────┬───────┬────────┬────────┘
                              │       │       │       │        │
                 ┌────────────▼─┐ ┌───▼───┐ ┌─▼─────┐ │        │
                 │    MySQL     │ │ Redis │ │Milvus │ │        │
                 │ 用户/角色卡/  │ │短期记忆│ │长期记忆│ │        │
                 │ 会话元数据    │ │会话缓存│ │知识库  │ │        │
                 └──────────────┘ └───────┘ └───────┘ │        │
                                                      │        │
                              ┌───────────────────────▼──┐ ┌───▼──────────┐
                              │  BGE-m3 服务(embedding)   │ │  BGE-rerank  │
                              │  dense + sparse 双向量     │ │  重排序服务   │
                              └──────────────────────────┘ └──────────────┘
                                                      │
                              ┌───────────────────────▼──────────────────┐
                              │   国内 LLM(云端 API:DeepSeek / 通义等)      │
                              └──────────────────────────────────────────┘
```

### 组件清单与职责

| 组件 | 用途 | 说明 |
|------|------|------|
| Nginx | 反向代理 + 前端静态托管 | 统一入口、SSE 流式转发 |
| FastAPI | 业务编排 | 认证、角色卡、会话、对话编排、记忆抽取 |
| MySQL | 持久化关系数据 | 用户、角色卡（人设/世界观）、会话、消息元数据 |
| Redis | 短期记忆 + 缓存 + 队列 | 最近 N 轮对话、会话状态、限流、ARQ 队列 |
| Milvus | 知识库 / 长期记忆 | `character_settings` 与 `long_term_memory` 两个 collection |
| BGE-m3 | 嵌入模型 | dense（1024 维）+ sparse（词权重）双向量，支撑混合检索 |
| BGE-rerank | 重排序 | 对混合检索候选精排 |
| 国内 LLM | 生成 | 角色扮演对话生成引擎（DeepSeek） |

## 3. 数据模型

### 3.1 MySQL（关系数据，`utf8mb4`）

**`users` 用户表**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT PK | 自增主键 |
| username | VARCHAR(64) UNIQUE | 登录名 |
| password_hash | VARCHAR(255) | bcrypt 哈希 |
| created_at / updated_at | DATETIME | 时间戳 |

**`characters` 角色卡表**（预设 + 自定义合一）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT PK | 角色 id |
| owner_user_id | BIGINT NULL | NULL = 预设角色；非空 = 某用户自建 |
| name | VARCHAR(64) | 角色名 |
| avatar | VARCHAR(512) | 头像 URL |
| persona | TEXT | 人设（性格/身份/说话风格） |
| worldview | TEXT | 世界观背景 |
| relationship | TEXT | 角色与用户的初始关系 |
| hidden_setting | TEXT | 隐藏设定（只影响行为，不主动剧透） |
| greeting | TEXT | 开场白 |
| sample_dialogue | TEXT | 示例对话（约束说话风格，可选） |
| is_preset | TINYINT(1) | 是否预设 |
| tags | VARCHAR(255) | 逗号分隔标签 |
| created_at / updated_at | DATETIME | 时间戳 |

> `persona` / `worldview` / `relationship` / `hidden_setting` / `sample_dialogue` 会切 chunk 进 Milvus 做检索；`greeting` 仅用于开场（新建会话时作为 assistant 首条消息），**不入库 Milvus**；`hidden_setting` 在详情接口默认不下发（见 §4.2）。

**`sessions` 会话表**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT PK | 会话 id |
| user_id | BIGINT | 属于谁 |
| character_id | BIGINT | 和哪个角色聊 |
| title | VARCHAR(128) | 自动生成的会话标题 |
| created_at / updated_at | DATETIME | 更新驱动会话列表排序 |

**`messages` 消息表**（全量落库，回溯 + 记忆抽取源）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT PK | 消息 id |
| session_id | BIGINT | 属于哪个会话 |
| role | ENUM('user','assistant') | 角色 |
| content | TEXT | 内容 |
| token_count | INT | 估算 token，用于计费/截断 |
| created_at | DATETIME | 时间 |

**`memory_tasks` 抽取任务表**（异步抽取进度台账）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT PK | |
| session_id | BIGINT | 待抽取的会话 |
| last_processed_msg_id | BIGINT | 已处理到哪条消息（断点续抽） |
| status | ENUM('pending','running','done','failed') | |
| created_at / updated_at | DATETIME | |

### 3.2 Redis（短期记忆 + 缓存 + 队列）

| Key 模式 | 类型 | 内容 | TTL |
|----------|------|------|-----|
| `session:{id}:history` | LIST | 最近 N 轮（user/assistant 交替）JSON 消息 | 跟随会话活跃期 |
| `session:{id}:summary` | STRING | 超出 N 轮后被压缩的历史摘要 | 长期 |
| `char:{id}` | STRING | 角色卡缓存（热点读） | 10min |
| `user:{id}:rate` | STRING | 限流计数器 | 窗口期 |
| `queue:memory` | LIST/STREAM | 记忆抽取任务队列（投递 target） | — |

> **短期/长期记忆分工**：Redis 管「最近上下文」（最近 N 轮原样拼 prompt + 滚动摘要），Milvus 管「跨会话/久远的事实」。

### 3.3 Milvus（两个 collection）

**Collection A：`character_settings`（角色/世界观设定，只读为主）**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INT64 PK | |
| dense_vector | FLOAT_VECTOR(1024) | BGE-m3 dense |
| sparse_vector | SPARSE_FLOAT_VECTOR | BGE-m3 sparse（BM25 用） |
| character_id | INT64 | 过滤键 |
| setting_type | VARCHAR | `persona` / `worldview` / `relationship` / `hidden_setting` / `sample_dialogue` |
| text | VARCHAR | chunk 原文 |
| chunk_index | INT64 | chunk 序号 |

**Collection B：`long_term_memory`（长期记忆，动态写入）**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INT64 PK | |
| dense_vector | FLOAT_VECTOR(1024) | |
| sparse_vector | SPARSE_FLOAT_VECTOR | |
| user_id | INT64 | 过滤键 |
| character_id | INT64 | 过滤键 |
| memory_type | VARCHAR | `fact` / `event` / `preference` |
| content | VARCHAR | 记忆内容 |
| importance | INT8 | 0-10 重要度（影响检索权重与遗忘） |
| source_msg_ids | VARCHAR | 来源消息 id（可回溯） |
| created_at | INT64 | 时间戳 |
| last_access_at | INT64 | 最后命中时间（可做遗忘） |

> 两个 collection 都建 dense（HNSW）+ sparse（BM25/inverted）索引，检索用 RRF 融合。`character_settings` 按 `character_id` 过滤；`long_term_memory` 按 `user_id + character_id` 过滤 —— **保证角色间记忆隔离**。

## 4. API 设计

统一约定：`/api/v1` 前缀；JWT（Bearer Token）认证；响应格式 `{ "code": 0, "data": ..., "message": "ok" }`；流式接口用 SSE。

### 4.1 认证与用户

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/auth/register` | 注册 |
| POST | `/api/v1/auth/login` | 登录，返回 JWT |
| GET | `/api/v1/users/me` | 当前用户信息 |

### 4.2 角色卡

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/characters` | 角色列表（预设 + 自建；`?tag=` `?keyword=`） |
| GET | `/api/v1/characters/{id}` | 角色详情 |
| POST | `/api/v1/characters` | 创建自定义角色卡 |
| PUT | `/api/v1/characters/{id}` | 编辑（仅 owner） |
| DELETE | `/api/v1/characters/{id}` | 删除（仅 owner；预设不可删） |
| POST | `/api/v1/characters/{id}/index` | 手动触发该角色设定写入 Milvus（后台自动为主，此为手动重试） |

> **`hidden_setting` 可见性**：owner 自己可见，其他人不可见，预设角色不下发。

### 4.3 会话与消息

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/sessions` | 会话列表 |
| POST | `/api/v1/sessions` | 新建会话（`character_id`） |
| GET | `/api/v1/sessions/{id}/messages` | 拉取历史消息 |
| DELETE | `/api/v1/sessions/{id}` | 删除会话 |

### 4.4 对话（核心）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/chat` | 非流式：一次性返回完整回复 |
| POST | `/api/v1/chat/stream` | 流式（SSE）：逐 token 返回（推荐） |
| POST | `/api/v1/chat/by_character` | 便捷接口：按 `character_id` 自动建/复用会话 |

请求体：`{ "session_id": 123, "content": "...", "stream": true }`

SSE 事件约定：

```
event: delta   data: {"token": "……"}
event: done    data: {"message_id": 456, "usage": {...}}
event: error   data: {"code": 50001, "message": "..."}
```

> **SSE 认证**：前端用 fetch-stream（能带 Authorization 头），后端输出标准 SSE 格式；`?token=` 作为兜底保留，非主方案。

### 4.5 记忆管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/sessions/{id}/memories` | 查看该会话抽取到的长期记忆 |
| DELETE | `/api/v1/memories/{id}` | 删除某条长期记忆（用户纠错） |
| POST | `/api/v1/sessions/{id}/extract` | 手动触发一次记忆抽取 |

## 5. RAG 编排与记忆抽取

### 5.1 对话编排流水线（回答时）

每轮对话走 5 步，检索步骤并行化 + 优雅降级。

```
用户消息到达
  ├─ ① 取短期上下文(Redis)：最近 N 轮 + 历史滚动摘要
  ├─ ② 并行检索 ── a. character_settings 混合检索角色/世界观设定 chunk
  │               b. long_term_memory 混合检索长期记忆
  ├─ ③ 融合 + 精排：dense + sparse(BM25) → RRF 融合 → BGE-rerank → top-M
  ├─ ④ 组装 Prompt
  └─ ⑤ LLM 生成 → SSE 流式返回
```

**混合检索细节（每个 collection 内）：**

```
查询文本 → BGE-m3 → dense 向量 ──► Milvus dense 检索(top-K)
                 → sparse 向量 ──► Milvus sparse/BM25 检索(top-K)
两条结果 → RRF 融合 → 去重排序 → 候选 top-K(默认 20)
候选 → BGE-rerank(query, candidate) → top-M(默认 5)
```

**Prompt 组装结构（顺序重要）：**

```
[System]
你扮演「{角色名}」。
## 角色设定        {persona}
## 世界观          {worldview}
## 你与用户的关系   {relationship}
## 隐藏设定(只影响行为,禁止主动透露)  {hidden_setting}
## 长期记忆(相关往事)  {检索到的 memory top-M}
## 说话风格示例      {sample_dialogue}

[滚动摘要](如有)  {summary}
[最近对话]        {最近 N 轮}
[用户]            {当前消息}
```

### 5.2 记忆抽取流水线（写入时，异步）

**触发条件**（满足其一，阈值配置化）：
- 会话新增消息数 ≥ N（默认 20 轮）未抽取；
- 会话空闲 > T 秒（默认 30s，用户停止发言后）。

```
触发 → 投递任务 → Redis queue:memory（幂等：同会话已有 pending 任务则跳过）
  ↓
后台 Worker(ARQ) 消费：
  1. 取 session 自 last_processed_msg_id 之后的新消息
  2. MemoryExtractor 组装抽取 Prompt → LLM 抽取结构化记忆
     输出 JSON：[{type, content, importance}]
  3. 相似度去重（与已有记忆对比，阈值可配，默认 0.92）
  4. 逐条 → BGE-m3 编码(dense+sparse) → 写入 Milvus long_term_memory
  5. 更新 memory_tasks.last_processed_msg_id（断点续抽）
  6. 失败重试（指数退避），最终失败写 status=failed
```

**抽取输出格式（三类记忆）：**

```json
{
  "memories": [
    {"type": "fact",       "content": "用户是程序员",     "importance": 6},
    {"type": "preference", "content": "用户喜欢猫",       "importance": 7},
    {"type": "event",      "content": "上个月用户去过海边", "importance": 5}
  ]
}
```

> `importance`（0-10）：① 检索时作加分项；② 后续记忆衰减/遗忘依据。

## 6. 接口抽象（LLM / Embedding / Rerank 可替换）

核心原则：业务代码只依赖抽象协议，不 import 具体 SDK，用依赖注入装配。

```
app/
  core/
    providers/
      base.py            # 抽象协议(Protocol / ABC)
      llm/               # deepseek.py / qwen.py / local.py（OpenAI 兼容）
      embedding/         # bge_m3_http.py
      rerank/            # bge_rerank_http.py
  services/
    memory_extractor.py  # 抽取 Prompt 组装、JSON 解析、去重逻辑（独立服务）
```

### 6.1 抽象协议

**`LLMProvider`**（通用生成，保持纯粹通用）

| 方法 | 说明 |
|------|------|
| `chat(messages, **opts) -> str` | 非流式生成 |
| `chat_stream(messages, **opts) -> AsyncIterator[str]` | 流式生成 |

**`EmbeddingProvider`**

| 方法 | 说明 |
|------|------|
| `encode_dense(texts) -> list[list[float]]` | dense 向量 |
| `encode_sparse(texts) -> list[sparse_vec]` | sparse 向量（BM25 融合） |
| `encode_query(text) -> (dense, sparse)` | 单条 query 编码 |

**`RerankProvider`**

| 方法 | 说明 |
|------|------|
| `rerank(query, passages, top_m) -> list[ranked]` | 重排返回 top-M |

**`MemoryExtractor`**（独立服务，注入 LLMProvider + EmbeddingProvider + Milvus 客户端）：
- 组装抽取 Prompt；
- 调 LLM 拿 JSON；
- 解析 + 相似度去重；
- 编码写入 Milvus。

> 抽取的 Prompt 组装、解析、去重逻辑全部在 `MemoryExtractor` 内，`LLMProvider` 保持通用（只做 chat + JSON 约束的薄封装）。

### 6.2 关键设计点

1. **LLM 走 OpenAI 兼容协议**：DeepSeek / 通义 / vLLM / Ollama 多支持 `/v1/chat/completions`，换模型主要改 `base_url` + `api_key` + `model`。
2. **BGE-m3 / BGE-rerank 是独立 HTTP 服务**（CPU 版），换 TEI / vLLM / 自封装 FastAPI 只改 `base_url`；后期上 GPU 只改配置与镜像。
3. **降级策略**：每个 Provider 调用包 `try/except`，失败时按 §8.1 降级矩阵处理。
4. **配置驱动**（`config.yaml` / 环境变量）：

```yaml
llm:
  provider: deepseek
  base_url: https://api.deepseek.com
  api_key: ${DEEPSEEK_API_KEY}
  model: deepseek-chat
embedding:
  provider: bge_m3_http
  base_url: http://bge-m3:8080
rerank:
  provider: bge_rerank_http
  base_url: http://bge-rerank:8080
memory:
  extract_every_n_rounds: 20
  extract_idle_seconds: 30
  dedup_threshold: 0.92
```

## 7. RAG 数据流总览

```
① 写入流(记忆抽取,异步):
  对话消息 → MemoryExtractor → LLM 抽取事实 → BGE-m3 编码 → Milvus long_term_memory

② 检索流(回答时):
  用户消息 → Redis 短期上下文
          → Milvus 混合检索(BGE-m3 dense + BM25 sparse) → RRF → BGE-rerank
          → 组装 Prompt → LLM → 流式返回
```

## 8. 错误处理、测试与部署

### 8.1 错误处理

**统一错误码**（响应 `{ code, message, data }`）：

| 区间 | 含义 | 示例 |
|------|------|------|
| 0 | 成功 | — |
| 1xxx | 客户端错误 | 1001 参数错误、1002 未认证、1003 无权限、1004 资源不存在 |
| 2xxx | 业务错误 | 2001 角色卡不属于你、2002 会话不存在、2003 内容为空 |
| 3xxx | 模型/向量服务错误 | 3001 LLM 超时、3002 embedding 不可用、3003 rerank 不可用 |
| 5xxx | 系统/依赖错误 | 5001 MySQL、5002 Redis、5003 Milvus |

**降级矩阵**（核心原则：部分依赖故障 ≠ 对话失败）：

| 故障 | 行为 |
|------|------|
| Milvus / Embedding 不可用 | 跳过长期记忆检索，仅用短期上下文 + 人设，对话继续 |
| Rerank 不可用 | 直接用 RRF 融合结果，不重排 |
| Redis 短期记忆不可用 | 短期上下文缺失，仅用系统人设，对话继续 |
| LLM 失败 | 返回 3001，不静默不编造 |
| MySQL 不可用 | 会话无法持久化，对话直接失败（一致性不可妥协） |

**可观测性**：结构化日志（structlog / JSON）+ 每轮记录检索命中数、重排后条数、token 用量、各步骤耗时。

### 8.2 测试策略

| 层级 | 工具 | 覆盖内容 |
|------|------|---------|
| 单元测试 | pytest | 抽象协议实现、MemoryExtractor 解析/去重、Prompt 组装、RRF 融合、配置加载 |
| 接口测试 | pytest + httpx(TestClient) | API 鉴权、CRUD、错误码、`hidden_setting` 可见性 |
| 集成测试 | pytest + testcontainers | MySQL/Redis/Milvus 容器化，测真实检索写入链路 |
| Provider 假实现 | fake/mock | LLM/Embedding/Rerank 用内存假实现，测试不依赖外部 API |
| RAG 效果评估 | **RAGAS** | faithfulness / answer_relevancy / context_precision / context_recall 等指标 |

> LLM / Embedding / Rerank 一律用假实现注入，单测/接口测不碰真实外部服务；Milvus/MySQL/Redis 用 testcontainers 做集成测试。RAG 效果评估用 **RAGAS** 框架（配合合成/标注评测集）。

### 8.3 部署与压测

**Docker Compose 编排：**

```
docker-compose.yml
  ├─ nginx         # 反向代理 + 前端静态托管
  ├─ frontend      # 前端构建产物
  ├─ api           # FastAPI(uvicorn)，可水平扩展
  ├─ worker        # ARQ 后台 worker（记忆抽取）
  ├─ mysql         # 用户/角色/会话
  ├─ redis         # 短期记忆 + 缓存 + ARQ 队列
  ├─ milvus        # Milvus 2.4+（依赖 etcd + minio）
  │   ├─ etcd
  │   └─ minio
  ├─ bge-m3        # CPU 版嵌入服务（自封装 FastAPI/TEI）
  └─ bge-rerank    # CPU 版重排服务
```

**部署要点：**
1. Milvus 用官方 standalone 编排（milvus + etcd + minio），版本锁 2.4+。
2. `api` 与 `worker` 跑同一份代码，启动命令区分（`uvicorn` vs `arq run`）。
3. BGE-m3 / BGE-rerank 用 CPU 版，容器内跑自封装 FastAPI 服务；接口预留，后期换 GPU/vLLM 只改 `base_url` 与镜像。
4. 密钥（DeepSeek key 等）用 `.env` 注入，不进 git；`config.yaml` 的 `${...}` 启动时替换。
5. 持久化卷：MySQL / Redis / Milvus（etcd+minio）全部挂 volume。

**压力测试：** 部署后用 **JMeter** 做压力测试，关注 QPS、P95/P99 延迟、流式接口吞吐，验证降级矩阵下的系统稳定性。
