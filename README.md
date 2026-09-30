# Role RAG · 多角色知识问答系统

一个可离线运行的多用户 / 多角色 RAG（Retrieval-Augmented Generation）项目：
本地大模型推断 + Milvus 混合检索 + Redis 分层记忆，提供流式问答 API 与零依赖 Web 前端。

- 模型全部来自 `D:\modelscope`，不联网下载：**BGE-M3**（稠密 + 稀疏向量）+ **Qwen3-0.6B**（生成）
- 向量库 **Milvus 3.0**：稠密 HNSW + 稀疏倒排，支持原生 `hybrid_search`
- 缓存 / 记忆 **Redis 8.0**：聊天记录、短期记忆、会话摘要、检索缓存、用户会话，覆盖 String/Hash/List/Set/zSet 五类结构
- 混合检索三路召回：稠密向量 + 稀疏词权重 + BM25 关键词，应用侧加权 RRF 融合
- 多用户 + 多角色：内置 11 个角色（本次随机启用 3 个）、RBAC 权限、无状态令牌认证

> 面向用户的说明文字一律使用简体中文；代码 / 命令 / 路径 / 日志保留原文。

## 目录

- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [技术栈映射](#技术栈映射)
- [模型说明](#模型说明)
- [角色与用户](#角色与用户)
- [API 一览](#api-一览)
- [检索与记忆](#检索与记忆)
- [已验证结论](#已验证结论)
- [更多文档](#更多文档)

## 快速开始

前置：Milvus 与 Redis 已在 WSL 中运行（见 `docs/运行手册.md`）。

```powershell
# 1) 切到项目根目录
cd 'D:\桌面D\专高\专高六\项目\RAG_'

# 2) 环境自检（Python / 依赖 / GPU / 模型 / Milvus / Redis / 配置）
& 'D:\an\envs\rags_\python.exe' tools\check_env.py

# 3) 重建知识库（14 篇文档 -> 78 个知识块）
& 'D:\an\envs\rags_\python.exe' tools\ingest_cli.py --all --recreate

# 4) 启动服务（默认 http://127.0.0.1:8020/ ）
.\start.ps1
```

启动后浏览器打开 `http://127.0.0.1:8020/`，用内置账号登录即可对话。

一键脚本快捷用法：

```powershell
.\start.ps1                 # 自检 + 启动
.\start.ps1 -Check          # 只做环境自检
.\start.ps1 -Ingest         # 先重建知识库再启动
.\start.ps1 -Port 8030      # 指定端口
.\start.ps1 -NoWarmup       # 关闭启动后台预热
```

## 目录结构

```text
RAG_/
├─ configs/
│  ├─ config.yaml            # 主配置：模型路径 / Milvus / Redis / 检索 / 记忆 / 安全
│  └─ roles.yaml             # 11 个角色注册表（persona / 知识目录 / 护栏 / 免责声明）
├─ src/role_rag/
│  ├─ config.py              # 分层配置：内置默认 < config.yaml < .env 环境变量
│  ├─ roles.py               # 角色模型、注册表、风险词护栏（RISK_KEYWORDS / SAFETY_TEMPLATE）
│  ├─ jobs.py                # 后台任务（入库 / 上传）进度登记
│  ├─ logging_conf.py        # 日志（控制台 + logs/app.log + logs/error.log）
│  ├─ errors.py              # 业务异常（DependencyError / PermissionError 等）
│  ├─ api/
│  │  ├─ app.py              # FastAPI 应用：全部路由 + SSE 流式 + 异常处理
│  │  ├─ auth.py             # PBKDF2 口令 + HMAC 无状态令牌（TokenCodec）
│  │  └─ schemas.py          # 请求 / 响应 Pydantic 模型
│  ├─ models/
│  │  ├─ embedder.py         # BGE-M3：单次前向输出稠密(1024) + 稀疏词权重
│  │  └─ llm.py              # Qwen3-0.6B：chat template + 流式生成
│  ├─ store/
│  │  ├─ milvus_store.py     # Milvus 封装：role_kb_chunks / role_memory 两集合
│  │  └─ redis_store.py      # Redis 封装：聊天记录 / 记忆 / 缓存 / 用户 / 限流 / 统计
│  ├─ ingest/
│  │  ├─ loaders.py          # md / txt / pdf 文档加载
│  │  ├─ chunking.py         # 结构感知分块（按标题 + 长度切分 + 重叠）
│  │  └─ pipeline.py         # 入库流水线（加载→分块→向量化→写 Milvus→刷新版本）
│  ├─ retrieval/
│  │  ├─ retriever.py        # 三路召回编排 + 缓存 + 版本失效
│  │  ├─ bm25.py             # jieba 分词 + rank_bm25，按 scope 分片内存索引
│  │  └─ fusion.py           # 加权 RRF 融合
│  ├─ memory/memory.py       # 分层记忆：最近轮次 / 摘要 / 事实向量 / 规则抽取
│  └─ rag/
│     ├─ prompts.py          # 提示词：人设 + 记忆 + 知识片段 + 引用协议
│     └─ pipeline.py         # 问答编排：召回→记忆→组装→生成→引用校验→护栏
├─ web/                      # 零 CDN 原生前端：index.html / style.css / app.js
├─ tools/                    # check_env / ingest_cli / kb_health / smoke_api / smoke_retrieval / eval_retrieval
├─ tests/                    # pytest 单元 + 集成测试
├─ data/kb/                  # 知识库源文件：shared + financial_planner + scientist + lawyer
├─ data_cache/               # 运行期缓存（分词过滤表等）
├─ logs/                     # 运行日志
├─ docs/                     # 架构设计 / 运行手册
├─ start.ps1 / start.sh      # 一键启动脚本
├─ requirements.txt          # 依赖清单（与现有环境版本一致）
└─ .env.example              # 环境变量覆盖示例
```

## 技术栈映射

| 需求 | 实现 | 位置 |
| --- | --- | --- |
| 本地模型 | BGE-M3 + Qwen3-0.6B，路径统一 `D:/modelscope` | `configs/config.yaml` → `paths.models_root` |
| 向量库 | Milvus 3.0（稠密 HNSW/COSINE + 稀疏倒排/IP） | `src/role_rag/store/milvus_store.py` |
| Redis 缓存 | 最近聊天记录 + 短期记忆 + 检索缓存 + 会话 | `src/role_rag/store/redis_store.py` |
| 混合检索 | dense + sparse + bm25 三路 → 加权 RRF | `src/role_rag/retrieval/` |
| 多用户 | 登录 + RBAC + 会话隔离 + 限流 | `src/role_rag/api/auth.py`、`api/app.py` |
| 多角色 | 11 角色注册表（启用 3 个） | `configs/roles.yaml`、`src/role_rag/roles.py` |
| 流式问答 | SSE（`text/event-stream`） | `src/role_rag/api/app.py` → `POST /api/chat` |
| 前端 | 原生 JS，无 CDN，4 个面板 tab | `web/` |
## 模型说明

| 用途 | 模型 | 路径 | 输出 |
| --- | --- | --- | --- |
| 向量化 | BGE-M3 | `D:/modelscope/bge-m3` | 稠密 1024 维（CLS 归一化）+ 稀疏词权重（`relu(W·h+b)` + max 池化） |
| 生成 | Qwen3-0.6B | `D:/modelscope/Qwen3-0.6B` | 流式文本，`enable_thinking=false` |

要点：

- BGE-M3 与官方权重同源：稠密向量与 `sentence-transformers` 实测 `cos ≥ 0.999999`
- 稠密 / 稀疏共用一次前向，fp16 单份权重，8G 显存可跑
- 模型懒加载 + 双检锁，避免并发加载；`app.warmup: true` 时启动后台线程预热，不阻塞服务

## 角色与用户

内置 **11 个角色** 定义在 `configs/roles.yaml`，本次按随机抽选启用 **3 个**（其余 8 个定义完整、`enabled: false`，把开关改为 `true` 并补上 `data/kb/<id>/` 目录即可启用）：

| id | 名称 | 状态 | 分类 |
| --- | --- | --- | --- |
| `financial_planner` | 金融理财师 | **启用** | 金融 |
| `scientist` | 科学家 | **启用** | 科研 |
| `lawyer` | 律师 | **启用** | 法律 |
| `customer_service` | 客服 | 未启用 | 服务 |
| `social` | 社交 | 未启用 | 生活 |
| `npc` | NPC（Non Player Character） | 未启用 | 娱乐 |
| `doctor` | 医生 | 未启用 | 医疗 |
| `psychologist` | 心理医生 | 未启用 | 心理 |
| `stock_analyst` | 股票（证券投资） | 未启用 | 金融 |
| `teacher` | 教师 | 未启用 | 教育 |
| `english_tutor` | 英语学习 | 未启用 | 教育 |

内置用户（`configs/config.yaml` → `security.users_seed`，首次启动写入 Redis，之后不再覆盖）：

| 用户名 | 口令 | 可访问角色 | 备注 |
| --- | --- | --- | --- |
| `admin` | `admin123` | `["*"]` 全部 | `is_admin: true` |
| `alice` | `alice123` | financial_planner / scientist / lawyer | 通用用户 |
| `bob` | `bob123` | scientist / lawyer | 无理财权限，用于验证 403 |
| `carol` | `carol123` | financial_planner | 单角色用户 |

> 角色护栏：每个角色可配置 `guardrails` 红线与 `disclaimer` 免责声明。答案命中风险词（如「保证收益」「满仓」「包赢」）时，程序会**强制追加**免责声明；问题本身命中风险词时，会插入 `SAFETY_TEMPLATE` 前置纠正指令（仅对当前问题生效，不污染多轮历史）。

## API 一览

统一前缀 `/api`，除 `/api/health` 与 `/api/auth/login` 外均需认证（`Authorization: Bearer <token>` 或 Cookie）。

| 方法 | 路径 | 说明 | 权限 |
| --- | --- | --- | --- |
| GET | `/api/health` | 健康检查（服务 / 依赖状态） | 公开 |
| POST | `/api/auth/login` | 登录，返回无状态令牌 | 公开 |
| POST | `/api/auth/logout` | 退出，清理在线集合 | 登录 |
| GET | `/api/auth/me` | 当前用户信息与可访问角色 | 登录 |
| POST | `/api/auth/users` | 新建用户 | 管理员 |
| GET | `/api/roles` | 列出当前用户可见的角色 | 登录 |
| POST | `/api/chat` | 流式问答（SSE：`meta` / `delta` / `citations` / `done`） | 登录（角色需授权） |
| GET | `/api/sessions` | 当前用户的会话列表 | 登录 |
| GET | `/api/sessions/{id}` | 会话详情（含消息历史） | 会话归属者 |
| DELETE | `/api/sessions/{id}` | 删除会话 | 会话归属者 |
| GET | `/api/memory` | 记忆总览（摘要 / 事实 / 轮数） | 登录 |
| POST | `/api/memory/clear` | 清空指定范围记忆 | 登录 |
| POST | `/api/retrieval/search` | 单次检索（可指定模式） | 登录 |
| POST | `/api/retrieval/compare` | 四路对比（dense / sparse / bm25 / hybrid） | 登录 |
| GET | `/api/kb/status` | 知识库状态（块数 / 版本 / 文档） | 登录 |
| POST | `/api/kb/ingest` | 触发入库（后台任务） | 管理员 |
| POST | `/api/kb/upload` | 上传文档并入库（multipart） | 管理员 |
| POST | `/api/kb/reset` | 重置知识库集合 | 管理员 |
| GET | `/api/jobs` | 任务列表 | 登录 |
| GET | `/api/jobs/{id}` | 任务详情与进度 | 登录 |
| GET | `/api/stats` | 运行统计 | 登录 |
| GET | `/api/logs` | 查询服务日志 | 登录 |
| GET | `/` | Web 前端首页 | 公开 |

## 检索与记忆

**混合检索（三路召回 + 加权 RRF）**

| 通路 | 实现 | 召回数 | 权重 |
| --- | --- | --- | --- |
| 稠密 | Milvus HNSW / COSINE（BGE-M3 CLS） | 20 | 0.55 |
| 稀疏 | Milvus SPARSE_INVERTED_INDEX / IP（BGE-M3 lexical） | 20 | 0.30 |
| 关键词 | jieba + rank_bm25 内存索引（按 scope 分片） | 20 | 0.15 |

- 融合：加权 RRF（`rrf_k=60`），最终取 `final_k=6`，同一文档最多贡献 3 块
- 另提供 `fusion=milvus` 走 Milvus 原生 `hybrid_search` + `RRFRanker` 做对照
- BM25 索引按 scope 分片，用 Redis `kb:version` 判断失效；入库后自增版本并清检索缓存
- 检索缓存键 `cache:retr:<hash>`，TTL 15 分钟

**分层记忆（Redis + Milvus）**

| 层 | 载体 | 说明 |
| --- | --- | --- |
| 最近聊天记录 | Redis List `sess:<sid>:msgs` | 每轮追加，取最近 N 轮（`short_term_turns=5`）进上下文 |
| 会话摘要 | Redis String `sess:<sid>:summary` | 超过 5 轮触发压缩刷新 |
| 结构化事实 | Redis Hash `user:<uid>:profile` | 昵称 / 偏好 / 累计提问数 |
| 长期事实向量 | Milvus `role_memory` | 规则抽取的事实（`FACT_PATTERNS` + 角色专属 `ROLE_FACT_PATTERNS`），相似度 ≥ 0.35 召回 4 条 |
| 会话元信息 | Redis Hash `sess:<sid>:meta` | owner / role / 标题 / 轮数 / 时间 |
| 在线用户 | Redis Set `online` | 心跳 TTL 300s |
| 引用去重 | Redis Set `sess:<sid>:cites` | 本会话引用过的文档 id |
| 用户会话列表 | Redis zSet `user:<uid>:sessions` | score = 最近活跃时间 |

## 已验证结论

以下均为本机实测通过（RTX 4060 Laptop 8G / Python 3.12 / `D:\an\envs\rags_`）：

- `tools/check_env.py`：Python / 依赖 / GPU / 模型 / Milvus / Redis / 配置 全部 `[ OK ]`
- `tools/kb_health.py`：知识库健康检查，输出各作用域源文件 / 知识块数量与状态
- `tools/ingest_cli.py --all --recreate`：14 篇文档 → **78 个知识块**，`kb_version=1`，耗时约 55.9s
- `tools/smoke_retrieval.py`：三路召回、RRF 融合正常；第二次同查询命中缓存（`cached=True`）
- `tools/smoke_api.py`：**23 项全 PASS**，含登录、401 / 403 RBAC、SSE 事件序列、会话与多轮上下文、长期记忆写入、四路对比、护栏命中（稳赚 / 保证收益 / 满仓 / 抄底）
- `pytest tests`：**75 项通过**（默认跳过 2 项 E2E 用例，耗时约 24s；设 `ROLE_RAG_E2E=1` 时 75 项全跑，约 41s）
- 样例：alice 以律师角色问「试用期最长多久？」回答带 `[1][2]` 引用；以金融理财师角色问「推荐一只稳赚不赔、能满仓抄底的基金」时先纠正「不存在保证收益的产品」并追加 ⚠️ 免责声明；「请记住：我偏好 R2 稳健型产品，我每月可以投入 3000 元」被抽取为长期事实并写入记忆

## 更多文档

- `docs/架构设计.md`：分层架构、数据流、Milvus 表结构、Redis 键设计、检索融合策略、记忆机制、权限模型
- `docs/运行手册.md`：环境准备、启动 / 停止、入库、常见问题排查、API 示例、测试命令
