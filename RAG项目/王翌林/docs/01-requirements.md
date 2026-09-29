# 01 · 需求规格说明

| 项目 | 内容 |
| :--- | :--- |
| 文档版本 | V1.0 |
| 编写日期 | 2026-09-15 |
| 系统名称 | 基于 RAG 的心理医生多角色陪伴系统 |
| 需求来源 | `心理医生需求文档.md`（原始需求规格说明书 V1.1） |
| 实现状态 | 已实现（本文档已按真实代码核对，与原始需求有出入处已标注） |

---

## 1. 项目概述

### 1.1 背景

构建一个基于 RAG（检索增强生成）的角色扮演聊天系统。首期聚焦心理健康陪伴场景，通过大语言模型、向量数据库、知识库检索、重排序、短期记忆、长期记忆，实现可切换、可配置、可扩展的心理医生角色。

系统**不是医疗诊断系统**，不能替代精神科医生、心理治疗师或线下心理咨询。

### 1.2 建设目标

1. 实现三个心理医生角色，多角色可切换、独立提示词、独立知识库、独立会话；
2. 支持多用户、多角色、多会话；
3. 用户管理连接 MySQL；
4. Redis 实现短期记忆，Milvus 实现长期记忆与知识库；
5. 支持离线文档解析、分块、向量化、入库；
6. 支持在线检索、重排序、提示词模板、大模型生成、后处理；
7. 支持 RAGAS 评测、接口测试、压力测试；
8. 可在 Windows 11 + WSL2 + Ubuntu 环境开发部署。

### 1.3 系统范围

**包含**：用户管理、心理医生角色管理、会话与多轮对话、RAG 离线知识库构建、RAG 在线检索生成、Redis 短期记忆、Milvus 长期记忆与知识库、MySQL 业务数据、管理后台/API、评测与测试、部署脚本与文档。

**不包含**：医疗诊断、处方开具、药物推荐、急救调度、真实医生在线接诊。

### 1.4 运行环境与固定路径

| 项目 | 内容 |
| :--- | :--- |
| 操作系统 | Windows 11 + WSL2 + Ubuntu 22.04 |
| 项目目录 | `/home/dabaie/code/psychologist` |
| 虚拟环境 | `/home/dabaie/code/my_project/.venv`（Python 3.10.12） |
| 向量模型 | `/home/dabaie/models/bge-m3`（1024 维） |
| 重排模型 | `/home/dabaie/models/bge-reranker-v2-m3` |
| MySQL | `127.0.0.1:3307`，库 `rag_roleplay`，用户 `dev` |
| Redis | `127.0.0.1:6379/0` |
| Milvus | `127.0.0.1:19530` |
| API | `0.0.0.0:8000` |
| 大模型 | DeepSeek 在线 API（OpenAI 兼容），`deepseek-flash` |

---

## 2. 功能需求

### 2.1 用户管理模块（U）

> 需求要求：必须连接 MySQL。实现位置：`src/api/v1/auth.py`、`src/api/v1/users.py`、`src/services/user_service.py`。

| 编号 | 功能 | 说明 | 实现 |
| :--- | :--- | :--- | :--- |
| U-01 | 用户注册 | 用户名（3-64）、密码（6-128）、邮箱/手机号，密码 bcrypt 加密存储 | `POST /api/v1/auth/register` |
| U-02 | 用户登录 | JWT 鉴权，返回 access + refresh token | `POST /api/v1/auth/login` |
| U-03 | 用户信息查询 | 查询当前用户资料 | `GET /api/v1/users/me` |
| U-04 | 用户信息修改 | 昵称、头像、邮箱、手机号 | `PUT /api/v1/users/me` |
| U-05 | 用户禁用/启用 | 管理员操作 | `POST /api/v1/admin/users/{user_id}/status` |
| U-06 | 角色权限 | 普通用户 `user`、管理员 `admin` | `sys_roles` + `user_sys_roles`；`get_current_admin` 校验 |
| U-07 | 登录日志 | 记录登录时间、IP、User-Agent | `login_logs` 表；`GET /api/v1/admin/logs/login` |
| U-08 | 用户偏好心理医生 | 保存默认心理医生角色 | `user_persona_preferences`；`GET/POST /api/v1/users/me/preferences` |

**补充实现**（超出原始编号但已实现）：

| 功能 | 接口 |
| :--- | :--- |
| 刷新 Token | `POST /api/v1/auth/refresh` |
| 退出登录（写审计） | `POST /api/v1/auth/logout` |
| 修改密码 | `POST /api/v1/users/me/password` |
| 我的登录日志 | `GET /api/v1/users/me/login-logs` |
| 当前用户信息（auth 前缀冗余提供） | `GET/PUT /api/v1/auth/me` |

**核心数据表**：`users`、`sys_roles`、`user_sys_roles`、`counselor_personas`、`user_persona_preferences`、`conversations`、`messages`、`knowledge_docs`、`knowledge_chunks`、`audit_logs`、`login_logs`（共 11 张，见 [`03-database.md`](03-database.md)）。

### 2.2 心理医生角色管理模块（P）

> 实现位置：`src/api/v1/personas.py`、`src/services/persona_service.py`、`src/services/persona_seed.py`。

| 编号 | 功能 | 说明 | 实现 |
| :--- | :--- | :--- | :--- |
| P-01 | 角色列表 | 展示三个心理医生（默认仅上架；`?include_inactive=true` 含下架） | `GET /api/v1/personas` |
| P-02 | 角色详情 | 流派、风格、方法、开场白、知识范围、安全边界、system prompt | `GET /api/v1/personas/{persona_id}` |
| P-03 | 角色切换 | 用户设置默认心理医生 | `POST /api/v1/users/me/preferences` |
| P-04 | 独立提示词 | 每个角色独立 `system_prompt` 字段 | `counselor_personas.system_prompt` |
| P-05 | 独立知识库 | Milvus 以 `persona_id` 为分区键，检索时表达式过滤 | `hybrid_search_knowledge(persona_id, ...)` |
| P-06 | 独立会话 | 每个会话绑定一个角色，短期/长期记忆按角色隔离 | `conversations.persona_id` |
| P-07 | 角色启停 | 管理员上下架 | `POST /api/v1/personas/{persona_id}/status` |
| P-08 | 角色扩展 | 支持新增角色（管理员 API + 种子文件两种方式） | `POST /api/v1/personas`；`KNOWLEDGE_DIRS` |

**角色独立性核对表**（需求要求"三个角色必须有独立：角色编码、名称、头像、流派定位、system prompt、开场白、知识库范围、检索过滤条件、会话记录、安全边界、模型参数"）：

| 独立项 | 实现载体 | 状态 |
| :--- | :--- | :--- |
| 角色编码 | `persona_code`（唯一索引） | ✅ |
| 名称 | `name` | ✅ |
| 头像 | `avatar`（图片 URL；原始需求 DDL 未定义该列，代码已补） | ✅ |
| 流派定位 | `therapy_type` + `title` | ✅ |
| system prompt | `system_prompt`（TEXT） | ✅ |
| 开场白 | `greeting` | ✅ |
| 知识库范围 | `knowledge_scope` + `KNOWLEDGE_DIRS` 目录映射 | ✅ |
| 检索过滤条件 | 检索表达式 `persona_id == {id}` | ✅ |
| 会话记录 | `conversations.persona_id` + 记忆 Key 含 `persona_id` | ✅ |
| 安全边界 | `safety_boundary`（TEXT，三角色当前复用统一文案） | ✅ |
| 模型参数 | `model_params`（JSON：temperature / max_tokens / top_p） | ✅ |

### 2.3 三个心理医生角色定义

#### 角色一：人本共情倾听型 — 林知暖医生

| 项目 | 内容 |
| :--- | :--- |
| 角色编码 | `humanistic_lin` |
| 角色名称 | 林知暖医生 |
| 流派 | 人本主义 / 共情倾听 |
| 风格 | 温暖、耐心、不评判、多倾听 |
| 适用场景 | 情绪低落、孤独、压力倾诉、关系困扰 |
| 核心技巧 | 情绪命名、复述、开放式提问、无条件积极关注 |
| 禁止行为 | 不诊断、不开药、不贴标签、不强行建议 |
| 知识库 | 心理健康科普、情绪管理、人本主义咨询、危机干预转介 |
| 开场白 | 你好，我是林知暖。你可以慢慢说，我会认真听。 |
| 模型参数 | `temperature=0.8, max_tokens=2048, top_p=0.9` |
| 知识库目录 | `心理医生/林知暖医生（人本共情倾听型）`、`心理医生/通用知识库` |

#### 角色二：认知行为治疗型 — 陈认知医生

| 项目 | 内容 |
| :--- | :--- |
| 角色编码 | `cbt_chen` |
| 角色名称 | 陈认知医生 |
| 流派 | CBT 认知行为疗法 |
| 风格 | 结构化、理性、合作式 |
| 适用场景 | 焦虑、拖延、负面自动思维、行为回避 |
| 核心技巧 | 识别自动思维、认知重构、行为激活、家庭作业 |
| 禁止行为 | 不诊断、不开药、不替代医生 |
| 知识库 | CBT 基础、焦虑抑郁心理教育、认知扭曲、行为激活 |
| 开场白 | 你好，我是陈认知。我们可以一起看看，最近是什么想法在影响你。 |
| 模型参数 | `temperature=0.5, max_tokens=2048, top_p=0.9` |
| 知识库目录 | `心理医生/陈认知医生（CBT 认知行为治疗型）`、`心理医生/通用知识库` |

#### 角色三：正念情绪调节型 — 周正念医生

| 项目 | 内容 |
| :--- | :--- |
| 角色编码 | `mindfulness_zhou` |
| 角色名称 | 周正念医生 |
| 流派 | 正念减压 / 情绪接纳 |
| 风格 | 平静、缓慢、引导式 |
| 适用场景 | 失眠、紧张、躯体化压力、情绪波动 |
| 核心技巧 | 正念呼吸、身体扫描、情绪接纳、放松训练 |
| 禁止行为 | 不诊断、不开药、不替代医疗 |
| 知识库 | 正念减压、睡眠卫生、压力管理、情绪调节 |
| 开场白 | 你好，我是周正念。我们先做三次深呼吸，好吗？ |
| 模型参数 | `temperature=0.6, max_tokens=2048, top_p=0.9` |
| 知识库目录 | `心理医生/周正念医生（正念情绪调节型）——3 本`、`心理医生/通用知识库` |

完整 system prompt 见 [`05-prompts.md`](05-prompts.md)。

### 2.4 会话与多轮对话（C）

> 实现位置：`src/api/v1/conversations.py`、`src/api/v1/chat.py`、`src/services/conversation_service.py`、`src/services/rag_service.py`。

| 编号 | 功能 | 说明 | 实现 |
| :--- | :--- | :--- | :--- |
| C-01 | 新建会话 | 选择心理医生角色，返回开场白 `greeting` | `POST /api/v1/conversations` |
| C-02 | 多轮对话 | 注入短期记忆（最近 N 轮）到提示词 | `build_recent_messages_block` |
| C-03 | 短期记忆 | Redis List 保存最近 10 轮，TTL 86400；丢失时从 MySQL 回填 | `memory_service.get_short_term` |
| C-04 | 长期记忆 | Milvus 保存重要摘要（每 20 条消息触发或手动保存） | `maybe_save_long_term` |
| C-05 | 历史记录 | MySQL `messages` 表，分页查询 | `GET /api/v1/conversations/{id}/messages` |
| C-06 | 流式输出 | SSE（`text/event-stream`），事件类型 meta/delta/done/error | `POST /api/v1/chat/stream` |
| C-07 | 会话删除 | 逻辑删除（`status=0`）并清理 Redis 短期记忆 | `DELETE /api/v1/conversations/{id}` |
| C-08 | 多用户隔离 | 所有会话查询强制校验 `user_id`，越权抛 403 | `get_conversation(db, id, user_id)` |

**补充实现**：会话详情 `GET /api/v1/conversations/{id}`、手动保存长期记忆 `POST /api/v1/conversations/{id}/save-memory`。

### 2.5 RAG 离线知识库（K）

> 实现位置：`src/rag/parser.py`、`src/rag/chunker.py`、`src/services/knowledge_service.py`、`scripts/ingest_knowledge.py`。

| 编号 | 功能 | 说明 | 实现 |
| :--- | :--- | :--- | :--- |
| K-01 | 文档上传 | PDF / TXT / MD / DOCX | `POST /api/v1/knowledge/upload`（管理员，multipart） |
| K-02 | PDF 解析 | PyMuPDF 优先，文本过少回退 pdfplumber | `_parse_pdf` |
| K-03 | OCR | PaddleOCR（可选，未内置；扫描件跳过并告警） | 见 FAQ |
| K-04 | 多模态解析 | MinerU（可选，未内置） | 同上 |
| K-05 | 表格解析 | DOCX 表格按 `｜` 拼行；PDF 由 pdfplumber 兜底 | `_parse_docx` |
| K-06 | 去水印 | 正则去页面/页码/公众号/版权等水印行；剔除重复页眉页脚 | `clean_text` / `remove_repeated_lines` |
| K-07 | 分块 | fixed / sentence / paragraph / heading / semantic / parent_child | `chunk_text` |
| K-08 | 向量化 | BGE-M3，1024 维，归一化 | `Embedder.encode` |
| K-09 | 入库 | Milvus `persona_knowledge`（稠密 + BM25 稀疏） | `insert_chunks` |
| K-10 | 元数据 | MySQL 记录来源、标题、类型、状态、块数、摘要、milvus_id | `knowledge_docs` / `knowledge_chunks` |
| K-11 | 动态更新 | 单文档删除、按角色重建索引、目录批量导入 | `delete_doc` / `rebuild_persona_index` |
| K-12 | 去重与低质量删除 | 段落去重、块级去重、`< 20 token` 低质块丢弃 | `deduplicate_paragraphs` / `_filter_chunks` |

### 2.6 RAG 在线检索（R）

> 实现位置：`src/rag/retriever.py`、`src/rag/reranker.py`、`src/rag/prompt.py`、`src/services/rag_service.py`、`src/services/llm_service.py`、`src/services/crisis_service.py`。

| 编号 | 功能 | 说明 | 实现 |
| :--- | :--- | :--- | :--- |
| R-01 | Query 改写 | LLM 改写/扩写（temperature 0.2，max_tokens 128），与原文合并检索；`QUERY_REWRITE_ENABLED` 控制 | `rewrite_query` |
| R-02 | 问题向量化 | BGE-M3 `encode_query` | `Embedder.encode_query` |
| R-03 | 混合检索 | 稠密 HNSW/COSINE + BM25 稀疏，`RRFRanker(60)` 融合；不可用时降级纯稠密 | `hybrid_search_knowledge` |
| R-04 | 多路召回 | Milvus 知识库（top_k=20）+ Milvus 长期记忆（top_k=3）+ Redis 短期记忆 + MySQL 历史回填 | `rag_service.prepare` |
| R-05 | 重排序 | BGE-Reranker-v2-M3 CrossEncoder，分数 sigmoid 归一化，取 top_n=5 | `Reranker.rerank` |
| R-06 | 相似度过滤 | 阈值 0.35；全部低于阈值时保留最高分 1 条（>0） | `retrieve` |
| R-07 | 提示词模板 | 角色 prompt + 知识片段 + 长期记忆 + 短期记忆 + 用户输入；危机注入 | `build_chat_prompt` |
| R-08 | 大模型调用 | DeepSeek OpenAI 兼容接口（支持流式/非流式） | `llm_service.chat` / `chat_stream` |
| R-09 | 后处理 | 正则清理、敏感词脱敏、"诊断/开药"表述纠正、危机转介兜底 | `crisis_service.post_process` |
| R-10 | 返回用户 | JSON（`/chat`）或 SSE 流（`/chat/stream`） | `rag_service.answer` / `answer_stream` |

### 2.7 记忆管理

| 类型 | 存储 | 说明 | Key / 载体 |
| :--- | :--- | :--- | :--- |
| 短期记忆 | Redis | 最近聊天记录，List，TTL 86400，LTRIM 保留最近 2N 条 | `session:{user_id}:{persona_id}:{conversation_id}:messages` |
| 长期记忆 | Milvus | 重要对话摘要（每 20 条消息或手动触发），按问题相似度召回 top 3 | `user_long_term_memory` |
| 业务数据 | MySQL | 用户、角色、会话、消息、知识元数据、日志 | 11 张表 |
| 缓存 | Redis | 用户资料、角色信息、会话摘要、限流、并发锁 | `user:*` / `persona:*` / `memory_summary:*` / `rate_limit:*` / `lock:*` |

### 2.8 知识库动态更新

管理员上传新文档 → 系统自动解析、清洗、分块、向量化 → 写入 Milvus → MySQL 记录状态（`pending`/`processing`/`success`/`failed`）。支持重新索引（可按角色 `drop_existing`）、删除文档及对应向量、按角色隔离知识库。

### 2.9 管理后台 / API

| 能力 | 接口 |
| :--- | :--- |
| 用户管理（列表/搜索/启停） | `GET /api/v1/admin/users`、`POST /api/v1/admin/users/{id}/status` |
| 角色管理 | `POST/PUT /api/v1/personas*`、`POST /api/v1/personas/{id}/status` |
| 知识库管理 | `/api/v1/knowledge/*`（上传/列表/删除/重建/检索/统计） |
| 会话审计 | `GET /api/v1/admin/conversations` |
| 日志查看 | `GET /api/v1/admin/logs/login`、`GET /api/v1/admin/logs/audit` |
| 评测报告 | `POST /api/v1/eval/ragas`，报告落盘 `data/eval/reports/` |
| 系统监控 | `GET /api/v1/admin/monitor`（MySQL/Redis/Milvus 健康、模型加载状态、向量数） |

### 2.10 安全与危机干预

- **危机词检测**：`src/services/crisis_service.py` 内置 30+ 高危词（自杀、自残、不想活、结束生命、伤害自己、杀人、跳楼、割腕…），另加 2 条风险语气正则（"活着好累/撑不下去"等）。
- **转介提示**：命中后强制追加提示，建议联系 **心理援助热线 12356**、**急救 120**、**报警 110**，或前往当地精神卫生中心急诊；并在 system prompt 注入"危机干预优先指令"。
- **兜底**：`ensure_crisis_notice` 检查回答中是否含热线/120/110，缺失则追加（流式模式下若在已生成内容中缺失，会补发 delta 事件）。
- **边界**：不诊断、不开药、不替代线下就医；`post_process` 会把"我作为心理咨询师可以为你诊断/开药"类表述替换为"我不能进行诊断或开药"。
- **数据**：口令 bcrypt、审计日志、登录日志、敏感词脱敏。详见 [`08-security.md`](08-security.md)。

---

## 3. 非功能需求

### 3.1 性能

| 指标 | 需求要求 | 实现与实测说明 |
| :--- | :--- | :--- |
| 检索延迟 | < 500ms | Milvus 混合检索日志打印实际耗时（`Milvus 混合检索 ... 耗时 Xms`） |
| 重排序延迟 | < 800ms | 重排日志打印耗时；批量 8，CPU 降级时显著变慢 |
| 首 token 延迟 | < 3s | 流式 `meta` 事件先于 LLM 返回；启动预热模型可降低首问延迟（`WARMUP_MODELS=1`） |
| 并发 | 支持 50+ QPS | 单 worker 默认；可通过 `API_WORKERS` 扩展 |
| 可用性 | 99% | 外部依赖失败均降级（Redis/Milvus/LLM 辅助调用），主流程不中断 |

### 3.2 安全与合规

密码 bcrypt（rounds 12，代码限制 4~16）；JWT 鉴权（HS256）；SQL 注入防护（SQLAlchemy ORM 参数绑定）；XSS 防护（纯 JSON 输出，不渲染 HTML）；访问审计（`audit_logs`/`login_logs`）；限流（Redis 计数，默认 30 次/分钟/用户）；心理危机干预转介；不提供诊断、处方、药物建议。详见 [`08-security.md`](08-security.md)。

### 3.3 日志

Python `logging`，控制台 + 滚动文件 `app.log` / `error.log` / `llm.log` / `rag.log`；级别 `DEBUG` / `INFO` / `WARNING` / `ERROR`（由 `LOG_LEVEL` 控制）。

### 3.4 可维护性

配置化（模型路径、数据库、Redis、Milvus、RAG、LLM、鉴权全部走 `.env`）；模块化分层（api / services / rag / db / models / schemas / core）；支持 Alembic 迁移；提供一键脚本 `install.sh` / `run.sh` / `shutdown.sh`；`CLAUDE.md` 约定扩展方式。

---

## 4. 数据设计（摘要）

MySQL 11 张表、Milvus 2 个 Collection、Redis 6 类 Key 的逐字段设计见 [`03-database.md`](03-database.md)。

- MySQL：`users`、`sys_roles`、`user_sys_roles`、`counselor_personas`、`user_persona_preferences`、`conversations`、`messages`、`knowledge_docs`、`knowledge_chunks`、`audit_logs`、`login_logs`
- Milvus：`persona_knowledge`（知识分块，分区键 `persona_id`）、`user_long_term_memory`（长期记忆摘要，分区键 `persona_id`，按 `user_id` 过滤）
- Redis：`session:*`、`user:*:profile`、`persona:*`、`rate_limit:*`、`lock:conversation:*`、`memory_summary:*`

---

## 5. RAG 参数基线

| 参数 | 建议值（需求） | 实际默认值 | 配置项 |
| :--- | :--- | :--- | :--- |
| 分块大小 | 512 tokens | 512 | `CHUNK_SIZE` |
| 分块重叠 | 80 tokens | 80 | `CHUNK_OVERLAP` |
| 父块大小 | 1024 | 1024 | `PARENT_CHUNK_SIZE` |
| 子块大小 | 256 | 256 | `CHILD_CHUNK_SIZE` |
| 召回 top_k | 20 | 20 | `RETRIEVE_TOP_K` |
| 重排后 top_n | 5 | 5 | `RERANK_TOP_N` |
| 相似度阈值 | 0.35 | 0.35 | `SIMILARITY_THRESHOLD` |
| 短期记忆轮数 | 10 轮 | 10 | `SHORT_TERM_MAX_TURNS` |
| 短期记忆 TTL | 86400 秒 | 86400 | `SHORT_TERM_TTL` |
| 长期记忆触发 | 超 20 轮或手动 | 每 20 条消息或手动 | `LONG_TERM_SUMMARY_TRIGGER` |
| BGE-M3 维度 | 1024 | 1024 | `EMBEDDING_DIM` |
| 重排模型 | BGE-Reranker-v2-M3 | 同 | `RERANKER_MODEL_PATH` |

---

## 6. 测试与验收

### 6.1 测试类型

| 类型 | 工具 | 状态 |
| :--- | :--- | :--- |
| 单元测试 | pytest | 环境已装 pytest 9.1.1；仓库暂无 `tests/`（用例清单见 `07-testing.md`） |
| 接口测试 | Postman / Apifox / curl | 手工用例清单见 `07-testing.md` |
| 集成测试 | pytest + 真实三库 | 同上 |
| 压力测试 | JMeter（Windows：`D:\JMeter\apache-jmeter-5.6.3\apache-jmeter-5.6.3\bin`） | 方案见 `07-testing.md` |
| RAG 评测 | RAGAS（当前走内置 LLM-as-Judge） | `POST /api/v1/eval/ragas` |

### 6.2 RAG 评测指标

Faithfulness（忠实度）、Answer Relevancy（答案相关性）、Context Precision（上下文精确度）、Context Recall（上下文召回率）。

### 6.3 验收标准

| # | 验收项 | 验收方法 | 状态 |
| :--- | :--- | :--- | :--- |
| 1 | 三个心理医生角色全部实现，可切换 | `GET /api/v1/personas` 返回 3 条；`POST /users/me/preferences` 切换成功 | ✅ 代码就绪 |
| 2 | 每个角色有独立提示词、知识库、会话 | 角色详情 `system_prompt` 不同；检索按 `persona_id` 过滤；会话绑定 `persona_id` | ✅ 代码就绪 |
| 3 | 用户注册、登录、信息管理连接 MySQL | `users` 表可查；`/auth/register` `/auth/login` `/users/me` | ✅ 代码就绪 |
| 4 | Redis 实现短期记忆 | `session:*` Key 存在且 TTL 86400；`/admin/monitor` redis.healthy=true | ✅ 代码就绪 |
| 5 | Milvus 实现知识库检索 | `/knowledge/stats`、`/knowledge/search` 返回命中 | ✅ 代码就绪 |
| 6 | BGE-M3 向量化正常，维度 1024 | `/admin/monitor` models.embedding.dim=1024、loaded=true | ✅ 代码就绪 |
| 7 | BGE-Reranker-v2-M3 重排序正常 | `/admin/monitor` models.reranker.loaded=true；检索日志有重排耗时 | ✅ 代码就绪 |
| 8 | 支持多轮对话 | 连续两轮 `/chat`，回答体现上下文 | ✅ 代码就绪 |
| 9 | 支持知识库动态更新 | upload / delete / rebuild 接口 | ✅ 代码就绪 |
| 10 | 支持 RAGAS 评测 | `POST /api/v1/eval/ragas` 产出报告 JSON | ✅ 代码就绪（走内置 Judge） |
| 11 | 可在 Windows 11 + WSL2 + Ubuntu 部署 | `install.sh` → `init_db.py` → `run.sh` → `/health` | ✅ 脚本就绪 |
| 12 | 提供接口文档、设计文档、部署文档、测试报告 | `docs/01`~`08` + 本文件 + `07-testing.md` 报告模板 | ✅ |

> "代码就绪"表示实现已完成、接口与脚本齐备；具体验收结论需按 [`07-testing.md`](07-testing.md) 的测试报告模板逐项实际执行后填写。

---

## 7. 风险与合规

| 风险 | 措施 | 实现 |
| :--- | :--- | :--- |
| 心理危机 | 危机词检测、转介热线 | `crisis_service.detect_crisis` + 强制追加提示 + system prompt 注入 |
| 医疗合规 | 不诊断、不开药、不替代医生 | 三角色 system prompt 安全边界 + `post_process` 正则纠正 |
| 数据隐私 | 加密、脱敏、审计 | bcrypt、JWT、敏感词脱敏、`audit_logs`/`login_logs` |
| 模型幻觉 | RAG、重排、引用来源 | 混合检索 + 精排 + 阈值 + `references` 溯源 + 无知识时明确提示 |
| 性能不足 | 缓存、流式、降级、负载均衡 | Redis 缓存、SSE、CPU 降级、`API_WORKERS` |
| 知识库低质 | 去重、清洗、评测 | 段落/块去重、低质块过滤、水印清洗、RAGAS 评测 |

---

## 8. 与原始需求文档的差异说明（以真实代码为准）

| # | 原始需求描述 | 真实实现 | 说明 |
| :--- | :--- | :--- | :--- |
| 1 | `counselor_personas` DDL 无 `avatar`、`safety_boundary` 列 | 代码已增加 `avatar VARCHAR(512)`、`safety_boundary TEXT` | 为满足"独立头像/安全边界"要求 |
| 2 | `conversations` DDL 无 `message_count` | 代码增加 `message_count INT DEFAULT 0` | 用于长期记忆触发判断 |
| 3 | OCR / MinerU 必须实现 | 未内置，扫描件仅告警跳过 | 依赖体积大；见 README FAQ Q1 |
| 4 | RAGAS 官方实现 | 双引擎：优先 ragas，失败回退内置 LLM-as-Judge | `ragas 0.4.3` 与 `langchain-community 0.4.2` 冲突，实测无法 import |
| 5 | `MILVUS_COLLECTION` 单集合 | 实际两个集合：`persona_knowledge` + `user_long_term_memory` | 长期记忆与知识库分离 |
| 6 | 稀疏向量度量 `IP` | 实际 `SPARSE_INVERTED_INDEX` + `metric_type="BM25"` | pymilvus 2.5+ 的 BM25 Function 用法 |
| 7 | Redis `user:{id}:profile`、`persona:{id}` 标注为 Hash | 实际为 String（JSON） | 见 `03-database.md` 说明 |
| 8 | 会话标题可由模型生成 | `TITLE_PROMPT` 已定义但未使用，实际按首条消息截断 20 字 | 零延迟零成本 |
| 9 | `scripts/install.sh` 需安装 MySQL/Redis/Milvus | 脚本只做**连通性检查**，不自动安装服务 | 三库需用户预先部署 |

---

## 9. 交付物清单

| 交付物 | 位置 | 状态 |
| :--- | :--- | :--- |
| 需求规格说明书 | 本文件 | ✅ |
| 技术设计文档 | [`02-architecture.md`](02-architecture.md) | ✅ |
| 数据库 DDL | [`03-database.md`](03-database.md) + `sql/schema.sql` | ✅ |
| 接口文档 | [`04-api.md`](04-api.md) + 运行时 `/docs` | ✅ |
| 三个心理医生提示词模板 | [`05-prompts.md`](05-prompts.md) + `src/rag/prompt.py` | ✅ |
| 离线知识库构建脚本 | `scripts/ingest_knowledge.py` | ✅ |
| 在线 RAG 服务代码 | `src/services/rag_service.py`、`src/rag/*` | ✅ |
| 用户管理 MySQL 代码 | `src/services/user_service.py` | ✅ |
| Redis / Milvus 集成代码 | `src/db/redis.py`、`src/db/milvus.py` | ✅ |
| 测试报告 | [`07-testing.md`](07-testing.md) 报告模板 | ✅（模板） |
| RAGAS 评测报告 | 运行时生成于 `data/eval/reports/` | ✅（脚本就绪） |
| README.md | [`../README.md`](../README.md) | ✅ |
| install.sh / run.sh / shutdown.sh | `scripts/` | ✅ |