# 技术文档 —— 基于 PDF 文档的 RAG 问答系统（工单1）

> 工单编号：**人工智能NLP-RAG-基于PDF 文档的问答系统**
> 语料：《招股说明书1.pdf》（548 页）
> 本文档面向前后端开发、评审人与二次开发者，描述系统架构、技术选型、模块职责、数据流、检索优化策略、数据库表结构、配置体系与开发流程。

---

## 1. 文档目的与范围

本文档回答四个问题：

1. **系统长什么样**——分层架构与组件职责（第 3、4 节）；
2. **数据怎么流动**——离线索引链路与在线问答链路（第 5 节）；
3. **为什么这样选**——技术选型理由与招股书特攻优化策略（第 3.4、6 节）；
4. **怎么继续开发**——数据库表结构、配置项、日志契约、阶段计划（第 7~13 节）。

**关于实现状态的约定**：项目正在开发中。第 4 节每个模块都标注了状态（✅ 已存在 / 🚧 待创建），标注依据是 2026-10-01 对代码仓的实际核对。对 🚧 模块，文中给出的是**接口契约**（由 `app/models/schemas.py` 中的数据契约与需求说明书推导），最终以实现代码为准。

---

## 2. 设计目标与约束

| 编号 | 目标/约束 | 设计响应 |
| --- | --- | --- |
| G1 | 对《招股说明书1.pdf》检索**又快又准** | 混合检索 + 领域词加权 + 表格单独索引 + 页码过滤（第 6 节） |
| G2 | 答案**必须基于 PDF**，带引用 | 检索片段注入 Prompt，引用结构 `Citation`（页码/chunk_id/摘要），前端可展开原文 |
| G3 | 不知道就回**“不清楚”** | 三处兜底：检索分数阈值、片段为空、模型输出无依据（第 5.3 节） |
| G4 | **首字返回 < 3 秒** | LLM 流式输出 + 本地 vLLM/SGLang + 冗余召回可裁剪（第 15 节） |
| G5 | 支持**多轮对话** | `conversations`/`messages` 表 + 问题改写 + 最近 5 轮历史 |
| G6 | 支持**中等并发** | vLLM 连续批处理 + SQLite WAL + 每请求独立连接 + 进程级单例 |
| G7 | **禁止静默失败**，函数级日志 | `@trace` 装饰器 + 三个日志文件（第 9 节） |
| G8 | 单篇文档特攻，但要**可扩展**到多文档 | `documents` 表 + `doc_id` 贯穿全链路，知识库管理接口预留 |
| G9 | 无 GPU/无模型时**链路仍可跑通** | 嵌入降级 hash 向量、向量库降级 numpy 内存索引、生成降级抽取式回答 |

---

## 3. 总体架构

### 3.1 分层架构

```text
┌──────────────────────────────────────────────────────────────────────┐
│  表现层        app/ui/streamlit_app.py        （Streamlit 网页界面）  │
│                文件选择/上传 · 提问框 · 答案流式展示 · 引用展开       │
│                多轮历史 · 点赞点踩 · 清空对话 · 首字响应时间          │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ 统一调用
┌───────────────────────────────▼──────────────────────────────────────┐
│  编排层        app/core/qa_engine.py          （问答引擎总装）        │
│                ask(question, conversation_id, doc_id, page_filter)    │
│                → QueryAnalysis → RetrievedChunk[] → Answer(流式)      │
└───┬───────────────┬───────────────┬───────────────┬──────────────────┘
    │               │               │               │
┌───▼─────┐  ┌──────▼──────┐  ┌─────▼──────┐  ┌─────▼──────┐
│ Query   │  │  检索层      │  │  生成层     │  │  引用层     │
│ 理解    │  │             │  │            │  │            │
│ query_  │  │ retriever   │  │ generator  │  │ citation   │
│ under-  │  │  ├ vector_  │  │ （OpenAI    │  │ （页码/摘要 │
│ standing│  │  │  store   │  │  兼容流式） │  │  /chunk_id）│
│         │  │  ├ bm25_    │  │            │  │            │
│         │  │  │  index   │  │            │  │            │
│         │  │  └ reranker │  │            │  │            │
└─────────┘  └──────┬──────┘  └────────────┘  └────────────┘
                    │
┌───────────────────▼──────────────────────────────────────────────────┐
│  索引层   pdf_parser → chunker → embedder → vector_store / bm25_index │
│           app/core/*.py    产物落 data/processed/ 与 data/index/      │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  存储层   app/storage/sqlite_manager.py   （SQLite：8 张表，WAL 模式） │
│           对话/消息/反馈/评估/分块/文档元数据/日志索引/标准答案        │
└──────────────────────────────────────────────────────────────────────┘
                    ▲
┌───────────────────┴──────────────────────────────────────────────────┐
│  横切层   config.py（配置） · logging_conf.py（日志+@trace）           │
│           models/schemas.py（数据契约） · prompts/*.txt（提示词）      │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 组件清单

| 层 | 组件 | 形态 | 运行位置 |
| --- | --- | --- | --- |
| 表现层 | Streamlit 应用 | Python 进程 | 本机 / 部署机（默认 8501 端口） |
| 编排层 | `qa_engine` | 库内模块 | 与 Streamlit 同进程 |
| 检索层 | `retriever` + `vector_store` + `bm25_index` | 库内模块 | 与 Streamlit 同进程 |
| 生成层 | `generator` → LLM 服务 | HTTP（OpenAI 兼容） | vLLM/SGLang 独立进程（默认 8000 端口） |
| 索引层 | `pdf_parser`/`chunker`/`embedder` + `研发/scripts/build_index.py` | 离线批处理 | 开发机或部署机 |
| 存储层 | SQLite（`data/index/rag.sqlite3`） | 单文件数据库 | 与 Streamlit 同机 |
| 向量库 | Chroma（`data/index/`） | 嵌入式（本地持久化） | 与 Streamlit 同机 |
| 评估 | `evaluator` + `研发/scripts/evaluate.py` + RAGAS | 离线批处理 | 开发机或部署机 |

### 3.3 部署拓扑（算力云 4090）

```text
算力云 4090 机器（Linux, 24GB 显存）
├─ 进程 1：vLLM OpenAI 兼容服务   :8000   ← Qwen2.5-7B-Instruct-AWQ（独立 conda 环境）
├─ 进程 2：Streamlit 应用         :8501   ← conda 环境 gao6gongdan
│            └─ 同进程内：检索 + SQLite + Chroma + 嵌入模型
└─ 磁盘：data/raw · data/processed · data/index · data/eval · logs
```

开发机（Windows）与部署机通过同一份代码运行，差异只由**环境变量**承载（如 `RAG_LLM__BASE_URL`、`RAG_EMBEDDING__DEVICE`）。

### 3.4 技术选型与理由

| 组件 | 选型 | 为什么选它 | 备选 |
| --- | --- | --- | --- |
| 网页框架 | **Streamlit** | 纯 Python 即可产出交互界面，内置流式输出（`st.write_stream`）、会话状态（`st.session_state`）、文件上传与折叠面板，契合“不需要移动端适配”的范围 | Gradio、FastAPI+前端 |
| LLM 服务 | **vLLM** | 连续批处理（continuous batching）+ PagedAttention，是“首字 < 3 秒 + 中等并发”的关键；提供 OpenAI 兼容接口，业务代码不绑定推理框架 | SGLang（同为 OpenAI 兼容，`研发/scripts/run_sglang.sh`） |
| 生成模型 | **Qwen2.5-7B-Instruct-AWQ** | 4 bit 量化后显存占用低，4090 单卡可留足 KV Cache；中文财报/招股书语料表现稳定 | GPTQ-Int4、ChatGLM3-6B、Yi-6B |
| 嵌入模型 | **BAAI/bge-small-zh-v1.5** | 约 95 MB，首次部署下载快、CPU 也能跑；中文检索效果好；后续可用环境变量无痛换大模型 | bge-large-zh-v1.5、bge-m3 |
| 重排模型 | **BAAI/bge-reranker-base** | 对“召回多、精度不够”的场景提升明显；作为**可选项默认关闭**，把延迟预算留给首字 | 不启用 |
| 向量库 | **Chroma** | 嵌入式、本地持久化、零运维，适合单机单文档场景；接口简单 | FAISS（`部署/环境配置/requirements.txt` 中已注释预留） |
| 关键词检索 | **rank-bm25 + jieba** | 招股书里“注册资本”“法定代表人”等**专有名词**命中，向量检索容易漏，BM25 正好互补 | Elasticsearch（重，超范围） |
| 元数据/对话存储 | **SQLite（WAL）** | 单文件、零运维；`journal_mode=WAL` 支持读写并发；分块、对话、反馈、评估统一落库便于审计 | PostgreSQL |
| 数据校验 | **Pydantic v2** | 全链路数据契约（`schemas.py`），模型即文档；序列化/反序列化与 JSON 字段存储天然契合 | dataclass |
| 配置 | **Pydantic 模型 + `RAG_` 环境变量覆盖** | 部署时改环境变量即可，不动代码；配置项集中在 `config.py`，是唯一的“魔法数字”来源 | YAML 文件 |
| 日志 | **loguru** | 一行接入 JSON 序列化（`serialize=True`）、轮转与保留策略；`@trace` 装饰器实现函数级追踪 | 标准库 logging |
| 评估 | **RAGAS** | 工单点名的评估体系；提供 faithfulness / answer_relevancy / context_precision / context_recall 四项 | 自研指标（作为补充） |
| RAG 框架 | **LangChain（按需使用）** | 仅在文本切分、文档加载等环节复用成熟实现；**核心链路保持自研**，避免框架黑盒掩盖问题、便于逐函数打日志 | LlamaIndex |

> 选型原则：**关键路径自研、外围能力借力**。检索融合、兜底判定、引用生成这三处决定验收结果，必须自己掌控并可逐函数打日志。

---

## 4. 模块说明

### 4.1 模块清单与实现状态（截至 2026-10-01 19:20 核对）

| 模块 | 职责 | 状态 |
| --- | --- | --- |
| `app/core/config.py` | 全局配置 + 环境变量覆盖 + 目录创建 | ✅ 已存在 |
| `app/core/logging_conf.py` | loguru 日志 + `@trace` 函数级追踪 | ✅ 已存在 |
| `app/core/text_utils.py` | 通用工具：文本清洗/分词/停用词/稳定 ID/数值格式化 | ✅ 已存在 |
| `app/core/bm25_index.py` | BM25 关键词索引（含纯 Python 兜底实现） | ✅ 已存在 |
| `app/models/schemas.py` | Pydantic 数据契约 | ✅ 已存在 |
| `app/storage/sqlite_manager.py` | SQLite 8 张表的建表与增删改查 | ✅ 已存在 |
| `app/ui/__init__.py` | 界面包标识 | ✅ 已存在 |
| `app/ui/streamlit_app.py` | Streamlit 网页界面（约 53 KB，含文档选择/上传、指标、引用、反馈、会话管理、demo 问题） | ✅ 已存在（**依赖 `qa_engine.py`，后者未落地时界面提示“问答引擎尚未就绪”**） |
| `app/main.py` | 启动入口 | 🚧 待创建 |
| `app/core/pdf_parser.py` | PDF 解析 | 🚧 待创建 |
| `app/core/chunker.py` | 分块 | 🚧 待创建 |
| `app/core/embedder.py` | 向量化 | 🚧 待创建 |
| `app/core/vector_store.py` | 向量库管理 | 🚧 待创建 |
| `app/core/retriever.py` | 混合检索 | 🚧 待创建 |
| `app/core/query_understanding.py` | Query 理解 | 🚧 待创建 |
| `app/core/generator.py` | LLM 生成 | 🚧 待创建 |
| `app/core/citation.py` | 引用处理 | 🚧 待创建 |
| `app/core/conversation.py` | 多轮对话管理 | 🚧 待创建 |
| `app/core/evaluator.py` | 评估 | 🚧 待创建 |
| `app/core/qa_engine.py` | 问答引擎总装 | 🚧 **待创建（当前最关键的缺口：界面已就绪并等待它）** |
| `app/prompts/qa_prompt.txt` | 问答提示词 | 🚧 待创建 |
| `app/prompts/query_rewrite_prompt.txt` | 多轮改写提示词 | 🚧 待创建 |
| `研发/scripts/build_index.py` / `evaluate.py` / `run_*.sh` | 索引、评估、启动脚本 | 🚧 待创建 |
| `测试/tests/offline` / `online` / `user` | 三层测试 | 🚧 待创建 |
| `logs/app.log` / `logs/rag_trace.jsonl` | 运行日志与函数级追踪 | ✅ 已产生（`logs/error.log` 需触发异常后生成） |
| `部署/环境配置/environment.yml` | conda 环境定义 | 🚧 待创建 |

### 4.2 已实现模块详解

#### 4.2.1 `app/core/config.py` —— 唯一配置来源

- 用 Pydantic 模型分层描述配置：`AppSettings` / `PathSettings` / `PDFSettings` / `ChunkSettings` / `EmbeddingSettings` / `RetrievalSettings` / `LLMSettings` / `ConversationSettings`，聚合为 `Settings`。
- `PROJECT_ROOT` 由 `Path(__file__).resolve().parents[2]` 推出，**所有路径以项目根为基准**，避免工作目录漂移。
- `get_settings()` 带 `lru_cache`，进程内单例；`reload_settings()` 清缓存后重载（测试与运行时切换用）；模块级 `settings` 为便捷别名。
- `ensure_directories()` 幂等创建 `data/raw`、`data/processed`、`data/index`、`data/eval`、`优化/评估结果/eval_results`、`logs`。
- **环境变量覆盖规则**：前缀 `RAG_`，`RAG_<段名>__<字段名>`，**双下划线**分隔，段名/字段名大小写不敏感；只支持两级。值按当前默认值的类型做转换（`bool` 识别 `1/true/yes/on`；`int`、`float` 直接转型；其余按字符串），转换失败或字段不存在时**静默忽略该变量**（不抛异常，保证服务可启动）。
- 覆盖在 `get_settings()` 内、`Settings()` 构造之后执行，因此**环境变量优先于默认值**。

```python
# 使用示例
from app.core.config import get_settings
s = get_settings()
print(s.retrieval.vector_top_k, s.llm.base_url, s.app.unknown_answer)
```

#### 4.2.2 `app/core/logging_conf.py` —— 日志与函数级追踪

三个输出目标（工单 5.9 要求）：

| 目标 | 级别 | 关键参数 | 用途 |
| --- | --- | --- | --- |
| 控制台 stderr | `RAG_APP__LOG_LEVEL`（默认 INFO） | 带颜色 | 开发实时观察 |
| `logs/app.log` | DEBUG | `serialize=True`（JSON 行）、50 MB 轮转、保留 10 份 | 全量日志 |
| `logs/error.log` | ERROR | `serialize=True`、`backtrace=True`、20 MB 轮转、保留 10 份 | 异常 + 堆栈 |
| `logs/rag_trace.jsonl` | — | 追加写入、自动 flush | 函数级输入/输出/耗时 |

- 对上层暴露统一门面 `logger`：`logger.info(module, message, **extra)`、`warning`、`error`、`exception`（自动附加 `traceback.format_exc()`）。
- **依赖降级**：环境里没有 loguru 时自动切到标准库 `logging`，**对外接口保持不变**——“禁止静默失败”这一硬性要求在缺依赖时依然成立（`HAS_LOGURU` 标记当前实现）。
- `@trace` 装饰器：进入时写 `event="enter"`（含 `args`/`kwargs`），正常返回写 `event="exit"`（含 `elapsed_ms`/`result`），异常时写 `event="error"`（含 `error_type`/`error`/`traceback`）**并把异常重新抛出**。`self`/`cls` 会被剔除，避免把整个对象写进日志。
- 日志值截断：`MAX_LOG_VALUE_CHARS = 800`，嵌套深度 > 3 折叠为 `"..."`，列表只留前 5 项、字典只留前 20 键——**防止把整页 PDF 文本写进日志**。
- `setup_logging()` 显式初始化；`log_stage(stage, message, **extra)` 记录阶段里程碑。

```python
from app.core.logging_conf import logger, trace, log_stage

@trace
def my_step(text: str) -> int:
    return len(text)

log_stage("阶段1", "索引构建开始", pdf="data/raw/招股说明书1.pdf")
```

#### 4.2.3 `app/models/schemas.py` —— 全系统数据契约

按层组织的 Pydantic 模型：

| 层 | 模型 | 关键字段 |
| --- | --- | --- |
| 解析 | `ParsedPage` | `page`（从 1 开始）、`text`、`tables`、`char_count` |
| 解析 | `ParsedTable` | `table_id`（形如 `p152_t1`）、`page`、`section`、`markdown`、`rows` |
| 解析 | `ParsedDocument` | `doc_id`、`source_path`、`title`、`page_count`、`pages`、`tables`、`parser`、`parsed_at` |
| 索引 | `Chunk` | `chunk_id`（形如 `c000123`）、`doc_id`、`page`、`section`、`type`(`text`/`table`)、`content`、`char_count`、`table_id`、`keywords` |
| 检索 | `RetrievedChunk` | `chunk`、`score`、`vector_score`、`bm25_score`、`rerank_score`、`source`(`vector`/`bm25`/`hybrid`/`table`) |
| 检索 | `QueryAnalysis` | `original`、`rewritten`、`intent`、`keywords`、`sub_questions`、`ambiguities`、`page_filter`、`is_followup` |
| 生成 | `Citation` | `page`、`chunk_id`、`snippet`、`section`、`score`；`label()` 输出 `[页码: N]` |
| 生成 | `Answer` | `answer`、`citations`、`is_unknown`、`unknown_reason`、`first_token_ms`、`total_ms`、`retrieved_count`、`pages`、`mode`(`llm`/`extractive`/`fallback`)、`query_analysis`、`retrieved` |
| 对话 | `Message` / `Conversation` | `conversation_id`、`role`(`user`/`assistant`/`system`)、`content`、`citations`、`first_token_ms`、`created_at` |
| 反馈 | `Feedback` | `conversation_id`、`message_id`、`rating`(`up`/`down`)、`comment`、`question` |
| 评估 | `EvalRecord` | `question_id`、`mode`(`rag`/`llm`/`extractive`)、`is_correct`、`is_unknown`、`should_be_unknown`、`citation_valid`、`first_token_ms`、`total_ms`、RAGAS 四项 |
| 知识库 | `DocumentMeta` | `doc_id`、`title`、`source_path`、`page_count`、`chunk_count`、`table_count`、`status`(`pending`/`parsed`/`indexed`/`failed`)、`is_default` |
| 评估 | `GoldenQA` | `id`、`question`、`answer`、`evidence`、`evidence_pages`、`category`、`should_be_unknown` |

**`Answer.retrieved` 的用途说明**：该字段保存中间检索结果，用于日志/评估/调试，**按工单要求不展示给最终用户**（对应配置 `RAG_APP__EXPOSE_INTERMEDIATE_STEPS=false`）。

#### 4.2.4 `app/core/text_utils.py` —— 通用纯函数工具

集中放置文本处理逻辑，避免各模块重复实现；**全部为纯函数，无第三方强依赖，便于离线单测**。

| 函数 | 作用 | 要点 |
| --- | --- | --- |
| `normalize_text(text)` | 归一化空白与全角字符 | 全角空格 `\u3000` 与 `\xa0` 转半角空格、统一换行、折叠连续空行 |
| `remove_punctuation(text)` | 去标点 | 保留中文、字母、数字 |
| `has_cjk(text)` | 是否含中日韩文字 | 用于选择分词策略 |
| `tokenize(text, use_jieba=True)` | 中文分词 | **优先 jieba 并过滤停用词；无 jieba 时降级为「单字 + 相邻二元组 + 英文小写词」**——bigram 召回效果接近分词，保证无 jieba 也能跑 |
| `STOPWORDS` | 内置精简中文停用词表 | 含高频虚词与「年、月、日、元、万、亿、%」等单位词，避免 BM25 被虚词主导 |
| `stable_id(prefix, *parts, length=10)` | 生成稳定短 ID | `sha1(parts)[:10]`，**同输入必然同输出**，是索引幂等的基础 |
| `format_amount(value)` | 金额格式化 | `1234.5` → `1,234.50` |
| `extract_numbers(text)` | 抽取数字串 | 兼容全角逗号千分位，用于**答案数值校验** |
| `truncate` / `dedupe_keep_order` | 截断 / 保序去重 | 工具函数 |
| `looks_like_heading(line)` | 判断是否章节标题 | 识别 `第一章`、`一、`、`（一）`、`1.1` 等招股书标题结构，供 `chunker` 生成 `section` |
| `safe_filename` / `display_width` | 文件名安全化 / 显示宽度 | 导出与对齐用 |

#### 4.2.5 `app/core/bm25_index.py` —— BM25 关键词索引

混合检索的另一路。**双后端设计**保证任何环境都能建索引：

| 后端 | 触发条件 | 说明 |
| --- | --- | --- |
| `rank_bm25.BM25Okapi` | 已安装 `rank_bm25`（默认） | 工单推荐依赖 |
| `PureBM25`（本文件内置） | 未安装 `rank_bm25` | 纯 Python Okapi BM25，`k1=1.5`、`b=0.75`，IDF 采用 `log(1 + (N - df + 0.5) / (df + 0.5))` 平滑形式（保证非负），**行为与 `BM25Okapi` 对齐** |

关键接口：

```python
index.build(chunks)                       # 用 Chunk[] 重建索引（分词走 text_utils.tokenize）
index.search(query, top_k=10)             # -> [(chunk_id, 归一化分数), ...]
index.save()                              # -> data/index/bm25_index.pkl
index = BM25Index.load()                  # 文件缺失时返回空索引并 warning，不抛异常
index.size                                # 索引文档数
```

**关键设计——分数归一化**：`search()` 把本次查询的分数**按最高分归一到 (0, 1]**，这样 BM25 分数与向量余弦分数才可以直接按 `0.6 / 0.4` 加权融合。这是第 6.1 节融合公式能成立的前提。

**持久化产物**：

| 文件 | 内容 |
| --- | --- |
| `data/index/bm25_index.pkl` | pickle：`chunk_ids`、`corpus_tokens`、`use_jieba` |
| `data/index/bm25_index.meta.json` | 可读元信息：`documents`、`backend`、`use_jieba`、`avg_tokens`（便于人工核对） |

边界处理：空语料时 `build()` 记 `logger.warning` 且不建索引；`search()` 在无索引、无 chunk、分词结果为空时一律返回 `[]`，不抛异常。`build`/`search`/`save`/`load` 均带 `@trace` 与 `logger.info` 日志。

#### 4.2.6 `app/ui/streamlit_app.py` —— 网页界面层

约 53 KB，**只负责界面**：不实现任何解析、检索、生成逻辑，全部通过惰性导入的引擎对象完成。按工单 5.4 节要求，界面**只展示最终答案与引用来源**，意图识别、Query 改写、检索中间步骤只写日志。

**(1) 与 `qa_engine.py` 的接口契约（界面已固定的调用，实现引擎时必须提供）**

| 界面功能 | 调用方式 | 约定 |
| --- | --- | --- |
| 流式回答 | `engine.stream(question, conversation_id)` | 产出事件序列，见下表；缺失时回退到 `ask()` |
| 一次性回答 | `engine.ask(question, conversation_id)` | 返回 `Answer` |
| 新建会话 | `engine.new_conversation()` | 返回新的 `conversation_id` |
| 清空会话 | `engine.clear_conversation(conversation_id)` | — |
| 会话列表 | `engine.list_conversations()` | 返回 `Conversation[]` 或 `(label, cid, ...)` 序列 |
| 切换会话 | `engine.switch_conversation(conversation_id)` | — |
| 知识库统计 | `engine.stats()` | 返回 dict，界面取其中的数值项展示 |

**流式事件契约**（`_normalize_event` 的宽容解析）：

| 事件名 | 负载 | 界面行为 |
| --- | --- | --- |
| `first_token` | `{"first_token_ms": float}` | 记录并展示首字响应时间 |
| `delta` | `{"text": str}` | 追加到答案缓冲区（流式渲染） |
| `done` | `{"answer": Answer}` | 取最终 `Answer`（引用、页码、片段数、总耗时） |

事件可用 `(事件名, 负载)` 元组、带 `event`/`payload` 属性的对象、或**直接产出字符串**（按 `delta` 处理）三种形态之一；无法识别的事件会被忽略并记 `logger.warning`，不会崩溃。

**(2) 界面元素清单（实际控件名）**

| 区域 | 控件 |
| --- | --- |
| 侧边栏 · 文档 | `📁 文档` 小节：`选择 PDF 文档`（下拉，默认《招股说明书1.pdf》，其它来自 `data/raw`）、`上传 PDF（可选）`（上传后存入 `data/raw`）、`文档 ID（选填）` + `应用文档 ID`、`当前文档` 信息（文件/路径/大小/文档 ID） |
| 侧边栏 · 统计 | `📊 知识库统计`：`engine.stats()` 指标卡 + `查看完整统计信息` 折叠面板 |
| 侧边栏 · 对话 | `💬 对话`：`➕ 新建对话`、`🗑 清空当前对话`、`切换历史对话` 下拉、`当前会话：xxxxxxxx` |
| 侧边栏 · 说明 | `ℹ️ 关于“不清楚”兜底` 折叠面板（写明只依据原文、兜底口径、`[页码: 129]` 标记、3 秒预算） |
| 侧边栏 · 演示 | `🎯 演示问题（一键提问）` 折叠面板：工单固定的 **10 个问题**各一个按钮，一键提问 |
| 侧边栏 · 恢复 | `🔄 重新初始化问答引擎` 按钮；底部显示日志目录 |
| 主区域 · 头部 | 标题 `📄 {项目名}`；副标题显示当前文档与“答案均来自原文并附带引用页码” |
| 主区域 · 指标 | 四列指标卡：`首字响应`（ms + s）、`总耗时`（s）、`检索片段`（个）、`命中页码`（页） |
| 主区域 · 预算提示 | 超预算时 `st.error`：`⏱ 首字响应 xxx 毫秒（x.xx 秒），已超出 3.0 秒预算！`；达标时 caption：`✅ 首字响应 …，满足 3.0 秒预算。` |
| 主区域 · 引用 | `📚 引用来源（N 条）`，每条为折叠面板，标题含 `[页码: N]`、章节、相关度；展开后显示 `片段 ID`、`原文片段`，并可勾选 `查看完整片段原文` |
| 主区域 · 反馈 | `👍 有帮助` / `👎 没帮助` 按钮；`补充评论（选填）` 输入框 + `提交评论` 按钮 |
| 主区域 · 提问 | 底部 `st.chat_input`，占位提示“请输入关于招股说明书的问题，例如：武汉兴图新科电子股份有限公司注册资本是多少？”；生成中显示“正在检索文档并生成回答…” |

**(3) 健壮性设计**（对应 G7「禁止静默失败」）

- 引擎构造失败时展示 `st.error` + `可能的原因与处理办法` 折叠面板 + `🔄 重新初始化问答引擎` 按钮，**界面不白屏**；
- `logging_conf` / `config` 导入失败时降级运行，并在界面上说明降级原因；
- 引擎未上报 `first_token_ms` 时，界面**自行计时兜底**；未上报片段数时用引用条数兜底，保证指标可读；
- 引擎调用统一经 `_call_engine()` 包装，异常转成中文错误信息返回，**绝不把异常抛到界面**；
- 流式生成中途报错时，已生成的部分内容保留展示，只记日志不打断阅读。



以下为按数据契约推导的接口约定，供实现时对齐（最终签名以实现为准）：

| 模块 | 关键接口（约定） | 输入 → 输出 |
| --- | --- | --- |
| `pdf_parser.py` | `parse_pdf(path) -> ParsedDocument` | PDF 路径 → 逐页文本 + 表格（`ParsedPage[]`/`ParsedTable[]`），同时落盘 `data/processed/` |
| `chunker.py` | `build_chunks(doc: ParsedDocument) -> list[Chunk]` | 解析结果 → 带 `page`/`section`/`type` 的分块列表 |
| `embedder.py` | `embed_texts(texts) -> list[list[float]]`、`embed_query(text) -> list[float]` | 文本 → 归一化向量；无模型时降级 hash 向量 |
| `vector_store.py` | `add(chunks, vectors)`、`search(vector, top_k, page_filter) -> list[RetrievedChunk]`、`persist()` | 向量 → 相似片段；缺依赖时降级 numpy 内存索引 |
| `bm25_index.py` | `build(chunks)`、`search(query, top_k) -> list[RetrievedChunk]`、`save()`/`load()` | 分词后建索引 → 关键词片段 |
| `retriever.py` | `search(query_analysis, top_n) -> list[RetrievedChunk]` | Query 分析结果 → 融合去重后的 top_n 片段 |
| `query_understanding.py` | `analyze(question, history) -> QueryAnalysis` | 原始问题 + 历史 → 改写后问题、意图、关键词、子问题、页码过滤 |
| `generator.py` | `generate(question, contexts, history) -> Iterator[str]` / `Answer` | 问题 + 片段 → 流式答案；无服务时降级抽取式 |
| `citation.py` | `build_citations(chunks) -> list[Citation]`、`validate(citations, doc) -> bool` | 片段 → 引用；校验页码真实性 |
| `conversation.py` | `get_history(conversation_id, rounds)`、`append(message)`、`clear(conversation_id)` | 会话读写 |
| `evaluator.py` | `evaluate(questions, modes) -> EvalRecord[]`、`report(records)` | 批量评测 → CSV/Markdown 报告 |
| `qa_engine.py` | `ask(question, conversation_id=None, doc_id=None, page_filter=None) -> Answer` | **对外统一入口**，编排全链路 |

---

## 5. 数据流

### 5.1 离线索引链路（`研发/scripts/build_index.py`）

```text
data/raw/招股说明书1.pdf
   │
   │ ① pdf_parser：PyMuPDF 逐页取文本与页码；pdfplumber 抽表格
   │    剔除页眉页脚（正则见 7.x 配置 header_footer_patterns）
   ▼
ParsedDocument（ParsedPage[] + ParsedTable[]）
   │        └──► 落盘 data/processed/*.json（可追溯、可复现）
   │
   │ ② chunker：chunk_size=700 / overlap=100；表格整表成块（type=table）
   ▼
Chunk[]（chunk_id / page / section / type / content / table_id / keywords）
   │
   ├──► ③ SQLite：写入 documents（元数据）+ chunks（全部分块）
   │
   ├──► ④ embedder → 向量 ──► vector_store（Chroma，持久化到 data/index/）
   │
   └──► ⑤ bm25_index（text_utils 分词 + rank_bm25，缺失时内置纯 Python 实现）
        持久化为 data/index/bm25_index.pkl 与 bm25_index.meta.json
```

**幂等性**：`insert_chunks(..., replace_doc=True)` 会先清空该 `doc_id` 的旧分块再写入；`upsert_document` 使用 `ON CONFLICT(doc_id) DO UPDATE`。因此重复建索引不会产生重复数据。

### 5.2 在线问答链路

```text
用户提问
   │
   │ ① query_understanding.analyze(question, history)
   │    多轮改写 → 意图识别 → 关键词抽取 → 复杂问题分解 → 消歧（报告期内/军用领域/主营业务收入）
   ▼
QueryAnalysis
   │
   │ ② retriever.search()：混合检索
   │    ├─ 向量召回 top_k=10（可带 page_filter 页码过滤）
   │    ├─ BM25 召回 top_k=10
   │    ├─ 合并去重 → 候选 ≤ fusion_top_k
   │    ├─ 领域关键词加权（keyword_boost）
   │    ├─ 可选重排 bge-reranker-base
   │    └─ 取 rerank_top_n=5
   ▼
RetrievedChunk[]（每项带 vector_score / bm25_score / score / source）
   │
   │ ③ 兜底判定：无片段 或 最高分 < min_relevance_score  → 直接返回“不清楚”
   ▼
   │ ④ generator.generate()：qa_prompt.txt + 片段 + 历史 → LLM 流式输出
   │    首个 token 到达即记录 first_token_ms（前端同步开始显示）
   ▼
   │ ⑤ citation.build_citations()：由命中片段生成 Citation（页码/chunk_id/摘要）
   ▼
Answer（answer + citations + first_token_ms + total_ms + retrieved_count + pages）
   │
   ├──► ⑥ 落库：messages（含 citations JSON、first_token_ms）；更新 conversations
   └──► ⑦ 前端：流式渲染答案 + 引用列表（可展开原文）+ 首字响应时间 + 命中片段数/页码
```

### 5.3 失败与兜底链路（G3）

兜底回复统一取自配置 `RAG_APP__UNKNOWN_ANSWER`（默认 **“不清楚”**），共三处触发点：

| 触发点 | 判定条件 | `Answer.mode` | 记录内容 |
| --- | --- | --- | --- |
| 检索为空 | 向量与 BM25 均无命中 | `fallback` | `unknown_reason="no_candidate"` |
| 相关性过低 | 最高分 < `min_relevance_score`（默认 0.08） | `fallback` | `unknown_reason="low_relevance"` + 实际分数 |
| 生成不可用/无依据 | LLM 服务不可达且抽取式也拿不到内容；或模型输出无依据 | `fallback` / `extractive` | 服务地址、异常类型、原始输出 |

**强制要求**：兜底原因只写日志（`logs/app.log`、`logs/rag_trace.jsonl`）与 `Answer.unknown_reason`，**不向用户展示中间步骤**（G2/G7 的展示口径见工单 5.4）。

---

## 6. 检索特攻优化策略

针对《招股说明书1.pdf》这一**单篇、长文档（548 页）、表格密集、专有名词密集**的语料，系统采用 7 项针对性优化：

### 6.1 混合检索：向量 + BM25 双路召回

| 项 | 值 | 配置键 |
| --- | --- | --- |
| 向量召回 | 10 | `RAG_RETRIEVAL__VECTOR_TOP_K` |
| BM25 召回 | 10 | `RAG_RETRIEVAL__BM25_TOP_K` |
| 合并去重后候选数 | 20 | `RAG_RETRIEVAL__FUSION_TOP_K` |
| 最终交给 LLM 的片段数 | 5 | `RAG_RETRIEVAL__RERANK_TOP_N` |
| 融合权重（向量 / BM25） | 0.6 / 0.4 | `RAG_RETRIEVAL__VECTOR_WEIGHT` / `__BM25_WEIGHT` |

融合公式（约定实现）：

```text
final_score = vector_weight * vector_score + bm25_weight * bm25_score
```

**为什么必须双路**：工单问题 8、9（“注册资本是多少”“法定代表人是谁”）是**精确专名匹配**，向量检索容易把语义相近但事实不同的段落排在前面；而问题 1、3（“军用领域收入及占比”）需要**跨表格的语义聚合**，BM25 又无能为力。两路互补后再融合，才能同时覆盖 10 个问题。

### 6.2 领域关键词加权（`keyword_boost`）

命中以下词条的片段在融合阶段获得权重加成（实际默认值取自 `config.py`）：

| 关键词 | 权重 | 关键词 | 权重 |
| --- | --- | --- | --- |
| 注册资本 | 1.8 | 上游 | 1.5 |
| 法定代表人 | 1.8 | 下游 | 1.4 |
| 募集资金 | 1.6 | 供应商 | 1.4 |
| 补充流动资金 | 1.6 | 客户 | 1.3 |
| 技术标准 | 1.6 | 军用领域 | 1.5 |
| 科技进步奖 | 1.6 | 主营业务收入 | 1.5 |
| 收入 | 1.5 | 占比 / 比重 | 1.5 |

加权命中结果写入 `Chunk.keywords`，既参与打分也便于事后审计（“这条为什么被召回”）。

### 6.3 表格单独建索引 + 整表成块

招股书的**收入构成、募集资金用途、股权结构**等关键事实几乎都在表格里。表结构若被按字符数切断，行列对应关系会被破坏，LLM 极易读错数字。因此：

- `chunk.table_as_single_chunk = true`：**表格整表作为一个 chunk**，不参与 700 字切分；
- 表格块 `type="table"`、带 `table_id`（形如 `p152_t1`）与 `page`，并保存 `markdown` 形式（行列结构显式保留，便于 LLM 阅读）；
- `chunks` 表对 `type` 与 `table_id` 建索引（`idx_chunks_type`、`idx_chunks_table`），支持“只在表格里找”的检索策略。

### 6.4 页码过滤

`QueryAnalysis.page_filter` 支持限定页码范围检索（例如用户问“第 150 页附近的上游企业”），直接下发到 `vector_store.search(..., page_filter=...)`，既提速又提准。`chunks` 表建了 `idx_chunks_page(doc_id, page)` 支撑该过滤。

### 6.5 相关性阈值兜底

`min_relevance_score = 0.08`：最高分低于该值即判定“检索不到相关内容”，直接回复“不清楚”，**不让 LLM 拿到弱相关片段去编答案**。这是 G3 的第一道闸门。

> 注意：嵌入模型降级为 hash 向量时，分数分布会整体偏低，此阈值需按实际模型重新标定（见 README 第 9 节 Q5）。

### 6.6 页眉页脚剔除

招股书每页重复出现“武汉兴图新科电子股份有限公司 招股意向书”与 `1-1-xxx` 形式的页码，这些噪声会污染 BM25 词频统计。`PDFSettings.header_footer_patterns` 用三条正则剔除：

```text
^\s*武汉兴图新科电子股份有限公司\s*招股意向书\s*$
^\s*1-1-\d+\s*$
^\s*\d{1,3}\s*$
```

同时 `min_line_chars = 2` 过滤孤立字符行。

### 6.7 分块参数与重叠

`chunk_size = 700`（落在工单要求的 500~800 区间中上部）、`chunk_overlap = 100`、`min_chunk_chars = 30`。700 字在中文语境下约合一个完整小节，配合 100 字重叠可避免“答案正好被切在边界上”。

---

## 7. 数据库表结构

存储位置：`data/index/rag.sqlite3`（由 `PathSettings.sqlite_path` 决定）。

连接级 PRAGMA（`sqlite_manager.py` 建表脚本开头执行）：

```sql
PRAGMA journal_mode=WAL;      -- 读写并发，支撑 Streamlit 多线程
PRAGMA synchronous=NORMAL;    -- 兼顾性能与安全
PRAGMA foreign_keys=ON;       -- 外键级联删除生效
```

连接参数：`check_same_thread=False`、`timeout=30.0`、`row_factory=sqlite3.Row`；每个操作独立连接（`session()` 上下文管理器负责 commit/rollback，异常时回滚并 `logger.exception` 记录堆栈后**重新抛出**）；`SQLiteManager` 内部用 `threading.RLock` 保护建表；`get_sqlite_manager()` 提供进程级单例（传入 `db_path` 时绕过单例，便于测试隔离）。

### 7.1 `documents` —— 文档元数据

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `doc_id` | TEXT | **PK** | 文档唯一 ID |
| `title` | TEXT | NOT NULL, 默认 `''` | 文档标题 |
| `source_path` | TEXT | NOT NULL, 默认 `''` | 原始文件路径 |
| `page_count` | INTEGER | 默认 0 | 页数（本语料为 548） |
| `chunk_count` | INTEGER | 默认 0 | 分块数 |
| `table_count` | INTEGER | 默认 0 | 表格数 |
| `status` | TEXT | 默认 `pending` | `pending`/`parsed`/`indexed`/`failed` |
| `is_default` | INTEGER | 默认 0 | 是否默认加载的文档（0/1） |
| `created_at` / `updated_at` | TEXT | NOT NULL | ISO 8601 时间字符串 |

### 7.2 `chunks` —— 分块内容与元数据

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `chunk_id` | TEXT | **PK** | 形如 `c000123` |
| `doc_id` | TEXT | NOT NULL, **FK** → `documents(doc_id)` ON DELETE CASCADE | 所属文档 |
| `page` | INTEGER | NOT NULL | 来源页码（引用展示用） |
| `section` | TEXT | 默认 `''` | 所属章节标题路径 |
| `type` | TEXT | 默认 `text` | `text` / `table` |
| `content` | TEXT | NOT NULL | 分块正文 |
| `char_count` | INTEGER | 默认 0 | 字符数（缺省时按 `len(content)` 回填） |
| `table_id` | TEXT | 可空 | 表格块来源，形如 `p152_t1` |
| `keywords` | TEXT | 默认 `'[]'` | 命中的领域关键词（JSON 数组） |

索引：`idx_chunks_doc(doc_id)`、`idx_chunks_page(doc_id, page)`、`idx_chunks_type(doc_id, type)`、`idx_chunks_table(table_id)`。

### 7.3 `conversations` —— 对话会话

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `conversation_id` | TEXT | **PK** | 会话 ID |
| `title` | TEXT | 默认 `新对话` | 会话标题 |
| `doc_id` | TEXT | 可空 | 绑定的文档 |
| `created_at` / `updated_at` | TEXT | NOT NULL | 时间戳 |
| `message_count` | INTEGER | 默认 0 | 消息数（增删消息时同步维护） |

### 7.4 `messages` —— 消息记录

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `message_id` | INTEGER | **PK AUTOINCREMENT** | 自增主键 |
| `conversation_id` | TEXT | NOT NULL, **FK** → `conversations` ON DELETE CASCADE | 所属会话 |
| `role` | TEXT | NOT NULL | `user` / `assistant` / `system` |
| `content` | TEXT | NOT NULL | 消息正文 |
| `citations` | TEXT | 默认 `'[]'` | 引用 JSON 数组（`Citation` 序列化） |
| `first_token_ms` | REAL | 默认 0 | 该条回答的首字耗时（毫秒） |
| `created_at` | TEXT | NOT NULL | 时间戳 |

索引：`idx_messages_conv(conversation_id, message_id)`。

### 7.5 `feedback` —— 用户反馈

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `feedback_id` | INTEGER | **PK AUTOINCREMENT** | 自增主键 |
| `conversation_id` | TEXT | NOT NULL | 所属会话 |
| `message_id` | INTEGER | 可空 | 被反馈的回答消息 |
| `rating` | TEXT | NOT NULL | `up`（点赞）/ `down`（点踩） |
| `comment` | TEXT | 默认 `''` | 文字评论 |
| `question` | TEXT | 默认 `''` | 冗余保存问题文本，便于统计 |
| `created_at` | TEXT | NOT NULL | 时间戳 |

配套方法：`add_feedback`、`list_feedback(limit=100)`、`feedback_stats()`（返回 `{"up": n, "down": m}`）。

### 7.6 `eval_results` —— 评估结果

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `eval_id` | INTEGER PK AUTOINCREMENT | 自增主键 |
| `question_id` | INTEGER NOT NULL | 工单问题 ID（260/95/33/34/957/793/795/543/531/207） |
| `question` | TEXT NOT NULL | 问题原文 |
| `mode` | TEXT NOT NULL | `rag` / `llm` / `extractive` |
| `answer` | TEXT NOT NULL | 系统回答 |
| `golden` | TEXT 默认 `''` | 标准答案 |
| `is_correct` | INTEGER 默认 0 | 是否判定正确 |
| `is_unknown` | INTEGER 默认 0 | 是否回复了“不清楚” |
| `should_be_unknown` | INTEGER 默认 0 | 本题是否**应该**回复“不清楚” |
| `citation_pages` | TEXT 默认 `'[]'` | 引用页码 JSON 数组 |
| `citation_valid` | INTEGER 默认 0 | 引用页码是否真实存在 |
| `first_token_ms` | REAL 默认 0 | 首字耗时（毫秒） |
| `total_ms` | REAL 默认 0 | 整体耗时（毫秒） |
| `faithfulness` | REAL 可空 | RAGAS：忠实度 |
| `answer_relevancy` | REAL 可空 | RAGAS：答案相关性 |
| `context_precision` | REAL 可空 | RAGAS：上下文精确率 |
| `context_recall` | REAL 可空 | RAGAS：上下文召回率 |
| `created_at` | TEXT NOT NULL | 时间戳 |

索引：`idx_eval_q(question_id, mode)`。配套方法：`add_eval_record`、`list_eval_records(mode=None)`、`clear_eval_records()`。

### 7.7 `golden_qa` —— 标准问答（`data/eval/golden_qa.jsonl` 的数据库镜像）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | INTEGER **PK** | 与工单问题 ID 一致 |
| `question` | TEXT NOT NULL | 问题原文 |
| `answer` | TEXT NOT NULL | 标准答案 |
| `evidence` | TEXT 默认 `''` | PDF 原文依据 |
| `evidence_pages` | TEXT 默认 `'[]'` | 依据所在页码 JSON 数组 |
| `category` | TEXT 默认 `''` | 问题类别（收入/占比/标准/上下游/注册资本/法定代表人/募资用途…） |
| `should_be_unknown` | INTEGER 默认 0 | 是否属于“应当回答不清楚”的样本 |

### 7.8 `logs` —— 关键日志索引

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `log_id` | INTEGER PK AUTOINCREMENT | 自增主键 |
| `ts` | TEXT NOT NULL | 时间戳 |
| `level` | TEXT 默认 `INFO` | 级别 |
| `module` | TEXT 默认 `''` | 模块名 |
| `function` | TEXT 默认 `''` | 函数名 |
| `message` | TEXT 默认 `''` | 摘要 |
| `payload` | TEXT 默认 `'{}'` | 结构化载荷 JSON |

索引：`idx_logs_ts(ts)`、`idx_logs_level(level)`。
**定位说明**：全量日志以文件（`logs/app.log`、`logs/rag_trace.jsonl`）为准，本表只存**关键节点索引**，避免数据库被日志淹没。

### 7.9 表关系

```text
documents 1 ──── n chunks                （doc_id，级联删除）
conversations 1 ──── n messages          （conversation_id，级联删除）
conversations 1 ──── n feedback          （conversation_id）
messages    1 ──── n feedback            （message_id）
eval_results / golden_qa / logs          （独立表，通过 question_id 关联）
```

---

## 8. 配置体系

全部配置项集中在 `app/core/config.py`，可用 `RAG_<段名>__<字段名>` 环境变量覆盖。下表为默认值清单。

### 8.1 `app`（AppSettings）

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `app_name` | 基于 PDF 文档的 RAG 问答系统 | 应用名 |
| `version` | 1.0.0 | 版本号 |
| `first_token_budget_seconds` | 3.0 | 首字时间预算（验收线） |
| `unknown_answer` | 不清楚 | 统一兜底回复 |
| `log_level` | INFO | 日志级别 |
| `log_json` | true | 是否 JSON 日志 |
| `expose_intermediate_steps` | false | 是否向用户展示中间步骤（工单要求 false） |

### 8.2 `paths`（PathSettings）

| 字段 | 默认值（相对项目根） |
| --- | --- |
| `project_root` | 项目根目录（自动推导） |
| `data_raw` | `data/raw` |
| `data_processed` | `data/processed` |
| `data_index` | `data/index` |
| `data_eval` | `data/eval` |
| `eval_results` | `优化/评估结果/eval_results` |
| `logs` | `logs` |
| `sqlite_path` | `data/index/rag.sqlite3` |
| `default_pdf` | `data/raw/招股说明书1.pdf` |

### 8.3 `pdf`（PDFSettings）

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `min_line_chars` | 2 | 低于该长度的行视为噪声 |
| `header_footer_patterns` | 3 条正则 | 剔除页眉、`1-1-xxx` 页码、孤立数字 |
| `extract_tables` | true | 是否用 pdfplumber 抽表格 |
| `max_table_pages` | 0 | 表格提取页数上限（0 = 不限制） |

### 8.4 `chunk`（ChunkSettings）

| 字段 | 默认值 |
| --- | --- |
| `chunk_size` | 700 |
| `chunk_overlap` | 100 |
| `table_as_single_chunk` | true |
| `min_chunk_chars` | 30 |

### 8.5 `embedding`（EmbeddingSettings）

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `model_name` | `BAAI/bge-small-zh-v1.5` | 可换 `bge-large-zh-v1.5` / `bge-m3` |
| `device` | cpu | GPU 机器建议改 `cuda` |
| `batch_size` | 32 | 批大小 |
| `normalize` | true | 归一化（余弦相似度） |
| `fallback_dim` | 512 | 降级 hash 向量维度 |

### 8.6 `retrieval`（RetrievalSettings）

| 字段 | 默认值 |
| --- | --- |
| `vector_top_k` | 10 |
| `bm25_top_k` | 10 |
| `fusion_top_k` | 20 |
| `rerank_top_n` | 5 |
| `vector_weight` | 0.6 |
| `bm25_weight` | 0.4 |
| `use_reranker` | false |
| `reranker_model` | `BAAI/bge-reranker-base` |
| `keyword_boost` | 15 个领域词的权重表（见 6.2） |
| `min_relevance_score` | 0.08 |

### 8.7 `llm`（LLMSettings）

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `base_url` | `http://127.0.0.1:8000/v1` | vLLM/SGLang 的 OpenAI 兼容地址 |
| `api_key` | EMPTY | 本地服务无需真实 key |
| `model` | `Qwen2.5-7B-Instruct-AWQ` | 模型名 |
| `temperature` | 0.1 | 低温保证稳定性 |
| `top_p` | 0.8 | 采样 |
| `max_tokens` | 512 | 单次生成上限 |
| `connect_timeout` | 3.0 | 连接超时（秒） |
| `read_timeout` | 60.0 | 读取超时（秒） |
| `allow_extractive_fallback` | true | 无 LLM 服务时降级为抽取式回答 |

### 8.8 `conversation`（ConversationSettings）

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `max_history_rounds` | 5 | 保留最近 5 轮对话 |
| `rewrite_history_rounds` | 3 | 改写当前问题时携带的历史轮数 |

---

## 9. 日志与可观测性

### 9.1 三个日志文件

| 文件 | 格式 | 内容 |
| --- | --- | --- |
| `logs/app.log` | JSON 行（loguru `serialize=True`） | 全量运行日志，DEBUG 起，50 MB 轮转、保留 10 份 |
| `logs/error.log` | JSON 行 + 堆栈 | 仅 ERROR 及以上，20 MB 轮转、保留 10 份 |
| `logs/rag_trace.jsonl` | 一行一 JSON | 函数级 enter / exit / error 记录 |

### 9.2 `rag_trace.jsonl` 记录结构

| 事件 | 字段 |
| --- | --- |
| `enter` | `ts`、`event`、`module`、`function`、`args`、`kwargs` |
| `exit` | `ts`、`event`、`module`、`function`、`elapsed_ms`、`result` |
| `error` | `ts`、`event`、`module`、`function`、`elapsed_ms`、`error_type`、`error`、`traceback` |

**“禁止静默失败”的落地方式**：任何被 `@trace` 装饰的函数抛异常，都会①写一条 `error` 追踪记录、②经 `logger.exception` 写 `logs/error.log`（带堆栈）、③**继续向上抛出**——调用方必须显式处理，不允许吞掉。

### 9.3 建议打点的关键函数

| 模块 | 函数 | 关注指标 |
| --- | --- | --- |
| `pdf_parser` | `parse_pdf` | 解析耗时、页数、表格数 |
| `chunker` | `build_chunks` | 分块数、表格块数 |
| `embedder` | `embed_texts` / `embed_query` | 批耗时、向量维度、是否降级 |
| `vector_store` | `search` | 召回数、耗时 |
| `bm25_index` | `search` | 召回数、耗时 |
| `retriever` | `search` | 融合前后候选数、最高分/最低分 |
| `query_understanding` | `analyze` | 意图、改写结果、是否识别为追问 |
| `generator` | `generate` | **首字耗时**、总耗时、是否降级 |
| `qa_engine` | `ask` | 端到端耗时、命中页码、是否兜底 |

---

## 10. 接口预留（工单 5.11）

以下能力本期**可以不做完整功能，但接口必须留好**，且不得让后续接入改动核心链路：

| 能力 | 预留点 | 约定 |
| --- | --- | --- |
| 知识库管理 | `documents` 表 + `SQLiteManager.upsert_document/delete_document/list_documents`（已实现） | 后续补 `研发/scripts/build_index.py --add/--delete/--rebuild` 与界面入口 |
| 用户反馈 | `feedback` 表 + `add_feedback/list_feedback/feedback_stats`（已实现） | 界面点赞/点踩直接调用 |
| 评估 | `eval_results`/`golden_qa` 表 + `evaluator` + `研发/scripts/evaluate.py` | 支持按 `mode` 查询、清空重跑 |
| 模型切换 | `EmbeddingSettings.model_name`、`RetrievalSettings.reranker_model`、`LLMSettings.model` | 环境变量即可切换，无需改代码 |
| 缓存 | 预留装饰器/中间件位置（如 `qa_engine` 前的查询缓存层） | 本期不实现 |
| 追踪（LangFuse/LangSmith） | `@trace` 已产出函数级事件流，可直接桥接 | 本期不实现 |
| 容器化 / CI | 未实现（工单明确「本工单暂不做」） | 预留 Dockerfile 与流水线位置 |

---

## 11. 评估方案

### 11.1 评估矩阵

对 10 个工单问题，**每种模式各跑一遍**：

| 模式 | 含义 | `EvalRecord.mode` |
| --- | --- | --- |
| RAG | 检索 + 生成（本系统的正常路径） | `rag` |
| 纯 LLM | **不读 PDF**，只凭模型自身知识回答（对比基线） | `llm` |
| 抽取式 | 无 LLM 服务时的降级路径 | `extractive` |

### 11.2 RAGAS 四项指标

| 指标 | 衡量的东西 | 高分的含义 |
| --- | --- | --- |
| `faithfulness` | 答案是否由检索上下文支撑 | 没有编造 |
| `answer_relevancy` | 答案与问题的相关程度 | 没有答非所问 |
| `context_precision` | 检索上下文中有用片段的排序质量 | 有用片段排在前面 |
| `context_recall` | 检索是否覆盖了标准答案所需信息 | 该找的都找到了 |

### 11.3 确定性指标（不依赖 RAGAS）

| 指标 | 定义 | 数据来源 |
| --- | --- | --- |
| 准确率 | `is_correct` 为真的题目占比 | `eval_results` |
| 首字响应时间 | `first_token_ms` 的均值/最大值，与 3000 ms 预算比较 | `eval_results` |
| 引用正确率 | `citation_valid` 为真的占比（引用页码必须真实存在于 PDF） | `eval_results` + `chunks.page_range` |
| “不清楚”回复正确率 | `is_unknown == should_be_unknown` 的占比 | `eval_results` |

### 11.4 产物

| 文件 | 内容 |
| --- | --- |
| `优化/评估结果/eval_results/rag_vs_llm.csv` | 逐题对比：问题、RAG 答案、纯 LLM 答案、标准答案、各指标 |
| `优化/评估结果/eval_results/ragas_report.md` | RAGAS 四项指标的汇总报告（含分模式均值） |
| `data/eval/golden_qa.jsonl` | 10 个问题的标准答案（从 PDF 提取，带依据页码） |

生成命令：

```powershell
python 研发/scripts/evaluate.py
```

---

## 12. 测试方案

### 12.1 离线测试（`测试/tests/offline/`，不依赖 LLM 服务）

| 文件 | 验证内容 |
| --- | --- |
| `test_pdf_parser.py` | 页数（548）、文本非空、表格提取结果 |
| `test_chunker.py` | 分块数量、页码范围、元数据完整性（`chunk_id`/`page`/`section`/`type`） |
| `test_embedding.py` | 向量维度、归一化、批量/单条一致性 |
| `test_retriever.py` | 对 10 个问题检索，检查 top_k 是否覆盖答案所在页 |
| `test_generator.py` | 模拟生成，检查引用格式 `[页码: N]` |
| `test_evaluator.py` | 评估流程可跑通、输出文件生成 |
| `test_sqlite.py` | 8 张表结构、增删改查、级联删除 |

### 12.2 在线测试（`测试/tests/online/`，需要服务在跑）

| 文件 | 验证内容 |
| --- | --- |
| `test_api.py` | 发送 10 个问题，检查答案、引用、首字时间 |
| `test_concurrency.py` | 5~10 并发，检查稳定性与错误率 |
| `test_multiturn.py` | 多轮对话上下文继承 |
| `test_unknown.py` | 无关问题是否回复“不清楚” |
| `test_citation.py` | 引用页码是否真实存在（对照 `chunks` 表页码范围） |

### 12.3 用户测试（`测试/tests/user/`）

| 文件 | 用途 |
| --- | --- |
| `user_acceptance_checklist.md` | 用户验收清单 |
| `simulate_user.py` | 模拟用户操作并记录日志 |
| `demo_questions.md` | 演示问题列表 |

### 12.4 统一执行命令

```bash
conda activate gao6gongdan
pytest 测试/tests/offline -v
pytest 测试/tests/online -v
python 测试/tests/user/simulate_user.py
```

> 本文档不声称任何测试已经通过；测试结果以实际执行输出为准，并应作为阶段验收材料留存。

---

## 13. 开发流程与阶段计划

| 阶段 | 内容 | 产出 |
| --- | --- | --- |
| **阶段 0 环境准备** | 创建 conda 环境 `gao6gongdan`（Python 3.11）、安装依赖、配置日志、创建目录结构 | 可导入的 `config.py`/`logging_conf.py`、`logs/` 落盘 |
| **阶段 1 PDF 解析与索引** | `pdf_parser` → `chunker` → `embedder` → Chroma + BM25 + SQLite | `data/processed/`、`data/index/`、`documents`/`chunks` 表有数据 |
| **阶段 2 检索与生成** | `query_understanding`、`retriever`、`generator`、`citation`、兜底、多轮对话 | `qa_engine.ask()` 端到端可用 |
| **阶段 3 网页界面** | Streamlit 界面、集成问答引擎、引用展示、响应时间、反馈 | 可交互的网页 |
| **阶段 4 评估与测试** | 离线/在线/用户测试、RAGAS 评估、对比报告 | `优化/评估结果/eval_results/` 报告 |
| **阶段 5 文档与交付** | 四份文档、交付物整理 | 本目录四份 Markdown |

**当前进度（2026-10-01 19:20 核对）**：

| 阶段 | 状态 |
| --- | --- |
| 阶段 0 环境准备 | 🔶 部分完成：`config.py`、`logging_conf.py`、`text_utils.py`、`schemas.py`、`sqlite_manager.py`、`部署/环境配置/requirements.txt` 已落地，`logs/app.log`、`logs/rag_trace.jsonl` 已产出；**conda 环境 `gao6gongdan`（Python 3.11）尚未创建**，项目根下另有一个 Python 3.10 的暂存环境目录 `.gao6gongdan-src\`（见第 17 节） |
| 阶段 1 PDF 解析与索引 | 🔶 部分完成：`bm25_index.py` 已实现；`pdf_parser`、`chunker`、`embedder`、`vector_store` 待创建；`data/` 下尚无语料与索引产物 |
| 阶段 2 检索与生成 | ⬜ 未开始：`retriever`、`query_understanding`、`generator`、`citation`、`conversation`、`qa_engine` 待创建 |
| 阶段 3 网页界面 | 🔶 部分完成：`app/ui/streamlit_app.py`（53 KB）已实现全部界面元素，**但因 `qa_engine.py` 缺失，当前启动后会提示“问答引擎尚未就绪”** |
| 阶段 4 评估与测试 | ⬜ 未开始：`tests/` 与 `研发/scripts/evaluate.py` 待创建，`data/eval/golden_qa.jsonl` 待生成 |
| 阶段 5 文档与交付 | 🔶 进行中：本目录四份文档已成稿，其余交付物待补齐 |

**当前最大阻塞点**：`app/core/qa_engine.py`——界面、数据契约、存储层、BM25 索引均已就位，只差把各模块总装成 `QAEngine`。其接口要求已在 4.2.6 节固定。

### 13.1 开发约定

1. 代码注释使用**中文**，关键文件头部注明工单编号 `人工智能NLP-RAG-基于PDF 文档的问答系统`；
2. 配置只写在 `config.py`，业务模块**不得内嵌魔法数字**；
3. 跨模块数据一律走 `schemas.py` 中的 Pydantic 模型，不传裸 dict；
4. 关键函数加 `@trace`，异常必须记录并向上抛出；
5. 变更检索/分块/嵌入模型参数后，**必须重建索引**（换嵌入模型尤其如此）；
6. 文档与代码同步更新，`设计/文档/` 下的五份文档属于交付物。

---

## 14. 部署要点（算力云 4090）

| 事项 | 做法 |
| --- | --- |
| 环境 | 部署机建同名 conda 环境 `gao6gongdan`（Python 3.11）；推断框架 vLLM/SGLang 另建独立环境 |
| 依赖离线安装 | 在联网机器 `pip download -r 部署/环境配置/requirements.txt -d wheels/`，部署机 `pip install --no-index --find-links wheels/ -r 部署/环境配置/requirements.txt` |
| 模型权重 | 提前下载 `Qwen2.5-7B-Instruct-AWQ` 与 `BAAI/bge-small-zh-v1.5`（约 95 MB），或用内网模型目录 |
| 配置注入 | 全部通过 `RAG_*` 环境变量注入，代码零改动 |
| 嵌入设备 | 部署机设 `RAG_EMBEDDING__DEVICE=cuda`（若显存充裕） |
| 服务编排 | vLLM 独立进程占 8000；Streamlit 占 8501；两者同机通过回环地址通信 |
| 数据持久化 | `data/`（索引与语料）与 `logs/` 需挂载到持久盘，避免重启丢索引 |
| 健康检查 | LLM 侧探测 `/v1/models`；应用侧检查 `data/index/rag.sqlite3` 与向量索引目录是否存在 |
| 端口与防火墙 | 仅对内网开放 8501；8000 不对外暴露 |

> 4090 为 24 GB 显存，7B 4bit 权重的显存占用可控，剩余显存用于 KV Cache；`--gpu-memory-utilization`、`--max-model-len`、`--max-num-seqs` 三个参数需按实际并发压测结果调整（**具体取值以实测为准，本文档不预设数字**）。

---

## 15. 性能设计与并发

### 15.1 首字 < 3 秒的实现手段

| 手段 | 说明 |
| --- | --- |
| 流式输出 | `generator` 边生成边推送，首 token 到达即开始渲染并记录 `first_token_ms`——**首字时间与总生成时间解耦** |
| 本地推理 | LLM 部署在本机/内网，避免公网 RTT |
| 冗余召回可裁剪 | 默认向量 10 + BM25 10 → 融合 → 取 5；延迟紧张时调小 `TOP_K` |
| 重排默认关闭 | `use_reranker=false`，把重排耗时排除在关键路径外 |
| 表格块不切分 | 减少片段数量，降低 Prompt 长度与 LLM 首 token 延迟 |
| 连接超时保护 | `connect_timeout=3.0`，LLM 服务不可达时快速失败并走降级，不让用户干等 |

### 15.2 并发设计

| 层 | 机制 |
| --- | --- |
| LLM | vLLM 连续批处理，天然支持并发请求排队 |
| 向量库 | Chroma 嵌入式本地读写 |
| 数据库 | `journal_mode=WAL` 允许读写并发；`check_same_thread=False`；每次操作独立连接；写操作由 `RLock` 与事务保护 |
| 配置 | `get_settings()` 进程级单例（`lru_cache`） |
| 索引 | BM25 索引进程内常驻，避免每次查询重建 |

---

## 16. 已知限制与风险

| # | 限制/风险 | 影响 | 应对 |
| --- | --- | --- | --- |
| R1 | 语料为**单篇长文档**，`doc_id` 维度上的多文档能力尚未被真实场景验证 | 扩到多文档时排序可能退化 | `doc_id` 已贯穿全链路；多文档时需引入文档级过滤与归一化打分 |
| R2 | 表格提取质量依赖 pdfplumber，跨页表格可能被拆成多个表 | 收入类问题可能漏数字 | 建索引后需人工抽查关键表格页；必要时补跨页合并逻辑 |
| R3 | 嵌入模型降级为 hash 向量时，`min_relevance_score=0.08` 阈值失真 | 可能出现大量误兜底 | 降级状态下需重新标定阈值，或强制要求部署环境装好 sentence-transformers |
| R4 | 首字时间依赖 LLM 服务预热状态 | 冷启动首问可能超 3 秒 | 部署后先发一次热身请求；或将预热纳入启动脚本 |
| R5 | `keyword_boost` 权重表为**经验值** | 可能对个别问题过加权 | 需用 10 个工单问题的检索命中情况反向调参 |
| R6 | SQLite 写入为单写者模型 | 高并发写入（如大量反馈）可能排队 | 中等并发下 WAL 足够；再高需换 PostgreSQL |
| R7 | RAGAS 评估需要 LLM 参与打分 | 评估耗时较长、依赖模型质量 | 保留确定性指标作为主口径，RAGAS 作为补充 |
| R8 | 重排模型默认关闭 | 召回精度未吃满 | 精度不足时开启 `USE_RERANKER=true` 并重新测首字时间 |

---

## 17. 待补充事项（供核心开发者确认）

以下内容本文档**无法验证或尚未定稿**，需要核心开发者补充：

1. **`data/eval/golden_qa.jsonl` 尚未生成**——10 个问题的标准答案与依据页码需从 PDF 提取后填入，这是评估与验收的前置条件；
2. **conda 环境 `gao6gongdan` 尚未创建**——README 第 5.1 节给出创建命令，实际创建结果需回填；
3. **解释器版本存在三处不一致，需统一**（2026-10-01 核对）：
   - 工单/需求说明书要求：**Python 3.11** + conda 环境 `gao6gongdan`；
   - `E:\Anaconda\envs` 下**没有** `gao6gongdan`，但项目根目录存在隐藏目录 `.gao6gongdan-src\`，实测为 **Python 3.10.21** 的 conda 环境树（含 `conda-meta`、`Library`、`Scripts`、`python.exe`，已装 streamlit 1.37.1 / chromadb 1.5.9 / sentence-transformers 6.0.1 / pymupdf / pandas / numpy 2.2.6 / torch 2.14.0+cpu 等）；
   - `app\core\__pycache__\` 中存在 `*.cpython-312.pyc`，说明相关模块曾被 **Python 3.12** 导入过。
   → 请确认：是补齐 Python 3.11 的 `gao6gongdan` 环境（推荐，符合工单），还是正式变更基线版本（需需求方确认）；并清理 `__pycache__` 避免混淆；
4. **`部署/环境配置/environment.yml` 尚未落地**——需按 `部署/环境配置/requirements.txt` 生成并核对版本；
5. **工单与需求说明书的差异需确认**：工单原件写"支持文字和**语音**输入"与"支持**中文和英文**的问答"，而需求说明书明确"不需要语音输入"且未要求英文问答——本文档按需求说明书口径描述（无语音、中文问答），请确认最终范围；
6. **`qa_prompt.txt` 与 `query_rewrite_prompt.txt` 的具体措辞**——本文档只描述其约束（只依据片段作答、无依据回"不清楚"、必须带 `[页码: N]` 引用），实际文本待落地；
7. **检索融合的具体归一化方式**——BM25 侧已实现「按本次查询最高分归一到 (0,1]」（见 4.2.5），但**向量分数的归一化方式尚未定稿**，`retriever.py` 需与之配合（建议同样做 min-max 或除以最高分），否则 0.6/0.4 权重不可比；
8. **意图类别枚举的最终取值**——`QueryAnalysis.intent` 目前为自由字符串（默认"其他"），建议收敛为固定枚举；
9. **vLLM 启动参数**（`--max-model-len`、`--gpu-memory-utilization`、`--max-num-seqs`）——需按 4090 实测确定；
10. **并发指标口径**——"中等并发"未量化，建议明确为具体并发数（如 5~10）与可接受的 P95 延迟；
11. **`logs/rag_trace.jsonl` 的轮转策略**——当前实现为追加写入不轮转，长期运行需补归档方案；
12. **`.gao6gongdan-src\` 与 `.venv\` 的去留**——两者都是开发期产物，交付前建议明确是否清理（尤其 `.gao6gongdan-src\` 体量很大且含完整 Python 运行时，不应进入交付包）。
