# CLAUDE.md — 项目约定（写给后续 AI / 开发者）

本文件是**协作契约**。修改本项目前请先读完本文件；本项目已有完整实现，新增功能请严格延续既有分层与风格。

---

## 0. 硬性约束（Do Not Break）

1. **不要修改 `src/` 下已实现的代码去"顺手重构"**。改动应最小化，且只针对需求。
2. **所有配置必须来自 `.env`**，通过 `src/core/config.py` 的 `settings` 读取。**禁止**在代码里硬编码数据库地址、模型路径、密钥、限流阈值、提示词参数。
3. **本地模型必须离线加载**：`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`（由 `config.get_settings()` 设置），禁止改成联网下载。
4. **三个心理医生角色不可删减**，其 `persona_code` 是稳定标识，被 `KNOWLEDGE_DIRS`、评测数据集、知识库分区键依赖。
5. **不得放宽医疗边界**：`src/services/crisis_service.py` 的危机词、转介提示、`post_process` 中的"不诊断/不开药"正则兜底不得删除或弱化。
6. **虚拟环境固定**：`/home/dabaie/code/my_project/.venv`。不要创建新 venv，不要 `pip install` 到系统 Python。

---

## 1. 固定路径与运行环境

| 项目 | 值 |
| :--- | :--- |
| 项目根目录 | `/home/dabaie/code/psychologist` |
| 虚拟环境 | `/home/dabaie/code/my_project/.venv` |
| Python 解释器 | `/home/dabaie/code/my_project/.venv/bin/python`（Python 3.10.12） |
| pip | `/home/dabaie/code/my_project/.venv/bin/pip` |
| BGE-M3 | `/home/dabaie/models/bge-m3` |
| BGE-Reranker-v2-M3 | `/home/dabaie/models/bge-reranker-v2-m3` |
| MySQL | `127.0.0.1:3307`，库 `rag_roleplay`，用户 `dev` |
| Redis | `127.0.0.1:6379/0` |
| Milvus | `http://127.0.0.1:19530` |
| API | `0.0.0.0:8000`，文档 `/docs` |
| 日志目录 | `/home/dabaie/code/psychologist/logs` |
| 数据目录 | `/home/dabaie/code/psychologist/data` |

**Windows 侧访问**：WSL 路径 `\\wsl.localhost\Ubuntu-22.04\home\dabaie\code\psychologist`。
**执行命令**：Shell 工具须用 `wsl -d Ubuntu-22.04 -- bash -lc '...'`，复杂命令建议写入 `/tmp/xxx.sh` 再执行（PowerShell 引号转义极易出错）。

---

## 2. 常用命令（复制即用）

```bash
cd /home/dabaie/code/psychologist

# 安装/检查环境
bash scripts/install.sh

# 初始化（建库建表 + 角色 + 管理员 + Milvus Collection）
/home/dabaie/code/my_project/.venv/bin/python scripts/init_db.py
# 危险：删表重建
/home/dabaie/code/my_project/.venv/bin/python scripts/init_db.py --drop-all

# 幂等刷新三个角色（改完 persona_seed.py 后必跑）
/home/dabaie/code/my_project/.venv/bin/python scripts/seed_personas.py

# 知识库构建
/home/dabaie/code/my_project/.venv/bin/python scripts/ingest_knowledge.py --all
/home/dabaie/code/my_project/.venv/bin/python scripts/ingest_knowledge.py --persona code=cbt_chen --drop-existing

# 启动/停止
bash scripts/run.sh            # 后台；--reload 开发热重载；--port 8001 换端口
bash scripts/shutdown.sh

# 健康检查
curl http://127.0.0.1:8000/health

# 数据库迁移
/home/dabaie/code/my_project/.venv/bin/alembic upgrade head
/home/dabaie/code/my_project/.venv/bin/alembic revision --autogenerate -m "描述"
```

---

## 3. 分层架构与模块职责边界

```
api/v1/*.py      只做：参数校验(chema) → 调 service → 用 ok()/异常 包装返回。禁止直接写 SQL、禁止拼提示词。
services/*.py    业务编排与事务边界。允许依赖 db/、rag/、models/、core/。禁止依赖 api/。
rag/*.py         纯 RAG 组件（解析/分块/向量/检索/重排/提示词），不碰 MySQL 事务，不碰 HTTP。
db/*.py          连接与存储原语（mysql/redis/milvus）。只依赖 core/ 与 models/。
core/*.py        config / security / logging / exceptions。不得依赖上层。
models/*.py      SQLAlchemy ORM，一张表一个类，集中在 models/__init__.py 导出。
schemas/*.py     Pydantic 请求/响应模型，集中在 schemas/__init__.py 导出。
utils/helpers.py 无状态纯函数。
```

**依赖方向（单向）**：`api → services → {rag, db, models, core}`，`rag → {db, core}`。任何反向依赖都是 bug。
**唯一例外**：`src/rag/pipeline.py` 是组合根（composition root），允许引用 services 层做编排转发——它只允许被 `scripts/` 与 api 层调用，内部不得出现业务分支。Query 改写依赖 LLM，因此位于 `services/retrieval_service.py`（不放在 rag 层）。

### 关键模块速查

| 文件 | 职责 | 关键函数 |
| :--- | :--- | :--- |
| `src/main.py` | FastAPI 入口、lifespan 初始化、全局异常、访问日志中间件 | `lifespan` / `health` |
| `src/api/deps.py` | JWT 鉴权、管理员校验、取客户端 IP | `get_current_user` / `get_current_admin` |
| `src/core/config.py` | 全部配置项 + 派生属性 | `settings` / `database_url` |
| `src/core/security.py` | bcrypt 哈希、JWT 签发/解析 | `hash_password` / `create_access_token` |
| `src/services/persona_seed.py` | **三角色种子数据 + 知识库目录映射** | `PERSONAS` / `KNOWLEDGE_DIRS` |
| `src/services/rag_service.py` | 在线问答编排（检索→提示词→LLM→后处理→落库） | `prepare` / `answer` / `answer_stream` |
| `src/services/memory_service.py` | 短期记忆 + 长期记忆摘要 | `get_short_term` / `maybe_save_long_term` |
| `src/services/knowledge_service.py` | 离线入库、动态更新、统计 | `ingest_file` / `rebuild_persona_index` |
| `src/services/crisis_service.py` | 危机词检测、转介、后处理 | `detect_crisis` / `post_process` |
| `src/services/eval_service.py` | RAGAS/内置 Judge 评测 | `run_eval` |
| `src/services/retrieval_service.py` | Query 改写（LLM）+ 改写后检索编排 | `rewrite_query` / `search_with_query_rewrite` |
| `src/rag/pipeline.py` | RAG 流水线组合根（离线/在线链路编排转发） | `offline_ingest_file` / `online_answer` |
| `src/rag/retriever.py` | 混合检索 + 精排 + 阈值过滤（不含 LLM 依赖） | `retrieve` |
| `src/db/milvus.py` | 两个 Collection 的建/写/搜/删 | `ensure_collections` / `hybrid_search_knowledge` |

---

## 4. 编码风格

- **语言**：注释、日志、docstring、异常信息全部使用**中文**。
- **模块 docstring**：每个文件首行一句话说明职责（现有文件均已如此）。
- **类型注解**：函数签名必须带类型注解（`from typing import ...`），使用 `Optional[...]`、`List[Dict[str, Any]]` 等。
- **命名**：模块/函数 `snake_case`，类 `PascalCase`，常量 `UPPER_SNAKE`，私有函数前缀 `_`。
- **行长**：建议 ≤ 100，现有代码为 100。
- **导入顺序**：标准库 → 第三方 → 本项目 `src.`；项目内使用绝对导入（`from src.services import ...`），**禁止**相对导入。
- **数据库会话**：API 层用 `Depends(get_db)`；脚本/后台任务用 `with session_scope() as db:`。不要在 service 里自己 `create_engine`。
- **单例**：Embedder / Reranker / MilvusClient / Redis / OpenAI 客户端均用**模块级全局变量 + 双检锁懒加载**，不要每次 new。
- **配置项新增**：在 `config.py` 的 `Settings` 加字段并给默认值 → 同步写入 `.env.example` → 必要时写入 `.env` → 在 `docs/06-deployment.md` 补说明。

---

## 5. 日志规范

统一使用 `src.core.logging.get_logger(name)`，**禁止** `print` 和裸 `logging.getLogger`。

```python
from src.core.logging import get_logger
logger = get_logger("service.xxx")   # 命名：层次.模块，如 service.memory / rag.retriever / db.mysql
```

- 命名空间约定：`main` / `api.*` / `service.*` / `rag*` / `rag.*` / `db.*` / `llm` / `crisis` / `eval.*` / `scripts.*`。
- 输出目标：控制台 + `logs/app.log`（全量，20MB×5 滚动）+ `logs/error.log`（ERROR）+ `logs/llm.log`（`llm`）+ `logs/rag.log`（`rag`）。
- 用 `%s` 惰性格式化，不要 f-string 拼日志：`logger.info("命中 %d 条", n)`。
- 异常必须 `logger.error("...: %s", exc)`；需要堆栈时加 `exc_info=True`。
- **禁止打印密钥、密码、token、完整对话内容到日志**。检索日志只记录条数、耗时、改写前后 query（`src/rag/retriever.py` 的 `Query 改写：... -> ...`）。

---

## 6. 异常处理规范

统一异常体系在 `src/core/exceptions.py`：

| 异常 | `code` | HTTP | 场景 |
| :--- | :--- | :--- | :--- |
| `AppError` | 400 | 400 | 通用业务错误（如不支持的文件类型） |
| `AuthError` | 401 | 401 | 未登录、token 失效、密码错误 |
| `PermissionError_` | 403 | 403 | 需要管理员、越权访问他人会话 |
| `NotFoundError` | 404 | 404 | 用户/角色/文档/会话不存在 |
| `ConflictError` | 409 | 409 | 用户名/邮箱/角色编码重复 |
| `RateLimitError` | 429 | 429 | 触发限流、角色未开放 |
| `ExternalServiceError` | 502 | 502 | 大模型等外部依赖失败 |

规矩：

1. **业务异常直接 raise**，由 `src/main.py` 的 `app_error_handler` 统一转成 `{"code":..., "message":..., "data":null}`，**不要在 API 层 try/except 后手写 JSONResponse**。
2. **响应成功统一用 `ok(data, message)`**，不要手写 dict。
3. **外部依赖调用必须降级**：
   - LLM 辅助调用（改写/摘要/打分/起标题）用 `llm_service.simple_complete()`，失败返回空串，不抛异常；
   - Redis 全部操作捕获 `RedisError`，失败返回空/False；
   - Milvus 检索失败返回 `[]`（`hybrid_search` 失败自动降级 `dense_search`）；
   - 模型加载 GPU 失败自动降级 CPU。
   **主流程不能因为辅助能力失败而中断。**
4. Redis 限流检查失败时**放行**（`check_rate_limit` 返回 True），这是有意为之。
5. 未捕获异常由 `unhandled_handler` 兜底返回 500，不要吞掉异常。

---

## 7. 如何新增一个心理医生角色

以新增"焦医生（焦点解决短期治疗，`sfbt_jiao`）"为例，**只改 3 处**：

### 步骤 1：在 `src/rag/prompt.py` 增加 system prompt（可选但推荐）

```python
SFBT_SYSTEM_PROMPT = """你是焦医生，一名焦点解决短期治疗取向的心理咨询师。
你的风格是：未来导向、资源取向、简洁。
你的核心技巧是：奇迹问句、例外询问、刻度化提问、小步行动。

安全边界：
1. 你不是精神科医生，不进行医学诊断，不开药，不替代线下就医。
2. 如果用户出现自伤、自杀、伤人风险，立即建议联系 120/110、当地精神卫生中心或心理援助热线 12356。
3. 不输出违法、暴力、歧视、色情内容。

对话规则：
1. 先共情，再澄清，再给建议。
2. 每次最多问 1-2 个问题。
3. 避免说教，使用用户能理解的语言。
4. 结合知识片段回答，不要编造。"""
```

> 若新角色复用标准话术，也可以直接在 `PERSONAS` 里写 prompt 字符串，不走 `prompt.py`。

### 步骤 2：在 `src/services/persona_seed.py` 的 `PERSONAS` 追加一项，并在 `KNOWLEDGE_DIRS` 注册知识库目录

```python
{
    "persona_code": "sfbt_jiao",              # 唯一，稳定标识
    "name": "焦医生",
    "title": "焦点解决短期治疗型心理咨询师",
    "therapy_type": "焦点解决短期治疗 SFBT",
    "style": "未来导向、资源取向、简洁",
    "methods": "奇迹问句、例外询问、刻度化提问、小步行动",
    "greeting": "你好，我是焦医生。我们一起来看看，你希望情况变成什么样。",
    "system_prompt": SFBT_SYSTEM_PROMPT,
    "knowledge_scope": "焦点解决、资源取向、目标设定、小步行动",
    "avatar": "",                              # 可选，图片 URL
    "safety_boundary": SAFETY_BOUNDARY,        # 复用统一安全边界
    "model_params": {"temperature": 0.6, "max_tokens": 2048, "top_p": 0.9},
    "status": 1,
}
```

```python
KNOWLEDGE_DIRS: Dict[str, List[str]] = {
    # ... 已有三个角色 ...
    "sfbt_jiao": [
        "心理医生/焦医生（焦点解决短期治疗型）",
        "心理医生/通用知识库",
    ],
}
```

### 步骤 3：写入数据库并建知识库

```bash
cd /home/dabaie/code/psychologist

# 幂等写入（已存在的角色会同步提示词/开场白/模型参数，并清 Redis 角色缓存）
/home/dabaie/code/my_project/.venv/bin/python scripts/seed_personas.py

# 把新角色的文档放到目录后入库
/home/dabaie/code/my_project/.venv/bin/python scripts/ingest_knowledge.py --persona code=sfbt_jiao

# 验证
curl -s http://127.0.0.1:8000/api/v1/personas
```

**注意事项**

- 新角色**无需改 Milvus Schema**：`persona_knowledge` 以 `persona_id` 作分区键，任意新角色自动隔离。
- 若希望角色可被前端切换，`GET /api/v1/personas` 会自动返回（`status=1`）。
- 也支持运行时通过 `POST /api/v1/personas`（管理员）动态新增，但**不会**自动获得知识库目录映射，`/knowledge/rebuild` 对新角色无效（`KNOWLEDGE_DIRS` 未配置）。持久化角色仍建议走 `persona_seed.py`。

---

## 8. 如何新增一个知识库

### 方式 A：新增目录（推荐，可复现）

1. 在 `心理医生/` 下新建目录，如 `心理医生/焦医生（焦点解决短期治疗型）/`；
2. 把 PDF / TXT / MD / DOCX 放进去（支持类型见 `src/rag/parser.py` 的 `SUPPORTED_TYPES = {pdf, txt, md, markdown, docx}`）；
3. 在 `persona_seed.py` 的 `KNOWLEDGE_DIRS` 对应角色下加入该目录（相对项目根目录，可多个）；
4. 执行入库：

```bash
/home/dabaie/code/my_project/.venv/bin/python scripts/ingest_knowledge.py --persona code=sfbt_jiao
# 或全量
/home/dabaie/code/my_project/.venv/bin/python scripts/ingest_knowledge.py --all
```

> 目录不存在时脚本只打印警告并跳过，不会报错。

### 方式 B：运行时上传（管理员，单文件）

```bash
curl -X POST http://127.0.0.1:8000/api/v1/knowledge/upload \
  -H "Authorization: Bearer <admin_token>" \
  -F "persona_id=2" -F "strategy=paragraph" \
  -F "file=@/path/to/book.pdf"
```

文件落到 `data/uploads/{时间戳}_{原文件名}`，解析入库后写 `knowledge_docs` / `knowledge_chunks` 元数据。

### 方式 C：共享知识库

想让所有角色共用某批资料，把文件放入 `心理医生/通用知识库/`，它已被三个（及后续）角色的 `KNOWLEDGE_DIRS` 共同引用，每个角色入库时会各自生成一份带自己 `persona_id` 的向量——**这是刻意的冗余设计**，用来保证检索时按分区键过滤。

### 分块策略选择

| 策略 | 适用场景 |
| :--- | :--- |
| `paragraph`（默认） | 中文心理类书籍，段落结构清晰 |
| `sentence` | 对话体、短句多的文本 |
| `fixed` | 结构混乱、无段落 |
| `heading` | 有"第 X 章 / 1.1 / # 标题"层级的教材 |
| `semantic` | 长段落、主题漂移明显（轻量近似实现） |

### 动态更新与清理

```bash
# 删除某文档（同时删 Milvus 向量与 MySQL 元数据）
curl -X DELETE http://127.0.0.1:8000/api/v1/knowledge/docs/12 \
  -H "Authorization: Bearer <admin_token>"

# 按角色重建（先清空该角色向量+元数据）
curl -X POST http://127.0.0.1:8000/api/v1/knowledge/rebuild \
  -H "Authorization: Bearer <admin_token>" -H "Content-Type: application/json" \
  -d '{"persona_id":2,"drop_existing":true}'
```

低质量与重复块过滤规则（`src/services/knowledge_service.py`）：`MIN_CHUNK_TOKENS = 20`（低于即丢弃）、完全重复块去重；解析后字符数 < 50 直接判失败（疑似扫描件）。

---

## 9. 测试与自检

### 现状

- 虚拟环境已安装 `pytest 9.1.1` / `pytest-asyncio 1.4.0`；
- `tests/` 已落地：11 个测试文件（单元 7 + 集成 4），共 174 项（参数化展开），当前全部通过；`pytest.ini` 已注册 `integration` 标记（真实 MySQL/Redis/Milvus/BGE 模型），可用 `pytest -m integration` 单独筛选；
- `tests/conftest.py` 会 session 级 monkeypatch 屏蔽真实大模型（严禁测试调用 DeepSeek），并提供 MySQL/Redis/Milvus 探测 fixture（不可用时自动 skip）；
- 测试策略、手工用例清单与报告模板见 [`docs/07-testing.md`](docs/07-testing.md)。

### 约定

- 新增测试放 `tests/`，文件命名 `test_*.py`，函数命名 `test_*`；异步用 `pytest.mark.asyncio`。
- 运行：
  ```bash
  cd /home/dabaie/code/psychologist
  PYTHONPATH=/home/dabaie/code/psychologist \
    /home/dabaie/code/my_project/.venv/bin/pytest -v
  # 仅跑不需要外部依赖的部分（推荐给纯函数）
  PYTHONPATH=$(pwd) /home/dabaie/code/my_project/.venv/bin/pytest -v tests/test_chunker.py
  ```
- **优先写"无外部依赖"的单元测试**（`estimate_tokens`、`chunk_text`、`clean_text`、`detect_crisis`、`post_process`、`build_chat_prompt`、`hash_password/verify_password`）；涉及 MySQL/Redis/Milvus/LLM 的用集成测试并允许在服务不可用时 `pytest.skip`。
- 改动后的**最低自检清单**：
  1. `python -m compileall src scripts`（语法）；
  2. 启动服务看 `logs/uvicorn.out` 与 `logs/error.log` 无 ERROR；
  3. `curl /health` 三库均为 true；
  4. 跑一次 `/api/v1/chat` 与 `/api/v1/chat/stream`；
  5. 改了角色/知识库 → 跑 `seed_personas.py` + `ingest_knowledge.py` + `/knowledge/stats` 验证。

---

## 10. 常见坑（踩过的）

1. **`ragas` 在本环境不可导入**（`ragas 0.4.3` 与 `langchain-community 0.4.2` 冲突：`No module named 'langchain_community.chat_models.vertexai'`）。评测走内置 LLM-as-Judge，**不要为了修 ragas 去动 `eval_service.py` 的双引擎回退逻辑**，也不要升级 `langchain-community`（会连带影响 `langchain-*` 全家桶）。
2. ~~`MILVUS_MEMORY_COLLECTION` 只存在于 `config.py` 默认值~~ 已修复：`.env.example` 已补齐该键（2026-09-16）。
3. **会话标题不是 LLM 生成**：`prompt.py` 里有 `TITLE_PROMPT`，但 `conversation_service.auto_title()` 实际用**用户首条消息截断 20 字符**（零延迟、零成本）。若改为 LLM 生成需自行接入并注意首问延迟。
4. ~~`.env.example` 缺 `ADMIN_USERNAME` / `ADMIN_PASSWORD`~~ 已修复（2026-09-16），并同步补了 `CORS_ALLOW_ORIGINS`。注意：自 2026-09-16 起 **安全校验已生效**——`JWT_SECRET_KEY`（≥32 字符）与 `ADMIN_PASSWORD` 缺失或为占位符时服务**拒绝启动**（`settings.validate_security()`，在 lifespan 调用；测试不进 lifespan 不受影响）。
5. **MySQL 端口是 3307 不是 3306**，写连接串时不要想当然。
6. ~~`sql/schema.sql` 与 ORM 有细微差异~~ 已修复（2026-09-16）：`schema.sql` 的 `audit_logs.detail` 已对齐为 `VARCHAR(4000)`，并在文件头注明唯一事实源是 `src/models` + Alembic。
7. ~~Redis "Hash" 注释与实现不符~~ 已修复（2026-09-16）：`redis.py` 头部 Key 规范已改为 String(JSON)，并补充 `token_blacklist:{jti}` 吊销黑名单键说明。
8. **服务启动时 Milvus/MySQL 初始化失败只记 ERROR 不退出**，服务仍会起来并对外提供（部分）接口——排查时务必先看 `logs/app.log` 的启动段。
9. **`WARMUP_MODELS=0` 可跳过模型预热**；调试接口时开启可让启动更快。
10. **目录名含中文、空格和全角破折号**（如 `周正念医生（正念情绪调节型）——3 本`），复制路径时务必加引号。
11. **`deepseek-flash` 是思考型模型**（2026-09-16 实测）：默认先输出 `reasoning_content`，`content` 为空——`simple_complete` 拿空串、`chat` 答复为空。已在 `llm_service.py` 统一传 `extra_body={"thinking": {"type": "disabled"}}`（由 `LLM_THINKING` 控制，默认 disabled）。可用模型以 `curl https://api.deepseek.com/models` 返回为准（当前：`deepseek-flash` / `deepseek-v4-pro`），不要沿用 `deepseek-v4.1-flash` 等已下线模型名。
12. **SSE 首 token 延迟优化**（阶段 4，2026-09-16）：三层——`rag_service.prepare_basic` 先建会话发 meta（检索前，实测 0.05s）；Query 改写短超时 `LLM_REWRITE_TIMEOUT=3s` 失败回退原问题；检索结果 Redis 缓存 `RETRIEVAL_CACHE_TTL=300s`（键 `retrieval_cache:{persona_id}:{sha256}`，`references` 随 done 事件下发，知识库入库/删除时 `clear_retrieval_cache` 失效）。同问题二次问答省 ~0.8s 检索段。
13. **登录接口按 IP 限流**（`LOGIN_RATE_LIMIT_PER_MINUTE=10`，键 `rate_limit:login:{ip}:{minute}`）：测试连错密码 12 次会把本 IP 打满 1 分钟 429。日志行带 `[rid=xxx]`（X-Request-ID 贯穿全链路，响应头回传）。