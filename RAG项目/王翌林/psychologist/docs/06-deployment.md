# 06 — 部署与运维

> 环境：Windows 11 + WSL2 + Ubuntu 22.04。项目根目录 `/home/dabaie/code/psychologist`。
> 所有配置从项目根 `.env` 读取（`src/core/config.py`），禁止硬编码。

---

## 1. 部署步骤总览

```text
① 准备 .env  →  ② bash scripts/install.sh（环境检查+依赖+连通性）  →  ③ python scripts/init_db.py（建库建表+角色+管理员+Milvus Collection）
→  ④ python scripts/ingest_knowledge.py --all（知识库入库）  →  ⑤ bash scripts/run.sh（启动）  →  ⑥ curl /health + 冒烟验证
```

### 1.1 配置 `.env`

```bash
cd /home/dabaie/code/psychologist
cp .env.example .env && vim .env
```

上线前必须修改：`LLM_API_KEY`、`JWT_SECRET_KEY`（≥32 字符）、`ADMIN_PASSWORD`。

### 1.2 环境安装与检查

`scripts/install.sh` 依次执行：读取 `.env` → 校验 Ubuntu → 校验虚拟环境与 Python → 安装 `requirements.txt` → 校验两个本地模型目录 → 检查 MySQL/Redis/Milvus 端口 → 调用 `init_db.py`。

### 1.3 初始化

```bash
/home/dabaie/code/my_project/.venv/bin/python scripts/init_db.py     # 幂等：建库建表+系统角色+三医生+管理员+Collection
/home/dabaie/code/my_project/.venv/bin/python scripts/init_db.py --drop-all   # 危险：删表重建
/home/dabaie/code/my_project/.venv/bin/python scripts/seed_personas.py        # 幂等刷新三角色（改 persona_seed.py 后必跑）
```

默认管理员：`admin / admin123456`（`.env` 的 `ADMIN_USERNAME`/`ADMIN_PASSWORD`），**上线前必须修改**。

### 1.4 启动 / 停止

```bash
bash scripts/run.sh                 # 后台启动，日志 logs/uvicorn.out，PID data/uvicorn.pid
bash scripts/run.sh --reload        # 开发热重载
bash scripts/run.sh --port 8001     # 换端口
bash scripts/shutdown.sh            # 停止
curl http://127.0.0.1:8000/health   # 三库均为 true 即正常
```

`run.sh` 会先探测 MySQL/Redis/Milvus 端口；不可用时打印告警但**不阻止启动**（相关功能降级）。服务 `lifespan` 自动完成：建表 → 初始化角色与管理员 → 创建 Milvus Collection → 预热 BGE-M3 / Reranker（`WARMUP_MODELS=0` 可关闭，调试时更快）。

---

## 2. 配置项逐项说明

### 2.1 项目

| 变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `PROJECT_ROOT` | 代码所在目录 | 项目根 |
| `APP_NAME` / `APP_VERSION` | 基于 RAG 的心理医生多角色陪伴系统 / 1.0.0 | 服务信息 |
| `DEBUG` | false | 调试模式 |
| `LOG_DIR` / `LOG_LEVEL` | `{root}/logs` / INFO | 日志目录与级别 |
| `DATA_DIR` | `{root}/data` | uploads / eval 报告 / uvicorn.pid |

### 2.2 本地模型（必须离线加载）

| 变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `EMBEDDING_MODEL_PATH` | /home/dabaie/models/bge-m3 | BGE-M3，dense 1024 维 |
| `RERANKER_MODEL_PATH` | /home/dabaie/models/bge-reranker-v2-m3 | CrossEncoder 重排 |
| `EMBEDDING_DEVICE` / `RERANKER_DEVICE` | cuda:0 | GPU 加载失败自动降级 CPU |
| `EMBEDDING_BATCH_SIZE` / `RERANKER_BATCH_SIZE` | 16（.env 可设 64）/ 8 | 吞吐与显存权衡 |
| `EMBEDDING_DIM` | 1024 | 向量维度 |
| `RERANK_TOP_N` | 5 | 重排后送入提示词条数 |
| `HF_HUB_OFFLINE` | 1 | 强制离线，禁止联网下载 |

### 2.3 MySQL / Redis / Milvus

| 变量 | 默认值 |
| :--- | :--- |
| `DB_HOST` / `DB_PORT` | 127.0.0.1 / **3307**（注意不是 3306） |
| `DB_NAME` / `DB_USER` / `DB_PASSWORD` | rag_roleplay / dev / 355359 |
| `DB_POOL_SIZE` / `DB_MAX_OVERFLOW` | 10 / 20 |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` | 127.0.0.1 / 6379 / 0 |
| `MILVUS_URI` / `MILVUS_COLLECTION` | http://127.0.0.1:19530 / persona_knowledge |
| `MILVUS_MEMORY_COLLECTION` | user_long_term_memory（**仅在 config.py 有默认值，`.env` 需手动添加才能改**） |

### 2.4 RAG 参数

| 变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 512 / 80 | 分块目标 token 与重叠 |
| `PARENT_CHUNK_SIZE` / `CHILD_CHUNK_SIZE` | 1024 / 256 | parent_child 策略 |
| `RETRIEVE_TOP_K` | 20 | 每路召回条数 |
| `SIMILARITY_THRESHOLD` | 0.35 | 重排分下限；全部低于时保留最高分 1 条 |
| `SHORT_TERM_MAX_TURNS` / `SHORT_TERM_TTL` | 10 / 86400 | Redis 短期记忆 |
| `LONG_TERM_SUMMARY_TRIGGER` | 20 | 消息数达该值整数倍自动摘要入库 |
| `QUERY_REWRITE_ENABLED` | true | LLM Query 改写 |

### 2.5 大模型（在线 DeepSeek，OpenAI 兼容）

| 变量 | 默认值 |
| :--- | :--- |
| `LLM_PROVIDER` | openai_compatible |
| `LLM_BASE_URL` | https://api.deepseek.com |
| `LLM_API_KEY` | （必填） |
| `LLM_MODEL` | deepseek-flash |
| `LLM_REWRITE_TIMEOUT` | 3.0（Query 改写短超时，秒） |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_TIMEOUT` | 0.7 / 2048 / 60 |

### 2.6 鉴权与安全

| 变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `JWT_SECRET_KEY` | change_me | **生产必须改**，建议 ≥32 字符 |
| `JWT_ALGORITHM` | HS256 | — |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` / `JWT_REFRESH_TOKEN_EXPIRE_MINUTES` | 1440 / 10080 | 分钟 |
| `BCRYPT_ROUNDS` | 12 | 密码哈希强度 |
| `CRISIS_HOTLINE` / `EMERGENCY_PHONE` | 12356 / 120,110 | 危机转介 |
| `RATE_LIMIT_PER_MINUTE` | 30 | 登录用户聊天限流（次/分钟） |
| `LOGIN_RATE_LIMIT_PER_MINUTE` | 10 | 登录接口按 IP 限流（次/分钟） |
| `RETRIEVAL_CACHE_TTL` | 300 | 检索结果 Redis 缓存秒数，0=禁用 |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | admin / admin123456 | 初始化管理员 |

### 2.7 服务

| 变量 | 默认值 |
| :--- | :--- |
| `API_HOST` / `API_PORT` / `API_WORKERS` | 0.0.0.0 / 8000 / 1 |
| `WARMUP_MODELS` | 1（启动预热；设 0 跳过） |

---

## 3. 知识库构建

```bash
# 按角色目录批量入库（目录映射见 persona_seed.py 的 KNOWLEDGE_DIRS）
/home/dabaie/code/my_project/.venv/bin/python scripts/ingest_knowledge.py --all
/home/dabaie/code/my_project/.venv/bin/python scripts/ingest_knowledge.py --persona code=cbt_chen
/home/dabaie/code/my_project/.venv/bin/python scripts/ingest_knowledge.py --all --drop-existing   # 重建
# 可选策略：fixed / sentence / paragraph(默认) / heading / semantic
/home/dabaie/code/my_project/.venv/bin/python scripts/ingest_knowledge.py --all --strategy heading
```

低质过滤：块 token < 20 丢弃、完全重复去重、解析后字符 < 50 判失败（疑似扫描件）。

---

## 4. 数据库迁移（Alembic）

```bash
/home/dabaie/code/my_project/.venv/bin/alembic upgrade head
/home/dabaie/code/my_project/.venv/bin/alembic revision --autogenerate -m "描述"
```

连接串与 metadata 均来自 `src.core.config.settings`。日常建表也可由 ORM `create_all` 完成（lifespan / init_db）。

---

## 5. 日志

| 文件 | 内容 |
| :--- | :--- |
| `logs/app.log` | 全量（20MB × 5 滚动） |
| `logs/error.log` | 仅 ERROR |
| `logs/llm.log` | 大模型调用 |
| `logs/rag.log` | 检索链路 |
| `logs/uvicorn.out` | 服务标准输出 |

排查顺序：启动是否正常看 `uvicorn.out` + `app.log` 启动段 → 业务错误看 `error.log` → 检索/生成问题看 `rag.log` / `llm.log`。

---

## 6. 故障排查（FAQ）

**Q：聊天报"未配置 LLM_API_KEY"（HTTP 502）**
`.env` 中 `LLM_API_KEY` 为空，填入后重启。

**Q：GPU 显存不足**
BGE-M3 / Reranker 在 `cuda:0` 加载失败（OOM、驱动不匹配）会自动降级 CPU 并记录警告，功能正常但延迟上升。可显式设 `EMBEDDING_DEVICE=cpu`、`RERANKER_DEVICE=cpu` 并调小 batch。当前机器 RTX 5060 Laptop 8GB，与其他模型同时占用显存易触发降级。

**Q：服务起来了但接口报数据库错误**
启动时 MySQL/Milvus 初始化失败**只记 ERROR 不退出**（降级策略）。先看 `logs/app.log` 启动段与 `/health`。

**Q：Redis 挂了影响聊天吗**
不中断。短期记忆读写失败静默（会从 MySQL messages 回填），限流失败放行。

**Q：知识库检索不到**
1) 确认执行过 `ingest_knowledge.py`；2) `GET /api/v1/knowledge/stats` 看 chunks/milvus_total；3) `POST /api/v1/knowledge/search` 直测；4) 重排分普遍低于 0.35 时可下调 `SIMILARITY_THRESHOLD`。

**Q：扫描版 PDF 入库失败**
无文本层时 PyMuPDF/pdfplumber 均读不出文本，状态记 `failed`。需先 OCR（PaddleOCR 为可选依赖），将 txt 放入角色目录再入库。

**Q：ragas 不可导入**
ragas 0.4.3 与 langchain-community 0.4.2 冲突。评测自动回退内置 LLM-as-Judge（指标口径一致），不要为修 ragas 升级 langchain 全家桶。

---

## 7. 健康与验收清单

```bash
curl http://127.0.0.1:8000/health          # mysql/redis/milvus 均 true
curl http://127.0.0.1:8000/api/v1/personas # 三个角色
# 管理员登录后：
curl http://127.0.0.1:8000/api/v1/admin/monitor -H "Authorization: Bearer <admin_token>"
# 应能看到 embedding/reranker loaded=true、各角色 vectors
```

压测（Windows 侧 JMeter 5.6.3）：`D:\JMeter\apache-jmeter-5.6.3\apache-jmeter-5.6.3\bin`，见 `docs/07-testing.md`。
