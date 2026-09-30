# 批次 20 验收报告：执行《法律RAG-项目清理任务书》+ 修订 A/B

> 日期：2026-09-21 ｜ 项目：`C:\Users\92842\Desktop\rag`
> 基线：批次 19 收尾 413 passed / 全 app 无 >300 行文件 / backend 根白名单
> 结果：**468 passed**（413 + 新增测试 55）、`check_services.py` 全绿退出码 0、项目根与 backend 根均达目标形态
> 全文只做"报告"的两项：**2.4 会话摘要方案**（见 `reports/batch20_session_summary_plan.md`）与 **2.5 依赖配置复核**（本文第六节）

---

## 零、一句话结论

| 项 | 状态 |
|---|---|
| 修订 A：review_status 单一来源 | ✅ 完成（`"pending_review"` 在 app/ 下只剩常量模块一处） |
| 修订 B：残留文件处置 | ⚠️ **部分完成**：删除 8 项、迁移 4 项 ✅；**5 个 e2e/清理脚本的幂等判定被内容授权拦截，未删除，已整体移入 `scripts/e2e/`** |
| 2.2 删死代码 | ⚠️ 删 1 处 ✅；**另 1 处是任务书误报，拒绝删（证据见第三节）** |
| 2.3 allowed_email 三份合一 | ✅ 完成（60 组行为比对全一致 + 新增 51 条测试） |
| 2.4 会话摘要 | ✅ 方案已交（未写代码，按任务书要求） |
| 2.5 依赖与配置复核 | ✅ 报告已出（只报告未改动） |
| 第三节 目录差异表 | ✅ 见第七节 |
| 终验 | ✅ 468 passed / check_services 全绿 / 目录达标 |

---

## 一、修订 A：review_status 单一来源

**做了什么**

| 动作 | 文件 |
|---|---|
| **新增**状态常量模块（只放常量，无 import、无逻辑） | `app/review/review_status.py`（`PENDING_REVIEW` / `APPROVED` / `REJECTED` / `REVIEW_STATUSES`） |
| 删本地字面量，改从常量模块导入 | `app/review/review_service.py`（原 `STATUS_PENDING/APPROVED/REJECTED` 四个常量 → import）、`app/review/review_detail.py`（原本地 `STATUS_PENDING` → import） |
| 顺带清掉两处**裸字面量**（否则 grep 验收过不了） | `app/api/review.py:93`（列表接口默认值 `Query(default="pending_review")` → `PENDING_REVIEW`）、`app/db/import_service.py:117`（新版本初始化 `version_status="pending_review"` → `PENDING_REVIEW`） |
| 新增测试 | `tests/test_review_status.py`（4 条） |

**证据 1 —— 字面量收敛（任务的验收口径）**

```
$ grep -rn '"pending_review"\|"approved"\|"rejected"' --include="*.py" app/
app/review/review_status.py:15:PENDING_REVIEW = "pending_review"
app/review/review_status.py:18:APPROVED = "approved"
app/review/review_status.py:21:REJECTED = "rejected"
```
→ `"pending_review"` 在 `app/` 下**只出现在常量模块**；`approved` / `rejected` 也只剩常量模块 + 下面 3 处**未动**的既有引用（见"遗留待裁决"）。

**证据 2 —— 常量与库中实际取值一致（真实库查询）**

```
库中 document_versions.version_status 实际取值：
  'approved'           行数=11
常量模块词表 : ('pending_review', 'approved', 'rejected')
库中出现过的值: ('approved',)
库取值 ⊆ 词表 : PASS
```

**证据 3 —— 测试锁定（防将来改状态名漏改）** `tests/test_review_status.py`：
1. 字面量值锁定（改名即破坏与既有 DB 行/接口的兼容）；
2. 四个模块（含子类）的常量是**同一个对象**（`is` 判定），且旧名 `STATUS_PENDING` 不得复活；
3. 6.3 列表接口的 `Query(default=...)` 必须取自词表；
4. 落库往返：新导入版本写进 `document_versions.version_status` 的就是词表值。
```
$ pytest tests/test_review_status.py -q  →  4 passed
```

**遗留待裁决（我未擅自扩范围）**：还有 3 处 `"approved"` 裸字面量：
`app/db/vector_index_service.py:54`、`app/retrieval/keyword_search.py:127`、`app/retrieval/vector_search.py:224`。
收敛它们需要 **db / retrieval 反向 import `app.review.review_status`**，方向反了（这两层与 review 是平级甚至更下层）。同理，`app/db/import_service.py` 现在也持有 `db → review` 的常量依赖（无循环、无逻辑，仅为一个字符串）。
**建议**：若你要彻底单一来源，把词表下沉到中立位置（如 `app/core/review_status.py` 或 `app/db/`），我再把 5 处一起改；否则就维持现状（review 域内单一来源 + 三处跨域字面量）。**等你裁决。**

---

## 二、修订 B：项目根残留文件处置

### 2.1 删除（8 项，删前逐文件 grep 零引用）

| 文件 | grep 结果（排除 `reports/`、`.workbuddy/`、自身后） |
|---|---|
| `b16a_acceptance.py` | 外部引用 0 |
| `b16b_backend.py` | 外部引用 0 |
| `b16b_prep.py` | 外部引用 0 |
| `b16b_restore.py` | 外部引用 0 |
| `b17_fingerprint.py` | 外部引用 0 |
| `frontend_build.log` | 外部引用 0 |
| `.pytest_cache/` | 缓存目录 |
| **`backend/app/ingest/chunker.py.broken`** | **任务书未列，我扫到的额外残留**（9/18 的中间态备份，零引用；因非 `.py` 后缀逃过了所有行数扫描）→ 一并删除 |
| `backend/tmp_demo_night.py` | 批次 19 已删（`demo_evidence/scripts/` 有同份拷贝） |

### 2.2 迁移到 `scripts/migrations/`（4 个 + README）

```
scripts/migrations/
├── README.md                              ← 新增：做了什么 / 执行顺序 / 是否已执行
├── migrate_auth_users.py                  （内容零改动，无路径依赖）
├── migrate_long_term_memory_setting.py    （内容零改动）
├── migrate_review_status.py               （内容零改动）
└── backfill_law_dates.py                  ⚠️ 唯一改动：路径锚点 parent → parents[2]
```

**为什么动了 `backfill_law_dates.py` 一行**：它原本用 `Path(__file__).resolve().parent` 定位项目根（`PROJECT_ROOT`），
移入子目录后该锚点会指向 `scripts/migrations`，`.env` / `data/` / `reports/` 全部找不到。已改为 `parents[2]`（= `rag/`），并加注释说明。其余三个脚本无路径依赖、内容零改动。

**验证**：`py_compile` 四个文件全通过；`backfill_law_dates.py` 预览模式实跑成功（打印各法规"无变化"并写出 `reports/batch10_law_dates.md`），证明路径修复有效。

**迁移是否已对当前库执行过（真实库证据）**：

| 迁移 | 当前库证据 | 结论 |
|---|---|---|
| `migrate_auth_users.py` | `users` 列 = `id, user_key, email, password_hash, is_admin, is_active, created_at, updated_at, long_term_memory_enabled`（**无 `username`**） | ✅ 已执行 |
| `migrate_long_term_memory_setting.py` | `users.long_term_memory_enabled` 列存在 | ✅ 已执行 |
| `migrate_review_status.py` | 留痕三列齐全；`version_status` 分布 = `approved: 11`（无 `new` 残留） | ✅ 已执行 |
| `backfill_law_dates.py` | `law_versions` 11 行：发布日期 11、生效日期 10（缺的 1 行是典型案例页面本身未写日期，属"待人工补录"） | ✅ 已执行 |

### 2.3 e2e / 清理脚本：⚠️ 未完成（阻塞原因如下）

这 5 个文件的**内容读取被内容授权机制拦截**（连续两次读取均返回"授权超时、内容已扣留"），
因此**无法按你要求逐文件判定"是否硬编码账号 / 是否依赖一次性数据 / 是否幂等"**。为不误删，我按最保守方式处置：

```
scripts/e2e/
├── e2e_auth_persist.py
├── e2e_create_admin.py
├── e2e_long_term_memory.py
├── e2e_review_publish.py
└── cleanup_e2e_probe_accounts.py
```
**未删除任何一个**（删除不可逆，宁留勿误删）。旁证（不是结论）：先前 grep 时片段显示 `e2e_auth_persist.py` 的账号带 `_{_RUN}` 运行后缀，像是"每次运行生成唯一账号"的可重复设计。

**需要你做的**：授权我读取这 5 个文件（或你自己看一眼），我立刻补做幂等判定 → 可重复的留在 `scripts/e2e/`、一次性的删除。判定依据会逐文件一行写进报告。

---

## 三、2.2 删真死代码

### ① `app/ingest/legal_meta.py` —— ✅ 已删（零引用成立）

```
$ grep -rn "ingest.legal_meta\|from .legal_meta" --include="*.py" . ../
（仅命中 legal_meta.py 自身 docstring 与 docs 的说明文字，无任何 import）
```
配套：`app/db/law_models.py:127` 的注释原写"（app/ingest/legal_meta.py）"，已改为实际模块 `app/ingest/article_number_rules.py`。

### ② `app/retrieval/keyword_search.py::build_keyword_searcher_rows` —— ❌ **任务书误报，拒绝删除**

任务书写"全项目（含 tests/）只有定义处出现，零调用"。**实际 grep 结果**：

```
app/retrieval/keyword_search.py:235:def build_keyword_searcher_rows(     ← 定义
tests/test_keyword_search_filters.py:5:   from app.retrieval.keyword_search import KeywordSearcher, build_keyword_searcher_rows
tests/test_keyword_search_filters.py:9/61/88:  searcher = build_keyword_searcher_rows(
tests/test_synonym_expansion.py:15:      from app.retrieval.keyword_search import build_keyword_searcher_rows
tests/test_synonym_expansion.py:155/253/268:   searcher = build_keyword_searcher_rows(
```
**6 处测试调用**，函数自己的 docstring 也写明"用内存行构造检索器，**供纯逻辑测试复用**"。
删它 → 两个测试文件直接红，且要改 `tests/`（违反任务书硬性约束 3"tests/ 内容一律不动"与约束 1"删前必须确认零引用含 tests/"）。
**结论：不删，等你裁决**（可选：保留现状 / 允许我把它移成测试夹具并同步改那两个测试文件）。

### 关于"误报说明"那条 ✅ 已遵守

框架回调（FastAPI 路由函数、Pydantic 校验器、HTMLParser 回调、property）一律未动。

---

## 四、2.3 allowed_email 三份合一

**做了什么**（`app/auth/schemas.py`）：
- 保留唯一实现 `validate_email_domain`（模块级函数，补了"为什么必须归一化"的注释）；
- 新增模块级 `_reject_disallowed_email`（供字段校验器引用）；
- **定义一次 `_allowed_email_validator = field_validator("email")(...)`，三个模型引用同一对象**，删掉三份逐字相同的 `allowed_email` 函数体。

**过程中的技术发现**（写进代码注释了）：`field_validator(...)(classmethod(fn))` 这种写法**被第二个类复用时**会触发
`PydanticUserError: Unrecognized field validator function signature ... () -> str`（pydantic 2.13）。
所以共享代理必须是**裸函数**，不能套 `classmethod`。这是本任务唯一值得一提的坑。

**证据 1 —— 行为完全一致（60 组输入矩阵比对）**
把改动前的文件另存为独立模块，与改动后逐条对拍（4 个模型 × 15 个输入 = 60 组）：

```
输入矩阵：4 个模型 × 15 个输入 = 60 组比对
模型： CodeRequest, RegisterRequest, LoginRequest, ResetPasswordRequest
  CodeRequest(email='  Foo@QQ.com  ') -> {"ok": true, "email": "foo@qq.com"}
  LoginRequest(email='x@gmail.com')   -> {"ok": false, "errors": [{"type": "value_error", "loc": ["email"], "msg": "Value error, 仅支持 QQ 邮箱或 Foxmail 邮箱"}]}
  RegisterRequest(email=123)          -> {"ok": false, "errors": [{"type": "string_type", "loc": ["email"], "msg": "Input should be a valid string"}]}
三个模型（含子类）的 allowed_email 校验器对象去重后数量： 1 (1 = 同一实现)
旧实现三处函数体逐字相同： True
BEHAVIOR DIFF: PASS —— 归一化结果与错误 type/msg/loc 逐条一致
```

**证据 2 —— 新增测试** `tests/test_email_validator.py`（51 条）：
非法邮箱 ×4 模型拒绝且错误信息逐字一致；合法邮箱归一化；非字符串输入仍 `string_type`；
共享实现可直调；**结构锁定**（四模型引用同一校验器对象，防止将来又抄成多份）。
```
$ pytest tests/test_email_validator.py tests/test_auth_api.py tests/test_create_admin.py -q  →  64 passed
```

**顺带登记的既存边界（未改，仅测试记录）**：现有实现**只校验域名、不校验本地部分**，所以 `"@qq.com"` 会被放行。
本任务边界是"行为零变化"，故故意保留，并用 `test_local_part_is_not_validated_legacy_behavior` 显式登记成未来决策点。

---

## 五、2.4 会话摘要：方案已交（未写代码）

详见 **`reports/batch20_session_summary_plan.md`**。要点：

- **不删**。现状是"实现了但没接线"：`read_summary` / `write_summary` 全项目只有 tests 调用；而 `messages` 窗口是**滑动 + 硬截断**（`ltrim -20 -1`），第 11 轮起最早的对话**永久丢失**，摘要正是补这个洞。
- **写**：满窗（10 轮）后每累积 ≥3 轮增量更新一次；落在 API 持久化层（`chat_persistence.persist_turn` 之后），后台线程，LLM 失败**保留旧摘要不覆盖**，任何异常只 warning。
- **读**：①检索侧（改写器在消息不足时用摘要补上下文）②生成侧（提示词新增"# 本次会话前情"段，插在长期记忆之后、法源清单之前，上限 300 字）。建议**先做 ①，② 需重跑评测确认引用合规**。
- **成本**：30 轮会话约 7 次短调用（输入 ≈2.3k / 输出 ≈0.5k tokens），自建 Qwen2.5-14B 边际成本≈0，后台执行不增加等待。
- **验收**：触发/节流/失败单测 + 贴出注入后的提示词真实内容 + "同一省略式追问"有/无摘要的改写与 top5 对比 + `redis-cli get ...:summary` 原文。
- **开关默认关**（`SESSION_SUMMARY_ENABLED=false`），关闭时行为与现状逐字一致。

---

## 六、2.5 依赖与配置复核（只报告，按任务书未擅自改）

### ① requirements.txt：无多余包、无漏包 ✅

```
import fastapi      -> ✅ 已声明(fastapi==0.138.1)
import httpx        -> ✅ 已声明(httpx==0.28.1)
import jieba        -> ✅ 已声明(jieba==0.42.1)
import pydantic     -> ✅ 已声明(pydantic==2.13.4)
import pymilvus     -> ✅ 已声明(pymilvus==2.5.18)
import pytest       -> ✅ 已声明(pytest==8.4.2)
import rank_bm25    -> ✅ 已声明(rank-bm25==0.2.2)
import redis        -> ✅ 已声明(redis==5.3.1)
import sqlalchemy   -> ✅ 已声明(SQLAlchemy==2.0.34)
import starlette    -> ⚠️ fastapi 的依赖（未显式声明，属传递依赖）
未被 import 的声明（隐式依赖，人工确认）：['PyMySQL==1.1.1', 'uvicorn[standard]==0.49.0']
```
→ 与任务书已知结论一致：**11 个包全部需要**；`uvicorn[standard]`（启动服务）与 `PyMySQL`（SQLAlchemy 连接串驱动）不出现 `import` 属正常。
**唯一新发现**：`starlette` 被代码直接 import（`api/chat.py`、`errors.py`、`main.py`），但未在 requirements 显式声明 —— 它由 fastapi 强制带入，**不算漏包**，只是"直接依赖未显式化"，要不要显式声明请你定。

### ② `.env` 与 `config.py`：无"读不到的键"造成故障，但有 28 个键**当前代码确实不读** ⚠️

`.env` 共 73 个键，代码实际读取 48 个。以下 28 个键**全项目零读取**：

```
部署类：APP_HOST、APP_PORT（uvicorn 启动参数，代码本就不该读）
Milvus 鉴权：MILVUS_USERNAME、MILVUS_PASSWORD、MILVUS_SECURE（本机无鉴权，AutoDL 部署时启用）
P1 未启用：MINERU_*（8 个）、QWEN_VL_*（5 个）
采集/存储类：FILE_STORAGE_ROOT、DOCUMENT_STORAGE_PATH、PARSED_STORAGE_PATH、
             BACKUP_STORAGE_PATH、MAX_UPLOAD_SIZE_BYTES、
             CRAWLER_USER_AGENT、CRAWLER_TIMEOUT_SECONDS、
             CRAWLER_MINIMUM_INTERVAL_SECONDS、CRAWLER_ENABLED、CRAWLER_NETWORK_ENABLED
```
判断：**不是配置错误，而是"超前登记"** —— 爬虫当前用**调用方传参**（`app/crawler/requester.py:64` 的 `user_agent`/`timeout_seconds` 由调用者注入），
`app/crawler/` 目录内只有 `collect_labor_law.py:128` 读一个目录环境变量，所以 `CRAWLER_*` 这一组今天确实没有消费者。
**风险提示**：改爬虫参数时改 `.env` 是**无效的**（会被忽略，且不会报错）—— 这正是"读不到的键"最容易踩的坑。

**新增配置接线抽查（全部已接上）**：

```
SYNONYM_EXPANSION_ENABLED            .env=有  读取点=['app/core/config.py']
SYNONYM_EXPANSION_WEIGHT             .env=有  读取点=['app/core/config.py']
SYNONYM_TABLE_PATH                   .env=有  读取点=['app/core/config.py']
EMBEDDING_RETRY_ATTEMPTS             .env=有  读取点=['app/models/embedding.py']
EMBEDDING_RETRY_BACKOFF_SECONDS      .env=有  读取点=['app/models/embedding.py']
EMBEDDING_RETRY_TOTAL_BUDGET_SECONDS .env=有  读取点=['app/models/embedding.py']
RECALL_VECTOR_LIMIT / RECALL_KEYWORD_LIMIT / RERANK_CANDIDATE_LIMIT  .env=有  读取点=['app/core/config.py']
REFUSAL_MIN_VECTOR_SCORE             .env=有  读取点=['app/core/config.py']
```

### ③ `.env.example` 严重滞后 ❌（只报告，未改）

`backend/.env.example` **只有 5 个键**（`APP_NAME / ENVIRONMENT / LOG_LEVEL / REDIS_URL / SESSION_TTL_SECONDS`），
而代码实际要读 **44 个**键（`DATABASE_URL`、全部 `MYSQL_*`、`MILVUS_*`、`EMBEDDING_*`（含 3 个 `EMBEDDING_RETRY_*`）、
`RERANKER_*`、`LLM_*`、`SMTP_*`、`SYNONYM_*`、`RECALL_*`、`RERANK_CANDIDATE_LIMIT`、`REFUSAL_MIN_VECTOR_SCORE`、
`MILVUS_LONG_TERM_COLLECTION_NAME`、`LONG_TERM_MEMORY_*` 等）全部缺失。
另外 `.env.example` 里没有"代码不读的键"（干净）。
**建议**：按 `.env` 的键名顺序重写 `.env.example`（值全留空/占位），并在 3 个 `MINERU_*` / `QWEN_VL_*` 段落标注"本阶段未启用"。
**我没有改它**（任务书写明 2.5"只报告，不擅自改"）——要改的话给我一句话即可。

---

## 七、第三节：目录结构 vs 文档目录约定的差异表

方法：`find app -name '*.py'` 取 basename 集合，与 `docs/目录与命名约定.md` 目录树段落做集合比对（97 个文档条目 vs 98 个实际文件）。

### 【实际有、文档没有】

| 项 | 说明 |
|---|---|
| `app/review/review_status.py` | **本批新增**（审核状态词表单一来源）。请在 review/ 段落加一行 |
| `__init__.py`（各包） | 文档树一贯不列包初始化文件，**属惯例，不算差异** |
| 根层级：`scripts/`（含 `migrations/`、`e2e/`）、`frontend/`、`demo_evidence/`、`CLAUDE.md`、`README.md`、`.env` | 文档只树化 `backend/`，`data/` `evaluation/` `reports/` 用文字块描述；根层级这几项**没有任何条目**。第 337 行只有一条"长期运维脚本 → scripts/，迁移脚本放 scripts/migrations/"的原则，没有目录树 |

### 【文档有、实际没有】

| 项 | 说明 |
|---|---|
| `app/ingest/legal_meta.py` | 第 58 行（"兼容 re-export 入口（待清理，见批次 20）"）与第 166 行（"legal_meta.py 仅保留 re-export"）→ **本批已按任务书删除**，请同步删掉这两处 |
| 第 191–192 行的命名禁止项（`legal_meta` / `legal_metadata` / `legal_metadata_extractor` 不得并存） | 规则本身仍有效（作为命名约束保留即可），但"三个同名文件并存"的现状已不存在 |

### 【名字/位置不一致】

**无**。文档目录树中列出的 97 个文件名，除去已删的 `legal_meta.py` 外**全部与实际情况逐一对上**（含批次 19 拆出的 19 个文件、5 处实体定义分置）。文档与代码高度一致。

---

## 八、终验

### ① 全量测试（本批跑 4 次，无一次回落）

```
批次20 起点（修订A 前）        : 413 passed, 5 warnings
修订A 后                      : 417 passed（+4：test_review_status）
2.2 死代码 + 修订B 后          : 417 passed
2.3 校验器合一后（终态）        : 468 passed, 6 warnings in 12.64s（+51：test_email_validator）
```

### ② 服务健康自检

```
$ python scripts/check_services.py
✅ 容器 my-redis / milvus-standalone
✅ Redis ping PONG
✅ MySQL 连接（127.0.0.1:3306/legal_rag）documents=11  document_chunks=1627  law_versions=11  document_versions=11
✅ Milvus 集合 legal_documents 存在
✅ 向量数一致性  Milvus=1627 == MySQL.document_chunks=1627
✅ 版本状态统计  approved=11（其它=0）
=== 结果 === ✅ 全部通过（退出码 0）
```

### ③ demo_ask 冒烟（改动涉及 chat/auth/db 后确认主链路无恙）

```
【检索统计】向量召回 11 条 / 关键词召回 20 条 / 融合后 27 条 / 重排后 5 条
【引用法源】共 5 条（劳动合同法 47 条、实施条例 27 条 …）
```

### ④ 文件规模（未被本批破坏）

```
$ find app -name '*.py' -exec wc -l {} + | awk '$1 > 300'
 13742 total          ← 只有总行数，无文件行：全 app/ 无 >300 行文件
```

### ⑤ 目录终态（ls 证明）

```
项目根：
  .claude  .env  .idea  .superpowers  .workbuddy  CLAUDE.md  README.md
  backend/  data/  demo_evidence/  docs/  evaluation/  frontend/  reports/  scripts/

backend/ 根（严格白名单）：
  .env.example  app  pyproject.toml  pytest.ini  requirements.txt  tests
```
> 说明：`.claude/`、`.idea/`、`.superpowers/`、`.workbuddy/` 是工具/IDE 生成的元数据目录（非源码产物），`CLAUDE.md`、`.env` 是配置/约定文件，均在允许之列。本批新增的所有临时脚本（`b20_*.py`）与 `.pytest_cache/` **已全部删除**。

---

## 九、未完成项与需你裁决（汇总）

| # | 项 | 状态 | 需要你做什么 |
|---|---|---|---|
| 1 | 5 个 e2e/清理脚本的幂等判定 | ⛔ 内容读取被授权拦截 | 授权我读取，或你自查后告知哪些是一次性的 → 我删除，其余留 `scripts/e2e/` |
| 2 | `build_keyword_searcher_rows` | ⛔ 任务书误报（6 处测试调用） | 裁决：保留 / 允许改这两个测试文件后移入测试夹具 |
| 3 | 3 处跨域 `"approved"` 字面量 | ⚠️ 未动（避免 db/retrieval → review 反向依赖） | 裁决：下沉词表到中立位置（我再改 5 处）/ 维持现状 |
| 4 | `db/import_service.py` 的 `db → review` 常量依赖 | ⚠️ 已引入（为满足 grep 验收） | 同上，一并裁决 |
| 5 | `.env.example` 缺 44 个键 | ⚠️ 只报告未改（任务书要求） | 说一声我就按 `.env` 键名重写示例文件 |
| 6 | `starlette` 直接 import 未显式声明 | 📝 提示 | 是否需要显式加进 requirements（当前不影响运行） |
| 7 | 文档需同步处 | 📝 已给出具体行号 | 第 58、166 行删 `legal_meta.py`；review/ 段加 `review_status.py`；根层级补 `scripts/` 目录树 |
