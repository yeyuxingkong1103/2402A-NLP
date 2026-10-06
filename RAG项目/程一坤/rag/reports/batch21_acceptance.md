# 批次 21 验收报告

- 运行时间：2026-09-21
- 范围：状态常量下沉、e2e 脚本去硬编码口令、.env/.env.example 配置清理、starlette 显式声明、会话摘要实现（默认关）
- 边界遵守：未改 `docs/`；未新增依赖（starlette 属既有传递依赖的显式化）；未动 `data/`、`tests/` 既有内容（仅新增测试文件）；临时脚本一律放项目根，未进 `backend/`
- 总览：**485 passed**（原 468 + 新增 17）｜`check_services.py` 全绿（退出码 0）｜20 条评测与基线逐条一致

---

## 【1】状态常量下沉到数据层 ✅

### 改动

| 项 | 内容 |
|---|---|
| 新建 | `backend/app/db/version_status.py`：`PENDING_REVIEW` / `APPROVED` / `REJECTED` + `REVIEW_STATUSES` 派生元组；模块 docstring 声明"document_versions.version_status 取值的**唯一定义处**"并说明下沉理由 |
| 删除 | `backend/app/review/review_status.py` |
| 导入改造 | 7 个文件由 `app.review.review_status` / 裸字面量改为 `app.db.version_status` |

改造点（逐行）：

```
app/api/review.py:30            from app.db.version_status import PENDING_REVIEW
app/db/import_service.py:20     from app.db.version_status import PENDING_REVIEW
app/db/vector_index_service.py:9   from app.db.version_status import APPROVED
app/retrieval/keyword_search.py:25 from app.db.version_status import APPROVED
app/retrieval/vector_search.py:19  from app.db.version_status import APPROVED
app/review/review_detail.py:17     from app.db.version_status import PENDING_REVIEW
app/review/review_service.py:28    from app.db.version_status import (...)
```

### 验收证据

1. **裸字面量清零**（`app/` 下，排除定义处自身）：

```
$ grep -rn '"pending_review"\|"approved"\|"rejected"' app/ --include=*.py | grep -v app/db/version_status.py
（无输出，退出码 1）
```

2. **依赖方向正确**（反向依赖清零）：

```
$ grep -rn "import app.review\|from app.review\|from ..review" app/db/       → 无（退出码 1）
$ grep -rn "import app.review\|from app.review\|from ..review" app/retrieval/ → 无（退出码 1）
```

3. **测试**：`tests/test_review_status.py` 由 4 条扩至 **6 条**，新增两条守边界：

- `test_data_and_retrieval_layers_do_not_import_review_layer` —— 以 AST 扫描 `db/`、`retrieval/` 的 import，出现 review 层即失败
- `test_no_bare_status_literals_in_app_source` —— 扫描 `app/` 源码，`version_status.py` 之外出现裸字面量即失败

---

## 【2】e2e 脚本去硬编码口令 ✅

### 改动

| 项 | 内容 |
|---|---|
| 新建 | `scripts/_env.py`：`load_project_env(project_root, *, strip_proxy=True)` 通用 .env 加载器（同时剔除沙箱注入的代理变量）+ `mysql_config()`（口令缺失则显式报错，不静默降级） |
| 新建 | `scripts/e2e/README.md`：作用 / 前置条件 / 可重复性 / 清理说明 |
| 去明文 | `scripts/e2e/e2e_auth_persist.py`、`scripts/e2e/e2e_review_publish.py`、`scripts/e2e/e2e_long_term_memory.py`、`scripts/e2e/cleanup_e2e_probe_accounts.py`、`scripts/migrations/migrate_auth_users.py`、`scripts/migrations/migrate_long_term_memory_setting.py`、`scripts/migrations/migrate_review_status.py` 全部改从 `scripts/_env` 取连接信息 |

### 验收证据

1. **全目录硬编码清零**（11 个 .py 全部扫过）：

```
$ grep -rn "password\s*=\s*[\"'][^\"']*[\"']\|passwd\s*=\s*[\"']\|api_key\s*=\s*[\"']sk-\|123456\|admin123" scripts/ --include=*.py | grep -v "os.getenv\|environ\|argv"
（无输出，退出码 1）
```

残留匹配仅两类，均非明文口令：`os.getenv("MYSQL_PASSWORD", "")`（读取）、`password_hash="e2e-only"`（夹具占位值）。

2. **真实连库**（证明凭据来自 .env 且可用）：

```
mysql_config = {'host': '127.0.0.1', 'port': 3306, 'user': 'legal_rag', 'password': '***', 'database': 'legal_rag'}
真实连库成功 document_chunks = 1627
```

---

## 【3】.env.example 重写 + .env 无用键清理 ✅（附一处待裁决）

### .env 键级 diff（73 → 67）

**删除 10 个**（零读取且属"改了不生效也不报错"的高危项）：

```
APP_HOST、APP_PORT                              （uvicorn 启动参数，代码本就不该读）
MILVUS_USERNAME、MILVUS_PASSWORD、MILVUS_SECURE  （本机无鉴权；AutoDL 部署时再按需启）
CRAWLER_ENABLED、CRAWLER_NETWORK_ENABLED、CRAWLER_MINIMUM_INTERVAL_SECONDS、
CRAWLER_TIMEOUT_SECONDS、CRAWLER_USER_AGENT      （爬虫当前走调用方传参，配置化从未接线）
```

**新增 4 个**（均为代码已读取但 .env 缺登记）：

```
MILVUS_LONG_TERM_COLLECTION_NAME
LONG_TERM_MEMORY_DEDUP_THRESHOLD
LONG_TERM_MEMORY_TOP_K
SESSION_SUMMARY_ENABLED=false
```

备份：`.dev_bak21/.env.bak`（改动前快照）。

### .env.example 重写

173 行，按"代码实际读取的键"分 12 组（服务基础 / Redis / MySQL / Milvus / 检索 / Embedding / Reranker / LLM / 认证 / 记忆 / 邮件 / 存储），敏感项统一 `<your-xxx>` 占位。

**密钥脱敏验证**（真实 key 字符串在 example 中出现次数）：

```
$ grep -n "<两个真实 key 串>" backend/.env.example
（无输出，退出码 1）
```

### ⚠️ 待你裁决：存储类 5 键仍是"零读取"状态

全仓 grep 确认以下 5 键**没有任何代码消费者**（仅 `docs/部署文档.md` 与 `reports/batch20_cleanup_report.md` 提到）：

```
FILE_STORAGE_ROOT、DOCUMENT_STORAGE_PATH、PARSED_STORAGE_PATH、
BACKUP_STORAGE_PATH、MAX_UPLOAD_SIZE_BYTES
```

现状不一致：`.env` 里**仍是生效赋值**（未删、未注释），`.env.example` 里已注释化。按本批"消灭改了不生效也不报错的键"的口径，`.env` 侧应二选一：①删除；②保留但在行尾加注释标明"当前代码未读取，属 P2 上传/备份功能预留"。**我未擅自改动，等你定。**

---

## 【4】starlette 显式声明 ✅

```
backend/requirements.txt:2:starlette==1.3.1
```

与既有 11 项同锁 `==`。背景：批次 20 报告指出 `starlette` 被 `api/chat.py`、`errors.py`、`main.py` 直接 import 却仅作为 fastapi 传递依赖存在——本批将其显式化为直接依赖。

---

## 【5】会话摘要实现（默认关）✅

### 交付物

| 文件 | 说明 |
|---|---|
| `backend/app/memory/summary_policy.py` | 新建，29 行。触发/长度常量单点：`SUMMARY_MIN_NEW_MESSAGES=6`（=3 轮节流）、`SUMMARY_KEEP_RECENT_MESSAGES=6`、`SUMMARY_MAX_CHARS=300`、`SUMMARY_INPUT_MESSAGE_CHARS=400` |
| `backend/app/memory/summary_service.py` | 新建，205 行。`maybe_update_session_summary(...)` 写入侧主逻辑 |
| `backend/tests/test_session_summary.py` | 新建，15 条用例 |

### 与滑动窗口的关系（写进代码注释的关键结论）

短期记忆是 `List + ltrim(key, -20, -1)` 的硬截断，窗口满后**长度恒定**，无法由长度推断"又过了几轮"，因此自持会话级状态 `summary_state`（记录 `turns` / `summarized_turns`），节流判断用两者之差——与窗口长度解耦。摘要输入取 `messages[:-6]`（含部分仍在窗口内的消息），每次把旧摘要与新滑出内容**一起重算**，保证跨轮连续无断点。状态与消息/摘要共用同一 TTL，会话沉睡后一并消失。

### 硬要求达成

| 要求 | 实现 |
|---|---|
| 默认关 | `SESSION_SUMMARY_ENABLED` 默认 `false`；`summary_policy._resolve_enabled` 三级兜底（显式传参 → 配置 → false） |
| 关闭时行为逐字一致 | `maybe_update_session_summary` 首行即 return；`memory_hooks._session_summary` 关闭时**连 Redis 都不读**；`prompt_builder` 不注入该段 |
| 写入不阻塞回答 | `api/chat_persistence.persist_turn` 末尾 `threading.Thread(daemon)` 起后台线程，判断条件在 service 内 |
| LLM 失败保留旧摘要 | 失败路径只 `warning` 并返回 None，**不写 summary、不推进 summarized_turns**（下轮继续尝试） |
| 超长截断 | 超 300 字截断 + 一次 warning，不报错不丢弃 |
| 任何异常不外抛 | service 内全包 `try/except`，`chat_persistence` 再兜一层 |

### 验收证据

**① 真实提示词对比**（真实检索 + 真实 LLM，探针跑完自动清理 Redis 探针 key）：

```
SESSION_SUMMARY_ENABLED = true  | 含前情段 True  | 提示词长度 1777 字
  # 本次会话前情（早前轮次的压缩摘要，仅供理解上下文，不得替代法源清单）
  用户是企业HR，正在处理一起协商解除劳动合同的案子；已讨论经济补偿按工作年限……（106 字探针摘要）
  # 法源清单（本次检索结果）      ← 前情段位于法源清单之前
  引用条数：5；护栏：['citation_check_passed', 'guardrails_applied']

SESSION_SUMMARY_ENABLED = false | 含前情段 False | 提示词长度 1632 字
  首行直接是 "# 法源清单（本次检索结果）"   ← 与未引入摘要前一致
  引用条数：5；护栏：['citation_check_passed', 'guardrails_applied']

探针 key 已删除：True
```

**② 单元测试**：`tests/test_session_summary.py` 15 条，覆盖窗口未满不摘要、满窗首触发、3 轮节流、会话隔离与 TTL、LLM 失败保留旧摘要、空回复保留旧摘要、Redis 异常不外抛、超长截断、开关关闭零读写、关闭时提示词无该段、开启时注入位置在法源清单前、持久化层开关的线程触发与否。

**③ 全量回归**：**485 passed**。

---

## 额外发现与修复：批次 19 拆分的跨目录旧导入残留 ✅

本批跑评测脚本时暴露：批次 19 拆分 `chat/service.py` → 新增 `bootstrap.py` 时，只 grep 了 `backend/`，遗漏 `evaluation/` 与 `scripts/` 下的调用方，导致运行时 `ImportError`。

修复 3 处：

```
evaluation/run_eval.py:503          build_default_chat_service → app.chat.bootstrap
evaluation/diagnose_rerank.py        fusion / parent_collapse 两函数的新归属
scripts/e2e/e2e_long_term_memory.py:82  同上
```

并新增 `b21_import_check.py` 做**全仓**（不只 backend/）AST 级 import 完整性校验：

```
扫描文件数: 85（backend/app 由 pytest 覆盖，此处跳过）
✅ 全部 app.* 导入均可解析（模块存在 + 符号存在）
```

---

## 评测对拍：20 条快速回归

摘要默认关，生产口径不变；为验证"零影响"，跑 20 条与基线对拍：

| 项 | 结果 |
|---|---|
| 评测集 | `data/evaluation/eval_set_v1.jsonl` 前 20 条（direct×16 + cross×4） |
| 检索指标 | 20 条 `golden_rank / hit_at_5 / mrr10` 与基线 `eval_20260920_194916_baseline95` **逐条完全一致（20/20）** |
| 本次汇总 | Recall@5 = 1.0000（20/20）、MRR@10 = 0.7825、引用总数 111、越界引用 0、引用正确率 1.0000 |
| 拒答准确率 | 前 20 条无拒答题，分母为 0，该值 0.0000 无意义（不作结论） |
| 新报告 | `reports/eval_20260921_105356_b21_summary_off.{json,md}` |

结论：检索链路**零改动**（摘要只作用于提示词构造之后），引用越界为 0。

> 若你还要"开 vs 关"的 20 条质量对拍（验证开启后引用正确率/拒答准确率不劣化），我另跑一轮即可——那是唯一还没取证的维度（需要 20 条 × 2 次真实 LLM）。

---

## 待裁决项汇总

| # | 事项 | 我的建议 |
|---|---|---|
| 1 | `.env` 里存储类 5 键仍生效但零读取 | 保留 + 行尾注释标明预留（与 example 的注释化对齐）；或直接删 |
| 2 | 本批临时脚本 `b21_*.py`（4 个）+ `.dev_bak21/` | 按前几批惯例（批次 19 结束时清理了全部 b18/b19 脚本与备份）应清理；但若你要留作复现证据，我保留 |
| 3 | 会话摘要"开 vs 关"20 条质量对拍 | 需要则跑；不需要则本批评测维度已闭合 |

---

## 可复现命令

```bash
# 全量测试（--basetemp 必须 Windows 风格路径，否则 tmp_path 全部 setup 失败）
cd backend && PYTHONPATH= python -m pytest tests -q --no-header -p no:cacheprovider \
  --basetemp="C:/Users/<user>/AppData/Local/Temp/pytest_b21"

# 服务体检（本地，不调外部 API）
python scripts/check_services.py

# 全仓 import 完整性
python b21_import_check.py

# 提示词对比探针
python b21_prompt_probe.py

# 20 条评测
python evaluation/run_eval.py --limit 20 --tag b21_summary_off
```
