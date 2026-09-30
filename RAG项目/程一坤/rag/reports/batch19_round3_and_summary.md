# 批次 19 第 3 轮验收 + 批次 19 总汇总

日期：2026-09-21
第 3 轮范围：⑦ `app/ingest/law_metadata_extractor.py`(309) ⑧ `app/ingest/offline_ingest.py`(308) ⑨ `app/review/review_service.py`(301)

---

## 一、第 3 轮验收输出

### a) AST 零改动比对（**本轮无任何白名单差异**：纯整对象搬家）
```
[law_metadata] PASS  对象数 原 12 → 新 12（顶层常量 5 → 5）
    缺失: 无
    不一致: 无
    新增: 无
[offline_ingest] PASS  对象数 原 8 → 新 8（顶层常量 0 → 0）
    缺失: 无
    不一致: 无
    新增: 无
[review_service] PASS  对象数 原 8 → 新 8（顶层常量 7 → 8）
    缺失: 无
    不一致: 无
    新增: 无

[状态字面量] review_detail.STATUS_PENDING == review_service.STATUS_PENDING: 一致
OVERALL: PASS
```

### b) 行数与注释覆盖率（新文件全部 ≥30% 且 ≤250 行）
```
law_metadata_extractor.py       198 行  docstring 32 + # 注释 15 = 覆盖率 23.7%   （瘦身后的原文件）
law_date_rules.py               133 行  docstring 33 + # 注释 14 = 覆盖率 35.3%   ← 新增
offline_ingest.py               217 行  docstring  9 + # 注释 10 = 覆盖率  8.8%   （瘦身后的原文件）
title_normalizer.py             102 行  docstring 19 + # 注释 14 = 覆盖率 32.4%   ← 新增
review_service.py               218 行  docstring 41 + # 注释 15 = 覆盖率 25.7%   （瘦身后的原文件）
review_detail.py                116 行  docstring 20 + # 注释 16 = 覆盖率 31.0%   ← 新增
```

### c) 全量测试
```
413 passed, 6 warnings in 12.69s
```

### d) demo_ask "经济补偿怎么算"
```
【检索统计】向量召回: 11 条 / 关键词召回: 20 条 / 融合后: 27 条 / 重排后: 5 条
【引用法源】共 5 条（劳动合同法 47/97 条、实施条例 27/25/10 条）
```

### e) 调用方 import 同步（不留兼容垫片）
- `app/db/legal_metadata_writer.py` 仍从 `law_metadata_extractor` 取 `extract_legal_metadata`（未动）
- `tests/test_law_metadata_dates.py`：日期两函数改从 `app.ingest.law_date_rules` 导入
- `tests/test_document_title_extraction.py`：`_extract_document_title` 改从 `app.ingest.title_normalizer` 导入
- `app/api/review.py`：`ReviewDetailNotFound` / `get_review_document_detail` 改从 `app.review.review_detail` 导入
- 局部回归：`tests/test_document_title_extraction.py` + `tests/test_law_metadata_dates.py` → `16 passed`

### f) 一处需你知晓的设计取舍（⑨）
`review_detail.py` 需要"待审核"状态字面量，而 `review_service.py` 持有状态词表。
若反向导入会构成 `review_service ↔ review_detail` 循环导入，故 `review_detail.py` 本地持有
`STATUS_PENDING = "pending_review"`（带注释说明同源），校验脚本含**同值断言**（已一致）。
若你更希望单一来源，可另开一个小批次把状态词表抽到 `review_status.py` —— 我没有擅自加文件。

---

## 二、批次 19 总汇总

### 2.1 三轮共新增 19 个文件（→ 职责一句话）

**第 1 轮（严重超标，11 个）**
| 新增文件 | 行数 | 职责一句话 |
|---|---|---|
| `chat/result.py` | 42 | ChatResult 结果类型与法条摘录组装 |
| `chat/memory_hooks.py` | 101 | 长期记忆读写钩子 Mixin |
| `chat/streaming.py` | 157 | 流式问答 chat_stream（replace 事件护栏衔接） |
| `chat/bootstrap.py` | 64 | build_default_chat_service 依赖装配 |
| `memory/memory_schema.py` | 94 | Milvus 长期记忆集合管理（常量/记录结构/建集合） |
| `memory/memory_summarize.py` | 49 | 写入前 LLM 摘要生成 |
| `memory/memory_settings.py` | 47 | 用户级长期记忆开关（MySQL） |
| `ingest/article_splitter.py` | 60 | 条文边界切分 |
| `ingest/paragraph_items.py` | 89 | 款/项识别 + 中文项号转数字 |
| `ingest/chunk_text.py` | 50 | 文本工具（归一化/检索文本/文档标识，防循环导入） |
| `ingest/chunk_fingerprint.py` | 45 | 切块指纹与 CHUNKER_VERSION |

**第 2 轮（中度超标，5 个）**
| 新增文件 | 行数 | 职责一句话 |
|---|---|---|
| `db/document_models.py` | 164 | 文档域实体 Document/DocumentVersion/DocumentChunk/CrawlRecord/ImportRecord |
| `db/law_models.py` | 175 | 法规域实体 Law/LawVersion/Article |
| `db/user_models.py` | 48 | 用户域实体 User |
| `retrieval/item_convert.py` | 109 | RankedItem 转换纯函数（向量结果转换/重排重建/锚点文本） |
| `db/import_version_status.py` | 83 | 数据包增量判定 new/unchanged/updated |

**第 3 轮（轻微超标，3 个）**
| 新增文件 | 行数 | 职责一句话 |
|---|---|---|
| `ingest/law_date_rules.py` | 133 | 法规日期识别规则（施行/公布/修订语境）+ HTML 文本预处理 |
| `ingest/title_normalizer.py` | 102 | 文档标题归一化（站点后缀剥离 + 站点名黑名单 + 可疑判定） |
| `review/review_detail.py` | 116 | 审核详情只读查询（版本定位 + 元数据 + 分块预览截断） |

**被改写的原文件（骨架瘦身，非新增）**：`chat/service.py` 516→217、`memory/long_term.py` 391→232、
`ingest/chunker.py` 383→181、`db/sql_models.py` 324→38（改统一导入入口）、`retrieval/service.py` 327→266、
`db/import_service.py` 318→277、`ingest/law_metadata_extractor.py` 309→198、
`ingest/offline_ingest.py` 308→217、`review/review_service.py` 301→218；
另 `db/base.py` 新增 `utc_now`（共用件）、`db/chat_models.py` 补"实体分置"docstring 注明（零逻辑）。

### 2.2 批次 19 删除的临时产物（全部已删）
| 项目 | 说明 |
|---|---|
| `.bak_b19r2/`（4 个文件） | 第 2 轮拆分前备份 |
| `.bak_b19r3/`（3 个文件） | 第 3 轮拆分前备份 |
| `b18_fusion_original.py` / `b18_verify.py` | 批次 18 遗留 |
| `b19_orig_chat_service.py` / `b19_orig_long_term.py` / `b19_orig_chunker.py` | 第 1 轮 AST 基线 |
| `b19_verify.py` / `b19r2_verify.py` / `b19r3_verify.py` | 第 1/2/3 轮校验脚本 |
| `b19r2_describe.py` | 第 2 轮 describe 校验脚本 |
| `backend/.pytest_cache/` | pytest 缓存目录（不在 backend 根白名单内） |
| `backend/tmp_demo_night.py` | 遗留演示脚本（`demo_evidence/scripts/` 已有同份拷贝，删除无信息损失） |

残留检查：`ls | grep -E "^b1[89]|b19|bak_b19"` → **无输出**

### 2.3 目录终检
```
=== backend/ 根（只允许 app/ tests/ requirements.txt pyproject.toml pytest.ini .env.example）===
.  ..  .env.example  app  pyproject.toml  pytest.ini  requirements.txt  tests
（已无 tmp_demo_night.py 与 .pytest_cache）

=== find app -name '*.py' -exec wc -l {} + | awk '$1 > 300' ===
 13709 total
（除总计行外无输出 → 全 app/ 无超过 300 行的文件）
```

### 2.4 项目根仍在、但**我没动**的脚本（等批次 20 任务书裁决）
`b16a_acceptance.py`、`b16b_backend.py`、`b16b_prep.py`、`b16b_restore.py`、`b17_fingerprint.py`、
`backfill_law_dates.py`、`cleanup_e2e_probe_accounts.py`、`e2e_auth_persist.py`、`e2e_create_admin.py`、
`e2e_long_term_memory.py`、`e2e_review_publish.py`、`migrate_auth_users.py`、
`migrate_long_term_memory_setting.py`、`migrate_review_status.py`、`frontend_build.log`、`.pytest_cache/`（根）

提醒：三个 `migrate_*.py` 被模型 docstring 明确引用为"MySQL 已用显式迁移脚本同步"的迁移历史，
建议**保留**；其余请以任务书清单为准。

---

## 三、过程记录
1. 三处调用方 import 全部同步，无兼容垫片；本轮 AST 比对**零白名单**（不需要任何人工放行的差异）。
2. 本轮改用"AST 定位 + 脚本删除"处理大段迁移（⑦⑧⑨ 的函数/类整体移出），比手写 Edit 更稳，
   未再出现丢盘问题。
3. `title_normalizer.py` / `review_detail.py` 初次覆盖率不足 30%，已补真实注释（规则动机、契约引用、判空原因），非凑数。
