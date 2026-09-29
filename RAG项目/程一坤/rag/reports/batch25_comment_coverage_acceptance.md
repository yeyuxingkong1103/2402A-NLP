# 批次 25 验收报告：注释补齐收尾（全 app/）

> 边界遵守：不改 `docs/`；临时脚本全在系统临时目录（未进 `backend/`）；
> 改动前备份到 `.dev_bak25/`（已 md5 校验）；做完停下汇报。

---

## 结论速览

| # | 任务书要求 | 结果 |
|---|---|---|
| 1 | 全 `app/` 每个文件的注释覆盖率统计表（低到高） | ✅ **109 个文件**，见第一节 / `reports/b25_comment_coverage_before.txt` |
| 2 | 点名 20 个文件覆盖率 ≥30% | ✅ **20/20 达标**（改动前 14 个已达标 + 本次补齐 **6** 个） |
| 3 | 标准：docstring（作用/参数/返回）+ 关键行"为什么"级行注 + 块级说明 | ✅ 见第二节逐文件清单 |
| 4 | 零逻辑改动（AST 剥 docstring 逐对象比对 PASS）+ 单文件 ≤300 行 | ✅ **6/6 PASS**；本批最长 **254 行** |
| 5 | 验收：前后覆盖率 + 指纹自查 + 598 passed | ✅ 见第二/三/六节 |

**两件与任务书前提不一致的事（有实测数字）**：

1. **点名"覆盖面低"的 20 个文件里，14 个本来就已 ≥30%**。
   实测未达标只有 **6 个**（`bootstrap` 16.67% / `paragraph_items` 17.98% / `streaming` 18.18% /
   `synonym_expansion` 19.55% / `memory_schema` 23.40% / `memory_settings` 29.79%）。
   → 我只改这 6 个（其余 13 个已达标 + `chunk_text` 正好 30.00%，**一律未动**，避免无谓改动引入风险），
   结果是 **20/20 ≥30%**。
2. **全 `app/` 实际有 55/109 低于 30%**，点名清单没覆盖到的还有 **49 个**；
   而且"补注释"与"单文件 ≤300 行"存在**硬冲突**（2 个已超限、3 个一补必超）。
   → 是否继续补、怎么补，见第七节待你定。

---

## 一、全 `app/` 注释覆盖率统计表（改动前，按覆盖率升序）

口径（任务书指定）：

```
覆盖率 = (# 注释行 ∪ docstring 行) / 总行
  · 总行        = splitlines() 条数（含空行，与 wc -l 一致）
  · 注释行      = tokenize 出的 COMMENT 所在行（纯注释行 + 行尾注释，按行去重）
  · docstring行 = Module / Class / Function / AsyncFunction 首个字符串常量覆盖的行区间
```

另附**「非空口径」**列 = 注释行 ∪ docstring 行 / 非空行数，
用来识别"总行低是因为空行多"的文件（例：`app/db/law_models.py` 含空行口径 38.29%、
非空口径 54.47%）。**排序与阈值判定一律用任务书口径。**

★ = 任务书点名的 20 个文件。

```
| 文件                                           |   总行 |   空行 |   注释行 | docstr |     覆盖率 |     非空口径 |
|----------------------------------------------|------|------|-------|--------|---------|----------|
| app/pipeline/__init__.py                     |   21 |    1 |     0 |      0 |  0.0000 |   0.0000 |
| app/pipeline/package_writer.py               |   88 |   11 |     0 |      0 |  0.0000 |   0.0000 |
| app/cli/import_mysql.py                      |  101 |   18 |     2 |      1 |  0.0198 |   0.0241 |
| app/api/legal_search.py                      |  194 |   25 |     4 |      3 |  0.0206 |   0.0237 |
| app/pipeline/package_models.py               |  162 |   33 |     5 |      0 |  0.0309 |   0.0388 |
| app/core/logging.py                          |   28 |    5 |     1 |      1 |  0.0357 |   0.0435 |
| app/cli/index_legal_documents.py             |  183 |   33 |     7 |      0 |  0.0383 |   0.0467 |
| app/pipeline/package_validator.py            |  180 |   30 |     8 |      0 |  0.0444 |   0.0533 |
| app/db/batch_import.py                       |   66 |   11 |     4 |      4 |  0.0606 |   0.0727 |
| app/db/milvus_store.py                       |  165 |   19 |    14 |      4 |  0.0848 |   0.0959 |
| app/auth/router.py                           |  168 |   25 |    16 |      8 |  0.0952 |   0.1119 |
| app/models/embedding.py                      |  194 |   20 |    19 |     14 |  0.0979 |   0.1092 |
| app/ingest/run_offline_ingest.py             |  123 |   24 |    13 |      9 |  0.1057 |   0.1313 |
| app/api/chat.py                              |  244 |   22 |    26 |      9 |  0.1066 |   0.1171 |
| app/db/import_service.py                     |  278 |   31 |    33 |     16 |  0.1187 |   0.1336 |
| app/ingest/offline_ingest.py                 |  276 |   31 |    33 |     19 |  0.1196 |   0.1347 |
| app/api/memory.py                            |  172 |   25 |    23 |     23 |  0.1337 |   0.1565 |
| app/api/review.py                            |  173 |   27 |    24 |     21 |  0.1387 |   0.1644 |
| app/db/base.py                               |   14 |    6 |     2 |      2 |  0.1429 |   0.2500 |
| app/db/engine.py                             |   14 |    4 |     2 |      2 |  0.1429 |   0.2000 |
| app/models/reranker.py                       |  159 |   25 |    24 |     22 |  0.1509 |   0.1791 |
| app/db/vector_index_service.py               |  250 |   42 |    38 |     11 |  0.1520 |   0.1827 |
| app/retrieval/service.py                     |  266 |   27 |    43 |     25 |  0.1617 |   0.1799 |
| app/chat/bootstrap.py ★                      |   66 |   10 |    11 |      7 |  0.1667 |   0.1964 |
| app/models/mineru.py                         |  446 |   57 |    75 |     60 |  0.1682 |   0.1928 |
| app/retrieval/keyword_search.py              |  295 |   41 |    50 |     25 |  0.1695 |   0.1969 |
| app/memory/long_term.py                      |  232 |   23 |    40 |     35 |  0.1724 |   0.1914 |
| app/models/qwen_vl.py                        |  297 |   40 |    52 |     30 |  0.1751 |   0.2023 |
| app/ingest/paragraph_items.py ★              |   89 |   21 |    16 |      9 |  0.1798 |   0.2353 |
| app/chat/streaming.py ★                      |  165 |   18 |    30 |     18 |  0.1818 |   0.2041 |
| app/models/llm.py                            |  247 |   34 |    45 |     36 |  0.1822 |   0.2113 |
| app/main.py                                  |  164 |   28 |    30 |      9 |  0.1829 |   0.2206 |
| app/auth/mailer.py                           |  133 |   23 |    25 |     23 |  0.1880 |   0.2273 |
| app/retrieval/synonym_expansion.py ★         |  179 |   33 |    35 |     29 |  0.1955 |   0.2397 |
| app/crawler/runner.py                        |  107 |   16 |    21 |      0 |  0.1963 |   0.2308 |
| app/crawler/whitelist.py                     |  140 |   18 |    30 |      1 |  0.2143 |   0.2459 |
| app/db/import_cleanup.py                     |   79 |   13 |    17 |      7 |  0.2152 |   0.2576 |
| app/ingest/cleaner.py                        |  124 |   26 |    28 |     17 |  0.2258 |   0.2857 |
| app/ingest/parser.py                         |  211 |   45 |    48 |      0 |  0.2275 |   0.2892 |
| app/core/config.py                           |  381 |   29 |    88 |     16 |  0.2310 |   0.2500 |
| app/memory/memory_schema.py ★                |   94 |   16 |    22 |     20 |  0.2340 |   0.2821 |
| app/ingest/law_metadata_extractor.py         |  198 |   32 |    47 |     32 |  0.2374 |   0.2831 |
| app/cli/demo_ask.py                          |  122 |   27 |    29 |     23 |  0.2377 |   0.3053 |
| app/ingest/chunker.py                        |  181 |   30 |    44 |     15 |  0.2431 |   0.2914 |
| app/retrieval/vector_search.py               |  290 |   32 |    72 |     27 |  0.2483 |   0.2791 |
| app/review/review_service.py                 |  220 |   31 |    57 |     41 |  0.2591 |   0.3016 |
| app/crawler/collect_labor_law.py             |  209 |   30 |    55 |     31 |  0.2632 |   0.3073 |
| app/auth/service.py                          |  193 |   31 |    51 |     27 |  0.2642 |   0.3148 |
| app/crawler/requester.py                     |  287 |   28 |    76 |      0 |  0.2648 |   0.2934 |
| app/db/legal_metadata_writer.py              |  166 |   25 |    46 |     19 |  0.2771 |   0.3262 |
| app/memory/summary_service.py                |  205 |   31 |    57 |     46 |  0.2780 |   0.3276 |
| app/crawler/rate_limiter.py                  |   56 |   13 |    16 |      0 |  0.2857 |   0.3721 |
| app/models/http_retry.py                     |  247 |   37 |    71 |     64 |  0.2874 |   0.3381 |
| app/auth/schemas.py                          |   79 |   28 |    23 |     12 |  0.2911 |   0.4510 |
| app/memory/memory_settings.py ★              |   47 |   11 |    14 |     14 |  0.2979 |   0.3889 |
| app/ingest/chunk_text.py ★                   |   50 |   10 |    15 |     11 |  0.3000 |   0.3750 |
| app/ingest/pdf_parser.py                     |   89 |   21 |    27 |      7 |  0.3034 |   0.3971 |
| app/api/users_me.py                          |   56 |   14 |    17 |     17 |  0.3036 |   0.4048 |
| app/memory/memory_summarize.py ★             |   49 |    6 |    15 |     13 |  0.3061 |   0.3488 |
| app/review/review_detail.py                  |  114 |   14 |    35 |     20 |  0.3070 |   0.3500 |
| app/chat/memory_hooks.py ★                   |  151 |   21 |    47 |     40 |  0.3113 |   0.3615 |
| app/retrieval/item_convert.py ★              |  109 |   12 |    34 |     28 |  0.3119 |   0.3505 |
| app/ingest/article_splitter.py ★             |   60 |   12 |    19 |      6 |  0.3167 |   0.3958 |
| app/auth/session_store.py                    |  102 |   27 |    33 |     21 |  0.3235 |   0.4400 |
| app/chat/service.py                          |  230 |   25 |    75 |     56 |  0.3261 |   0.3659 |
| app/crawler/crawl_record.py                  |   60 |   19 |    20 |      0 |  0.3333 |   0.4878 |
| app/db/import_version_status.py ★            |   83 |   10 |    28 |     16 |  0.3373 |   0.3836 |
| app/db/sql_models.py                         |   38 |    4 |    13 |     13 |  0.3421 |   0.3824 |
| app/chat/prompt_builder.py                   |   90 |   12 |    31 |     25 |  0.3444 |   0.3974 |
| app/ingest/title_normalizer.py ★             |  116 |   14 |    40 |     20 |  0.3448 |   0.3922 |
| app/chat/guard.py                            |  230 |   54 |    81 |     62 |  0.3522 |   0.4602 |
| app/ingest/law_date_rules.py ★               |  133 |   21 |    47 |     33 |  0.3534 |   0.4196 |
| app/db/document_models.py ★                  |  164 |   27 |    59 |     24 |  0.3598 |   0.4307 |
| app/retrieval/assembly.py                    |  232 |   37 |    84 |     72 |  0.3621 |   0.4308 |
| app/auth/current_user.py                     |   55 |   16 |    20 |     14 |  0.3636 |   0.5128 |
| app/auth/sql_store.py                        |  115 |   17 |    42 |     31 |  0.3652 |   0.4286 |
| app/errors.py                                |  112 |   18 |    41 |      8 |  0.3661 |   0.4362 |
| app/retrieval/filters.py                     |  106 |   19 |    40 |     31 |  0.3774 |   0.4598 |
| app/db/law_models.py ★                       |  175 |   52 |    67 |     32 |  0.3829 |   0.5447 |
| app/crawler/robots_policy.py                 |   39 |    9 |    15 |      1 |  0.3846 |   0.5000 |
| app/crawler/html_text.py                     |   93 |   13 |    36 |      7 |  0.3871 |   0.4500 |
| app/retrieval/parent_collapse.py ★           |  250 |   26 |    98 |     61 |  0.3920 |   0.4375 |
| app/cli/create_admin.py                      |   92 |   20 |    37 |     32 |  0.4022 |   0.5139 |
| app/chat/citation_check.py                   |  235 |   38 |    95 |     54 |  0.4043 |   0.4822 |
| app/pipeline/hashing.py                      |   12 |    4 |     5 |      5 |  0.4167 |   0.6250 |
| app/retrieval/query_rewrite.py               |  201 |   31 |    85 |     62 |  0.4229 |   0.5000 |
| app/api/session_routes.py                    |  261 |   31 |   111 |     74 |  0.4253 |   0.4826 |
| app/chat/result.py ★                         |   42 |    6 |    18 |     16 |  0.4286 |   0.5000 |
| app/crawler/response_reader.py               |   53 |    8 |    23 |      6 |  0.4340 |   0.5111 |
| app/retrieval/fusion.py                      |  162 |   22 |    72 |     49 |  0.4444 |   0.5143 |
| app/db/chat_models.py                        |  114 |   28 |    52 |     18 |  0.4561 |   0.6047 |
| app/db/redis_client.py                       |   39 |   12 |    18 |     17 |  0.4615 |   0.6667 |
| app/api/chat_persistence.py                  |  225 |   37 |   105 |     75 |  0.4667 |   0.5585 |
| app/ingest/chinese_number.py                 |  136 |   23 |    67 |     48 |  0.4926 |   0.5929 |
| app/ingest/article_number_rules.py           |  184 |   24 |    91 |     70 |  0.4946 |   0.5687 |
| app/db/user_models.py ★                      |   48 |    9 |    24 |     15 |  0.5000 |   0.6154 |
| app/chat/chat_store.py                       |  279 |   45 |   140 |     83 |  0.5018 |   0.5983 |
| app/retrieval/context_builder.py             |  197 |   41 |    99 |     47 |  0.5025 |   0.6346 |
| app/chat/chat_runtime.py                     |   56 |   14 |    29 |     19 |  0.5179 |   0.6905 |
| app/ingest/chunk_fingerprint.py ★            |   45 |    8 |    24 |     21 |  0.5333 |   0.6486 |
| app/chat/chat_history.py                     |  174 |   21 |    96 |     53 |  0.5517 |   0.6275 |
| app/chat/chat_title.py                       |   50 |   13 |    30 |     20 |  0.6000 |   0.8108 |
| app/memory/summary_policy.py                 |   29 |    7 |    18 |      8 |  0.6207 |   0.8182 |
| app/memory/short_term.py                     |  183 |   37 |   114 |    106 |  0.6230 |   0.7808 |
| app/db/version_status.py                     |   35 |    8 |    27 |     21 |  0.7714 |   1.0000 |
| app/auth/__init__.py                         |    1 |    0 |     1 |      1 |  1.0000 |   1.0000 |
| app/memory/__init__.py                       |    1 |    0 |     1 |      1 |  1.0000 |   1.0000 |
| app/retrieval/__init__.py                    |    1 |    0 |     1 |      1 |  1.0000 |   1.0000 |
| app/review/__init__.py                       |    1 |    0 |     1 |      1 |  1.0000 |   1.0000 |

低于 30% 的文件 = 55 / 109
```

改动后同一张表见 `reports/b25_comment_coverage_after.txt`：**低于 30% 的文件 = 49 / 109**。

---

## 二、点名 20 个文件：实测前提与补齐结果

### 2.1 改动前后对照

| 文件 | 总行 前→后 | 注释行 前→后 | 覆盖率 前→后 | 结论 |
|---|---|---|---|---|
| app/chat/streaming.py | 165 → 202 | 30 → 67 | 0.1818 → **0.3317** (+0.1499) | ✔ 本次补齐 |
| app/chat/bootstrap.py | 66 → 86 | 11 → 31 | 0.1667 → **0.3605** (+0.1938) | ✔ 本次补齐 |
| app/ingest/paragraph_items.py | 89 → 116 | 16 → 43 | 0.1798 → **0.3707** (+0.1909) | ✔ 本次补齐 |
| app/memory/memory_schema.py | 94 → 133 | 22 → 61 | 0.2340 → **0.4586** (+0.2246) | ✔ 本次补齐 |
| app/memory/memory_settings.py | 47 → 72 | 14 → 39 | 0.2979 → **0.5417** (+0.2438) | ✔ 本次补齐 |
| app/retrieval/synonym_expansion.py | 179 → 254 | 35 → 110 | 0.1955 → **0.4331** (+0.2376) | ✔ 本次补齐 |
| app/chat/memory_hooks.py | 151 → 151 | 47 → 47 | 0.3113 → 0.3113 | ✔ 原已达标（未动） |
| app/chat/result.py | 42 → 42 | 18 → 18 | 0.4286 → 0.4286 | ✔ 原已达标（未动） |
| app/db/document_models.py | 164 → 164 | 59 → 59 | 0.3598 → 0.3598 | ✔ 原已达标（未动） |
| app/db/law_models.py | 175 → 175 | 67 → 67 | 0.3829 → 0.3829 | ✔ 原已达标（未动） |
| app/db/user_models.py | 48 → 48 | 24 → 24 | 0.5000 → 0.5000 | ✔ 原已达标（未动） |
| app/db/import_version_status.py | 83 → 83 | 28 → 28 | 0.3373 → 0.3373 | ✔ 原已达标（未动） |
| app/ingest/article_splitter.py | 60 → 60 | 19 → 19 | 0.3167 → 0.3167 | ✔ 原已达标（未动） |
| app/ingest/chunk_text.py | 50 → 50 | 15 → 15 | 0.3000 → 0.3000 | ✔ 正好 30.00%（未动） |
| app/ingest/chunk_fingerprint.py | 45 → 45 | 24 → 24 | 0.5333 → 0.5333 | ✔ 原已达标（未动） |
| app/ingest/law_date_rules.py | 133 → 133 | 47 → 47 | 0.3534 → 0.3534 | ✔ 原已达标（未动） |
| app/ingest/title_normalizer.py | 116 → 116 | 40 → 40 | 0.3448 → 0.3448 | ✔ 原已达标（未动） |
| app/memory/memory_summarize.py | 49 → 49 | 15 → 15 | 0.3061 → 0.3061 | ✔ 原已达标（未动） |
| app/retrieval/item_convert.py | 109 → 109 | 34 → 34 | 0.3119 → 0.3119 | ✔ 原已达标（未动） |
| app/retrieval/parent_collapse.py | 250 → 250 | 98 → 98 | 0.3920 → 0.3920 | ✔ 原已达标（未动） |

```
合计：20/20 ≥30%（改动前 14 个已达标 + chunk_text 正好 30.00%，本次补齐 6 个）
全 app/ 低于 30% 的文件数：55/109 → 49/109
```

### 2.2 本次补的 6 个文件：补了什么

| 文件 | 补的内容 |
|---|---|
| `chat/bootstrap.py` | 模块 docstring 补「为什么坚持函数内延迟导入」（评测/启动两条路径都会引用本文件，顶层建连接会污染单测）；函数补 **参数/返回 + 装配顺序为什么把记忆放最后**；行注 5 处（检索与 LLM 是硬依赖故放 try 之外、`ensure_collection` 幂等所以可反复调、`memory_gate` 传的是**可调用对象**而非当期取值以保证运行中改设置立刻生效、失败只告警不抛出） |
| `chat/streaming.py` | 模块 docstring 补「为什么用 Mixin」（与同步 `chat()` 必须共用同一套配置属性，独立类会把属性转发一遍且两边易漂移）；`chat_stream` 补 **参数/返回**（含 `stream_context` 为什么是可变字典）；新增 `finish` 闭包 docstring（为什么要闭包：三条出口写同一份结构，避免字段写漏）；行注 8 处（记忆块与检索的顺序与依赖关系、`llm_client` 校验为什么放在拒答分支之后、`messages[0]/[1]` 分别是什么、`emitted_len` 为什么只发增量、允许清单的语义、三条 `except` 的分级理由、收尾标记的用途、写记忆为什么放最后） |
| `ingest/paragraph_items.py` | 模块 docstring 补「为什么款/项要单独成文件」（父子块质量的地基 + ≤300 行）；`_split_into_paragraphs` 补 **参数/返回 + 为什么 `\n\s*\n\|\n` 双分支切**；`_extract_items` 补 **参数/返回 + 为什么无项返回 `None` 而不是空列表 + 为什么延迟导入 `to_arabic_number`**；两个模式的差异注释；行注 5 处 |
| `memory/memory_schema.py` | `MemoryRecord` 补**字段说明 + 为什么 `frozen`**（拦住"以为改了内存就等于改了库"）；`LongTermMemoryError` 补为什么单独定义类型；`MemoryCollectionMixin` 补为什么用 Mixin；`ensure_collection` 补 **参数/返回 + 幂等 + 为什么末尾必须 `load_collection`**；`_expected_fields` 补为什么写成字面量而非从 schema 反推；行注 3 处（`auto_id=False` 与去重、`dense_vector` 维度不能写死、COSINE 与法条集合同口径） |
| `memory/memory_settings.py` | `__init__` 新增 docstring（**参数/返回 + 为什么只存会话工厂**）；`is_enabled` 补 **参数/返回 + 为什么"查不到用户"返回 `False` 而不抛错**（它挂在每轮问答前，宁可安静不写记忆也不能让接口 500）；`set_enabled` 补 **参数/返回 + 为什么强制 `bool()`** |
| `retrieval/synonym_expansion.py` | 模块 docstring 补「为什么扩展是单向的」；`SynonymGroup` / `SynonymTable` / `ExpansionResult` 三个 dataclass 补**字段说明 + frozen 取舍**（前两个是只读快照、后一个是本次调用的累积容器）；`load_synonym_table` 补 **参数/返回/异常 + 为什么不静默降级为空表**；`expand` / `explain` / `_matches` 补 **参数/返回 + 为什么先做去空白压缩 + ASCII 为什么用 lookaround 而不是 `\b`**；行注 5 处 |

---

## 三、零逻辑改动证明（AST 剥 docstring 逐对象比对）

`fingerprint_check.py` → `reports/b25_fingerprint.txt` + `reports/b25_fingerprint.json`

做法：把 6 个文件各自与 `.dev_bak25/` 里的原件分别 `ast.parse`，
**剥掉所有 docstring**（Module / Class / Function / AsyncFunction 的 `body[0]` 为字符串常量时删除），
再 `ast.dump(include_attributes=False)`，先比整模块、再比**每个顶层对象**。

```
文件                                           | 行数 旧→新         | 注释行 旧→新        | 整体 AST | 逐对象
----------------------------------------------------------------------------------------------------------------------
app/chat/bootstrap.py                        |   66 → 86      |    4 → 12      |    ✔     | ✔
app/chat/streaming.py                        |  165 → 202     |   12 → 20      |    ✔     | ✔
app/ingest/paragraph_items.py                |   89 → 116     |    7 → 13      |    ✔     | ✔
app/memory/memory_schema.py                  |   94 → 133     |    2 → 5       |    ✔     | ✔
app/memory/memory_settings.py                |   47 → 72      |    0 → 0       |    ✔     | ✔
app/retrieval/synonym_expansion.py           |  179 → 254     |    6 → 16      |    ✔     | ✔
----------------------------------------------------------------------------------------------------------------------
比对文件数 = 6
结论：AST（剥 docstring）逐对象完全一致 = ✅ 全部 PASS
```

> 说明：`#` 注释不进 AST，剥 docstring 后 AST 必须逐字节相同 —— 这是"只动注释、没动代码"的硬证明。
> `memory_settings.py` 的"注释行 0 → 0"是因为它的补充全部落在 docstring 里（覆盖率仍从 29.79% → 54.17%）。

---

## 四、行数检查（单文件 ≤300 行）

```
✔ app/chat/bootstrap.py          66 → 86
✔ app/chat/streaming.py         165 → 202
✔ app/ingest/paragraph_items.py  89 → 116
✔ app/memory/memory_schema.py    94 → 133
✔ app/memory/memory_settings.py  47 → 72
✔ app/retrieval/synonym_expansion.py  179 → 254   ← 本批最长，余量 46 行

全部 ≤300 = True
```

### ⚠️ 补注释与"≤300 行"的硬冲突（全 app/ 层面，本批未动）

| 文件 | 行数 | 状态 |
|---|---|---|
| `app/models/mineru.py` | 446 | **已超限**（既有） |
| `app/core/config.py` | 381 | **已超限**（既有） |
| `app/models/qwen_vl.py` | 297 | 余量 3 行，补注释必超 |
| `app/retrieval/keyword_search.py` | 295 | 余量 5 行 |
| `app/retrieval/vector_search.py` | 290 | 余量 10 行 |

→ 这 5 个文件都在"低于 30%"的 49 个里面。**"覆盖率 ≥30%"与"≤300 行"对它们不可能同时满足**，
要么先拆文件再补注释，要么对这几个文件豁免行数上限。需要你定（第七节）。

---

## 五、改动文件清单

只改了 6 个，全部是注释 + docstring，无一行代码变动。

| # | 文件 | 行数 | 覆盖率 | 备注 |
|---|---|---|---|---|
| 1 | `backend/app/chat/bootstrap.py` | 66 → 86 | 16.67% → 36.05% | |
| 2 | `backend/app/chat/streaming.py` | 165 → 202 | 18.18% → 33.17% | |
| 3 | `backend/app/ingest/paragraph_items.py` | 89 → 116 | 17.98% → 37.07% | |
| 4 | `backend/app/memory/memory_schema.py` | 94 → 133 | 23.40% → 45.86% | |
| 5 | `backend/app/memory/memory_settings.py` | 47 → 72 | 29.79% → 54.17% | |
| 6 | `backend/app/retrieval/synonym_expansion.py` | 179 → 254 | 19.55% → 43.31% | 本批最长，254 行 |

---

## 六、体检基线（三项，逐项贴真实输出）

```bash
# ① 全量测试（任务书指定形式：项目根 + PYTHONPATH= + backend/tests）
cd C:/Users/92842/Desktop/rag
PYTHONPATH= C:/Users/92842/anaconda3/envs/rag/python.exe -m pytest backend/tests -q --no-header -p no:cacheprovider
# → 598 passed, 3 warnings in 12.47s        退出码 = 0        （reports/b25_final_pytest.log）

# ② 服务自检
python scripts/check_services.py
# → ✅ 全部通过（退出码 0）：Redis PONG / MySQL documents=11 document_chunks=1627
#    law_versions=11 / Milvus 1627 == MySQL 1627 / approved=11

# ③ 导入自检
python scripts/check_imports.py
# → ✅ 全部 app.* 导入均可解析（模块存在 + 符号存在）  扫描 107 文件  退出码 = 0
```

**① 两处口径说明（都不是问题，但要说清）**：

- 任务书的命令形式（`backend/tests` + `PYTHONPATH=`）**直接可用**，不需要 `--basetemp`；
  我另外验证了带 `--basetemp` 的形式也是 598 passed（两种都贴了日志）。
- **`check_imports` 报 107 而不是 101**：脚本的 `SKIP_DIRS` 里有 `.dev_bak21` 但**没有 `.dev_bak25`**，
  于是把我本批的 6 个备份副本也扫了。实测**排除 `.dev_bak25` 后正好 101 个文件、同样全绿**
  （多出来的 6 个逐一列出确认过：就是这 6 个文件的备份）。
  → 备份按任务书要保留到验收通过，清理后自然回到 101。

---

## 七、待你裁决 / 需你处理

1. **其余 49 个低于 30% 的文件要不要继续补？**
   它们**不在你的 20 个点名清单里**。最惨的一批是：
   `app/pipeline/package_writer.py` 0.00%、`app/pipeline/__init__.py` 0.00%、
   `app/cli/import_mysql.py` 1.98%、`app/api/legal_search.py` 2.06%、`app/pipeline/package_models.py` 3.09%。
   继续的话建议**分批按目录推进**（一轮一个目录：api / pipeline+cli / models / crawler / ingest / retrieval / db / auth / memory / chat / review），
   每批都出"前后覆盖率 + 指纹 + 598 passed"。
   **要不要开批次 25-2？从哪个目录开始？**
2. **"覆盖率 ≥30%"与"单文件 ≤300 行"的冲突**（第四节 5 个文件）怎么处理 ——
   先拆文件再补注释 / 对这几个文件豁免行数 / 只补到不超限为止？
3. **`scripts/check_imports.py` 的 `SKIP_DIRS` 加一个 `.dev_bak*`**（或 `.dev_bak25`）——
   1 行改动，能让后续每批的备份目录不再干扰自检计数。**按你的规矩我不擅自改代码，等你批。**
4. **`.dev_bak25/` 的清理时机** —— 我按任务书留到"验收通过后"；你说清就清。

---

## 八、复现命令与证据文件

```bash
PY=C:/Users/92842/anaconda3/envs/rag/python.exe        # 项目唯一正式环境（§七 规则 7）

# 覆盖率统计（全 app/ 按低到高；--targets 只看点名的 20 个；--json 落盘供前后对比）
python <临时目录>/count_comment_coverage.py --json <临时目录>/coverage_before.json
python <临时目录>/count_comment_coverage.py --targets

# 零逻辑改动指纹（比对 .dev_bak25/ 里的原件）
python <临时目录>/fingerprint_check.py

# 体检三项
cd C:/Users/92842/Desktop/rag
PYTHONPATH= $PY -m pytest backend/tests -q --no-header -p no:cacheprovider
python scripts/check_services.py
python scripts/check_imports.py
```

| 证据文件 | 内容 |
|---|---|
| `reports/b25_comment_coverage_before.txt` | 改动前全 app/ 覆盖率表（109 文件，含阈值判定与 300 行冲突预警） |
| `reports/b25_comment_coverage_after.txt` | 改动后同一张表（低于 30% 的从 55 → 49） |
| `reports/b25_comment_coverage_targets_after.txt` | 点名 20 个文件改动后覆盖率（低于 30% = **0/20**） |
| `reports/b25_target_coverage_compare.txt` | 20 个文件前后对照（含达标/未达标判定与行数上限检查） |
| `reports/b25_fingerprint.txt` / `b25_fingerprint.json` | AST 剥 docstring 逐对象比对（6/6 PASS） |
| `reports/b25_baseline_pytest.log` / `b25_pytest_nobasetemp.log` | 基线测试原始输出（两种命令形式，都是 598 passed） |
| `reports/b25_final_pytest.log` | 改动后回归测试（598 passed） |
| `reports/b25_check_services.txt` | check_services 原始输出（全绿，退出码 0） |
| `reports/b25_check_imports.txt` | check_imports 原始输出（全绿，退出码 0，扫描 107 含 6 个备份副本） |
| 备份 | `.dev_bak25/`（6 个改动文件的改动前版本，md5 已校验一致） |
