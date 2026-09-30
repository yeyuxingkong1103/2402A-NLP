# 批次 10 验收报告（2026-09-20）

范围：①补生效日期 ②召回窗口 40/40/40 配置化 ③护栏零引用分级 ④评测集扩样候选（待审） ⑤Faithfulness 轻量实现。
测试：**338 passed**（323 + 新增 15）。

---

## 一、补生效日期（✅ 通过）

**根因不是过滤逻辑失效，是抽取缺陷**（你的判断正确）：
1. 真实页面排版是"本条例自 2004 年 1 月 1 日起施行"——数字与年月日之间有空格，旧正则 `\d{4})年` 不容空格 → 6 部法规生效日期漏抽；
2. "自公布之日起施行"（实施条例/特别规定）没有映射到公布日期；
3. 最高法新闻页只有"发布时间：2025-08-01"标注，旧词表不含"发布时间"；
4. `legal_metadata_writer` 对已存在的 law_version 不刷新日期字段。

**修复**（`app/ingest/law_metadata_extractor.py` + `app/db/legal_metadata_writer.py`）：
- 所有日期正则各段容忍空白；
- ② "自公布之日起施行"→ 生效=公布日期（页面必须有这句话且公布日期已知，否则留空）；
- ③ 修订语境"…修订，自X年X月X日起施行"优先；
- "公布"日期语义优先于"会议通过"日期；
- writer 对已存在 law_version 按新抽取值刷新（仍只写页面抽到的值）。

**回填后 11 部法规日期表**（详见 `reports/batch10_law_dates.md`）：

| 法规 | 公布 | 生效 | 变化 |
|---|---|---|---|
| 解释（一） | 2020-12-30（发布时间） | 2021-01-01 | 公布补上 |
| 典型案例（新闻页） | 2025-08-01 | **（待人工补录）** | 无 |
| 解释（二）+典型案例 | 2025-08-01 | 2025-09-01 | 公布补上 |
| 劳动争议调解仲裁法 | 2007-12-29 | 2008-05-01 | 无 |
| 劳动合同法 | 2012-12-28 | 2008-01-01 | 生效补上 |
| 劳动法 | 2018-12-29 | 1995-01-01 | 生效补上 |
| 工资支付暂行规定 | 1994-12-06 | 1995-01-01 | 无 |
| 工伤保险条例 | 2003-04-27 | 2004-01-01 | 生效补上 |
| 劳动合同法实施条例 | 2008-09-18 | 2008-09-18（自公布之日起施行） | 生效补上 |
| 女职工劳动保护特别规定 | 2012-04-28 | 2012-04-28（自公布之日起施行） | 生效补上 |
| 职工带薪年休假条例 | 2007-12-14 | 2008-01-01 | 无 |

**待人工补录清单（1 条）**：
- 《最高法发布劳动争议典型案例》：缺**生效日期**（新闻页无施行条款，"典型案例"本身也没有施行日期概念，建议人工判定后置为 N/A 或填发布日）；页面链接见 `reports/batch10_law_dates.md`。
- 另：11 部法规的**效力状态**页面均无"现行有效/已废止"字样，status 全部留空——如需 status 参与过滤/展示，也需人工补录。

**重跑**：导入走增量判定（分块未变→unchanged 短路，故用回填脚本刷新元数据）；索引 `--recreate-collection` 显式重建：**1627 chunks / 11 versions，数量一致**。Milvus payload 复核：486 个 parent 块中 effective_date=0 的只剩 1 条（典型案例）。

**as_of 复测**：
- asof-035（2010-06-01 问产假）：劳动法62条 **top1**，特别规定第七条不再越界（violation=None）✅
- asof-036（2020-06-01 问产假）：特别规定第七条 **rank2**，无越界 ✅

## 二、召回窗口 40/40/40（✅ 配置化完成，⚠️ 收益未达预期，如实报告）

- 已抽成配置：`.env` 新增 `RECALL_VECTOR_LIMIT/RECALL_KEYWORD_LIMIT/RERANK_CANDIDATE_LIMIT=40`，代码默认值同步（`config.py`/`retrieval/service.py`/`chat/service.py`），传参可覆盖，便于调参。
- 窗口前后同题对比（`evaluation/compare_window.py` 可复现）：候选池明显变大（asof-031 融合 30→59；confuse-049 融合 34→58）。
- **全量评测前后**（同一 50 条，旧=窗口20+日期未修；新=窗口40+日期已修）：

| 指标 | 前 | 后 | 变化 |
|---|---|---|---|
| Recall@5 | 0.9286 | **0.9048** | ↓（direct-015 rank 5→7） |
| MRR@10 | 0.7351 | 0.7330 | ≈平（asof-031 rank 7→9） |
| 引用正确率 | 1.0000 | 1.0000 | 平 |
| 拒答准确率 | 1.0000 | 1.0000 | 平 |
| as_of 越界 | 1 | **0** | 日期修复收益 |

- **你的期望（confuse-046/049/asof-031 至少两条进 top5）没有实现**：三条分别 rank None / None / 9。瓶颈已不是融合窗口（候选池 56~59 条都装得下），而是**重排质量**——golden 条文进了候选但重排分排不进前 10。窗口加大反而把 2 条从 top5 边缘挤了出去。
- 建议：窗口保留 40 配置（无害且可调）；把"提召回窗口"的后续精力转投**重排环节**（如 rerank 前 query 扩展、或对 rerank 分与 RRF 分做加权融合），另立项验证。

## 三、护栏零引用分级（✅ 代码+测试通过）

- 新增 `NoCitationError` 子类 + `NO_CITATION_WARNING`：有候选但回答零引用 → **保留回答 + 末尾追加"本回答未引用具体法条编号"警示**，不再整段替换；无候选/低于阈值 → 拒答行为不变；引用越界/空回答 → 仍整段替换。
- chat() 与 chat_stream() 两条路径同口径（流式下追加警示 token，不再触发 replace）。
- 三场景验收：
  1. **无候选拒答（真实输出）**：refusal-037"上海2025年最低月工资标准"→ `refuse_low_score`，输出固定拒答文案 ✅
  2. **正常带引用（真实输出）**：direct-001 试用期问题 → `citation_check_passed`，带 [1][2][3] 引用正常回答 ✅
  3. **有候选零引用**：本次 50 条评测中该场景出现 0 次（42 条回答全部有引用），生产侧无真实样本；分级行为由单测覆盖（`tests/test_guard_zero_citation_tier.py` 3 例 + `test_guardrail_graded.py` 2 例，含流式断言）。

## 四、评测集扩样候选（⏸ 已交审，等批准后跑指标）

- `data/evaluation/eval_set_v1_batch10_candidates.jsonl`：35 条新增（051–085），合并后 85 条，拒答 15 条占 **17.6%**（≥15%）。
- 全部带 source_quote，逐字取自库内条文原文；审阅文档 `reports/batch10_candidates_review.md`。
- **待你批准后**：合并进 eval_set_v1.jsonl → 全量跑指标 → 用扩后样本复核 REFUSAL_MIN_VECTOR_SCORE（两簇间距+保守取值建议）。

## 五、Faithfulness 轻量实现（✅ 完成，未装 ragas）

- `evaluation/faithfulness.py`：复用现有 DeepSeek 客户端，对每条非拒答回答判"每个结论是否被注入的法条原文支持"，0~1 分+一句话理由；`run_eval.py` 明细新增 `context_excerpts`（注入原文，[n] 对齐），打分离线完成。
- **本次结果（50 条，42 条打分，8 条拒答跳过）**：
  - **Faithfulness 均值 0.9869**，解析失败 0
  - 最差 5 条：asof-035(0.75)、direct-009(0.85)、direct-008(0.90)、asof-034(0.95)、direct-001(1.00)
  - 共性问题：模型在法条之外做延伸断言（如"孕期禁夜班"漏掉"七个月以上"限定）
- **费用**：42 次 LLM 调用，估算输入 ~114k token / 输出 ~2.5k token，**约 0.09 元/轮**（单价假设：输入1元/百万token、输出2元/百万token，以账单为准）。报告：`reports/latest_faithfulness.md`。

## 改动清单

| 文件 | 改动 |
|---|---|
| `backend/app/ingest/law_metadata_extractor.py` | 日期正则空白容忍、自公布之日起施行、修订语境、发布时间标注、公布优先于通过 |
| `backend/app/db/legal_metadata_writer.py` | 已存在 law_version 刷新时效字段 |
| `backend/app/core/config.py` | 3 个召回窗口配置项 |
| `backend/app/retrieval/service.py`、`backend/app/chat/service.py` | 窗口参数化（None→读配置）；零引用分级接入两条路径；ChatResult.context_excerpts |
| `backend/app/chat/citation_check.py`、`backend/app/chat/guard.py` | NoCitationError 子类 + 追加警示 |
| `.env` | RECALL_VECTOR_LIMIT/RECALL_KEYWORD_LIMIT/RERANK_CANDIDATE_LIMIT=40 |
| `backend/tests/`（3 个文件） | +15 测试（日期 11、护栏分级 5 中 2 更新 3 新增、零引用分级 3） |
| `backfill_law_dates.py`（项目根，一次性） | 时效字段回填 |
| `evaluation/run_eval.py`、`evaluation/faithfulness.py`（新）、`evaluation/compare_window.py`（新）、`evaluation/README.md` | 评测工具 |
| `data/evaluation/eval_set_v1_batch10_candidates.jsonl`（新） | 35 条候选，待审 |
| `reports/` | batch10_law_dates.md、batch10_candidates_review.md、latest_faithfulness.md、eval_*_batch10.*、本报告 |

未动 docs/；未新增依赖；临时脚本均在项目根/evaluation/，未进 backend/。
