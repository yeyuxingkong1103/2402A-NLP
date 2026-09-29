# 评测工具（阶段 8.3 / 8.4）

| 文件 | 用途 |
|---|---|
| `run_eval.py` | CLI 装配 + 主循环：跑四指标 + 批次 23 的 3 个 multi_turn 指标，输出 `reports/` |
| `legal_matching.py` | 法规名 / 条号匹配（检索侧判定原语：`first_golden_rank` / `absent_violation` …） |
| `answer_judging.py` | 回答侧判定原语（拒答口径 R2+R1′、引用越界审计） |
| `item_runner.py` | 单问题题型的执行（`with_retry` + `run_item`） |
| `aggregate.py` | 指标汇总与最差样本（`summarize` / `worst_samples`，数字口径唯一来源） |
| `render_report.py` | Markdown 渲染（`render_markdown`） |
| `multi_turn_runner.py` | multi_turn 题目执行器：真实多轮链路 + 短期记忆回写 + 会话清理（run_eval 只负责接线） |
| `multi_turn_grading.py` | multi_turn 的判定原语 / 汇总 / 报告渲染（纯函数，口径单一来源） |
| `calibrate_refusal.py` | 拒答阈值校准：采集分数分布 → 搜分界点 → 出校准报告 |

评测集：`data/evaluation/eval_set_v1.jsonl`（100 条 = 原 95 条 + 5 条 multi_turn，口径见 `_schema.md`）
报告输出：项目根 `reports/`（JSON 机器可读 + Markdown 人可读）

## 一条命令

```bash
# 全量评测（约 30 分钟，含真实 LLM 调用）
python evaluation/run_eval.py

# 加标记（报告文件名带后缀，便于校准前后对比）
python evaluation/run_eval.py --tag before
python evaluation/run_eval.py --tag after

# 只跑若干条（冒烟）
python evaluation/run_eval.py --limit 3

# 只跑某类（按 id 子串）
python evaluation/run_eval.py --only refusal-

# 拒答阈值校准（只跑检索，不调 LLM，约 10 分钟）
python evaluation/calibrate_refusal.py
```

## 注意

- 运行前请确认 MySQL / Milvus / Redis 在跑，`.env` 里 Embedding、Reranker、LLM 配置可用；
- 脚本会自动剔除沙箱注入的 `http_proxy` / `https_proxy`（代理会劫持模型 API 请求导致 SSL 断连）；
- 上游 Embedding/LLM 偶发挂起，脚本内置重试（5 次退避）；
- 指标口径以 `data/evaluation/_schema.md` 为准，改口径请先改文档。

## 批次 23：multi_turn（真实多轮 + 摘要/改写链路）

原 `followup_rewrite` 类型**验不到改写**：它的 `context` 只是字符串前缀，而评测器从不回写
短期记忆（`short_term_memory.append_message` 的唯一调用方是 SSE 接口的 `persist_turn`），
改写器读到的上文永远是空的。`multi_turn` 用 `turns` 数组按真实多轮顺序连跑，
每轮结束后回写短期记忆（等价 `persist_turn` 的 Redis 写侧，**不写 MySQL**），链路才真的接上。

```bash
# 正向：回写开启，改写应真的生效
python evaluation/run_eval.py --only multiturn- --tag mt_on

# 反向对照：禁掉回写，改写应彻底失效（rewrite_rate 掉到 0）
python evaluation/run_eval.py --only multiturn- --no-context-writeback --tag mt_off
```

- 会话清理：每题**跑前清一次残留、跑完在 `finally` 里再清一次**，中途异常也会删掉
  `eval_<题号>` 的 Redis key，不留评测会话；`detail["session_cleanup_before/after"]` 记录结果。
- 三个指标（口径见 `_schema.md`）：`multi_turn_rewrite_rate`（分母 = 有 `rewrite_expect` 的题）、
  `multi_turn_turn2_hit_at_5`、`multi_turn_context_carryover`（只记录不判定）。
- 明细里保留**每轮改写后查询原文**（`detail["turns"][i]["rewrite"]["rewritten_query"]`），
  报告第二节之二会把 5 条题的改写后查询逐行列出来。

## 批次 10 新增工具

```bash
# Faithfulness 打分（对评测报告里的每条回答判"结论是否被注入法条支持"）
python evaluation/faithfulness.py --eval-json reports/latest_eval_batch10.json
# 产物：reports/faithfulness_<时间戳>.json / latest_faithfulness.md（含均值、最差5条、LLM 调用次数与费用估算）

# 召回窗口前后对比（调 RECALL_VECTOR_LIMIT / RECALL_KEYWORD_LIMIT / RERANK_CANDIDATE_LIMIT 用）
python evaluation/compare_window.py --questions asof-031 confuse-046 --old 20 --new 40
```

- 召回窗口已配置化：`.env` 里 `RECALL_VECTOR_LIMIT` / `RECALL_KEYWORD_LIMIT` / `RERANK_CANDIDATE_LIMIT`。
  **当前默认 20**（批次 11 按实测回退：40/40/40 让 Recall@5 0.9286→0.9048、MRR 持平，瓶颈在重排质量）。
- `run_eval.py` 的明细里新增 `context_excerpts`（注入提示词的法条原文，[n] 编号与回答引用一致），供 Faithfulness 离线打分，不用二次检索。

## 批次 13 新增工具：术语扩写消融

术语扩写（口语 → 法条用语，只作用于关键词路）由 `.env` 的 `SYNONYM_EXPANSION_ENABLED` 控制（**默认 false**）。

```bash
# 检索层消融：每 arm 单独起进程（开关在 import 期生效），两遍取均值
python evaluation/compare_synonym_ablation.py --arm off --tag pass1
python evaluation/compare_synonym_ablation.py --arm on  --tag pass1
# 待审候选集预览（多集可用空格并列）
python evaluation/compare_synonym_ablation.py --arm on --tag preview \
  --eval-set data/evaluation/eval_set_v1.jsonl data/evaluation/eval_set_v1_batch13_candidates.jsonl

# 回答层对比（含引用正确率/拒答准确率，需要 LLM）
SYNONYM_EXPANSION_ENABLED=true python evaluation/run_eval.py --tag b13_on
```

- 产物：`reports/synonym_ablation_<arm>_<tag>.json`（每题 golden_rank / top10 / 扩展词条 / 两路召回统计）。
- 术语表：`data/legal_synonyms_v1.json`（带版本号、负向规则、易混对、known_gaps），改表不用改代码。
- 只跑检索时指标仍有轻微波动（Milvus ANN + 重排服务），所以两遍取均值；上游偶发失败已内置重试（与 run_eval 同口径）。

## 批次 24：`run_eval.py` 按职责拆分（628 → 193 行）

`run_eval.py` 拆分前 628 行，超出项目"单文件 ≤300 行"硬规则（`docs/目录与命名约定.md` §3.4）。
拆成 5 个职责单一的模块（见上方文件表）+ `run_eval.py` 只留"装配 + CLI"。

**硬要求：零逻辑改动。** 两层证明（脚本用完即弃，放在系统临时目录）：

1. **源码逐字比较**：旧文件里每个被搬移的函数/常量，与新模块里同名对象的
   `inspect.getsource` / `repr` **逐字相同**（16 个函数 + 4 个常量，含 `main` 与 `prepare_env`）。
   为此搬移时**连 docstring 都不许改动**（新增说明一律写进模块 docstring）。
2. **行为等价**：拿历史真实 payload 喂给旧实现与新实现，比较
   `summarize` / `worst_samples` / `render_markdown` 的输出 —— 7 份报告（含批次 23 全量 100 条）
   **逐字节一致**。
3. 另跑一遍**全量 100 条 A/B**（拆分前实现 vs 拆分后实现，同一份 backend）逐条对照：
   差异只出现在 LLM 采样会抖的条目上（同批次 23 的取证口径）。

**调用方无需改动**：`python evaluation/run_eval.py` 的用法、参数、产物路径全部不变；
同级模块之间按顶层模块名互相导入（各自把 `evaluation/` 挂上 `sys.path`），
所以脚本入口与 `import evaluation.run_eval` 两种进入方式都拿到同一份模块状态。

## 批次 24：改写器两处调整（都有 A/B 数字）

1. **锚点词（评测侧，只改题目字段）**：`multiturn-003` → `["怀孕的女员工"]`、
   `multiturn-004` → `["离职后不去同行"]`（取自各自第 1 轮原文子串）。题目一字未改。
   正向 `rewrite_rate` 2/4 → **4/4**。
2. **`_compose_query` 去掉「怎么算」尾巴（生产代码）**：10 条 followup_rewrite + 5 条 multi_turn，
   加/不加各跑三轮，逐位复现 —— 加了会让 `multiturn-003`/`004` 的第 2 轮各掉 1 位、
   子集 MRR@10 0.6444 → 0.6278，故去掉。`tests/test_query_rewrite.py` 的两条期望值随之更新。
3. **`_extract_topic` 的"去疑问后缀"分支保留**（批次 23 报告里说它"永不触发"，批次 24 实测**前提不成立**）：
   - `"？"` / `"?"` 两个标点项确实结构性不可达（前一步已把标点换成空格）→ **已删**；
   - 但 7 个词后缀（吗/呢/怎么算/是什么/有哪些/如何/怎么回事）**可达**：
     输入"……女员工吗"（无尾标点）就会命中，`tests/test_query_rewrite.py` 有两条用例正依赖它；
     删整段会让这两条变红、且改写结果出现重复的「怎么算」。故只删死标点，保留循环。
