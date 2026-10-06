# 批次 23 验收报告：multi_turn 评测接入 + 收尾三件

> 边界遵守：不改 `docs/`（环境规范已由你写入）；临时脚本放系统临时目录（未进 `backend/`）；
> 不写任何 key；做完停下汇报。

---

## 结论速览

| # | 事项 | 结果 |
|---|---|---|
| 【多轮题裁决】 | 4 条题目 + 备用 1 条落地评测集（95 → **100 条**） | ✅ 5 条全部写入，golden 逐条 SQL 核对在 approved 版本 |
| 【run_eval 5 项改动】 | multi_turn 分支 / 每轮回写 / 记录改写 / 两项判定 / try-finally 清理 | ✅ 全部落地，另加 `--no-context-writeback` 反向对照开关 |
| 【正反对照】 | 正向 rewrite_rate **2/4**、turn2 top5 **1.0**；反向 **0/4**、**0.8** | ✅ 反向两数都掉，测试有效 |
| 【全量评测】 | 100 条不回归 | ✅ 既有 95 条 Recall@5 / 引用正确率 / 拒答准确率**逐位一致**；13 条回答侧翻转经三次运行对照证明是 LLM 采样（详见第四节） |
| 【1】`probe_fetch_pdf.py` 升格 | → `scripts/e2e/fetch_pdf_samples.py` + README | ✅ 真实重下 4 份，页数/文字层 4/4 精确命中，退出码 0 |
| 【2】`requests` 进 requirements | `requests==2.34.2`（rag 环境实测已装） | ✅ dry-run 自洽，退出码 0 |
| 【3】Qwen-VL 无 Markdown 层级 | 不修，登记为已知限制 | ✅ 写入 `data/pdf_samples/README.md` + 批次 22 报告第七节 |
| 【终验】 | conda `rag` 环境全量测试 + check_services | ✅ 598 passed / check_services 退出码 0 / 全仓 import 96 文件全绿 |

> 说明：全量 100 条那次是在"单文件拆分"之前启动的（拆分是**纯文件搬移、逻辑零改动**），
> 拆完后我用终版代码把正反两轮**重跑了一遍** —— 逐题结果与拆分前**完全一致**（见第三节末），
> 故全量数据仍有效。

**两个必须先说的结论**（与你的预期不一致，有实测证据）：

1. **approved 的锚点词有 2 条在真实改写输出里不可能出现** —— 003 期望「女职工/产假」、
   004 期望「竞业限制」，但改写器注入的主题词是**上一轮问题原文**（不是名词短语），
   上轮问的是「怀孕的女员工」「离职后不去同行」，所以锚点命不中。正向 `rewrite_rate = 2/4`
   是**锚点词的问题，不是链路的问题**（同一批数据里 001~004 四条的 `context_carryover` 全是 True，
   即改写后查询确实带上了上一轮问题原文）。
   两条修法见第七节，都很小，等你选。
2. **`multiturn-005` 的第 2 轮按设计不触发改写** —— 「这笔钱从什么时候开始算…」既不以
   `那/这个/上述/前面` 开头，也不以指代词结尾，过不了改写器的指代信号闸门。已如实按
   "无 `rewrite_expect`" 落地（不进 `rewrite_rate` 分母），只计入多轮检索命中。

---

## 一、题目落地（评测集 95 → 100）

新增 5 条 `multi_turn`（4 条正式 + 1 条备用），题面/锚点词**逐字按你批准的草案**，未改一字：

| id | 第 1 轮 | 第 1 轮 golden | 第 2 轮 | 第 2 轮 golden | 锚点词 | `rewrite_expect` |
|---|---|---|---|---|---|---|
| `multiturn-001` | 签三年期的劳动合同，试用期最长不能超过多久？ | 劳动合同法 第十九条 | 那这个期间工资最低不能低于多少？ | 劳动合同法 第二十条 / 实施条例 第十五条 | 试用期 | ✅ turn 2 |
| `multiturn-002` | 工作满十年，年休假有几天？ | 职工带薪年休假条例 第三条 | 那些天没休成，钱按什么标准算？ | 职工带薪年休假条例 第五条 | 年休假 | ✅ turn 2 |
| `multiturn-003` | 公司可以辞退怀孕的女员工吗？ | 女职工劳动保护特别规定 第五条 | 那她能休多久？ | 女职工劳动保护特别规定 第七条 | 女职工 / 产假 | ✅ turn 2 |
| `multiturn-004` | 公司要我在离职后不去同行，需要给我钱吗？ | 劳动合同法 第二十三条 | 那这个最长能约定几年？ | 劳动合同法 第二十四条 | 竞业限制 | ✅ turn 2 |
| `multiturn-005`（备用） | 公司一直没跟我签书面劳动合同，我能要双倍工资吗？ | 劳动合同法 第八十二条 | 这笔钱从什么时候开始算，算到什么时候为止？ | 实施条例 第六条 | — | 无（按设计不触发改写） |

- 每条 `turns` 里的 `golden` 已逐条 SQL 核对存在于 approved 版本（`document_chunks` +
  `document_versions.version_status='approved'`）；
- `source_quote` 全部取自库内父块正文原文；
- 类型分布：`direct_article 38 / refusal 15 / cross_law 14 / followup_rewrite 10 / as_of_date 9 /
  confusable 9 / multi_turn 5 = 100`；
- `data/evaluation/_schema.md` 已同步：新增 `multi_turn` 类型与 `turns` / `rewrite_expect` / `notes`
  字段说明、三个新指标口径、正反对照口径。

---

## 二、`run_eval.py` 改造（5 项 + 对照开关）

**为什么把逻辑放新文件**：`run_eval.py` 已 569 行（项目硬规则"单文件 ≤300 行"，
见 `docs/目录与命名约定.md` §3.4 与 §七拆分清单），继续往里堆不合适。故新增两个模块，
按职责拆开（都守住 300 行）：

| 文件 | 行数 | 职责 |
|---|---|---|
| `evaluation/multi_turn_grading.py` | 190 | **纯函数**：`rewrite_hit` / `context_carryover` / `merge_citation_audits` / `summarize_multi_turn` / `render_multi_turn_section` / 会话清理 |
| `evaluation/multi_turn_runner.py` | 255 | 多轮执行循环：逐轮检索+问答、回写短期记忆、`try/finally` 清会话、整题 `_rollup` |
| `evaluation/run_eval.py` | 569 → **628**（+59） | 只加接线：分支调用、指标汇总、`--no-context-writeback`、渲染调用 |

判定原语（`retry` / `citation_audit` / `refused_by_text` / `golden_rank`）由 `run_eval` **注入**，
**口径仍是同一份实现**，不复制逻辑；渲染与指标口径也同在一文件，改一处不会漏改另一处。

| # | 你的要求 | 落地位置 |
|---|---|---|
| 1 | 新增 `multi_turn` 分支按 `turns` 逐轮跑 | `run_eval.py` main 循环的 `if item.get("type") == "multi_turn"` → `multi_turn_runner.run_multi_turn_item` |
| 2 | 每轮结束后回写短期记忆（不写 MySQL） | `run_multi_turn_item` 第 3 步：`append_message(user) + append_message(assistant)`，窗口由 `ShortTermMemoryStore(max_messages=20)` 决定，与生产一致 |
| 3 | 记录每轮 `query_rewrite`（保留 `rewritten_query` 原文） | `turn_detail["rewrite"] = {changed, original_query, rewritten_query, reasons}` |
| 4 | 新增 `rewrite_hit` / `turn_hit_at_5` 两项判定 | `rewrite_hit()`：`changed == expect_changed` **且**锚点命中 **且** 反向断言不命中；`_rollup()` 里出整题 `turn_hit_at_5` |
| 5 | 会话清理放 `try/finally`，异常也要删 Redis key | `run_multi_turn_item` 的 `finally` → `cleanup_eval_session`；`run_eval` 主循环每个题目**再兜一次**清理 |
| + | （追加）不得残留 eval 会话 | 除跑完清，**跑前也清一次**（清掉上次 Ctrl+C/被杀留下的残留）；结果记进 `session_cleanup_before/after` |
| + | （新增）反向对照开关 | `--no-context-writeback`（`action="store_true"`） |

**离线冒烟（替身驱动，不联网不连 Redis）**：**40 项断言全过**
（脚本与输出：`reports/b23_smoke.txt`，退出码 0）—— 覆盖 `rewrite_hit` 六种组合、
`context_carryover`、引用审计合并、清理失败降级（不抛）、异常路径（第 1 轮 chat 抛错仍清会话）、
空 `turns`、回写后窗口条数、跑前/跑后各清一次、以及与既有 `summarize()` 的兼容性
（老题型 Recall@5 不受影响）。
其中"整题引用合并"首次断言写错（我把每轮 1 条引用误当 2 条），**是我的测试期望值错，代码无误**，
已改正后全绿。

---

## 三、正向 / 反向对照（真跑，同一批 5 条题）

```bash
# 正向（回写开启）
python evaluation/run_eval.py --only multiturn- --tag b23_mt_on
# 反向（禁回写）
python evaluation/run_eval.py --only multiturn- --no-context-writeback --tag b23_mt_off
```

| 指标 | 正向（回写开） | 反向（禁回写） | 说明 |
|---|---|---|---|
| `rewrite_rate` | **0.5000（2/4）** | **0.0000（0/4）** | ⬇ 掉到 0：改写判定**确实依赖上下文**，测试有效 |
| `turn2_hit_at_5` | **1.0000（5/5）** | **0.8000（4/5）** | ⬇ 掉 1 条：`multiturn-004` 第 2 轮 golden 从 rank 3 变为**未召回** |
| `context_carryover`（只记录） | **4/5 接通** | **0/5** | 正向 001~004 都真带上了上一轮问题正文；005 本就无改写 |
| `changed=True` 的轮次 | 4（001~004 的第 2 轮） | 0 | 反向 4 条断言题 `changed` 全为 False |

**逐题对照**

| 题号 | 第 2 轮问题 | 正向改写后查询（原文） | 正向 top5 | 反向 changed / top5 |
|---|---|---|---|---|
| 001 | 那这个期间工资最低不能低于多少？ | `签三年期的劳动合同 试用期最长不能超过多久 这个期间工资最低不能低于多少 怎么算` | ✔ rank 1 | False / ✔ rank 1 |
| 002 | 那些天没休成，钱按什么标准算？ | `工作满十年 年休假有几天 些天没休成，钱按什么标准算 怎么算` | ✔ rank 2 | False / ✔ rank 1 |
| 003 | 那她能休多久？ | `公司可以辞退怀孕的女员工吗 她能休多久 怎么算` | ✔ rank 4 | False / ✔ rank 2 |
| 004 | 那这个最长能约定几年？ | `公司要我在离职后不去同行 需要给我钱吗 这个最长能约定几年 怎么算` | ✔ rank 3 | False / **未召回** |
| 005 | 这笔钱从什么时候开始算，算到什么时候为止？ | 无改写（无指代信号，按设计） | ✔ rank 3 | False / ✔ rank 3 |

**两点值得记下**：

1. `multiturn-004` 是唯一能区分正反的题 —— 第 2 轮「那这个最长能约定几年？」离开上文几乎无法检索，
   反向直接掉出 top10；001/002/003 的第 2 轮即使不改写也还能命中（问题本身自足性够），
   所以 `turn2_hit_at_5` 从 1.0 只掉到 0.8。**若要更强的判别力，需要更多"纯指代"题**（见第七节）。
2. **`multiturn-004` 的第 1 轮本身就是真实弱点**：口语化的"公司要我在离职后不去同行，需要给我钱吗？"
   让检索侧直接**拒答**（`refused=True`、引用 0 条、golden 第二十三条未进 top10），
   而第 2 轮改写（带上了第 1 轮问题）反而命中了第二十四条 rank 3。
   即：**首轮没有上文可用时，口语化问法是裸的** —— 这属于同义词/口语扩写（批次 13 开关）的靶心，
   与本批无关，登记为观察项。

**终版复跑（拆分后代码，验证拆分零改动）**：

```
正向：改写命中率=0.5（分母 4）  第2轮 top5=1.0（分母 5）  回写=开   退出码 0
反向：改写命中率=0.0（分母 4）  第2轮 top5=0.8（分母 5）  回写=关   退出码 0
逐题 rewrite_hit / carryover / rank：001(T,T,1) 002(T,T,2) 003(F,T,4) 004(F,T,3) 005(-,F,3)
```
与拆分前**逐题完全一致**（报告：`reports/eval_*_b23_mt_on_final.json` / `..._off_final.json`）。

---

## 四、全量评测（100 条）与不回归

### 4.1 做法

基线用昨天的全量报告 `reports/eval_20260920_194916_baseline95.json`（95 条，
Recall@5=0.9625 / MRR@10=0.7776 / 引用=1.0000 / 拒答=0.9333）。

不回归核验**不靠眼看数字**：
1. 用 `run_eval.summarize()` 这**同一份实现**，对本次全量结果里的"既有 95 条"子集重算四项指标，
   与基线直接比（口径同源，避免我手写算法引入偏差）；
2. 再逐条比 `golden_rank` / `hit_at_5` / 越界引用 / 误拒 / 异常。

```bash
python evaluation/run_eval.py --tag b23_full100          # 100 条，退出码 0
python <不回归核验脚本>                                    # 输出 reports/b23_nonregression.txt，退出码 0
```

### 4.2 既有 95 条：指标对照

| 指标 | 基线 95 条 | 本次同 95 条 | 一致 |
|---|---|---|---|
| Recall@5 | 0.9625 | **0.9625** | ✔ |
| MRR@10 | 0.7776 | 0.7755 | ✗ 差 −0.0021 |
| 引用正确率 | 1.0000 | **1.0000** | ✔ |
| 拒答准确率 | 0.9333 | **0.9333** | ✔ |
| 误拒率 | 0.1500 | 0.1125 | ✗ 差 −0.0375 |
| 计分题数 / 拒答题数 | 80 / 15 | 80 / 15 | ✔ |
| 越界引用 | 0 | 0 | ✔ |
| 引用总数 | 351 | 362 | ✗ |
| 无引用回答 | 12 | 9 | ✗ |

### 4.3 逐条差异：13 条，且**没有一条是"检索侧回归"**

| 类别 | 条数 | 明细 |
|---|---|---|
| golden_rank 漂移（均在 top5 内） | **1** | `followup-072` 2 → 3 |
| `hit_at_5` 变化 | **0** | — |
| 越界引用变化 | **0** | — |
| `refusal_by_text` 翻转 | **13**（含上条那 1 条） | direct-004/009/012/016、cross-021/066、asof-033/074、confuse-045、direct-057/058/090、followup-072 |

### 4.4 这些差异不是本批引入的 —— 三重证据

**证据 1（代码路径）**：`multi_turn` 分支只对 `type == "multi_turn"` 生效；其余题走的是
**一字未改的原 `run_item(...)`**（`evaluation/run_eval.py:572-578`，只是被包进 if/else，
后面多了一行清理调用）。本批没有碰检索、改写、提示词、护栏任何一行。

**证据 2（差异的形态）**：13 条里的 12 条只翻 `refusal_by_text`，而 `refusal_by_text` 是**纯回答侧判定**。
看基线当时的回答内容就明白了 —— 这些题的基线答案是**护栏兜底文案**：

```
direct-012 基线（134 字，引用 0 条 / sources 5 条 / refused_flag False）：
  抱歉，本次回答未通过引用校验：回答缺少可核查的法条引用，无法保证内容可靠性。……
direct-012 本次（609 字，引用 7 条）：
  遇到这种情况确实挺让人头疼的。根据你提供的法源，你可以主张的主要是这几块：……
```

统计两轮里出现该兜底文案的题数：**基线 10 条 / 本次 7 条，但两次重叠只有 2 条**
（`cross-067`、`direct-051`）—— 8 条消失、5 条新增。这是"LLM 这一轮有没有给出 `[n]` 引用"
的**逐题随机性**（给了就正常作答，没给就被护栏兜底），不是结构性问题：
真要是代码回归，差异会按题型聚集，而不是 8 出 5 进、只有 2 条稳定。

**证据 3（同样的代码跑第三遍）**：把那 13 条单独抽出来**再跑一遍**（同一份代码、同一份库，13 条 ≈1.5 分钟）：

| 结果 | 条数 |
|---|---|
| 重跑后**回到基线值** | 5 |
| 出现"基线→变→又变"来回跳 | 5 |
| golden_rank 有过变化 | 1（`followup-072` 2→3→3） |

明细见 `reports/b23_stability.txt`。若是本批引入的确定性回归，重跑应稳定复现同一结果；
实测多数回到基线或来回跳，属上游（Embedding / 重排 / LLM）波动。
（另：这次子集重跑日志里恰好有 1 次 `重排服务不可用，已退回 RRF 融合顺序：RerankerApiError`，
也说明重排服务本身会抖 —— 全量 100 条那轮 0 次降级、正反两轮 0 次。）

### 4.5 全量 100 条总指标（口径与基线不同，仅供对照）

| 指标 | 数值 |
|---|---|
| Recall@5 | 0.9647（计分 85 条） |
| MRR@10 | 0.7583 |
| 引用正确率 | 1.0000（越界 0） |
| 拒答准确率 | 0.9333（15 条拒答题） |
| 误拒率 | 0.1059 |
| 无引用回答 | 9 |

新增 5 条 `multi_turn` 在这轮全量里的表现：`rewrite_rate=0.5（分母 4）`、
`turn2_hit_at_5=1.0（分母 5）`、`context_carryover=4/5`，逐题 golden_rank 为 1 / 2 / 4 / 3 / 3。

**结论**：检索侧与判定口径**不回归**（Recall@5、引用正确率、拒答准确率逐位一致，0 条
`hit_at_5` 变化、0 条越界引用）；回答侧 13 条翻转经三重证据判定为 LLM 采样波动，与本批改动无关。

---

## 五、终验：conda `rag` 环境全量测试 + 服务自检

全部用 `C:/Users/92842/anaconda3/envs/rag/python.exe`（§七 规则 7「运行环境唯一化」）：

```
########## ① 全量测试（conda rag）##########
598 passed, 4 warnings in 13.93s

########## ② 服务自检 ##########
=== 结果 ===
✅ 全部通过（退出码 0）
退出码=0

########## ③ 全仓 import 校验（清理完 probe_fetch_pdf.py 后复跑）##########
扫描文件数: 96（backend/app 由 pytest 覆盖，此处跳过）
✅ 全部 app.* 导入均可解析（模块存在 + 符号存在）
退出码=0
```

数量说明：测试 598 条与批次 22 持平（本批未新增后端测试，新增逻辑的验证靠
①离线冒烟 40 项断言 + ②正反两轮真实对照 + ③全量不回归逐条核验）；
import 校验 94 → **96** 文件（+3 新模块 `multi_turn_runner.py` / `multi_turn_grading.py` /
`fetch_pdf_samples.py`，−1 被升格掉的 `probe_fetch_pdf.py`）。

**项目根已清干净**：只剩 `.env` / `CLAUDE.md` / `README.md`。
`probe_fetch_pdf.py` 已随升格移除 —— ⚠️ **项目无 `.git`，删除不可回滚**，
所以我先做了 md5 一致的备份（`C:\Users\92842\AppData\Local\Temp\b23_dev_bak\probe_fetch_pdf.py`，
md5 `7603b470419c913854aba978496cefc8`）再走回收站；两个新生成的 `__pycache__` 一并清掉。
（`SHFileOperationW` 返回码仍是 2，但三项目标均已确认消失、备份可核对 —— 与批次 22 同一现象。）

**Redis 残留**：`scan('short_memory:*')` 里含 `eval_` 的 key = **0 条**
（本批累计跑了 3 遍 multi_turn —— 全量那遍 + 正反各一遍，会话全部清干净）。
另见第七节第 5 条：11 个非本批产生的历史残留 key，待你决定是否清。

---

## 六、收尾三件

### 1) `probe_fetch_pdf.py` → `scripts/e2e/fetch_pdf_samples.py` ✅

项目根那个一次性探针升格成长期脚本：加 `argparse`（`--out-dir/--force/--verify-only/--timeout`）、
退出码（0 通过 / 1 与预期不符 / 2 前置不满足）、并**把"下载"和"校验"合并成一个动作** ——
预期页数与文字层写在脚本 `TARGETS` 常量里，与 `data/pdf_samples/README.md` 清单表两处对应。

真实重下 4 份（`--force`，走网络）实测：

```
--- 中华人民共和国劳动合同法                      下载：OK（282,029 B）
    校验：页数 27（预期 27）✔    文字层 12,375 字 / 逐页不 strip（预期 12,375）✔
--- 中华人民共和国劳动合同法实施条例              下载：OK（261,817 B）
    校验：页数 10（预期 10）✔    文字层 4,617 字 / 逐页不 strip（预期 4,617）✔
--- 中华人民共和国劳动法                          下载：OK（273,566 B）
    校验：页数 21（预期 21）✔    文字层 8,952 字 / 逐页不 strip（预期 8,952）✔
--- 劳动合同（通用）示范文本（人社部编制版）      下载：OK（150,930 B）
    校验：页数 7（预期 7）✔    文字层 3,591 字 / 逐页不 strip（预期 3,591）✔
✅ 4 份样本全部校验通过（页数与文字层均与 README 清单一致）   退出码=0
```

**顺带发现并修正了一个口径坑**：我第一版脚本按"逐页 `get_text().strip()`"统计文字层，
结果每份都比 README 少**正好 1 字/页**（27 页少 27 字），第一眼像"样本变了"。
实测三种口径（逐页 strip / 逐页不 strip / 拼接后 strip）后确认 README 与批次 22 报告
用的是**逐页不 strip**，已把脚本改成同口径（现在 4/4 精确命中、偏差 0），
并在两边文档写清"别改成 strip 口径，否则会被误读成样本变化"。
`scripts/e2e/README.md` 已补该脚本的作用/前置/可重复性/清理与退出码。

### 2) `requests` 补进 requirements ✅

- rag 环境实测**已装** `requests 2.34.2`（不是 base 的版本），按其版本锁 `==`；
- 用在哪：`scripts/e2e/e2e_auth_persist.py` / `e2e_create_admin.py` / `e2e_review_publish.py`
  三个 HTTP 客户端脚本（真实打本机后端）；**后端本体与评测链路一律用标准库**
  （`urllib` / `http.client`），所以注释里写清了"为 e2e 脚本而加"；
- `pip install --dry-run -r backend/requirements.txt` → 全部 `already satisfied`、退出码 0。

### 3) Qwen-VL 兜底输出无 Markdown 层级 → 登记不修 ✅

写进 `data/pdf_samples/README.md`「已知限制」+ 批次 22 报告第七节第 3 条（并把已修的
`page_count` 那条同步标记为"批次 23 已修"）。登记内容：MinerU 返回 `full.md` 有层级，
Qwen-VL 只吐纯 OCR 文本，兜底分支的 38 条/19 项完全靠文本形态切出，章节标题只能靠 cleaner
白名单保住；**不修的理由**是兜底属故障降级路径，在纯文本上做启发式结构补全误判风险大于收益。

---

## 七、待你裁决

1. **锚点词（唯一挡在 4/4 前面的东西）** —— 003 期望「女职工/产假」、004 期望「竞业限制」，
   但改写器注入的主题词是**上一轮问题原文**，上轮口语是"怀孕的女员工"/"离职后不去同行"，
   故锚点命不中。两条修法：
   - **方案 A（推荐，改锚点词，不碰题面）**：003 → `["怀孕的女员工"]`、004 → `["离职后不去同行"]`；
     零风险、立刻 4/4，且锚点仍能证明"上下文被带进来了"。
   - **方案 B（改题面，不碰锚点词）**：把第 1 轮换成规范术语（"竞业限制"/"女职工产假"），
     但会牺牲"口语化真实问法"这个出题意图（真实 HR 就是这么问的）。
   我建议 A。**你点头我就改**（一条命令、只动 `must_contain_any`）。
2. **反向判别力偏弱**：反向 `turn2_hit_at_5` 只掉 0.2（只有 004 掉出）。要不要补 2~3 条
   "纯指代"题（如「那这个呢？」「它怎么算？」这类无自足性的第 2 轮），让反向掉得更狠？
   要的话我出草案交审（不直接写集）。
3. **改写器的两个观察项（本批未动）**：
   - `_extract_topic` 里"去掉疑问后缀"的分支**实际永不触发** —— 它前面的标点替换先把
     "？" 换成了空格，`endswith("吗"/"？")` 就永远不成立，所以主题词里残留"多久/几天/吗"；
   - `_compose_query` 对非疑问结构一律追加「怎么算」，于是出现
     `… 这个期间工资最低不能低于多少 怎么算` 这种尾巴（语义上无害，但检索词噪音增加）。
   两者都不影响本批结论，修不修由你定（改动会波及既有 10 条 `followup_rewrite` 的改写结果）。
4. **`run_eval.py` 超 300 行**（改造后 **628 行**，实测）：本批已把多轮逻辑外移，但它自己仍超线。
   要不要另开一批做"运行器拆分（判定/渲染/汇总各自成模块）+ 全量回归"？
5. **Redis 残留（不是本批产生，顺手报备）**：`scan('short_memory:*')` 有 **11 个非评测残留 key**
   —— 10 个是 `ablation_runner:ablation_followup-*`（`compare_synonym_ablation.py` 探针，
   TTL 约 3.2 小时后自然过期），1 个 `u-verify-5c:s-5c:messages` **TTL = -1（永不过期）**，
   内容只有 1 条 user 消息「经济补偿怎么算」（7 字）。要清我就清（会先报备清单再动）。
   **本批自己的 `eval_*` key：0 残留**（实测 `含 eval_ 的 key = []`）。

---

## 八、复现命令与证据文件

```bash
PY=C:/Users/92842/anaconda3/envs/rag/python.exe   # 项目唯一正式环境（§七 规则 7）

# ① 正反两轮对照（终版代码）
python evaluation/run_eval.py --only multiturn- --tag b23_mt_on_final
python evaluation/run_eval.py --only multiturn- --no-context-writeback --tag b23_mt_off_final

# ② 全量 100 条
python evaluation/run_eval.py --tag b23_full100

# ③ 不回归逐条核验（对"既有 95 条"用同一份 summarize 重算 + 逐条比 rank/引用/误拒）
python <不回归核验脚本>

# ④ 13 条差异题的第三次运行（稳定性对照，需先用差异 id 生成子集 jsonl）
python evaluation/run_eval.py --eval-set <子集.jsonl> --tag b23_flip13

# ⑤ 收尾三件
python scripts/e2e/fetch_pdf_samples.py --force     # 重下并校验 4 份 PDF
python scripts/e2e/fetch_pdf_samples.py --verify-only
pip install --dry-run --no-input -r backend/requirements.txt

# ⑥ 终验
cd backend && PYTHONPATH= python -m pytest tests -q --no-header -p no:cacheprovider \
  --basetemp="C:/Users/92842/AppData/Local/Temp/pytest_b23_final"
python scripts/check_services.py
python scripts/check_imports.py
```

| 证据文件 | 内容 |
|---|---|
| `reports/eval_*_b23_full100.json` / `.md` | 全量 100 条明细与分类型指标 |
| `reports/eval_*_b23_mt_on_final.json` / `..._off_final.json` | 终版正反两轮明细（含每轮 `rewritten_query` 原文） |
| `reports/b23_nonregression.txt` | 不回归逐条对照（退出码 0） |
| `reports/b23_stability.txt` | 13 条差异题的第三次运行对照 |
| `reports/b23_smoke.txt` | 离线冒烟 40 项断言（退出码 0） |
| `reports/b23_full100.log` / `b23_flip13.log` / `b23_mt_forward.log` / `b23_mt_reverse.log` / `b23_mt_final.log` | 各次运行原始日志（含重排降级等告警） |
