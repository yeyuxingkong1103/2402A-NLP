# 混合检索评测脚本说明（`测试/混合检索/`）

> 任务编号：**人工智能NLP-RAG-混合检索任务**
> 所属工单：**工单6**（仓库根 `E:\gao6gongdan\工单6`）
> 文档状态：**已按本机实跑复核**（脚本参数、产物路径、复现顺序逐条实跑/实读）
> 最后更新：2026-10

本目录是工单6 的评测层，共 **3 个脚本**。它们都只**读**代码与索引、把结果**写**到
`测试/测试数据/混合检索评测/` 下，不修改任何研发代码或既有索引。

所有命令均须在**仓库根** `E:\gao6gongdan\工单6` 执行，统一走 `run_py.ps1`
（它负责指定解释器并设置 `PYTHONPATH=研发` / `PYTHONUTF8=1` 等环境变量）。

---

## 一、三个脚本速查

| 脚本 | 一句话用途 | 产物 | 实测耗时 |
| --- | --- | --- | --- |
| `run_strategy_eval.py` | 评测主驱动：10 题 × 3 策略 + 多轮回归 + 异常输入 | `测试/测试数据/混合检索评测/<run-tag>/` 下 7 个文件（`截图页/` 由第三个脚本另生成） | 全量 **88.49 秒** |
| `check_evidence_reachability.py` | 穷举扫描证明「缺的那 1 题」是结构不可达 | `测试/测试数据/混合检索评测/证据可达性.json` | 数秒（不调模型） |
| `生成截图页.py` | 把评测 JSON 渲染成自包含 HTML 供截图留痕 | `测试/测试数据/混合检索评测/<run-tag>/截图页/*.html`（5 个） | 1 秒内 |

---

## 二、`run_strategy_eval.py` —— 评测主驱动

### 2.1 用途

对工单指定的 **10 道题**，分别用 **3 种检索策略**（`vector` / `fulltext` / `hybrid`）
各跑一遍，输出准确率、召回率、拒答数、检索与端到端耗时分位；
另附两组回归：**工单5 的五轮多轮序列** 与 **14 类异常输入**。

判分纪律（沿用 工单4/5，**不另造判据**）：

* **准确率**：`测试/common/golden.py::judge_answer` —— 确定性关键字 / 正则判定，**禁止 LLM 自评**；
  判据来自 `研发/data/eval/golden_qa.jsonl`。
* **召回率**：独立指标，用 `common.golden.check_retrieval_hit_layers` 的 `hit_complete`
  （权威层：返回的前 K 条块中含**完整证据原文**）；K 取 `record["min_hit_top_k"]`（golden 恒为 5）。
* **响应时间**：逐轮墙钟毫秒，`use_cache=False` 强制真实执行，不用缓存值充数。

### 2.2 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--run-tag` | `t6-local` | 本次运行标签，**决定输出目录名** |
| `--strategies` | `vector,fulltext,hybrid` | 要评测的策略，逗号分隔 |
| `--ids` | 无（= 工单十题） | 只跑指定题号，逗号分隔 |
| `--limit` | `0`（= 全部） | 只跑前 N 题 |
| `--only` | `all` | `all` / `single` / `multiturn` / `abnormal` |
| `--out-dir` | 无（默认 `测试/测试数据/混合检索评测/<tag>`） | 自定义输出目录 |
| `--no-warmup` | 关 | 跳过预热（**不推荐**） |

工单十题题号（脚本内 `T6_QUESTION_IDS`）：
`5, 6, 1, 2, 3, 4, 260, 95, 33, 34`。

### 2.3 常用命令

```powershell
cd E:\gao6gongdan\工单6

# 全量（10 题 × 3 策略 + 多轮 + 异常；实测 88.49 秒）
pwsh -NoProfile -File run_py.ps1 测试/混合检索/run_strategy_eval.py --run-tag t6-full

# 冒烟：只跑混合检索、只跑前 3 题
pwsh -NoProfile -File run_py.ps1 测试/混合检索/run_strategy_eval.py --strategies hybrid --limit 3 --run-tag smoke

# 只跑多轮回归
pwsh -NoProfile -File run_py.ps1 测试/混合检索/run_strategy_eval.py --only multiturn --run-tag smoke

# 只跑异常输入
pwsh -NoProfile -File run_py.ps1 测试/混合检索/run_strategy_eval.py --only abnormal --run-tag smoke

# 指定题号（例如只验三个被混合检索修正的题）
pwsh -NoProfile -File run_py.ps1 测试/混合检索/run_strategy_eval.py --ids 260,33,34 --run-tag smoke
```

### 2.4 产物

输出目录：`测试/测试数据/混合检索评测/<run-tag>/`

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 每策略的准确率 / 各级召回率 / 拒答数 / 检索与端到端耗时分位；`hybrid_gains`（相对提升、修正与退化题号）；`overall`；`targets`；`reachability`；`meta`（run-tag、题号、预热耗时、总耗时） |
| `results.jsonl` | 逐题逐策略明细（全量 = 30 行 = 10 题 × 3 策略），每行含 `id` / `strategy` / `correct` / `question` / `answer` / `retrieval_ms` / `wall_ms` |
| `qa_results.csv` | 扁平表，便于用表格软件打开 |
| `multiturn.json` | 工单5 五轮序列的回归：`turns_definition` + `by_strategy`（每策略 5 轮的改写策略、改写后查询、答案是否判对） |
| `abnormal.json` | 异常输入（14 用例 × 3 策略 = **42 条**），每条含 `strategy` / `tag` / `ok` / `refused` / `answer_preview` / `wall_ms` |
| `证据可达性.json` | 该轮评测自带的可达性结论（供 `summary.json` 的「仅可达题」口径使用）。**只跑 `--only multiturn` 或 `--only abnormal` 时不会产生**（`single` 与 `all` 才有） |
| `report.md` | 中文汇总报告：结论速览表、达标判定、混合 vs 单路的提升、多轮与异常小结、复现命令 |
| `run_meta.json` | 本轮运行元信息（run-tag、工作单号、生成时间、题号、是否命中缓存、预热耗时、总耗时） |
| `截图页/` | 由 `生成截图页.py` 生成的 HTML（见第四节） |

已存在的两轮产物：`t6-full`（全量正式轮，含 `截图页/`）与 `smoke`（冒烟轮，只有 7 个文件、无 `截图页/`）。

### 2.5 实测结果（`t6-full`，2026-10-07，用时 88.49 秒）

| 检索策略 | 准确率 | 召回率(权威层) | 召回率(仅可达题) | 召回率(诊断层A) | 拒答 | 检索P95 | 端到端P95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vector | 70.0% (7/10) | 60.0% (6/10) | 66.67% (6/9) | 90.0% | 1 | 69.83 ms | 2418.87 ms |
| fulltext | 100.0% (10/10) | 90.0% (9/10) | 100.0% (9/9) | 100.0% | 0 | 13.74 ms | 2269.06 ms |
| hybrid | 100.0% (10/10) | 90.0% (9/10) | 100.0% (9/9) | 100.0% | 0 | 65.67 ms | 2392.43 ms |

* 混合 vs vector：准确率 **+30.00 个百分点**，修正题号 **[33, 34, 260]**，退化题号：**无**。
* 混合 vs fulltext：**+0.00 个百分点**（这 10 题上持平，如实说明）。
* 端到端 P50 **908.72** / P95 **2392.43** / max **2418.87** ms，超 3000 ms 轮次 = **0**。
* 多轮回归：改写策略正确 3/3 策略均 **5/5**；答案正确 vector **4/5**、fulltext **5/5**、hybrid **5/5**。
* 异常输入：**42/42 = 100%**。

---

## 三、`check_evidence_reachability.py` —— 证据可达性穷举核查

### 3.1 用途

`report.md` 里「召回率（权威层）90%」看起来像没达标。要判断这是**检索系统的缺陷**
还是**判定口径的结构性限制**，必须回答一个可证伪的问题：

> golden 要求的「完整证据原文」，在 2676 个已索引块里**到底存不存在**？

本脚本做**穷举扫描**（不做任何检索、不调任何模型）：对每道题的每条 `retrieval_evidence`，
用与 `common.golden.check_retrieval_hit_layers` **逐字相同**的判定谓词，逐块检查该证据能否在
**某一单块**内被完整满足：

* 找得到 ⇒ 该证据「可达」，检索没进 Top-5 才是检索的问题；
* 找不到 ⇒ 该证据「不可达」，**任何**检索策略、任何 Top-K 都不可能命中权威层。

判定谓词（与 golden 严格同口径）：`kind=verbatim` 用归一化后的连续子串；
`kind=chart_pairs|pairs|labels` 要求**每个 pair 的每个分量**都出现在**同一块**里。

### 3.2 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--index-dir` | 默认索引目录（`研发/data/index/bge-m3_1024`） | 要扫描的索引目录 |
| `--ids` | 无（= 工单十题） | 逗号分隔题号 |
| `--out` | 无（脚本内推导为 `测试/测试数据/混合检索评测/证据可达性.json`） | 输出 JSON 路径 |

### 3.3 常用命令

```powershell
cd E:\gao6gongdan\工单6

# 全量（工单十题）
pwsh -NoProfile -File run_py.ps1 测试/混合检索/check_evidence_reachability.py

# 只查指定题号，输出到自定义路径
pwsh -NoProfile -File run_py.ps1 测试/混合检索/check_evidence_reachability.py --ids 4 --out 测试/测试数据/混合检索评测/证据可达性_仅题4.json
```

### 3.4 产物与实测结论

默认输出 `测试/测试数据/混合检索评测/证据可达性.json`。实测结论：

| 项 | 值 |
| --- | --- |
| 题级可达 | **9/10** |
| 证据条可达 | 13/14 |
| 不可达题号 | **[4]** |
| 可达率 | 90.0% |

题号 4 的完整证据原文在**全部 2676 个块里都不存在**连续（或按 golden 配对口径）匹配——
根因是分块结构：golden 要求「表格标题 + 表格正文」连读，而标题落在表格块、正文落在文本块，
**任何单块都不含整段**。

因此检索系统该背的指标是**「可达题上的召回率」**（全文与混合已达 100%），
权威层那个 90% 是**判定口径的上限**，不是检索缺陷。

---

## 四、`生成截图页.py` —— 评测 JSON → HTML 截图页

### 4.1 用途

把评测产物渲染成**自包含 HTML**（内联 CSS、无外部依赖），供截图留痕。
不直接截终端的原因：评测产物是 JSON/CSV，终端截图在长表格上会换行错位，
且中文字体在不同控制台下宽窄不一；渲染成固定宽度的 HTML 再截图，数字与中文都可读，
且页面本身就是可复现的产物。

### 4.2 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--run-tag` | `t6-full` | 要渲染哪一轮的产物 |
| `--run-dir` | 无（默认按 run-tag 推导） | 直接指定产物目录 |

### 4.3 常用命令

```powershell
cd E:\gao6gongdan\工单6
pwsh -NoProfile -File run_py.ps1 测试/混合检索/生成截图页.py --run-tag t6-full
```

### 4.4 产物

输出目录：`测试/测试数据/混合检索评测/<run-tag>/截图页/`，共 **5 个** HTML：

| 文件 | 内容 |
| --- | --- |
| `准确率汇总.html` | 三策略准确率 / 召回率 / 拒答 / 耗时分位汇总表 |
| `响应时间测试.html` | 检索与端到端耗时分位、超 3000 ms 轮次统计 |
| `异常输入测试.html` | 14 用例 × 3 策略的 42 条结果 |
| `多轮对话回归.html` | 五轮序列 × 3 策略的改写与答案判定 |
| `逐题逐策略明细.html` | 10 题 × 3 策略的逐条明细 |

**不要用 `file://` 打开**（浏览器限制且相对路径易错），走 Web 服务的只读路由：

```powershell
# 需先启动服务：pwsh -NoProfile -File run_py.ps1 部署/界面/web_app.py --host 127.0.0.1 --port 8123
Start-Process 'http://127.0.0.1:8123/reports/混合检索评测/t6-full/截图页/准确率汇总.html'
Start-Process 'http://127.0.0.1:8123/reports/混合检索评测/t6-full/截图页/'
```

> `GET /reports/<相对路径>` 只读暴露 `测试/测试数据/**`，带路径穿越防护
> （越界返回 403 `E_FORBIDDEN`）。
> **注意**：直接对 `截图页` 目录发请求会返回一个链接列表页；
> 若地址栏里的 `..` 被浏览器/客户端先行归一化，越界请求会落到 404 而不是 403 —— 两者都是拒绝。

---

## 五、复现顺序（推荐）

整套结论的复现顺序如下，命令可逐条复制执行：

```powershell
cd E:\gao6gongdan\工单6

# ① 单元测试基线（实测 255 passed，约 2.4 秒）
pwsh -NoProfile -File run_py.ps1 -m pytest 研发/tests -q

# ② 证据可达性穷举（不调模型，数秒）
pwsh -NoProfile -File run_py.ps1 测试/混合检索/check_evidence_reachability.py

# ③ 主评测（10 题 × 3 策略 + 多轮 + 异常；实测 88.49 秒）
pwsh -NoProfile -File run_py.ps1 测试/混合检索/run_strategy_eval.py --run-tag t6-full

# ④ 渲染截图页（5 个 HTML）
pwsh -NoProfile -File run_py.ps1 测试/混合检索/生成截图页.py --run-tag t6-full
```

顺序说明：

1. **两个脚本各自独立算一遍可达性，没有先后依赖**，但**两个都建议跑**：
   * `run_strategy_eval.py` 内部会**直接 import 并调用** `check_evidence_reachability.check()`
     （扫描目录固定为 `<索引根>/bge-m3_1024`），所以每轮评测都会自带可达性结论，
     并写进该轮目录的 `证据可达性.json` 与 `summary.json` 的 `reachability` 字段。
     这一步失败**不阻断**评测（只打印「跳过」并把「仅可达题」一列留空）。
   * 独立的 `check_evidence_reachability.py` 写的是**共享的规范结论**
     `测试/测试数据/混合检索评测/证据可达性.json`（`--out` 可改），
     供人工复核与报告引用。
2. **③ 的 `--run-tag` 决定目录名**，后续 ④ 必须用同一个 tag。
3. ③ 会真实调用本地 Ollama（嵌入 + 生成）。请在评测期间**独占 Ollama**，
   不要并发跑其它问答脚本，否则耗时数据不可用。
4. ③ 内部会先做预热（实测 8360.67 ms），所以不必单独预热。

跑完后建议核对的位置：

| 要核对什么 | 看哪里 |
| --- | --- |
| 三策略准确率 / 召回率 / 拒答 / 耗时 | `测试/测试数据/混合检索评测/t6-full/summary.json` 的 `per_strategy` |
| 混合相对单路的提升与修正/退化题号 | 同上，`hybrid_gains` |
| 逐题逐策略明细 | `results.jsonl`（30 行）或 `qa_results.csv` |
| 多轮回归 | `multiturn.json` 的 `by_strategy` |
| 异常输入 | `abnormal.json` 的 `cases`（42 条） |
| 汇总报告 | `report.md` |
| 可截图页面 | `截图页/` 下 5 个 HTML |

---

## 六、口径与边界（如实登记）

* **权威层召回率 90% 不等于「检索故障」**：题号 [4] 结构不可达，由
  `check_evidence_reachability.py` 穷举证明；该脚本输出必须与报告同时给出。
* **混合 vs 全文在这 10 题上持平（+0.00 个百分点）**：如实登记，不粉饰。
  混合的价值是「不弱于任一单路，并在单路失效时兜底」。
* **不得声称「策略层默认 == 工单4 基线」**：不传 `strategy` 时走工单4 原生 `hybrid_search`，
  行为与基线一致；一旦显式传策略就走新链路，两者**不等价**。
* 本目录脚本不修改研发代码与既有索引；所有写入都落在
  `测试/测试数据/混合检索评测/` 下。
