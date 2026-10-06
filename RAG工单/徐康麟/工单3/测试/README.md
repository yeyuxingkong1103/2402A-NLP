# 测试/ —— 离线 / 在线 / 用户 三级测试套件

> 工单：**人工智能NLP-RAG-PDF文档的表格解析及检索优化**

## 1. 目录约定

| 路径 | 用途 | 是否依赖服务 |
| --- | --- | --- |
| `离线/` | 确定性测试：14 题固定输入自洽性、表格归一化锚点、分块元数据、BM25 可复现、配置契约、结构化日志、分类口径（N-7/N-8） | ❌ 不需要 Ollama |
| `在线/` | 端到端：14 题混合检索 top-5 命中、问答准确率、引用可回溯、**每题**首字、多轮+中英文、HTTP/SSE、启动预热 | ✅ 需要 Ollama |
| `用户/` | 用户视角：不可答负例、同页互污染（N-5）、闸门 fail-open（N-6）、编造红线（N-4/⑨）、两套界面 | ✅ 需要 Ollama |
| `common/` | **判定口径唯一实现**（见 §3）：`assertions` / `golden` / `negatives` / `paths` / `pdf_probe` / `artifacts` / `reports` / `online_gate` | —— |
| `测试数据/` | 固定输入与期望：`golden_qa_14.jsonl`、`unknown_questions.jsonl`、`leakage_cases.jsonl` + 逐字取证副本 | —— |
| `留痕/` | 执行留痕（原始 stdout、JSON 报告）。**tester 独占写目录** | —— |

## 2. 运行方式（工作目录 = `E:\gao6gongdan\工单3`）

```powershell
# 三级全跑（含 simulate_user），原始 stdout 自动落 测试/留痕/
pwsh -NoProfile -File 测试/run_tests.ps1

# 单级
pwsh -NoProfile -File 测试/run_tests.ps1 -Tier offline
pwsh -NoProfile -File 测试/run_tests.ps1 -Tier online
pwsh -NoProfile -File 测试/run_tests.ps1 -Tier user

# 等价的分段命令（验收 9 的「三命令重跑」）
pwsh -NoProfile -File run_py.ps1 -m pytest 测试/离线 -v
pwsh -NoProfile -File run_py.ps1 -m pytest 测试/在线 -v
pwsh -NoProfile -File run_py.ps1 测试/用户/simulate_user.py
```

注意：`run_py.ps1` 的参数是 `ValueFromRemainingArguments`，**不要**传 `-p no:cacheprovider` 这类单字母参数
（会被 PowerShell 当成 `run_py.ps1` 自己的参数名而报错）。

在线/用户级在 Ollama 不可用时**默认判失败**（不允许「跳过即通过」）；只有在明确知道自己在做什么时，
才可设 `RAG_TEST_ALLOW_ONLINE_SKIP=1` 降级为 skip。

## 3. 判定口径（唯一实现：`common/assertions.py`）

同一口径若各测试文件各写一份必然漂移，最终把**正确实现判成失败**。因此：

| 口径 | 实现 | 关键约束 |
| --- | --- | --- |
| 命中 | `assertions.evidence_hit` → `retrieval_utils.is_evidence_hit` → `text_utils.evidence_contains` | **禁止**用「引用页 == `evidence_pages`」判命中（会把 957 误计为命中）；题 207 优先 `evidence_verbatim` |
| 引用可回溯 | `assertions.check_citation_traceable` | 1-based **物理页**、该页真实含支撑原文；抓 0-based 错页；页脚 `1-1-128` 与裸数字 `128` 禁止用作页码 |
| 首字 | `assertions.check_first_token` | **逐题** ≤ 3000 ms，取最大值；冷启动不豁免（预热缺失即缺陷） |
| 表格缺陷 | `assertions.table_horizontal_defects` | **只认行内横向重复**；跨行纵向同值不算；无数据占位符（`-` 等）不算 |
| 非缺陷白名单 | `assertions.non_defect_verdicts()` | 退化表不进索引 / 引用页 ≠ evidence_pages / 闸门 `counted=False` / `[ ◆ ]` =「未披露」/ 计数口径差异 / 冷启动 |
| 判分 | `assertions.judge_answer` → `evaluator_bridge.check_answer` | 门槛 ≥13/14 = 92.9%；加载失败才用本地副本并在理由里标注来源 |
| RAGAS | `assertions.ragas_banner()` | 固定原文「RAGAS 未运行（依赖不可用，本机断网）」；`reports.assert_no_ragas_numbers` 禁止任何 RAGAS 数值字段 |

## 4. 14 题与负例集

* `测试数据/golden_qa_14.jsonl`：14 题固定输入，含 `evidence`（判定用）、**`evidence_verbatim`（逐字原文，命中优先用它）**、
  `citation_quote`（引用回溯核验）、`evidence_pages`（1-based 物理页）、`required/forbidden_substrings`。
  id 260/95/33/34/957/793/795/543/531/207 沿用工单1 golden；id 1~4（力源信息 / PDF2）由本目录从真实页逐字取证。
  题 207 的 golden `evidence` 是**合成串+编辑注记**（14 题中唯一非逐字原文）→ 该题命中判定必须用 `evidence_verbatim`。
* `测试数据/unknown_questions.jsonl`：不可答负例（含 N-4 编造红线、语料外主体、时间范围外）。
* `测试数据/leakage_cases.jsonl`：同页互污染与要素齐备（N-5①/②/③、N-6、跨语料串题、占位符口径）。
* `测试数据/pdf1_evidence_pages.txt` / `pdf2_evidence_pages.txt`：逐字取证的**只读提取副本**（含物理页号），供复核。

> id 3/4 的**题面**在 `设计/需求分析.md` §4.2 里是摘要（id 1/2 有原文）。本目录按同语义还原了完整问句并
> 在 fixture 的 `question_source` 字段标注，等待 captain 裁定后同步 T9 的 `golden_qa.jsonl`。

## 5. 留痕与报告纪律

* 执行留痕落 `测试/留痕/`：`*_<时间戳>.txt`（run_tests.ps1 的原始 stdout）、`online_*.json`（在线逐题结果）、
  `user_*.json`、`simulate_user.txt`、`run_tests_<时间戳>.json`（汇总）。
* **不写** `优化/评估结果/`（optimizer 的产物目录）；测试侧结论只进 `测试/留痕/`。
* 每份报告首行固定 `RAGAS 未运行（依赖不可用，本机断网）`，并用确定性指标替代（命中率 / 准确率 / 引用正确率 / 首字 / 拒答率）。
* 只读纪律：工单1/工单2 目录不得写入（含 `__pycache__`，统一入口已设 `PYTHONDONTWRITEBYTECODE=1`）。

## 6. 验收项对照

| 验收项 | 判定方式 | 用例 |
| --- | --- | --- |
| 1 表格解析/归一化 | 锚点页结构断言 + 退化表不丢证据 | `离线/test_t7_offline_parse.py` |
| 2 混合检索与过滤 | 14 题 top-5 命中 ≥13 + 硬过滤 | `在线/test_t7_online_hybrid14.py` |
| 3 准确率 ≥90% | `evaluator_bridge.check_answer` ≥13/14 | `在线/test_t7_online_qa14.py` |
| 4 首字 ≤3 s | 逐题 `first_token_ms` + 预热事件 | `在线/test_t7_online_qa14.py`、`test_t7_online_warmup.py` |
| 5 引用可回溯 | 四点核验，正确率 = 1.0 | `在线/test_t7_online_qa14.py` |
| 6 无依据回「不清楚」 | 负例集全拒答 + 14 题误拒答 = 0 | `用户/test_t7_user_negative.py` |
| 7 多轮 + 中英文 | 3 轮问答 + 窗口/落库 + 分类取自本轮原文 | `在线/test_t7_online_multiturn.py`、`离线/test_t7_offline_classify.py` |
| 9 三级测试通过并留痕 | `run_tests.ps1` 退出码 0 + stdout 留痕 | `测试/run_tests.ps1` |
| 10 工程纪律 | 结构化日志 / 禁静默 except / 预热调用点 / 只读 | `离线/test_t7_offline_config_logging.py` |

## 7. 当前状态（重要）

**本轮只做「不依赖最终代码」的准备工作**：三级骨架、`conftest.py`、断言工具、14 题与负例集、
`run_tests.ps1`。**尚未**对定稿代码正式开跑，也**没有**给出任何「已通过」结论。

本地自检（非产品结论，仅证明测试基线自身可用；2026-10-04 实测）：

* `pytest --collect-only 测试`：**82 条用例**全部可收集；
* 离线级自检：**44 通过 / 2 失败 / 1 跳过**，两条失败与一条跳过均为**已知且可解释**：
  1. `test_warmup_call_sites_exist` —— `研发/app/main.py`、`研发/app/ui/streamlit_app.py` 尚未交付
     （T7 进行中），预热调用点自然还不存在；
  2. `test_no_silent_exception_swallow` —— `研发/app/core/reranker.py:196` 的 `except ImportError: pass`
     分支体只有 `pass`，与 `errors.py` 自己写的「允许的降级必须在日志中留下 `*.degrade` 事件」相冲突
     （**待 engineer 在 t12/T7 中处理**）；
  3. 跳过项 `test_dev_golden_kept_in_sync_when_present` —— T9 的 `研发/data/eval/golden_qa.jsonl` 尚未生成，
     生成后该用例自动生效，用于防止「评估一套题、测试另一套题」。
* 在线级 / 用户级用例需 `qa_engine` / `main.py` / 两个界面到位后才能执行（本轮**未**对定稿代码开跑）。

题面**单一来源**：id 1~4 的题面沿用 `测试/测试数据/eval_retrieval_14.jsonl`（T5 版，已用于 T5/T6/T9 评估），
`离线/test_t7_offline_fixture.py::test_question_texts_are_single_sourced` 会把任何措辞漂移钉成失败。
