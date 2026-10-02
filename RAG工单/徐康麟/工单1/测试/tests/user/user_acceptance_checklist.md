# 用户验收清单（工单1 · 基于 PDF 文档的 RAG 问答系统）

> 用法：逐条执行命令 → 对照「预期结果」→ 把 `- [ ]` 改成 `- [x]`。
> 全部命令都在**项目根目录** `E:\gao6gongdan\工单1` 下执行。
> 最近一次完整验收结果见文末「附录 A：验收记录（自动化实测）」。

## 0. 环境准备（每次验收前执行一次）

```powershell
$env:PYTHONPATH="E:\gao6gongdan\工单1"
cd E:\gao6gongdan\工单1
$PY = "E:\gao6gongdan\工单1\.gao6gongdan-src\python.exe"

# 如索引未生成（data/index/rag.sqlite3 的 chunks 表为空），先建索引：
& $PY 研发/scripts/build_index.py
```

说明：`测试/pytest.ini` 只设 `testpaths = tests` 与 `-p no:cacheprovider`；测试临时目录由
`测试/tests/conftest.py` 通过 `PYTEST_DEBUG_TEMPROOT` 指向 `tests/.pytest_run_<进程号>`，
避免受限环境无法访问系统临时目录（会令 `tmp_path` 报 PermissionError），
也避免两个并发的 pytest 会话互相清空临时目录；会话结束自动删除。
另外，`conftest.py` 会把标准库临时目录（`tempfile.tempdir`）一并重定向到该目录内。

## 1. 三条统一命令（工单 9.4）

```powershell
# ① 离线测试：不依赖 LLM 服务；缺索引的用例会明确 skip
& $PY -m pytest 测试/tests/offline -v

# ② 在线测试：引擎直连（本机无 LLM 服务，走抽取式降级）+ 网页界面无头渲染
& $PY -m pytest 测试/tests/online -v

# ③ 用户测试：模拟真实用户操作，日志写入 logs/user_simulation.log
& $PY 测试/tests/user/simulate_user.py
```

- [ ] 三条命令均执行完毕，且输出中没有 `FAILED` / `ERROR`
- [ ] `测试/tests/offline` 结果为 `135 passed, 4 xfailed`
- [ ] `测试/tests/online` 结果为 `101 passed, 1 skipped, 2 xfailed`
- [ ] `测试/tests/user/simulate_user.py` 退出码为 0，且 `logs/user_simulation.log` 已生成

> `xfailed` 是**已知缺陷用例**（真实执行、真实断言，缺陷修复后自动变为 `XPASS`），
> 明细见第 5 节；`skipped` 是"默认不启动网页服务"的可选分支，原因见第 4 节。

## 2. 逐条对应工单第 13 节验收标准

### 13.1 PDF 解析能准确提取文字和表格
- [ ] `& $PY -m pytest 测试/tests/offline/test_pdf_parser.py -v` 全部通过（16 项）
- [ ] 校验点：`page_count == 548`；页码 1..548 连续；表格带 `table_id / page / markdown`；页眉页脚被清洗

### 13.2 10 个工单问题答案准确，且基于 PDF 内容
- [ ] `& $PY 测试/tests/user/simulate_user.py`：10 题全部拿到非空答案，且与标准答案判分一致（实测 12/12 一致）
- [ ] `& $PY -m pytest 测试/tests/offline/test_qa_engine.py -v` 通过（含逐题判分与准确率基线）
- [ ] 校验点：`测试/tests/offline/test_qa_engine.py::test_answer_accuracy_baseline`（要求 ≥ 9/10，实测 10/10）

### 13.3 答案带引用，引用页码真实
- [ ] `& $PY -m pytest 测试/tests/online/test_citation.py -v` 全部通过（10 题 × 6 项校验）
- [ ] 校验点：引用页码 ⊆ 1..548（PyMuPDF 实测页数）；`chunk_id` 能在 SQLite 中查到；引用摘要来自原文
- [ ] 校验点：绝不生成"带 chunk_id 的幻觉引用"（越界页码不会出现在引用列表中）

### 13.4 不知道时回复"不清楚"
- [ ] `& $PY -m pytest 测试/tests/online/test_unknown.py -v` 全部通过（8 项）
- [ ] 校验点：未建索引 / 无检索片段 / 检索结果为空时，答案恰为「不清楚」且不带引用
- [ ] 校验点：与语料无关的问题（"今天天气怎么样""红烧肉怎么做""推荐一本小说"）均回复「不清楚」
      （依据检索器原始余弦阈值 `min_confidence_cosine=0.55`：相关问题 ≥0.73、无关问题 ≤0.39）
- [ ] `& $PY 测试/tests/user/simulate_user.py` 日志中「无关问题是否兜底：是」

### 13.5 首字返回时间 < 3 秒
- [ ] `& $PY -m pytest 测试/tests/online/test_api.py -v` 通过
- [ ] 校验点：10 题最大首字耗时 < 3000ms（实测平均 ≈ 0.9ms、最大 ≈ 1.5ms）
- [ ] 日志证据：`logs/user_simulation.log` 中「首字耗时」一行

### 13.6 支持多轮对话
- [ ] `& $PY -m pytest 测试/tests/online/test_multiturn.py -v`：11 项通过、2 项 xfail（会话计数/标题缺陷）
- [ ] 校验点：历史按 user/assistant 交替落库；追问"那注册资本呢？"被还原成完整问句
      「武汉兴图新科电子股份有限公司注册资本是谁？」，意图切换到「注册资本」，答案给出 5,520 万元
- [ ] 校验点：无关问题（"今天天气怎么样？"）不会被误判为追问

### 13.7 支持中等并发
- [ ] `& $PY -m pytest 测试/tests/online/test_concurrency.py -v` 全部通过（5 项）
- [ ] 校验点：8 并发调用无异常、同题答案一致、无 `database is locked`、消息不丢失

### 13.8 日志完整，函数输入输出都有记录
- [ ] `logs/app.log`、`logs/error.log`、`logs/rag_trace.jsonl` 均存在且非空
- [ ] 快速检查：`Get-Content logs/rag_trace.jsonl -Tail 3`（每行 JSON，含 `enter`/`exit`、`args`、`kwargs`、`elapsed_ms`）
- [ ] 校验点：异常必须落 `logs/error.log`（带堆栈），不允许静默失败

### 13.9 离线、在线、用户测试全部通过
- [ ] 第 1 节的 3 条命令均无 `failed`
- [ ] `skipped` 与 `xfailed` 的数量、原因已在本清单登记

### 13.10 RAG vs 纯 LLM 对比报告和 RAGAS 评估结果完整
- [ ] `python 研发/scripts/evaluate.py` 生成的 `优化/评估结果/eval_results/rag_vs_llm.csv` 存在且含 10 行题目
- [ ] `优化/评估结果/eval_results/ragas_report.md` 存在
- [ ] 校验点：RAGAS 四项指标在无裁判 LLM 时明确标注「未运行」，不得伪造数值
- [ ] （算力云）起 vLLM 后执行 `python 研发/scripts/evaluate.py --ragas` 补齐四项指标

## 3. 网页界面人工验收

```powershell
& $PY -m streamlit run app/ui/streamlit_app.py --server.port 8501
# 浏览器打开 http://127.0.0.1:8501
```

- [ ] 页面加载默认语料《招股说明书1.pdf》，侧边栏显示页数 548、分块数、向量库后端（numpy）
- [ ] 输入问题后能看到答案、引用来源（页码 + 可展开原文）、首字响应时间
- [ ] 多轮对话历史保留；「清空对话」按钮生效
- [ ] 点赞 / 点踩按钮可提交反馈
- [ ] 服务启动后执行 `& $PY -m pytest 测试/tests/online/test_api.py -v -k streamlit`，HTTP 分支由 skip 变为 passed

> 无需手工启动服务也可验证界面：`测试/tests/online/test_api.py::test_streamlit_app_renders_without_exception`
> 用 Streamlit 官方 `AppTest` 无头执行 `app/ui/streamlit_app.py`，校验标题、提问框、按钮与引用说明均正常渲染。

## 4. 跳过（skip）说明

| 跳过条件 | 提示信息 | 处理方式 |
| --- | --- | --- |
| 索引未生成（`data/index/rag.sqlite3` 无分块） | `索引未就绪：请先运行 python 研发/scripts/build_index.py` | 先执行 `python 研发/scripts/build_index.py` |
| 未启动网页服务（默认不启动，避免测试过慢） | `未启动网页服务（127.0.0.1:8501）：为避免自动化测试过慢，HTTP 分支默认不执行…` | 需要时 `streamlit run app/ui/streamlit_app.py --server.port 8501` |
| 语料 PDF 或标准答案文件缺失 | 对应文件路径提示 | 补齐 `data/raw/招股说明书1.pdf`、`data/eval/golden_qa.jsonl` |

## 5. 已知问题（对应 xfail 用例，缺陷修复后自动转 XPASS）

| 编号 | 现象 | 影响 | 位置 | 对应用例 |
| --- | --- | --- | --- | --- |
| 1 | 表格首列是「序号」时，抽取式回答把序号当金额（实测返回「3 万元」而非「15,000.00 万元」） | 募资用途类问题可能答错数字 | `app/core/generator.py:389-400` | `测试/tests/offline/test_generator.py::test_extractive_fund_usage_table_with_index_column` |
| 2 | 数值写法不同导致判分误判：「5,520.00 万元」与标准答案「5,520 万元」被判为不一致 | 评估准确率被低估 | `app/core/evaluator.py:177-180` | `测试/tests/offline/test_evaluator.py::test_check_answer_equivalent_amount_with_decimals` |
| 3 | 判分口径偏宽：只答出 2/4 个比重、字符相似度 0.84 即判为正确 | 评估准确率可能虚高 | `app/core/evaluator.py:194-199` | `测试/tests/offline/test_evaluator.py::test_check_answer_partial_number_match_should_be_wrong` |
| 4 | SQLite 外键未生效（普通连接 `PRAGMA foreign_keys=0`），孤儿分块可写入、级联删除不生效 | 数据一致性风险 | `app/storage/sqlite_manager.py:185-189` | `测试/tests/offline/test_sqlite.py::test_foreign_key_constraint_rejects_orphan_chunk` |
| 5 | 会话 `message_count` 被重置为 0 | 界面会话列表计数显示为 0 | `app/core/conversation.py` `auto_title` + `app/storage/sqlite_manager.py` `create_conversation`（INSERT OR REPLACE） | `测试/tests/online/test_multiturn.py::test_message_count_matches_stored_messages` |
| 6 | 会话标题被每一轮问题覆盖 | 会话列表中标题不是首轮问题 | 同问题 5（守卫依赖 message_count） | `测试/tests/online/test_multiturn.py::test_title_is_not_overwritten_by_later_rounds` |

> 以上问题只做记录，本次测试**未修改** `app/` 下任何代码（工单要求：发现缺陷先汇报）。
> 另有一处**注释与实现不一致**（非缺陷用例）：`app/core/citation.py` 的 `build()` 注释称幻觉引用会
> 「保留但标注分数 0」，实际实现是丢弃该页码、只写告警日志（引用不再进入答案，评估也就统计不到它）。
>
> 已修复（本轮测试期间确认）：无关问题不回复「不清楚」的相关性阈值缺陷 —— 现已改为基于原始余弦
> `min_confidence_cosine`，`测试/tests/online/test_unknown.py` 3 个用例由 xfail 转为 passed。

## 6. 验收结果记录

| 项目 | 结果 | 记录人 | 日期 |
| --- | --- | --- | --- |
| 离线测试 | | | |
| 在线测试 | | | |
| 用户模拟 | | | |
| 网页界面 | | | |
| 评估报告 | | | |

## 附录 A：验收记录（自动化实测）

执行环境：`.gao6gongdan-src\python.exe`（Python 3.10.21）、索引 `doc_3302796a`（548 页 / 3024 分块 /
语义嵌入 `bge-small-zh-v1.5` + numpy 向量库）、无 LLM 服务（抽取式生成）。

| 命令 | 实测输出 |
| --- | --- |
| `& $PY -m pytest 测试/tests/offline -v` | `135 passed, 4 xfailed in 25.77s` |
| `& $PY -m pytest 测试/tests/online -v` | `101 passed, 1 skipped, 2 xfailed in 22.22s` |
| `& $PY 测试/tests/user/simulate_user.py` | 退出码 0；10 题全部非空答案，12/12 判分一致；首字耗时平均 0.8ms、最大 1.2ms；无关问题回复「不清楚」 |

完整输出留档：`logs/test_offline_run.txt`、`logs/test_online_run.txt`、`logs/user_simulation.log`。
