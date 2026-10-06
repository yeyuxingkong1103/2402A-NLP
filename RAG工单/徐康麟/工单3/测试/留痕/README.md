# 测试/留痕/ —— 测试执行留痕（tester 独占写目录）

> 工单：**人工智能NLP-RAG-PDF文档的表格解析及检索优化**

本目录是验收 9「三级测试全部通过并留痕」的证据落点。**只允许 tester 写入**；
其它角色（尤其 optimizer）请不要把评估产物放这里（`优化/评估结果/` 才是 T9 的目录）。

## 会出现的文件

| 文件 | 产生者 | 内容 |
| --- | --- | --- |
| `offline_<时间戳>.txt` / `online_<时间戳>.txt` / `user_<时间戳>.txt` | `测试/run_tests.ps1` | 各段 pytest 的**原始 stdout**（含 RAGAS 标注行） |
| `run_tests_<时间戳>.json` | `测试/run_tests.ps1` | 三段退出码汇总（全 0 才算三级通过） |
| `online_retrieval_top5.json` | `在线/test_t7_online_hybrid14.py` | 14 题 top-5 命中、命中块、判据字段 |
| `online_qa14.json` | `在线/test_t7_online_qa14.py` | 14 题答案、判分理由与来源、逐题首字、逐条引用核验 |
| `online_multiturn.json` / `online_warmup.json` / `online_first_question_after_warmup.json` | `在线/` | 多轮与中英文、预热事件取证、预热后首题 |
| `user_unknown_cases.json` / `user_leakage_cases.json` / `user_subject_gate.json` / `user_bank_loan.json` | `用户/` | 拒答、互污染、闸门 fail-open、跨语料对照 |
| `simulate_user.txt` / `simulate_user.json` | `用户/simulate_user.py` | 用户旅程会话记录与逐场景判定 |
| `bm25_rank_diagnostic.json` | `离线/test_t7_offline_index_bm25.py` | 逐题 BM25 单路名次（诊断，非门槛） |
| `baseline_table_markdown.json` | `离线/test_t7_offline_parse.py` | 表块 markdown 非空率（基线值，非门槛） |
| `config_thresholds_record.json` | `离线/test_t7_offline_config_logging.py` | 可答性阈值实测值（与设计 §8 初值的差异记录） |
| `app_log_parse_audit.json` | `离线/test_t7_offline_config_logging.py` | app.log 可解析比例与异常行（并发写日志的截断现象取证） |
| `degenerate_on_extended_evidence_pages.json` | `离线/test_t7_offline_parse.py` | 落在**延伸**证据页上的退化表（只留痕、不判缺陷） |

## 纪律

* 每份 JSON 报告都带 `work_order` 与 `ragas` 字段，首行/首字段固定
  `RAGAS 未运行（依赖不可用，本机断网）`；**严禁**出现 `ragas_score` / `faithfulness` 等伪造指标字段
  （`common/reports.assert_no_ragas_numbers` 会把它们直接判失败）。
* 时间戳命名保证多次重跑不覆盖历史留痕（验收 8「重跑一致」可对比）。
* 留痕里出现的失败必须带**可执行的整改线索**（文件:行号 / 题号 / 期望 vs 实测），不允许只写「失败」。
