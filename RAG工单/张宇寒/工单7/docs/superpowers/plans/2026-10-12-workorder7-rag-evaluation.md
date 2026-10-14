# 工单7 RAG测试与检索评估 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个读取CCF年报结构化文本、运行10道固定题并展示/导出RAG评测结果的本地网页项目。

**Architecture:** 以工单6的FastAPI、BGE、FAISS和TF-IDF代码为稳定基线，新增CCF JSONL数据适配层和独立评测模块。检索评测与DeepSeek生成解耦，保证无API Key仍可得到真实检索指标。

**Tech Stack:** Python、FastAPI、sentence-transformers、FAISS、scikit-learn、原生HTML/CSS/JavaScript。

**Spec:** `docs/superpowers/specs/2026-10-12-workorder7-rag-evaluation-design.md`

## Global Constraints

- 使用现有 `C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat`，不安装依赖。
- 数据默认来自给定附件目录，项目内不复制原始PDF/TXT。
- 固定10题，评测结果可重复。
- DeepSeek失败时检索评测仍可运行。
- 端口8018；密钥、索引、缓存不提交。

## Review Focus

- 中文乱码文件名必须通过股票代码稳定映射到正确报告。
- 空行或损坏JSONL行不能导致整个建库失败。
- 无DeepSeek Key时必须返回证据摘要且指标仍完整。
- 目标证据未召回时MRR必须为0，不能误算为1。
- 单题失败不能中断其他9题，导出仍包含错误记录。

---

### Task 1: 建立基线并解析CCF年报

**Files:**
- Create: `corpus.py`
- Create: `tests/test_corpus.py`
- Modify: `prepare.py`

**Interfaces:**
- Produces: `iter_jsonl_rows(path) -> tuple[list[dict], int]`、`load_corpus(root) -> list[Chunk]`、`REPORTS`。
- Consumes: 工单6的 `Chunk` 与索引保存接口。

- [ ] 写失败测试，覆盖股票代码映射、损坏行跳过、页码保留和缺失目录错误。
- [ ] 运行 `python -m unittest tests.test_corpus -v`，确认因接口不存在而失败。
- [ ] 实现最小JSONL解析和分块，并把 `prepare.py` 改为从附件TXT建库。
- [ ] 再运行测试，预期全部通过。
- [ ] 提交 `feat: parse ccf annual report corpus`。

### Task 2: 固定10题与评测指标

**Files:**
- Create: `evaluation.py`
- Create: `evaluation_questions.json`
- Create: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `RAGSystem.search(question, config)` 返回命中片段。
- Produces: `load_questions()`、`score_retrieval()`、`score_facts()`、`evaluate_all()`。

- [ ] 写失败测试，覆盖恰好10题、Hit@K、Recall@K、MRR、事实覆盖率和单题异常隔离。
- [ ] 运行 `python -m unittest tests.test_evaluation -v`，确认因接口不存在而失败。
- [ ] 实现固定题加载、逐题评测、汇总指标和无LLM证据摘要降级。
- [ ] 再运行测试，预期全部通过。
- [ ] 提交 `feat: add repeatable rag evaluation`。

### Task 3: API、网页和JSON导出

**Files:**
- Modify: `app.py`
- Modify: `index.html`
- Create: `tests/test_web.py`

**Interfaces:**
- Consumes: `evaluation.evaluate_all()` 和最近一次报告缓存。
- Produces: `/api/evaluation/questions`、`/api/evaluation/run`、`/api/evaluation/export`及网页操作流程。

- [ ] 写失败测试，覆盖10题接口、评测接口、JSON下载、未建库提示和页面关键控件。
- [ ] 运行 `python -m unittest tests.test_web -v`，确认新路由/元素缺失导致失败。
- [ ] 实现API和一页式评测面板，保留单题问答与PDF查看。
- [ ] 再运行测试，预期全部通过。
- [ ] 提交 `feat: add evaluation dashboard and export`。

### Task 4: 运行说明与整体核验

**Files:**
- Create: `README.md`
- Create: `.env`
- Create: `.gitignore`
- Create: `tests/test_project.py`

**Interfaces:**
- Consumes: 前三项任务提供的脚本、API和页面。
- Produces: 可直接运行的桌面交付项目。

- [ ] 写失败测试，覆盖配置模板、端口8018、Git忽略规则和10题文件结构。
- [ ] 运行 `python -m unittest tests.test_project -v`，确认缺失交付文件导致失败。
- [ ] 编写启动说明与配置文件，清理不应交付的数据和密钥。
- [ ] 运行 `python -m unittest discover -s tests -v` 和Python语法编译，预期全部通过。
- [ ] 提交 `docs: add workorder7 run guide`。
