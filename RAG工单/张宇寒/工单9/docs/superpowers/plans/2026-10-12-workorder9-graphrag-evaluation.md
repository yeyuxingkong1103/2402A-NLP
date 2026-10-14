# 工单9 GraphRAG 优化评测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在工单8基础上交付可运行的 GraphRAG 优化、参考答案评测和优化前后对比网页。

**Architecture:** `evaluation.py` 负责基准题、查询改写与指标；`graph_rag.py` 提供基线和优化检索；`app.py` 编排回答与评测；原生网页展示对比结果。

**Tech Stack:** Python、FastAPI、scikit-learn、NumPy、原生 HTML/CSS/JavaScript、可选 DeepSeek API

**Spec:** `docs/superpowers/specs/2026-10-12-workorder9-graphrag-evaluation-design.md`

## Global Constraints

- 不安装新依赖，不修改现有 Conda 环境。
- `.env` 中 API Key 保持空白。
- 端口使用 8020，工单8的8019不受影响。
- 评测可在无网络、无API Key时完整运行。

## Review Focus

- 匿名实体 `xx银行` 必须稳定映射到平安银行。
- 自定义问题没有参考答案时不伪造评测分数。
- 空检索结果的 precision/recall 返回0而非异常。
- DeepSeek 异常时仍返回本地证据和评测。
- 页面不可把检索文本当HTML执行。

### Task 1: 基准题与指标

**Files:**
- Create: `sample_questions.md`
- Create: `evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `load_benchmarks()`, `rewrite_query()`, `evaluate_hits()`

- [ ] 先写基准题解析、匿名实体改写、precision/recall 的失败测试。
- [ ] 运行测试确认因接口缺失而失败。
- [ ] 实现最小评测模块并运行完整测试。

### Task 2: 双通道检索与优化

**Files:**
- Modify: `graph_rag.py`
- Test: `tests/test_optimization.py`

**Interfaces:**
- Produces: `search_baseline()`, `search_optimized()`

- [ ] 先写基线/优化输出和去重的失败测试。
- [ ] 实现查询扩展和关系加权。
- [ ] 验证优化检索不破坏旧接口。

### Task 3: 评测API与网页

**Files:**
- Modify: `app.py`
- Modify: `index.html`
- Test: `tests/test_workorder9_web.py`

**Interfaces:**
- Produces: `/api/benchmarks`, `/api/evaluate`, `/api/evaluate-all`

- [ ] 先写接口结构、端口和页面区域的失败测试。
- [ ] 实现API、优化Prompt和可视化页面。
- [ ] 验证异常回退和自定义问题。

### Task 4: 实际评测与桌面交付

**Files:**
- Modify: `README.md`
- Create: `data/evaluation_report.json`

- [ ] 对四道题执行真实评测并保存报告。
- [ ] 运行完整测试、编译、健康检查和浏览器检查。
- [ ] 复制到桌面独立目录并从桌面版本复验。
