# 工单8 Graph RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建读取CCF金融年报、生成本地知识图谱并展示答案、证据和图谱子图的Graph RAG网页项目。

**Architecture:** `corpus.py`从ZIP读取并切分年报，`graph_rag.py`建立TF-IDF与JSON知识图谱并完成图增强检索，`questions.py`解析固定Markdown题库，`app.py`提供API，`index.html`完成交互和SVG图谱展示。

**Tech Stack:** Python、FastAPI、scikit-learn、NetworkX数据模型兼容JSON、DeepSeek OpenAI兼容API、原生HTML/CSS/JavaScript。

**Spec:** `docs/superpowers/specs/2026-10-12-workorder8-graph-rag-design.md`

## Global Constraints

- 使用 `C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe`，不安装依赖。
- 默认读取 `C:\Users\lenovo\Desktop\RAG 工单 (1)\RAG 工单\附件\ccf_competition.zip`。
- `.env` 只保留空白 `DEEPSEEK_API_KEY=`。
- 服务端口为8019；数据、索引、附件、缓存和真实密钥不提交Git。

## Review Focus

- ZIP文件名乱码但股票代码仍可识别时应正常匹配报告。
- TXT中含空行或损坏JSON行时应跳过而非中断建库。
- 问题未提公司或同时提多家公司时应保留多报告证据。
- 图谱过大时API只能返回Top K相关子图，浏览器不能接收全量文本节点。
- 未配置DeepSeek或未建索引时页面和API应给出可操作的中文提示。

---

### Task 1: 题库与语料解析

**Files:**
- Create: `eval_questions.md`
- Create: `questions.py`
- Create: `corpus.py`
- Test: `tests/test_questions.py`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Produces: `load_questions(path) -> list[dict]`、`load_zip_corpus(path) -> list[Chunk]`。

- [ ] 先写题库解析、股票代码识别、损坏JSON容错和切块测试并确认失败。
- [ ] 实现Markdown题库解析和ZIP语料解析。
- [ ] 运行Task 1测试并确认通过。

### Task 2: 知识图谱与Graph RAG检索

**Files:**
- Create: `graph_rag.py`
- Create: `prepare.py`
- Test: `tests/test_graph_rag.py`

**Interfaces:**
- Consumes: `Chunk`。
- Produces: `build_graph(chunks) -> dict`、`GraphRAG.search(question, top_k) -> dict`。

- [ ] 先写节点/关系生成、公司年份指标加权、空问题和子图边界测试并确认失败。
- [ ] 实现图谱、TF-IDF索引、保存加载和图增强检索。
- [ ] 运行Task 2测试并确认通过。

### Task 3: 问答API与网页

**Files:**
- Create: `app.py`
- Create: `index.html`
- Create: `.env`
- Create: `.gitignore`
- Test: `tests/test_web.py`

**Interfaces:**
- Consumes: `GraphRAG.search`、`load_questions`。
- Produces: `/api/health`、`/api/questions`、`/api/ask`和8019启动入口。

- [ ] 先写API、错误提示、空白密钥、端口和页面图谱控件测试并确认失败。
- [ ] 实现FastAPI、DeepSeek可选生成和原生SVG图谱页面。
- [ ] 运行Task 3测试并确认通过。

### Task 4: 实际建库、说明与交付验证

**Files:**
- Create: `README.md`
- Create: `tests/test_project.py`

**Interfaces:**
- Consumes: 所有前述模块。
- Produces: 可直接运行的桌面项目及建库结果。

- [ ] 先写README、忽略规则和项目结构测试并确认失败。
- [ ] 完成README并运行 `prepare.py` 处理真实ZIP。
- [ ] 运行完整测试、10题检索烟雾测试和网页健康检查。
- [ ] 复制到桌面并保留空白 `.env`。

