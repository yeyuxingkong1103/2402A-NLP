# 工单01 · 基于 PDF 文档的问答系统

> 工单编号：**人工智能NLP-RAG-基于PDF文档的问答系统**
> 项目：八维 NLP-RAG 项目（共 18 份工单，本仓库为工单 01）

针对《武汉兴图新科电子股份有限公司招股意向书》（548 页）的 RAG 问答系统：
PDF 文字+表格解析 → 句子边界分块 → 去重 → bge-m3 向量化 → Milvus 检索 →
qwen3:8b 流式生成 → 三轨评估。

---

## 快速开始

```bash
# 0. 环境自检（第一道拦截，必须先跑）
python scripts/preflight.py

# 1. 启动向量库（Milvus Standalone + Attu）
docker compose -f docker-compose.milvus.yml up -d
#    Attu 可视化： http://localhost:8000

# 2. 入库（548 页约 1–3 分钟）
python scripts/ingest.py

# 3. 起服务
uvicorn app.main:app --port 8080
#    问答界面： http://localhost:8080

# 4. 10 题对比评估
python scripts/eval.py

# 5. 测试
pytest tests/ -v
```

## 目录结构

```
rag-project/
├── app/
│   ├── main.py              FastAPI 入口（生命周期、静态挂载、路由注册）
│   ├── config.py            配置（宿主 IP 动态发现）
│   ├── schemas.py           Pydantic 模型
│   ├── api/                 chat · ingest · kb · evaluate · asr · health
│   ├── core/                ollama_client · pdf_parser · chunker · dedup ·
│   │                        embedder · vectorstore · retriever · generator ·
│   │                        evaluator · pipeline
│   └── static/              index.html · app.js · style.css
├── docs/                    技术文档 · 用户手册 · 验收对照表 · 流程图
├── eval/questions.json      10 题题面 + 人工核实真值
├── scripts/                 preflight · ingest · eval
├── tests/                   pytest 冒烟
└── docker-compose.milvus.yml
```

数据目录（WSL 原生盘，避开 `/mnt/c` 的 9p 慢速）：
`~/rag-data/{raw,parsed,eval,feedback,venv}`

## 关键设计取舍

详见 `docs/01-技术文档.md`，此处只列结论：

| 决策 | 选择 | 理由（均为实测） |
|---|---|---|
| 向量库 | **Milvus Standalone** | Lite 不监听端口 → Attu 连不上；且单进程文件锁过不了高并发验收 |
| 分块 | **句子边界优先**（300–600 字） | 固定字数会切断 p128 那句同时装着两道题答案的 137 字金句 |
| 表格 | **`find_tables()` + 表头下推** | 纯文本流抽出的是粘连的 `发行前每股净资产3.55元/股…` |
| 页眉清洗 | **锚定整行模板** | 按关键词删会误伤正文（公司全称在正文反复出现）→ 静默答错 |
| 去重 | **入库只压精确重复；冗余控制放检索时** | 实测"重复段落"chunk 间海明距离 14–34，放宽阈值会误删真实内容 |
| 语音 | **后端 faster-whisper** | Chrome Web Speech 走 Google 服务器，国区不可用且报错误导 |
| 评估 | **自实现三轨** | ragas 走 Ollama 有三个必炸点；规则化数值命中比 LLM 打分更可信 |
| 响应口径 | **TTFT**（并主动披露完整耗时） | 工单写"返回答案的时间"，流式下首 token 即开始返回 |

## 环境要求

- WSL2（`/etc/wsl.conf` 需 `systemd=true`）、Python 3.11、Docker
- Windows 侧 Ollama，模型：`qwen3:8b`、`bge-m3`
- `.wslconfig` 建议 `memory=8GB`、`swap=8GB`

## 代码规范

每个源文件首行均含工单编号注释（工单硬约束）：

```bash
grep -rn "人工智能NLP-RAG-基于PDF文档的问答系统" --include="*.py" --include="*.js" --include="*.css" --include="*.html" app/ scripts/ tests/
```
