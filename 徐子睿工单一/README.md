# 招股说明书智能问答系统（RAG）

> **工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化**
> 输入语料：《招股说明书1.pdf》——武汉兴图新科电子股份有限公司 科创板招股意向书（548 页）

一个本地可跑的 **PDF 文档问答**系统：上传 PDF → MinerU 解析 → 分块 → BGE-M3 向量化 → 混合检索（向量 + BM25 → 加权 RRF）→ qwen2:7b 生成带页码引用的答案，无依据则拒答；并提供 **RAG vs 纯 LLM** 对照与 10 题检索评估。

---

## 1. 技术栈

| 环节 | 方案 |
|---|---|
| PDF 解析 | **MinerU 3.4**（pipeline 后端，GPU）+ PyMuPDF 兜底 |
| 分块 | 标题感知滑窗（600 字 / overlap 120），带页码+章节元数据 |
| 嵌入 | **bge-m3**（Ollama，1024 维，多语） |
| 向量检索 | numpy 余弦（内存常驻） |
| 关键词检索 | jieba 分词 + 自实现 BM25 |
| 融合 | 加权 RRF（λ=2.5, k=60） |
| 生成 | **qwen2:7b**（Ollama） |
| 服务 | FastAPI + uvicorn（REST + SSE 流式） |
| 前端 | 单文件 `static/index.html`（文字 + 语音输入 + 对比模式 + 反馈） |

## 2. 目录结构

```
徐子睿工单一/            （招股说明书 RAG 问答系统）
├─ app/
│  ├─ config.py      # 路径/模型/参数
│  ├─ parse.py       # MinerU 解析（+PyMuPDF 兜底）
│  ├─ clean.py       # 清洗/规范化
│  ├─ chunk.py       # 标题感知分块
│  ├─ llm.py         # Ollama 客户端（嵌入+生成，含流式）
│  ├─ bm25.py        # jieba + BM25
│  ├─ kb.py          # 向量库 + BM25 索引（构建/加载/检索）
│  ├─ prompts.py     # 提示词模板
│  ├─ engine.py      # 问答引擎（Query理解+混合检索+生成+拒答）
│  ├─ build_kb.py    # 一键构建知识库
│  └─ server.py      # FastAPI 服务
├─ static/index.html # 前端
├─ evaluation/
│  ├─ eval_questions.json     # 中文 10 题 + 金标准答案（含页码）
│  ├─ eval_questions_en.json  # 英文 10 题（同 id / 同金标准，测多语言）
│  ├─ run_eval.py             # 中文检索指标（Hit@k / MRR）
│  ├─ run_eval_en.py          # 英文检索指标（跨语检索）
│  ├─ build_bilingual_qa.py   # 中英双语问答 → report_bilingual.md
│  ├─ run_ragas.py            # RAGAS 标准指标（可选依赖，见 README_ragas.md）
│  ├─ compare_before_after.py  # 优化前后检索精确度对比（消融）
│  ├─ run_answer_accuracy.py  # 答案准确率（验收口径 ≥90%）
│  ├─ run_qa_report.py        # RAG vs 纯 LLM 对照
│  ├─ load_test.py            # 并发/高可用压测
│  ├─ calibrate.py / tune_fusion.py / smoke_test.py
│  └─ 评估报告.md / 优化前后对比.md / 答案准确率.md / report_*.md
├─ data/             # pdfs / mineru 输出 / 解析块 / 索引
├─ docs/             # 技术文档 / 用户手册
├─ 设计方案.md        # 01 工单：系统设计
├─ 优化方案.md        # 02 工单：优化方案 + 前后对比（产出物一）
└─ 演示视频分镜脚本.md # 02 工单：演示视频脚本（产出物二）
```

## 3. 快速开始

### 3.1 前置
- 本机 **Ollama** 已就绪，且已拉取模型：
  ```bash
  ollama list          # 需有 bge-m3、qwen2:7b
  ollama pull bge-m3
  ollama pull qwen2:7b
  ```
- PDF 解析用 **MinerU**（`pip install -U mineru`）。
  - ⚠️ 本机 torch 为 **CPU 版**（`torch.cuda.is_available() == False`），MinerU 全量 548 页约需 8 小时；
    如需 GPU 加速，请在带 CUDA 的机器/实例上跑解析，或装 CUDA 版 torch。
  - 解析产物缓存在 `data/mineru*/`，二次构建秒级复用。
  - **PyMuPDF 兜底通道**：当 `data/mineru` 下没有 MinerU 输出时，`parse.py` 自动改用 PyMuPDF（含内置表格识别）
    抽取，保证系统随时可用。

### 3.2 构建知识库（解析 → 向量化 → 索引）
```bash
# 方式一：先调用 MinerU 解析再构建（慢，但表格结构最好）
python app/build_kb.py --parse

# 方式二：用已有 MinerU 输出 / PyMuPDF 兜底直接构建（快）
python app/build_kb.py
```

### 3.3 启动服务
```bash
python -m uvicorn app.server:app --host 0.0.0.0 --port 8100
# 或
python app/server.py
```
浏览器打开 <http://localhost:8100>

### 3.4 评估与压测
```bash
python evaluation/compare_before_after.py # 优化前后检索精确度对比（消融）
python evaluation/run_answer_accuracy.py  # 答案准确率（验收口径 ≥90%）
python evaluation/run_eval.py      # 中文检索指标 Hit@k / MRR
python evaluation/run_eval_en.py   # 英文检索指标（跨语）
python evaluation/build_bilingual_qa.py  # 中英双语问答（顺便产出 RAGAS 输入）
python evaluation/run_qa_report.py # RAG vs 纯 LLM 对照报告
python evaluation/calibrate.py     # 拒答阈值标定
python evaluation/tune_fusion.py   # 融合策略对比
python evaluation/load_test.py     # 并发压测（需先起服务）
# RAGAS 标准指标（独立 venv，见 evaluation/README_ragas.md）
.venv_ragas\Scripts\python.exe evaluation\run_ragas.py --lang zh
```

### 3.5 实测结果

| 指标 | 中文 | 英文（跨语） |
|---|---|---|
| Hit@1 | 60% | 80% |
| Hit@3 | 100% | 100% |
| **Hit@5** | **100%** | **100%** |
| MRR | 0.750 | 0.883 |
| 引用命中金标准页（生成端） | 10/10 | 10/10 |

- **答案准确率 100%**（中英各 10 题，规则判分，验收要求 ≥90%）→ `evaluation/答案准确率.md`
- **优化前后**：中文 Hit@3 80%→100%、MRR 0.733→0.750；英文 Hit@5 80%→100%、MRR 0.720→0.883 → `evaluation/优化前后对比.md`
- 拒答阈值 0.58（库内最低 0.644 / 库外最高 0.509，可分）
- RAG 平均响应 ≈1.8 s（流式首字更快）；纯 LLM 约 2.9 s 且波动更大
- 并发压测：检索无错误、生成层受单卡 7B 限制（QPS ≈1.3，见 `load_test_result.json`）
- RAGAS 标准指标见 `evaluation/ragas_result_zh.json` / `_en.json`

详见 `evaluation/评估报告.md` 与 `evaluation/report_rag_vs_llm.md`。

## 4. 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 模型与索引状态 |
| GET | `/api/kb/overview` | 知识库概览（块数/维度/页码范围） |
| GET | `/api/kb/chunks` | 分块浏览 |
| POST | `/api/search` | 仅检索（返回 top-k 片段） |
| POST | `/api/ask` | RAG 问答（同步） |
| POST | `/api/ask/stream` | RAG 问答（SSE 流式） |
| POST | `/api/ask/llm` | 纯 LLM 对照 |
| POST | `/api/feedback` | 👍/👎 反馈 |
| POST | `/api/upload` | 上传新 PDF |

## 5. 关键参数（`app/config.py`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `CHUNK_SIZE / CHUNK_OVERLAP` | 600 / 120 | 分块长度与重叠 |
| `RECALL_K` | 20 | 每路召回条数 |
| `RRF_LAMBDA` | 2.5 | 向量路权重 |
| `TOP_K` | 5 | 送入生成的片段数 |
| `REFUSE_SCORE` | 0.58 | 置信度（最高余弦）低于此则拒答 |
| `EMBED_MODEL / GEN_MODEL` | bge-m3 / qwen2:7b | 可换 |

## 6. 演示要点（对应工单交付物）

1. **功能演示**：上传 PDF → 构建 → 提问 → 带页码引用作答 → 语音输入 → 中英双语 → 反馈。
2. **10 题检索**：`evaluation/run_eval.py` 输出 Hit@k / MRR 与命中页码。
3. **RAG vs 纯 LLM**：前端勾选“对比纯LLM”，同屏两路作答。
4. **评估体系**：检索指标 + 生成忠实度（见 `docs/技术文档.md`）。

## 7. 已知注意事项

- MinerU 首次解析慢（GPU 约 5-7 s/页）；解析结果缓存在 `data/mineru/`，二次构建无需重跑。
- 本机 8090/8100 端口若被占用，改 `config.PORT`。
- `jieba` 会打印 `pkg_resources` 弃用告警，无害；如需静默：`set PYTHONWARNINGS=ignore`。
