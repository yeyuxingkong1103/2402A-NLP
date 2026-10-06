# 基于 PDF 文档的问答系统优化（工单 02）

**工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化**

面向《招股说明书1.pdf》（548 页，武汉兴图新科电子股份有限公司）的检索增强问答系统优化版：
在工单 01 基线之上，从 **PDF 解析、分块、检索** 三个层面做优化，
目标 **回答准确率 ≥90%、响应时间 ≤3 秒**，并提供优化前后对比与 RAGAS 评估。

> **硬约束**：全流程只使用本机已下载的模型，任何环节都不触发模型下载。

---

## 快速开始

```bash
# 1) 确认 Ollama 在跑且本机已有 qwen2.5:3b
ollama list

# 2) 建库（首次或需要重建时）
#    MinerU 3.4.5 经独立 venv 解析 548 页（约 1~2 分钟）→ 分块 → 向量化 → Qdrant
python build_index.py

# 3) 启动中英双语界面
streamlit run app.py
```

> MinerU 环境（已配置好，换机时按此准备）：
> - 独立 venv：`D:\model\mineru-venv`（`--system-site-packages` 复用主环境 CUDA torch）
>   ```
>   python -m venv --system-site-packages D:\model\mineru-venv
>   D:\model\mineru-venv\Scripts\pip install "mineru[core]==3.4.5" "transformers==4.57.6"
>   D:\model\mineru-venv\Scripts\pip install --no-deps --ignore-installed mineru==3.4.5
>   python scripts/patch_mineru_venv.py     # 跳过缺失的印章模型（本机快照无 seal 权重）
>   ```
> - 详细原因（为什么是 3.4.5、为什么需要独立 venv）见 `docs/02-优化方案.md` 2.1 节。

浏览器打开 `http://localhost:8501` 即可提问（中文/英文），回答带**页码引用**与**全链路耗时**。

---

## 系统架构

```
【离线建库】
 招股说明书1.pdf
   └─ 解析  MinerU 3.4.5（pipeline：PP-DocLayoutV2 版面 + OCR + 表格 + 公式）
   │        独立 venv D:\model\mineru-venv（复用主环境 CUDA torch，隔离 transformers 4.57）
   │        失败时自动回退 PyMuPDF 增强引擎（src/pymupdf_parser.py）
   │        输出：Markdown + content_list.json（含 text_level 标题层级 / 表格 / 页码）
       └─ OCR 兜底  PaddleOCR-VL（低文本页，独立 venv 子进程）
       └─ 分块  标题层级 + 父子块（子块 300–450 字检索 / 父块 ≤1800 字上下文）+ 表格独立
           └─ 向量化  bge-m3（1024 维，CUDA，local_files_only）
               └─ 入库  Qdrant（本地模式；collection zhaogu_v2_opt）

【在线问答】
 提问（中文/英文）
   └─ 查询改写  意图识别 / 消歧 / 分解 / 领域扩展（规则，<1ms）
       └─ 混合检索  向量 top-30 + BM25 top-30 → RRF 融合（k=60）
           └─ 精排  bge-reranker-v2-m3（top-20 → 4）
               └─ 上下文  子块→父块去重 + 同页去重 + 裁剪（≤3200 字）
                   └─ 生成  qwen2.5:3b（Ollama）+ 数字校验重试 + 抽取式降级
                       → 答案 + [n] 页码引用 + 耗时明细
```

---

## 目录结构

```
工单2/
├── src/
│   ├── bootstrap.py        # 离线环境 + Windows 栈溢出修复（必须最先导入）
│   ├── config.py           # 全部配置（路径/模型/参数）
│   ├── pdf_parser.py       # 解析入口（MinerU 可选 → PyMuPDF 增强）
│   ├── pymupdf_parser.py   # PyMuPDF 五遍扫描引擎
│   ├── ocr_fallback.py     # PaddleOCR-VL 兜底（独立 venv）
│   ├── chunker.py          # 标题层级 + 父子块分块
│   ├── embedder.py         # bge-m3 向量化
│   ├── vector_store.py     # Qdrant（local/server 双模式）
│   ├── retriever.py        # 向量+BM25+RRF 混合检索
│   ├── reranker.py         # bge-reranker-v2-m3 精排
│   ├── query_rewrite.py    # 查询改写（意图/消歧/分解/扩展）
│   ├── context.py          # 父块上下文组装与压缩
│   ├── llm.py              # Ollama 封装
│   ├── rag.py              # 全链路编排（计时/缓存/校验/降级）
│   └── baseline.py         # 「优化前」桥接（只读调用工单1）
├── build_index.py          # 建库
├── app.py                  # Streamlit 中英双语界面
├── evaluate_compare.py     # 三方对比（准确率/时间/HitRate/MRR/NDCG）
├── evaluate_ragas.py       # RAGAS 评估（本地裁判，离线）
├── scripts/
│   ├── mineru_compat.py    # MinerU VLM 后端兼容层
│   └── baseline_ask.py     # 工单1 单题问答（子进程）
├── tests/                  # pytest 单元测试
├── data/                   # 解析/分块/向量库/评估产物
└── docs/
    ├── 01-本机模型扫描结果.md
    ├── 02-优化方案.md
    ├── 03-优化前后对比分析.md
    ├── 04-RAGAS评估报告.md
    ├── 技术文档.md
    └── 用户手册.md
```

---

## 主要命令

| 命令 | 说明 |
| --- | --- |
| `python build_index.py` | 建库（解析→分块→向量化→入库） |
| `python build_index.py --rebuild` | 强制重新解析 |
| `streamlit run app.py` | 启动双语问答界面 |
| `python evaluate_compare.py` | 三方对比评估（优化后/优化前/纯LLM） |
| `python evaluate_compare.py --skip-baseline` | 跳过工单1（更快） |
| `python evaluate_ragas.py` | RAGAS 评估（本地裁判） |
| `python -m pytest tests/ -v -m "not slow"` | 单元测试 |

---

## 关键设计

- **父子块**：小块（300–450 字）做向量检索保证命中精度；命中后展开所属父块（≤1800 字）送 LLM，兼顾上下文完整；
- **混合检索**：精确事实（数字/专名）靠 BM25，语义问法靠向量，RRF 融合无需调参；
- **数字防幻觉**：生成后校验答案中的数字是否都来自上下文、列表是否重复，可疑则定向重试；
- **性能**：预热 + 规则版查询理解 + 单次编码 + 缓存 + 抽取式降级，实测平均 ~1 秒。

---

## 文档索引

| 文档 | 内容 |
| --- | --- |
| `docs/02-优化方案.md` | 三层面优化点、参数、取舍理由、MinerU 实测记录 |
| `docs/03-优化前后对比分析.md` | 10 题对比表格（准确率/时间/HitRate/MRR/NDCG） |
| `docs/04-RAGAS评估报告.md` | RAGAS 四指标 |
| `docs/技术文档.md` | 架构、选型、开发流程、使用方法、工程问题记录 |
| `docs/用户手册.md` | 界面操作、上传文档、常见问题 |
