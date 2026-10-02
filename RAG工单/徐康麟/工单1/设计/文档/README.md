# 基于 PDF 文档的 RAG 问答系统（工单1）

> 工单编号：**人工智能NLP-RAG-基于PDF 文档的问答系统**
> 项目根目录：`E:\gao6gongdan\工单1`
> 语料：《招股说明书1.pdf》（武汉兴图新科电子股份有限公司招股意向书，**548 页**，约 13.65 MB）
> 开发环境：Windows 本机 ｜ 部署目标：算力云 4090 机器

---

## 1. 项目简介

本项目是一个**面向单篇招股意向书的特攻型 RAG（检索增强生成）问答系统**。它把 548 页的《招股说明书1.pdf》解析、分块、向量化并建立关键词索引，用户用自然语言提问后，系统先做 Query 理解与混合检索，再把命中的原文片段交给本地大模型生成答案，并附上**可展开核对的页码引用**。

系统的核心主张是：**答案必须来自 PDF 原文**。检索不到相关内容、或模型无法依据片段作答时，统一回复 **“不清楚”**，不做任何编造。

### 1.1 关键指标

| 指标 | 目标值 | 说明 |
| --- | --- | --- |
| 首字返回时间 | **< 3 秒** | 从用户点击提问到界面出现第一个字，采用流式输出 |
| 答案准确性 | 基于 PDF 内容回答 | 10 个工单固定问题为主要验收样本 |
| 引用真实性 | 引用页码必须真实存在于 PDF | 前端可展开查看原文片段 |
| 兜底行为 | 无依据时回复“不清楚” | 禁止编造答案 |
| 并发能力 | 支持中等并发 | vLLM 连续批处理 + SQLite WAL + 每请求独立连接 |
| 日志完备性 | 函数级输入/输出/耗时全记录 | 禁止静默失败 |

### 1.2 工单固定测试问题（10 个）

系统的验收以以下 10 个问题为准，逐条列出如下（**演示操作清单**与**用户手册**中均可直接勾选使用）：

| # | 问题 |
| --- | --- |
| 1 | 报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？ |
| 2 | 武汉兴图新科电子股份有限公司参与制定了哪个技术标准？ |
| 3 | 报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？ |
| 4 | 根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？ |
| 5 | 武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？ |
| 6 | 根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？ |
| 7 | 武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？ |
| 8 | 武汉兴图新科电子股份有限公司注册资本是多少？ |
| 9 | 武汉兴图新科电子股份有限公司法定代表人是谁？ |
| 10 | 武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？ |

---

## 2. 文档导航

| 文档 | 内容 | 适合谁看 |
| --- | --- | --- |
| [README.md](README.md)（本文） | 项目介绍、环境安装、快速启动、目录结构、常见问题 | 所有人，第一次接触项目先读这篇 |
| [TECH_DOC.md](TECH_DOC.md) | 系统架构、技术选型理由、模块说明、数据流、检索特攻优化、数据库表结构、开发流程 | 开发、评审、二次开发者 |
| [USER_MANUAL.md](USER_MANUAL.md) | 界面操作指南：上传/选择 PDF、提问、看引用与原文、多轮对话、点赞点踩、清空对话、看首字响应时间 | 演示者、最终用户 |
| [DEMO_CHECKLIST.md](DEMO_CHECKLIST.md) | 逐步勾选式演示清单：环境启动 → 建索引 → 起界面 → 10 问 → 引用 → 多轮 → 反馈 → 兜底 → 响应时间 → 评估 | 演示者本人 |
| **[VOICE_AND_ENGLISH.md](VOICE_AND_ENGLISH.md)** | 语音输入与英文问答（追加需求）的实现、配置与限制 |

---

## 3. 目录结构

```text
工单1/                                # 项目根目录
│
├─ 设计/                              # 【阶段一】设计：需求、架构与规格说明
│  ├─ 文档/
│  │  ├─ README.md                    # 项目介绍、安装、启动、目录结构（本文件）
│  │  ├─ TECH_DOC.md                  # 技术文档：架构、模块、数据流、数据库、配置
│  │  ├─ USER_MANUAL.md               # 用户手册：操作指南
│  │  ├─ DEMO_CHECKLIST.md            # 演示操作清单（勾选式）
│  │  └─ VOICE_AND_ENGLISH.md         # 语音输入与英文问答的实现说明
│  ├─ 规格说明/
│  │  ├─ check_local_env.py           # 本地环境自检（依赖/索引/模型/端口）
│  │  ├─ dump_pdf.py                  # PDF 全文导出（带页码）
│  │  ├─ search_pdf.py                # 关键词检索 PDF 文本
│  │  └─ pdf_text.py                  # 轻量 PDF 取文
│  └─ screenshot_ui.png               # 界面截图（验收留档）
│
├─ 研发/                              # 【阶段二】研发：源码与开发脚本
│  ├─ app/                            # Python 包（导入名为 app.*）
│  │  ├─ main.py                      # 启动入口（web / ask / doctor）
│  │  ├─ ui/streamlit_app.py          # Streamlit 网页界面
│  │  ├─ core/                        # 核心模块
│  │  │  ├─ config.py                 # 配置管理（唯一路径基准 PROJECT_ROOT）
│  │  │  ├─ logging_conf.py           # 日志 + @trace 装饰器
│  │  │  ├─ pdf_parser.py             # PDF 解析（文字/页码/表格，含断行修复）
│  │  │  ├─ chunker.py                # 分块（700 字/重叠 100，表格整表成块）
│  │  │  ├─ embedder.py               # 向量化（BGE，本地模型优先）
│  │  │  ├─ vector_store.py           # 向量库（numpy 精确检索 / Chroma 可选）
│  │  │  ├─ bm25_index.py             # BM25 关键词索引（纯 Python 实现）
│  │  │  ├─ text_utils.py             # 文本清洗 / 分词 / 停用词
│  │  │  ├─ number_utils.py           # 数值规范化（金额等价写法、单位换算）
│  │  │  ├─ retriever.py              # 混合检索 + 多查询变体 + 加权合并
│  │  │  ├─ query_understanding.py    # Query 理解（意图/消歧/分解/多轮改写）
│  │  │  ├─ language.py               # 多语言：语言检测、英文意图、中文问句桥接
│  │  │  ├─ english_answer.py         # 英文答案构造（模板 + 确定性单位换算）
│  │  │  ├─ generator.py              # 生成（LLM 优先，抽取式降级，中英双语）
│  │  │  ├─ asr.py                    # 语音识别（OpenAI 接口 / 本地 Whisper）
│  │  │  ├─ citation.py               # 引用处理（[页码: N] / [Page: N]）
│  │  │  ├─ conversation.py           # 多轮对话管理
│  │  │  ├─ evaluator.py              # 评估（确定性指标 + RAGAS）
│  │  │  └─ qa_engine.py              # 问答引擎总装（对外统一入口）
│  │  ├─ storage/sqlite_manager.py    # SQLite：documents/chunks/conversations/
│  │  │                               #   messages/feedback/eval_results/logs/golden_qa
│  │  ├─ models/schemas.py            # Pydantic 数据模型
│  │  └─ prompts/                     # 提示词模板（qa_prompt、query_rewrite_prompt）
│  └─ scripts/
│     ├─ build_index.py               # 建索引（解析→分块→向量→BM25→SQLite）
│     ├─ evaluate.py                  # 评估（中文 10 题 + 英文 5 题 + RAGAS）
│     ├─ run_app.sh                   # 起网页界面
│     ├─ run_vllm.sh                  # 起 vLLM（LLM 服务）
│     ├─ run_sglang.sh                # 起 SGLang（备选 LLM 服务）
│     ├─ run_whisper.sh               # 起 Whisper（语音识别服务）
│     └─ setup_env.ps1                # 环境构建脚本
│
├─ 测试/                              # 【阶段三】测试
│  ├─ pytest.ini                      # pytest 配置
│  └─ tests/
│     ├─ conftest.py                  # 公共 fixture（索引就绪判定、库隔离、沙箱兼容）
│     ├─ offline/                     # 离线测试：解析/分块/向量/检索/生成/评估/SQLite/ASR
│     ├─ online/                      # 在线测试：接口/并发/多轮/兜底/引用/双语
│     └─ user/                        # 用户测试：验收清单、模拟用户、演示问题
│
├─ 优化/                              # 【阶段四】优化：评估与性能改进产物
│  └─ 评估结果/eval_results/
│     ├─ rag_vs_llm.csv               # 逐题对比（RAG vs 纯 LLM）
│     ├─ ragas_report.md              # 评估报告（含中英文两组指标）
│     └─ eval_records.json            # 完整明细（答案、引用、耗时）
│
├─ 部署/                              # 【阶段五】部署
│  ├─ run.ps1                         # **统一启动入口**（web/index/eval/test/doctor/ask/envs）
│  └─ 环境配置/
│     ├─ 部署/环境配置/requirements.txt             # Python 依赖清单
│     └─ 部署/环境配置/environment.yml              # conda 环境定义
│
├─ data/                              # 运行产物（与阶段无关，便于算力云挂载数据盘）
│  ├─ raw/招股说明书1.pdf             # 原始语料（548 页）
│  ├─ processed/                      # 解析结果：pages.txt / parsed.json / tables.jsonl / chunks.jsonl
│  ├─ index/                          # 向量索引 vectors.npy、BM25 索引、rag.sqlite3
│  └─ eval/golden_qa.jsonl            # 10 题标准答案（含 PDF 原始依据与页码）
│
├─ logs/                              # app.log / error.log / rag_trace.jsonl / user_simulation.log
└─ models/                            # 本地模型
   ├─ bge-small-zh-v1.5/              # 嵌入模型（必需）
   ├─ whisper-tiny/                   # 本地语音识别（可选）
   └─ Qwen3-0.6B/                     # 本地翻译/润色（可选）
```

> **为什么 data / logs / models 留在根目录**：这三者是**运行期产物或大体积资产**，
> 不属于任何单一交付阶段（设计不产出它们，测试与部署都要读写它们）。
> 放在根目录还便于在算力云上单独挂载数据盘与模型盘。
> 源码中的路径基准只有一处 —— `研发/app/core/config.py` 的 `PROJECT_ROOT`，
> 其余模块一律通过 `settings.paths.*` 取路径，因此搬迁目录只需改这一处。

### 3.1 开发期辅助目录（非交付物）

| 路径 | 用途 |
| --- | --- |
| `_tools/pdf_text.py` | 小工具：抽取 PDF 每页文本，页间以 `===PAGE n===` 分隔 |
| `_tools/search_pdf.py` | 小工具：在 PDF 文本中按关键词定位页码 |
| `_tools/dump_pdf.py` | 小工具：批量导出 PDF 文本 |
| `_tools/install_missing.py` | 在外网较慢时逐个安装缺失依赖（每包限时，失败跳过） |
| `_tools/link_local_packages.py` | 把本机其它 conda 环境里已有的包以目录联接方式复用 |
| `_work/prospectus.txt` | 《招股说明书1.pdf》全文文本导出（548 页，开发期核对用） |
| `.gao6gongdan-src/` | 开发期暂存的一个 **Python 3.10.21** conda 环境目录（含 `conda-meta`/`Library`/`Scripts`/`python.exe`），内含 streamlit、chromadb、sentence-transformers、pymupdf、pandas、numpy、torch(CPU) 等依赖。**它不是交付环境**：工单要求的是 **Python 3.11** 的 conda 环境 `gao6gongdan`，两者不一致，详见 3.2 与 5.1 节 |
| `.venv/` | 项目内临时虚拟环境（Python 3.11，仅含 jieba / loguru / pymupdf），开发期小工具用 |

> 这三个目录是开发过程的脚手架，**不参与交付评审**，也不被 `app/` 下任何模块导入。

### 3.2 实现状态（截至 2026-10-01 19:20 核对）

项目**正在开发中**，多个模块由不同开发者在并行推进。本节如实标注核对时刻代码仓中的文件情况，方便读者判断哪些命令现在就能跑。

| 状态 | 文件 |
| --- | --- |
| ✅ 已存在 | `app/core/config.py`、`app/core/logging_conf.py`、`app/core/text_utils.py`、`app/core/bm25_index.py`、`app/models/schemas.py`、`app/storage/sqlite_manager.py`、`app/ui/__init__.py`、`app/ui/streamlit_app.py`、`部署/环境配置/requirements.txt` |
| ✅ 已产生 | `logs/app.log`、`logs/rag_trace.jsonl`（`logs/error.log` 尚未出现，需触发一次异常才会创建） |
| 🚧 待创建 | `app/main.py`、`app/core/` 下 `pdf_parser`、`chunker`、`embedder`、`vector_store`、`retriever`、`query_understanding`、`generator`、`citation`、`conversation`、`evaluator`、`qa_engine`、`app/prompts/` 两个提示词、`data/` 下的语料与索引、`scripts/` 五个脚本、`tests/` 三个测试目录、`部署/环境配置/environment.yml` |

> **关键依赖提醒**：网页界面 `app/ui/streamlit_app.py` 已实现，但它需要 `app/core/qa_engine.py` 提供引擎对象。**在 `qa_engine.py` 落地之前**，启动界面会显示“问答引擎尚未就绪”，并给出“🔄 重新初始化问答引擎”按钮；此时无法提问。
>
> 因此：本文第 4~6 节的安装步骤现在即可执行；第 7 节的“快速启动”中，**建索引与提问两步需要对应模块落地后才能跑通**。各模块的详细状态见 [TECH_DOC.md](TECH_DOC.md) 第 4 节。

---

## 4. 环境要求

### 4.1 硬件与系统

| 项 | 开发环境（本机） | 部署环境（算力云） |
| --- | --- | --- |
| 操作系统 | Windows | Linux |
| GPU | 无（可纯 CPU 跑通解析/索引/降级链路） | NVIDIA RTX 4090（24 GB 显存） |
| 内存 | 建议 ≥ 16 GB | 建议 ≥ 32 GB |
| 磁盘 | 建议 ≥ 20 GB 可用 | 建议 ≥ 50 GB 可用 |

### 4.2 软件

| 软件 | 版本 | 说明 |
| --- | --- | --- |
| Anaconda / Miniconda | 任意较新版本 | 本机安装于 `E:\Anaconda`，环境目录 `E:\Anaconda\envs` |
| Python | **3.11** | conda 环境名固定为 **`gao6gongdan`** |
| CUDA 驱动 | 支持 CUDA 12.x | 仅 GPU 部署机器需要 |
| vLLM 或 SGLang | 独立环境安装 | 与主环境隔离，避免 torch/CUDA 版本冲突 |

### 4.3 模型清单

| 用途 | 默认模型 | 可选替代 | 体积/说明 |
| --- | --- | --- | --- |
| 生成（LLM） | `Qwen2.5-7B-Instruct-AWQ` | `Qwen2.5-7B-Instruct-GPTQ-Int4`、`ChatGLM3-6B`、`Yi-6B` | 以 OpenAI 兼容服务方式提供 |
| 嵌入（Embedding） | `BAAI/bge-small-zh-v1.5` | `BAAI/bge-large-zh-v1.5`、`BAAI/bge-m3` | 默认模型约 95 MB，便于快速部署 |
| 重排（Reranker） | `BAAI/bge-reranker-base` | — | **可选，默认关闭**（`RAG_RETRIEVAL__USE_RERANKER=true` 开启） |

---

## 5. 环境安装

> 以下命令在 **PowerShell** 中执行（Windows 开发机）。算力云 Linux 机器把 conda 激活命令换成 `source activate gao6gongdan` 即可。

### 5.1 创建 conda 环境

```powershell
# 环境名固定为 gao6gongdan，Python 3.11
conda create -n gao6gongdan python=3.11 -y

# 激活（Windows 用 activate，Linux 用 source activate）
conda activate gao6gongdan

# 校验
python -V
```

预期输出形如：

```text
Python 3.11.x
```

> **环境现状（2026-10-01 核对）**：
> - `E:\Anaconda\envs` 下**当前还没有** `gao6gongdan` 这个环境，请先执行上面的创建命令；
> - 项目根目录下另有 `.venv`（Python 3.11，仅含 jieba / loguru / pymupdf），是开发期临时环境，**不作为交付环境**；
> - 项目根目录下还有一个隐藏目录 `.gao6gongdan-src\`，它是开发期暂存的一个 **Python 3.10.21** conda 环境目录（内含 `python.exe` 与 streamlit/chromadb/sentence-transformers 等依赖）。它**不是**工单要求的交付环境，因为工单/需求说明书明确规定 **Python 3.11** 与 conda 环境名 `gao6gongdan`；
> - 另需注意：`app\core\__pycache__\` 中现存 `*.cpython-312.pyc`，说明这些模块曾被 **Python 3.12** 解释器导入过。建议统一解释器后删除 `__pycache__` 目录，避免混淆。
>
> **结论：请以本节创建 Python 3.11 的 `gao6gongdan` 环境为准**；若确实要改用 Python 3.10 或 3.12，请先与需求方确认（见 [TECH_DOC.md](TECH_DOC.md) 第 17 节）。

### 5.2 安装依赖

```powershell
cd E:\gao6gongdan\工单1

# 方式 A：直接安装（推荐，使用清华镜像加速）
pip install -r 部署/环境配置/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方式 B：离线安装（先在能联网的机器上下载 wheel）
# pip download -r 部署/环境配置/requirements.txt -d wheels/
# pip install --no-index --find-links wheels/ -r 部署/环境配置/requirements.txt

# 校验关键依赖
python -c "import streamlit, chromadb, sentence_transformers, rank_bm25, loguru, pymupdf, pdfplumber; print('deps ok')"
```

### 5.3 安装推理框架（与主环境隔离）

vLLM 或 SGLang **不要装进 `gao6gongdan`**，另建环境：

```bash
# 在算力云 4090 机器上（Linux）
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm
```

仓库提供了现成的启动脚本（需对应脚本落地后使用）：

```bash
bash 研发/scripts/run_vllm.sh      # 启动 vLLM 的 OpenAI 兼容服务
bash 研发/scripts/run_sglang.sh    # 备选：启动 SGLang
```

### 5.4 准备模型权重

```powershell
# HuggingFace 直连较慢时可先设置镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 下载嵌入模型（约 95 MB，默认；首次运行 embedder 时也会自动下载）
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5')"

# 可选：更大的嵌入模型 / 重排模型
# python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-large-zh-v1.5')"
# python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3')"
# python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-base')"
```

若暂时不下载嵌入模型，`embedder` 会自动降级为 **hash 向量**，整条链路仍可离线跑通（检索质量会下降，仅用于联调）。

### 5.5 放置语料 PDF

系统默认读取 `data/raw/招股说明书1.pdf`：

```powershell
cd E:\gao6gongdan\工单1
New-Item -ItemType Directory -Force -Path data\raw, data\processed, data\index, 优化\评估结果\eval_results, logs | Out-Null

# 把《招股说明书1.pdf》放到 data\raw\ 下（文件名必须一致）
Copy-Item "C:\Users\徐康麟\.dsh\attachments\v1\files\9d\9d69903ca390654c0994f60c0a72904ea9b2229c5ecbf44caa73bdc2434a9b98\招股说明书1.pdf" "data\raw\招股说明书1.pdf"

# 校验
Get-Item "data\raw\招股说明书1.pdf" | Select-Object Name, Length
```

> 该附件路径是开发期语料的来源位置，请以实际保存位置为准；**只要 `data\raw\招股说明书1.pdf` 存在且是那 548 页的招股意向书即可**。

---

## 6. 配置说明

全部配置集中在 `app/core/config.py`，并用 `RAG_<段名>__<字段名>` 形式的环境变量覆盖（**两个下划线**分隔段名与字段名，大小写不敏感）。

```powershell
# 示例：把 LLM 服务地址指到本机 vLLM
$env:RAG_LLM__BASE_URL = "http://127.0.0.1:8000/v1"

# 示例：扩大检索召回
$env:RAG_RETRIEVAL__VECTOR_TOP_K = "20"

# 示例：开启重排模型
$env:RAG_RETRIEVAL__USE_RERANKER = "true"

# 示例：换用更大的嵌入模型
$env:RAG_EMBEDDING__MODEL_NAME = "BAAI/bge-large-zh-v1.5"
```

Linux 上对应写法：

```bash
export RAG_LLM__BASE_URL=http://127.0.0.1:8000/v1
```

常用配置速查（完整清单见 [TECH_DOC.md](TECH_DOC.md) 第 8 节）：

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `RAG_LLM__BASE_URL` | `http://127.0.0.1:8000/v1` | LLM 服务地址（vLLM/SGLang 的 OpenAI 兼容接口） |
| `RAG_LLM__MODEL` | `Qwen2.5-7B-Instruct-AWQ` | 调用的模型名 |
| `RAG_LLM__TEMPERATURE` | `0.1` | 生成温度，越低越稳定 |
| `RAG_EMBEDDING__MODEL_NAME` | `BAAI/bge-small-zh-v1.5` | 嵌入模型 |
| `RAG_EMBEDDING__DEVICE` | `cpu` | 嵌入/重排设备（`cpu` / `cuda`） |
| `RAG_RETRIEVAL__VECTOR_TOP_K` | `10` | 向量召回条数 |
| `RAG_RETRIEVAL__BM25_TOP_K` | `10` | BM25 召回条数 |
| `RAG_RETRIEVAL__RERANK_TOP_N` | `5` | 融合后交给 LLM 的片段数 |
| `RAG_RETRIEVAL__USE_RERANKER` | `false` | 是否启用 bge-reranker-base |
| `RAG_RETRIEVAL__MIN_RELEVANCE_SCORE` | `0.08` | 低于该分数判定为“无相关内容”，走兜底 |
| `RAG_CHUNK__CHUNK_SIZE` | `700` | 分块长度 |
| `RAG_CHUNK__CHUNK_OVERLAP` | `100` | 分块重叠 |
| `RAG_CONVERSATION__MAX_HISTORY_ROUNDS` | `5` | 多轮对话保留轮数 |
| `RAG_APP__LOG_LEVEL` | `INFO` | 日志级别 |
| `RAG_APP__UNKNOWN_ANSWER` | `不清楚` | 统一兜底回复 |

---

## 7. 快速启动

> **前置条件**：`研发/scripts/build_index.py`、`app/core/qa_engine.py`、`app/main.py` 等模块需先就位。`app/ui/streamlit_app.py` 已实现，但它依赖 `qa_engine.py`；引擎缺失时界面会提示“问答引擎尚未就绪”。当前实现状态见 3.2 节。

整体分 4 步，建议开 **3 个终端**。

### 步骤 1：准备语料

确认 `data/raw/招股说明书1.pdf` 已存在（见 5.5）。

### 步骤 2：建立索引

```powershell
conda activate gao6gongdan
cd E:\gao6gongdan\工单1

python 研发/scripts/build_index.py --pdf "data/raw/招股说明书1.pdf"
```

该脚本依次完成：PDF 解析 → 分块 → 写入 SQLite（`chunks` 表）→ 生成向量写入 Chroma（`data/index/`）→ 建立 BM25 索引。预期输出形如：

```text
[阶段1] 解析完成：page_count=548, table_count=...
[阶段1] 分块完成：chunk_count=...
[阶段1] 向量索引完成：collection=..., vectors=...
[阶段1] 索引构建结束，耗时 ...s
```

### 步骤 3：启动本地大模型服务（GPU 机器）

```bash
# 终端 A（算力云 4090）
bash 研发/scripts/run_vllm.sh
# 服务就绪后，http://127.0.0.1:8000/v1/models 应能返回模型列表
```

没有 GPU 时，可跳过本步骤：系统会降级为**抽取式回答**（直接把命中片段整理后返回），链路仍可演示，但答案自然度会下降。

### 步骤 4：启动网页界面

```powershell
# 终端 B（本机）
conda activate gao6gongdan
cd E:\gao6gongdan\工单1
streamlit run app/ui/streamlit_app.py
```

浏览器打开终端提示的地址（Streamlit 默认 `http://localhost:8501`），即可开始提问。完整操作流程见 [USER_MANUAL.md](USER_MANUAL.md)，演示时请对照 [DEMO_CHECKLIST.md](DEMO_CHECKLIST.md) 逐项勾选。

### 一键脚本

```bash
bash 研发/scripts/run_app.sh    # 启动界面（Linux / Git Bash）
```

---

## 8. 日志与排查

| 文件 | 内容 | 何时看 |
| --- | --- | --- |
| `logs/app.log` | 全量日志，**JSON 行格式**，级别 DEBUG 起 | 常规排查 |
| `logs/error.log` | 仅异常，带完整堆栈 | 报错时先看这个 |
| `logs/rag_trace.jsonl` | **函数级**输入/输出/耗时（`@trace` 装饰器写入） | 定位“哪一步慢”“哪一步返回空” |

`rag_trace.jsonl` 每条记录形如：

```json
{"ts":"2026-01-01T10:00:00.000+08:00","event":"exit","module":"app.core.retriever","function":"Retriever.search","elapsed_ms":123.456,"result":{...}}
```

常用排查命令：

```powershell
# 看最近的报错
Get-Content logs\error.log -Tail 50

# 找耗时最长的步骤（按 elapsed_ms 排序，取前 20）
Get-Content logs\rag_trace.jsonl | Select-String "elapsed_ms" | Select-Object -Last 200

# 统计各函数平均耗时（示例：只看检索器）
Select-String -Path logs\rag_trace.jsonl -Pattern '"function":"Retriever' | Select-Object -Last 50
```

排查思路：**先看 `logs/error.log` 有没有堆栈 → 再看 `logs/rag_trace.jsonl` 中 `event":"error"` 的记录定位失败函数 → 最后看 `rag_trace.jsonl` 的 `elapsed_ms` 找性能瓶颈。**

---

## 9. 常见问题（FAQ）

### Q1：`conda activate gao6gongdan` 报错「Could not find conda environment」

环境还没创建。执行 `conda create -n gao6gongdan python=3.11 -y` 重建（见 5.1）。若 `conda` 命令本身不可用，在 Windows 上可用完整路径 `& "E:\Anaconda\Library\bin\conda.bat" activate gao6gongdan`，或先运行 `conda init powershell` 后重开终端。

### Q2：`data/raw/招股说明书1.pdf` 找不到，建索引直接失败

语料未就位。按 5.5 节把 PDF 复制到 `data\raw\` 下，文件名必须是 `招股说明书1.pdf`。系统的默认路径写在 `app/core/config.py` 的 `PathSettings.default_pdf`，也可以用 `--pdf` 参数指定别的路径。

### Q3：装不上 `chromadb` / `sentence-transformers`

- 先换镜像：`pip install -r 部署/环境配置/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`；
- 仍失败时，系统**不强依赖**这两个包：`vector_store` 缺依赖时降级为 numpy 内存索引，`embedder` 无模型时降级为 hash 向量，链路仍可跑通（用于联调，检索质量下降）；
- 本机另有 `_tools/install_missing.py`、`_tools/link_local_packages.py` 两个开发期脚本，可在外网很慢时逐个装包或复用其它环境里已有的纯 Python 包。

### Q4：首字响应时间超过 3 秒

按可能性从高到低排查：

1. **LLM 服务未预热**——vLLM 首次请求包含模型加载/CUDA 图捕获，第一条问题必然慢，先随便问一次热身；
2. **走的是 CPU 嵌入**——548 页语料的查询向量化在 CPU 上有开销，GPU 机器上把 `RAG_EMBEDDING__DEVICE=cuda`；
3. **检索过重**——把 `RAG_RETRIEVAL__VECTOR_TOP_K`/`BM25_TOP_K` 调小，或保持 `USE_RERANKER=false`；
4. **网络绕路**——确认 `RAG_LLM__BASE_URL` 指向本机/内网，而不是公网地址。

具体是哪一步慢，看 `logs/rag_trace.jsonl` 的 `elapsed_ms` 字段。

### Q5：答案总是“不清楚”

按顺序检查：

1. 索引是否建好——`data/index/` 下是否有向量索引文件、`rag.sqlite3` 的 `chunks` 表是否有数据（见 Q6）；
2. 检索阈值是否过严——`RAG_RETRIEVAL__MIN_RELEVANCE_SCORE` 默认 `0.08`，若嵌入模型降级成 hash 向量，分数分布会整体偏低，可临时调低验证；
3. LLM 服务是否可用——服务不通时会走抽取式降级；若片段本身为空，仍然返回“不清楚”；
4. 问题是否真的超出 PDF 范围——这种情况下“不清楚”就是**正确行为**。

### Q6：怎么快速看数据库里有什么

```powershell
python -c "from app.storage.sqlite_manager import get_sqlite_manager as g; s=g().stats(); print(s)"
```

预期输出形如：

```text
{'documents': 1, 'chunks': ..., 'conversations': ..., 'messages': ..., 'feedback': ..., 'eval_results': ..., 'golden_qa': 10, 'logs': ...}
```

### Q7：端口冲突（8000 被占 / 8501 被占）

```powershell
# 查看占用
netstat -ano | findstr :8000
netstat -ano | findstr :8501

# 界面换端口
streamlit run app/ui/streamlit_app.py --server.port 8502

# LLM 服务换端口后，同步改配置
$env:RAG_LLM__BASE_URL = "http://127.0.0.1:8001/v1"
```

### Q8：想换嵌入模型或开启重排，要重建索引吗？

- **换嵌入模型：必须重建索引**（向量维度与语义空间都变了）；
- **开启重排模型：不需要重建索引**（重排作用于召回结果之上）。

### Q9：日志文件越来越大怎么办

`app.log` 与 `error.log` 已配置轮转与保留（分别 50 MB / 20 MB 轮转、保留 10 份），无需手工清理。`logs/rag_trace.jsonl` 为追加写入、不做轮转，长期运行请自行归档或清理。

### Q10：评估报告在哪里、怎么生成

```powershell
python 研发/scripts/evaluate.py
```

产物：

- `优化/评估结果/eval_results/rag_vs_llm.csv` —— RAG 与纯 LLM 的逐题对比
- `优化/评估结果/eval_results/ragas_report.md` —— RAGAS 四项指标报告

评估指标与判定口径见 [TECH_DOC.md](TECH_DOC.md) 第 11 节。

---

## 10. 工单信息

| 项 | 内容 |
| --- | --- |
| 工单编号 | 人工智能NLP-RAG-基于PDF 文档的问答系统 |
| 工单来源 | 八维文化与产业研究院 / 北京八维信息集团（2025 年 1 月） |
| 工时预估 | 2 人日 |
| 代码约定 | 代码注释使用**中文**，并包含工单编号 |
| 交付物 | 代码、`部署/环境配置/requirements.txt`、`部署/环境配置/environment.yml`、`data/processed/`、`data/index/`、`优化/评估结果/eval_results/`、`tests/`、四份 `docs/` 文档、`logs/` 日志示例、启动脚本与评估脚本 |

## 11. 语音输入与英文问答（追加需求）

工单追加要求“支持语音输入”与“支持英文问答”，两者均已实现，详见
**[`设计/文档/VOICE_AND_ENGLISH.md`](VOICE_AND_ENGLISH.md)**。

- **语音输入**：界面提供录音控件（Streamlit < 1.40 时退化为音频文件上传），
  识别后端优先调用 vLLM 的 Whisper 服务（`研发/scripts/run_whisper.sh`），
  也可离线用本地 transformers Whisper（只支持 WAV）。
  未部署语音服务时界面给出明确中文提示，文字问答不受影响。
- **英文问答**：英文问题经“英文意图识别 → 中文问句模板 → 中文混合检索 →
  英文答案模板”链路作答，引用格式为 `[Page: N]`。
  金额单位由代码确定性换算（避免小模型把“5,520 万元”译成 “5,520 million yuan”）。
  实测英文 5 题准确率 100%，中文 10 题不受影响。

```bash
# 英文对照评估
python 研发/scripts/evaluate.py --no-llm --english
```
