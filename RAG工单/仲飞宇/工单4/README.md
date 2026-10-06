# RAG 项目 · 基于 PDF 文档的问答系统（工单01 → 工单04）

> 工单编号：**人工智能NLP-RAG-基于PDF文档的问答系统**（工单01）
> 工单编号：**人工智能NLP-RAG-基于PDF文档的问答系统优化**（工单02）
> 工单编号：**人工智能NLP-RAG-PDF文档的表格解析及检索优化**（工单03）
> 工单编号：**人工智能NLP-RAG-图像内容解析及检索优化**（工单04）
> 项目：八维 NLP-RAG 项目（共 18 份工单，本仓库覆盖工单 01、02、03、04）

针对两份招股意向书的 RAG 问答系统：
PDF 文字+表格解析 → **图表/结构图多模态转写（入库期）** → 句子边界分块 → 去重 →
bge-m3 向量化 → Milvus 检索 → **实体路由** → 同事实聚合 + 邻块扩展 →
qwen3:8b 流式生成 → 三轨评估。

| 文档 | 页数 | 块数 | 公司 |
|---|---|---|---|
| `招股说明书1.pdf` | 548 | 1132 | 武汉兴图新科电子股份有限公司 |
| `招股说明书2.pdf` | 350 | 716 | 武汉力源信息技术股份有限公司 |

- **工单01**：把系统从零搭起来并跑通（10/10 命中）。
- **工单02**：在 01 的代码上做检索优化，产出「优化前后检索精确度对比」。
  方案见 **[`docs/工单02-优化方案.md`](docs/工单02-优化方案.md)**，取证见
  **[`docs/工单02-演示截图清单.md`](docs/工单02-演示截图清单.md)**。
- **工单03**：加入第二份 PDF，用表格解析把答案定位到表，并按问题里的公司专名
  自动分流。方案见 **[`docs/工单03-表格解析与多文档检索.md`](docs/工单03-表格解析与多文档检索.md)**，
  取证见 **[`docs/工单03-演示截图清单.md`](docs/工单03-演示截图清单.md)**。
- **工单04**：用**多模态模型**把图（组织图/图表）在**入库期**转写成文字，
  问答期链路一行不改（3 秒预算不受影响）。方案见
  **[`docs/工单04-图像内容解析及检索优化.md`](docs/工单04-图像内容解析及检索优化.md)**，
  取证见 **[`docs/工单04-演示截图清单.md`](docs/工单04-演示截图清单.md)**。

---

## 快速开始

```bash
# 0. 环境自检（第一道拦截，必须先跑；共 29 项）
$PY scripts/preflight.py

# 0b. 多模态模型（工单04，一次性）
ollama pull qwen2.5vl:3b

# 1. 启动向量库（Milvus Standalone + Attu）
docker compose -f docker-compose.milvus.yml up -d
#    Attu 可视化： http://localhost:8000

# 2. 图像转写作业台（工单04，**不碰 Milvus**，可反复跑）
$PY scripts/transcribe_images.py --run    # 跑 VLM（缓存命中的会跳过）
$PY scripts/transcribe_images.py --list   # 一览：页/类型/题注/是否已人工核对

# 3. 入库（图像解析随流水线自动发生）
$PY scripts/ingest_all.py                 # ★ 工单03/04：把已配置的两份 PDF 都灌进去
$PY scripts/ingest_all.py --status        #    只看库里有哪些文档
#    旧的单文档入口仍在（只灌书1）：$PY scripts/ingest.py

# 4. 起服务（注意 8080 —— Attu 占着 8000）
$PY -m uvicorn app.main:app --host 0.0.0.0 --port 8080
#    问答界面： http://localhost:8080

# 5. 评估（题库 16 题：10 道书1 + 4 道书2 + 2 道图题）
$PY scripts/eval.py --profile optimized   # ★ 工单04：16 题口径
$PY scripts/eval.py --compare all         # 工单02：三剖面横向对比

# 6. 工单02 专项实验
$PY scripts/chunker_ablation.py           # 分块层对照（离线，不需要向量库）
$PY scripts/sweep.py                      # 检索参数扫描（两阶段）

# 7. 测试
$PY -m pytest tests/ -v                   # 58 passed
```

（`$PY` = `~/rag-data/venv/bin/python`，数据与 venv 都在 WSL 原生盘，避开 `/mnt/c` 的 9p 慢速。）

## 16 题实测（工单04 核心产出）

`$PY scripts/eval.py --profile optimized`，报告 `~/rag-data/eval/eval-20261003-193522.json`
（本目录 `eval报告/` 下附了存档 json + csv）：

| 指标 | 实测 | 工单要求 |
|---|---|---|
| 规则化命中（准确率） | **16/16 = 100%** | ≥ 90% ✅ |
| TTFT ≤ 3s | **16/16**（P50 1945 / P95 2704 / 最大 2842 ms） | ≤ 3s ✅ |
| 答案正确性（LLM judge） | 0.9875 | — |
| 上下文召回 / 忠实度 | 0.975 / 1.0 | — |

其中 2 道图题（id 5/6）靠**入库期多模态转写**拿到答案；书1 的 10 道题无回归。

## 知识库构成

| 文档 | 块数 | text | table | **image** |
|---|---|---|---|---|
| 招股说明书1.pdf | 1132 | — | — | **16** |
| 招股说明书2.pdf | 716 | 463 | 225 | **30** |
| 合计 | 1848 | — | — | **46** |

## 检索剖面（工单02）

同一份向量库、同一套 prompt，只切换检索策略 —— 这是"优化前后对比"的地基：

| 剖面 | 召回 | 重排 | 冗余过滤 | 窗口 | 新增 |
|---|---|---|---|---|---|
| `baseline` 朴素基线 | 单路 · pool 10 | 无 | 无 | 300 | — |
| `delivered` 工单01 交付版 | 双路 · pool 30 | 词法 0.5/0.5 | Jaccard 0.85 | 600 | — |
| `optimized` 工单02 优化版 | 双路 · pool 30 | 词法 0.5/0.5 | **同事实聚合** | 600 | **邻块扩展** |

用法：前端问答页/评估页的剖面下拉框，或 `--profile` / `--compare` 命令行参数，
或 `/api/chat`、`/api/evaluate/run` 的 `profile=` 参数。

10 题实测（详见优化方案第四章）：

| 剖面 | 规则命中 | CKC | 页精确 | 页召回 | TTFT-P50 | ≤3s |
|---|---|---|---|---|---|---|
| `baseline` | 6/10 | 66.7% | 16.7% | 30.0% | 1166 ms | 10/10 |
| `delivered` | 10/10 | 96.7% | 26.7% | 53.3% | 1184 ms | 10/10 |
| `optimized` | **10/10** | **96.7%** | **30.0%** | **56.7%** | **1104 ms** | **10/10** |

## 目录结构

```
（仓库根目录就是项目根）
├── app/
│   ├── main.py              FastAPI 入口（生命周期、静态挂载、路由注册）
│   ├── config.py            配置（宿主 IP 动态发现、剖面默认值）
│   ├── schemas.py           Pydantic 模型
│   ├── api/                 chat · ingest · kb · evaluate · asr · health
│   ├── core/                ollama_client · pdf_parser · chunker · dedup ·
│   │                        embedder · vectorstore · retriever · generator ·
│   │                        evaluator · pipeline · profiles ·
│   │                        doc_profiles【工单03】· router【工单03】·
│   │                        figure_locator【工单04】· vision【工单04】·
│   │                        image_cache【工单04】
│   └── static/              index.html · app.js · style.css
├── docs/
│   ├── 工单02-优化方案.md          【工单02 产出物一】含优化前后对比
│   ├── 工单02-演示截图清单.md       【工单02 产出物二】20 张逐条步骤
│   ├── 工单03-表格解析与多文档检索.md  【工单03 产出物一】
│   ├── 工单03-演示截图清单.md        【工单03 产出物二】19 张逐条步骤
│   ├── 工单04-图像内容解析及检索优化.md 【工单04 产出物一】
│   ├── 工单04-演示截图清单.md        【工单04 产出物二】19 张逐条步骤
│   └── 演示截图/                   ← 截图存这里（.gitignore 已白名单放行）
├── eval/questions.json      16 题题面 + 人工核实真值（10 书1 + 4 书2 + 2 图题）
├── scripts/                 preflight · ingest · ingest_all【工单03】·
│                            transcribe_images【工单04】·
│                            eval · sweep · chunker_ablation
├── tests/                   pytest 冒烟（58 例）
└── docker-compose.milvus.yml
```

数据目录（WSL 原生盘，避开 `/mnt/c` 的 9p 慢速）：
`~/rag-data/{raw,parsed,eval,feedback,venv}`；评估与扫描报告写在 `~/rag-data/eval/`。

## 关键设计取舍

| 决策 | 选择 | 理由（均为实测） |
|---|---|---|
| 向量库 | **Milvus Standalone** | Lite 不监听端口 → Attu 连不上；且单进程文件锁过不了高并发验收 |
| 分块 | **句子边界优先**（300–600 字） | 固定字数会切断 p128 那句同时装着两道题答案的 137 字金句（离线对照：句边界 10/10 vs 固定 500 字 8/10） |
| 表格 | **`find_tables()` + 表头下推** | 纯文本流抽出的是粘连的 `发行前每股净资产3.55元/股…` |
| 页眉清洗 | **锚定整行模板** | 按关键词删会误伤正文（公司全称在正文反复出现）→ 静默答错 |
| 去重 | **入库只压精确重复；冗余控制放检索时** | 实测"重复段落"chunk 间海明距离 14–34，放宽阈值会误删真实内容 |
| 语音 | **后端 faster-whisper** | Chrome Web Speech 走 Google 服务器，国区不可用且报错误导 |
| 评估 | **自实现三轨** | ragas 走 Ollama 有三个必炸点；规则化数值命中比 LLM 打分更可信 |
| 响应口径 | **TTFT**（并主动披露完整耗时） | 工单写"返回答案的时间"，流式下首 token 即开始返回；**n=10 时 P95 = 最大值，达标看 P50 与「≤3s 的题数」** |
| 同事实聚合 | **句子级 6-gram + 并查集** | 整块 Jaccard 只有 0.055，抓不到"同一句话的三种矛盾说法"；传递闭包必需（316~355=0.48 但都能连到 358） |
| 邻块扩展 | **`chunk_index±1` + 互补性过滤** | 不填 `parent_id`：`section_path` 已编码章节，真填要全量重入库而嵌入无重试 |
| 嵌入放 CPU | **`embed_query_on_cpu=True`** | bge-m3 上 GPU 会挤掉 qwen3，TTFT 从 35ms 劣化到 6117ms |
| 响应缓存 | **放 `api/` 不放 `core/`** | 评测器直接调 core，因此评测天然不吃缓存、TTFT 数字是真的 |
| 多文档分流【工单03】 | **按公司专名硬过滤 `doc_name`** | 两份招股书章节结构雷同（"发行股数"两边都有），排序无法可靠区分；过滤后零召回才回退全库 |
| 文档差异【工单03】 | **收进 `doc_profiles.py`，一份 PDF 一条** | 页眉/页脚/页码格式/专名/证据正则/提示词口径原先散在 6 个文件里写死书1；不配的文档**当场报错**，不静默套用 |
| 提示词口径【工单03】 | **system prompt 按文档拼** | 写死"基于《兴图新科招股意向书》"会让模型对力源的题**系统性拒答**（实测 CKC=100% 却答"未提及"） |
| 表格邻块【工单03】 | **表格邻块不受互补性过滤** | 引子句"拟投资以下项目："与表格被切在相邻两块时，表格不含新查询词会被过滤掉 —— 可答案是它 |
| 图像解析【工单04】 | **多模态模型只在入库期用** | 3B VLM 读一张图 ≈ 2s，放查询期会直接击穿"≤3s"；入库期转写成文字后，问答链路一行没改 |
| 图像解析【工单04】 | **不用 CLIP，用多模态大模型** | CLIP 只给图文相似度、**没有计数能力**（答不了"销售部有几个部门"），且要加一路 CLIP 向量就得重建 schema |
| VLM 结果可覆写【工单04】 | **`reviewed_text` 优先于 `vlm_text`** | 实测模型把柱状图 `-2.0%` 的归属记错（记成"计算机"，实际是 IC卡），而那道题问的恰恰是它 |
| 转写缓存【工单04】 | **内容寻址**（PNG 字节 + 模型 + 提示词版本） | 全量重建时书2 的 33 张图 **33 次缓存命中 / 0 次 VLM 调用 / 0.6 秒** |

## 环境要求

- WSL2（`/etc/wsl.conf` 需 `systemd=true`）、Python 3.11、Docker
- Windows 侧 Ollama，模型：`qwen3:8b`、`bge-m3`、**`qwen2.5vl:3b`（工单04）**
- `.wslconfig` 建议 `memory=8GB`、`swap=8GB`
- **不需要**设 `OLLAMA_MAX_LOADED_MODELS=2`：嵌入跑在 CPU、不占显存，两个模型本来就并存
  （实测 `基准 39ms → 交替1 40ms → 交替2 46ms`）。只有把 `embed_query_on_cpu` 改成
  `False` 时才需要。

## 代码规范

每个源文件头部均含工单编号注释（工单硬约束；四个工单的编号不同，被改到的文件要都写上）：

```bash
grep -rn "人工智能NLP-RAG-基于PDF文档的问答系统" --include="*.py" --include="*.js" --include="*.html" app/ scripts/ tests/
grep -rn "人工智能NLP-RAG-基于PDF文档的问答系统优化" --include="*.py" --include="*.js" --include="*.html" app/ scripts/ tests/
grep -rn "人工智能NLP-RAG-PDF文档的表格解析及检索优化" --include="*.py" --include="*.js" --include="*.html" app/ scripts/ tests/
grep -rn "人工智能NLP-RAG-图像内容解析及检索优化" --include="*.py" --include="*.js" --include="*.html" app/ scripts/ tests/
```
