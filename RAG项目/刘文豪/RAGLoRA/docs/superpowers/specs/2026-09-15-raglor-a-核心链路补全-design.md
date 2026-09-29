# RAGLoRA 核心链路补全 · 设计文档

> 日期：2026-09-15
> 场景：课程/毕设答辩，老师按课程清单逐条核查
> 环境约束：可安装新 pip 包、可启动 Docker
> 状态：已与用户确认设计，待转实施计划

---

## 一、背景与目标

### 1.1 背景

RAGLoRA 是一个已完成的、可运行的多用户多角色 RAG 角色扮演系统，现有能力：

- 混合检索（bge-m3 dense + learned sparse → RRF 融合）
- 精排（bge-reranker-v2-m3，GPU 分时复用）
- 多轮记忆（Redis 短期 + MySQL 回源）
- 流式输出（SSE）、链路可视、知识库管理、检索调试台
- 自研评测脚本，实测 43 题：来源命中率 100%、引用标注率 100%

对照课程清单逐条核查后，存在四类缺口：

1. **向量库**：清单要求 Milvus，现用 Qdrant 嵌入式
2. **评测**：清单要求 RAGAS，现为自研脚本
3. **多路召回**：清单要求跨数据源（Milvus/MySQL/Neo4j/…），现为单库内两路
4. **文档解析**：清单要求 OCR / MinerU / PaddleOCR / pdfplumber，现仅 PyMuPDF
5. **框架**：清单要求 LangChain 等，现为纯手写

### 1.2 目标

在不破坏现有已验证功能的前提下，把上述缺口补齐到**可现场演示、能被追问**的程度。

### 1.3 范围（2026-09-15 用户确认扩大）

初版设计（方案 B + LangChain）曾把文档类、压测、云部署、角色扩展列为非目标。
用户随后确认：**这些也要做**，总要求是「快速、完善地改出来，贴近清单」。

因此本轮范围为**全清单覆盖**，组织为五个工作流：

| 工作流 | 内容 |
|---|---|
| **W1 技术链路** | LangChain 双链路、Milvus 双写、多路召回、OCR 分流、RAGAS |
| **W2 数据增强** | 知识库数据增强（摘要 / 父子块）、去重与低质量过滤。**角色保持现有 2 个不扩** |
| **W3 工程化** | pytest 单测、Postman 集合、压测与 QPS、nginx 负载均衡、Redis 补齐 5 种数据类型 |
| **W4 文档** | 需求规格说明书、业务流程图、思维导图、各项对比报告 |
| **W5 部署** | Linux 部署脚本、WSL Ubuntu 实测、云部署方案与文档 |
| **W6 嵌入优化** | 嵌入模型横向对比（3 个）、bge-m3 嵌入微调、微调前后指标对比 |

**明确仍不纳入本轮**（成本过高或平台级产品，经权衡后排除）：

| 项 | 原因 |
|---|---|
| RagFlow / LightRAG / LinearRAG 实机部署 | 平台级产品；LightRAG 的图增强思路已由 W1 的 Neo4j 图谱落地覆盖 |
| 云服务器真实开通 | 需用户云账号与计费；本轮交付**可直接执行的部署脚本 + 在 WSL Ubuntu 上验证过的完整流程**，云上只需替换 IP 与密钥 |

> 这三项如后续要补，均可独立增量进行，不影响已完成的工作流。

---

## 二、总体策略：不替换，做可切换的双链路

现有手写链路已经产出 100% 命中率/引用率的实测结果，是**已验证基线**。直接替换为 LangChain 会把一个已知可用的系统变成未知状态，且一旦改造引入回归，将同时损失功能与已有指标。

因此采用**双链路并存 + 配置切换**：

| 配置 | 链路 | 用途 |
|---|---|---|
| `RAGLORA_CHAIN=langchain`（**默认**） | LCEL 重写版 | 日常运行 |
| `RAGLORA_CHAIN=manual` | 现有手写链路 | 已验证基线 / 出问题时的退路 |

> **2026-09-16 变更**：默认值由 `manual` 改为 `langchain`（用户指定）。
> 依据是实测两条链路的检索与引用指标一致（见 `docs/06-双链路对比.md`），
> 且 `manual` 随时可切回，退路始终保留。

**收益**：可现场切换演示两条链路并讲清「框架省了什么、又藏了什么」，比单纯声称「使用了 LangChain」更有说服力；改造失败不影响现有功能。

---

## 三、详细设计

### 3.1 模块清单

新增文件：

```
backend/app/services/
  lc_chain.py       LangChain 版链路（LCEL 编排）
  lc_retrievers.py  三路 Retriever 封装（向量 / MySQL / Neo4j）
  milvus_store.py   Milvus 客户端（与 qdrant_store 同接口）
  graph_store.py    Neo4j 图谱存储 + 召回
  ocr.py            OCR 分流解析
backend/eval/
  run_ragas.py      RAGAS 评测
backend/app/core/
  config.py         扩展：链路 / 向量库 / 模型开关
docker/
  docker-compose.milvus.yml
  docker-compose.neo4j.yml
tools/demo/
  up.sh / down.sh   按需启停 Milvus / Neo4j
```

### 3.2 ① LangChain 链路（`lc_chain.py`）

**实现方式**：LCEL 编排

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | ChatOllama(model=settings.LLM_MODEL, streaming=True)
    | StrOutputParser()
)
```

**关键取舍（必须在文档与答辩中讲清）**：
`langchain_huggingface.HuggingFaceEmbeddings` 只输出 dense 向量，**bge-m3 的 learned sparse 分支仍由手写代码产出**。因此 LangChain 版是混合架构——dense 走框架、sparse 走手写、RRF 融合走手写、精排走手写。

这**不是缺陷**，而是对框架能力边界的准确认识：LangChain 未覆盖 bge-m3 的稀疏检索能力，强行套用会丢失混合检索能力。

**分块对照**：同时引入 `langchain_text_splitters`，与现有三策略（条文/段落/固定长度）做对照实验，输出分块策略对比结论。

**接口约定**：`lc_chain.ask()` / `lc_chain.ask_stream()` 与现有 `rag_chain` 保持同名同签名，路由层无需感知具体实现，由 `config.py` 的工厂函数选择。

### 3.3 ② Milvus 双写（`milvus_store.py`）

**部署**：Docker standalone（`milvusdb/milvus:v2.6.6`），单容器模式。

**该镜像已在本地 Docker 缓存中（3.55 GB），无需拉取。** Neo4j 同理（`neo4j:latest`，1.06 GB 已在本地）。

**Collection Schema**（按清单要求建全）：

| 字段 | 类型 | 说明 | 对应清单项 |
|---|---|---|---|
| `id` | INT64, `auto_id=True`, `is_primary=True` | 主键 | 主键：唯一性、自增 ID |
| `vector` | FLOAT_VECTOR(dim=1024) | 稠密向量 | 向量 |
| `text` | VARCHAR(8192) | 原文 | 向量对应的原文 |
| `sparse` | SPARSE_FLOAT_VECTOR | 稀疏向量 | 混合检索 BM25 |
| `source` | VARCHAR(512) | 文档来源 | 文档来源 |
| `summary` | VARCHAR(2048) | 摘要 | 摘要 |
| `created_at` | INT64 | 创建时间（Unix 时间戳） | 创建时间 |
| `updated_at` | INT64 | 修改时间（Unix 时间戳） | 修改时间 |

**索引**：`vector` 建 HNSW（COSINE），`sparse` 建 SPARSE_INVERTED_INDEX（IP）。

**数据初始化**：现有 Qdrant 中的 636（medical）+ 14319（legal）条向量**直接导出回灌 Milvus，不重新编码**。

- 理由：重新编码需约 4 分钟 GPU 时间，且结果与现有向量等价
- 导出内容：Qdrant point 的 vector（dense + sparse）与 payload
- 一致性策略：**以 Qdrant 为权威源**，Milvus 可随时由 Qdrant 全量重建

**双写时机**：`POST /api/kb/ingest` 新增文档时双写；存量数据走一次性回灌脚本。

**检索切换**：`RAGLORA_VECTOR_STORE=qdrant|milvus|both`。`both` 模式用于现场对比两库召回结果。

### 3.4 ③ 多路召回（`lc_retrievers.py`）

```
问题 ──┬─→ 路1 向量库（dense + sparse 混合检索）  ──┐
       ├─→ 路2 MySQL（结构化过滤）                 ──┼─→ RRF 融合 ─→ 精排 ─→ 生成
       └─→ 路3 Neo4j（图谱关系召回）              ──┘
```

**路2 — MySQL 结构化召回**：
按角色绑定的文档集合、时间范围、文档元数据（`kb_documents` 表）做过滤召回。适用于「只看 2024 修订版」「只看某一部法律」这类带约束的查询。

**路3 — Neo4j 图谱召回**：
图模式：`(法条/章节) -[依据|引用|修订自]-> (法条/章节)`

**关键决策：用规则抽取图谱，不跑 LLM 抽取。**

- 理由：法条间的「依据《X》第Y条」是强模式，正则抽取可达高准确率，成本近零；LLM 抽取需对 14319 条 chunk 逐条调用，耗时与成本都不划算
- 医疗侧：抽取指南的章节层级（章 → 节 → 条）构建层级图
- 兜底：若规则抽取覆盖率低于预期，再评估是否引入 `neo4j-graphrag` 的 LLM 抽取（作为可选项，不在本轮承诺）

**融合**：现有 RRF（Reciprocal Rank Fusion）从两路扩展到三路，各路权重可配置。

**已知限制（需在文档中写明）**：RRF 融合分是量化值，存在大量并列，**相关性过滤仍必须使用精排分**（沿用现有设计决策 §5.4）。

### 3.5 ④ OCR 分流解析（`ocr.py`）

```
PDF ─→ PyMuPDF 抽取文本
        │
        ├─ 文本量 ≥ 阈值 → 判定为文本层 PDF → 走现有分块链路
        │
        └─ 文本量 <  阈值 → 判定为扫描件
              ├─ MinerU   (D:\anaconda3\envs\mineru, vlm-engine, GPU) ← 高精度
              └─ PaddleOCR (rag_env 已装 paddleocr 3.4.0)           ← 轻量兜底

表格提取 → pdfplumber
```

**MinerU 跨环境调用**：MinerU 位于独立 conda 环境 `mineru`，通过 `subprocess` 调用
`D:\anaconda3\envs\mineru\Scripts\mineru.exe`。可复用父项目 `D:\桌面\RAG\mineru_parse.py` 的成熟写法。

**阈值定义**：按**页均字符数**判定，初始取 **50 字符/页**。低于该值判为扫描件。
该阈值以配置项暴露（`RAGLORA_OCR_TEXT_THRESHOLD`），可用现有 113 页医疗 PDF 与 176 部法律 txt 做回归校准。

**接口变更**：`POST /api/kb/ingest` 新增参数 `ocr=auto|force|off`（默认 `auto`，即按上述阈值自动判定）。

**降级策略**：MinerU 不可用时自动降级到 PaddleOCR；两者都不可用时明确报错，不静默产出空结果。

### 3.6 ⑤ RAGAS 评测（`run_ragas.py`）

**安装**：`pip install ragas pymilvus pdfplumber`（**已于 Task 1 执行**，详见 §5.1）

**⚠️ 硬性调用约定（实测确认，2026-09-15）**

`ragas 0.4.3` 与本环境存在两处不兼容，已通过仓库内垫片 `backend/app/core/ragas_compat.py` 绕过：

1. ragas 在 `ragas/llms/base.py` 模块顶层硬导入 `langchain_community.chat_models.vertexai`，
   而 `langchain-community 0.4.2` 已删除该模块。ragas 从 0.2.15 到 0.4.3 **全部版本**皆如此，降级无效。
2. **更凶险的一处**：本环境 `torch 2.5.1+cu121` 与 `pyarrow 24.0.0` 存在原生 DLL 加载顺序冲突。
   若 `torch` 先于 `pyarrow.dataset` 加载，随后 `import ragas` 会**直接段错误（SIGSEGV，退出码 139）**，
   **不是 Python 异常，无法用 try/except 捕获**。

**实测验证**：

| 顺序 | 结果 |
|---|---|
| `ensure_ragas_compat()` → `import ragas` | ✅ 通过，ragas 0.4.3 |
| `import torch` → `ensure_ragas_compat()` → `import ragas` | ❌ **段错误，退出码 139** |

**因此评测脚本必须遵守**：`ensure_ragas_compat()` 必须是**第一步**，
在**任何会拉起 torch 的导入之前**调用（`app.services.embed`、`app.services.rerank` 都会拉起 torch）。

```python
# backend/eval/run_ragas.py 的正确开头
from app.core.ragas_compat import ensure_ragas_compat
ensure_ragas_compat()          # ← 必须最先，早于任何 app.services 导入
import ragas                    # 此后才安全
```

垫片在检测到该状态时会**主动抛出 `RuntimeError`**（而非仅记一条告警、然后任由进程被段错误杀死）——
调用方因此能在进程死掉之前拿到一个可捕获的明确错误。**但仍不要依赖这个兜底**：
正确做法是让它压根不进入该状态，即把 `ensure_ragas_compat()` 放在最前。

**该垫片在以下任一条件满足时应删除**：ragas 上游改为惰性导入、langchain-community 恢复该模块、
或本项目不再使用 ragas。详见该文件顶部 docstring。

**⚠️ 裁判并发必须调低（2026-09-17 实测踩坑）**

ragas 的 `RunConfig.max_workers` **默认 16**，而本机是单张 8G 显卡跑单个 7B 模型
（Ollama 对单模型默认串行）。16 路并发把请求堆在队列里 → 撞 180s 超时 →
重试 10 次仍失败 → **该题该指标记为 NaN**，且 `raise_exceptions=False` 让它**静默消失**。

**实测后果**（首轮 40 题）：

| 指标 | 首次（并发 16） | 重跑（并发 2） |
|---|---|---|
| `faithfulness` | 0.7596（40/40） | 0.8173（40/40） |
| `answer_relevancy` | 0.7760（40/40） | 0.7666（40/40） |
| **`context_precision`** | **1.0000（仅 3/40）** ❌ | **0.7469（40/40）** ✓ |
| `context_recall` | 0.7667（40/40） | 0.7417（40/40） |

`context_precision` 是唯一暴露的，因为它要对**每个上下文单独裁判**（5 条 = 5 倍调用量），
最先被压垮。**脚本把 3 个样本的均值报成了 1.0000 —— 一个看起来满分、实则无意义的数字。**

**处置**：
1. `max_workers=2`、`timeout=600`（可用 `RAGLORA_RAGAS_WORKERS` 覆盖）
2. **脚本内置守门**：任一指标有效样本 < 80% 时，控制台与报告**双双标记为「不可信」**，
   不再输出一个孤零零的均值

> 这与本项目此前记录的教训同类：**「3/40 样本仍报出一个漂亮数字」和
> 「评测脚本自身缺陷伪装成系统缺陷」是同一类错误 —— 指标的口径与样本量
> 必须与数值一起呈现。**

**数据集**：复用现有 `backend/eval/qa_set.json`（43 题：medical 20 / legal 20 / 拒答 3）

**指标**：`faithfulness`、`answer_relevancy`、`context_precision`、`context_recall`

**模型分工**：
- **被测模型**：`qwen2.5:7b`
- **裁判模型（judge）**：`qwen2.5:7b`

> **2026-09-15 变更**：原计划用 `qwen2.5:3b` 当被测模型以省内存，**已取消**（见 §3.7），
> 改为被测与裁判同为 `qwen2.5:7b`。

**关键说明（必须写进报告）**：同一模型既当被测又当裁判，会引入**自我偏好（self-preference）偏差** ——
模型倾向给自己的输出打高分，RAGAS 分数可能系统性偏高。
因此报告中**不得把 RAGAS 分数当作绝对结论**，必须与自研评测指标对照呈现，并明确标注该局限。

**产出**：`docs/05-RAGAS评测报告.md`，**与现有自研指标并列呈现**，不取代现有评测脚本。

### 3.7 ⑥ 模型规格切换（已取消）

原计划拉取 `qwen2.5:3b` 并产出「3b vs 7b 模型规格影响对比表」。

**2026-09-15 用户决定取消，全程使用 `qwen2.5:7b`。**

理由：现有成绩单（来源命中率 100%、引用标注率 100%、平均延迟 7.1s）由 7b 产出。
切换小模型会使现场演示质量低于报告记载的指标 —— 答辩时展示的反而比写下来的差。

代价与应对：

| 代价 | 应对 |
|---|---|
| 「模型规格影响对比表」交付物取消 | 从 §8.2 交付物清单移除 |
| 多路召回演示时内存吃紧（Milvus 2.5G + Neo4j 1.5G + 7b 4.4G + 模型 3.4G） | **只演示检索链路、不生成**，规避生成模型的内存占用，而非降模型规格 |
| 磁盘不再新增 2GB | — |

---

### 3.8 角色与数据增强（W2）

**角色范围（2026-09-16 用户确认收窄）**：**保持现有的两个角色，不再新增。**

| 角色 | 类型 | 知识库 | 说明 |
|---|---|---|---|
| 心血管内科医生 | 专业 | `kb_medical` | 现有，指标基线 |
| 执业律师 | 专业 | `kb_legal` | 现有，指标基线 |

原计划扩到 6 个（心理医生 / 金融理财师 / NPC / 英语学习助教），**已取消**。
理由：新增专业角色需配套建设独立知识库语料，成本高；
而答辩清单里「多角色」这条**已被现有两个角色满足**（各自拥有独立知识库与独立人格模板）。

> 若答辩时被追问「能否加新角色」，可直接演示现有设计：
> 角色 = 人格三层（身份/风格/约束）+ 知识库绑定，**新增角色零代码改动**，
> 只需在 `seed.py` 加一条数据并上传知识库文档即可。这是现有架构的既有能力，无需为此真的加角色。

**数据增强**（对应清单「知识库数据增强」）：

| 项 | 做法 |
|---|---|
| 摘要 | 每个 chunk 生成摘要写入 payload（Milvus `summary` 字段已预留） |
| 父子块 | 子块用于检索、父块用于生成，提升上下文完整性 |
| 去重 | 对现有 14955 条做相似度去重，清理低质量 chunk |

**验收**：增强前后跑同一套 QA 集，**指标不得下降**；下降则回滚该项增强。

---

### 3.9 嵌入模型对比与微调（W6）

**本地可用嵌入模型**（`D:\桌面\模型\嵌入模型\`）：

| 模型 | 维度 | 用途 |
|---|---|---|
| bge-m3 | 1024 | 当前主力（dense + sparse） |
| m3e-base | 768 | 对比项 |
| bert-base-chinese | 768 | 对比项 |

#### 3.9.1 横向对比（低成本，先做）

用现有 43 题 QA 集，对三个模型分别编码并检索，对比**来源命中率**。bge-m3 是唯一能出 sparse 的，因此对比限定在 **dense 单路**，保证可比性。

产出：嵌入模型对比表（并入 `docs/11-嵌入优化报告.md`）。

#### 3.9.2 bge-m3 嵌入微调（本项主要成本）

**训练库已就绪**：`peft 0.19.1` / `trl 1.3.0` / `sentence-transformers 2.6.1` / `accelerate 1.13.0` / `bitsandbytes 0.49.2`，**无需安装任何新包**。

**⚠️ 关键前提：训练对必须自己造。**

父项目 `D:\桌面\RAG\finetune\data\train.jsonl` 是 **chat 格式的生成微调数据**（`messages` 字段，356 train / 87 val），**不能直接用于嵌入微调**——嵌入需要的是 `(query, positive_passage)` 对。

**训练对构造方案**：

1. 从 medical / legal 语料**分层采样 1200 条 chunk**（按现有 636 : 14319 的比例分配）
2. 用 Ollama `qwen2.5:7b` 为每条 chunk 生成 **2 个"用户会怎么问"的问题** → 得到约 2400 个 `(问题, chunk)` 对
3. **43 题 QA 集严格作为验证集，不参与训练**（避免数据泄漏，否则指标虚高）

**耗时估算**：2400 次 LLM 调用，按每次约 3 秒计 ≈ **2 小时**，后台批跑。这是本项最大的时间开销，期间可并行推进其他工作流。

**训练方式**：

- 目标：只微调 bge-m3 的 **dense 分支**；**sparse 分支保持原模型不变**
  - 理由：sparse 是 learned 词权重，与检索粒度的语义匹配目标不同，一起微调风险大且收益不明
- 损失：`sentence_transformers.losses.MultipleNegativesRankingLoss`（批内负样本）
- 参数：LoRA（`peft`），r=16 / alpha=32，与父项目保持一致
- 设备：GPU（8G 显存下 bge-m3 微调需 fp16 + 小 batch）

**评估与回滚**：

- 重编码：微调后全量重编码 14955 条写入**新 collection**（`kb_medical_ft` / `kb_legal_ft`），**不覆盖原 collection**
  - 编码耗时约 1 分钟（GPU 234 chunks/s）
- 对比：微调前后跑同一套 43 题，对比来源命中率与精排分
- **回滚条件**：若微调后指标下降，保留原 collection，微调版仅作为实验结论呈现
  - 「微调后指标反而降了」本身也是有价值的答辩结论（小数据集过拟合），**不视为失败**

**已知风险**：仅 2400 个训练对，对 bge-m3 这种 5.7 亿参数模型而言数据量偏小，过拟合风险实质存在。这一点会在报告中明确说明，不粉饰。

---

## 四、内存编排

### 4.1 实测硬件

- 总内存 15.7 GB，Docker 启动前可用 5.0 GB
- D 盘剩余 149 GB，C 盘剩余 30 GB（Docker 镜像默认落 C 盘）
- GPU：RTX 4060 Laptop，8 GB 显存

### 4.2 组合与内存账

全程使用 `qwen2.5:7b`（4.4 GB，见 §3.7）：

| 组合 | 估算内存 | 用途 |
|---|---|---|
| Qdrant 嵌入式 + qwen2.5:7b + bge-m3 + 精排 | ~8.8 GB | **常驻**，日常开发与主演示 |
| ＋ Milvus standalone | ~11.3 GB | 演示双写时临时开启 |
| ＋ Milvus + Neo4j（含生成） | ~12.8 GB | ⚠️ 逼近上限（15.7 GB），且需关掉浏览器多开 |
| ＋ Milvus + Neo4j（**只检索不生成**） | ~8.4 GB | ✅ 演示多路召回的推荐方式 |

**多路召回演示的推荐姿势**：用检索调试台（`/api/search`）展示三路召回与融合结果，
**不触发大模型生成** —— 此时 Ollama 的 4.4 GB 不占用，内存回到安全区。
这比「换个 3b 小模型」更划算：既保住演示质量，又保住内存。

### 4.3 本地已有 Docker 镜像（实测清点）

Docker 本地缓存中已存在下列镜像，**本轮设计所需的基础设施无需重新拉取**：

| 镜像 | 大小 | 用途 |
|---|---|---|
| `milvusdb/milvus:v2.6.6` | 3.55 GB | 本轮向量库 |
| `neo4j:latest` | 1.06 GB | 本轮图谱库 |
| `nginx:latest` | 237 MB | （未在本轮范围，但负载均衡可用） |
| `langgenius/dify-api:1.9.1` + `dify-web:1.9.1` + `dify-sandbox` + `dify-plugin-daemon` | ~6 GB | （未在本轮范围，见 §9） |
| `postgres:15-alpine`、`redis:latest`、`apache/kafka`、`weaviate:1.19.0` | — | 备用 |

> 这意味着本轮的「Docker 拉取失败」风险实际不存在——组件都已经在本地了。

### 4.4 编排方式

`tools/demo/up.sh <milvus|neo4j|all>` 按需拉起，`tools/demo/down.sh` 停止。
主链路（Qdrant 嵌入式）不依赖 Docker，Docker 未启动时系统仍可完整运行。

**设计原则**：Docker 组件是**可选增强**，不是运行前提。这样答辩现场即使 Docker 出问题，主演示仍可继续。

---

## 五、技术风险与应对

### 5.1 依赖安装风险（已探明）

`pip install --dry-run ragas pymilvus pdfplumber` 结果：

- **新增** 11 个包：appdirs 1.4.4、cffi 2.1.1、cryptography 50.0.1、docstring_parser 0.18.0、instructor 1.17.0、pdfminer.six、pdfplumber 0.11.10、pycparser 3.0、pymilvus 3.0.1、ragas 0.4.3、scikit-network 0.33.5
- **变更** 4 个已装包，均为向后兼容的小版本：

| 包 | 现在 | 变为 |
|---|---|---|
| requests | 2.32.5 | 2.34.2 |
| click | 8.1.8 | 8.5.0 |
| idna | 3.11 | 3.19 |
| pypdfium2 | 5.6.0 | 5.13.0 |

**无降级，无版本链断裂。** 安装前执行 `pip freeze > requirements-lock-before.txt` 保留回滚依据。

### 5.2 风险清单

| 风险 | 影响 | 应对 |
|---|---|---|
| pymilvus 3.0.1 与 Milvus server 版本不匹配 | 客户端连不上服务端 | **风险已显著降低**：本地镜像为 `milvusdb/milvus:v2.6.6`，`pymilvus 3.x` 客户端对 2.6.x 服务端属匹配代际。仍需在 S0 做一次真实连接验证；若不兼容，降 pymilvus 版本即可，不影响设计 |
| Milvus 镜像大 / 内存吃紧 | 演示时 OOM | 后台预拉镜像；按需启停；主链路不依赖 Docker |
| Neo4j 图谱构建成本高 | 工期不可控 | 规则抽取替代 LLM 抽取 |
| RAGAS judge 模型不稳 | 指标可信度低 | 7b 当裁判；报告中标注局限 |
| 双写不一致 | 两库数据漂移 | 以 Qdrant 为权威源，Milvus 可全量重建 |
| LangChain 改造引入回归 | 破坏现有功能 | 双链路并存，`manual` 随时可切回；改造前后跑同一套 QA 集 |
| MinerU 跨环境调用失败 | OCR 链路不可用 | 降级到 PaddleOCR；两者皆不可用时明确报错 |
| 磁盘占用（C 盘仅剩 30G） | 镜像拉取失败 | 关注镜像体积；必要时迁移 Docker 数据目录到 D 盘 |

---

## 六、测试与验收

### 6.1 验收标准

| 项 | 验收方式 |
|---|---|
| LangChain 链路 | `RAGLORA_CHAIN=langchain` 下完成一轮完整对话，输出与 manual 版对比 |
| Milvus 双写 | 回灌后 Milvus 向量数与 Qdrant 一致（636 + 14319）；同一查询两库召回结果可对比 |
| 多路召回 | 三路各自单独可查；融合结果优于任一路单独结果 |
| OCR 分流 | 文本层 PDF 走 PyMuPDF、扫描件走 OCR，路由判定可验证 |
| RAGAS | 43 题跑出 4 个指标，报告落盘 |
| RAGAS 与自研指标并列 | 43 题同时产出两套指标，报告中对照呈现 |
| 健康检查 | `/api/health?deep=1` 扩展覆盖新组件 |

### 6.2 回归保护

每项改造完成后，跑**同一套 43 题 QA 集**，确保来源命中率、引用标注率不低于改造前基线（100% / 100%）。

### 6.3 单元测试

为核心新模块编写 pytest 用例（`rag_env` 已装 pytest 9.1.1），覆盖：
- OCR 文本量阈值判定逻辑
- 三路 RRF 融合的权重与排序正确性
- Milvus schema 建表与读写往返

---

## 七、建议实施顺序

本设计覆盖六个子系统，规模较大，需分批实施、每批独立可验收。顺序按「依赖关系 + 风险前置」排列：

本设计覆盖五个工作流，需分批实施、每批独立可验收。按「依赖关系 + 风险前置」排列：

| 阶段 | 工作流 | 内容 | 依赖 | 完成标志 |
|---|---|---|---|---|
| **S0** | — | 环境准备：`pip freeze` 备份 → 装依赖 → 确认 qwen2.5:7b 就绪 → 写 compose 并验证 Milvus/Neo4j 真实连通 | — | 三个包可导入；7b 就绪；Milvus 连通 |
| **S1** | W1 | LangChain 双链路（`lc_chain.py` + 配置开关） | S0 | `RAGLORA_CHAIN=langchain` 跑通一轮对话 |
| **S2** | W1 | Milvus 接入 + 存量向量回灌 + 双写 | S0 | 两库向量数一致；对比查询可跑 |
| **S3** | W1 | 多路召回（MySQL 路 + Neo4j 路 + 三路 RRF） | S2 | 三路各自可查，融合结果可解释 |
| **S4** | W1 | OCR 分流解析链 | S0 | 扫描件走 OCR 成功入库 |
| **S5** | W1 | RAGAS 评测（单一 7b 模型） | S1–S4 | 报告落盘 |
| **S6** | W2 | 数据增强（摘要 / 父子块 / 去重）。**角色不扩，保持现有 2 个** | S3 | 增强后指标不降 |
| **S7** | W3 | pytest 单测 + Postman 集合 + Redis 补齐 5 种数据类型 | S1–S6 | 用例通过；集合可导入执行 |
| **S8** | W3 | nginx 负载均衡 + 多 worker + 压测出 QPS | S2, S7 | 压测报告落盘 |
| **S9** | W5 | Linux 部署脚本 + WSL Ubuntu 实测 + 云部署方案 | S7 | WSL 上完整跑通 |
| **S10** | W4 | 需求规格说明书 + 业务流程图 + 思维导图 | S1–S8 | 文档落盘 |
| **S11** | W6 | 嵌入模型横向对比（bge-m3 / m3e-base / bert-base-chinese，dense 单路） | S0 | 对比表落盘 |
| **S12** | W6 | bge-m3 嵌入微调 + 重编码到新 collection + 前后指标对比 | S11 | 微调版可对比；**原 collection 未被改动** |
| **S13** | — | 回归验证：43 题 QA 集 | 全部 | 命中率/引用率不低于基线 |

**⚠️ 并行任务（关键路径优化）**：S12 的瓶颈是**训练对生成**（约 2400 次 LLM 调用 ≈ 2 小时）。
该步骤**在 S0 完成后立即后台启动**，与 S1–S10 并行推进，不阻塞主线。
否则整个项目会被 2 小时的串行等待拖住。

### 7.1 排序理由

- **S1 风险前置**：LangChain 对现有代码侵入性最高、最可能暴露回归。早做早发现，且双链路开关就位后，后续所有改造都在开关保护下进行。
- **S8 依赖 S2**：Qdrant 嵌入式持有独占文件锁，后端只能单 worker——**必须先换 Milvus 才能解除这一限制**，nginx 负载均衡与多 worker 才有意义。这是 S2 的下游收益。
- **S10 文档殿后**：文档需引用 S1–S8 的实测数据与截图，提前写会反复返工。
- **S6 排在 S3 后**：数据增强的效果要用 S5 的指标验证。
- **S11 只依赖 S0**：嵌入模型对比不需要任何前置改造，S0 一完成就能做，能最早消掉一条清单项。
- **微调排在最后且可回滚**：微调是唯一「可能让指标变差」的改造，放在末尾保证前面所有成果已锁定；且写入新 collection，失败不影响主线。

### 7.2 风险隔离

- **Milvus 不是单点**：若 S2 因 pymilvus 兼容性问题受阻，S3 的路 1 退回 Qdrant，**路 2（MySQL）/ 路 3（Neo4j）不受影响**，多路召回的设计依然成立。仅 S8 的「多 worker」收益会推迟。
- **文档类工作零技术风险**：W4 不依赖任何代码改动，任何阶段都可插入。
- **每个 S 独立可交付**：任一批次完成后系统都处于可运行、可演示状态。

---

## 八、交付物

### 8.1 代码

见 §3.1 模块清单，另含 W2/W3/W5 的新增脚本与部署脚本。

### 8.2 文档

| 交付物 | 路径 | 工作流 |
|---|---|---|
| 设计文档 | 本文件 | — |
| 实施计划 | `docs/superpowers/plans/` | — |
| **需求规格说明书**（功能/业务流程/流程图/业务规则） | `docs/08-需求规格说明书.md` | W4 |
| **思维导图** | `docs/09-思维导图.md`（+ 可导入 Xmind 的源文件） | W4 |
| RAGAS 评测报告 | `docs/05-RAGAS评测报告.md` | W1 |
| 双链路对比说明（manual vs LangChain） | `docs/06-双链路对比.md` | W1 |
| 自研 vs Dify 对比 | `docs/07-自研vsDify对比.md` | W1 |
| 多路召回设计说明 | 并入本设计文档 §3.4 | W1 |
| 角色与知识库说明 | 并入需求规格说明书 | W2 |
| **嵌入优化报告**（3 模型对比 + 微调前后） | `docs/11-嵌入优化报告.md` | W6 |
| **压测报告（QPS / 并发 / 负载均衡）** | `docs/10-压测报告.md` | W3 |
| **部署文档（Linux/云）** | 更新 `docs/04-部署文档.md`，补 Linux 实测章节 | W5 |
| **接口文档** | 更新 `docs/03-接口文档.md`（新增参数与端点） | W1 |

### 8.3 脚本

| 交付物 | 路径 |
|---|---|
| Milvus / Neo4j 按需启停 | `tools/demo/up.sh` / `down.sh` |
| Docker compose | `docker/docker-compose.milvus.yml` / `neo4j.yml` / `dify.yml` |
| Linux 部署脚本 | `deploy/install.sh`（检查配置 → 装环境 → 拉起依赖 → 建库 → 导数据） |
| Postman 集合 | `docs/postman/RAGLoRA.postman_collection.json` |

---

## 九、Dify 对比演示（纳入本轮）

镜像清点发现 Dify 全套镜像已在本地（api / web / sandbox / plugin-daemon，约 6 GB），
把该项成本从「重写主干」降到「起一份 compose」。**纳入本轮**，作为 W1 的收尾演示。

**定位**：不做功能对齐，只做**平台级产品 vs 自研系统**的对比演示。

**执行方式**（S5 之后）：
1. 起 Dify compose（`langgenius/dify-api:1.9.1` + `dify-web:1.9.1` + 依赖的 postgres/redis，均在本地）
2. 让 Dify 接入**同一份本地 Milvus 数据**（S2 已建好）
3. 同一批问题分别问自研系统与 Dify，对比回答、引用、检索链路可解释性

**产出**：`docs/07-自研vsDify对比.md`

**预期结论方向**（答辩时的讲法）：Dify 在编排速度、可视化配置上占优；自研系统在**混合检索（sparse 分支）、精排策略、链路可解释性、指标可测性**上更可控——这正好印证「为什么自研而不是直接用平台」。

> 内存提示：Dify 全套约需 2–3 GB。演示时先停 Milvus 之外的重组件，或按 §4.4 的按需启停脚本逐个开。

---

---

## 十、确认记录

1. 双链路并存（非直接替换）— 用户认可
2. 图谱用规则抽取（不跑 LLM 抽取）— 用户认可
3. RAGAS 与自研指标并列（不删旧）— 用户认可
4. 范围扩大：需求规格说明书 / 思维导图 / 压测 / 云部署 **均纳入本轮** — 用户 2026-09-15 确认
   （**角色扩展除外**，见第 9 条）
5. 总要求：**快速、完善地改出来，贴近清单** — 用户 2026-09-15 确认
6. **嵌入微调纳入本轮**（W6）：做 bge-m3 嵌入微调，写新 collection，不覆盖原库 — 用户 2026-09-15 确认
7. 本地模型路径确认：`D:\桌面\模型\嵌入模型\`（bge-m3 / m3e-base / bert-base-chinese）、`D:\桌面\模型\精排模型\bge-reranker-v2-m3`、微调产物在 `D:\桌面\RAG\finetune\out\` — 用户 2026-09-15 提供

8. **取消 3b 模型，全程使用 qwen2.5:7b** — 用户 2026-09-15 确认。
   理由：现有 100% 指标由 7b 产出，换小模型会让现场演示低于报告记载。
   多路召回内存压力改用「只检索不生成」规避。详见 §3.7。

9. **角色不扩展，保持现有 2 个（心血管内科医生 / 执业律师）** — 用户 2026-09-16 确认。
   取消原计划的心理医生 / 金融理财师 / NPC / 英语学习助教 4 个角色。
   理由：新增专业角色需配套建设独立知识库语料，成本高；而清单「多角色」这条已由现有两角色满足。

**无未决事项。**

无。设计已与用户逐项确认：

1. 双链路并存（非直接替换）— 用户认可
2. 图谱用规则抽取（不跑 LLM 抽取）— 用户认可
3. RAGAS 与自研指标并列（不删旧）— 用户认可
