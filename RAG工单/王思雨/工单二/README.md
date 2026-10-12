# 基于 PDF 文档的问答系统（优化版）

**工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化**
**对应工单**：《人工智能NLP-RAG项目-02-基于PDF文档的问答系统的优化任务工单V1.1-20250206》
**知识库**：`data/招股说明书1.pdf`（武汉兴图新科电子股份有限公司招股意向书，548 页）

从 **PDF 解析处理、分块、检索** 三个层面优化检索准确率，并对同一批工单问题做优化前后的实测对比。

**实测结果（工单 10 题，`results/eval_20261007_112004.json`）**

| 验收项（工单原文） | 目标 | 优化前 | 优化后 | 结论 |
|---|---|---|---|---|
| 回答准确率 | ≥ 90% | 60%（6/10） | **100%（10/10）** | ✅ |
| 检索上下文命中率 | — | 70% | **100%** | ✅ |
| 出处页命中率 | — | 30% | **100%** | ✅ |
| 响应时间（提问→答案） | ≤ 3 秒 | 平均 1.769 s | 平均 2.455 s，9/10 题 ≤3 s | ⚠️ 见下 |
| 中文与英文问答 | 均支持 | — | 英文提问英文作答 3/3 | ✅ |

RAGAS 四项（优化前后）：faithfulness 0.6667→**1.0000**、answer_relevancy 0.6167→**0.8585**、
context_recall 0.7000→**1.0000**、context_precision 0.8667→**0.9867**。

> ⚠️ **响应时间的如实说明**：本项目可控部分（检索+重排）稳定在 **0.76～1.07 s**；
> 端到端长尾由在线 DeepSeek API 的响应波动决定——同一环境 30 分钟内实测生成耗时从 1.4 s 涨到 3.2 s。
> 正式评测中 10 题有 9 题 ≤3 s，唯一超出的 3.043 s 超了 0.04 s。
> 高频问题走 Redis 缓存后实测 **0.00 s**。完整数据与对照测量见 `docs/03-测试.md` §6。

---

## 1. 文件清单

### 源码（Python，含工单编号注释，每行代码均有中文注释）

| 文件 | 职责 |
|---|---|
| `src/config.py` | 从 `.env` 与环境变量集中读取参数，密钥不落代码 |
| `src/pdf_parser.py` | MinerU 主解析；PyMuPDF / pdfplumber / PaddleOCR 三级降级 |
| `src/retrieval.py` | bge-m3 编码、Milvus 读写、BM25、加权融合、bge-reranker-v2-m3 重排 |
| `src/rag_pipeline.py` | 语义分块、索引构建、Redis 缓存、DeepSeek 生成、问答主流程与容错 |
| `src/main.py` | FastAPI 接口：`/api/ask`、`/api/compare`、`/api/health` |
| `src/evaluate.py` | 10 题评测 + RAGAS 四项指标 + 优化前后汇总 |

### 前端（Vue 3 + Vite，不计入 Python 行数）

| 文件 | 职责 |
|---|---|
| `frontend/src/App.vue` | 问答界面：提问、中英切换、答案与耗时、优化前后对比、命中片段 |
| `frontend/vite.config.js` | 把 `/api` 代理到 `http://localhost:8000` |
| `frontend/package.json` / `index.html` / `src/main.js` | 工程与入口 |

### 配置与数据

| 文件 | 说明 |
|---|---|
| `requirements.txt` | Python 依赖 |
| `.env.example` | 配置模板（复制为 `.env` 后填密钥） |
| `docker-compose.yml` | Milvus（etcd + minio + milvus）与 Redis |
| `data/招股说明书1.pdf` | 知识库 |
| `data/eval/questions.json` | 工单 10 题 + 3 道英文题（含标准答案、关键字段、出处页码） |
| `data/parsed/` | MinerU 解析产物（自动生成） |
| `data/index.json`、`data/bm25.pkl` | 两套分块结果与 BM25 索引（自动生成） |
| `results/` | 评测结果 JSON（自动生成） |

### 文档

| 文件 | 内容 |
|---|---|
| `docs/01-设计.md` | 优化前后架构对比、解析/分块/检索/重排/缓存/3 秒/高可用/验收方案 |
| `docs/03-测试.md` | 工单 10 题逐题结果、RAGAS 四项指标、响应时间、并发与异常场景 |
| `docs/04-优化.md` | 调参过程、优化前后检索精确度对比表、逐题答案对比、优化前后代码结构变化 |
| `docs/05-部署.md` | docker-compose、后端/前端启动、`.env` 说明、高可用与资源控制、排障 |
| `docs/演示视频脚本.md` | 分幕演示脚本（启动→解析→中英提问→逐题检索→前后对比→耗时→异常） |

---

## 2. 运行命令

```bash
cd "d:/Users/雨子/Desktop/专高六/liu-专高六/RAG/工单二"

# ① 安装依赖
pip install -r requirements.txt

# ② 配置密钥：复制模板后填入 DEEPSEEK_API_KEY
copy .env.example .env

# ③ 启动 Milvus 与 Redis（本机若已有同端口服务，跳过对应服务）
docker compose up -d

# ④ 构建索引（首次会调 MinerU 解析 548 页，约 4~5 分钟；再次运行复用解析产物）
python -m src.rag_pipeline

# ⑤ 启动后端
uvicorn src.main:app --host 0.0.0.0 --port 8000

# ⑥ 启动前端（另开终端）
cd frontend && npm install && npm run dev     # http://localhost:5173
```

**冒烟测试**

> ⚠️ **不要把中文写在 `-d '...'` 里。** 在 Windows 的 Git Bash 中，命令行会把中文参数按本地代码页编码，
> 服务端收到乱码问题后**会正常地返回「资料中未找到相关信息」**——看起来像检索坏了，实际是调用方式的问题。
> 正确做法：先把 JSON 写成 UTF-8 文件，再用 `--data-binary @文件` 发送。

```bash
# 1) 用 Python 写请求体，确保是 UTF-8
python -c "import pathlib;pathlib.Path('q1.json').write_text('{\"question\":\"武汉兴图新科电子股份有限公司注册资本是多少？\"}', encoding='utf-8')"
curl -X POST http://localhost:8000/api/ask -H "Content-Type: application/json" --data-binary @q1.json

# 2) 前后对比同理
python -c "import pathlib;pathlib.Path('q2.json').write_text('{\"question\":\"报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？\"}', encoding='utf-8')"
curl -X POST http://localhost:8000/api/compare -H "Content-Type: application/json" --data-binary @q2.json
```

> 更省事的办法：直接用浏览器界面（`npm run dev` → <http://localhost:5173>）。
> 界面走的是 Vue 发出的 UTF-8 请求，不存在这个问题，也是演示时推荐的方式。

接口文档：<http://localhost:8000/docs>

---

## 3. 如何配置 DeepSeek API Key

密钥**只从环境变量读取，代码里没有任何硬编码**。

方式一（推荐）：编辑项目根目录的 `.env`

```
DEEPSEEK_API_KEY=sk-你的密钥
```

方式二：设为系统环境变量，**优先级高于 `.env`**

```bash
export DEEPSEEK_API_KEY=sk-你的密钥        # Git Bash / Linux
set DEEPSEEK_API_KEY=sk-你的密钥           # Windows cmd
```

未配置时提问会返回明确提示，而不是抛堆栈：

```
未配置 DEEPSEEK_API_KEY，请参考 .env.example 配置后重试
```

> `.env` 已在 `.gitignore` 中，不会进版本库。

---

## 4. 如何执行测试与 RAGAS 评估

```bash
python -m src.evaluate
```

脚本会依次执行：

1. **预热**一次问答链路（首次调用含模型加载，不计入 3 秒验收），保证测的是稳态耗时；
2. 依次跑 `baseline`（优化前）与 `optimized`（优化后）两种模式，各 10 道工单问题；
3. 每题**关闭缓存**，测真实端到端耗时；
4. 输出两项自动指标：
   - **回答准确率** = 答案包含该题全部关键字段的题数 / 10；
   - **检索精确度** = ①上下文已含答案关键字段的比例、②上下文覆盖标准答案出处页的比例；
5. 输出 **RAGAS 四项指标**：`faithfulness`、`answer_relevancy`、
   `context_precision`（`LLMContextPrecisionWithReference`）、`context_recall`（`LLMContextRecall`），
   判官模型用 DeepSeek，嵌入模型用本地 bge-m3；
6. 完整结果落盘：`results/eval_<时间戳>.json`（含逐题答案、命中页码、分阶段耗时、RAGAS 明细）。

耗时参考：本机 20 个样本的 RAGAS 评估约 12～15 分钟（判官为在线 API，受网络影响）。

---

## 5. 如何查看优化前后对比结果

| 方式 | 位置 |
|---|---|
| **界面并排对比** | 启动前后端 → 输入问题 → 点「优化前后对比」，左右卡片分别是优化前与优化后的答案、耗时、命中片段 |
| **逐题答案对比表** | `docs/04-优化.md` —— 10 道题逐条列出优化前答案、优化后答案、是否命中出处页 |
| **检索精确度对比表** | `docs/04-优化.md` —— 准确率、上下文命中率、出处页命中率、响应时间、RAGAS 四项指标的优化前后对照 |
| **原始评测数据** | `results/eval_<时间戳>.json` |
| **命令行单题对比** | `curl -X POST .../api/compare -H "Content-Type: application/json" --data-binary @q.json`（中文请求体走文件，原因见 §2 冒烟测试的提示） |

---

## 6. 实现要点（一页速览）

- **解析**：MinerU 版面还原 + 表格识别为主；表格内容在 `table_body`（HTML），拼接**最近章节标题 + 表题**补足检索语境；
  失败降级 PyMuPDF/pdfplumber，文本层为空再降级 PaddleOCR；中文用户名导致 fasttext 打不开模型的问题用 `MINERU_LANGFIX_DIR` 补丁解决。
- **分块**：基于 bge-m3 的语义边界切分（相邻单元相似度低于阈值即切块）+ 长度上下限兜底 + 标题作块首语境 + 表格独立成块 + 过短块向后合并。
- **检索**：bge-m3 向量（Milvus，强一致读）+ BM25（jieba 分词）双路召回，各自归一化后加权融合。
- **重排**：bge-reranker-v2-m3 交叉编码精排，取 Top-5 作为上下文。
- **缓存**：Redis 缓存 `rag:{mode}:{md5(问题)}`，TTL 一天，写入时剔除耗时的变量值；Redis 不可用时静默降级。
- **生成**：DeepSeek（OpenAI 兼容接口），按提问语言作答，要求数字/名称与资料完全一致，无依据时明确拒答。
- **容错**：空输入、超长输入、PDF 缺失、解析失败、索引未构建、Milvus/Redis/API 异常均返回中文友好提示。

---

## 7. 行数与工单约束说明

**统计口径**：有效行 = 非空且非纯注释的物理行（行尾注释不重复计数）。

| 文件 | 有效行 | 对应工单要求 |
|---|---|---|
| `src/config.py` | 32 | 密钥不落代码、参数可调 |
| `src/pdf_parser.py` | 83 | PDF 解析优化 + 解析失败容错 |
| `src/retrieval.py` | 75 | 分块/检索/重排优化（向量 + BM25 + 融合 + 重排） |
| `src/rag_pipeline.py` | 138 | 语义分块、缓存、生成、主流程与容错 |
| `src/main.py` | 36 | 交互接口、中英文问答 |
| `src/evaluate.py` | 68 | 准确率对比、RAGAS 评估 |
| `src/__init__.py` | 1 | 包声明 |
| **合计** | **433** | — |
| `frontend/`（不计入） | 145 | Vue 3 问答界面 |

**为什么是 433 行**：本次交付的形式要求是「Python 总有效行数不超过 300 行，超出时优先保留核心链路、
删除非必要代码、把非核心内容写进文档」。已按此执行——删掉了工单未要求的建索引接口，
非核心内容（探针细节、扩展方式、调参过程）全部移入文档，**每个模块都在 138 行以内**。
但总行数由工单要求的模块数量决定：解析降级链、语义分块、混合检索、交叉重排、Redis 缓存、
RAGAS 评估、前后对比双链路、FastAPI 层、全链路容错，共 8 类能力，删任何一项都会缺失一条验收项。
（工单 PDF 本身没有行数条款，其备注只要求注释含工单编号；300 行是本次交付的形式约束。）

**若必须严格 ≤300 行**，裁剪顺序与代价见 `docs/01-设计.md` §11.3：
删 PaddleOCR 兜底（−16）、删 RAGAS 模块（−16）、接口合并（−14）、删优化前基线（−30）后为 357 行，
再靠紧凑写法压 57 行才能到 300——第 4 项会直接删掉工单要求的「优化前后对比」。

> **结论（数学事实）**：四项裁剪全执行后为 **357 行，仍高于 300**；剩余 57 行只能靠「一行多语句」压缩，
> 与工单「注释需说明每行作用」的要求直接冲突。**300 行与工单验收项二者不可兼得**，
> 需在「裁剪项」与「行数」之间择一。当前权衡是**优先保住验收项**：
> 实测准确率 100%、检索命中率 100%、RAGAS 四项 0.859～1.000。
