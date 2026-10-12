# 第二步 · 研发

> 工单编号：**人工智能NLP-RAG-功能测试及评估**
> 代码位置：`app/`（核心链路）、`app/main.py`（FastAPI 路由）、`frontend/index.html`（Vue 演示）、`tests/`、`scripts/`

---

## 1. 行数口径（对应工单「Python 代码 ≤ 300 行，每行带详细中文注释」）

| 文件 | 物理行 | 是否计入 300 行 |
|---|---|---|
| `app/config.py` | 36 | ✅ |
| `app/ingest.py` | 99 | ✅ |
| `app/rag_core.py` | 97 | ✅ |
| `app/evaluate.py` | 68 | ✅ |
| **核心链路小计** | **300** | **含全部注释行** |
| `app/main.py` | 69 | ❌ 工单另行要求的 FastAPI 路由层，与 Vue 同属演示层 |
| `app/__init__.py` | 2 | ❌ 包标记 |
| `frontend/index.html` | — | ❌ 按工单不计（Vue） |
| `tests/`、`scripts/` | — | ❌ 测试与启动入口，非交付链路 |

**注释策略**：全部采用**行尾注释**，即「一行代码 + 一行注释写在同一条物理行上」。这样 300 行内同时装下了代码与注释，无需为注释额外占行——这是能满足「≤300 行」且「每行都有详细中文注释」的唯一可行写法。

**为守住 300 行而放弃的非核心功能**（均已在代码注释中标注）：

| 放弃项 | 原因 | 替代 |
|---|---|---|
| 关键词判分器（expect/expect_not 自动判分） | 每题需额外维护期望词表，约占 20 行 | 改用 RAGAS 四指标 + 第四步逐题人工复核 |
| 多轮会话历史 | 本工单是单轮功能测试，不需要 | — |
| 反馈票箱 / 自适应重排 | 属 06 工单范畴 | — |
| PDF 表格结构化重建、图像解析 | 属 03/04 工单范畴 | 表格以 MinerU `table_body` / 文本行形式进入分块 |

---

## 2. 各模块实现要点

### 2.1 `app/config.py`（36 行）—— 唯一配置源

- `load_dotenv(override=False)`：真实环境变量优先，`.env` 只补缺失项（便于 CI 注入）。
- 所有路径**正斜杠**，值统一走 `_e()` 取参口并 `.strip()`，避免行尾空格污染路径。
- `CODE2NAME`：股票代码 → 公司名。**这不是装饰，是必需**：语料文件名中的中文已被 `U+FFFD` 破坏（见 `docs/01-design.md` §1）。
- `MILVUS_COLLECTION` 默认 `gd7_chunks`，与工单六的 `gd6_chunks` 物理隔离。

### 2.2 `app/ingest.py`（99 行）—— 解析与分块

**四级解析降级链**（工单要求 MinerU 为主 + 备用方案）：

```
parse_pdf(pdf)
  ├─ ① parse_mineru   MinerU 官方 CLI（子进程）→ 版面/表格/图内文字
  ├─ ② parse_pymupdf  pymupdf 逐页取文本层（本语料文本层完好，快且稳）
  └─ ③ parse_txt      附件自带的同名 .txt（最终兜底）
```

要点：
- MinerU 以 `subprocess.run` 调用，`timeout=7200`；环境须同时设 `MINERU_LANGFIX_DIR` **和**把该目录前置到 `PYTHONPATH`（只设变量无效，`sitecustomize.py` 不会加载）。
- MinerU 产物目录布局新旧不一（`<out>/<名>/<名>/auto/` 与 `<out>/<名>/hybrid_auto/`），用 `rglob("*content_list.json")` 找，**不写死路径**。
- 表格内容在 `table_body` 字段而非 `text`，需单独取。
- 每一级失败都 `print` 失败类型后继续降级，**不静默**；三级全失败才抛错。
- `MINERU_ENABLED=0` 可临时关闭主路径（本单批量建索引时使用，理由见 `CLAUDE.md` §4）。

**语义分块 `sem_chunk(text, encode)`**：
1. 按句末标点断句（超长句按 200 字符硬切，应对表格行）；
2. **批量**编码句子（逐句编码 7 万句会慢一个数量级）；
3. 逐句累积，与「当前块语义中心」（块内句向量的均值）算余弦相似度；
4. `相似度 < SEM_THRESHOLD(0.62)` **且** 块已超过 `CHUNK_SIZE(512)` → 切块，新块以尾部 `CHUNK_OVERLAP(80)` 字符开头，防答案被切断；
5. 超过 `SEM_MAX_CHUNK(900)` 强制切；丢弃 < 30 字符的噪声块。

### 2.3 `app/rag_core.py`（97 行）—— 混合检索 + 生成 + 缓存

**融合为什么先归一化**：向量通道分数量纲（COSINE，0~1）与 BM25 分数量纲（无上界，实测可达 40+）不相交。若直接把权重乘在原始分上，等于只有向量通道生效、BM25 被架空。因此 `fuse()` 先对**每个通道内**做 min-max 归一化，再加权相加；两通道都命中的片段来源标记成 `vec+bm25`，分数叠加使其自然上浮。

**重排**：`bge-reranker-v2-m3`（FlagReranker，fp16），输入按 `RERANK_MAX_CHARS=512` 截断——不截断会按最长序列 padding，30 条候选打分从 0.6 秒拖到 2 秒以上。

**Redis 缓存键带结构指纹**：`gd7:ans:{源码指纹16位}:{参数哈希32位}`。源码指纹 = `rag_core.py` + `config.py` 的字节哈希 + BM25 索引 mtime。**改动代码或重建索引会自动让旧缓存失效**，避免「新字段被旧缓存静默吞掉」（历史踩坑：`degraded` 字段被旧缓存吞掉 → 降级无提示）。

**客户端与模型全部单例**（`_M` 字典）：每请求新建 OpenAI 客户端会多付约 1 秒 TLS 握手（实测 4.3s → 0.7s）；bge-m3 / reranker 同理，加载一次约 10 秒。

**生成约束**：system prompt 强制「只能依据检索片段作答，句末用【来源：公司 第X页】标注，片段里没有就回答『年报中未提及』」；温度 0 保证评测可复现。

### 2.4 `app/evaluate.py`（68 行）—— 评估

三条纪律（写在文件头注释里）：
1. **先落盘**：`collect()` 作答阶段就写 `results/raw_*.json`，RAGAS 阶段失败不丢数据；
2. **逐指标串行**：`for name, m in plan: evaluate(ds, metrics=[m], ...)`，绝不把四个指标塞进一次 `evaluate()`（并发翻 4 倍 → 限流 → 样本静默变 `nan`）；
3. **nan 记 None**：`None if v != v else round(float(v), 4)`，并单独上报 `metric_valid`（如 `9/10`）。

`collect()` 调 `answer(..., use_cache=False)`：**评估必须关缓存**，否则第二轮全命中缓存，测不出链路真实表现。

### 2.5 `app/main.py`（69 行）—— FastAPI 路由

| 路由 | 方法 | 用途 |
|---|---|---|
| `/health` | GET | Milvus / Redis / LLM / 集合块数 |
| `/questions` | GET | 10 题题面 + ground_truth |
| `/retrieve` | POST | 只跑检索链路（演示「检索结果」这一步） |
| `/ask` | POST | 检索 + 生成完整链路 |
| `/results` | GET | 历次评估文件列表 |
| `/results/latest` | GET | 最近一次评估完整结果（前端主页数据源） |
| `/` | — | `StaticFiles` 挂载 Vue 前端 |

只做分发，不含算法；异常一律转成可读的 HTTP 500 信息。

### 2.6 `frontend/index.html` —— Vue 3 演示页

单文件、本地 `vue.global.prod.js`（不走 CDN，离线可用）。页面结构：

- 顶栏：Milvus / Redis / LLM 健康灯 + 集合块数；
- 左栏：10 题列表（题号 / 题源 / 类型 / 目标文档）；
- 右栏：
  - 题面 + 折叠的标准答案（ground_truth）；
  - **① 运行检索** → 展示每个片段来自哪个公司哪一页、由哪个通道召回、重排分多少；
  - **② 生成答案** → 展示 DeepSeek 答案与耗时；
  - **RAGAS 指标卡**（四指标 + 各指标有效样本数）；
  - **逐题明细表**（时延、召回条数、四指标逐题分值，`nan` 显示 `—`）。

### 2.7 `tests/test_pipeline.py` —— 自检

依据 `RAG/CLAUDE.md` 第二条：**自造的产物进评估前，先过「已知正例全过 + 已知反例全拦」**。自检覆盖四个关键件：

| # | 被检对象 | 正例 | 反例 |
|---|---|---|---|
| 1 | 解析三级降级链 | 真实 PDF 解析出 >100 页、页码从 1 起无空页 | 不存在的文件 → 每级都返回 `None` 而非抛异常；`MINERU_ENABLED=0` 时 MinerU 被跳过 |
| 2 | 文档身份回认 | 中文已损坏的文件名仍能认出 `000001 / 平安银行 / 2019` | 无股票代码的文件名 → `unknown`（不瞎猜） |
| 3 | 混合融合 | 双通道命中的片段得分最高、来源合并、分数归一到 ≤1 | 空通道参与融合不报错 |
| 4 | 缓存键指纹 | 同输入 → 同指纹 | 不同输入 → 不同键 |
| 5 | RAGAS 样本构造 | 10 题四元组齐全、题号唯一、sample 原题恰 4 道 | 空 ground_truth 可被检出 |

**任一用例失败即以非 0 退出**——自检不通过时不得给出任何评估结论。
