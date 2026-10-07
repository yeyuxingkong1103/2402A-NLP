# 招股说明书 RAG 问答系统 · 工单6（混合检索）

> 工单编号：**人工智能NLP-RAG-混合检索任务**
> 前置工单：人工智能NLP-RAG-基于PDF文档的问答系统 / …优化 / …表格解析及检索优化 /
> 　　　　　…图像内容解析及检索优化 / …Query理解优化任务
> 项目：RAG · 创建人：王洪荣

基于 PDF 的问答系统，知识库为两份招股说明书：

| 文档键 | 文件 | 公司 | 页数 |
|---|---|---|---|
| `xingtu` | `data/raw/zhaogu_yixiangshu.pdf` | 武汉兴图新科电子股份有限公司 | 548 |
| `liyuan` | `data/raw/zhaogu2.pdf` | 武汉力源信息技术股份有限公司 | 350 |

**本工单的核心是混合检索**：把原来「一条固定的检索链路」变成
**三种检索策略 × 五种融合算法 × 五种重排器**的可配置组合，
并量化它给检索精确度带来的变化。
方案与实验报告见 [`docs/工单6-混合检索方案.md`](docs/工单6-混合检索方案.md)。

---

## 一、5 分钟跑起来

### 1. 环境

| 项 | 说明 |
|---|---|
| Python | 3.10+（本项目在 3.12 / 3.13 上验证） |
| 依赖 | 见 `requirements.txt`（pypdf / pdfplumber / sentence-transformers / jieba / rank-bm25 / openai / fastapi / uvicorn） |
| 向量模型 | 本机已存在 `D:\bge-small-zh-v1.5`（512 维） |
| 重排模型（可选） | `D:\models\bge-reranker-v2-m3`（fp16，默认关闭） |
| 大模型 | 任一 OpenAI 兼容服务（DashScope / Moonshot / 硅基流动） |

> 本机已有一套现成的虚拟环境，包含上述全部依赖，可直接用它作为解释器：
> `D:\roleplay-rag\.venv\Scripts\python.exe`
> 若要在干净环境重建：`python -m venv .venv && .venv\Scripts\pip install -r requirements.txt`

### 2. 配置

编辑根目录 `.env`，只需确认两项：

```ini
LLM_API_KEY=sk-...                          # 大模型 Key（必填）
EMBEDDING_MODEL_KEY=bge-small-zh            # 嵌入模型（注册表里的 key）
```

工单6 新增的可调项（都有默认值，按需改）：

```ini
RETRIEVAL_STRATEGY=hybrid        # vector | fulltext | hybrid
HYBRID_FUSION=weighted           # weighted | rrf | borda | vote | evidence
HYBRID_VECTOR_WEIGHT=0.6         # 向量路权重，0 = 纯全文，1 = 纯向量
RERANKER=tfidf                   # none | tfidf | llm | feedback | cross
FULLTEXT_W_TITLE=2.0             # 全文检索「标题」字段权重（BM25F）
FULLTEXT_W_SUMMARY=1.5           # 「摘要」字段权重
RETRIEVAL_MIN_EVIDENCE=0.70      # 阈值闸门
```

### 3. 建索引 + 启动

```bash
# 建索引（首次或换 PDF / 换嵌入模型时；已存在可跳过）
python scripts/build_index.py

# 启动服务（预热约 25~30 秒：索引 + BM25 + 向量模型 + 倒排索引 + 停用词表 + CLIP）
python main.py            # http://127.0.0.1:8010/ui
```

---

## 二、界面怎么用

`/ui` 左侧问答、右侧「检索链路」与「工单验收问题」。
输入框下方那一排是**工单6 新增的检索策略配置**：

| 控件 | 作用 |
|---|---|
| 检索策略 | 向量检索 / 全文检索 / 混合检索 |
| 融合算法 | 加权平均 / RRF 倒数排名 / Borda 排序投票 / 多数投票 / 依据分 |
| 重排器 | TF-IDF / LLM / 用户反馈自适应 / 交叉编码器 / 不重排 |
| 向量权重 | 滑块，0 = 纯全文，1 = 纯向量 |
| 嵌入模型 | 切换嵌入模型（切到未建索引的模型会提示先跑 `build_index`） |
| 对比模式 | **三策略对比（工单6）** / **融合算法对比（工单6）**，另保留工单2~4 的对比项 |

改任何一个控件，下一次提问就换一套检索链路；
右侧「检索链路」面板会显示每一路的耗时、候选池大小、两路同时命中了几条、
以及重排器的耗时 —— 这些数字就是调参时的直接反馈。

### 全文查询语法（在问题里直接写）

| 写法 | 含义 |
|---|---|
| `"军用领域"` | 短语匹配，要求相邻 |
| `收入 AND 军用` / `OR` / `NOT` | 布尔，支持括号嵌套 |
| `section:风险提示` | 字段限定（title / body / summary） |
| `注删资本~1` | 模糊匹配，编辑距离 ≤1（差一个字时用） |
| `销售*` | 前缀通配 |

**错别字还能自动容错**：不打 `~1` 也行。比如把「注册资本」打成「注册酱」
（整字被同音替换，差 2 步，`~1` 是修不了的），系统会发现「酱」在索引里 df=0、
前面紧挨着「注册」，自动把整段补全成「注册资本」。
改写记录会显示在界面的检索链路面板上。开关：`FULLTEXT_REPAIR`（默认开）。

### 常用接口

```bash
curl -s http://127.0.0.1:8010/api/retrieval/options | jq     # 全部可配置项
curl -s -X POST http://127.0.0.1:8010/api/search \
  -H "Content-Type: application/json" \
  -d '{"question":"报告期内来自军用领域的收入分别是多少？","strategy":"hybrid",
       "fusion":"rrf","reranker":"feedback","vector_weight":0.5}'
curl -s -X POST http://127.0.0.1:8010/api/compare-strategy \
  -H "Content-Type: application/json" \
  -d '{"question":"……","reranker":"tfidf"}'                  # 三条链路并排返回
curl -s -X POST http://127.0.0.1:8010/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"question":"……","chunk_idx":123,"useful":true}'       # 自适应重排器在线学习
```

---

## 三、常用命令

```bash
########## 评测（本工单的核心交付物）
python scripts/eval_strategies.py                # 全量网格，约 90 秒，产出 md + json
python scripts/eval_strategies.py --grid strategies     # 只看策略对比
python scripts/eval_strategies.py --grid fusions        # 只看融合算法
python scripts/eval_strategies.py --grid weights        # 只看权重扫描
python scripts/eval_strategies.py --grid rerankers      # 只看重排器
python scripts/eval_strategies.py --grid syntax         # 全文查询语义（OR / AND / 关语法）
python scripts/eval_strategies.py --with-llm            # 额外评测 LLM 重排器（慢、耗 token）
python scripts/eval_strategies.py --no-clip --limit 5   # 关 CLIP、只跑 5 题（试水）

########## 索引
python scripts/build_index.py                          # 主线索引（data/index）
python scripts/build_index.py --embed-model m3e-base   # 用指定模型建索引（data/index_m3e_base）

########## 其它（沿用前几个工单）
python scripts/run_eval.py                             # LLM-as-judge 的 RAG 质量评估
python scripts/load_test.py                            # 并发压测
python scripts/ablation_retrieval.py                   # 工单2 检索层消融
```

---

## 四、本工单新增了什么

| 文件 | 职责 |
|---|---|
| `src/fulltext.py` | 倒排索引 + BM25F + 查询语法（布尔 / 短语 / 模糊 / 字段限定） |
| `src/reranker.py` | 五种重排器 + 用户反馈在线学习与落盘 |
| `src/hybrid.py` | 策略编排、融合算法、权重、排序先验、阈值闸门 |
| `src/embed_registry.py` | 多嵌入模型注册表 + 按模型分目录的索引 |
| `scripts/eval_strategies.py` | 策略 / 融合 / 权重 / 重排器评测与报告生成 |
| `docs/工单6-混合检索方案.md` | 方案、设计取舍与踩坑记录 |
| `docs/工单6-原文提取.txt` | 工单 PDF 原文提取 + 要点归纳 |
| `docs/工单6-演示视频台本.md` | 两条演示视频的分镜与口播稿 |

改动的既有文件：`src/config.py`（新增配置）、`src/retriever.py`（`RetrievedChunk` 增加
`sources / rerank_score / rerank_reason`）、`src/rag.py`（检索入口切换到 `hybrid`，
接口对齐所以其余零改动）、`main.py`（新增 4 个接口 + 预热）、`static/index.html`（策略面板）。

---

## 五、验收对照

| 工单验收项 | 要求 | 本方案 |
|---|---|---|
| 准确率 | ≥ 90% | **95.24%**（优化前基线 85.71%），见方案文档 §7 |
| 召回率 | ≥ 95% | **98.51%**（优化前基线 92.54%），见方案文档 §7 |
| 响应时间 | ≤ 3 秒 | 检索端 **113~125 ms**（p95 143 ms），基线约 1 s |
| 三种检索策略可配置 | 向量 / 全文 / 混合 | ✅ |
| 多种嵌入模型 | bge / m3e / 其它（注册表，未下载的会提示） | ✅ |
| 至少 3 种重排算法 | TF-IDF / LLM / 用户反馈自适应（+ 交叉编码器 + 不重排对照） | ✅ |
| 倒排索引 + 布尔/短语/模糊 + 多字段 | 三字段 BM25F | ✅ |
| 权重调整 + 融合算法 | 权重滑块 + 5 种融合算法 | ✅ |
| 代码注释含工单编号 | 硬性要求 | ✅ 新增模块头部均标注 |

---

## 六、目录

```
data/
  raw/          源 PDF
  index/        bge-small-zh 的索引（默认，沿用工单1~5）
  index_<key>/  其它嵌入模型的索引（按需构建）
  images/       裁剪出的图 + 语义描述 + CLIP 向量（工单4）
  eval/         评测集与评测报告
docs/           原文提取 / 方案 / 演示台本
scripts/        索引构建、评测、消融、压测
src/            检索、重排、融合、生成、配置
static/         零构建前端
```
