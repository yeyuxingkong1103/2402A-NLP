# 招股说明书 RAG 问答系统 · 工单3

> 工单编号：**人工智能NLP-RAG-PDF文档的表格解析及检索优化**
> 工单1/2 编号：人工智能NLP-RAG-基于PDF文档的问答系统 / …优化

基于 PDF 的问答系统，知识库为两份招股说明书：

| 文档键 | 文件 | 公司 | 页数 |
|---|---|---|---|
| `xingtu` | `data/raw/zhaogu_yixiangshu.pdf` | 武汉兴图新科电子股份有限公司 | 548 |
| `liyuan` | `data/raw/zhaogu2.pdf` | 武汉力源信息技术股份有限公司 | 350 |

**本工单的核心是表格解析**：把 PDF 表格从「一段被展平的文本」变成「列名与值绑定的结构化块」，
并量化它给检索精确度带来的变化。方案与实验报告见
[`docs/工单3-表格解析与检索优化方案.md`](docs/工单3-表格解析与检索优化方案.md)。

---

## 快速开始

```bash
# 依赖：本机 D:\roleplay-rag\.venv 已具备全部依赖，可直接复用
#   pip install -r requirements.txt   （干净环境重建时）

# 1) 建索引（约 5 分钟/份，CPU 向量化 1800+ 块）
python scripts/build_index.py                                    # 主线：表格结构化
python scripts/build_index.py --no-tables --out notable          # 对照：表格不结构化（约 3 分钟）

# 2) 启动服务
python main.py            # http://127.0.0.1:8010/ui
```

界面（`/ui`）左侧问答、右侧「检索链路」与「工单验收问题」（14 题）。
「对比模式」下拉里选 **表格解析前后（工单3）** 即可对任意问题做两条链路的实链路对比。

---

## 常用命令

```bash
python scripts/ablation_table.py              # 表格解析消融（单变量受控，约 30 秒）
python scripts/ablation_table.py --with-gate  # 带阈值闸门口径（线上实际配置）
python scripts/ablation_table.py --top-k 12   # 上下文预算敏感性

python scripts/report_wot3.py                 # 逐题「检索到的答案 + 检索精确度」报告
python scripts/report_wot3.py --no-answer     # 不调大模型，零成本

python scripts/demo_walkthrough.py --compare-table        # 演示台本（对应演示视频）
python scripts/demo_walkthrough.py --compare-table --id 4

python scripts/run_eval.py                    # RAGAS 四项指标 + RAG vs 纯 LLM 对比
python scripts/load_test.py                   # 并发 / 串行化 / 持续负载
```

---

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 服务信息（裸 JSON，不被静态前端覆盖） |
| GET | `/health` `/ready` `/config` | 探针 |
| GET | `/api/kb/status` | 知识库状态（含对照索引是否就绪） |
| GET | `/api/questions` | 14 道验收题 |
| POST | `/api/ask` | 完整问答（非流式） |
| POST | `/api/ask/stream` | SSE 流式问答 |
| POST | `/api/search` | **只检索不生成**（纯本地路径，用于分离系统层与模型层耗时） |
| POST | `/api/compare` | 本系统 vs 纯 LLM |
| POST | `/api/compare-opt` | 优化前（工单2 朴素 RAG）vs 优化后 |
| POST | `/api/compare-table` | **表格不结构化 vs 结构化**（工单3 核心） |
| POST | `/api/upload` | 上传 PDF 并重建索引 |

`/api/compare-table` 返回 `{table, no_table, fact_hits:{table,no_table}}`：
两条链路**只差一个变量**（喂给检索与生成的分块来自哪一份索引），
检索式、融合公式、prompt、生成模型完全相同。

```bash
curl -s --noproxy '*' -X POST http://127.0.0.1:8010/api/compare-table \
  -H "Content-Type: application/json" \
  -d '{"question":"与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？","top_k":8}'
```

> ⚠️ 本机环境变量里有 `http_proxy`，`curl` 访问 `127.0.0.1` 要加 `--noproxy '*'`，
> 否则请求会被代理截走去连一个不存在的上游（表现为 `HTTP 502 upstream connect failed`）。

---

## 目录结构

```
rag-pdf-table/
├─ main.py                     FastAPI 服务入口
├─ static/index.html           零构建静态前端
├─ src/
│  ├─ config.py                配置中心（工单标识 / 多文档 / 表格参数 / 检索参数）
│  ├─ table_parser.py          ★ 工单3 核心：表格解析
│  ├─ pdf_parser.py            PDF → 干净文本 + 结构化表格（表格区域与表名从正文剔除）
│  ├─ chunker.py               分块（标题感知 + 表格结构化成块）
│  ├─ index_store.py           索引落盘 / 加载 / BM25 / DF 过滤 / 文档识别
│  ├─ retriever.py             混合检索（余弦 + BM25 + 文档共识 + 文档消歧 + 阈值闸门）
│  ├─ rag.py                   问答链路（Query 理解 → 检索 → 生成）
│  ├─ query_understanding.py   Query 理解（含高置信直通）
│  ├─ evaluator.py             RAGAS 四项指标（LLM-as-judge）
│  └─ baseline.py              工单2 的朴素 RAG 对照实现
├─ scripts/
│  ├─ build_index.py           多文档索引构建（--no-tables / --out notable）
│  ├─ ablation_table.py        ★ 表格解析消融实验
│  ├─ report_wot3.py           ★ 逐题答案与精度报告
│  ├─ demo_walkthrough.py      演示台本（--compare-table）
│  ├─ run_eval.py              评估
│  ├─ load_test.py             压测
│  └─ compare_optimization.py  优化前后答案对比
└─ data/
   ├─ raw/                     两份招股说明书
   ├─ index/                   主线索引（表格结构化）
   ├─ index_notable/           对照索引（表格不结构化）
   └─ eval/                    评测集与报告
```

---

## 关键结论（详见方案文档第五节）

| 指标（top_k=8，闸门关） | 优化前 T0 | 优化后 T1 |
|---|---|---|
| 正确页命中率 | 1.000 | 1.000 |
| **MRR（正确页平均排名倒数）** | **0.552** | **0.788（+43%）** |
| 关键事实命中率 | 1.000 | 0.964 |

- 4 道力源表格题：MRR **0.708 → 0.875**，关键事实 4/4 全中；
- 10 道兴图正文题：MRR **0.489 → 0.753** —— 说明表格结构化没有拖累正文题；
- `top_k` 提到 12 后关键事实命中率回到 **1.000**。

已知代价：表格块的余弦相似度天然偏低（表块中位 0.415 vs 正文 0.439），
在 top-8 预算下会让个别数字排不进来；闸门阈值在双文档语料上无法完全分离真题与离题问题。
两者都已在方案文档第六节量化记录。

---

## 常见问题

**Q：为什么不改 PDF 里的页码口径？**
引用页码统一用 **PDF 物理页序（1-based）**。力源 PDF 书眉上印的是「3-21」这类原书章节页码，
与物理页序不同；两种混用会导致引用无法核对。

**Q：两份文档里都有「第 22 页」，引用会不会串？**
不会。每个块带 `doc_key`，引用面板、评测的 gold 标注、`build_context` 的来源行都带文档名。

**Q：多人同时用会不会把服务压垮？**
不会。本地检索路径（`/api/search`）实测 8 并发 56.2 rps、P99 146ms、零失败；
端到端耗时的大头是上游大模型往返（串行化系数 ≈ 并发数），不是本系统。
