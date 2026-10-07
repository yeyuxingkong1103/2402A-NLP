# 招股说明书 RAG 问答系统 · 工单5

> 工单编号：**人工智能NLP-RAG-Query理解优化任务**（多轮对话）
> 前置工单：人工智能NLP-RAG-基于PDF文档的问答系统 / …优化 / …表格解析及检索优化 / …图像内容解析及检索优化

**本工单的核心是 Query 理解优化：实现多轮对话。**
把「一句话一句话各自独立理解」的系统，改成「记得前面问过什么」的系统，
使下面这类**单独拿出来就没法检索**的话能被正确处理：

| 轮 | 用户说的 | 难点 |
|---|---|---|
| Q2 | **他**参与的哪个工程荣获了国家科技进步一等奖？ | 指代消解：「他」= 上一轮的兴图新科 |
| Q3 | **这个公司**的法定代表人是谁？ | 指代消解 |
| Q4 | **那武汉力源信息技术股份有限公司呢？** | 省略补全：只给了新主体、没给谓词 |

方案与实验报告见 [`docs/工单5-Query理解优化与多轮对话方案.md`](docs/工单5-Query理解优化与多轮对话方案.md)。

**验收结果**：准确性 **100%（23/23）**（要求 ≥90%）；HTTP 实测 5 轮响应
**1.80 / 1.27 / 2.64 / 1.51 / 0.96 秒**（要求 ≤3 秒）。

---

基于 PDF 的问答系统，知识库为两份招股说明书：

| 文档键 | 文件 | 公司 | 页数 |
|---|---|---|---|
| `xingtu` | `data/raw/zhaogu_yixiangshu.pdf` | 武汉兴图新科电子股份有限公司 | 548 |
| `liyuan` | `data/raw/zhaogu2.pdf` | 武汉力源信息技术股份有限公司 | 350 |

**本工单的核心是图像内容解析**：把 PDF 里的图（组织结构图、柱状图、流程图…）
从「一堆不可检索的位图 + 打散的文字碎片」变成**可检索的结构化描述与跨模态向量**，
并量化它给检索精确度带来的变化。
方案与实验报告见 [`docs/工单4-图像内容解析与检索优化方案.md`](docs/工单4-图像内容解析与检索优化方案.md)。

---

## 快速开始

```bash
# 依赖：本机 D:\roleplay-rag\.venv 已具备全部依赖，可直接复用
#   pip install -r requirements.txt   （干净环境重建时）

# 1) 检测 + 裁剪图区域（约 10 秒/份；表格 bbox 有缓存）
python scripts/build_images.py

# 2) 多模态语义解析（68 张图，qwen-vl-plus，约 8 分钟；已解析的会跳过）
python scripts/parse_images.py

# 3) CLIP 图向量（可选；需要先下好 chinese-clip 权重到 CLIP_MODEL_PATH）
python scripts/parse_images.py --clip-only

# 4) 建索引
python scripts/build_index.py                                    # 主线：表格 + 图像都解析
python scripts/build_index.py --no-images --out noimage          # 对照：图像不解析（约 3 分钟）

# 5) 启动服务
python main.py            # http://127.0.0.1:8010/ui
```

界面（`/ui`）左侧问答、右侧「检索链路」与「工单验收问题」（16 题）。
「对比模式」下拉里选 **图像解析前后（工单4）** 即可对任意问题做三条链路的实链路对比，
并把命中的裁剪图直接显示出来。

---

## 常用命令

### 工单5（本工单：多轮对话）

```bash
python scripts/demo_dialogue.py --save         # 工单原文的 5 轮演示 + 落台本
python scripts/demo_dialogue.py --mode off     # 对照：完全不做多轮（第2~4轮必崩）

python scripts/report_wot5.py                  # 全量评测（5 场 / 23 轮）
python scripts/report_wot5.py --dialogue D1    # 只跑工单原题那 5 轮
python scripts/report_wot5.py --no-answer      # 不调大模型，只看消解 + 检索

python scripts/ablation_dialogue.py --group A  # 消解方式消融（off/concat/rule/rule+llm）
python scripts/ablation_dialogue.py --group B  # 子优化项消融（doc_filter / subject_first）
```

界面（`/ui`）右上角可开关 **多轮对话**、切换 4 种消解模式；
右侧新增 **「对话上下文」** 面板，逐轮显示系统把每句话的主体/话题理解成了什么；
「▶ 连播工单5轮对话」一键跑完工单原文的 5 轮。

### 工单1~4

```bash
python scripts/ablation_image.py               # 图像解析消融（三条臂，约 30 秒）
python scripts/ablation_image.py --with-gate   # 带阈值闸门口径（线上实际配置）
python scripts/ablation_image.py --clip        # 打开 CLIP 跨模态召回臂
python scripts/ablation_image.py --top-k 12    # 上下文预算敏感性

python scripts/report_wot4.py                  # 逐题「检索到的答案 + 检索精确度」报告
python scripts/report_wot4.py --id 5 --id 6    # 只跑新增的两道图像题
python scripts/report_wot4.py --no-answer      # 不调大模型，零成本

python scripts/demo_walkthrough.py --compare-image       # 演示台本（对应演示视频）
python scripts/demo_walkthrough.py --compare-image --id 6

python scripts/demo_walkthrough.py --compare-table       # 工单3 的表格对比仍可复跑
python scripts/run_eval.py                     # RAGAS 四项指标 + RAG vs 纯 LLM 对比
python scripts/load_test.py                    # 并发 / 串行化 / 持续负载
```

---

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | JSON 根探针（不要被前端覆盖） |
| GET | `/health` `/ready` `/config` | 系统探针 / 就绪检查 / 非敏感配置 |
| GET | `/api/kb/status` | 知识库状态（含两份对照索引、图像块数、CLIP 状态） |
| GET | `/api/questions` | 工单验收题目（16 题） |
| POST | `/api/ask` | RAG 问答（带 `session_id` 即走多轮） |
| POST | `/api/ask/stream` | SSE 流式问答（同上） |
| GET | `/api/session/{session_id}` | 查看会话状态（逐轮主体/话题/检索式） |
| POST | `/api/session/reset` | 清空会话上下文 |
| GET | `/api/session/stats` | 会话存储指标（对应「资源消耗合理」验收项） |
| POST | `/api/search` | **只检索、不生成**（本地路径，可压测；`clip=false` 可关跨模态通道） |
| POST | `/api/compare` | RAG vs 纯 LLM |
| POST | `/api/compare-opt` | 优化后 vs 朴素 RAG（工单2） |
| POST | `/api/compare-table` | 表格结构化 vs 不结构化（工单3） |
| POST | **`/api/compare-image`** | **图像多模态解析 vs 不解析（工单4 核心）** |
| POST | `/api/upload` | 上传 PDF 重建索引 |
| GET | `/ui` | 零构建静态问答界面 |
| GET | `/images/<doc>/<file>.png` | 裁剪出的图（演示时直接看原图） |

---

## 目录

```
src/
  session.py            多轮会话状态（限长 + 线程安全 + LRU）  （工单5 新增）
  dialogue.py           上下文消解：指代消解 + 省略补全        （工单5 新增）
  image_detector.py     图区域检测 / 裁剪 / 正文剔除        （工单4 新增）
  image_semantics.py    多模态大模型（qwen-vl-plus）语义解析（工单4 新增）
  diagram_topology.py   方框-连接线图的几何拓扑抽取（层级归属的权威来源）
  clip_encoder.py       Chinese-CLIP 跨模态向量            （工单4 新增）
  table_parser.py       表格结构化（工单3）
  retriever.py          稠密+稀疏+文档先验+CLIP 跨模态 融合检索
  chunker.py            分块（正文/表格/图像三类）
  rag.py                串联链路与答案组装
scripts/
  build_images.py       检测+裁剪+拓扑+正文剔除
  parse_images.py       多模态解析 + CLIP 向量
  build_index.py        建索引（--no-images 出对照组）
  ablation_image.py     图像解析消融（三条臂、两组单变量对照）
  demo_dialogue.py      工单原文的 5 轮对话演示              （工单5 新增）
  report_wot5.py        多轮评测 + 报告（JSON/MD）           （工单5 新增）
  ablation_dialogue.py  多轮消融 A/B 两组                    （工单5 新增）
  report_wot4.py        逐题「检索到的答案 + 检索精确度」报告
  demo_walkthrough.py   演示台本（--compare-image）
data/
  raw/                  两份 PDF
  images/               裁剪图 + figures.json（图清单/描述/CLIP 行号）
  index/                主线索引（表格 + 图像）
  index_noimage/        对照组索引（图像不解析）
  index_notable/        工单3 对照索引（表格不结构化）
  eval/                 评测集 + 各类报告
static/index.html       零构建前端控制台
```
