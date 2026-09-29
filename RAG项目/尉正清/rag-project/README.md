# 基于 RAG 的角色扮演系统

对标 character.ai 的角色扮演对话系统，但要求专业领域回答**有据可依**——所有回答
严格基于知识库检索结果，资料不足时如实说明而不是编造。

当前内置三个角色：**律师**、**心理医生**、**金融理财师**。角色人设与提示词模板
存放在 MySQL，新增角色无需改代码。

## 技术栈

| 环节 | 选型 |
|---|---|
| 大模型 | DeepSeek `deepseek-flash`（推理模型） |
| RAG 框架 | LangChain（Models / Prompts / Chains / Memory 已落地） |
| 向量化 | BGE-M3（一次前向同时产出稠密向量与稀疏权重） |
| 向量库 | Milvus 2.4.17 standalone |
| 检索 | 稠密 + 稀疏双路召回 → RRF 融合 → BGE-reranker-v2-m3 精排 |
| 短文本分块 | 不切——法条按「条」、问答按「一对」，本身就是原子语义单元 |
| 长文本分块 | 语义分块（按相邻句相似度的低谷切分） |
| PDF 解析 | MinerU 主路径（版面 / OCR / 公式 / 表格一体，独立 venv 子进程调用）+ PyMuPDF 去水印；兜底为 PyMuPDF 文本层 + PaddleOCR 图表文字 + pdfplumber 表格 |
| 短期记忆 | Redis List（最近 10 轮对话） |
| 长期记忆 | Milvus 集合 `long_term_memory`（对话摘要 + 语义召回） |
| 关系库 | MySQL 8.0（用户、角色、会话、消息、法条索引） |
| Web 框架 | FastAPI + Uvicorn |
| 日志 | Python 标准库 logging（控制台 + 滚动文件 + 错误日志） |

## 目录结构

```
rag-project/
├── app/
│   ├── main.py              FastAPI 入口与路由挂载
│   ├── config/
│   │   ├── settings.py      统一配置（读 .env）
│   │   └── components.py    重资源懒加载单例（BGE-M3 / Reranker / LLM / 切分器）
│   ├── db/
│   │   ├── mysql_conn.py    SQLAlchemy 引擎、Session、建库建表
│   │   ├── milvus_conn.py   Milvus 封装：集合管理、写入、混合检索
│   │   └── redis_conn.py    Redis 封装：短期记忆、会话状态、热度、缓存
│   ├── models/tables.py     ORM 表定义
│   ├── core/
│   │   ├── chain_service.py     LangChain 集成（Models/Prompts/Chains）
│   │   ├── llm_service.py       DeepSeek 调用（含推理模型空回复补偿）
│   │   ├── prompt.py            系统提示词组装
│   │   ├── query_service.py     追问改写 + 查询扩写
│   │   ├── retrieve_service.py  多路召回 + 加权 RRF 融合 + 精排
│   │   ├── metadata_service.py  元数据路召回（法条号精确命中）
│   │   ├── chunk_service.py     语义分块
│   │   ├── memory_service.py    双层记忆（含 LangChain Memory 接口）
│   │   ├── role_service.py      角色读取与种子数据（返回可缓存快照）
│   │   ├── session_service.py   会话与消息持久化
│   │   ├── ingest_service.py    知识入库
│   │   ├── update_service.py    知识增量更新
│   │   ├── pdf_service.py       PDF 解析编排（MinerU 主路径 + 兜底三件套）
│   │   ├── mineru_service.py    MinerU 调用（独立 venv + 子进程）
│   │   ├── ocr_service.py       兜底路径：PaddleOCR 补图表文字
│   │   ├── table_service.py     兜底路径：pdfplumber 还原表格为 Markdown
│   │   ├── watermark.py         PDF 水印检测与擦除
│   │   └── rag_service.py       RAG 主流程编排
│   ├── api/
│   │   ├── chat.py              问答（同步 + SSE 流式）
│   │   ├── role.py              角色与会话
│   │   ├── knowledge.py         知识入库与更新
│   │   └── system.py            用户、健康检查、统计
│   └── schemas/             出入参模型
├── data/                    知识库源数据（三个角色 JSONL）
├── docs/                    项目文档
│   ├── 01-需求规格说明书.md     02-功能需求.md
│   ├── 03-业务流程与规则.md     04-思维导图.md（含思维导图.png）
│   ├── 05-系统设计.md          06-数据集说明.md
│   ├── 07-接口文档.md          08-设计文档.md
│   └── 09-部署文档.md          10-测试报告.md
├── evals/                   RAGAS 评测（评测集、采集、判分、对比）
│   ├── dataset.jsonl            56 题专业评测集
│   ├── collect.py               采集回答与检索上下文
│   ├── score.py                 RAGAS 判分（独立 venv）
│   └── compare.py               优化前后对比
├── mineru/                  MinerU 独立 venv
│   └── .venv/                   MinerU 及其依赖（与主环境冲突，单独安装）
├── jmeter/                  压测计划
│   ├── read_api.jmx             读接口压测（高并发）
│   ├── chat_api.jmx             问答接口压测（低并发长响应）
│   └── questions.csv            压测用问题集
├── tests/                   接口测试
│   ├── test_*.py                pytest 用例
│   ├── build_postman.py         生成 Postman 集合
│   └── run_all.sh               一键跑三类测试
├── scripts/
│   ├── init_db.py               初始化数据库与角色
│   ├── knowledge_base.py        知识库构建 / 清理 / 统计
│   ├── dedup_kb.py              知识库去重（重复切片清理）
│   ├── fetch_statutes.py        法条抓取与按条切分
│   ├── fetch_finance_extra.py   金融知识补充抓取（Fin-Eva）
│   ├── build_law_index.py       构建 MySQL 法条索引
│   ├── prepare_dataset.py       开源语料清洗（数据血缘记录）
│   ├── dataset_io.py            清洗脚本的 IO 工具
│   └── draw_mindmap.py          生成思维导图图片
├── uploads/                 上传文档暂存
├── deploy/                  部署
│   ├── install.sh               安装脚本（环境检查 → 依赖 → 初始化）
│   ├── run.sh                   启动脚本
│   ├── shutdown.sh              停止脚本
│   └── docker-compose.yml       依赖服务编排
└── run.py                   Python 启动入口
```

## 快速开始

### 1. 依赖服务

三个服务均以 Docker 运行，本项目默认连接：

```bash
docker start redis mysql8 milvus-standalone
```

| 服务 | 地址 | 凭据 |
|---|---|---|
| MySQL | localhost:3306 | 用户 `root`，密码见 `.env` 的 `MYSQL_PASSWORD`，库 `rag_roleplay` |
| Redis | localhost:6379 | 密码见 `.env` 的 `REDIS_PASSWORD` |
| Milvus | localhost:19530 | 无 |

> WSL 中若长时间无操作，发行版会休眠并带走容器。建议在 WSL 里执行一次
> `docker update --restart unless-stopped redis mysql8 milvus-standalone`，
> 或在 Windows 侧保持一个 WSL 终端常开。

### 2. 安装依赖

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# PDF 解析用的 MinerU 单独装在项目内的独立 venv：
# 它要求 openai<3，与主环境的 openai 3.x 冲突，不能装在一起
python3 -m venv mineru/.venv
mineru/.venv/bin/pip install mineru==4.0.5

# 再下模型（约 0.23GB，basic 档小模型）。不下载的话首次解析会现下，很慢
MINERU_MODEL_SOURCE=modelscope mineru/.venv/bin/mineru-models-download \
    --tier basic --small-backend torch -s modelscope
```

> 也可用 `./deploy/install.sh --with-mineru` 一步装完（该脚本默认跳过这步）。

> **兜底三件套的依赖在主环境里**：`requirements.txt` 的 `pymupdf` / `pdfplumber` /
> `paddlepaddle` / `paddleocr` 四项，装完主环境即可拥有完整的兜底解析能力，
> 无需额外步骤。

> `torch` 需与机器 CUDA 版本匹配，建议按 PyTorch 官网指令单独安装。

> **关于 MinerU**：模型（模型目录约 1.4GB）落在 `/root/.mineru/models/`，模型源默认 `modelscope`
> ——本机 huggingface.co 直连不通。没装也能跑：`pdf_service` 会自动退回**兜底三件套**
> ——PyMuPDF 取文本层、PaddleOCR 补图表文字、pdfplumber 还原表格，扫描件与图表
> 照样能读出内容（只是阅读顺序与公式弱于 MinerU）。相关开关见 `settings.py` 的
> `MINERU_*`（可用环境变量覆盖）：`MINERU_BIN` / `MINERU_TIER` / `MINERU_MODEL_SOURCE` / `MINERU_TIMEOUT`。

### 3. 配置文件

复制并修改 `.env`：

```ini
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_MODEL=deepseek-flash
MYSQL_PASSWORD=your-mysql-password
REDIS_PASSWORD=your-redis-password
EMBEDDING_MODEL_PATH=/root/models/bge-m3
RERANKER_MODEL_PATH=/root/models/bge-reranker-v2-m3
```

> `deepseek-flash` 是**推理模型**，会先输出 `reasoning_content`。
> 若 `LLM_MAX_TOKENS` 给小了，token 会被推理耗尽导致正文为空，
> 因此默认设为 4096；`llm_service` 内也做了空正文自动补偿重试。

### 4. 初始化

```bash
.venv/bin/python -m scripts.init_db              # 建库建表 + 写入三个角色 + 演示用户
.venv/bin/python -m scripts.knowledge_base       # 构建知识库（全量重建，约 2 分钟）
.venv/bin/python -m scripts.dedup_kb --apply     # 去重（重建后必跑，见下方说明）
.venv/bin/python -m scripts.build_law_index      # 构建 MySQL 法条索引
```

> **为什么要单独跑去重**：源数据里有 68 条记录归一化后（去空白与标点）
> 文本完全相同——它们只是标点写法不同，在 JSONL 里是两条独立记录。
> 全量重建会得到 **13432** 条，去重后才是 **13364** 条。文档中引用的
> 13364 / 4555 / 4675 / 4134 均为**去重后**的数字，少了这一步对不上。

### 5. 启动

```bash
.venv/bin/python run.py
```

接口文档：<http://127.0.0.1:8000/docs>

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat/ask` | 角色扮演问答（返回完整回答） |
| POST | `/api/chat/stream` | 流式问答（SSE） |
| GET | `/api/roles` | 角色列表 |
| GET | `/api/roles/{role_key}` | 角色详情（含人设提示词） |
| GET | `/api/roles/hot` | 角色热度排行 |
| GET | `/api/sessions` | 会话列表 |
| GET | `/api/sessions/{sid}/messages` | 会话历史消息 |
| GET | `/api/sessions/{sid}/memory` | 查看短期/长期记忆 |
| DELETE | `/api/sessions/{sid}` | 删除会话 |
| POST | `/api/users/register` | 注册 |
| POST | `/api/users/login` | 登录 |
| POST | `/api/ingest/dataset` | 按角色批量导入内置数据集 |
| POST | `/api/ingest/upload` | 上传文档入库（PDF 自动去水印，MinerU 解析，不可用时走三件套兜底） |
| POST | `/api/update/document` | 增量更新单个文档 |
| POST | `/api/update/dataset/{role_key}` | 批量更新角色数据集 |
| DELETE | `/api/update/document` | 删除某文档的全部知识 |
| GET | `/api/health` | 各依赖健康状态 |
| GET | `/api/stats` | 知识库与业务统计 |

### 调用示例

```bash
# 首轮
curl -X POST http://127.0.0.1:8000/api/chat/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"酒驾撞人要判多久？","role_key":"lawyer","user_id":1}'

# 多轮追问（带上返回的 session_id，指代会被自动补全）
curl -X POST http://127.0.0.1:8000/api/chat/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"那要是逃逸了呢？","role_key":"lawyer","user_id":1,"session_id":"<上一步返回>"}'
```

演示账号：`demo` / `demo123`

## 核心机制

### 混合检索
BGE-M3 一次前向同时产出稠密向量与稀疏权重（BM25 风格），两路各自召回后
用 RRF 融合，再交给 BGE-reranker 交叉编码器精排取 top-5。纯向量检索作为降级路径。

### 双层记忆
- **短期**：Redis List 保存最近 10 轮原文，读写 O(1)，保证"刚说过的话"一字不差。
- **长期**：短期窗口滑出的旧对话，先用大模型压缩成摘要，再连同向量写入 Milvus，
  按 `user_id + role_id` 隔离。每轮按当前问题语义召回最相关的若干条注入提示词。

### 多轮追问改写
用户问"那他呢"这类省略句时，直接用原句检索必然跑偏。系统会结合历史把追问改写成
独立完整的问题再检索（实测：`那要是逃逸了呢？` → `酒驾撞人后逃逸的，要判多久？`）。

### 查询扩写与元数据路（可关闭）
把一个问题扩成 3 个查询分别召回，并新增一路「按法条号精确命中」。
两项优化都可用 `.env` 开关控制：

```ini
QUERY_EXPAND_N=3              # 1 = 关闭扩写
ENABLE_METADATA_ROUTE=true   # false = 只用向量检索
```

> **实测结论（详见 `docs/10-测试报告.md`）**：扩写这项优化**未带来显著提升**，
> 四项 RAGAS 指标平均仅 +0.0048，落在判分噪声内。分角色看差异相反——
> 心理医生角色明显受益（相关性 +0.205），金融理财师角色明显受损（−0.143）。
> 过程还证伪了「给原问题加权」这一看似合理的修正（实测更差，已回退）。
> 保留开关是为了便于按角色单独调优。

**元数据路的定位修正（本轮）**：原先把法条元数据路当成普通召回路径，
和其它路一起参与 RRF 融合与重排。实测发现这样**用不上**——问「民法典第
一百八十八条」，它确实被召回了，却在重排后掉到第 5 位：

| 路径 | RRF 后重排得分 |
|---|---|
| 向量路 第一千一百七十八条 | 2.965 |
| 向量路 第一千零五十八条 | 2.953 |
| 向量路 第一百二十八条 | 2.908 |
| **元数据路 第一百八十八条（正解）** | **2.543** |

根因是**拿语义相似度去排精确事实**：条号是确定信息，而重排器只看语义，
「诉讼时效」这类主题相近的条文天然比条号本身更"像"问题。
现改为：**识别到明确条号并精确命中时，该条直接置顶，不参与重排排序**；
没有条号的问题仍走纯语义，不误触发。

> 另外发现评测集存在覆盖盲区：原 50 题全是口语化提问（"开车撞了人之后跑了，
> 会被判几年？"），**含条号的题目为 0**——所以之前那次消融实际上从未触发过
> 元数据路，「未带来显著提升」的结论对它并不成立。现已补充 6 道条号题
> （`law-21`~`law-26`，答案取自知识库真实法条原文），使该路径可被评测覆盖。

### PDF 去水印
针对三类情况：多页重复出现的文字水印（含斜向旋转）、重复铺版的图片水印、
元数据与注释。仅"重复"不足以判定——正文页眉表头也会重复——因此还会校验
形态特征（旋转 / 大字 / 浅灰低对比 / 极短），避免误删正文。

> **顺序**：MinerU 只做解析、**不做水印擦除**，水印不去掉会被当成正文解析进去。
> 所以流程是「先用 PyMuPDF 擦水印 → 另存一份干净 PDF → 再交给 MinerU」。

### PDF 解析（MinerU 主路径 + 三件套兜底）
主路径是 **MinerU**——版面、OCR、公式、表格一次解析成 Markdown。原先的自研三件套
（PyMuPDF 取文本层、PaddleOCR 补图表文字、pdfplumber 还原表格）**没有废弃**，
而是退居兜底路径，在 MinerU 不可用时接管。同一份 28 页 WHO 报告实测：

| 管线 | 页数 | 字数 | 耗时 | 备注 |
|---|---|---|---|---|
| **MinerU basic（主路径，默认）** | 1（整篇 Markdown） | **15174** | **约 38 秒** | 剥掉 25 张 base64 内嵌图 |
| 兜底三件套（PyMuPDF + PaddleOCR + pdfplumber） | 27 | 14925 | 约 283 秒 | OCR 补充 125 行 / 6 页；提取表格 1 个 |
| 改造前（只有自研三件套时） | 27 | 14830 | 254 秒 | OCR 补充 110 行 / 5 页 |

> 兜底路径恢复得比原管线还完整一点点（14925 字 vs 14830 字，OCR 多补 15 行、
> 多覆盖 1 页），因为现在是「MinerU 优先、失败才兜底」，而不是当初的唯一路径。

> **为什么用独立 venv + 子进程**：MinerU 要求 `openai<3`，与本项目的
> `openai==3.13.0`（`llm_service` / `chain_service` 在用）硬冲突；且这台机器只有
> 7.6G 内存，子进程解析完即退出，内存立刻归还。沿用项目已有的 `evals/.venv` 模式。

> **为什么只开 basic 档**：standard 要多跑一个 VLM，而 VLM 引擎只能用 llama.cpp
> 的 CPU 路径（没装 vLLM），单次加载就要 361 秒；子进程模式**每次调用都要重付**
> 这笔开销，因此不实用（同一份 28 页文档：basic 约 38 秒，standard 实测 431 秒，
> 而字数完全一样）。

> **CUDA 版本要对齐（踩过的坑）**：本机驱动 566.36 只支持到 CUDA 12.7，而
> `pip install mineru` 默认拉到的 torch 是 **cu130** 构建，`torch.cuda.is_available()`
> 会是 `False`，MinerU 静默退回 onnx/CPU。必须显式装 cu126 的 torch：
>
> ```bash
> mineru/.venv/bin/pip uninstall -y torch torchvision triton $( \
>     mineru/.venv/bin/pip list | awk '/^nvidia-/{print $1}')
> mineru/.venv/bin/pip install --force-reinstall --no-deps \
>     --index-url https://download.pytorch.org/whl/cu126 torch torchvision
> mineru/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu126 \
>     --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple/ torch==2.14.0
> ```
>
> 装完用 `mineru/.venv/bin/mineru-kit models show` 确认 `Effective small backend`
> 是 `torch` 而不是 `onnx`。换后端后**首次解析会现下 torch 版模型**（227MB），
> 那次耗时不代表常态。

> **必须剥掉 base64 内嵌图**：MinerU 会把图内嵌进 Markdown。实测 28 页文档嵌了
> 25 张、约 7MB——剥图前 710 万字符，剥后才是 1.5 万字。这些数据对检索毫无价值。

> **兜底路径**：MinerU 未安装或解析失败时，自动退回**与改造前等价的自研三件套**
> ——PyMuPDF 逐页取文本层、PaddleOCR 补图表文字（`ocr_service.py`）、pdfplumber
> 还原表格（`table_service.py`），接口不报错。这条路径是刻意保留的：MinerU 是独立
> venv 里的重依赖，开发环境不装它也要能把项目跑起来。
>
> 上传接口的 `use_ocr` / `use_tables` 两个参数**只作用于兜底路径**：走 MinerU 时
> 它自带这两项且无法单独关闭，两个开关会被忽略，并在 `watermark.notes` 里记一条
> 「use_ocr/use_tables 在主路径下不生效」。
>
> **两个踩过的坑**：① `PaddleOCR(enable_mkldnn=False)` 是**必须的**，不是性能选项
> ——paddlepaddle 3.3.1 的 oneDNN 指令在 PP-OCRv5/v6 上会抛
> `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support ...`
> （`onednn_instruction.cc`），关掉后走通用 CPU 路径才能跑通；另外 PaddleOCR 3.x
> 已移除 `show_log` 参数、`use_angle_cls` 改名为 `use_textline_orientation`。
> ② PaddleOCR 3.x 的返回结构变了——文本在同级的 `rec_texts` 里
> （`[{'rec_texts': [...], 'rec_scores': [...]}, ...]`），2.x 的 `[文本, 置信度]`
> 嵌套结构已经没有了。

### 生成结果校验（防幻觉引用）
大模型有时会「顺手」写出一条读起来很像真的、知识库里根本没有的法条号。
生成之后再过一道关：把回答里的「法律名 + 条文号」抽出来，逐条到 MySQL 的
`law_index` 核对，核对不上的在回答末尾附提示，并在响应里以 `citation_check`
字段返回明细：

```
提示：以上回答引用的《中华人民共和国刑法》第九百九十九条未能在本系统知识库中
核对到，请以现行法律法规原文为准。
```

实测：「酒驾撞人要判多久」核对 4 条全部命中，「什么情况下可以判缓刑」核对 6 条全部命中。
`ENABLE_CITATION_CHECK=false` 可关闭。

### 分块策略
知识库 **99.9% 的记录不需要分块**——法条按「条」、问答按「一对」，
本身就是不可再分的语义单元（实测律师 0/4555、心理 0/4675、金融 9/621 条超长）。
只有 PDF 导入的连续正文需要切分，用**语义分块**：算相邻句的向量相似度，
在相似度低谷（话题转折处）下刀。MinerU 产出的是**整篇 Markdown**，
`extract_pages()` 因此返回单元素列表 `[{page: 1, text: Markdown}]`，整篇交给
语义分块——这比按页拆更好，MinerU 的价值就是连贯的阅读顺序，按页重拆会把
跨页表格与段落切断（走兜底路径时仍逐页返回、逐页切块）。对比固定长度切分：

```
固定 900 字：块1 结尾「…提交」→ 块2 开头「商业使用」   ← 把句子拦腰截断
语义分块：  5 块中 4 块结尾落在完整句号上
```

### LangChain 集成
六大功能逐项对应到现有实现，而非推倒重来：

| 功能 | 落地方式 |
|---|---|
| Models | `DeepSeekChatModel`（自定义 `BaseChatModel`，保留推理模型空回复补偿） |
| Prompts | `ChatPromptTemplate` + `MessagesPlaceholder` |
| Chains | LCEL：`提示词 \| 模型 \| StrOutputParser`，同步与流式共用一条链 |
| Memory | `RedisChatHistory` 实现 LangChain 消息历史接口，后端仍是项目的 Redis |
| Indexes | 保留自研检索 |
| Agents | 保留自研路由 |

> **为什么 Indexes 与 Agents 不套框架**：本项目的检索是「多查询 × 多路召回
> （稠密 + 稀疏 + 法条元数据）→ 加权 RRF → 交叉编码器精排」，
> LangChain 的 `EnsembleRetriever` 只支持多检索器等权融合，表达不了元数据路与权重；
> Agents 的 tool-calling 在这里就是一次确定性规则路由（识别到法条号才走元数据路），
> 套上 Agent 只会引入不确定性。取舍写在 `app/core/chain_service.py` 的模块注释里。

## 知识库

共 **13364** 个切片，三个角色均衡在 4000 条以上：

| 角色 | 切片数 | 数据来源 |
|---|---|---|
| psychologist | 4675 | MentalGLM：认知歪曲识别 3407、自杀风险识别 1249；WHO《世界精神卫生报告》22（含图表内容） |
| lawyer | 4555 | **现行法条全文 4068 条**（lawtext/laws，源自国家法律法规数据库）+ 刑法罪名条目 483 + 法律问答 2 + 裁判文书 2 |
| financial_advisor | 4134 | DISC-FinLLM：金融咨询/计算/检索问答 + 286 篇金融材料；**Fin-Eva**：1281 条保险条款/产品分析/事件解读 + 2250 条金融问答与从业资格题 |

原始数据来自四个 GitHub 开源项目，经 `scripts/prepare_dataset.py`（开源语料）与
`scripts/fetch_statutes.py`（法条）清洗为入库友好的 JSONL
（`embed_text` 作检索目标、`display_text` 喂给大模型）。

**法条数据集**覆盖 21 部法规，只取「有效 / 尚未生效」状态，整部纳入，按「条」切分
（含 53 条「第X条之一」增补条款）：

| 法域 | 法规 |
|---|---|
| 民事 | 民法典（1254 条）、民事诉讼法、民事诉讼证据规定 |
| 刑事 | 刑法、刑事诉讼法 |
| 商事 | 公司法 |
| 劳动 | 劳动法、劳动合同法、劳动争议调解仲裁法、劳动争议解释（一）、社会保险法 |
| 行政 | 行政处罚法、行政诉讼法、治安管理处罚法、道路交通安全法 |
| 消费者 | 消费者权益保护法 |
| 宪法 | 宪法（2018 年修正文本） |
| 司法解释 | 民法典婚姻家庭编 / 继承编 / 担保制度 / 合同编通则解释 |

> **已知局限**：法条覆盖 21 部，而数据源中现行有效法规共 1517 部；且**没有判例库**
> （仅 2 篇裁判文书样例），无法检索类案。扩围只需在 `scripts/fetch_statutes.py` 的
> `PRIORITY` 列表追加法规名并提高 `TARGET`。

## 常用脚本

```bash
.venv/bin/python -m scripts.init_db                    # 初始化数据库与角色
.venv/bin/python -m scripts.knowledge_base             # 全量重建知识库
.venv/bin/python -m scripts.knowledge_base lawyer      # 只重建指定角色
.venv/bin/python -m scripts.knowledge_base --keep      # 保留旧数据，追加
.venv/bin/python -m scripts.knowledge_base --clean     # 清理遗留集合
.venv/bin/python -m scripts.knowledge_base --stats     # 查看知识库统计

# 知识库去重（全量重建后建议跑一次）
.venv/bin/python -m scripts.dedup_kb                   # 只统计，不动数据
.venv/bin/python -m scripts.dedup_kb --apply           # 执行清理

# 重新拉取法条（法规修订后）
git clone --depth 1 https://github.com/lawtext/laws /tmp/laws
.venv/bin/python -m scripts.fetch_statutes --src /tmp/laws/content
.venv/bin/python -m scripts.knowledge_base lawyer
```

## 安全提醒

`.env` 中的 DeepSeek API Key 与数据库密码为明文。若要分享或公开本项目，
请先轮换 Key 并改用环境变量注入。
