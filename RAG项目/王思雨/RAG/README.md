# RAG 电力维修问答助手

基于 RAG（检索增强生成）的电力维修问答系统：把电力行业标准 PDF 切块入库，
用「混合检索 + 多路召回 + 重排」找依据，交给大模型生成**有出处、可追溯**的答案。

## 一、项目简介

电力检修人员查标准，痛点是「知道要查什么，但翻不到那一条」。
本项目把 GB/T 系列标准 PDF 解析、切块、向量化后存进 Milvus，
提问时先检索出最相关的条款片段，再让大模型**只依据这些片段**作答，
并在回答下方列出引用来源（文件名 + 相关度分数）。

面向的教学/演示目标：完整走一遍 RAG 的离线入库与在线问答全链路，
覆盖解析、分块、向量化、混合检索、重排、提示词、流式生成、多轮记忆、
多用户多角色、缓存优化、前端页面与 RAGAS 评测。

## 二、功能特性

| 功能 | 说明 |
|---|---|
| 资料入库 | PDF 解析（含扫描页 OCR）→ 语义分块 → 离线增强 → BGE-m3 向量化 → Milvus 入库 |
| 智能问答 | 查询改写 → 混合检索（稠密 + 稀疏）→ 多路召回 → RRF 融合 → 交叉编码器精排 → 流式生成 |
| 引用来源 | 每条回答附带来源文件名、片段摘要与相关度分数，前端可折叠查看 |
| 多轮记忆 | Redis 短期记忆（最近 N 条）+ Milvus 长期记忆（按语义召回），追问不断片 |
| 多用户隔离 | 用户 / 角色 / 会话 / 消息四张表，记忆与缓存按 `user_id` 精确隔离 |
| 多角色 | 电力维修专家、通用助手两套系统提示词，回答边界与安全提醒规则不同 |
| 答案缓存 | 按「用户 + 角色 + 归一化问题」建键，命中时跳过检索与生成，但**照常写回记忆** |
| 知识库管理 | 查看集合条数与更新时间、后台重建知识库、按用户清空缓存 |
| 效果评测 | RAGAS 四指标（忠实度 / 答案相关度 / 上下文精确率 / 上下文召回率），结果前端可视化 |
| PDF 解析对比（可选） | 第 12 步补跑了 MinerU 并与 PyMuPDF + pdfplumber 逐文档对比；**结论是 MinerU 会丢数字，不接入主链路**，只保留 `python -m ingest --compare-mineru` 作为对比工具，详见设计文档 11.5 |
| 前端页面 | 登录注册、角色选择、问答、历史、知识库、上传、评测、后台、用户中心共 9 个页面 |

## 三、技术栈

| 层次 | 技术选型 | 版本 / 说明 |
|---|---|---|
| 大模型 | DeepSeek `deepseek-v4-flash` | 走 OpenAI 兼容接口，支持流式 |
| 向量模型 | BGE-m3 | 本地加载，同时产出稠密（1024 维）与稀疏向量 |
| 重排模型 | BGE-reranker-v2-m3 | 本地交叉编码器，对候选做精排 |
| 向量库 | Milvus | standalone 容器，集合 `power_repair_docs` |
| 关系库 | MySQL | 4 张表：users / roles / conversations / messages |
| 缓存与记忆 | Redis | 短期记忆 `mem:short:{user_id}` + 答案缓存 `cache:rag:answer:*` |
| 后端框架 | FastAPI + Uvicorn | 16 个 HTTP 接口，自带 `/docs` 交互文档 |
| 前端框架 | Vue 3 + Vite + Element Plus | 组合式 API + vue-router，教学版不用状态管理库（存 localStorage） |
| PDF 解析 | PyMuPDF / pdfplumber / MinerU / PaddleOCR | 分工与选型理由见设计文档 11.1 |
| 评测框架 | RAGAS | 0.4.3，四指标；评估 LLM 用 DeepSeek，评估 Embedding 用 BGE-m3 |
| 部署环境 | WSL + Ubuntu + Conda | 三个脚本：install / run / shutdown |

## 四、目录结构

```
RAG电力维修问答助手/
├── .env.example              # 配置模板（39 项，每项带中文注释），复制为 .env 后填写
├── requirements.txt          # Python 依赖清单
├── README.md                 # 本文件
├── config.py                 # 配置读取（统一从 .env 读，代码里不写死密钥）
├── logger.py                 # 日志配置（同时输出控制台与 logs/app.log）
├── schemas.py                # 请求/响应数据模型（Pydantic）
├── app.py                    # FastAPI 应用入口，16 个接口
│
├── ingest.py                 # PDF 解析（PyMuPDF / pdfplumber / MinerU / PaddleOCR）
├── ingest_chunk.py           # 语义切分
├── enrich.py                 # 离线数据增强（摘要、低质过滤、去重、父子块）
├── vector_store.py           # BGE-m3 向量化 + Milvus 建库与入库
├── retrieval.py              # 混合检索 + 多路召回 + RRF 融合 + BGE-reranker 精排
├── prompt.py                 # 系统提示词模板（专家 / 通用两套）
├── postprocess.py            # 生成结果后处理与正则清洗
├── optimize.py               # 查询改写、查询扩写、Redis 答案缓存
├── memory.py                 # 短期记忆（Redis）+ 长期记忆（Milvus）
├── rag.py                    # 问答主流程：检索 → 生成 → 写回
├── db.py                     # MySQL / Redis 连接与建表
├── db_user.py                # 用户、角色、会话、消息的数据访问
├── cli.py                    # 命令行演示入口
├── eval_ragas.py             # RAGAS 评测：15 条评测集 → 四指标 → 结果落盘
│
├── data/                     # 运行期数据产物
│   ├── parsed_chunks.json    # PDF 解析与切分结果
│   ├── enriched_chunks.json  # 增强后的子块（入库用）
│   ├── eval_dataset.json     # RAGAS 评测集（15 条，参考答案人工摘录）
│   └── ragas_result.json     # RAGAS 评测结果（summary + details）
├── docs/                     # 文档
│   ├── 需求规格说明书.md
│   ├── 设计文档.md
│   ├── 接口文档.md
│   ├── 流程图.md
│   ├── 业务规则.md
│   ├── 思维导图.md
│   └── 测试成果.md
├── scripts/                  # 部署脚本
│   ├── install.sh            # 环境安装（系统依赖 / conda / Redis / MySQL / Milvus 提示）
│   ├── run.sh                # 后台启动前后端，PID 写入 logs/
│   └── shutdown.sh           # 按 PID 停止，先 SIGTERM 后 SIGKILL
├── tests/                    # 测试
│   ├── test_unit.py          # pytest 单元测试（59 条）
│   └── postman_collection.json   # Postman 接口测试集合
├── logs/                     # 运行日志（app.log / backend.log / frontend.log）
└── frontend/                 # 前端工程
    ├── index.html
    ├── package.json
    ├── vite.config.js        # /api 代理到 8000，端口 5173
    └── src/
        ├── main.js           # 入口：注册路由 + Element Plus + 图标
        ├── router.js         # 路由表与登录守卫
        ├── api.js            # 接口封装（含手写 SSE 解析）
        ├── App.vue           # 根组件：顶部导航 + 路由出口
        └── views/            # 9 个页面
            ├── Login.vue        # 登录 / 注册
            ├── RoleSelect.vue   # 角色选择
            ├── Chat.vue         # 智能问答（流式 + 引用来源）
            ├── History.vue      # 历史记录
            ├── KbManage.vue     # 知识库管理
            ├── UploadStatus.vue # 资料上传
            ├── Eval.vue         # 评测结果
            ├── Admin.vue        # 后台状态（只读）
            └── User.vue         # 用户中心
```

## 五、快速开始

### 5.1 环境要求

| 组件 | 要求 | 说明 |
|---|---|---|
| 操作系统 | Ubuntu 22.04+ / Debian / CentOS（或 WSL2 里的 Ubuntu） | `install.sh` 识别发行版并选对应包管理器 |
| Python | 3.10 | 由 conda 环境 `power_rag` 提供 |
| Conda | Miniconda 即可 | 未安装时 `install.sh` 自动装 |
| Node.js | 18+（实测 24 可用） | 前端 Vite 需要 |
| Docker | 用于跑 Milvus | `install.sh` 只给命令，不自动装 Docker |
| 内存 | 建议 16 GB 以上 | BGE-m3 与 reranker 两个本地模型同时驻留 |
| 磁盘 | 建议 20 GB 以上空闲 | 两个本地模型 + Milvus 容器 |

### 5.2 安装

```bash
bash scripts/install.sh              # 真安装
bash scripts/install.sh --dry-run    # 只看会执行哪些命令，不实际安装
```

装完脚本会提示 MySQL 手动建库（脚本不替用户决定密码）：

```bash
mysql -u root -p -e "CREATE DATABASE rag CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
```

以及 Milvus 的启动命令（需先自备 Docker）：

```bash
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.6.2 standalone
```

### 5.3 配置

```bash
cp .env.example .env
vi .env
```

至少要填：`LLM_API_KEY`（大模型 Key）、`MYSQL_HOST` / `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DB`、
`REDIS_HOST`、`MILVUS_HOST`，以及两个本地模型路径 `BGE_M3_PATH`、`RERANKER_PATH`。
`.env.example` 里 39 个配置项都带中文注释，逐项对着填即可。

### 5.4 首次灌数据

```bash
conda activate power_rag
python -m ingest                     # 解析 PDF 并分块，产出 data/parsed_chunks.json
python -m enrich                     # 离线增强，产出 data/enriched_chunks.json
python -m vector_store --rebuild     # 向量化并写入 Milvus
```

### 5.5 启动

```bash
bash scripts/run.sh                  # 后台起前后端，PID 写入 logs/
bash scripts/shutdown.sh             # 停止
```

### 5.6 访问

| 服务 | 地址 |
|---|---|
| 前端页面 | http://localhost:5173 |
| 后端服务 | http://localhost:8000 |
| API 交互文档 | http://localhost:8000/docs |
| 接口文档（Markdown） | `docs/接口文档.md` |

首次打开前端会跳转到登录页，注册一个账号即可开始使用。

## 六、常用命令

```bash
# 依赖与环境
bash scripts/install.sh              # 安装环境（--dry-run 只看命令）
bash scripts/run.sh                  # 启动前后端
bash scripts/shutdown.sh             # 停止前后端

# 知识库
python -m ingest                     # 解析 PDF 并语义分块
python -m enrich                     # 离线增强（摘要 / 低质过滤 / 去重 / 父子块）
python -m vector_store --rebuild     # 用增强后的子块重建 Milvus 集合
curl http://localhost:8000/kb/status # 查看集合条数与更新时间

# 问答（命令行演示）
python -m rag "绝缘油气相色谱分析步骤是什么" "那具体操作步骤呢"

# RAGAS 评测（15 条评测集，约 15 分钟）
python -m eval_ragas                 # 结果写入 data/ragas_result.json，前端 /eval 页面可看

# MinerU 对比（可选工具，约 2~4 分钟；不进主链路）
python -m ingest --compare-mineru    # 结果写入 data/mineru_compare.json

# 测试
python -m pytest tests/test_unit.py -q          # 单元测试（59 条）
python -m pytest tests/test_unit.py -q -k api   # 只跑接口相关用例
```

## 七、项目步骤日志

| 步骤 | 一句话概括 |
|---|---|
| 0 | 项目骨架：配置、日志、依赖清单、文档与脚本占位 |
| 1 | MySQL / Redis / Milvus 三个基础连接的封装与建表 |
| 2 | PDF 解析（含 OCR）与语义切分 |
| 3 | BGE-m3 向量化 + Milvus 集合建立与入库 |
| 4 | 混合检索 + 多路召回 + RRF 融合 + BGE-reranker 精排 |
| 5 | 系统提示词 + DeepSeek 调用 + 流式输出 + 后处理校验 |
| 6 | 多轮记忆（短期 Redis / 长期 Milvus）+ 多用户 + 多角色 |
| 7 | RAG 优化：查询改写、查询扩写、答案缓存 |
| 8 | FastAPI 后端，15 个 HTTP 接口 + 流式 SSE |
| 8.5 | 修复长期记忆干扰精排、缓存键不稳定、缓存误删全体用户；lifespan 迁移 |
| 9 | Vue3 + Vite + Element Plus 前端，9 个页面 + 路由 + 接口封装 |
| 9.5 | 修复 save_turn 的 `or` 短路（助手回答全丢）；新增 `/cache/stats` |
| 9.6 | 修复删会话不级联删消息；清理 11 个半截会话；Admin 展示缓存分组 |
| 9.7 | 修复缓存命中时不写记忆（多轮断片） |
| 10 | RAGAS 评测：15 条评测集 + 四指标 + 结果可视化 |
| 11 | 部署脚本（install / run / shutdown）+ 文档收尾 + 交付物核对 |
| 12 | 补跑 MinerU 并与 PyMuPDF 对比（结论：不接入主链路，只作可选对比工具） |

> 每一步的详细记录（bug 根因、修复方式、实测数据）见文末「附录 A：分步开发记录」。

## 八、已知局限

各局限的成因、实测数据与改进方向，详见 **`docs/设计文档.md` 第 16.9 节「历次暴露问题的修复」**
（16.9.1 – 16.9.8 共 8 条）与第 17.6 节「RAGAS 评测的已知局限」。

| 类别 | 局限 |
|---|---|
| 评测 | 评测集只有 15 条；`ground_truth` 由人工摘录；RAGAS 的 LLM 评判本身有随机性，分数只看量级 |
| 生成 | `faithfulness` 偏低的主因是回答含大量上下文之外的引申与安全提醒（提示词风格与指标口径的张力） |
| 检索 | 语料里混有封面 / 目次 / 前言页，会挤进 top 5，压低上下文精确率 |
| 评测覆盖 | 只做单轮评测，多轮记忆能力未被覆盖；评测时按设计关掉长期记忆，与线上多轮场景有差异 |
| 鉴权 | 教学版：`/auth/login` 返回 token 但各接口**不校验**，`X-User-Id` 请求头同样不被校验 |
| 前端状态 | 用 `localStorage` 存登录态（生产应换 Pinia + HttpOnly Cookie）；不做路由级权限 |
| 历史来源 | `messages` 表不存 `sources`，历史页只能看到问答正文，来源仅在当次对话内可见 |
| 结构化输出 | 答案靠提示词约束格式，不保证 100% 稳定 |
| PDF 解析 | 主链路用 PyMuPDF + pdfplumber。已知局限：PyMuPDF 对表格的还原较弱（表头会被压成一行、行列关系丢失），但数字与章节号完整；MinerU 表格更强却会丢数字（最差只保留 52%），两害相权取了前者 |

---

## 附录 A：分步开发记录

> 以下是第 0–10 步逐步开发时留下的记录，含当时的 bug 根因分析、
> 修复前后的实测对照与验证证据，作为正文的补充材料保留。
> 各节标题沿用了当时的「二之X」编号。

### 二之一、日志说明
程序运行时会在项目根目录自动创建 `logs/` 目录，并把日志写入 `logs/app.log`，
该目录由 `logger.py` 在首次写日志时自动生成，不需要手工创建，也不需要提交到版本库。

### 二之二、文档解析与分块说明

### 解析工具分工
| 工具 | 负责哪一类解析 | 为什么选它 |
|---|---|---|
| MinerU | 复杂版面、多栏、公式、图表标题 | 版面分析能力强，能还原阅读顺序，标准类 PDF 多栏排版友好 |
| PyMuPDF（fitz） | 正文文本层 | 速度最快，逐页取文本稳定，作为主解析通道 |
| pdfplumber | 表格 | 按坐标切格子，能还原表格的行列结构，PyMuPDF 做不到 |
| PaddleOCR | 扫描件、图片文字 | 标准 PDF 常有纯图片页（本批 5 个文件就有 18 页无文本层），需 OCR 补 |
| PyMuPDF 规则清洗 | 去水印、去页眉页脚 | 规则可解释、可复现，不额外引入模型依赖 |

### 分块方式
本步**只实现了语义切分**：先按段落切，段落超长再按句子切，然后合并过短块，单块不超过 500 字。
固定长度分块、句子分块、段落分块、标题分块这四种**没有实现**，它们的区别与优缺点记录在
`docs/设计文档.md` 的「文档解析与分块」一节。

### 解析产物
`python ingest.py` 会解析 `PDF_DIR` 下指定的 5 个 GB/T 标准文件，输出 `data/parsed_chunks.json`，
包含每个 PDF 的页数、逐页文本、表格和语义块，供第 3 步入库使用。

### 二之三、向量化与入库说明

| 项目 | 值 |
|---|---|
| 向量模型 | BGE-M3 |
| 模型路径 | `D:\模型\bge-m3`（由 `.env` 的 `BGE_M3_PATH` 指定） |
| 向量维度 | 1024 维 dense 向量 |
| 运行设备 | 优先 CUDA，无显卡自动回退 CPU |
| 向量库 | Milvus |
| 集合名 | `power_repair_docs`（由 `.env` 的 `MILVUS_COLLECTION` 指定） |
| 入库块数 | 原始 894 块，去重后 **880 块**，入库 880 行 |

入库命令：

```bash
python ingest.py                    # 第一步：解析 PDF，生成 data/parsed_chunks.json
python -m vector_store --rebuild    # 第二步：向量化并写入 Milvus（--rebuild 表示重建集合）
```

`--rebuild` 会先删除 `power_repair_docs` 集合再重建，保证每次入库前数据干净；
不加该参数则直接往已有集合里追加。

**OCR 模型路径必须放在纯英文目录**：PaddleOCR 的 C++ 推理层打不开含中文的路径，
默认模型目录 `C:\Users\雨子\.paddleocr` 会导致「模型文件找不到」。
因此模型已复制到 `C:/paddleocr_models`，由 `.env` 的 `OCR_MODEL_DIR` 指定。

### 二之四、检索与重排说明

### 检索链路

```
用户问题
  ├─ 路 1：Milvus 混合检索
  │    ├─ dense 向量检索（COSINE，取 top 2K）
  │    └─ sparse 向量检索（IP，取 top 2K）
  │    └─ RRF 融合 → 取 top 20
  ├─ 路 2：MySQL 历史对话模糊匹配 → 取 top 10
  ├─ 合并去重 → 30 条
  └─ BGE-reranker-v2-m3 精排 → 过滤 score < 0.3 → 取 top 5
```

| 配置项 | 值 | 说明 |
|---|---|---|
| 混合检索 | dense + sparse | 稠密语义 + 稀疏关键词，互补 |
| 稀疏方案 | **BGE-m3 lexical_weights** | 学习式稀疏检索 |
| 融合方式 | RRF | `score = Σ 1/(k + rank)`，k=60 |
| 多路召回 | Milvus + MySQL | MySQL 那路查历史提问 |
| 重排模型 | BGE-reranker-v2-m3 | 路径 `D:\模型\bge-reranker-v2-m3` |
| 得分过滤阈值 | 0.3 | reranker 输出 logits，sigmoid 转 0~1 后比较 |

命令：`python -m retrieval "绝缘油气相色谱分析步骤是什么"`

### 为什么用 BGE-m3 sparse，而不是 Milvus 内置 BM25

| 原因 | 说明 |
|---|---|
| 客户端不支持 | 本机 pymilvus 是 **2.4.9**，`Function` / `FunctionType` 导不进来，`schema.add_function` 不存在，表达不了内置 BM25 语法 |
| 中文分词是内置 BM25 的软肋 | 内置 BM25 走标准分析器，中文按空格/标点切，`绝缘油气相色谱分析步骤` 会被切成一个整 token；官方建议另配 jieba 分析器 |
| BGE-m3 sparse 中文开箱即用 | `lexical_weights` 是学习出来的稀疏权重，中文无需额外分词配置 |
| 不动环境 | 升级 pymilvus 会影响本机其他 11 个共享该 Milvus 的项目，风险不值 |
| 一个模型出两种向量 | dense 与 sparse 同源，不用再挂分词器，架构更简洁 |

技术选型里的「支持混合检索、BM25」由 BGE-m3 sparse 落实：它属于**学习式稀疏检索**，
在中文上优于字面 BM25，方向一致。

### 二之五、提示词与生成说明

| 项目 | 值 |
|---|---|
| 大模型 | DeepSeek API，模型 `deepseek-v4-flash` |
| 接入方式 | OpenAI 兼容 SDK，`base_url=https://api.deepseek.com/v1` |
| 生成温度 | 0.3 |
| 最大生成 | 2000 token |
| 流式输出 | 支持 |
| 提示词文件 | `prompt.py`（`SYSTEM_PROMPT` + `USER_PROMPT_TEMPLATE`） |

### 提示词设计
`SYSTEM_PROMPT` 把模型设定为**电力维修专家**（擅长变压器、绝缘油、SF6、分接开关），并明确四件事：

- **性格**：严谨务实，只讲有依据的结论，不确定就说不确定
- **口吻**：专业但易懂，用工程语言，不用网络流行语，不闲聊
- **回答边界**：只答电力维修/检测/标准问题；涉及人身安全必须提醒先断电验电挂接地线；
  知识库没依据时明说「标准文档中没有找到相关条款」**且不许编造标准编号与参数**；
  不给未经标准验证的偏方
- **回答格式**：先结论 → 再依据（引用标准编号）→ 必要时分步骤 → 末尾列引用清单

`USER_PROMPT_TEMPLATE` 含 `{context}` 与 `{question}` 两个占位符，
context 由检索结果拼成 `[序号] 来源：xxx | 内容：xxx` 的形式。

### 后处理
正则清洗：去 markdown 代码块标记 → 全角空格转半角 → 去行尾空格 →
去中文标点后多余空格 → 连续 3 个以上空行合并为 2 个 → 超过 8000 字截断。

> **注意**：清理「中文标点后空格」用的是 `[ \t]+` 而**不是 `\s+`**。
> `\s` 在 Python 中包含 `\n`，用它会连段落换行一起吃掉，把整篇答案压成一行。

### 校验
`validate_answer()` 检查四项：非空、不超过 8000 字、不含违规表述
（「绝对安全」「保证不出问题」）、是否声明了无依据（标记 `is_no_evidence`）。

### 命令
```bash
python -m rag "绝缘油气相色谱分析步骤是什么"
python -m rag "变压器漏油怎么处理" --top_k 3
```

### 二之六、记忆与多用户多角色说明

大模型本身**没有记忆**：每次请求都是独立的，上一次说过什么它不知道。
记忆必须由外部系统提供，本项目用两个存储分别承担短期与长期记忆。

| 类型 | 存储 | 键 / 集合 | 保留策略 | 作用 |
|---|---|---|---|---|
| 短期记忆 | Redis list | `mem:short:{user_id}` | 最近 `LLM_HISTORY_LIMIT`(10) 条，1 天过期 | 让模型记住刚才几轮说了什么 |
| 长期记忆 | Milvus | `power_repair_memory` | 永久，按 `user_id` 过滤 | 跨会话沉淀问答，下次问相似问题能复用 |
| 用户/角色/会话/消息 | MySQL | `users`/`roles`/`conversations`/`messages` | 永久 | 账号、角色、会话归档 |

### 多用户隔离
`user_id` 同时是 **Redis 键前缀**（`mem:short:5`）和 **Milvus 过滤字段**（`user_id == 5`），
两处都做隔离，且 Milvus 检索结果还会二次核对 `user_id` 归属。

### 多角色
由 `roles` 表驱动，预置两条：`power_repair_expert`（电力维修专家）、`general_assistant`（通用助手）。
角色当前只作为会话归属记录，尚未切换提示词。

### 密码哈希
`sha256(盐值 + 密码)`，盐值从 `.env` 的 `PASSWORD_SALT` 读。
**这是教学简化版**：固定盐值意味着相同密码产生相同哈希，且 SHA256 计算太快、抗暴力破解弱。
生产应用必须换成 **bcrypt / argon2** 这类自带随机盐与慢哈希的算法。

### 命令
```bash
# 两轮问答（第二轮自动带上第一轮的历史）
python -m rag "绝缘油气相色谱分析步骤是什么" "那具体操作步骤呢"
# 指定用户、角色、会话
python -m rag "变压器漏油怎么处理" --user-id 5 --role-id 1 --conversation-id 3
```

### 二之七、RAG 优化说明

| 优化项 | 位置 | 做法 | 效果 |
|---|---|---|---|
| 查询改写 | `optimize.rewrite_query` | 用 LLM 把历史 + 追问合并成自足问题 | 解决模糊追问检索不到内容 |
| 查询扩写 | `optimize.expand_query` | 生成 n 个同义问题 | 提高召回（**本步未接入主链路**） |
| 答案缓存 | `optimize.cache_get/set` | Redis，键 `cache:rag:answer:{角色}:{改写后问题}` | 相同问题免重复生成 |
| 摘要 | `enrich.generate_summary` | 规则版：首句 + 高频关键句 | 不调 LLM，零成本 |
| 语义去重 | `enrich.dedup_chunks` | BGE-m3 余弦相似度 > 0.95 只留一条 | 去掉近似重复块 |
| 删低质 | `enrich.remove_low_quality` | 长度、空白占比、数字占比三规则 | 去掉碎片与表格噪声 |
| 父子块 | `enrich.build_parent_child` | 子块入库，父块回溯 | 检索命中子块时可取回完整上下文 |
| 按角色切换提示词 | `prompt.get_system_prompt` | `role_name → prompt` 映射 | 补第 6 步遗留 |

### 离线增强与入库
```bash
python -m enrich                     # 生成 data/enriched_chunks.json
python -m vector_store --rebuild     # 用增强后的子块重建 Milvus 集合
```

增强效果：`894 块 →（删低质丢弃 80）→ 814 →（语义去重丢弃 90）→ 724 子块 + 242 父块`，
**Milvus 只入 724 个子块**，父块不入库、仅用于检索时回溯上下文。

### 查询改写实测效果
```
追问「那具体操作步骤呢」
  改写前：召回 21 条 → 精排命中 1 条（且那条是 MySQL 历史提问，不是知识）
  改写后：改写为「绝缘油气相色谱分析标准第9章中仪器准备、取气与检测条件的具体操作步骤是什么？」
          召回 20 条 → 精排命中 5 条，全部为 GBT 17623-2026 的真实条款
```

### ⚠️ 重要：模型是推理模型，`max_tokens` 给小了会静默返回空
`deepseek-v4-flash` 是**推理模型**，`max_tokens` **包含思维链消耗**。
实测改写任务光推理就要烧掉约 390 token，若按常规给 `max_tokens=128`，
返回的 `content` 是**空字符串**——不报错、不抛异常，只是静默失败。
因此 `optimize.py` 里的 `REWRITE_MAX_TOKENS=512`、`EXPAND_MAX_TOKENS=1024` 不能调小。

### 二之八、HTTP 服务说明

### 启动方式
```bash
uvicorn app:app --reload --port 8000
```
监听地址与端口由 `.env` 的 `API_HOST` / `API_PORT` 控制，跨域白名单由 `API_CORS_ORIGINS` 控制。

### 路由清单（15 个）
| # | 方法 | 路径 | 说明 |
|---|---|---|---|
| 1 | GET | `/health` | 健康检查，返回 Milvus / MySQL / Redis 三项状态 |
| 2 | POST | `/auth/register` | 注册，用户名已存在返回 400 |
| 3 | POST | `/auth/login` | 登录，失败返回 401 |
| 4 | GET | `/user/{user_id}` | 查用户，不存在返回 404 |
| 5 | GET | `/roles` | 角色列表 |
| 6 | POST | `/conversation` | 创建会话 |
| 7 | GET | `/conversation/{user_id}` | 会话列表 |
| 8 | DELETE | `/conversation/{id}?user_id=` | 删除会话（只能删自己的） |
| 9 | GET | `/message/{conversation_id}?limit=` | 消息列表 |
| 10 | POST | `/chat` | 问答（`stream=true` 走 SSE） |
| 11 | POST | `/kb/upload` | 上传 PDF 并解析（不入库） |
| 12 | POST | `/kb/rebuild` | 后台重建知识库，立即返回 |
| 13 | GET | `/kb/status` | 知识库状态 |
| 14 | GET | `/eval/result` | 评测结果（第 10 步产出） |
| 15 | GET | `/cache/clear?user_id=` | 清空该用户的短期记忆与答案缓存 |

### 流式接口格式（SSE）
`POST /chat` 传 `{"stream": true}` 时返回 `text/event-stream`：

```
data: <正文片段>\n\n
data: <正文片段>\n\n
...
data: {"sources": [...], "conversation_id": 1, "long_memory_used": 2}\n\n
data: [DONE]\n\n
```

**每帧以空行（`\n\n`）结尾**，最后以 `data: [DONE]` 收尾。
出错时推送 `data: {"error": "..."}` 帧，之后**仍然推送 `[DONE]`**，保证前端循环能正常退出。

### 本步同时补的第 7 步遗留
| 遗留项 | 处理 |
|---|---|
| 父子块回溯没接上 | `retrieval.attach_parent_text` 给命中子块附加父块全文，`rag.build_messages` 拼进 context |
| MySQL 召回压过知识 | `config.MYSQL_RECALL_WEIGHT=0.5`，`retrieval.fuse_multi_route` 按路加权融合 |

### 二之九、第 8.5 步修复说明

### ① 长期记忆不再干扰精排
**问题**：长期记忆里存的就是「问题+答案」，检索 query 又是同一个问题，
reranker 给它接近满分（0.9742），把真实标准条款（0.4382）挤到后面，还出现 3 条重复。

**修复**：长期记忆**完全不参与 reranker**。
- `contexts` 只由 `retrieval.retrieve` 产生（Milvus + MySQL 精排后结果）
- 长期记忆走 `rag.collect_long_memory` 单独取，按问题去重（同问题只留最高分）
- `prompt.USER_PROMPT_TEMPLATE` 增加 `{long_memory}` 占位符，
  在标准条款之后单独成段，段内明确写「优先级低于标准条款，冲突时以标准条款为准」
- 没有长期记忆时该段整体不出现

**效果**：sources 里不再出现 `source=历史对话记忆` 的条目；
而提示词里仍有该段（两处需求都满足）。

### ② 缓存键改用原始 query 归一化
**问题**：缓存键是 `answer:{角色}:{改写后问题}`，而改写是 LLM 输出、不稳定——
同一个问题 4 次改写得到 4 个语义相同、文本不同的串，缓存几乎不命中。

**修复**：
- 新增 `optimize.normalize_query()`：去中英文标点、合并空白、转小写
- 缓存键改为 `cache:rag:answer:{user_id}:{role_name}:{归一化原始问题}`
- 用**原始 query**（用户原话）而不是改写结果；键里加 `user_id` 实现用户隔离

**效果**：`变压器漏油怎么处理`、`变压器漏油怎么处理？`、`变压器漏油，怎么处理！`、
` 变压器漏油怎么处理。 ` 四种问法生成**同一个键**，第 2 次起即命中缓存。

### ③ /cache/clear 只清该用户
**修复**：用 Redis `SCAN`（不用 `KEYS`，避免阻塞）匹配 `cache:rag:answer:{user_id}:*`，
逐个删除；同时清该用户的短期记忆 `mem:short:{user_id}`。
返回结构变为 `{"msg", "cache_keys", "short_memory"}`。

> 注意：`MsgResp` 只有 `msg` 字段，而本接口要额外返回 `cache_keys` 与 `short_memory`。
> 第 8.5 步因不允许改 `schemas.py`，该路由暂时**不挂 response_model**；
> 第 9 步新增了 `schemas.ClearCacheResp`，该路由已改为 `response_model=schemas.ClearCacheResp`。

### ④ on_event 迁移到 lifespan
**修复**：`@app.on_event("startup")` 已废弃，改用 `@asynccontextmanager` 的 `lifespan`，
由 `FastAPI(lifespan=lifespan)` 挂载。启动初始化逻辑不变，**DeprecationWarning 消失**。

### ⑤ MySQL 召回降权说明
`fuse_multi_route` 上加了注释：**权重只影响 RRF 融合阶段，最终排序由 rerank 决定**。
MySQL 路降权后若仍能进精排，说明它与 query 相关度确实高；
若发现它仍压过知识条款，需要在 rerank 前过滤掉 `source=mysql` 的结果。

---

### 二之十、第 9 步：前端说明

### 启动方式

```bash
# 1) 先启动后端（端口 8000）
python -m uvicorn app:app --reload --port 8000

# 2) 再启动前端（端口 5173）
cd frontend && npm install && npm run dev
```

浏览器打开 <http://localhost:5173/>，会自动跳转到登录页。

### 端口与代理

| 项目 | 值 |
|---|---|
| 前端端口 | `5173`（`vite.config.js` 的 `server.port`） |
| 后端地址 | `http://127.0.0.1:8000` |
| 代理规则 | 前端以 `/api` 开头的请求全部转发到后端，并去掉 `/api` 前缀 |
| 路径别名 | `@` → `frontend/src` |

前端代码里统一写 `baseURL = '/api'`，例如 `api.kbStatus()` 实际请求 `/api/kb/status`，
经 Vite 代理后到达后端的 `/kb/status`。这样浏览器侧不存在跨域，后端 CORS 白名单也不必改动。

### 页面清单

| 路由 | 文件 | 功能 |
|---|---|---|
| `/login` | `views/Login.vue` | 登录 / 注册（两个页签），成功后写 `user_id`、`username` 并跳 `/role` |
| `/role` | `views/RoleSelect.vue` | 角色卡片选择，默认高亮 `power_repair_expert` |
| `/chat` | `views/Chat.vue` | 智能问答：流式输出、引用来源折叠展示、停止生成、清空另开会话 |
| `/history` | `views/History.vue` | 按会话查看历史消息，用户 / 助手气泡样式区分 |
| `/kb` | `views/KbManage.vue` | 知识库状态、重建知识库、清空我的缓存 |
| `/upload` | `views/UploadStatus.vue` | 上传 PDF，多文件排队，逐文件显示进度与解析结果 |
| `/eval` | `views/Eval.vue` | RAGAS 评测结果表格，无数据时提示「暂无评测数据」 |
| `/admin` | `views/Admin.vue` | 三项服务状态、MySQL 表清单、Milvus 集合、Redis 缓存（全部只读） |
| `/user` | `views/User.vue` | 账号信息、我的会话列表、删除会话、退出登录 |

未登录访问除 `/login` 之外的任何路由，都会被 `router.js` 的前置守卫拦回登录页。

### 状态存储

教学版**不使用 Pinia / Vuex**，登录态与当前角色、当前会话直接存在 `localStorage`：

| 键 | 含义 |
|---|---|
| `user_id` | 用户编号，所有请求通过 `X-User-Id` 头带给后端 |
| `username` | 用户名，导航栏展示 |
| `role_id` / `role_name` | 当前角色，问答请求携带 |
| `conversation_id` | 当前会话，保证多轮续在同一会话里 |

生产环境应换成 Pinia，并把 token 放进内存 + HttpOnly Cookie，理由见 `docs/设计文档.md`。

### 本步新增的后端改动（仅两处）

1. `schemas.py`：`ChatResp` 增加 `long_memory_used`；新增 `ClearCacheResp`。
2. `app.py`：`/chat` 非流式返回补上 `long_memory_used`；`/cache/clear` 挂上
   `response_model=schemas.ClearCacheResp`。

### 遗留问题：助手回答没有落库（第 9.5 步已修复）

前端联调时发现 `/history` 只能看到用户提问，看不到助手的回答。定位到
`rag.py` 的 `save_turn()`：

前端联调时发现 `/history` 只能看到用户提问，看不到助手的回答。定位到
`rag.py` 的 `save_turn()`：

```python
("会话消息落库", lambda: db_user.save_message(conversation_id, "user", query) or
                         db_user.save_message(conversation_id, "assistant", answer)),
("短期记忆", lambda: memory.push_short_memory(user_id, "user", query) or
                     memory.push_short_memory(user_id, "assistant", answer)),
```

`save_message()` 返回新消息编号（如 `57`），`push_short_memory()` 返回当前条数（如 `2`），
都是**真值**，所以 `A(...) or B(...)` 在 A 求值后即短路，**B 永远不会执行**。
后果有两个：

- MySQL 里只写入了用户提问，助手回答丢失 → 前端历史页只有半截对话；
- Redis 短期记忆里只有用户提问，多轮对话时模型看不到自己上一轮的回答。

该问题属于第 6 步引入、第 8.5 步未发现，**不是第 9 步引入的**。
第 9 步明确要求不改 `rag.py`，因此当时只记录、不修复；**第 9.5 步已修复**，见下一节。

---

### 二之十一、第 9.5 步：修复 save_turn 短路 bug + 新增 /cache/stats

### 一、bug 根因：`or` 短路吞掉了助手回答

**现象**：前端历史页只显示用户提问，看不到助手回答。直接数 MySQL `messages` 表，
会话 14/15/16/17 全都只有一条 `role='user'`。

**根因**（`rag.py` 的 `save_turn()`，第 6 步引入）：

```python
("会话消息落库", lambda: db_user.save_message(conversation_id, "user", query) or
                         db_user.save_message(conversation_id, "assistant", answer)),
```

`save_message()` 返回 `cursor.lastrowid`（新消息编号，如 `68`），
`push_short_memory()` 返回 `client.llen(key)`（当前条数，如 `1`），**都是非 0 真值**。
Python 的 `A or B` 在 A 为真时**短路**，B 不求值 —— 写 user 成功之后，
写 assistant 那句**永远不执行**，而且不报任何错。

写这个 lambda 的人本意是把 `or` 当顺序连接符，但 `or` 是逻辑运算符；
`lambda` 体只能放一个表达式，塞不下两条语句，就被逼出了这个错误写法。

**后果两处，都是静默丢数据**：

| 位置 | 后果 |
|---|---|
| MySQL `messages` | 助手回答全丢 → 历史页只有半截对话，看着像系统没回答 |
| Redis `mem:short:{user_id}` | 同样只存提问 → 多轮对话时模型看不到自己上一轮说了什么，「那具体步骤呢」这类追问会失准 |

**修复方式**：拆成三个内层函数，组内两条**顺序执行**，组间用 `try/except` 隔离：

```python
def _write_messages() -> None:                # 第 1 组：会话消息落库（MySQL）
    db_user.save_message(conversation_id, "user", query)          # 写用户提问
    db_user.save_message(conversation_id, "assistant", answer)    # 写助手回答

def _write_short_memory() -> None:            # 第 2 组：短期记忆（Redis）
    memory.push_short_memory(user_id, "user", query)          # 推用户提问
    memory.push_short_memory(user_id, "assistant", answer)    # 推助手回答

writes = [("会话消息落库", _write_messages), ("短期记忆", _write_short_memory),
          ("长期记忆", _write_long_memory)]   # 三组互相独立
for label, action in writes:
    try: action()
    except Exception as exc: logger.warning("%s写入失败：%s", label, exc)   # 一组失败不影响其他组
```

`save_turn` 的**签名与三处调用位置都没动**，只改函数体。

**教训**：`lambda` + `or` 串联有副作用的函数是隐蔽陷阱。

1. `or` / `and` 的右边**不保证执行**，不能当语句分隔符用；
2. **返回真值的函数最容易踩**——返回 `None`/`False`/`0` 的函数会「看起来正常」地跑通，
   返回 `lastrowid`、`len()` 这类恒真值的，右边就永远不跑；
3. `lambda` 放不下两步操作，需要顺序执行就老老实实写 `def` 或多写一行。

修复前后 A/B 实测对比（同一份代码路径，只差 `or`）：

```
旧写法（or 串联）  -> MySQL 角色： ['user']
新写法（顺序语句）-> MySQL 角色： ['user', 'assistant']
```

### 二、新增接口 GET /cache/stats

| 项目 | 值 |
|---|---|
| 路径 | `GET /cache/stats` |
| 参数 | 无 |
| 响应模型 | `schemas.CacheStatsResp` |
| 实现 | 用 Redis `SCAN` 遍历 `cache:rag:*` 按第 4 段（`user_id`）分组，再遍历 `mem:short:*` 计数 |

响应示例：

```json
{"total_keys": 1, "by_user": {"40": 1}, "short_memory_users": 12}
```

`Admin.vue` 现在显示真实的 `缓存键数量` 与 `短期记忆用户数`，原来那句
「需后端新增统计接口，第 9 步未提供」的提示已删除。

`Admin.vue` 目前只展示 `total_keys` 与 `short_memory_users` 两个值；
`by_user` 已在响应里返回但**前端未展示**，需要按用户展开时可直接用。

### 三、新增测试

| 用例 | 断言 |
|---|---|
| `test_save_turn_writes_both` | 调 `save_turn` 后 MySQL 恰好 2 条（user + assistant）、Redis 至少 2 条且最近两条角色齐全，助手内容非空；MySQL 或 Redis 不可用则 skip |
| `test_cache_stats_api` | 造两个用户的缓存键，断言 `total_keys >= 2`、`by_user` 里两个用户都统计到；Redis 不可用则 skip |

两个用例跑完都会清理自己造的数据（会话、消息、Redis 键）。测试总数 51 → 53。

---

### 二之十二、第 9.6 步：修复 delete_conversation 级联删消息 + 清理半截会话 + Admin 展示 by_user

### 一、bug 根因：删会话不删消息，消息变孤儿

**现象**：删掉会话后，它的消息还留在 `messages` 表里；`GET /message/{id}`
对一个**已经删除的会话**仍然返回内容。

**根因**（`db_user.py` 的 `delete_conversation()`，第 6 步引入）：函数里只写了一条

```python
cursor.execute("DELETE FROM conversations WHERE id = %s AND user_id = %s", (conversation_id, user_id))
```

而 `messages` 表的建表语句里**没有外键约束**，只有一个普通索引
`KEY idx_msg_conv (conversation_id)`。索引只加速查询，**不产生级联行为**；
父表删了子表不动，MySQL 也不报错。

**后果**：数据只增不减，越用越脏 —— 第 9.6 步清理时一次性发现 29 条历史孤儿消息，
分布在 11 个会话里。

**修复方式**：同一个事务里按顺序三步（先校验、再删子表、最后删父表）：

```python
cursor.execute("SELECT id FROM conversations WHERE id = %s AND user_id = %s FOR UPDATE", ...)
if not cursor.fetchone():                     # 不是自己的会话
    conn.rollback(); return 0                 # 一条都不删
cursor.execute("DELETE FROM messages WHERE conversation_id = %s", ...)            # 先删子表
cursor.execute("DELETE FROM conversations WHERE id = %s AND user_id = %s", ...)   # 再删父表
conn.commit()                                 # 两条删除一次提交，要么都成要么都不成
```

**顺序是关键，不能反**。如果先删 `messages` 再校验归属，碰到"会话不属于该用户"时，
**别人的消息已经被删掉了** —— 校验必须发生在任何删除之前。
`FOR UPDATE` 给会话行加行锁，避免"校验通过 → 会话被转走 → 再删"这种竞态。

**教训**：删除操作必须考虑关联表。没有 `FOREIGN KEY ... ON DELETE CASCADE` 时，
应用层必须手动级联，且校验要放在最前面。同类风险：`users` → `conversations`
也没有外键级联，将来做"注销账号"时要一并处理两层。

### 二、清理半截会话（一次性工具）

`db_user.py` 新增 `cleanup_orphan_conversations(expected_ids=None)`，
**一次性工具，跑完即弃，不接入主流程**，也不会被 `app.py` 调用。

清理逻辑：找出 `messages` 里「有 user、无 assistant」的会话 → 删消息 → 删会话，同一事务提交。

**内置保护**：`expected_ids` 传入预期会话编号清单，实际结果与它不一致时**中止不删**，
避免误删其他用户的正常数据（比如某个用户提问后服务异常、消息没写全，那不是本工具该动的）。

**执行结果**（清理前先做只读预演核对）：

```
清理前核对：待清理会话与消息数 {6: 3, 7: 3, 8: 8, 10: 3, 11: 1, 12: 3, 13: 1, 14: 3, 15: 1, 16: 1, 17: 2}
清理完成：删除会话 [6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]，共删除消息 29 行
```

清理后复查：全库已无「有 user 无 assistant」的半截会话。
`step9_demo`（user_id 40）名下**能正常验收的会话 18、25 完整保留**；
`step9_demo` 用户本身未动。

> 说明：会话 17（「第9步流式演示」）虽归属 user_id 40，但它是第 6 步 bug 造的半截数据，
> 答案本身没存过、补不回来，经确认属于本次清理范围，一并删除。
> 会话 7 和 16 的会话行早先已被删，只剩孤儿消息，本次一并清掉。

### 三、Admin.vue 展示 by_user

Redis 卡片里在 `缓存键数量`、`短期记忆用户数` 下方加了一个 `el-table`：

| 用户 ID | 缓存键数量 |
|---|---|
| 40 | 1 |

数据源是 `cacheStats().by_user`（形如 `{"40": 1, "41": 3}`），
在页面里用 `computed` 转成 `[{user_id, keys}]` 喂给表格；空数据时显示「暂无缓存」。

### 四、新增测试

| 用例 | 断言 |
|---|---|
| `test_delete_conversation_cascade` | 建会话 + 2 条消息，删除后该会话消息为 0；越权删除返回 0 且**不动别人的消息** |
| `test_cleanup_orphan_conversations` | 造半截会话后被清理工具清掉；预期清单与实际不符时中止、一条都不删 |

两个用例跑完都会清掉自己造的数据。测试总数 53 → 55。

---

### 二之十三、第 9.7 步：修复缓存命中时多轮记忆断片

### 一、bug 根因：命中缓存直接 `return`，绕过了 `save_turn`

**现象**：在新会话里问一个**之前问过的问题**，`/history` 看不到这一轮；
再追问时模型像是"没看见上一句"，指代（「那具体步骤呢」）会答偏。

**根因**（`rag.py` 的 `ask()`，第 7 步引入缓存时留下）：

```python
if not stream:
    cached = optimize.cache_get(key)
    if cached:
        logger.info("缓存命中：%s", key)
        cached["cache_hit"] = True
        return cached            # ← 直接返回，函数末尾的 save_turn 根本没机会执行
```

而 `save_turn()` 是三处副作用的**唯一入口**，于是这一轮：

| 副作用 | 目标 | 后果 |
|---|---|---|
| 会话消息落库 | MySQL `messages` | 这一轮 user + assistant 都不写 → `/history` 缺一轮 |
| 短期记忆 | Redis `mem:short:{user_id}` | 这一轮不进记忆 → **下一轮 history 缺一段，多轮对话断片** |
| 长期记忆 | Milvus `power_repair_memory` | 这一轮不进向量库 → 以后按语义召回不到它 |

**最严重的是第二条**，它会产生连锁反应：第一次问 A 写入记忆 → 第二次问 A 命中缓存、
不写记忆 → 第三次追问 B 时 history 里只有第一次那段，模型看不见第二次的问答。
命中缓存本是加速手段，却把上下文撕掉一块。

### 二、修复方式：命中缓存只跳过昂贵计算，不跳过副作用

```python
if cached:                                   # 命中
    # 缓存只该跳过 LLM 调用，不该跳过副作用
    save_turn(user_id, conversation_id, query, cached["answer"])   # 与未命中路径完全一致
    logger.info("缓存命中：命中键=%s，仍写入 MySQL/Redis/Milvus", key)
    cached["cache_hit"] = True
    return cached
```

**一个容易写错的细节**：`save_turn()` 内部**已经包含 Milvus 长期记忆写入**
（它的第三组 `_write_long_memory`）。所以这里**只调它一次**，不能"再单独调一次
`memory.save_long_memory`"——那样长期记忆一轮写两条，两条路径反而不再对称。
实测确认：缓存命中时 Milvus 恰好新增 **1** 条。

**副作用对称性对照**：

| 步骤 | 缓存未命中 | 缓存命中 |
|---|---|---|
| 取短期记忆 / 查询改写 | 执行 | 执行 |
| 检索 + 重排 | 执行 | **跳过**（昂贵） |
| 调大模型生成 | 执行 | **跳过**（昂贵） |
| 写回 MySQL / Redis / Milvus | 执行 | **照常执行** |
| 写答案缓存 | 执行 | 跳过（已在缓存里） |

**教训**：缓存的职责边界是"跳过昂贵计算"，不是"跳过副作用"。
写库、写记忆虽然廉价，但**有语义后果**——它们决定了下一轮能看见什么，
和"检索 + 生成"这种纯计算不能一起跳过。
另外 `ask()` 里所有副作用都在函数末尾，**任何提前 `return` 都会静默漏掉全部写回**，
以后新增提前返回的分支时，必须回头检查它跳过了哪些副作用。

**一个可接受的连带代价**：同一问题反复提问时，每次命中都会再写一轮消息与记忆。
MySQL 与 Redis 里出现多轮相同记录是**正确的**（每次提问本来就是一个真实轮次）；
Milvus 长期记忆也会累积同题记忆，但 `collect_long_memory()` 已按 question 去重，
不会挤占召回名额。

### 三、验证

**A/B 对照**（只在缓存命中那一次把 `save_turn` 换成空操作，精确复刻旧行为）：

```
旧行为：第一轮后 2 条 -> 第二轮(命中)后 2 条，命中这轮写入 0 条
新行为：第一轮后 2 条 -> 第二轮(命中)后 4 条，命中这轮写入 2 条
```

**三处副作用分别核对**（缓存命中时）：MySQL 多 2 条、Redis 多 2 条、Milvus 恰好多 1 条。

**多轮实测**（会话 42 连续四轮）：

```
问答开始：...，历史 0 条                                    ← 第一轮 未命中
缓存命中：命中键=cache:rag:answer:40:...，仍写入 MySQL/Redis/Milvus   ← 第二轮 命中
缓存命中：命中键=cache:rag:answer:40:...，仍写入 MySQL/Redis/Milvus   ← 第三轮 命中
问答开始：...，历史 6 条                                    ← 第四轮 追问，带上前三轮记忆
```

第四轮 `历史 6 条` 是关键证据：中间两轮都是缓存命中，**未修复时这里只会有 2 条**。
`/message/42` 返回 4 个 user + 4 个 assistant，完整无缺。

### 四、新增测试

| 用例 | 断言 | 耗时 |
|---|---|---|
| `test_cache_hit_still_writes` | 第一轮未命中、第二轮命中；断言命中这轮 MySQL 多 2 条（user + assistant），Redis 短期记忆最近两条为 assistant + user | ~26 秒 |
| `test_cache_hit_writes_long_memory` | 第一轮后清空该用户长期记忆，第二轮命中缓存；断言 Milvus 里能搜到这一轮 Q&A | ~62 秒 |

两个用例跑完都会清掉会话、消息、Redis 键、缓存与长期记忆。
**注意**：这两个用例会真实调用大模型（各 1~2 次），不是 mock，
因此比其余用例慢；测试总数 55 → 57。

---

### 二之十四、第 10 步：RAGAS 评测

### 一、评测集

`data/eval_dataset.json`，**15 条**中文问答，只含 `question` 与 `ground_truth` 两个字段，
覆盖语料里的 **5 个 PDF 标准**：

| 标准 | 条数 |
|---|---|
| GB/T 17623 绝缘油中溶解气体气相色谱测定法 | 4 |
| GB/T 25438 三相油浸式立体卷铁芯电力变压器 | 3 |
| GB/T 27743 变压器专用设备检测方法 | 3 |
| GB/T 44653 六氟化硫（SF6）气体现场循环再利用导则 | 3 |
| GB/T 46131 特高压变压器用分接开关 | 2 |

**难度分层**（混四种题型，避免只考"抄一句"）：
简单事实查询 4 条、流程步骤 2 条、参数数值 5 条、判断标准 4 条。

**`ground_truth` 全部从 `data/parsed_chunks.json` 的原文逐句摘录，没有编造**，
数字与条款内容一字不改。摘录时只做两件事：去掉 PDF 排版产生的多余空格、
把 `≥` 写成中文"不小于"。

### 二、四个指标

| 指标 | 查的是哪一环 | 它在问什么 |
|---|---|---|
| `faithfulness` 忠实度 | **生成** | 回答里的每个论断，能不能在检索到的上下文里找到依据 |
| `answer_relevancy` 答案相关度 | **生成** | 回答是否正面回答了问题 |
| `context_precision` 上下文精确率 | **检索** | 检索回来的上下文里，相关的排在前面了吗 |
| `context_recall` 上下文召回率 | **检索** | 参考答案里的信息，在检索上下文里都出现了吗 |

前两个查"生成"，后两个查"检索"。分开看才能定位是哪一环拖后腿。

### 三、评估模型选型

| 角色 | 选型 | 理由 |
|---|---|---|
| 评估 LLM | **DeepSeek**（`.env` 里的 `LLM_MODEL`，走 OpenAI 兼容接口） | 项目本来就在用，Key 已在 `.env`，不额外引入供应商 |
| 评估 Embedding | **BGE-m3**（复用 `vector_store.embed_texts`） | `answer_relevancy` 要算语义相似度；复用已有模型实例，不重复加载、不联网 |

### 四、运行方式

```bash
python -m eval_ragas
```

流程：读评测集 → 逐条跑完整 RAG（检索 + 生成）→ RAGAS 四指标打分 →
写 `data/ragas_result.json` → 结果在**前端 `/eval` 页面**展示
（汇总卡片 + 逐条明细表，长文本悬浮展开）。

跑完会打印每条问题的四指标分数与四个指标的均分。

### 五、两个必须注意的实现细节

1. **每条问题之前要清空评测用户的记忆**。`rag.ask()` 的 history 与长期记忆
   都是按 `user_id` 读的、**不是按会话读的**，只新建会话并不能隔离。
   实测第一次冒烟测试时第 2 条问题的改写混进了第 1 条的内容，`faithfulness` 直接被拉到 0。
   现在每条之前都清一次短期记忆（Redis）与长期记忆（Milvus）。
2. **`answer_relevancy` 必须设 `strictness=1`**。它默认 `strictness=3`，
   会向模型请求 `n=3` 个生成，而 **DeepSeek 只支持 `n=1`**，直接返回 400。
   实测未设之前 15 条里有 10 条报 `Invalid n value`，指标整片算不出来。

### 六、指标算不出来时的处理

RAGAS 的指标靠 LLM 逐条判定，**偶发的超时或返回格式异常会让某一条的某一项算不出来**。
这种情况写 `null`（前端显示 `-`），**不用 0 冒充"得了 0 分"**——两者含义完全不同。
汇总均分只对**算出来的那些条**求平均，并在日志里打印每项跳过了几条。
`schemas.py` 里对应的四个字段因此是 `Optional[float]`。

### 七、已知局限

- 评测集只有 15 条，均分对单条异常值敏感（一条 0 分能拉动均分约 0.067）；
- `ground_truth` 由人工摘录，虽是原文摘抄但仍是人工选择的结果；
- RAGAS 的 LLM 评判有随机性，分数适合看量级与横向对比，不适合比小数点后两位；
- 只做**单轮**评测，第 6 步的多轮记忆能力没有被这项评测覆盖；
- 评测时按设计关掉了长期记忆（每条清空），与线上多轮场景有差异。

详细分数与逐条分析见 `docs/测试成果.md`。
