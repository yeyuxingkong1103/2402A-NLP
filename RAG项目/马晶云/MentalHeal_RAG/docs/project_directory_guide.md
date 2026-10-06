# MentalHeal RAG 项目代码目录说明（口语化版）

这份文档不讲太多抽象概念，主要就是带着你认一下：项目每个文件夹是干什么的、平时改功能应该去哪里找，以及用户发出一句话以后，代码大概是怎么跑起来的。

## 一、先看整体目录

```text
MentalHeal_RAG/
├── backend/                 后端：FastAPI、RAG、数据库和业务逻辑
├── frontend/                前端：Vue 3 聊天页面
├── data/                    PDF、清洗结果、分块结果和评测数据
├── docs/                    项目说明、架构图、部署文档和评测报告
├── scripts/                 启动、停止、健康检查和评测脚本
├── configs/                 配置文件目录，目前主要是占位目录
├── logs/                    日志目录
├── notebooks/               实验 Notebook 目录，目前主要是占位目录
├── backups/                 数据库备份文件
├── .env                     本机真实配置，不能提交或公开
├── .env.example             配置模板，不放真实密钥
├── requirements.txt         Python 依赖
└── README.md                项目总说明
```

简单说就是：

- `frontend` 负责“页面长什么样、用户怎么操作”。
- `backend` 负责“接口怎么处理、怎么检索知识、怎么调用大模型”。
- `data` 负责“知识库原文件和处理过程中的数据”。
- `docs` 负责“项目怎么部署、架构是什么、测试结果怎么样”。
- `scripts` 负责“怎么启动、怎么检查、怎么停止”。

---

## 二、后端目录：`backend/`

后端是整个项目的核心，主要使用 Python + FastAPI。

```text
backend/
├── app/
│   ├── api/
│   ├── core/
│   ├── db/
│   ├── ingestion/
│   ├── memory/
│   ├── models/
│   ├── rag/
│   ├── schemas/
│   ├── security/
│   ├── services/
│   └── main.py
└── tests/
```

### 1. `backend/app/main.py`：后端总入口

这个文件可以理解成后端的“总开关”。启动 Uvicorn 的时候，最终就是加载这里的 FastAPI 应用。

它主要做这些事：

- 创建 FastAPI 应用。
- 注册登录、聊天、历史记录、角色、文档、评测和记忆等接口。
- 配置前端跨域访问。
- 初始化默认的 AI 角色。
- 提供 `/health` 健康检查接口。
- 检查 MySQL、Redis、Milvus 和 `knowledge_chunks` 集合是否正常。

如果你想知道“后端到底注册了哪些接口”，优先看这个文件和 `backend/app/api/`。

### 2. `backend/app/api/`：接口层

这里放的是对外暴露的 API 路由。可以把它理解成前端和后端之间的“窗口”。

```text
backend/app/api/
├── auth.py          登录、注册、当前用户信息
├── chat.py          普通问答和流式问答
├── documents.py     文档上传、解析、向量化
├── evaluations.py   RAGAS 评测接口
├── history.py       历史会话接口
├── memories.py      长期记忆接口
├── retrieval.py     单独的知识检索接口
└── roles.py         AI 角色接口
```

最重要的文件是 `chat.py`：

- `POST /api/v1/chat`：一次性返回完整答案。
- `POST /api/v1/chat/stream`：通过 SSE 流式返回答案，前端可以边生成边显示。

目前页面上的“我的偏好”和“管理中心”入口已经去掉了，但是后端的文档、评测和记忆接口还保留着。这样以后如果要重新做管理页面，不需要从后端重新开始写。

### 3. `backend/app/core/`：配置和基础设置

目前主要是：

```text
backend/app/core/
└── config.py
```

`config.py` 负责读取环境变量和项目配置，比如：

- DeepSeek API 配置。
- MinerU API 配置。
- MySQL、Redis、Milvus 地址。
- RAG 检索数量和分数阈值。
- Rerank 候选数量。
- 上传文件大小限制。

像 `RAG_RERANK_CANDIDATES=8` 这种性能配置，就应该在这里找对应的配置字段。真实密码和 API Key 放在 `.env`，不要直接写进 Python 文件。

### 4. `backend/app/db/`：数据库连接和表结构

```text
backend/app/db/
├── session.py
├── schema.sql
└── migrations/
    ├── 001_auth_roles.sql
    ├── 002_knowledge_evaluation.sql
    └── 003_long_term_memories.sql
```

这里主要处理 MySQL 数据库：

- `session.py`：创建数据库连接和 Session。
- `schema.sql`：基础表结构。
- `migrations/`：后续补充的数据库变更脚本。

用户、角色、会话、聊天消息、文档、知识分块和评测记录等结构化数据，主要都放在 MySQL 里。

### 5. `backend/app/models/`：数据库模型

```text
backend/app/models/
└── chat.py
```

这里是 Python 代码里的数据库模型，主要对应用户、角色、会话、消息等表。

简单说：`db/` 更像是数据库连接和 SQL 文件，`models/` 是后端程序操作这些表时使用的 Python 对象。

### 6. `backend/app/schemas/`：接口请求和返回格式

```text
backend/app/schemas/
├── auth.py
├── chat.py
├── documents.py
├── evaluations.py
├── history.py
├── memories.py
├── retrieval.py
└── roles.py
```

这里定义接口接收什么参数、返回什么字段。比如聊天请求里的：

- 用户消息。
- 会话 ID。
- 角色 ID。
- `top_k` 检索数量。

如果前端传参报校验错误，或者接口返回字段需要调整，可以先看这里。

### 7. `backend/app/services/`：真正的业务逻辑

```text
backend/app/services/
├── chat_history.py
├── chat_service.py
├── deepseek_client.py
├── document_ingestion.py
└── ragas_evaluation.py
```

这里是后端最值得重点看的地方，因为接口文件一般只是接收请求，真正干活的是 Service。

- `chat_service.py`：聊天总编排。
  - 读取聊天历史。
  - 判断是不是危机表达。
  - 读取 Redis 缓存。
  - 调用知识检索。
  - 组织 DeepSeek 提示词。
  - 保存聊天记录。
  - 通过流式事件把结果返回给前端。
- `deepseek_client.py`：封装 DeepSeek API 调用。
- `chat_history.py`：处理聊天历史。
- `document_ingestion.py`：处理文档解析、分块、向量化和入库。
- `ragas_evaluation.py`：执行 RAGAS 评测。

如果以后要修改“回答逻辑”，优先看 `chat_service.py`；如果要修改“文档入库逻辑”，优先看 `document_ingestion.py`。

### 8. `backend/app/rag/`：RAG 检索核心

```text
backend/app/rag/
└── retriever.py
```

`retriever.py` 是在线知识检索的核心文件。现在的检索流程大概是：

1. 用本地 BGE-m3 把用户问题转成向量。
2. 去 Milvus 做向量检索。
3. 同时从本地 chunk 数据做 BM25 关键词检索。
4. 合并向量和关键词候选。
5. 先用快速分数筛选候选。
6. 只把前 8 条左右候选交给 BGE-reranker 精排。
7. 按综合分数、文档和页面限制选出最终引用内容。

这里的模型当前固定使用 CPU，主要是为了避开 Apple Silicon 上 MPS/Metal 偶发崩溃的问题。

### 9. `backend/app/ingestion/`：离线知识库处理

```text
backend/app/ingestion/
├── chunking.py
├── cleaning.py
├── cli.py
├── config.py
├── embedding.py
├── milvus_store.py
├── mineru_client.py
├── native_pdf.py
├── ocr_pdf.py
├── pipeline.py
├── schemas.py
├── table_pdf.py
├── text_processing.py
└── vision_ocr.py
```

这一组文件负责把 PDF 变成可以检索的知识库内容：

- `pipeline.py`：整条 PDF 处理流水线的入口。
- `mineru_client.py`：调用 MinerU 做 PDF 解析和 OCR。
- `cleaning.py`、`text_processing.py`：清洗和规范化文本。
- `chunking.py`：把长文档切成一个个 chunk。
- `embedding.py`：用 BGE-m3 生成向量。
- `milvus_store.py`：把向量写入 Milvus。
- `cli.py`：从命令行启动文档解析。
- `native_pdf.py`、`ocr_pdf.py`、`table_pdf.py`、`vision_ocr.py`：不同类型 PDF、OCR 和表格处理的辅助模块。

这部分主要是“离线处理”，不是用户每次提问时才临时解析 PDF。提前处理好以后，在线问答只需要检索已经入库的内容。

### 10. `backend/app/memory/`：Redis 短期记忆

```text
backend/app/memory/
└── redis_memory.py
```

这里使用 Redis 保存：

- 最近几轮聊天记录。
- 相同问题的 RAG 检索缓存。
- 部分文档处理任务的状态。

Redis 缓存命中时，就不用每次都重新走完整的检索流程，所以第二次问相同问题会明显更快。

### 11. `backend/app/security/`：登录鉴权和安全策略

```text
backend/app/security/
├── auth.py
└── safety.py
```

- `auth.py`：Bearer Token、当前用户和权限校验。
- `safety.py`：识别自伤、自杀、伤害他人等危机表达。

命中高风险表达时，系统不会继续调用普通知识检索和大模型生成，而是优先返回固定的安全提醒。这是产品的安全边界，不是正式医疗诊断或急救服务。

---

## 三、后端测试目录：`backend/tests/`

```text
backend/tests/
├── integration/
└── unit/
    ├── test_long_term_memory.py
    ├── test_online_safety.py
    └── test_retriever.py
```

目前重点测试包括：

- 长期记忆相关逻辑。
- 在线危机安全分流。
- RAG 检索和快速候选筛选。

`integration/` 目前保留为集成测试目录，后面如果要加真实数据库、Redis 或 Milvus 联调测试，可以放在这里。

---

## 四、前端目录：`frontend/`

前端使用 Vue 3 + Vite + TypeScript。

```text
frontend/
├── src/
│   ├── App.vue
│   └── main.ts
├── public/
├── index.html
├── package.json
├── package-lock.json
├── tsconfig.json
└── vite.config.ts
```

### 1. `frontend/src/App.vue`：目前的主要页面

现在项目的聊天页面主要集中在这一个 Vue 单文件组件里，包括：

- 登录和注册界面。
- 当前用户信息。
- AI 角色选择。
- 新建会话和历史会话。
- 用户发送消息。
- SSE 流式读取回答。
- 展示知识库引用来源。
- 危机安全提示。
- 退出登录。

“我的偏好”和“管理中心”页面入口及其前端逻辑已经删除，目前页面只保留核心聊天功能。

如果要改页面文案、聊天布局、按钮、消息气泡或流式显示效果，通常先看这个文件。

### 2. `frontend/src/main.ts`：前端启动入口

这个文件负责创建 Vue 应用，并把 `App.vue` 挂载到页面上。一般不需要频繁修改。

### 3. `frontend/index.html`：网页外壳

这是 Vite 使用的 HTML 入口，主要提供挂载 Vue 应用的根节点。

### 4. `frontend/package.json`：前端命令和依赖

主要命令是：

```bash
npm run dev       # 启动开发服务器
npm run build     # 类型检查并打包
npm run preview   # 预览打包后的前端
```

`vite.config.ts` 里配置了开发服务器，默认端口是 `5173`。

### 5. 其他前端目录

当前 `src/api/`、`src/assets/`、`src/components/`、`src/stores/`、`src/types/`、`src/views/` 目录已经预留出来，但当前核心页面主要还是集中在 `App.vue` 中。以后如果页面继续变大，可以再把 API、组件、状态和页面拆出去。

---

## 五、数据目录：`data/`

```text
data/
├── raw/
├── processed/
├── cleaned/
├── chunks/
├── vectorized/
├── failed/
└── evaluation/
```

这一块可以按照“PDF 从进来到变成知识库”的顺序理解：

```text
raw → processed → cleaned → chunks → vectorized / Milvus
```

- `raw/`：原始 PDF 文件，最开始上传的文件放这里。
- `processed/`：PDF 解析后的页面和结构化结果。
- `cleaned/`：清洗后的文本数据。
- `chunks/`：切分好的知识片段，在线 BM25 检索会用到相关数据。
- `vectorized/`：向量化过程的清单或中间信息。
- `failed/`：解析失败的任务及错误信息。
- `evaluation/`：RAGAS 评测数据和评测结果。

真正在线使用的向量会写入 Milvus，文件目录里的 JSON 更多是离线处理和本地追踪用的。

不要随便删除 `raw/`、`processed/`、`cleaned/` 和 `chunks/` 里的数据，除非已经确认不再需要，或者已经做好备份。

---

## 六、文档目录：`docs/`

```text
docs/
├── architecture/
│   ├── mentalheal_rag_architecture.html
│   └── technical_architecture.md
├── evaluation/
│   └── ragas_report.md
├── requirements/
│   ├── pdf_ingestion_guide.md
│   └── requirements_spec.md
├── deployment.md
└── project_directory_guide.md
```

- `architecture/mentalheal_rag_architecture.html`：刚生成的可直接用浏览器打开的架构图。
- `architecture/technical_architecture.md`：技术架构说明。
- `requirements/`：需求说明和 PDF 导入说明。
- `deployment.md`：本地和 Ubuntu 部署说明。
- `evaluation/ragas_report.md`：RAGAS 评测报告。
- `project_directory_guide.md`：就是当前这份目录说明。

如果以后要给老师、同学或项目答辩展示，架构图和需求文档主要从 `docs/` 里找。

---

## 七、脚本目录：`scripts/`

```text
scripts/
├── evaluation/
│   └── run_ragas.py
├── install/
│   └── check_env.sh
├── ops/
│   ├── health_check.sh
│   └── stop_local.sh
└── start/
    ├── all.sh
    ├── backend.sh
    └── frontend.sh
```

这些脚本是为了避免每次都手动敲一长串命令：

- `scripts/start/backend.sh`：启动后端。
- `scripts/start/frontend.sh`：启动前端。
- `scripts/start/all.sh`：前后端一起启动。
- `scripts/ops/health_check.sh`：检查后端以及 MySQL、Redis、Milvus 是否正常。
- `scripts/ops/stop_local.sh`：停止本地服务。
- `scripts/install/check_env.sh`：检查运行环境和依赖。
- `scripts/evaluation/run_ragas.py`：执行 RAGAS 评测。

---

## 八、配置、依赖和备份

### `.env` 和 `.env.example`

- `.env`：本机真正使用的配置，里面可能有 API Key、密码和数据库地址，不能发给别人，也不要提交到仓库。
- `.env.example`：给别人参考的配置模板，只放变量名和示例值。

### `requirements.txt`

后端 Python 依赖清单，比如 FastAPI、SQLAlchemy、Milvus、Redis、Sentence Transformers 等都在这里声明。

### `backups/`

保存数据库备份文件。修改数据库表结构、重置测试数据之前，最好先备份。

### `configs/`、`logs/`、`notebooks/`

这几个目录已经预留：

- `configs/`：以后可以放更完整的配置文件。
- `logs/`：运行日志。
- `notebooks/`：实验和分析 Notebook。

目前它们主要是项目结构上的预留位置。

---

## 九、一句话请求到底怎么走

用户在页面里发送一句话后，大概会按照下面这条路线走：

```text
浏览器
  ↓
Vue 3 的 App.vue
  ↓ POST /api/v1/chat/stream
FastAPI 的 api/chat.py
  ↓
ChatService
  ↓
危机安全检测
  ├─ 命中危机表达：直接返回安全提醒
  └─ 普通问题：继续往下走
       ↓
Redis 读取历史和检索缓存
       ↓
KnowledgeRetriever
       ├─ BGE-m3 向量化
       ├─ Milvus 向量检索
       ├─ BM25 关键词检索
       └─ BGE-reranker 精排
       ↓
DeepSeek 生成回答
       ↓
SSE 流式返回前端
       ↓
MySQL 保存会话和聊天记录，Redis 保存短期记忆
```

离线知识库则是另一条路线：

```text
PDF
  ↓
MinerU 解析 / OCR
  ↓
文本清洗
  ↓
文档分块
  ↓
BGE-m3 向量化
  ├─ 写入 Milvus，供在线语义检索
  └─ 写入 MySQL，保存文档和 chunk 元数据
```

---

## 十、以后改功能应该去哪里

| 想改的内容 | 优先查看的位置 |
|---|---|
| 修改页面布局、按钮和聊天气泡 | `frontend/src/App.vue` |
| 修改前端请求地址 | `frontend/src/App.vue` 里的 API 基地址配置 |
| 修改登录注册接口 | `backend/app/api/auth.py`、`backend/app/security/auth.py` |
| 修改聊天接口或 SSE 格式 | `backend/app/api/chat.py` |
| 修改回答整体流程 | `backend/app/services/chat_service.py` |
| 修改 DeepSeek 调用 | `backend/app/services/deepseek_client.py` |
| 修改向量检索、BM25、Rerank | `backend/app/rag/retriever.py` |
| 修改 PDF 解析和分块 | `backend/app/ingestion/`、`backend/app/services/document_ingestion.py` |
| 修改危机表达处理 | `backend/app/security/safety.py` |
| 修改 Redis 缓存和短期记忆 | `backend/app/memory/redis_memory.py` |
| 修改数据库表结构 | `backend/app/db/schema.sql`、`backend/app/db/migrations/` |
| 修改接口参数格式 | `backend/app/schemas/` |
| 修改启动和健康检查 | `scripts/start/`、`scripts/ops/` |
| 修改测试 | `backend/tests/unit/` |

最简单的记忆方式就是：**页面问题看 frontend，接口问题看 api，业务问题看 services，检索问题看 rag，PDF 问题看 ingestion，数据库问题看 db。**

---

## 十一、当前项目状态补充

现在项目的核心链路已经跑通：

- 前端默认运行在 `http://localhost:5173`。
- 后端默认运行在 `http://localhost:8000`。
- 健康检查会验证 MySQL、Redis、Milvus 和知识库集合。
- 在线问答支持 SSE 流式返回。
- 知识库支持向量检索、BM25 和 Rerank。
- 页面已经去掉“我的偏好”和“管理中心”入口。
- 后端单元测试和前端构建已经通过。

不过当前更准确的说法是：**项目已经达到可以演示和继续迭代的阶段，距离完整生产化还可以继续补浏览器自动化测试、部署守护、日志监控、压力测试和密钥轮换。**
