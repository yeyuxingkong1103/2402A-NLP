# RAG PDF 问答系统 v1 MVP 任务清单

> 基于 `docs/v1/spec.md` 与 `docs/v1/plan.md` 拆分。

## 任务总览

- [ ] 任务 1：项目基础文件
- [ ] 任务 2：配置加载与路径边界
- [ ] 任务 3：领域模型
- [ ] 任务 4：本地状态存储
- [ ] 任务 5：PDF 上传保存
- [ ] 任务 6：MinerU 解析契约与分块
- [ ] 任务 7：bge-m3 嵌入适配器
- [ ] 任务 8：Qdrant 向量库适配层
- [ ] 任务 9：构建流水线服务
- [ ] 任务 10：任务、文档和向量库接口
- [ ] 任务 11：问答服务与 Chat 接口
- [ ] 任务 12：文档与 v1 基线结构
- [ ] 任务 13：全量验证

---

## 任务 1：项目基础文件

**目标：** 建立最小项目骨架、依赖清单、配置样板和运行说明。

**涉及文件：**
- `requirements.txt`
- `.env.example`
- `.gitignore`
- `README.md`

**验收标准：**
- 依赖版本被精确锁定
- `.env.example` 含本地运行所需配置项
- `.gitignore` 忽略本地产物但不忽略 `.specify`
- `README.md` 说明本地运行与主要接口

## 任务 2：配置加载与路径边界

**目标：** 提供统一配置对象，并限制文件路径必须位于项目目录内。

**涉及文件：**
- `backend/app/__init__.py`
- `backend/app/config.py`
- `backend/tests/test_config.py`

**验收标准：**
- 默认配置与宪法一致
- 路径检查能拒绝项目目录外路径
- 测试覆盖默认值与越界场景

## 任务 3：领域模型

**目标：** 建立文档、任务、chunk、引用与问答请求响应模型。

**涉及文件：**
- `backend/app/models.py`
- `backend/tests/test_models.py`

**验收标准：**
- `Document`、`BuildTask`、`Chunk`、`Citation`、`ChatRequest`、`ChatResponse` 可被验证
- `Chunk.page` 必须为正整数
- `BuildTask` 状态流转可测试

## 任务 4：本地状态存储

**目标：** 提供本地 JSON 存储，保存文档、任务与 chunk 元数据。

**涉及文件：**
- `backend/app/storage.py`
- `backend/tests/test_storage.py`

**验收标准：**
- 文档、任务、chunk 可以读写往返
- 能按文档 ID 过滤 chunk
- 测试覆盖保存与读取路径

## 任务 5：PDF 上传保存

**目标：** 接收 PDF 上传、校验文件名并保存到 `data/`。

**涉及文件：**
- `backend/app/files.py`
- `backend/app/routes_files.py`
- `backend/tests/test_files.py`

**验收标准：**
- 仅允许 PDF 上传
- 文件名被安全清洗
- 上传后返回文档 ID 和任务 ID

## 任务 6：MinerU 解析契约与分块

**目标：** 定义 MinerU 结构化块模型，并把清洗与分块逻辑做成可测试函数。

**涉及文件：**
- `backend/app/mineru.py`
- `backend/app/chunking.py`
- `backend/tests/test_chunking.py`

**验收标准：**
- MinerU 块保留页码、标签和来源片段
- 分块结果保留页码与类别
- 缺少页码时拒绝入库

## 任务 7：bge-m3 嵌入适配器

**目标：** 封装本地 bge-m3 dense+sparse 嵌入接口，并提供测试替身。

**涉及文件：**
- `backend/app/embeddings.py`
- `backend/tests/test_embeddings.py`

**验收标准：**
- 支持生成 dense 与 sparse 表示
- 提供测试用 fake embedder
- 可转换为 Qdrant 命名向量结构

## 任务 8：Qdrant 向量库适配层

**目标：** 封装向量库写入、检索和类别读取能力。

**涉及文件：**
- `backend/app/vector_store.py`
- `backend/tests/test_vector_store.py`

**验收标准：**
- 支持测试内存实现
- 支持本地 Qdrant collection 初始化
- 支持 dense+sparse 检索与 RRF 融合
- 可查看类别列表

## 任务 9：构建流水线服务

**目标：** 组织清洗 → 分块 → 向量化 → 入库的真实任务流转。

**涉及文件：**
- `backend/app/pipeline.py`
- `backend/tests/test_pipeline.py`

**验收标准：**
- 任务状态会随步骤更新
- 成功后文档状态更新为已入库
- 失败时任务和文档都进入失败状态

## 任务 10：任务、文档和向量库接口

**目标：** 提供任务进度、文档列表、向量库列表与类别查询接口。

**涉及文件：**
- `backend/app/routes_tasks.py`
- `backend/app/routes_documents.py`
- `backend/app/main.py`
- `backend/tests/test_api.py`

**验收标准：**
- 健康检查可访问
- 任务可查询进度
- 文档、向量库和类别接口可访问
- 上传接口拒绝非 PDF

## 任务 11：问答服务与 Chat 接口

**目标：** 提供带引用约束的问答服务与聊天接口。

**涉及文件：**
- `backend/app/qa.py`
- `backend/app/routes_chat.py`
- `backend/app/main.py`
- `backend/tests/test_qa.py`

**验收标准：**
- 无证据时返回兜底文案
- 提示词中包含文档名和页码引用
- Chat 接口可返回 `ChatResponse`

## 任务 12：文档与 v1 基线结构

**目标：** 补齐需求、版本、架构图和数据说明文档。

**涉及文件：**
- `docs/需求说明.md`
- `docs/版本迭代.md`
- `docs/架构/架构图-v1.md`
- `data/README.md`
- `eval/sets/README.md`
- `eval/baseline/README.md`

**验收标准：**
- 需求说明作为唯一需求文档存在
- 架构图版本与需求版本一致
- 数据与评测目录说明齐全

## 任务 13：全量验证

**目标：** 通过测试和导入验证整个 MVP 基线可用。

**涉及文件：**
- 全部已创建代码与文档

**验收标准：**
- `pytest backend/tests -v` 通过
- `python -c "from backend.app.main import app; print(app.title)"` 可执行
- 核心文档存在
- 覆盖所有接口和关键约束
