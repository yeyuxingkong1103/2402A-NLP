# RAG PDF 问答系统 v1 MVP 检查清单

> 基于 `docs/v1/spec.md`、`docs/v1/plan.md`、`docs/v1/tasks.md`。
>
> 通过本清单后，才允许进入实现阶段。

## 总体门禁

- [ ] `docs/v1/spec.md`、`docs/v1/plan.md`、`docs/v1/tasks.md`、`docs/v1/checklist.md` 均存在且命名一致
- [ ] `docs/需求说明.md` 已存在且作为唯一需求文档
- [ ] `docs/架构/架构图-v1.md` 已存在且与 v1 需求一致
- [ ] `docs/版本迭代.md` 已存在并记录 v1
- [ ] `data/README.md`、`eval/sets/README.md`、`eval/baseline/README.md` 已存在
- [ ] 所有新增内容都落在宪法允许的目录中

## 任务 1：项目基础文件

- [ ] `requirements.txt` 的依赖版本与任务 1 完全一致
- [ ] `.env.example` 包含 APP_HOST、APP_PORT、DATA_DIR、QDRANT_PATH、QDRANT_COLLECTION、BGE_M3_MODEL_PATH、OLLAMA_BASE_URL、OLLAMA_MODEL、MAX_UPLOAD_MB、TOP_K、MIN_RETRIEVAL_SCORE
- [ ] `.gitignore` 忽略 `.env`、`.venv/`、缓存、`data/*.pdf`、`data/qdrant/`、`eval/results/`、`eval/baseline/*.json`
- [ ] `.gitignore` 没有忽略 `.specify`
- [ ] `README.md` 说明了本地运行步骤、Ollama 要求和主要接口

## 任务 2：配置加载与路径边界

- [ ] `backend/app/config.py` 提供 `AppSettings`
- [ ] `AppSettings` 默认值与 `data/`、`data/qdrant/`、`rag_documents`、`deepseek-r1:7b` 一致
- [ ] `ensure_project_path()` 能允许项目内路径
- [ ] `ensure_project_path()` 能拒绝项目外路径
- [ ] `backend/tests/test_config.py` 覆盖默认值与越界场景

## 任务 3：领域模型

- [ ] `backend/app/models.py` 定义 `Document`、`BuildTask`、`Chunk`、`Citation`、`ChatRequest`、`ChatResponse`
- [ ] `Chunk.page` 必须是正整数
- [ ] `BuildTask` 能表达 pending、running、completed、failed
- [ ] `Document.new()` 和 `BuildTask.new()` 可生成初始记录
- [ ] `backend/tests/test_models.py` 覆盖模型验证与状态流转

## 任务 4：本地状态存储

- [ ] `backend/app/storage.py` 可以保存和读取文档
- [ ] `backend/app/storage.py` 可以保存和读取任务
- [ ] `backend/app/storage.py` 可以保存和列出 chunk
- [ ] `backend/app/storage.py` 可以按 document_id 过滤 chunk
- [ ] `backend/tests/test_storage.py` 覆盖保存与读取往返

## 任务 5：PDF 上传保存

- [ ] `backend/app/files.py` 能清洗 PDF 文件名
- [ ] `backend/app/files.py` 能拒绝非 PDF 文件
- [ ] `backend/app/files.py` 能把 PDF 字节保存到 `data/`
- [ ] `backend/app/routes_files.py` 能接收上传并创建文档与任务记录
- [ ] `backend/tests/test_files.py` 覆盖文件名清洗、拒绝非 PDF、保存路径

## 任务 6：MinerU 解析契约与分块

- [ ] `backend/app/mineru.py` 定义 `MinerUBlock`
- [ ] `MinerUBlock` 包含 `page`、`label`、`text`、`source_span`
- [ ] `backend/app/chunking.py` 能清洗文本中的空行与空白
- [ ] `backend/app/chunking.py` 能将块转换为可溯源 chunk
- [ ] `backend/app/chunking.py` 在页码无效时拒绝入库
- [ ] `backend/tests/test_chunking.py` 覆盖清洗、分块、页码校验

## 任务 7：bge-m3 嵌入适配器

- [ ] `backend/app/embeddings.py` 定义 `EmbeddingResult`
- [ ] `EmbeddingResult` 能输出 Qdrant 命名向量结构
- [ ] `backend/app/embeddings.py` 提供本地 bge-m3 嵌入器封装
- [ ] `backend/app/embeddings.py` 提供测试用 fake embedder
- [ ] `backend/tests/test_embeddings.py` 覆盖 dense 与 sparse 输出

## 任务 8：Qdrant 向量库适配层

- [ ] `backend/app/vector_store.py` 提供内存实现用于测试
- [ ] `backend/app/vector_store.py` 提供本地 Qdrant 实现
- [ ] `backend/app/vector_store.py` 能创建 dense 与 sparse collection
- [ ] `backend/app/vector_store.py` 能执行 dense+sparse 检索并用 RRF 融合
- [ ] `backend/app/vector_store.py` 能列出类别
- [ ] `backend/tests/test_vector_store.py` 覆盖写入、检索和类别读取

## 任务 9：构建流水线服务

- [ ] `backend/app/pipeline.py` 能组织清洗 → 分块 → 向量化 → 入库
- [ ] `backend/app/pipeline.py` 能更新任务步骤与进度
- [ ] `backend/app/pipeline.py` 成功后将文档状态更新为已入库
- [ ] `backend/app/pipeline.py` 失败后将任务和文档标记为失败
- [ ] `backend/tests/test_pipeline.py` 覆盖完整流转

## 任务 10：任务、文档和向量库接口

- [ ] `backend/app/routes_tasks.py` 提供任务启动与进度查询接口
- [ ] `backend/app/routes_documents.py` 提供文档列表、向量库列表、类别查询接口
- [ ] `backend/app/main.py` 注册任务、文档、文件接口
- [ ] 健康检查接口可访问
- [ ] `backend/tests/test_api.py` 覆盖健康检查、非 PDF 拒绝、文档列表返回

## 任务 11：问答服务与 Chat 接口

- [ ] `backend/app/qa.py` 提供兜底答案
- [ ] `backend/app/qa.py` 能构造带文档名与页码的提示词
- [ ] `backend/app/routes_chat.py` 提供 `POST /api/chat`
- [ ] `backend/app/main.py` 注册 chat 路由
- [ ] `backend/tests/test_qa.py` 覆盖兜底与提示词内容

## 任务 12：文档与 v1 基线结构

- [ ] `docs/需求说明.md` 是唯一需求文档
- [ ] `docs/版本迭代.md` 记录 v1
- [ ] `docs/架构/架构图-v1.md` 与 v1 需求一致
- [ ] `data/README.md` 说明真实 PDF 和本地 Qdrant 数据的放置方式
- [ ] `eval/sets/README.md` 说明评测集必须来自真实数据
- [ ] `eval/baseline/README.md` 说明基线结果的存放规则

## 任务 13：全量验证

- [ ] `pytest backend/tests -v` 通过
- [ ] `python -c "from backend.app.main import app; print(app.title)"` 可执行
- [ ] `docs/v1/spec.md`、`docs/v1/plan.md`、`docs/v1/tasks.md`、`docs/v1/checklist.md` 均存在
- [ ] 宪法要求的目录结构已创建
- [ ] 没有未批准的范围扩展
