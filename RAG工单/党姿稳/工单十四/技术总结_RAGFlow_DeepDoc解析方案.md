# RAGFlow 文档解析方案技术总结

> 本文基于 RAGFlow 开源仓库源码（`ragflow-main/ragflow-main/`）逐文件精读产出，所有结论均标注源码路径与函数名，关键片段直接摘自源码。分析范围覆盖：解析任务触发与 Redis Stream 入队链路、各 `parser_id` 的分块策略、`do_handle_task` 任务处理主流程、DeepDoc 解析器体系与 PDF 深度解析技术，以及对图片型工业 PDF 的解析局限分析。

---

## 一、RAGFlow 整体架构与解析触发链路

### 1.1 总体架构

RAGFlow 采用"API 服务 + 任务执行器 + 存储中间件"的分离式架构。从源码目录可以看到清晰的分层：

| 层 | 目录 | 职责 |
|---|---|---|
| API 服务层 | `api/apps/`、`api/db/services/` | Flask 蓝图暴露 REST 接口，业务逻辑与数据库读写 |
| 任务执行层 | `rag/svr/task_executor.py` | 独立进程，消费 Redis Stream 消息，执行解析/向量化/入库 |
| 文档解析层 | `deepdoc/parser/`、`deepdoc/vision/` | 各类文件解析器 + OCR/版面/表格识别模型 |
| 分块应用层 | `rag/app/` | 按 `parser_id` 把解析结果切成 chunk |
| NLP/检索层 | `rag/nlp/`、`rag/utils/` | 分词、docStoreConn 抽象（ES/Infinity/OpenSearch） |
| 中间件 | Redis / MySQL / MinIO / ES/Infinity | 队列、元数据、文件存储、向量库 |

### 1.2 解析触发链路（从点击"解析"到入队）

**入口 API**：前端点击解析后调用 `POST /document/run`，对应 `api/apps/document_app.py: run()`（第 615 行）。其核心逻辑是遍历 `doc_ids`，取出文档后调用：

```python
# api/apps/document_app.py: run() 第 666 行
DocumentService.run(tenant_id, doc_dict, kb_table_num_map)
```

SDK 侧另有对应接口 `api/apps/sdk/doc.py: parse()`（第 782 行），其第 847 行直接调用 `queue_tasks(doc, bucket, name, 0)`。

**分发入口**：`api/db/services/document_service.py: DocumentService.run()`（第 1088 行）：

```python
# api/db/services/document_service.py: run() 第 1103-1107 行
if doc.get("pipeline_id", ""):
    queue_dataflow(tenant_id, flow_id=doc["pipeline_id"], task_id=get_uuid(), doc_id=doc["id"])
else:
    bucket, name = File2DocumentService.get_storage_address(doc_id=doc["id"])
    queue_tasks(doc, bucket, name, 0)
```

即：若文档绑定的是"数据流（dataflow / 画布 Pipeline）"，走 `queue_dataflow`；否则走标准解析 `queue_tasks`。

**任务生成与入队**：`api/db/services/task_service.py: queue_tasks()`（第 360 行）是标准解析的核心入队函数，它负责：
1. 按文档类型计算任务切分粒度（见第二节）；
2. 生成 `parse_task_array` 任务数组，计算每个任务的 `digest`（xxhash 摘要，用于断点复用）；
3. `bulk_insert_into_db(Task, parse_task_array, True)` 把任务写入 MySQL 的 `task` 表；
4. `DocumentService.begin2parse(doc["id"])` 把文档置为 RUNNING；
5. 逐个 `REDIS_CONN.queue_product(...)` 推入 Redis Stream。

```python
# api/db/services/task_service.py: queue_tasks() 第 460-464 行
unfinished_task_array = [task for task in parse_task_array if task["progress"] < 1.0]
for unfinished_task in unfinished_task_array:
    assert REDIS_CONN.queue_product(
        settings.get_svr_queue_name(priority), message=unfinished_task
    ), "Can't access Redis. Please check the Redis' status."
```

### 1.3 Redis Stream 的 key 名与消息体结构

**Stream key**：定义于 `common/constants.py` 第 232-233 行与 `common/settings.py`：

```python
# common/constants.py
SVR_QUEUE_NAME = "rag_flow_svr_queue"
SVR_CONSUMER_GROUP_NAME = "rag_flow_svr_task_broker"

# common/settings.py: get_svr_queue_name() 第 128-134 行
def get_svr_queue_name(priority: int) -> str:
    if priority == 0:
        return SVR_QUEUE_NAME
    return f"{SVR_QUEUE_NAME}_{priority}"

def get_svr_queue_names():
    return [get_svr_queue_name(priority) for priority in [1, 0]]
```

即默认优先级队列为 `rag_flow_svr_queue`，高优先级队列为 `rag_flow_svr_queue_1`。消费组固定为 `rag_flow_svr_task_broker`。

**入队实现**：`rag/utils/redis_conn.py: RedisDB.queue_product()`（第 350 行）使用 Redis Stream 的 `XADD`，消息体被 JSON 序列化后放入 `message` 字段：

```python
# rag/utils/redis_conn.py: queue_product() 第 350-361 行
def queue_product(self, queue, message) -> bool:
    for _ in range(3):
        try:
            payload = {"message": json.dumps(message)}
            self.REDIS.xadd(queue, payload)
            return True
        ...
```

**标准解析消息体**（来自 `task_service.py: queue_tasks()` 内部的 `new_task()`，第 381-389 行，补入 digest/priority 后）：

```json
{
  "id": "<task uuid>",
  "doc_id": "<document id>",
  "progress": 0.0,
  "from_page": <起始页>,
  "to_page": <结束页>,
  "begin_at": "YYYY-MM-DD HH:MM:SS",
  "digest": "<xxhash64 摘要>",
  "priority": 0
}
```

注意：标准解析消息体**不含** `tenant_id`/`kb_id`/`parser_id` 等，这些字段在消费端由 `TaskService.get_task()` 从数据库关联查询补齐（见第三节）。而 dataflow/raptor/graphrag 类任务则会在消息体里附加 `task_type`、`tenant_id`、`kb_id`、`dataflow_id`、`doc_ids`、`file` 等字段（`queue_dataflow` 第 532 行、`queue_raptor_o_graphrag_tasks` 第 1110 行）。

### 1.4 任务执行器如何消费

任务执行器是独立进程 `rag/svr/task_executor.py`，入口 `main()`（第 1330 行）。其事件循环模式为：

```python
# rag/svr/task_executor.py: main() 第 1368-1383 行
report_task = asyncio.create_task(report_status())   # 心跳上报
while not stop_event.is_set():
    await task_limiter.acquire()                     # 并发闸门
    t = asyncio.create_task(task_manager())
    tasks.append(t)
```

`task_manager()`（第 1323 行）→ `handle_task()`（第 1191 行）→ `collect()`（第 173 行）→ `do_handle_task()`（第 948 行）。

**消费实现**：`collect()` 按顺序尝试两个队列，关键调用：

```python
# rag/svr/task_executor.py: collect() 第 177-188 行
svr_queue_names = settings.get_svr_queue_names()          # [rag_flow_svr_queue_1, rag_flow_svr_queue]
if not UNACKED_ITERATOR:
    UNACKED_ITERATOR = REDIS_CONN.get_unacked_iterator(svr_queue_names, SVR_CONSUMER_GROUP_NAME, CONSUMER_NAME)
try:
    redis_msg = next(UNACKED_ITERATOR)
except StopIteration:
    for svr_queue_name in svr_queue_names:
        redis_msg = REDIS_CONN.queue_consumer(svr_queue_name, SVR_CONSUMER_GROUP_NAME, CONSUMER_NAME)
        if redis_msg:
            break
```

底层 `RedisDB.queue_consumer()`（`redis_conn.py` 第 363 行）使用 `XREADGROUP`（`block=5` 阻塞、`count=1`），并自动 `XGROUP CREATE ... id=0 mkstream=True` 创建消费组。`get_unacked_iterator()`（第 410 行）用 `msg_id=0` 遍历消费组的 PENDING（未 ACK）消息，实现**崩溃重启后的消息重投**。任务处理完成后由 `handle_task()` 第 1231 行调用 `redis_msg.ack()` 触发 `XACK`。

**消费者命名与并发**：`CONSUMER_NAME = "task_executor_" + CONSUMER_NO`（第 113 行），并发受 `asyncio.Semaphore` 控制：

```python
# rag/svr/task_executor.py 第 122-129 行
MAX_CONCURRENT_TASKS = int(os.environ.get('MAX_CONCURRENT_TASKS', "5"))
MAX_CONCURRENT_CHUNK_BUILDERS = int(os.environ.get('MAX_CONCURRENT_CHUNK_BUILDERS', "1"))
task_limiter = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
chunk_limiter = asyncio.Semaphore(MAX_CONCURRENT_CHUNK_BUILDERS)
embed_limiter = asyncio.Semaphore(MAX_CONCURRENT_CHUNK_BUILDERS)
minio_limiter = asyncio.Semaphore(MAX_CONCURRENT_MINIO)
kg_limiter = asyncio.Semaphore(2)
```

---

## 二、各 parser_id 的分块策略与入队机制

### 2.1 任务切分粒度（queue_tasks 中的分页逻辑）

`task_service.py: queue_tasks()`（第 393-424 行）按文件类型与 `parser_id` 决定"一个文档拆成几个 Task"：

```python
# api/db/services/task_service.py: queue_tasks() 第 393-413 行
if doc["type"] == FileType.PDF.value:
    do_layout = doc["parser_config"].get("layout_recognize", "DeepDOC")
    pages = PdfParser.total_page_number(doc["name"], file_bin)
    page_size = doc["parser_config"].get("task_page_size") or 12
    if doc["parser_id"] == "paper":
        page_size = doc["parser_config"].get("task_page_size") or 22
    if doc["parser_id"] in ["one", "knowledge_graph"] or do_layout != "DeepDOC" \
            or doc["parser_config"].get("toc_extraction", False):
        page_size = 10 ** 9
    page_ranges = doc["parser_config"].get("pages") or [(1, 10 ** 5)]
    for s, e in page_ranges:
        ...
        for p in range(s, e, page_size):
            task = new_task()
            task["from_page"] = p
            task["to_page"] = min(p + page_size, e)
            parse_task_array.append(task)

elif doc["parser_id"] == "table":
    rn = RAGFlowExcelParser.row_number(doc["name"], file_bin)
    for i in range(0, rn, 3000):          # Excel 按 3000 行一段
        ...
else:
    parse_task_array.append(new_task())   # 其余类型整篇一个任务
```

要点：
- **PDF 默认每 12 页一个 Task**；`paper` 每 22 页；
- `one` / `knowledge_graph`、或非 DeepDOC 的 layout_recognize、或开启 `toc_extraction` 时，**整篇只发一个 Task**（`page_size=10**9`）；
- `table`（Excel/CSV）按 **3000 行**切段；
- 其他文档类型（docx/txt/md/html…）整篇一个 Task。

每个 Task 都会计算 `digest`（xxhash64，混入 `chunking_config` 与 `doc_id/from_page/to_page`），`reuse_prev_task_chunks()`（第 467 行）会在重跑时复用 digest 一致的已完成任务 chunk，避免重复向量化。

### 2.2 parser_id → chunker 类映射（FACTORY）

消费端 `rag/svr/task_executor.py` 中的 `FACTORY`（第 83-100 行）把 `parser_id` 映射到 `rag/app/` 下的分块模块：

```python
FACTORY = {
    "general": naive,                    ParserType.NAIVE.value: naive,
    ParserType.PAPER.value: paper,       ParserType.BOOK.value: book,
    ParserType.PRESENTATION.value: presentation,
    ParserType.MANUAL.value: manual,     ParserType.LAWS.value: laws,
    ParserType.QA.value: qa,             ParserType.TABLE.value: table,
    ParserType.RESUME.value: resume,     ParserType.PICTURE.value: picture,
    ParserType.ONE.value: one,           ParserType.AUDIO.value: audio,
    ParserType.EMAIL.value: email,       ParserType.KG.value: naive,   # 注意：KG 复用 naive
    ParserType.TAG.value: tag
}
```

`ParserType` 枚举定义于 `common/constants.py` 第 93-108 行。注意 **`knowledge_graph`（KG）的分块本身复用 `naive`**，它的"知识图谱"能力是在文档解析完成后由独立的 graphrag 任务（`queue_raptor_o_graphrag_tasks`，`api/apps/kb_app.py` 第 632 行）构建的。

### 2.3 四种重点 parser_id 的分块策略

#### （1）paper —— 学术论文

- 模块：`rag/app/paper.py`。
- PDF 解析器：`paper.py: class Pdf(PdfParser)`（第 31 行），`model_speciess = ParserType.PAPER.value`，因此加载的版面模型是 `layout.paper.onnx`（见第四节）。
- 默认 `parser_config`：`{"chunk_token_num": 512, "delimiter": "\n!?。；！？", "layout_recognize": "DeepDOC"}`（第 150-152 行）。
- 分块逻辑（`chunk()` 第 206-250 行）：识别标题（title）、作者（authors）、摘要（abstract）、章节（sections）。**摘要整体作为一个 chunk 不切分**（`paper.py` 文档字符串明确说明 "The abstract of the paper will be sliced as an entire chunk"）；正文按 `bullets_category` + `title_frequency` 推断章节层级，把同一章节层级的文本合并，再 `tokenize_chunks` 输出。表格经 `tokenize_table` 单独成 chunk。

#### （2）table —— 表格型文件（Excel/CSV/TXT）

- 模块：`rag/app/table.py`；解析器 `deepdoc/parser/excel_parser.py: RAGFlowExcelParser`。
- 分块逻辑（`table.py: chunk()` 第 361 行）：**每一行数据作为一个 chunk**（文档字符串："Every row in table will be treated as a chunk"）。列头做拼音化与类型后缀映射（`fields_map = {"text": "_tks", "int": "_long", ...}`），文本列分词、数值/日期列按类型存储。ES/OS 下启用 `_tks/_long/_flt/_dt` 类型化字段，Infinity 下存入 `chunk_data` JSON 列（第 462-495 行）。
- 注意 `table` 的 `chunk_token_num` 不适用（按行切），任务按 3000 行分页。

#### （3）one —— 整篇合并为一个 chunk

- 模块：`rag/app/one.py`。
- 默认 `parser_config`：`{"chunk_token_num": 512, "delimiter": "\n!?。；！？", "layout_recognize": "DeepDOC"}`。
- 核心：`tokenize(doc, "\n".join(sections), eng); return [doc]`（第 165-166 行）——**整篇文档只产出一个 chunk**，"One file forms a chunk which maintains original text order"。它的 PDF 解析器 `one.py: class Pdf`（第 30 行）调用 `self._layouts_rec(zoomin, drop=False)`（保留所有版面块），再用 `get_position` 按坐标排序输出。
- 由于只有一个 chunk，`queue_tasks` 对 PDF 强制 `page_size=10**9`（整篇一个 Task）。

#### （4）knowledge_graph —— 知识图谱

- 分块复用 `naive`（见 FACTORY）；默认 `parser_config` 由 `api/utils/api_utils.py` 第 407-413 行给出：`{"chunk_token_num": 8192, "delimiter": r"\n", "entity_types": ["organization","person","location","event","time"], ...}`。
- **chunk_token_num 显著更大（8192）**，因为知识图谱需要更长的上下文来抽取实体/关系。
- PDF 同样整篇一个 Task；解析完成后由独立的 `graphrag` 任务（`task_type="graphrag"`，`do_handle_task` 第 1053 行分支，调用 `run_graphrag_for_kb`）构建图谱。

#### （5）其余 parser_id 速览

| parser_id | 模块 | 支持文件 | 说明 |
|---|---|---|---|
| naive / general | `rag/app/naive.py` | docx/pdf/excel/txt/md/html/json/csv/jsonl | 通用：按 delimiter 切分再按 chunk_token_num 合并 |
| book | `rag/app/book.py` | docx/pdf/txt/html/doc | 按标题层级分块 |
| laws | `rag/app/laws.py` | docx/pdf/txt/md/html/doc | 按"第X条"等法条结构分块 |
| manual | `rag/app/manual.py` | pdf/docx | 手册：层级标题 |
| presentation | `rag/app/presentation.py` | pptx/pdf | 每页幻灯片一个 chunk |
| qa | `rag/app/qa.py` | xlsx/txt/csv/pdf/md/docx | 问答对，每问一答一 chunk |
| picture | `rag/app/picture.py` | 图片 | 图片 → 视觉多模态描述 |
| resume | `rag/app/resume.py` + `deepdoc/parser/resume/` | pdf/doc/docx/txt | 简历结构化抽取 |
| tag / email / audio | `rag/app/tag.py` / `email.py` / `audio.py` | 标签集/邮件/音频 | 标签是 KB 级能力，非文档分块 |

### 2.4 layout_recognize 参数

`layout_recognize`（默认 `"DeepDOC"`）决定 PDF 走哪条解析管线。其归一化在 `common/parser_config_utils.py: normalize_layout_recognizer()`：以 `@mineru` / `@paddleocr` 后缀区分同名视觉模型。可选值与对应解析器见 `rag/app/naive.py` 的 `PARSERS` 字典（第 221-228 行）：

```python
PARSERS = {
    "deepdoc": by_deepdoc,      # 默认，本地 DeepDoc（OCR+版面+TSR）
    "mineru": by_mineru,        # MinerU 服务
    "docling": by_docling,      # Docling
    "tcadp": by_tcadp,          # 腾讯云 TCADP
    "paddleocr": by_paddleocr,  # PaddleOCR-VL
    "plaintext": by_plaintext,  # 纯文本（pypdf 直接抽取，无 OCR）
}
```

其中 `by_plaintext` 使用 `deepdoc/parser/pdf_parser.py: PlainParser`（第 1419 行）或 `VisionParser`（第 1453 行）。**`PlainParser` 直接 `page.extract_text()`，不做 OCR** —— 这是理解第五节"图片型 PDF 丢信息"的关键。

---

## 三、do_handle_task 任务处理逻辑

`rag/svr/task_executor.py: do_handle_task()`（第 948 行）是整个解析任务的处理主体，挂有 `@timeout(60*60*3, 1)`（3 小时超时）。逐段梳理如下。

### 3.1 任务拉取与字段绑定（第 950-995 行）

`collect()` 已把消息经 `TaskService.get_task(msg["id"])` 补齐为完整任务字典（`task_service.py` 第 73 行：联表 Document/Knowledgebase/Tenant，取出 `parser_id`、`parser_config`、`embd_id`、`language`、`llm_id`、`pagerank` 等）。`do_handle_task` 开头先做特殊任务分流：

```python
# 第 952-958 行
if task_type == "memory":  await handle_save_to_memory_task(task); return
if task_type == "dataflow" and task.get("doc_id","") == CANVAS_DEBUG_DOC_ID:
    await run_dataflow(task); return
```

随后提取 `task_from_page/task_to_page/tenant_id/embd_id/language/llm_id/kb_id/doc_id` 等局部变量，并构造进度回调：

```python
# 第 977 行
progress_callback = partial(set_progress, task_id, task_from_page, task_to_page)
```

### 3.2 绑定 embedding 模型、初始化索引（第 984-995 行）

```python
embedding_model = LLMBundle(task_tenant_id, LLMType.EMBEDDING, llm_name=task_embedding_id, lang=task_language)
vts, _ = embedding_model.encode(["ok"])     # 探测向量维度
vector_size = len(vts[0])
init_kb(task, vector_size)                  # 建索引
```

`LLMBundle`（`api/db/services/llm_service.py`）是模型抽象层；`init_kb()`（第 563 行）调用 `settings.docStoreConn.create_idx(index_name, kb_id, vector_size, parser_id)` 创建/校验向量索引。

### 3.3 分支：RAPTOR / GraphRAG / 标准分块

- `task_type == "raptor"`（第 1001 行）：读 KB 的 `raptor` 配置，`should_skip_raptor()` 判断是否跳过，然后 `run_raptor_for_kb()`（第 765 行）用 `Raptor` 对已入库 chunk 做递归聚类摘要，生成 `raptor_kwd="raptor"` 的树状 chunk 再入库。
- `task_type == "graphrag"`（第 1053 行）：`run_graphrag_for_kb()`（`graphrag/general/index.py`）抽取实体关系、建图、可选社区报告，受 `kg_limiter` 限流。
- `task_type == "mindmap"`（第 1101 行）：占位。
- **`else` 标准分块**（第 1105-1127 行）：

```python
chunks = await build_chunks(task, progress_callback)     # 解析+分块
progress_callback(msg="Generate {} chunks".format(len(chunks)))
token_count, vector_size = await embedding(chunks, embedding_model, task_parser_config, progress_callback)
if task["parser_id"].lower() == "naive" and task["parser_config"].get("toc_extraction", False):
    toc_thread = executor.submit(build_TOC, task, chunks, progress_callback)
```

### 3.4 解析与分块：build_chunks（第 242 行）

`build_chunks()` 是"文件 → chunks 列表"的核心，挂 `@timeout(60*80, 1)`：

1. **体积校验**：`task["size"] > settings.DOC_MAXIMUM_SIZE`（默认 128MB）直接报错返回。
2. **取 chunker 类**：`chunker = FACTORY[task["parser_id"].lower()]`。
3. **拉取文件**：`File2DocumentService.get_storage_address()` 取 bucket/name，`get_storage_binary()` 经线程池从 MinIO 读取（超时/缺文件都会 `progress_callback(-1, ...)` 上报错误）。
4. **执行分块**（受 `chunk_limiter` 限流）：

```python
# 第 268-280 行
async with chunk_limiter:
    cks = await thread_pool_exec(
        chunker.chunk, task["name"], binary=binary,
        from_page=task["from_page"], to_page=task["to_page"],
        lang=task["language"], callback=progress_callback,
        kb_id=task["kb_id"], parser_config=task["parser_config"], tenant_id=task["tenant_id"])
```

即真正调用 `rag/app/<parser>.py: chunk(...)`。注意分块在工作线程池执行（CPU 密集），避免阻塞 asyncio 事件循环。

5. **图片落盘**：`upload_to_minio()`（第 299 行）对每个 chunk 计算 `id = xxhash.xxh64(content + doc_id)`，若含 `image` 字段则 `image2id()` 上传到 MinIO 并把 `img_id` 写回。
6. **可选增强**（受 parser_config 开关）：`auto_keywords`（关键词抽取）、`auto_questions`（问题生成）、`enable_metadata`（元数据抽取）、`tag_kb_ids`（打标签），均通过 `LLMBundle` 调 LLM，用 `chat_limiter` 限流，并用 `get_llm_cache/set_llm_cache` 缓存结果。

### 3.5 向量化：embedding（第 569 行）

```python
# 第 594-603 行
for i in range(0, len(cnts), settings.EMBEDDING_BATCH_SIZE):   # 默认 16
    async with embed_limiter:
        vts, c = await thread_pool_exec(batch_encode, cnts[i: i + settings.EMBEDDING_BATCH_SIZE])
    ...
    callback(prog=0.7 + 0.2 * (i + 1) / len(cnts), msg="")
```

要点：
- 文本先做标题向量 `tts` 与正文向量 `cnts` 的**加权融合**：`vects = title_w * tts + (1 - title_w) * cnts`，`title_w` 来自 `parser_config.filename_embd_weight`（默认 0.1）。
- 批量大小 `EMBEDDING_BATCH_SIZE`（默认 16，`common/settings.py` 第 121 行）；单批 `batch_encode` 有 60 秒超时。
- 结果写入每个 chunk 的 `q_<维度>_vec` 字段；进度在 0.7~0.9 区间回传。
- 正文会被 `re.sub(r"</?(table|td|caption|tr|th)...>", " ", c)` 剥离 HTML 标签后再编码。

### 3.6 写入向量库：insert_chunks（第 870 行）

`insert_chunks()` 负责把 chunk 写入 docStore（ES/Infinity/OpenSearch，由 `settings.docStoreConn` 抽象）：

```python
# 第 911-927 行
for b in range(0, len(chunks), settings.DOC_BULK_SIZE):     # DOC_BULK_SIZE 默认 4
    doc_store_result = await thread_pool_exec(
        settings.docStoreConn.insert, chunks[b:b + settings.DOC_BULK_SIZE],
        search.index_name(task_tenant_id), task_dataset_id)
    if has_canceled(task_id): ...
    if b % 128 == 0:
        progress_callback(prog=0.8 + 0.1 * (b + 1) / len(chunks), msg="")
    ...
    TaskService.update_chunk_ids(task_id, chunk_ids_str)   # 记录已写入 chunk id
```

其中 `search.index_name(uid)`（`rag/nlp/search.py` 第 34 行）= `f"ragflow_{uid}"`。索引按 tenant 建、按 kb_id（`task_dataset_id`）做逻辑分区。此外还会先单独写入"母块"（`mom`/`mom_with_weight`，`available_int=0`），支持父子分块检索。

### 3.7 收尾、进度回传与异常处理

- **进度回传**：全程通过 `set_progress()`（第 141 行）→ `TaskService.update_progress()`（`task_service.py` 第 298 行）把 `progress`（0~1 浮点）与 `progress_msg`（追加式文本，最多 3000 行）写回 MySQL。`prog<0` 表示错误（如 `-1`）；`has_canceled()` 会检查 Redis 中的 `<task_id>-cancel` 键实现取消。最终 `progress_callback(prog=1.0, msg="Task done ...")`。
- **计数更新**：`DocumentService.increment_chunk_num(doc_id, kb_id, token_count, chunk_count, 0)` 累加 chunk 数与 token 消耗。
- **异常处理**：`handle_task()`（第 1191 行）包裹 `do_handle_task`，异常时展开 `exceptiongroup.ExceptionGroup`、调用 `set_progress(task_id, prog=-1, msg=f"[Exception]: {err_msg}")` 并用 `logging.exception` 记录；无论成败最终在 `finally` 里 `redis_msg.ack()`（第 1231 行）——**即使失败也 ACK**，避免消息无限重投（失败重试由 `TaskService.get_task` 的 `retry_count>=3` 机制在第 129 行控制）。
- **取消清理**：`do_handle_task` 的 `finally`（第 1171-1188 行）在任务被取消时，删除已写入 docStore 的该文档 chunk。

### 3.8 关键技术小结

| 技术 | 实现位置 | 作用 |
|---|---|---|
| Redis Stream + 消费组 | `redis_conn.py: queue_product/queue_consumer` | 队列解耦、`XREADGROUP` 竞争消费、`XPENDING` 重投 |
| 分布式任务分发 | `task_executor.py: main/collect` | 多执行器按优先级队列抢任务，`task_limiter` 控并发 |
| 进度上报 | `set_progress` → `TaskService.update_progress` | 实时进度/日志回写 MySQL |
| 心跳与存活检测 | `report_status()`（第 1245 行） | `zadd`/`sadd TASKEXE` + `RedisDistributedLock` 清理死节点 |
| 向量库抽象 | `settings.docStoreConn`（`common/settings.py` 第 244-268 行） | 屏蔽 ES/Infinity/OpenSearch/OB 差异 |
| embedding 批处理 | `embedding()` + `embed_limiter` | 分批+限流+超时，标题/正文加权融合 |
| 断点复用 | `digest` + `reuse_prev_task_chunks` | 跳过未变配置的已完成分块 |
| LLM 结果缓存 | `get_llm_cache`/`set_llm_cache` | 关键词/问题/元数据/标签去重 |

---

## 四、DeepDoc 解析器体系与 PDF 解析技术

### 4.1 内置文件解析器一览（deepdoc/parser/）

| 解析器类 | 文件 | 解析类型 |
|---|---|---|
| `RAGFlowPdfParser`（别名 `PdfParser`） | `pdf_parser.py` | PDF（OCR+版面+表格+公式） |
| `PlainParser` | `pdf_parser.py:1419` | PDF 纯文本（pypdf） |
| `VisionParser` | `pdf_parser.py:1453` | PDF → 视觉大模型逐页描述 |
| `RAGFlowDocxParser` | `docx_parser.py` | docx（段落/图片/表格） |
| `RAGFlowExcelParser` | `excel_parser.py` | xls/xlsx/csv |
| `RAGFlowPptParser` | `ppt_parser.py` | ppt/pptx |
| `RAGFlowHtmlParser` | `html_parser.py` | html/htm |
| `RAGFlowMarkdownParser` / `MarkdownElementExtractor` | `markdown_parser.py` | md/markdown/mdx |
| `RAGFlowJsonParser` | `json_parser.py` | json/jsonl/ldjson |
| `RAGFlowTxtParser` | `txt_parser.py` | txt 及代码文件 |
| `figure_parser.py`（含 `VisionFigureParser`） | `figure_parser.py` | 图片/图表视觉增强包装器 |
| `MinerUParser` | `mineru_parser.py` | PDF（第三方 MinerU） |
| `DoclingParser` | `docling_parser.py` | PDF（IBM Docling） |
| `PaddleOCRParser` | `paddleocr_parser.py` | PDF（PaddleOCR-VL） |
| `TCADPParser` | `tcadp_parser.py` | PDF/表格（腾讯云 TCADP） |
| resume 子模块 | `resume/step_one.py`、`step_two.py`、`entities/` | 简历结构化抽取（教育/工作/项目等） |

导出集中在 `deepdoc/parser/__init__.py`。`resume` 子模块用两步流程 + 实体识别把简历拆成 basic/education/work/certificate/project 等结构。

### 4.2 支持的文件类型（按扩展名）

从 `rag/app/naive.py: chunk()`（第 789-983 行）的正则可归纳：`docx`、`pdf`、`csv/xlsx/xls`、`txt/py/js/java/c/cpp/h/php/go/ts/sh/cs/kt/sql`、`md/markdown/mdx`、`htm/html`、`json/jsonl/ldjson`、`doc`（经 Apache Tika）、`ppt/pptx`（presentation）。

### 4.3 RAGFlowPdfParser 完整解析流程

核心类 `deepdoc/parser/pdf_parser.py: RAGFlowPdfParser`（第 55 行）。其初始化（`__init__` 第 56-108 行）加载三类模型：

```python
self.ocr = OCR()                        # OCR：文本检测 + 文字识别
self.layouter = LayoutRecognizer("layout")  # 版面分析（DOMAIN 可为 layout / layout.paper）
self.tbl_det = TableStructureRecognizer()   # 表格结构识别 TSR
# 另加载 XGBoost 版面/文本拼接模型：
self.updown_cnt_mdl.load_model(os.path.join(model_dir, "updown_concat_xgb.model"))
```

**完整流程**（以 `naive.py: Pdf.__call__()` 第 547-579 行为例，与 `RAGFlowPdfParser.__call__` 第 1187 行一致）：

```
__images__()  →  _layouts_rec()  →  _table_transformer_job()
   →  _text_merge()  →  _naive_vertical_merge()  →  _extract_table_figure()
```

#### 步骤 1：`__images__()`（第 1047 行）——渲染与 OCR

- 用 `pdfplumber` 打开 PDF，逐页 `to_image(resolution=72*zoomin)`（zoomin 默认 3）渲染为 PIL 图；同时用 `page.dedupe_chars().chars` 抽取**PDF 内嵌文本字符**，并用 `_has_color()`（第 192 行）过滤"白底白字/无意义"字符。
- OCR 判定逻辑在 `__img_ocr_launcher`（第 1128 行）：`chars = self.page_chars[i] if not self.is_english else []`——**若判定为英文文档，则传空字符强制整页 OCR**；`is_english` 由第 1098-1105 行对随机抽样的字符做正则匹配统计得出。
- `__ocr()`（第 285 行）：先 `self.ocr.detect()` 检测文本框，再把内嵌字符按重叠关系归入对应框（`Recognizer.find_overlapped`）；对**没有文本的框**裁剪后送入 `self.ocr.recognize_batch()` 做 OCR 识别。这意味着**即使 PDF 内嵌 0 字符，只要文本框检测到内容，仍会走 OCR**。
- 若 `len(self.boxes)==0 and zoomin<9`，则递归以 `zoomin*3` 重试（第 1184 行），提高小字识别率。
- 同时用 `pypdf` 读取 `pdf.outline` 作为目录（outlines）。

#### 步骤 2：`_layouts_rec()`（第 350 行）——版面识别

```python
self.boxes, self.page_layout = self.layouter(self.page_images, self.boxes, ZM, drop=drop)
```

`LayoutRecognizer`（`deepdoc/vision/layout_recognizer.py`）输出 10 类版面：`Text / Title / Figure / Figure caption / Table / Table caption / Header / Footer / Reference / Equation`（labels 见第 34-46 行）。`header/footer/reference` 被标记为 garbage（第 56 行），`drop=True` 时被丢弃。识别结果按页保存到 `self.page_layout`。`paper` 模式加载的是 `layout.paper` 域模型。

#### 步骤 3：`_table_transformer_job()`（第 199 行）——表格结构识别 TSR

对版面识别为 `table` 的区域裁剪放大后送 `self.tbl_det(imgs)`（`TableStructureRecognizer`），识别表格的行/列/表头/跨行跨列单元格（labels：`table / table column / table row / table column header / table projected row header / table spanning cell`，见 `table_structure_recognizer.py` 第 31-38 行）。随后给每个单元格打 `R/H/C/SP` 标签（行号/表头/列/合并）。

#### 步骤 4：`_text_merge()` + `_naive_vertical_merge()`——文本合并

- `_text_merge()`（第 444 行）先 `_assign_column()`（第 358 行）用 **KMeans + silhouette_score** 聚类 x0 坐标判断多栏排版（单栏/双栏/多栏），再横向合并同一行相邻同版面的框。
- `_naive_vertical_merge()`（第 482 行）用规则（标点续行、缩进、`layoutno` 一致性、重叠率 0.3 阈值等）做上下行拼接，是当前版本实际生效的跨行合并逻辑。
- `_concat_downward()`（第 586 行）**在本版本中第 588 行直接 `return`**，其后的 XGBoost `_updown_concat_features` DFS 合并为**死代码**——即 `updown_concat_xgb.model` 被加载但主流程未实际使用。这一点需特别注意：跨行拼接实际由规则化的 `_naive_vertical_merge` 完成。

#### 步骤 5：`_extract_table_figure()`（第 762 行）——表格/图片抽取

- 从 `self.boxes` 中剥离 `table`、`figure` 框，按 `page-layoutno` 分组；跨页表格会合并（第 798-817 行）。
- 用 `TableStructureRecognizer.is_caption()` 识别题注，按最近邻（y/x 距离平方）把题注并入表格或图片（第 822-859 行）。
- `cropout()`（第 861 行）把区域裁剪为图片；表格再用 `self.tbl_det.construct_table(bxs, html=return_html)` 重建为 **HTML 或句子化文本**（便于 LLM 理解）。
- 返回 `(crop_image, [table_text])` 列表；`separate_tables_figures=True` 时图表分列。

#### 步骤 6：`__filterout_scraps()`（第 976 行）——过滤碎片

对剩余文本框做"有用性"判断（`usefull()`：有版面类型、宽度 > 页宽/3、高度 > 平均行高），并按 `proj_match()` 识别的标题层级做段落重组，剔除孤立碎片（"REMOVED"）。这是信息丢失的潜在环节之一。

#### 位置标签与图片提取

`_line_tag()`（第 961 行）为每个框生成 `@@页码\tx0\tx1\ttop\tbottom##` 位置标签；`extract_positions()`（第 1286 行）解析标签；`crop()`（第 1294 行）据位置标签从 `page_images` 裁剪出 chunk 对应图片（`pdf_parser.crop(ck, need_position=True)`，在 `naive.py: tokenize_chunks` 中被调用）。这样每个 chunk 都能携带其在原 PDF 中的截图，供多模态检索。

### 4.4 deepdoc/vision/ 下的模型与 ONNX 文件

`deepdoc/vision/` 的模型文件通过 `huggingface_hub.snapshot_download(repo_id="InfiniFlow/deepdoc", ...)` 下载到 `rag/res/deepdoc/`（`download_deps.py` 中 `repos` 含 `InfiniFlow/text_concat_xgb_v1.0` 与 `InfiniFlow/deepdoc`）。

| ONNX / 模型文件 | 加载类 | 作用 |
|---|---|---|
| `det.onnx` | `ocr.py: TextDetector`（第 415 行）| OCR 文本检测（DBNet，定位文本框） |
| `rec.onnx` | `ocr.py: TextRecognizer`（第 133 行）| OCR 文字识别（CRNN/SVTR，配 `ocr.res` 字典） |
| `ocr.res` | `TextRecognizer` | 识别字符字典（中英文等） |
| `layout.onnx` | `layout_recognizer.py: LayoutRecognizer` | 通用版面分析（10 类） |
| `layout.paper.onnx` | `layout_recognizer.py`（域 `layout.paper`）| 论文专用版面分析 |
| `tsr.onnx` | `table_structure_recognizer.py: TableStructureRecognizer`（第 42 行）| 表格结构识别（6 类） |
| `updown_concat_xgb.model` | `pdf_parser.py`（XGBoost Booster）| 上下行文本拼接判定（当前版本主流程未启用） |

其他实现变体（非 ONNX 默认路径）：

- `layout_recognizer.py: LayoutRecognizer4YOLOv10`（第 164 行）：YOLOv10 版版面识别（`deepdoc` 仓库内的 `layout.onnx` 亦可对应此结构）。
- **昇腾（Ascend）后端**：`AscendLayoutRecognizer`（第 241 行，加载 `layout.<domain>.om`）、`TableStructureRecognizer._run_ascend_tsr`（第 577 行，加载 `tsr.om`）。由环境变量 `LAYOUT_RECOGNIZER_TYPE` / `TABLE_STRUCTURE_RECOGNIZER_TYPE` 切换（`pdf_parser.py` 第 74-88 行）。
- `ocr.py: OCR`（第 536 行）是多 GPU 封装：`PARALLEL_DEVICES>0` 时按 GPU 数创建多个 `TextDetector`/`TextRecognizer` 实例。
- `figure_parser.py: VisionFigureParser` 不依赖 ONNX，而是调用视觉大模型（`LLMType.IMAGE2TEXT`）对图表生成描述，用于 `layout_recognize` 选择视觉模型（如 Qwen-VL）时。

---

## 五、对低质量/图片型工业 PDF 的解析局限与改进方向

### 5.1 现象：图片型 PDF 每页 0 可提取字符

工业 PDF（扫描件、CAD 导出图、图纸、表格截图、白底线条图）通常**没有文本层**：`pdfplumber.page.chars` 为空、`pypdf.extract_text()` 返回空字符串。这类文件在 RAGFlow 中的解析结果高度依赖于配置的解析管线。

### 5.2 RAGFlow 默认流程会怎么处理

**（A）默认 `layout_recognize = "DeepDOC"` 时——会走 OCR，原则上不丢**

- `RAGFlowPdfParser.__images__()` 会把每页渲染成位图，`page_chars` 为空 → `is_english` 判定为非英文（`sum(...)>len/2` 不成立）→ `__ocr()` 仍执行 `det.onnx` 文本框检测。
- 由于所有框 `chars` 为空，`__ocr` 会把它们全部送入 `rec.onnx` 做识别（第 332-343 行），因此**默认 DeepDOC 管线对纯扫描件是执行 OCR 的**，理论上能提取文字。
- 但 `mean_height` 在无内嵌字符时初值为 0，靠 `np.median([b["bottom"]-b["top"]])` 兜底（第 346-347 行）；若文本框检测本身失败（`bxs` 为空），则该页 `self.boxes.append([])`（第 291-293 行），**整页直接为空**。

**（B）若用户把 `layout_recognize` 设为 "Plain Text"——必然全丢**

- `PlainParser.__call__`（第 1419 行）仅 `page.extract_text()`，对无文本层 PDF 返回空行。
- 空 sections 经 `naive_merge` 后无 chunk，`do_handle_task` 走到 `progress_callback(1., msg=f"No chunk built from {name}")`（第 1110-1112 行），**文档解析"成功"但 chunk 数为 0**，信息完全丢失且不报错。

### 5.3 为什么会丢信息——具体环节分析

1. **OCR 前端（det.onnx）对工业图不友好**：DBNet 类检测器在密集细线、低对比、旋转/倾斜、表格线干扰、极小字号（工程图标注常 <6px）上召回率下降，导致框漏检；页面 `mean_width` 在无字符时硬编码为 8（第 1132 行），后续按宽度过滤会误删。
2. **OCR 识别（rec.onnx）字典受限**：`ocr.res` 主要覆盖中英文常用字；工业 PDF 中的符号（Φ、±、°、公差、专业符号）、特殊字体、竖排文字识别率低，易产出乱码。
3. **版面模型误判**：`LayoutRecognizer` 仅 10 类，且 `header/footer/reference` 直接丢弃（`drop=True`）。工业图纸中的图框、标题栏、明细表常被误判为 `figure` 或 `header` 而被裁剪/丢弃。
4. **`__filterout_scraps()` 主动删碎片**（第 1029 行：`mw/pw<0.35 且 mw<200` 的段落被 `REMOVED`）。工程图中零散标注被判定为"scrap"删除。
5. **`_extract_table_figure()` 的跨页/题注逻辑**：无题注的表格难以正确归属，表格被当作图片时只剩 OCR 文本，**表结构信息丢失**；`construct_table` 依赖 TSR 的单元格检测，对无框线表格（工业表格常用细线或无线）失效。
6. **公式识别**：`Equation` 仅是版面类别，DeepDoc **没有专门的公式识别模型**，公式区域只会被 OCR 成散字符，结构（上下标、分式）丢失。
7. **`_concat_downward` 的 XGBoost 跨行合并被短路**（第 588 行 `return`），复杂多栏/图文混排的阅读顺序由规则化 `_naive_vertical_merge` 处理，规则无法覆盖的排版会拼接错乱或断句错误。
8. **Chunk 体积与图片丢失**：若 chunk 图片裁剪失败（`crop()` 无有效位置标签时返回 None），该 chunk 只保留文本、丢失视觉信息（`image2id` 不触发）。
9. **任务分页导致上下文割裂**：默认每 12 页一个 Task，跨页表格/公式在任务边界处被切断，`reuse_prev_task_chunks` 的 digest 机制也意味着**若只改了 OCR 相关环境（模型升级）而未变 parser_config，旧 chunk 会被复用而不重算**。

### 5.4 改进方向

**配置层（零改造）**
- 图片型 PDF **必须使用默认 `DeepDOC`**，切勿选 `Plain Text`；可尝试 `MinerU` / `PaddleOCR-VL` / `Docling` / `TCADP` 等 `layout_recognize` 选项（`rag/app/naive.py: PARSERS`），它们对扫描件的版面与公式支持更强。
- 对纯图纸类文档考虑 `parser_id = "picture"` 或 `one`，走视觉大模型描述路径（`VisionParser` / `picture.py`），保留图形语义。
- 用 `task_page_size` 调小分页（缓解跨页割裂）；`chunk_token_num` 调小避免长 chunk 稀释检索。

**OCR/模型层**
- 提升渲染分辨率：`__images__` 的 `zoomin` 默认 3，可传入更大值（内部有 `zoomin<9` 递归放大重试逻辑，第 1184 行），改善小字召回。
- 替换/增强 OCR：`ocr.py` 的 det/rec 可换成 PP-OCRv5、含专业符号字典的模型；为工业场景训练专用字符集字典 `ocr.res`。
- 补充公式识别（如引入 LaTeX-OCR/UniMERNet）填补 `Equation` 类别的空白。
- 去掉 `_concat_downward` 的短路（第 588 行 `return`）或重新启用 XGBoost 拼接模型，恢复复杂排版的跨行合并。

**工程层**
- 增加**空 chunk 告警**：`do_handle_task` 在 "No chunk built" 时应标记文档为失败而非完成，避免"解析成功但 0 内容"的静默丢数据。
- 对图片型 PDF 增加前置探测（`page_chars` 全空即判定为扫描件），自动切换/提示更合适的解析器。
- 强化表格：对无框线工业表格引入基于视觉的 TSR（如基于 Transformer 的表格识别）替代仅靠框线几何的 `construct_table`。
- 保留原始页图并与 chunk 强绑定：确保 `crop()` 失败时回退到整页截图，避免视觉信息丢失，配合多模态检索弥补文本 OCR 损失。

---

## 附：关键源码位置索引

| 主题 | 位置 |
|---|---|
| 解析 API 入口 | `api/apps/document_app.py: run()`；`api/apps/sdk/doc.py: parse()` |
| 任务分发 | `api/db/services/document_service.py: DocumentService.run()` |
| 任务生成/入队 | `api/db/services/task_service.py: queue_tasks()`、`queue_dataflow()` |
| Redis Stream 名与消费组 | `common/constants.py`（`SVR_QUEUE_NAME`/`SVR_CONSUMER_GROUP_NAME`）、`common/settings.py: get_svr_queue_name()` |
| 队列读写 | `rag/utils/redis_conn.py: queue_product/queue_consumer/get_unacked_iterator` |
| 任务消费主循环 | `rag/svr/task_executor.py: main/handle_task/collect` |
| 任务处理主体 | `rag/svr/task_executor.py: do_handle_task()` |
| 解析分块 | `rag/svr/task_executor.py: build_chunks()` |
| 向量化 | `rag/svr/task_executor.py: embedding()` |
| 入库 | `rag/svr/task_executor.py: insert_chunks()` |
| 进度上报 | `rag/svr/task_executor.py: set_progress()`；`api/db/services/task_service.py: update_progress()` |
| parser_id 映射 | `rag/svr/task_executor.py: FACTORY`；`common/constants.py: ParserType` |
| 各分块器 | `rag/app/paper.py`、`table.py`、`one.py`、`naive.py` |
| PDF 解析 | `deepdoc/parser/pdf_parser.py: RAGFlowPdfParser` |
| 版面/OCR/TSR 模型 | `deepdoc/vision/layout_recognizer.py`、`ocr.py`、`table_structure_recognizer.py`、`recognizer.py` |
| 模型下载清单 | `download_deps.py`（`InfiniFlow/deepdoc`、`InfiniFlow/text_concat_xgb_v1.0`） |
