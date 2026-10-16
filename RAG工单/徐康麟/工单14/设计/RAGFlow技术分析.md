# RAGFlow 技术分析（工单14 · 任务一）

> 工单编号：人工智能NLP-RAG-修复低质量工业PDF 的解析与信息丢失工单
>
> 分析对象：`E:\gao6gongdan\_共享输入\ragflow-main\`（只读，Apache-2.0，Infiniflow RAGFlow）
> 分析方法：**逐文件 grep/read 实读源码**，所有函数名、行号均来自真实源码；
> 凡与「常见说法」不一致之处，一律**以源码为准**并在文中标注。
> 分析日期：2026-10-12

---

## 0. 阅读须知：三处必须纠正的常见误解

工单原文的问法里隐含了几个流传较广但不准确的表述。为了「准确」，先把它们标出来：

| 常见说法（工单/网上） | 源码实际 | 证据 |
| --- | --- | --- |
| 「`task_executor.py` 里有 `chunk()` 分发，用 if/else 判断 parser_id」 | **不存在 `chunk()` 函数做分发**。分发靠模块级字典 `FACTORY`，在 `build_chunks()` 里一次查表 | `rag/svr/task_executor.py:83-100`（FACTORY）、`:248`（查表） |
| 「解析任务用 `task_executor.py` 的 `queue_consumer` 触发」 | `queue_consumer` 是 **Redis Stream 消费原语**，属 `RedisDB` 类，在 `rag/utils/redis_conn.py:363`；触发入队的是 **API 侧** `queue_tasks()` | `api/db/services/task_service.py:360`、`rag/utils/redis_conn.py:363` |
| 「`rag/app/knowledge_graph.py` 是 knowledge_graph 的分块器」 | **该文件不存在**。`rag/app/` 下无 `knowledge_graph.py`；`ParserType.KG = "knowledge_graph"` 在 `FACTORY` 里**映射到 `naive` 模块** | `ls rag/app/` 无该文件；`common/constants.py:107`、`task_executor.py:98` |

另外，工单提到「用到了 `@timeit`」——**源码中不存在名为 `timeit` 的装饰器**。实际用的是
`from timeit import default_timer as timer`（`rag/svr/task_executor.py:56`）+ 分布式锁装饰器
`@timeout`（来自 `common/connection_utils.py`）。下文一律按源码写。

---

## 1. 问题一：PDF 分块策略与任务触发入队

### 1.1 总览链路

```
[Web/API] 点击「解析」
   → api/apps/sdk/doc.py:782  parse()
   → api/db/services/task_service.py:360  queue_tasks(doc, bucket, name, priority)
        ├─ 按 parser_id / 文件类型切「任务页区间」（parse_task_array）
        ├─ bulk_insert_into_db(Task, ...)        # 任务落库
        ├─ DocumentService.begin2parse(doc_id)   # 文档置为解析中
        └─ REDIS_CONN.queue_product(settings.get_svr_queue_name(priority), message=task)
   → (Redis Stream: rag_flow_svr_queue)
[Task Executor 进程]
   → rag/svr/task_executor.py:1330 main() → :1323 task_manager() → :1191 handle_task()
   → :173 collect()  → RedisDB.queue_consumer(...)  # XREADGROUP 消费
   → :948 do_handle_task(task)
        → :242 build_chunks() → :248 chunker = FACTORY[task["parser_id"].lower()]
```

### 1.2 分块器分发：`FACTORY` 字典（不是 if/else）

`rag/svr/task_executor.py:83-100`：

```python
FACTORY = {
    "general": naive,
    ParserType.NAIVE.value: naive,          # "naive"
    ParserType.PAPER.value: paper,          # "paper"
    ParserType.BOOK.value: book,
    ParserType.PRESENTATION.value: presentation,
    ParserType.MANUAL.value: manual,
    ParserType.LAWS.value: laws,
    ParserType.QA.value: qa,
    ParserType.TABLE.value: table,          # "table"
    ParserType.RESUME.value: resume,
    ParserType.PICTURE.value: picture,
    ParserType.ONE.value: one,              # "one"
    ParserType.AUDIO.value: audio,
    ParserType.EMAIL.value: email,
    ParserType.KG.value: naive,             # "knowledge_graph" -> naive !!
    ParserType.TAG.value: tag
}
```

`ParserType` 定义在 `common/constants.py:93-108`，其中 `KG = "knowledge_graph"`（`:107`）。

分发点 `rag/svr/task_executor.py:248`（在 `build_chunks()` 内）：

```python
chunker = FACTORY[task["parser_id"].lower()]
```

随后以线程池方式调用统一入口 `chunker.chunk(...)`（`task_executor.py:269-280`）：

```python
cks = await thread_pool_exec(
    chunker.chunk,
    task["name"],
    binary=binary,
    from_page=task["from_page"],
    to_page=task["to_page"],
    lang=task["language"],
    callback=progress_callback,
    kb_id=task["kb_id"],
    parser_config=task["parser_config"],
    tenant_id=task["tenant_id"],
)
```

> 即：**每个 `rag/app/*.py` 模块都必须暴露 `chunk(filename, binary, from_page, to_page, lang, callback, **kwargs)`**，
> 这是分块器的统一契约。本文分析的四个 parser 全部符合该契约。

### 1.3 四种 parser_id 的分块策略

#### （1）`paper` —— 论文式结构化分块

文件：`rag/app/paper.py`，类 `Pdf(PdfParser)`（`:39` 起）、函数 `chunk()`（`:144`）。

解析阶段 `Pdf.__call__`（`paper.py:47-141`）的顺序是：

```python
self.__images__(...)                 # OCR/渲染
self._layouts_rec(zoomin)            # 版面识别
self._table_transformer_job(zoomin)  # 表格结构识别
self._text_merge()                   # 文本合并
tbls = self._extract_table_figure(True, zoomin, True, True)
column_width = np.median([b["x1"] - b["x0"] for b in self.boxes])
self._concat_downward()
self._filter_forpages()
```

分块逻辑（`paper.py:206-250`）——**先切章节，再按小节聚合**：

```python
bull = bullets_category([txt for txt, _ in sorted_sections])
most_level, levels = title_frequency(bull, sorted_sections)
sec_ids = []
sid = 0
for i, lvl in enumerate(levels):
    if lvl <= most_level and i > 0 and lvl != levels[i - 1]:
        sid += 1
    sec_ids.append(sid)

chunks = []
last_sid = -2
for (txt, _), sec_id in zip(sorted_sections, sec_ids):
    if sec_id == last_sid:
        if chunks:
            chunks[-1] += "\n" + txt      # 同一节内的行合并进同一 chunk
            continue
    chunks.append(txt)
    last_sid = sec_id
res.extend(tokenize_chunks(chunks, doc, eng, pdf_parser))
```

特点：
- **摘要（abstract）单独成一个不可再切的 chunk**（`paper.py:216-225`，注释见 `:148`）；
- 用「标题层级 + 出现频率」自动判定章节边界，**不按固定 token 数硬切**；
- 双栏排版会被重排（`paper.py:68-71` 调 `self.sort_X_by_page`）；
- 表格经 `tokenize_table()` 单独成块，并支持 `table_context_size` / `image_context_size`
  给表格/图片补上下文（`paper.py:251-254` 调 `attach_media_context`）。

#### （2）`table` —— 按行/表切分（Excel/CSV 为主）

文件：`rag/app/table.py`，函数 `chunk()`（`:361`），类 `Excel(ExcelParser)`（`:39`）。

docstring 明确（`:373`）：**"Every row in table will be treated as a chunk."**
分块粒度在任务层就已经按 **3000 行**切任务片（见 1.4 节 `task_service.py:415-422`）。

#### （3）`one` —— 整篇一个 chunk

文件：`rag/app/one.py`，函数 `chunk()`（`:58`）。
docstring（`:60-62`）：**"One file forms a chunk which maintains original text order."**

关键代码 `one.py:163-166`：

```python
doc = {"docnm_kwd": filename, "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))}
doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
tokenize(doc, "\n".join(sections), eng)   # 全文拼成一个 chunk
return [doc]
```

同时它支持 docx / pdf / xlsx / txt|md / html / doc 六类文件（`:66/88/126/131/138/144`）。
其内部 `Pdf.__call__`（`one.py:31-55`）在 `_layouts_rec` 时用 **`drop=False`**（保留全部版面框），
并按 `(page, top, left)` 排序输出，保证**原始阅读顺序**。

#### （4）`knowledge_graph` —— 分块仍走 `naive`，图谱由**独立任务类型**构建

这是最容易搞错的一点，分三层说明：

**a) 分块层：映射到 `naive`**（`task_executor.py:98`）。
`knowledge_graph` 作为 `parser_id` 提交解析时，实际执行的是 `rag/app/naive.py` 的 `chunk()`（`:736`），
即通用的「按 `chunk_token_num` + `delimiter` 做 `naive_merge`」策略（`:1043-1045`）。

**b) 任务层：`knowledge_graph` 被特殊对待成「整篇一个任务」**
`api/db/services/task_service.py:400-403`：

```python
page_size = doc["parser_config"].get("task_page_size") or 12
if doc["parser_id"] == "paper":
    page_size = doc["parser_config"].get("task_page_size") or 22
if doc["parser_id"] in ["one", "knowledge_graph"] or do_layout != "DeepDOC" or doc["parser_config"].get("toc_extraction", False):
    page_size = 10 ** 9        # 等价于「不切页，整篇一次处理」
```

**c) 图谱层：由 `task_type == "graphrag"` 的独立任务构建**
`task_executor.py:1053-1100`（`do_handle_task` 内的 `elif task_type == "graphrag":` 分支）
读取 `kb_parser_config["graphrag"]`（`use_graphrag`、`entity_types` 等）并调用 GraphRAG 抽取器；
抽取产物以 `knowledge_graph_kwd` 字段写入文档库
（取值 `graph` / `subgraph` / `entity` / `relation` / `community_report`，
见 `api/apps/kb_app.py:413,451` 与 `api/apps/chunk_app.py:466`）。
检索侧再据此切换检索器：`api/db/services/dialog_service.py:1040-1041`

```python
is_knowledge_graph = all([kb.parser_id == ParserType.KG for kb in kbs])
retriever = settings.retriever if not is_knowledge_graph else settings.kg_retriever
```

### 1.4 任务如何被切分、触发并放入 Redis Stream

#### Step 1：API 入口

`api/apps/sdk/doc.py:780-847`（`/datasets/<dataset_id>/chunks`，POST，函数 `parse()`）
在 `:847` 调用 `queue_tasks(doc, bucket, name, 0)`。
Web 端对应 `api/apps/document_app.py:860  async def parse()`。

#### Step 2：`queue_tasks()` 切任务

`api/db/services/task_service.py:360-464`。PDF 分支（`:393-424`）：

```python
if doc["type"] == FileType.PDF.value:
    file_bin = settings.STORAGE_IMPL.get(bucket, name)
    do_layout = doc["parser_config"].get("layout_recognize", "DeepDOC")
    pages = PdfParser.total_page_number(doc["name"], file_bin)
    page_size = doc["parser_config"].get("task_page_size") or 12
    if doc["parser_id"] == "paper":
        page_size = doc["parser_config"].get("task_page_size") or 22
    if doc["parser_id"] in ["one", "knowledge_graph"] or do_layout != "DeepDOC" or doc["parser_config"].get("toc_extraction", False):
        page_size = 10 ** 9
    page_ranges = doc["parser_config"].get("pages") or [(1, 10 ** 5)]
    for s, e in page_ranges:
        s -= 1; s = max(0, s); e = min(e - 1, pages)
        for p in range(s, e, page_size):
            task = new_task()
            task["from_page"] = p
            task["to_page"] = min(p + page_size, e)
            parse_task_array.append(task)
elif doc["parser_id"] == "table":
    file_bin = settings.STORAGE_IMPL.get(bucket, name)
    rn = RAGFlowExcelParser.row_number(doc["name"], file_bin)
    for i in range(0, rn, 3000):
        task = new_task(); task["from_page"] = i; task["to_page"] = min(i + 3000, rn)
        parse_task_array.append(task)
else:
    parse_task_array.append(new_task())
```

**默认 PDF 每 12 页一个任务；`paper` 22 页；`one`/`knowledge_graph` 整篇一个任务；
`table` 每 3000 行一个任务。**

随后计算 `digest`（`xxhash.xxh64`，`:428-438`）用于**重解析时复用旧 chunk**
（`reuse_prev_task_chunks()`，`:467-511`），并：
`:457` `bulk_insert_into_db(Task, parse_task_array, True)`
`:458` `DocumentService.begin2parse(doc["id"])`

#### Step 3：生产者——`queue_product`（XADD）

`api/db/services/task_service.py:460-464`：

```python
unfinished_task_array = [task for task in parse_task_array if task["progress"] < 1.0]
for unfinished_task in unfinished_task_array:
    assert REDIS_CONN.queue_product(
        settings.get_svr_queue_name(priority), message=unfinished_task
    ), "Can't access Redis. Please check the Redis' status."
```

`rag/utils/redis_conn.py:350-361`：

```python
def queue_product(self, queue, message) -> bool:
    for _ in range(3):
        try:
            payload = {"message": json.dumps(message)}
            self.REDIS.xadd(queue, payload)      # <-- Redis Stream 写入
            return True
        except Exception as e:
            logging.exception("RedisDB.queue_product " + str(queue) + " got exception: " + str(e))
            self.__open__()
    return False
```

#### Step 4：队列名与消费者组

| 常量 | 值 | 位置 |
| --- | --- | --- |
| `SVR_QUEUE_NAME` | `"rag_flow_svr_queue"` | `common/constants.py:232` |
| `SVR_CONSUMER_GROUP_NAME` | `"rag_flow_svr_task_broker"` | `common/constants.py:233` |
| `settings.get_svr_queue_name(priority)` | priority==0 → `SVR_QUEUE_NAME`；否则 `f"{SVR_QUEUE_NAME}_{priority}"` | `common/settings.py:128-131` |
| `settings.get_svr_queue_names()` | `[queue_1, queue_0]`（**优先消费高优先级队列**） | `common/settings.py:133-134` |

#### Step 5：消费者——`collect()` 与 `queue_consumer`（XREADGROUP）

`rag/svr/task_executor.py:173-234`（`collect()`）核心：

```python
svr_queue_names = settings.get_svr_queue_names()
if not UNACKED_ITERATOR:
    UNACKED_ITERATOR = REDIS_CONN.get_unacked_iterator(svr_queue_names, SVR_CONSUMER_GROUP_NAME, CONSUMER_NAME)
try:
    redis_msg = next(UNACKED_ITERATOR)      # ① 先补投递「已投递未确认(PEL)」的消息
except StopIteration:
    for svr_queue_name in svr_queue_names:
        redis_msg = REDIS_CONN.queue_consumer(svr_queue_name, SVR_CONSUMER_GROUP_NAME, CONSUMER_NAME)
        if redis_msg:
            break                            # ② 再从新消息 ">" 读
...
task = TaskService.get_task(msg["id"])       # ③ 用 msg 里的 id 回库取完整任务
```

`rag/utils/redis_conn.py:363-408`（`queue_consumer`）核心：

```python
try:
    group_info = self.REDIS.xinfo_groups(queue_name)
    if not any(gi["name"] == group_name for gi in group_info):
        self.REDIS.xgroup_create(queue_name, group_name, id="0", mkstream=True)   # 自动建组+建流
except redis.exceptions.ResponseError as e:
    if "no such key" in str(e).lower():
        self.REDIS.xgroup_create(queue_name, group_name, id="0", mkstream=True)
    elif "busygroup" in str(e).lower():
        logging.warning("Group already exists, continue.")

args = {"groupname": group_name, "consumername": consumer_name,
        "count": 1, "block": 5, "streams": {queue_name: msg_id}}
messages = self.REDIS.xreadgroup(**args)
```

`get_unacked_iterator()`（`redis_conn.py:410-434`）负责遍历 **PEL（pending entries list）**
把上次崩溃未 ack 的消息重投，`current_min` 从 `0` 开始递增推进。

**消息确认（XACK）**：`task_executor.py:1231  redis_msg.ack()`，位于 `handle_task()` 的 `finally` 语义之后
（`:1222-1231`）——即任务成功或异常都会 ack，异常信息已通过 `set_progress` 写回 DB。
另有 `requeue_msg()`（`redis_conn.py:447-453`）实现「重新入队 + ack 旧消息」。

#### Step 6：消费并发与会话管理

`rag/svr/task_executor.py:122-131`：

```python
MAX_CONCURRENT_TASKS = int(os.environ.get('MAX_CONCURRENT_TASKS', "5"))
MAX_CONCURRENT_CHUNK_BUILDERS = int(os.environ.get('MAX_CONCURRENT_CHUNK_BUILDERS', "1"))
MAX_CONCURRENT_MINIO = int(os.environ.get('MAX_CONCURRENT_MINIO', '10'))
task_limiter = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
chunk_limiter = asyncio.Semaphore(MAX_CONCURRENT_CHUNK_BUILDERS)
embed_limiter = asyncio.Semaphore(MAX_CONCURRENT_CHUNK_BUILDERS)
minio_limiter = asyncio.Semaphore(MAX_CONCURRENT_MINIO)
kg_limiter = asyncio.Semaphore(2)
```

消费者名称由启动参数决定：`:112-113`

```python
CONSUMER_NO = "0" if len(sys.argv) < 2 else sys.argv[1]
CONSUMER_NAME = "task_executor_" + CONSUMER_NO
```

心跳与积压上报：`:1245 report_status()` —— 每轮 `REDIS_CONN.sadd("TASKEXE", CONSUMER_NAME)` 注册自身，
并用 `REDIS_CONN.queue_info(settings.get_svr_queue_name(0), SVR_CONSUMER_GROUP_NAME)` 取
`pending` / `lag`（`:1262-1264`）。

---

## 2. 问题二：`do_handle_task` 主要逻辑与技术方案

`rag/svr/task_executor.py:948-1188`。装饰器：`@timeout(60 * 60 * 3, 1)`（**3 小时超时**）。

### 2.1 步骤拆解

| # | 步骤 | 代码位置 | 说明 |
| --- | --- | --- | --- |
| 0 | 任务类型短路 | `:950-958` | `memory` → `handle_save_to_memory_task`；画布调试 → `run_dataflow` |
| 1 | 构建 task context | `:960-977` | 取出 task_id / from_page / to_page / tenant_id / embd_id / language / llm_id / kb_id / doc_id / name / parser_config；建 `progress_callback = partial(set_progress, task_id, from_page, to_page)`（`:977`） |
| 2 | 取消检查 | `:979-982` | `has_canceled(task_id)` → 回调 -1 并 return |
| 3 | 绑定 embedding 模型并探针取维度 | `:984-993` | `LLMBundle(...)`（`:986`）+ `embedding_model.encode(["ok"])`（`:987`）→ `vector_size` |
| 4 | 初始化索引 | `:995` | `init_kb(task, vector_size)` → `:563 init_kb()` → `settings.docStoreConn.create_idx(...)` |
| 5 | 分支：dataflow / raptor / graphrag / mindmap | `:997-1104` | 见下 |
| 6 | **标准分块** | `:1105-1108` | `chunks = await build_chunks(task, progress_callback)` |
| 7 | 生成 embedding | `:1114-1125` | `token_count, vector_size = await embedding(chunks, embedding_model, task_parser_config, progress_callback)`，回调文案 `"Embedding chunks ({:.2f}s)"` |
| 8 | TOC 抽取（可选，异步线程） | `:1126-1127` | `parser_id=="naive" and parser_config["toc_extraction"]` → `executor.submit(build_TOC, ...)` |
| 9 | 写入文档库（ES/Infinity） | `:1132-1146` | `insert_chunks()`（`:870`）；日志 `"Indexing doc(...), page(...), chunks(...)"` |
| 10 | 更新计数与进度 | `:1148-1150` | `DocumentService.increment_chunk_num(doc, kb, token_count, chunk_count, 0)`；`progress_callback(msg="Indexing done ...")` |
| 11 | 收尾 TOC 结果入库 | `:1152-1157` | `toc_thread.result()` 再插入一次并 `increment_chunk_num(...,0,1,0)` |
| 12 | 完成埋点 | `:1159-1169` | `progress_callback(prog=1.0, msg="Task done ({:.2f}s)")` + 汇总日志 |
| 13 | `finally` 取消清理 | `:1171-1188` | 若任务被取消，从 docStore 删除该 `doc_id` 的所有 chunk |

任务类型分支细节：

- **`raptor`**（`:1001-1051`）：读 `kb_parser_config["raptor"]`，缺失则补默认配置
  （`max_token=256, threshold=0.1, max_cluster=64, random_seed=0, scope="file"`，`:1011-1020`）；
  结构化数据可用 `should_skip_raptor()` 跳过（`:1031`）；执行体 `run_raptor_for_kb()`
  （`:765`）包在 `async with kg_limiter:`（`:1040`）中。
- **`graphrag`**（`:1053-1100`）：读 `kb_parser_config["graphrag"]`，默认 `entity_types`
  为 `["organization","person","geo", ...]`（`:1065-1068`）。
- **`mindmap`**（`:1101-1104`）：仅 `progress_callback(1, "place holder")`。

### 2.2 用到的技术方案（逐项定位）

#### （a）Redis Stream 消息队列
生产者 `XADD`：`redis_conn.py:354`；消费者 `XREADGROUP`：`:388`；建组 `XGROUP CREATE`：`:371/374`；
确认 `XACK`：`:47`（`RedisMsg.ack`）与 `task_executor.py:1231`。
队列名/组名见 1.4 节表格。

#### （b）MinIO 对象存储
- 取文件：`task_executor.py:237-238`
  ```python
  async def get_storage_binary(bucket, name):
      return await thread_pool_exec(settings.STORAGE_IMPL.get, bucket, name)
  ```
  由 `build_chunks()` 在 `:251-252` 调用（`File2DocumentService.get_storage_address(doc_id=...)`）。
- 存图片：`:298-322` `upload_to_minio()` 内 `image2id(d, partial(settings.STORAGE_IMPL.put, tenant_id=...), d["id"], d["kb_id"])`。
- 存储实现可切换：`common/settings.py:125-126`
  ```python
  STORAGE_IMPL_TYPE = os.getenv('STORAGE_IMPL', 'MINIO')
  STORAGE_IMPL = None
  ```

#### （c）Elasticsearch / Infinity 双引擎
分块写入统一走抽象层 `settings.docStoreConn.insert(...)`（`:904`、`:912`），
索引名由 `search.index_name(tenant_id)` 生成（`:564`）。
引擎实现在 `rag/utils/es_conn.py` 与 `rag/utils/infinity_conn.py`；
`infinity_conn.py:37-39` 有对 `*_kwd` 关键词列的特殊处理。
切换方式（见仓库 `CLAUDE.md`）：环境变量 `DOC_ENGINE=infinity` 后重启容器。
**注意：没有 `@timeit` 装饰器**；计时用 `from timeit import default_timer as timer`（`:56`）。

#### （d）线程池并发
统一封装 `common/misc_utils.py:116-132`：

```python
def _thread_pool_executor():
    max_workers_env = os.getenv("THREAD_POOL_MAX_WORKERS", "128")
    max_workers = int(max_workers_env) if ... else 128
    if max_workers < 1: max_workers = 1
    return ThreadPoolExecutor(max_workers=max_workers)

async def thread_pool_exec(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    if kwargs:
        func = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(_thread_pool_executor(), func)
    return await loop.run_in_executor(_thread_pool_executor(), func, *args)
```

`do_handle_task` 内另建一个独立线程池用于 TOC：`:974  executor = concurrent.futures.ThreadPoolExecutor()`。

#### （e）`LLMBundle` 统一模型绑定
`api/db/services/llm_service.py` 的 `LLMBundle`，在 `task_executor.py` 中被用于：
- 绑定 embedding：`:986`（`LLMType.EMBEDDING`）
- 绑定 chat（关键字/问题/元数据抽取、RAPTOR）：`:342`、`:1038` 等
- 绑定 rerank：`rag/nlp/search.py` 检索链路

#### （f）embedding 计算与「标题加权」
`task_executor.py:569-620`（`embedding()`）：

```python
tts.append(d.get("docnm_kwd", "Title"))            # 标题文本
c = "\n".join(d.get("question_kwd", []))
if not c: c = d["content_with_weight"]
c = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", c)   # 去表格标签
...
vts, c = await thread_pool_exec(mdl.encode, tts[0:1])
tts = np.tile(vts[0], (len(cnts), 1))               # 标题向量广播
...
filename_embd_weight = parser_config.get("filename_embd_weight", 0.1)
title_w = float(filename_embd_weight)
vects = title_w * tts + (1 - title_w) * cnts        # 标题与正文的加权融合
```

批大小 `settings.EMBEDDING_BATCH_SIZE`（`common/settings.py:121`，默认 16），
写入字段 `d["q_%d_vec" % len(v)]`（`:619`）。

#### （g）chunk 插入与图片去重（`insert_chunks`，`:870-945`）
- **母块（mom / mom_with_weight）** 先单独入库，`available_int=0`（`:883-901`），`mom_id` 用
  `xxhash.xxh64(mom.encode("utf-8"))`；
- 批量写入按 `settings.DOC_BULK_SIZE`（`common/settings.py:120`，默认 4）；
- 每批写入后 `TaskService.update_chunk_ids(task_id, chunk_ids_str)`（`:927`）做**断点续传**；
- 失败/任务不存在时回滚：删除已插入 chunk + 并发 `delete_image`（`:930-942`）。
- chunk 主键：`:303-304`
  ```python
  d["id"] = xxhash.xxh64((chunk["content_with_weight"] + str(d["doc_id"])).encode("utf-8", "surrogatepass")).hexdigest()
  ```

#### （h）进度埋点与异常
`set_progress()`（`:141-170`）：

```python
cancel = has_canceled(task_id)
if cancel: msg += " [Canceled]"; prog = -1
if to_page > 0 and from_page < to_page:
    msg = f"Page({from_page + 1}~{to_page + 1}): " + msg
msg = datetime.now().strftime("%H:%M:%S") + " " + msg
TaskService.update_progress(task_id, {"progress_msg": msg, "progress": prog})
close_connection()
if cancel: raise TaskCanceledException(msg)
```

进度常用刻度：`0.7 + 0.2*(i+1)/len(cnts)`（embedding 批，`:603`）、
`0.8 + 0.1*(b+1)/len(chunks)`（入库批，`:919`）。

外层 `handle_task()`（`:1191-1231`）捕获异常并展开 `exceptiongroup.ExceptionGroup`：

```python
except Exception as e:
    FAILED_TASKS += 1
    err_msg = str(e)
    while isinstance(e, exceptiongroup.ExceptionGroup):
        e = e.exceptions[0]; err_msg += ' -- ' + str(e)
    set_progress(task_id, prog=-1, msg=f"[Exception]: {err_msg}")
finally:
    ...
redis_msg.ack()
```

#### （i）Task 状态流转

| 状态/字段 | 位置 | 含义 |
| --- | --- | --- |
| 建任务 | `task_service.py:381-389` | `progress=0.0`，`from_page/to_page`，`begin_at` |
| 落库 | `task_service.py:457` | `bulk_insert_into_db(Task, parse_task_array, True)` |
| 文档置解析中 | `task_service.py:458` | `DocumentService.begin2parse(doc_id)` |
| 运行中 | `task_executor.py:161` | `TaskService.update_progress(task_id, {...})` |
| 取消 | `task_executor.py:145` + `task_service.py:514-519` | Redis key `f"{t.id}-cancel"` = `"x"`，`has_canceled()` 读（`:522`） |
| 分块 id 记录 | `task_executor.py:927` | `TaskService.update_chunk_ids()` |
| 完成 | `task_executor.py:1164` | `progress=1.0` |
| 失败 | `task_executor.py:1217` | `progress=-1` + `[Exception]:` 前缀 |

`Task` 的 `chunk_num` 累加在 `DocumentService.increment_chunk_num(...)`（`:1148`）。

---

## 3. 问题三：DeepDoc 深度解析模块

### 3.1 内置 parser 清单（`deepdoc/parser/` 实读）

`deepdoc/parser/__init__.py:17-39` 导出 9 个公共 parser：

| 源码文件 | 类 / 别名 | 能解析的文件类型 |
| --- | --- | --- |
| `pdf_parser.py` | `RAGFlowPdfParser`（别名 `PdfParser`，`:24`） | **PDF**（含扫描件，走 OCR） |
| `pdf_parser.py` | `PlainParser`（`:23`） | PDF 的**纯文本层**快速路径（不 OCR） |
| `pdf_parser.py` | `VisionParser(RAGFlowPdfParser)`（`:1453`） | PDF，改用视觉大模型做版面/文本理解 |
| `docx_parser.py` | `RAGFlowDocxParser`（`DocxParser`） | docx / doc |
| `excel_parser.py` | `RAGFlowExcelParser`（`ExcelParser`） | xls / xlsx / csv |
| `ppt_parser.py` | `RAGFlowPptParser`（`PptParser`） | ppt / pptx |
| `html_parser.py` | `RAGFlowHtmlParser`（`HtmlParser`） | html / htm |
| `json_parser.py` | `RAGFlowJsonParser`（`JsonParser`） | json |
| `markdown_parser.py` | `RAGFlowMarkdownParser`（`MarkdownParser`）+ `MarkdownElementExtractor` | md / markdown |
| `txt_parser.py` | `RAGFlowTxtParser`（`TxtParser`） | txt |
| `figure_parser.py` | `VisionFigureParser`（`:172`）及 4 个 `vision_figure_parser_*_wrapper` | 图/表区域，交给视觉模型生成描述 |
| `resume/` | 简历专用子模块（`entities/res/` 内含 `schools.csv`、`corp_baike_len.csv` 等词典资源） | 简历（PDF/DOCX）结构化抽取 |
| `docling_parser.py` | Docling 适配器 | 交由 Docling 处理的文档 |
| `mineru_parser.py` | MinerU 适配器 | 交由 MinerU 处理的 PDF |
| `paddleocr_parser.py` | PaddleOCR 适配器 | 交由 PaddleOCR 处理的 PDF |
| `tcadp_parser.py` | 腾讯云文档解析适配器 | 交由腾讯云 TCADP 处理的文档 |
| `utils.py` | 工具（`get_text` 等） | — |

> 工单里提到的 `picture_parser` 在 deepdoc 中**不存在独立文件**：图片解析由
> `deepdoc/parser/figure_parser.py` 的 `VisionFigureParser`（`:172`）承担，
> 对应分块器是 `rag/app/picture.py`（`ParserType.PICTURE`）。

### 3.2 重点：PDF 解析技术（`RAGFlowPdfParser`）

#### 3.2.1 `__call__` 主流程（`pdf_parser.py:1187-1195`）

```python
def __call__(self, fnm, need_image=True, zoomin=3, return_html=False):
    self.__images__(fnm, zoomin)                 # ① 渲染 + OCR
    self._layouts_rec(zoomin)                    # ② 版面识别
    self._table_transformer_job(zoomin)          # ③ 表格结构识别
    self._text_merge()                           # ④ 文本行合并
    self._concat_downward()                      # ⑤ 跨页/上下行拼接
    self._filter_forpages()                      # ⑥ 剔除页眉页脚
    tbls = self._extract_table_figure(need_image, zoomin, return_html, False)  # ⑦ 抽表与图
    return self.__filterout_scraps(deepcopy(self.boxes), zoomin), tbls          # ⑧ 去碎片
```

对照另一条更强的入口 `parse_into_bboxes()`（`:1197-1279`）——它在 `_text_merge/_concat_downward`
之后**多调用了 `_naive_vertical_merge(zoomin)`**（`:1216`），并额外做：
- `_extract_table_figure(True, zoomin, True, True, separate_tables_figures=True)`（`:1221`）
- 用 `min_rectangle_distance()`（`:1224`）把表/图**按最小矩形距离插回阅读顺序**（`:1243-1276`）
- 给每个 box 打 `position_tag` 与 `positions`（`:1270-1273`），支撑后续按位置裁剪。

#### 3.2.2 各子步骤的技术细节

**① `__images__`（`:1047-1185`）—— 渲染 + OCR 调度**
- 逐页渲染为图像，`zoomin` 默认 3（即 3 倍缩放）；
- OCR 以 `asyncio.gather` + 线程池并发（`:1107-1123`）：
  ```python
  async def __img_ocr(i, id, img, chars, limiter): ...
      await thread_pool_exec(self.__ocr, i + 1, img, chars, zoomin, id)
  ```
- 多 GPU 时按 `settings.PARALLEL_DEVICES` 做设备分片与信号量限流（`:71-72`、`:1142-1147`）；
- OCR 失败降级：`:1159 logging.error(f"Error in OCR: {e}")`，仍继续后续流程。

**② `__ocr`（`:285-348`）—— 检测 + 识别 + 合并**
```python
bxs = self.ocr.detect(np.array(img), device_id)            # 文本检测
...
b["box_image"] = self.ocr.get_rotate_crop_image(img_np, np.array([[left, top], ...], dtype=np.float32))
texts = self.ocr.recognize_batch([b["box_image"] for b in boxes_to_reg], device_id)   # 批量识别
```
本步骤依赖 `deepdoc/vision/ocr.py` 的 `OCR` 类，而 `OCR` 内部使用 **ONNX 模型**
（`deepdoc/vision/ocr.py`、`recognizer.py`、`layout_recognizer.py`、`tsr.py`）。

**③ `_layouts_rec`（`:350-357`）—— 版面区域识别**
```python
def _layouts_rec(self, ZM, drop=True):
    assert len(self.page_images) == len(self.boxes)
    self.boxes, self.page_layout = self.layouter(self.page_images, self.boxes, ZM, drop=drop)
```
`self.layouter` 来自 `LayoutRecognizer(recognizer_domain)`（`:79-88`），
由环境变量 `LAYOUT_RECOGNIZER_TYPE`（默认 `"onnx"`，可选 `"ascend"`）选择后端。
`RAGFlowPdfParser.__init__` 中（`:69, 88-89`）：
```python
self.ocr = OCR()
self.layouter = LayoutRecognizer(recognizer_domain)   # "layout" 或 "layout." + model_speciess
self.tbl_det = TableStructureRecognizer()
```

**④ `_table_transformer_job`（`:199-284`）—— 表格结构识别（TSR）**
```python
recos = self.tbl_det(imgs)                       # TSR 模型前向
...
eles = Recognizer.layouts_cleanup(self.boxes, eles, 5, ption)     # 版面框清洗
clmns = sorted([r for r in self.tb_cpns if re.match(r"table column$", r["label"])],
               key=lambda x: (x["pn"], x["layoutno"], x["x0"]))    # 表头列排序
clmns = Recognizer.layouts_cleanup(self.boxes, clmns, 5, 0.5)
```
并给表格内 box 打上 `R/H/C/SP` 标记（`R`=行、`C`=列、`H`=表头、`SP`=跨格，见 `:248` 注释）。

**⑤ `_text_merge`（`:444-481`）—— 同行合并**
按 `layoutno` 与 `layout_type` 合并相邻框；遇到 `table` / `figure` / `equation` 类型不合并（`:467`）。

**⑥ `_concat_downward`（`:586-689`）—— 向下拼接成段**
判定跨框是否应拼接，特征函数是 `_updown_concat_features(up, down)`（`:134-179`），
其中包含 `up["layout_type"] == down["layout_type"]`、是否同为 `text`、是否同为 `table` 等布尔特征
（`:147-151`）；拼接判定还要求同一 `layoutno`（`:644-645`）：
```python
if i - dp < 5 and up.get("layout_type") == "text":
    if up.get("layoutno", "1") == down.get("layoutno", "2"):
```
断连特征 `detach_feats = [b["x1"] < b_["x0"], b["x0"] > b_["x1"]]`（`:539`）。
该模型可由 XGBoost 模型提供（`:100-104`，模型文件 `rag/res/deepdoc/updown_concat_xgb.model`，
缺失时从 `InfiniFlow/text_concat_xgb_v1.0` 下载）。

**⑦ `_naive_vertical_merge`（`:482-562`）** —— 朴素的垂直相邻合并，仅在 `parse_into_bboxes` 中被调用。
另有 `_final_reading_order_merge`（`:563-585`）与 `_merge_with_same_bullet`（`:736-761`）。

**⑧ `_filter_forpages`（`:690-735`）—— 剔除页眉/页脚**
若某 box 未带 `layout_type`，则继承其子框类型（`:684-685`）。

**⑨ `_extract_table_figure`（`:762-936`）—— 抽表与图**
```python
if TableStructureRecognizer.is_caption(self.boxes[i]) or self.boxes[i]["layout_type"] in \
        ["table caption", "title", "figure caption", "reference"]:
    ...
if self.boxes[i]["layout_type"] == "table": ...
if need_image and self.boxes[i]["layout_type"] == "figure": ...
```
其中图注/表注识别依赖 `TableStructureRecognizer.is_caption()`；
`layout_type` 取值包含 `table caption`、`figure caption`。
表格还原：`:921`
```python
res.append((cropout(bxs, "table", poss),
            self.tbl_det.construct_table(bxs, html=return_html, is_english=self.is_english)))
```

**⑩ `__filterout_scraps`（`:976-1036`）—— 去碎片**：`proj_match()`（`:937`）与 `_line_tag()`（`:961`）配合。

#### 3.2.3 位置标记体系（对「图文关联」很关键）

- `_line_tag(bx, ZM)`（`:961-975`）生成形如 `@@页码\t左\t右\t上\t下##` 的位置标签；
- `extract_positions(txt)`（`:1286-1292`）用正则 `r"@@[0-9-]+\t[0-9.\t]+##"` 反解出位置；
- `remove_tag(txt)`（`:1281-1283`）用 `re.sub(r"@@[\t0-9.-]+?##", "", txt)` 去掉标签；
- `crop(text, ZM, need_position)`（`:1294-1404`）按位置把图裁出来；
- `get_position(bx, ZM)`（`:1405-1418`）。

> **这正是 RAGFlow 把「图」和「说明文字」关联起来的底层机制**：位置标签既用于裁剪图片，
> 也用于在最终 chunk 里回溯「这段文字/这张图在哪一页哪一区域」。
> 本工单自研流水线的 `figure_parser.py` 与 `numeral_table.py` 借鉴了同一思路
> （见 `设计/低质量PDF解析修复方案.md`）。

#### 3.2.4 ONNX 模型清单

`RAGFlowPdfParser.__init__`（`:56-108`）加载的模型组件与用途：

| 组件 | 类 | 来源 | 用途 |
| --- | --- | --- | --- |
| 文本检测 `det` | `OCR`（`deepdoc/vision/ocr.py`） | ONNX | 定位文本行框 |
| 文本识别 `rec` | `OCR.recognize_batch` | ONNX | 行内字符识别 |
| 版面分析 `layout` | `LayoutRecognizer`（`layout_recognizer.py`） | ONNX | 区域分类（text/title/table/figure/…） |
| 表格结构 `tsr` | `TableStructureRecognizer`（`tsr.py`） | ONNX | 表格行列与跨格还原 |
| 上下拼接 | `updown_cnt_mdl` | XGBoost `updown_concat_xgb.model` | `_updown_concat_features` 打分 |

**重要事实**：这些 ONNX 模型**不在源码仓库里**（`ragflow-main` 全仓无 `.onnx` 文件，
仓库总大小约 68.7 MB）。`pdf_parser.py:100-104` 显示其获取方式：
先找 `rag/res/deepdoc/`，找不到就 `snapshot_download(repo_id="InfiniFlow/text_concat_xgb_v1.0", ...)`
从 HuggingFace 在线拉取。**这就是本机无法直接跑 DeepDoc 的根因**（见 `设计/环境事实.md`）。

#### 3.2.5 `naive_merge` 家族

`naive_merge` / `naive_vertical_merge` 是 RAGFlow 的**通用文本合并策略**，
定义在 `rag/nlp/__init__.py`，被 `rag/app/naive.py` 使用：

```python
# rag/app/naive.py:1043-1045
chunks = naive_merge(sections, int(parser_config.get("chunk_token_num", 128)),
                     parser_config.get("delimiter", "\n!?。；！？"))
res.extend(tokenize_chunks(chunks, doc, is_english, pdf_parser, child_delimiters_pattern=child_deli))
```

带图版本 `naive_merge_with_images`（`naive.py:1040`）；
docx 版本 `naive_merge_docx`（`naive.py:812`）。

而 `RAGFlowPdfParser` 侧的方法名是 **`_naive_vertical_merge`**（`:482`），两者不同名、不同层：
前者在 `rag/nlp`（chunk 级合并），后者在 `deepdoc`（box 级纵向合并）。

---

## 4. 小结与本工单的落点

1. **分发**：`FACTORY[parser_id]` 查表（`task_executor.py:83-100, 248`）；
   `paper`→章节聚合、`table`→按行、`one`→整篇一块、`knowledge_graph`→**走 naive 分块 + 独立 graphrag 任务**。
2. **触发入队**：Web「解析」→ `queue_tasks()` 切页区间 → `REDIS_CONN.queue_product()` 即 `XADD` 到
   `rag_flow_svr_queue`；执行器 `collect()` 用 `XREADGROUP`（组 `rag_flow_svr_task_broker`）消费，
   并优先补投递 PEL 中的未确认消息。
3. **`do_handle_task`**：绑定 embedding → 建索引 → `build_chunks` → `embedding` → `insert_chunks`
   → 计数与进度 → 3 小时超时 + 取消清理；技术栈为 Redis Stream + MinIO + ES/Infinity +
   线程池（`THREAD_POOL_MAX_WORKERS` 默认 128）+ `LLMBundle`。
4. **DeepDoc PDF**：8 步串行流水线，OCR(det/rec) + layout + tsr 三套 ONNX 模型 + XGBoost 拼接模型，
   **模型需联网下载**——这是本机复现的硬约束。

对应到本工单任务二/三：由于本机无 ONNX/OCR 运行时，自研流水线**改用「字体模板匹配 + 版面投影」
在纯 Python 环境下恢复附图标记与图注关联**，设计对照关系见
`设计/RAGFlow解析流水线架构.md` 与 `设计/低质量PDF解析修复方案.md`。
