# RAGFlow DeepDoc 解析与任务调度技术总结

> 工单：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失（工单 14，V1.1-20260123）
> 依据源码：`D:\_RAG新工单素材\ragflow-main`（RAGFlow，infiniflow/ragflow）
> 面向对象：任务一「产出物 一、技术总结文档」。本文所有结论均可回源码逐条核对（行号附后）。

---

## 1. 背景：为什么低质量工业 PDF 会「信息丢失」

IMDR 数据集（`original_problems/documents`，1700 份）以**不可编辑的图片型 PDF** 呈现：文档通常**分辨率低、版式不一、图文交错**。
本工单目标文档 `CN100342976C.pdf`（静电除尘器专利，共 8 页）经实测**无文本层**，用 PyMuPDF 抽取文本为空 —— 意味着：

- 任何「直接抽文本」的解析路径（如 RAGFlow 的 `PlainParser` / `Plain Text` 选项）对它是**完全失效**的，图与文都拿不到；
- 必须走 **DeepDoc 的「版面识别 + OCR + 表格结构识别」** 多模型流水线，把图像像素还原成带位置标记（position tag）的结构化文本块；
- 而「第 7 页图片中部件 4 与部件 5 的位置关系」这类问题，答案**只存在于图里**，要求解析阶段把**图注/图像描述**也抽成可检索的文本，并在检索阶段能把「图 3 / 第 11 页图示」这类视觉引用与对应图像块关联起来。

下文分三部分对应工单的三项文档要求。

---

## 2. PDF 四种 `parser_id` 的分块策略与 Redis Stream 触发链路

### 2.1 触发入口与队列投递

「解析」动作最终落到 `TaskService.queue_tasks(doc, bucket, name, priority)`（`api/db/services/task_service.py`）。
它做三件事：**按类型切任务 → 算 digest → 投 Redis Stream**。

```python
# api/db/services/task_service.py: queue_tasks(...)
unfinished_task_array = [t for t in parse_task_array if t["progress"] < 1.0]
for unfinished_task in unfinished_task_array:
    assert REDIS_CONN.queue_product(                      # ← 投递
        settings.get_svr_queue_name(priority), message=unfinished_task
    ), "Can't access Redis..."
```

- 队列名：`common/constants.py` 定义 `SVR_QUEUE_NAME = "rag_flow_svr_queue"`；
  `common/settings.py` 的 `get_svr_queue_name(priority)`：priority==0 → `rag_flow_svr_queue`，否则 `rag_flow_svr_queue_{priority}`；
  `get_svr_queue_names()` 返回 `[svr_queue_1, svr_queue_0]`（**高优先级队列先被消费**）。
- 投递实现：`rag/utils/redis_conn.py: queue_product()` → `payload = {"message": json.dumps(task)}` → `REDIS.xadd(queue, payload)`
  —— 即 **Redis Stream**（不是 List），字段固定为 `message`（JSON 字符串），失败重试 3 次。

### 2.2 任务如何按 `parser_id` 切分（分块策略的「粒度」）

`queue_tasks` 内 `if doc["type"] == FileType.PDF.value:` 分支决定**一个文档切成几个解析任务**（每个任务 = 一段页范围）：

| `parser_id` | page_size（每个任务页数） | 含义 |
|---|---|---|
| 默认（naive/book/manual/laws/qa/…） | `task_page_size` 默认 **12** 页 | 按 12 页滚动切任务，并行解析 |
| `paper` | `task_page_size` 默认 **22** 页 | 论文类，页块更大，减少跨页结构被切断 |
| `one` / `knowledge_graph` | **10^9**（整篇一个任务） | 不分页，**全文/整文档一个任务**，保证不跨任务丢上下文 |
| 非 DeepDOC 版式识别（MinerU/Docling/PaddleOCR 等）或开启 `toc_extraction` | **10^9** | 整篇一个任务 |

```python
page_size = doc["parser_config"].get("task_page_size") or 12
if doc["parser_id"] == "paper":        page_size = ... or 22
if doc["parser_id"] in ["one","knowledge_graph"] or do_layout!="DeepDOC" \
   or doc["parser_config"].get("toc_extraction", False):
    page_size = 10 ** 9
```

- 非 PDF：`parser_id == "table"` 的 Excel/CSV 按 **每 3000 行** 一个任务；其余类型整篇一个任务。
- 每个任务带上 `from_page/to_page`，并计算 **task digest**（`xxhash.xxh64` 对分块配置 + doc_id + 页范围哈希）。`TaskService.reuse_prev_task_chunks` 用 digest 匹配历史任务，**命中则复用已算好的 chunk**，只对未完成的任务投队列（增量/幂等）。

### 2.3 消费者：Task Executor 如何消费

`rag/svr/task_executor.py`：
- 消费者名 `CONSUMER_NAME = "task_executor_" + CONSUMER_NO`，消费组 `SVR_CONSUMER_GROUP_NAME = "rag_flow_svr_task_broker"`（`common/constants.py:233`）。
- `collect()` 先 `REDIS_CONN.get_unacked_iterator(...)` 捞取**未 ACK 的历史消息**（崩溃/重启后重放），再 `queue_consumer(queue, group, consumer)` 走 `XREADGROUP`（count=1, block=5ms）阻塞拉取；处理完 `redis_msg.ack()`（`XACK`）确认。
- 心跳：`REDIS_CONN.zadd(CONSUMER_NAME, heartbeat, now_ts)` 维持存活，供「解析任务管理器」页展示。

> 一句话链路：**WebUI「解析」→ API → `queue_tasks` 按 parser 切页任务 → `XADD rag_flow_svr_queue[_priority]` → Task Executor `XREADGROUP` 消费 → `do_handle_task`**。

---

## 3. `do_handle_task`：解析→分块→向量化→索引全流程

入口 `rag/svr/task_executor.py:949 do_handle_task(task)`。主逻辑：

1. **按 `task_type` 分流**（决定走哪条流水线）：
   - `memory` → `handle_save_to_memory_task`；
   - `dataflow`（Canvas 画布调试）→ `run_dataflow`；
   - `raptor` → `run_raptor_for_kb`（层次化摘要树，对 chunk 做聚类 + LLM 摘要）；
   - `graphrag` → `run_graphrag_for_kb`（实体/关系抽取 → 知识图谱，支持 resolution/community）；
   - `mindmap` → 占位；
   - **其它（标准分块）** → `build_chunks` → `embedding` → 索引。
2. **公共前置**：`LLMBundle(tenant, LLMType.EMBEDDING, ...)` 绑定嵌入模型并 `encode(["ok"])` **探测向量维度**；`init_kb(task, vector_size)` 初始化知识库（写入 embedding 维度、校验 parser_id）。
3. **标准分块分支**：
   ```python
   chunks = await build_chunks(task, progress_callback)          # 解析 + 切块
   token_count, vector_size = await embedding(chunks, embedding_model, ...)  # 向量化
   # 后续写入 DocStore（ES / Infinity）+ 建索引；naive 且开启 toc_extraction 时另起线程 build_TOC
   ```
   - `build_chunks` 依据 `FACTORY[task["parser_id"].lower()]` 选分块器，从 **MinIO** 取文件二进制，用 `thread_pool_exec` 调 `chunker.chunk(...)`，并用 `chunk_limiter` 做**并发限流**。
   - `FACTORY`（`task_executor.py:82`）：`general/naive→naive`、`paper→paper`、`book→book`、`manual→manual`、`laws→laws`、`qa→qa`、`table→table`、`resume→resume`、`picture→picture`、`one→one`、`audio→audio`、`email→email`、`knowledge_graph(KG)→naive`、`tag→tag`。
4. **进度与取消**：全程 `set_progress(task_id, from_page, to_page, prog, msg)` 经 Redis 回写进度；`has_canceled(task_id)` 检查 `{task_id}-cancel` 键实现取消。

**用到的关键技术**：Redis Stream 生产者-消费者 + 消费组 + 未 ACK 重放；`xxhash` digest 复用；MinIO/S3 对象存储；`asyncio` + `ThreadPoolExecutor` + `chunk_limiter/kg_limiter` 信号量并发控制；`LLMBundle` 统一模型封装；DocStore 抽象（Elasticsearch / Infinity）承载向量与全文混合检索。

---

## 4. DeepDoc 内置解析器清单与 PDF 解析技术

### 4.1 内置解析器（`deepdoc/parser/`）

| 解析器 | 文件 | 能力 |
|---|---|---|
| PDF | `pdf_parser.py`（`RAGFlowPdfParser` / `PlainParser` / `VisionParser`） | 版式+OCR+表格结构，PDF 专用 |
| DOCX | `docx_parser.py` | Word，含表格/图片抽取 |
| Excel | `excel_parser.py` | xlsx，输出 DataFrame/HTML |
| PPT | `ppt_parser.py` | 演示文稿 |
| HTML | `html_parser.py` | 网页 |
| Markdown | `markdown_parser.py` | md（含 `MarkdownElementExtractor`） |
| JSON | `json_parser.py` | 结构化 |
| TXT | `txt_parser.py` | 纯文本 |
| 扩展（非默认导出） | `mineru_parser.py` / `docling_parser.py` / `paddleocr_parser.py` / `tcadp_parser.py` / `figure_parser.py` | 接第三方解析后端 / 图形解析 |

视觉模型（`deepdoc/vision/`）：`layout_recognizer.py`（版面元素检测）、`ocr.py` / `t_ocr.py`（文字识别）、`recognizer.py` / `t_recognizer.py`（检测/识别基类）、`table_structure_recognizer.py`（表格结构 TSR）、`postprocess.py`、`operators.py`、`seeit.py`。

> 注意：工单里 `paper_id` 应为 `parser_id` 笔误，取值 `paper/table/one/knowledge_graph` 与 `ParserType`（`common/constants.py:93`）一致：`PAPER="paper"`、`TABLE="table"`、`ONE="one"`、`KG="knowledge_graph"`。

### 4.2 PDF 解析技术流水线（`deepdoc/parser/pdf_parser.py`）

核心类 `RAGFlowPdfParser`，主链路：

1. `__images__(fnm, zoomin=3, page_from, page_to)`：用 PyMuPDF 把每页**按 3 倍缩放渲染成图像**（低分辨率 PDF 在这里被放大以利识别），并读取原生文本框（若有）。
2. `_layouts_rec(ZM)`：**版面识别**——用 DeepDoc 版面模型（XGBoost 检测器）输出文本/标题/表格/图/公式等 **layout 类别 + bbox**。
3. `__ocr(pagenum, img, chars, ZM)`：对文本区域做 **OCR**（`deepdoc/vision/ocr.py`），把像素转成字符 + 位置。
4. `_assign_column(boxes)`：**分栏归属**，把每块归到正确的列，避免多栏 PDF 串行。
5. 阅读顺序/合并类：`_text_merge`、`_naive_vertical_merge`、`_final_reading_order_merge`、`_concat_downward`（**跨页拼接**：把被分页切断的段落接回）、`_merge_with_same_bullet`。
6. `_table_transformer_job` + `_extract_table_figure`：**表格结构识别（TSR）**与**表格/图抽取**，可用 `crop()` 按位置裁出图像并打 **position tag**（`@@页码-x0x1y0y1##`），实现「文本块 ↔ 图/表」的精确关联。
7. `crop(text, ZM, need_position=True)` / `get_position(bx, ZM)`：把块还原成图像 + 位置，供前端高亮溯源。

- `PlainParser`：只做最朴素的原生文本抽取（**图片型 PDF 会得到空**，这正是工单 14 要避免的路径）。
- `VisionParser`：用 VLM 直接「看图成文」（`__images__` 被重写为调视觉模型），对应工单 16 的专用 VLM。

### 4.3 分块器（`rag/app/*.py`）与 parser_id 的对应

| parser_id | 模块 | 分块策略要点 |
|---|---|---|
| `paper` | `paper.py` | 专利/论文：先出 `title/abstract/sections/tables`；**摘要单独成整块不切**；正文按标题层级（`bullets_category` + `title_frequency` 判层级）聚合分段；`chunk_token_num` 默认 512，分隔符 `\n!?。；！？` |
| `one` | `one.py` | **整个文件一个 chunk，保持原文顺序**；docx/pdf/excel/txt 都支持；对第三方解析器（mineru 等）会把 `chunk_token_num` 置 0（不二次切） |
| `table` | `table.py` | Excel/CSV：**每一行 = 一个 chunk**；首行必须表头；按列类型加后缀（`_tks/_long/_kwd/_flt/_dt`）生成字段映射 |
| `knowledge_graph` | → `naive.py` | `KG` 在 FACTORY 映射到 `naive`；图表框 → 实体/关系抽取（graphrag 分支） |
| `naive`（通用） | `naive.py` | 通用：按 `delimiter` 切片 + `chunk_token_num` 控制大小；支持 `table_context_size`/`image_context_size` 把**表/图上下文折进 chunk**；`analyze_hyperlink`、`toc_extraction` |

---

## 5. 面向本工单的优化方案（6 题 → 100%）

> 说明：以下为**方案设计 + 代码/配置位点**；最终数值须在 RAGFlow 实例跑通后回填（见 `00_提交说明与自检表.md` 待补事项）。

**A. 解析方法（根因，对应故障「图文丢失」）**
- 强制 `layout_recognize = DeepDOC`（或改用 `MinerU`/`VisionParser`），**禁用 `Plain Text`**；`task_page_size` 调大或对目标文档用 `one`/整篇，避免图与图注被分到不同任务。
- 打开图像语义：`vision_figure_parser_pdf_wrapper`（`rag/app/paper.py` / `naive.py` 内 `vision_figure_parser_*_wrapper_*`），为图块生成**文本描述**，让「图的语义」进入可检索文本。

**B. 分块策略**
- 关键页（第 7 页含图 3/图注）用 **整页块 / one 模式**，保证「图 + 图注 + 部件编号文本」在同一 chunk；
- 增大 `chunk_token_num`（512 → 1024）并在 `image_context_size`/`table_context_size` 设正数，把图注折进相邻文本块。

**C. 向量相似度权重 / 混合检索**
- 调 `filename_embd_weight`、`hybrid_similarity`（向量分与全文分加权），提高含「第 X 页图」「部件编号」的块命中；
- 可对查询做**术语/位置扩展**（识别「图3」「第11页」→ 生成图注增强查询）。

**D. ReRank**
- 开启 rerank（cross-encoder / bge-reranker）对 Top-K 精排，把真正含图注的块顶到 Top-1/3。

**E. Prompt**
- 命中含图像上下文时，在 Prompt 明确要求「结合以下技术图纸的描述分析」，并附图像描述摘要。

对应代码补丁位点见 `01_代码/RAGFlow_关键源码片段/`（含 `task_service.queue_tasks`、`task_executor.do_handle_task`、`pdf_parser`、`app/*.py`）。

---

## 6. 结论

- RAGFlow 以 **「API Server 投递 + Task Executor 消费」** 的解耦架构，通过 **Redis Stream** 实现解析任务的异步/并行/可重放；分块粒度由 `parser_id`（`paper`=22 页 / 默认 12 页 / `one`+`knowledge_graph`=整篇 / `table`=每 3000 行）控制。
- 低质量工业图片型 PDF 的信息丢失**根因在解析层**：必须走 DeepDoc 的 **版面识别 + OCR + 表格结构识别 + 位置标记** 流水线，而不是纯文本抽取；图/表要经视觉描述进入可检索文本。
- 6 题要 100%，需**解析（DeepDOC + 图像描述）→ 分块（关键页整块）→ 检索（混合权重）→ 重排（ReRank）→ Prompt** 全链路配合。
