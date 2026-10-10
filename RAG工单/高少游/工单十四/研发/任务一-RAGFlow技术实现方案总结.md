# 任务一 · RAGFlow deepdoc 技术实现方案总结

**工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失工单**
**参考项目：RAGFlow（https://github.com/infiniflow/ragflow）**

---

## 1. RAGFlow 系统架构

RAGFlow 的系统架构由两个核心组件构成：

- **API Server**（`api/ragflow_server.py`）：提供外部接口与平台基本功能——知识库管理、文件上传、解析触发、检索问答编排；
- **Task Executor**（`rag/svr/task_executor.py`）：作为异步处理流水线，负责文件的解析、分块、向量化与索引写入。

两者通过 **Redis Stream 消息队列**解耦：API Server 把解析任务写入队列，Task Executor 从队列消费并处理。文件本体存储在 MinIO，元数据存储在 MySQL，任务经 Redis 队列分发，deepdoc 抽取内容与结构，分块后写入文档引擎 [$TRAE_REF](https://deepwiki.com/ragflow/ragflow)。

```
文件上传 → 存储(MinIO) + 元数据(MySQL)
        → 点击「解析」→ API Server 生成任务
        → 写入 Redis Stream 消息队列
        → Task Executor 消费
        → deepdoc 解析 → 分块 → 向量化 → 写入文档引擎
```

---

## 2. 解析任务如何触发并入队 Redis Stream

### 2.1 触发入口

文件上传后，在文件列表中点击「解析」按钮触发解析。API Server 端处理该请求时：

1. 读取文档的 `parser_id`（分块方法）与 `parser_config`；
2. 调用任务服务（`TaskService`）把文档切分为若干**任务（Task）**——通常按页范围切分，形成多个可并行处理的任务单元；
3. 对每个任务调用 `REDIS_CONN.queue_product(SVR_QUEUE_NAME, message=...)`，把任务序列化后写入 Redis Stream 队列。

Redis 连接层（`rag/utils/redis_conn.py`）提供 Stream 的写入、消费组、待处理消息遍历与取消标志等能力 [$TRAE_REF](https://doc.holiday/library/ragflow/anatomy-of-ingestion/)。

### 2.2 消费端

Task Executor 启动时以**消费者组**方式订阅队列：

```python
# rag/svr/task_executor.py（示意）
REDIS_CONN.queue_consumer(SVR_QUEUE_NAME, CONSUMER_NAME)
```

消费循环从 Stream 中读取消息（`PAYLOAD`），处理完成后调用 `PAYLOAD.ack()` 确认；同时周期性上报状态。任务执行器还负责收集任务、构建分块、补充增强信息、嵌入内容并写入文档引擎 [$TRAE_REF](https://doc.holiday/library/ragflow/the-big-picture/)。

### 2.3 parser_id 与分块策略

`parser_id` 决定文档采用哪种分块器（chunker）。RAGFlow 内置多种 parser_id，对应 `rag/app/` 下不同的分块模块：

| parser_id | 分块模块 | 适用文档 | 分块策略 |
|---|---|---|---|
| `naive` | `rag/app/naive.py` | 通用文档 | 按扩展名分派解析器，DeepDoc 解析后按通用规则分块 |
| `paper` | `rag/app/paper.py` | 学术论文 | 面向论文结构（标题/摘要/章节/参考文献）分块 |
| `table` | `rag/app/table.py` | 表格类文件 | 表格结构化抽取，按行/键值对分块 |
| `one` | `rag/app/one.py` | 任意文件 | 整份文件作为**单个**块 |
| `knowledge_graph` | `rag/app/knowledgegraph.py` | 知识图谱抽取 | 分块后做实体/关系抽取，构建图谱 |
| `book` / `laws` / `manual` / `picture` / `qa` / `resume` / `presentation` / `tag` / `email` / `audio` | 对应模块 | 书籍 / 法规 / 手册 / 图片 / 问答 / 简历 / 演示 / 标签 / 邮件 / 音频 | 各自的结构化分块策略 |

**四种重点 parser_id 的分块策略说明：**

- **paper（论文）**：DeepDoc 的 PDF 解析器以「论文模式」运行，重点识别标题、摘要、章节层级与参考文献；分块时保留章节上下文，使块内语义完整。
- **table（表格）**：对表格类文件（Excel/CSV 等）做结构化抽取，识别多级表头与合并单元格，通常以「一行一块」或「键值对」形式分块，便于字段型问题检索。
- **one（整体）**：不切分，把整份文件作为一个块。适用于短文档或需要整体语义的场景。
- **knowledge_graph（知识图谱）**：先按常规方式分块，再对块内容做实体与关系抽取（GraphRAG），构建知识图谱。该模式下任务除常规分块外，还会触发图谱抽取流程。

> 工单原文中的「paper_id」指 `parser_id`。

---

## 3. do_handle_task 的主要逻辑与技术实现

`do_handle_task` 是 Task Executor 的核心任务处理函数，负责单个任务从解析到入库的完整流程。主要逻辑如下：

1. **取任务与文档**：通过 `TaskService.get_task(task_id)` 获取任务，关联文档，建立进度回调 `progress_callback`；
2. **准备文件**：按 `doc.location` 从存储下载文件到临时目录，读取 `parser_config`；
3. **选择分块器**：依据 `parser_id` 从分块器字典中选取对应模块（如 `paper` → `rag/app/paper.py`），调用其 `chunk()` 函数。`chunk()` 返回 `(chunks, timed_out)`，`chunks` 为块列表，每块包含 `content_with_weight`、文档名、标题分词等字段；
4. **知识图谱分支**：若 `parser_id == knowledge_graph`，额外触发图谱抽取流程（`run_graphrag`）；
5. **向量化**：对块文本批量调用嵌入接口 `embedding_encode`，得到稠密向量；
6. **组装文档记录**：为每块组装含向量、分词字段、元数据的文档记录；
7. **写入文档引擎**：调用 `settings.docStoreConn.insert(docs, index_name, ...)` 写入索引；
8. **更新进度与计数**：`TaskService.update_progress()` 更新进度，`increment_indexed_chunk_num` / `update_chunk_num` 累加块计数；
9. **增强特性（可选）**：RAPTOR 递归摘要、自动关键词、自动问题生成等。

用到的关键技术实现方案：

| 技术点 | 说明 |
|---|---|
| 异步任务队列 | Redis Stream + 消费者组，支持确认（ack）、待处理消息重投与取消 |
| 分块器插件化 | `parser_id` → 分块模块的字典映射，按扩展名与配置分派 |
| 视觉解析 | deepdoc 提供 OCR、版面识别、表格结构识别三类视觉模型 |
| 批量嵌入 | 块文本批量编码，降低调用开销 |
| 文档引擎写入 | 向量与分词字段统一写入，支撑混合检索 |
| 进度上报 | 任务级进度回调，前端可见解析进度 |

---

## 4. DeepDoc 深度解析模块

### 4.1 内置解析器与可解析文件类型

DeepDoc 是 RAGFlow 内置的文档解析模块，覆盖 12 类文档、80+ OCR 语言 [$TRAE_REF](https://iideas18.github.io/wiki_repo/wiki/ragflow/20260506_151323/deepdoc_doc/index.html)。其 `deepdoc/parser/` 目录内置多类解析器：

| 解析器 | 文件 | 可解析类型 |
|---|---|---|
| `PdfParser` | `deepdoc/parser/pdf_parser.py` | PDF（含扫描件、复杂版式） |
| `PlainParser` | `deepdoc/parser/pdf_parser.py` | 纯文本 PDF（跳过 OCR） |
| `VisionParser` | `deepdoc/parser/pdf_parser.py` | 基于视觉 LLM 的 PDF 理解 |
| `DocxParser` | 基于 python-docx | Word 文档 |
| `ExcelParser` | `deepdoc/parser/excel_parser.py` | Excel / CSV |
| `PptParser` | `deepdoc/parser/ppt_parser.py` | PowerPoint |
| `MarkdownParser` | `deepdoc/parser/markdown_parser.py` | Markdown |
| `TxtParser` | —— | 纯文本 |
| `JsonParser` / `HtmlParser` | —— | JSON / HTML |

`rag/app/naive.py` 的 `chunk()` 函数是解析分派中心：按扩展名正则匹配文件类型，读取 `parser_config["layout_recognize"]` 选择 PDF 解析引擎，再映射到 `PARSERS` 注册表；未命中则回退 `by_plaintext()` [$TRAE_REF](https://deepwiki.com/wenjiecome/ragflow/4.2-document-parsing)。

### 4.2 PDF 解析引擎

RAGFlow 为 PDF 提供多种解析引擎，通过 `layout_recognize` 配置选择：

| 引擎 | 适用场景 | 处理方式 |
|---|---|---|
| **deepdoc**（默认） | 通用、高质量 | OCR + 版面识别 + 表格识别 |
| **naive** | 纯文本 PDF | 跳过 OCR/TSR/DLR，速度快 |
| **mineru** | 复杂版式、公式 | 外部 MinerU 服务 |
| **docling** | IBM Docling | Docling 流水线 |
| **tcadp** | 腾讯云 | 腾讯云文档解析 API |

DeepDoc（默认）在 PDF 上执行 **OCR、TSR（表格结构识别）、DLR（文档版面识别）** 三类视觉任务，适合含复杂版式、图像、扫描内容或表格的 PDF；Naive 则跳过这些任务，适合纯文本 PDF [$TRAE_REF](https://ragflow.docs-hub.com/docs/v0.20.5/select_pdf_parser/)。

### 4.3 PDF 解析技术要点

**（1）图像生成**：将 PDF 页面按可配置缩放（默认 3 倍）渲染为图像，使用 `pdfplumber` 渲染，配合全局锁保证线程安全。

**（2）OCR**：文本检测与识别均使用 ONNX 模型（`det.onnx` / `rec.onnx`）。检测模型将图像缩放到最大 960px、归一化到 [-0.5, 0.5]，输出二值掩码后经 DBPostProcess 转为多边形框并 NMS 去重；识别模型把文本区域裁剪缩放为 48×320，经 CTC 解码输出字符序列，字典约 6600+ 字符，批量处理 `rec_batch_num=16`。

**（3）版面识别**：使用 **YOLOv10** 目标检测模型，输入 1024×1024，输出 11 类版面标签——Text、Title、Figure、Figure caption、Table、Table caption、Header、Footer、Reference、Equation、Background。对每个 OCR 框计算与版面框的重叠率，取重叠最高（阈值 > 50%）的版面类型赋给该框；同时过滤噪声（如 `^•+$`、页码 `^[0-9]{1,2} / ?[0-9]{1,2}$`、URL），并可丢弃页眉页脚。

**（4）表格结构识别**：检测表格的整表、列、行、表头、跨行跨列单元等组件，为框打上 R（行）/ C（列）/ H（表头）/ SP（跨单元格）标签，再生成带 `rowspan`/`colspan` 的 HTML 表格，支持多级表头。

**（5）文本合并与阅读序**：`_text_merge()` 合并同一行水平相邻的框；`_naive_vertical_merge()` 合并同列垂直堆叠的框；`_assign_column()` 用 K-means 聚类检测多栏版式；`_concat_downward()` 借助 XGBoost 模型按阅读顺序拼接框。

**（6）输出格式**：解析输出标准化的「章节 + 表格」结构。章节为 `(text, position_tag)` 元组，位置标签语法为 `@@{page}\t{x_left}\t{x_right}\t{y_top}\t{y_bottom}##`，可拼接多个位置标签；表格为 `((image, html_or_rows), position_list)` 元组 [$TRAE_REF](https://deepwiki.com/wenjiecome/ragflow/4.2-document-parsing)。

### 4.4 DeepDoc 在本工单低质量 PDF 上的局限

本工单的工业 PDF 是**整页位图、无文本层**的低质量扫描件，且包含大量**工程附图**（部件编号 + 空间关系）。DeepDoc 的通用 PDF 解析在此类文档上存在两点不足：

- 附图页的视觉信息（部件编号之间的空间关系）未被结构化，导致「部件 4 相对部件 5 的位置」这类问题缺少可检索依据；
- 附图上的部件编号与正文中的部件名称缺少显式对应，图文关联错误。

本工单的研发系统在 deepdoc 方案基础上，补充了**附图语义解析**（部件编号 ↔ 部件名 ↔ 空间/次序关系）与**专利选择题定向作答**，正是针对这两点不足的修复。详细实现见 [`系统设计文档.md`](../设计/系统设计文档.md) 与 [`优化报告.md`](../优化/优化报告.md)。

---

## 5. 小结

RAGFlow 以 API Server + Task Executor 的双组件架构，通过 Redis Stream 解耦「任务生成」与「任务执行」；`parser_id` 决定分块策略，`do_handle_task` 串联解析、分块、向量化与索引写入的完整流程；DeepDoc 作为内置深度解析模块，以 OCR + 版面识别（YOLOv10）+ 表格结构识别为核心，覆盖 PDF 等 12 类文档，是处理复杂版式与扫描件的默认引擎。对低质量工业 PDF 的附图语义与图文关联，本工单在 deepdoc 方案上做了针对性增强。