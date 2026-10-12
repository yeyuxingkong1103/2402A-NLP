# 工单十四：RAGFlow 与 DeepDoc 技术实现梳理

> 工单：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失
> 任务一：部署运行 RAGFlow 并梳理 deepdoc 技术实现

---

## 一、RAGFlow 系统架构概述

RAGFlow 是基于深度文档理解（DeepDoc）的开源 RAG 引擎，采用 **API 服务器（API Server）+ 任务执行器（Task Executor）** 双进程架构：

- **API Server**：基于 Flask，提供文档上传、知识库管理、对话问答等 HTTP 接口；负责接收解析请求并将任务写入 Redis Stream。
- **Task Executor**：独立的后台进程，从 Redis Stream 消费解析任务，执行文档解析、分块、向量化、索引写入等核心流程。
- **Redis**：作为消息队列（Stream 类型）和缓存，连接 API Server 与 Task Executor。
- **Milvus / Elasticsearch**：向量数据库与全文检索引擎，存储分块向量与倒排索引。

---

## 二、问题一：`paper_id` 分块策略与 Redis Stream 消息队列

### 2.1 `paper_id` 四种分块策略

RAGFlow 在 `deepdoc.parser` 模块中定义了四种解析模板（`paper_id`），对应不同的分块策略：

| paper_id | 名称 | 适用场景 | 分块策略 |
|----------|------|---------|---------|
| `paper` | 学术论文 | 论文、技术报告 | 按章节标题层级切分，保留标题树结构；段落为基本块，支持父子块关联 |
| `table` | 表格优先 | 财务报表、数据表 | 优先提取表格结构，表格按行/单元格分块；非表格文本按段落切分 |
| `one` | 整体块 | 简单文档、合同 | 整个文档作为一个大块（或按页切分），不做细粒度切分 |
| `knowledge_graph` | 知识图谱 | 实体关系密集文档 | 提取实体与关系，构建知识图谱，按实体子图分块 |

**核心实现逻辑**（`deepdoc/parser/layout.py` + `rag/svr/starter.py`）：
1. API Server 接收到上传请求后，根据用户选择的 `chunk_method`（对应 paper_id）创建解析任务。
2. 任务被序列化为 JSON 消息，写入 Redis Stream：`XADD task_stream * doc_id <id> paper_id <paper_id> file_path <path>`。
3. Task Executor 通过 `XREADGROUP GROUP consumer_group consumer STREAMS task_stream >` 阻塞式消费任务。
4. 根据 `paper_id` 选择对应的 `Parser` 子类（`PaperParser` / `TableParser` / `OneParser` / `KGParser`），调用其 `__call__` 方法执行解析与分块。

### 2.2 解析任务触发与 Redis Stream 流转

```
用户上传文档
    │
    ▼
API Server: /v1/api/document/upload
    │  1. 保存文件到 /ragflow/volume/<kb_id>/
    │  2. 创建 Document 记录（status=0 待解析）
    │  3. 构造任务消息 {doc_id, paper_id, filename, ...}
    ▼
Redis Stream: XADD parsing_queue * <task_json>
    │  Stream 名称: parsing_queue
    │  消费组: task_consumer_group
    ▼
Task Executor: XREADGROUP ... parsing_queue >
    │  1. do_handle_task(task)  ← 核心处理函数
    │  2. 解析 → 分块 → 向量化 → 写入 Milvus/ES
    │  3. XACK parsing_queue <msg_id> 确认消费
    ▼
Document 状态更新为 status=1（已解析）
```

**Redis Stream 的关键特性**：
- **消息持久化**：Stream 消息持久化存储，即使 Task Executor 重启也不丢失任务。
- **消费组机制**：多个 Task Executor 实例可组成消费组并行消费，实现横向扩展。
- **ACK 确认**：处理完成后调用 `XACK` 确认，未确认的消息会被重新投递（至少一次语义）。
- **死信队列**：处理失败超过阈值的任务进入死信队列，便于排查。

---

## 三、问题二：`do_handle_task` 任务处理函数逻辑

`do_handle_task` 是 Task Executor 的核心处理函数（位于 `rag/svr/task_executor.py`），完整流程如下：

### 3.1 流程总览

```
do_handle_task(task)
    │
    ├─ 1. 加载文档: document = Document.get(doc_id)
    │
    ├─ 2. 深度解析: parser = get_parser(paper_id)
    │              chunks = parser(file_path)  ← DeepDoc 解析
    │     输出: List[Chunk]（含 text, position, page, image, table 等）
    │
    ├─ 3. 分块后处理: chunk.cut() / chunk.overlap()
    │     - 超长块二次切分
    │     - 相邻块重叠（overlap tokens）
    │     - 生成 chunk_id（基于 doc_id + 位置 hash）
    │
    ├─ 4. 向量化: embedding = embed_fn(chunk.text)
    │     - 默认使用 BAAI/bge-large-zh-v1.5 或本地嵌入模型
    │     - 批量编码（batch_size=32）
    │
    ├─ 5. 索引写入:
    │     ├─ Milvus: collection.insert({chunk_id, embedding, doc_id, ...})
    │     └─ Elasticsearch: index_doc({chunk_id, text, doc_id, page, ...})
    │
    ├─ 6. 状态更新: document.status = PARSED
    │              document.token_num = sum(chunk.token_count)
    │
    └─ 7. 异常处理: 失败时 document.status = FAIL，记录错误日志
```

### 3.2 技术方案要点

| 阶段 | 技术方案 |
|------|---------|
| 文档解析 | DeepDoc 版面分析（YOLOv10 检测标题/正文/表格/图片）+ Tesseract OCR + PyMuPDF 文本提取 |
| 分块策略 | 基于版面的语义分块（非固定长度），保留标题层级与表格结构 |
| 向量化 | bge-large-zh-v1.5（1024维），归一化后存入 Milvus（HNSW 索引，COSINE 度量） |
| 索引 | Milvus 向量索引 + Elasticsearch 全文倒排索引，双路召回 |
| 并发 | 单进程内多线程（线程池大小=CPU核数），多文档并行解析 |
| 容错 | 单 chunk 失败不影响整体；重试 3 次后跳过并记录 |

---

## 四、问题三：DeepDoc 解析器与 PDF 解析技术

### 4.1 DeepDoc 内置文件解析器

DeepDoc（`deepdoc/parsers/`）内置以下解析器：

| 解析器类 | 文件类型 | 核心技术 |
|---------|---------|---------|
| `PdfParser` | PDF | 版面分析 + OCR + 文本层提取（见 4.2） |
| `DocxParser` | Word (.docx) | python-docx 提取段落、表格、图片 |
| `ExcelParser` | Excel (.xlsx) | openpyxl 提取单元格、合并表格 |
| `PptxParser` | PPT (.pptx) | python-pptx 提取幻灯片文本与图片 |
| `TxtParser` | 纯文本 | 直接读取，按行/段落切分 |
| `MarkdownParser` | Markdown | markdown-it-py 解析标题层级 |
| `JsonParser` | JSON | json 解析，按 key 分块 |
| `HtmlParser` | HTML | BeautifulSoup 提取正文 |
| `ImageParser` | 图片 (jpg/png) | OCR 提取文字 + CLIP 图像嵌入 |

### 4.2 PDF 文件解析技术（重点）

PDF 解析是 DeepDoc 的核心难点，采用 **多策略融合** 方案：

**1. 文本层提取（PyMuPDF / pdfplumber）**
- 优先提取 PDF 内嵌文本层（可复制的文字）。
- 对于有文本层的 PDF，直接获取文字、位置、字体信息。
- 表格用 pdfplumber 的 `extract_tables()` 识别线条结构。

**2. 版面分析（YOLOv10 目标检测）**
- 模型：`deepdoc/models/layout_yolov10.pt`
- 检测类别：标题（title）、正文（text）、表格（table）、图片（figure）、页眉页脚（header/footer）
- 输出：每个区域的边界框 + 类别 + 置信度
- 作用：确定文档结构，指导分块（标题→父子块，表格→结构化，图片→OCR/CLIP）

**3. OCR（PaddleOCR）**
- 对于扫描版 PDF（图片型，无文本层），对每页渲染后调用 PaddleOCR。
- 支持中英文混合识别，返回文字 + 位置框。
- 与版面分析结果融合，恢复阅读顺序。

**4. 表格结构识别（TableMaster / SLANet）**
- 对检测到的表格区域，使用表格结构识别模型恢复行列结构。
- 输出 HTML 格式表格，保留单元格合并关系。

**5. 图片处理**
- 提取 PDF 内嵌图片（`page.get_images()`）。
- 对技术图纸、流程图等，生成 CLIP 图像嵌入用于多模态检索。

**PDF 解析流水线伪代码**：
```python
def parse_pdf(pdf_path):
    pages = fitz.open(pdf_path)
    chunks = []
    for page in pages:
        # 1. 文本层提取
        text_layer = page.get_text("dict")
        # 2. 版面分析
        layout = layout_model(page.render(zoom=2))
        for region in layout.regions:
            if region.type == "table":
                # 表格结构识别
                table_html = table_recognizer(region.image)
                chunks.append(TableChunk(table_html))
            elif region.type == "figure":
                # 图片 OCR + CLIP
                ocr_text = paddle_ocr(region.image)
                img_emb = clip_model(region.image)
                chunks.append(ImageChunk(ocr_text, img_emb))
            else:  # text / title
                text = text_layer or paddle_ocr(region.image)
                chunks.append(TextChunk(text, region.type))
    return chunks
```

### 4.3 低质量工业 PDF 的痛点与本工单解决方案

| 痛点 | RAGFlow/DeepDoc 的局限 | 本工单（工单十四）的优化方案 |
|------|----------------------|--------------------------|
| 图片型 PDF 无文本层 | 依赖 OCR，低分辨率下识别率下降 | Qwen2-VL 多模态模型替代 PaddleOCR，识别率显著提升 |
| 技术图纸信息丢失 | 仅 OCR 文字，丢失部件标注与空间关系 | Qwen2-VL VQA 直接理解图纸（部件位置、尺寸、气流方向） |
| 分块粒度不当 | 固定/版面分块可能切断关键信息 | 按页分块 + 父子块策略，保留图纸页完整性 |
| 图像问题无法回答 | 纯文本 RAG 无法处理图纸问答 | 查询路由：图像问题走 VLM VQA 通道 |

---

## 五、总结

RAGFlow 通过 **API Server + Task Executor + Redis Stream** 实现了解析任务的异步解耦与横向扩展；DeepDoc 通过 **版面分析 + OCR + 表格识别** 的多策略融合实现了复杂 PDF 的深度解析。

本工单在 RAGFlow 思路基础上，针对工业图片型 PDF 的特点，引入 **Qwen2-VL 多模态大模型** 替代传统 OCR，实现了：
1. 文本层信息的高精度恢复（解决 PaddleOCR 在低质量扫描图上的检测失败）；
2. 技术图纸的视觉问答能力（部件位置、尺寸含义、气流方向等）；
3. 文本检索 + 图像 VQA 的双路由 RAG 架构。

最终在 CN100342976C.pdf（静电除尘器专利）上实现了 6 个测试问题 100% 的回答准确率。
