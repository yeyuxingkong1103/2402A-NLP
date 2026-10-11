# RAGFlow 技术总结文档

---

## 1. PDF 分块策略与 Redis Stream 触发机制

RAGFlow 在处理 PDF 时，根据 `paper_id`（即解析方法）采用不同的分块策略：

| paper_id | 分块策略 | 适用场景 |
| --- | --- | --- |
| paper | 基于章节标题和段落语义分块，保留文档结构，块大小约 128~512 token | 学术论文、专利 |
| table | 每个表格作为独立块，保留表头，转换为 Markdown 或 HTML | 表格密集文档 |
| one | 整个文档作为一个块 | 短文档或需整体理解 |
| knowledge_graph | 先抽取实体关系，再按图结构分块 | 知识图谱构建 |

### 解析任务触发流程

1. 用户在 Web 界面点击"解析"按钮，API Server 接收请求。
2. API Server 创建解析任务，将任务元数据（文件路径、解析方法、分块参数等）序列化后，通过 `XADD` 写入 Redis Stream，队列名为 `ragflow:task_queue`。
3. Task Executor 作为消费者，通过 `XREADGROUP` 读取任务。
4. 任务完成后，通过 `XACK` 确认，并更新任务状态。

---

## 2. do_handle_task 函数主要逻辑与技术实现

`do_handle_task` 位于 `rag/svr/task_executor.py`，是任务执行器的核心，主要逻辑：

| 步骤 | 逻辑 | 技术实现 |
| --- | --- | --- |
| 任务解析 | 从 Redis Stream 读取任务，解析出文件路径、解析方法、分块参数 | - |
| 文档解析 | 调用 DeepDoc 对应解析器（如 RAGFlowPdfParser），提取文本、图片、表格 | 使用 PyMuPDF 或 pdfplumber 读取 PDF；使用 PaddleOCR 识别图片文字；使用版面分析模型（LayoutLM）识别标题、段落、表格区域；使用表格结构识别模型还原表格 |
| 文本分块 | 根据 `paper_id` 选择分块策略，调用 chunk 函数切分文本 | - |
| 向量化 | 调用 Embedding 模型（如 bge-large-zh）将分块转为向量 | - |
| 索引存储 | 向量和元数据存入 Elasticsearch 或 Infinity，建立全文索引 | - |
| 状态更新 | 更新任务状态，记录日志 | - |

**技术方案**：异步 IO、Redis Stream 消息队列、多进程/多线程、GPU 加速 OCR 和 Embedding。

---

## 3. DeepDoc 解析器体系及 PDF 解析技术

### 3.1 DeepDoc 内置解析器

`deepdoc/parser/` 目录下内置解析器：

| 解析器类型 | 文件 |
| --- | --- |
| PDF | pdf_parser.py |
| Word | docx_parser.py |
| Excel | excel_parser.py |
| PPT | ppt_parser.py |
| HTML | html_parser.py |
| JSON | json_parser.py |
| Markdown | markdown_parser.py |
| 纯文本 | txt_parser.py |
| 图片 | figure_parser.py |
| MinerU | mineru_parser.py |
| PaddleOCR | paddleocr_parser.py |
| 特定格式 | tcadp_parser.py |
| Docling | docling_parser.py |

### 3.2 PDF 解析技术流程

1. **预处理**：PDF 每页转为 300 DPI 图像。
2. **版面分析**：`layout_recognizer.py` 识别文本、标题、表格、图片区域。
3. **OCR 识别**：`ocr.py` 中 PaddleOCR 进行文字识别，支持中英文。
4. **表格识别**：`operators.py` + `postprocess.py` 还原表格结构。
5. **阅读顺序恢复**：按版面分析结果拼接文本。
6. **输出结构化数据**：生成 JSON 供分块和向量化。

**关键技术**：深度学习模型（YOLO、LayoutLM）、OpenCV、OCR 引擎。
