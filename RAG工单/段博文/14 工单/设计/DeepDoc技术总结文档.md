# RAGFlow DeepDoc 深度解析模块技术总结

> 工单编号：人工智能NLP-RAG项目-修复低质量工业PDF的解析与信息丢失
> 分析基于 RAGFlow 源码（Go 重构版）：`internal/deepdoc/`、`internal/parser/`、`internal/ingestion/`

## 一、RAGFlow 系统架构与任务触发机制

RAGFlow 由两个核心组件构成：**API Server**（外部接口与平台功能）和 **Task Executor**（文件解析与切片）。新版采用 Go 重构，旧版 Python 的 `do_handle_task` + Redis Stream 在 Go 版中由 `internal/ingestion/task/pipeline_executor.go` 取代。

### 1.1 任务触发流程

用户在文件列表点击「解析」按钮后，任务按以下链路触发：

1. **HTTP 入口**：`internal/handler/document.go` 接收解析请求，调用 `internal/service/document/document.go`；
2. **任务入队**：document service 将解析任务写入任务表（数据库持久化），状态置为 `PENDING`；
3. **任务消费**：`internal/ingestion/task/pipeline_executor.go` 的 `Run()` 循环轮询任务表，取出 PENDING 任务；
4. **流水线执行**：加载对应 chunk_method 的 Ingestion Pipeline DSL 模板（`internal/ingestion/pipeline/template/*.json`），按 DAG 拓扑依次执行 `File → Parser → Chunker → Extractor → Tokenizer`；
5. **状态回写**：每阶段执行结果写回数据库，最终状态置为 `DONE`，chunks 入库并向量化。

新版用数据库任务表替代了旧版的 Redis Stream 消息队列，简化了部署依赖。

### 1.2 `do_handle_task`（新版 PipelineExecutor）的主要逻辑

`pipeline_executor.go` 的核心执行步骤（对应旧版 `do_handle_task`）：

| 步骤 | 说明 | 关键代码 |
| --- | --- | --- |
| 1. 加载 DSL | 按 chunk_method 选择 pipeline 模板 JSON | `template/ingestion_pipeline_*.json` |
| 2. 解析文件 | 按文件类型分发到对应 Parser（PDF→DeepDoc） | `component/parser_dispatch.go` |
| 3. 分块 | 将解析输出送入 Chunker（按策略切片） | `component/chunker/` |
| 4. 提取元数据 | 可选的关键词/标签/摘要提取（Extractor） | `agent/templates/chunk_summary.json` |
| 5. 向量化 | Tokenizer 对 chunk 做 embedding + 全文索引 | `component/tokenizer/` |
| 6. 入库 | chunks 写入 Milvus/Elasticsearch | `service/chunk/chunk.go` |

## 二、不同 chunk_method 的分块策略

RAGFlow 通过 Ingestion Pipeline 模板定义分块策略。每个 `parser_id`（即旧版 `paper_id`）对应一套 `File → Parser → XxxChunker → Extractor → Tokenizer` 的 DAG 模板。

| chunk_method | Chunker 类型 | 分块规则 | 适用场景 |
| --- | --- | --- | --- |
| **paper**（论文） | TitleChunker | 按标题层级（#/##/###、第X章/节、1.1/1.2）递归切分，`method=group` 将同级标题下内容合并为一个 chunk | 学术论文、研究报告 |
| **table**（表格） | TableChunker | 按表格单元切片，每个表格或表格行组作为独立 chunk | 财务报表、统计文档 |
| **one**（全文） | OneChunker | 整个文档作为**单个 chunk**，不切分 | 短文档、需保留全文上下文 |
| **book**（书籍） | TitleChunker | 按章/节/条层级切分（中文编号规则：第X章/第X节/(一)(二)） | 长篇书籍、法典 |
| **manual**（手册） | ManualChunker | 按结构化章节/条款切分，保留编号层级 | 技术手册、操作指南 |
| **general**（通用） | GeneralChunker/TokenChunker | 按 token 数量切分（默认 500 token，重叠 100） | 无固定结构的普通文档 |
| **qa**（问答） | QAChunker | 按 Q&A 对切分，每个问答对独立成块 | FAQ、问答对文档 |
| **presentation**（演示） | PageChunker | 按幻灯片页面切分，每页一个 chunk | PPT、演示文稿 |
| **picture**（图片） | OneChunker | 图片/视频整体作为单个 chunk（含 OCR 文本） | 图片、扫描件 |
| **laws**（法律） | TitleChunker | 按条/款/项切分（第X条、第X款） | 法律法规 |
| **knowledge_graph**（知识图谱） | — | 旧版支持，新版通过 GraphRAG 插件实现实体/关系抽取 | 需图谱推理的场景 |

### 关键模板文件
- `internal/ingestion/pipeline/template/ingestion_pipeline_paper.json`
- `internal/ingestion/pipeline/template/ingestion_pipeline_table.json`
- `internal/ingestion/pipeline/template/ingestion_pipeline_one.json`
- `internal/ingestion/pipeline/template/ingestion_pipeline_book.json`
- `internal/ingestion/pipeline/template/ingestion_pipeline_qa.json`
- `internal/ingestion/component/schema/chunker.go`（Chunker 组件类型声明）

## 三、DeepDoc 深度解析模块分析

### 3.1 DeepDoc 内置解析器清单

`internal/parser/parser/` 目录下的解析器：

| 解析器 | 文件 | 支持格式 | 说明 |
| --- | --- | --- | --- |
| **PDF 解析器** | `pdf_parser_common.go` + DeepDoc | `.pdf` | 核心解析器，支持 DeepDOC/PaddleOCR/Marker/Docling 等多种解析后端 |
| XLSX 解析器 | `xlsx_parser.go` | `.xlsx` | 电子表格，支持 column_mode/column_roles |
| XLS 解析器 | `xls_parser.go` | `.xls` | 旧版 Excel |
| CSV 解析器 | `csv_parser.go` | `.csv` | 逗号分隔值 |
| PPTX 解析器 | `pptx_parser.go` | `.pptx` | PowerPoint 演示文稿 |
| Markdown 解析器 | `markdown_parser.go` | `.md` | Markdown 文档 |
| Email 解析器 | `email_parser.go` | `.eml/.msg` | 邮件 |
| Office 解析器 | `office/parser.go` | `.doc/.docx` | Word 文档（Go 绑定 LibreOffice/Unoconv） |

### 3.2 PDF 解析技术栈（重点）

DeepDoc 的 PDF 解析位于 `internal/deepdoc/parser/pdf/`，采用**多引擎协同**架构：

```
PDF 文件
  │
  ├─ 渲染层：renderer.go → renderer_pdfium.go（pdfium 渲染页面为图像）
  │           或 pdfoxide（纯 Go PDF 渲染引擎，pdf_oxide_engine.go）
  │
  ├─ OCR 层：parser_ocr.go（PaddleOCR/RapidOCR 文字识别）
  │           支持旋转矫正（parser_ocr_rotate.go）、warp 对齐（warp.go）
  │
  ├─ 版面分析层：layout/（boxes_sections.go、combined_column.go）
  │              检测标题/正文/图片/表格区域，支持多栏排版
  │
  ├─ 表格识别层：table/deepdoc_table_builder.go（TSR 表格结构识别）
  │              table_construct.go 构建行列网格，支持跨行跨列合并
  │
  ├─ 后处理层：util/garbled.go（乱码检测）、toc_header_footer.go（页眉页脚去除）
  │            dehyphen.go（断词修复）、watermark（水印去除）
  │
  └─ 输出：JSON / Markdown / HTML（含 bbox、page、type 元数据）
```

**核心技术实现要点**：

1. **PDF 渲染**：默认用 pdfium（C++ PDF 引擎，通过 cgo 绑定），备用 pdfoxide（纯 Go）。渲染分辨率可调（DPI），低分辨率 PDF 可放大渲染提升 OCR 准确率。

2. **OCR 识别**：`parser_ocr.go` 调用 PaddleOCR 中文模型，输出每个字符的 bbox + 文本。支持 `garbled.go` 乱码检测自动重试、`rotate.go` 旋转矫正（0/90/180/270）。

3. **版面分析**：`layout.go` + `boxes_sections.go` 将 OCR 结果按空间聚类为行/段，`combined_column.go` 处理多栏排版（工业文档常见双栏）。

4. **表格结构识别**：`deepdoc_table_builder.go` 用 TSR（Table Structure Recognition）检测 `table row`/`table column`/`table column header`/`spanning cell`，构建行列网格并处理合并单元格，输出 HTML 表格。

5. **图表与图片**：图片区域保留为 base64 或独立文件，供多模态推理使用（`flatten_media_to_text` 选项可将图片转文字）。

### 3.3 低质量工业 PDF 的解析难点与修复

工业 PDF（如 IMDR 数据集）的特点：**图片型（扫描件）、低分辨率、格式复杂、图文混排**。RAGFlow 解析时的信息丢失风险及 DeepDoc 修复策略：

| 问题 | 表现 | DeepDoc 修复 |
| --- | --- | --- |
| 图片型 PDF 无文本层 | `get_text()` 返回空或极少字符 | 强制走 OCR 路径（`parse_method=DeepDOC` 自动检测） |
| 低分辨率文字模糊 | OCR 识别错误率高 | 渲染时提高 DPI（200-300），`garbled.go` 乱码检测重试 |
| 多栏排版断句错误 | 左右栏文字串行拼接 | `combined_column.go` 版面分析按列分离 |
| 表格结构丢失 | 表格文字散乱无结构 | `deepdoc_table_builder.go` TSR 还原行列 |
| 页眉页脚干扰 | 重复内容进入 chunk | `toc_header_footer.go` 自动检测去除 |
| 跨页段落被切断 | 语义断裂 | 后处理按标点/缩进合并跨页段落 |

---

*本文档基于 RAGFlow Go 重构版源码分析，旧版 Python 的 `do_handle_task`/Redis Stream/paper_id 在新版分别对应 PipelineExecutor/数据库任务表/chunk_method。*
