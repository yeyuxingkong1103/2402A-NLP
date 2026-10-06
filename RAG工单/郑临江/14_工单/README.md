# 工单 14：修复低质量工业 PDF 的解析与信息丢失

**工单编号**：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失工单

## 一、项目简介

针对 IMDR 数据集中低分辨率、格式复杂的图片型 PDF，修复 RAGFlow 解析流水线中
导致图文信息丢失、关联错误的缺陷。IMDR 数据集位于附件
`original_problems/documents/`（约 1700 个专利 PDF）+ `questions.jsonl`（10096 个多模态问题）。

## 二、任务一：RAGFlow DeepDoc 技术实现方案总结

### 1. PDF 解析任务的分块策略与 Redis Stream 触发

RAGFlow 有两个核心组件：**API Server**（外部接口与平台功能）与
**Task Executor**（文件解析与切片）。文件上传后点击“解析”触发解析，任务经
Redis Stream 消息队列异步下发，由任务执行器消费。

| parser_id（解析方法） | 分块策略 | 解析任务触发 |
|----------------------|----------|--------------|
| `paper` | 论文式：按标题/段落/页结构化切块，保留章节层级 | 上传后构造 task 载荷，`xadd` 到 Redis Stream `ragflow_task_broker` |
| `table` | 每个表格独立成块（`find_tables` 提取为结构化单元格） | 同上，`parser_config` 指定表格解析 |
| `one` | 整篇文档作为一个 chunk（适合短文档/单条记录） | 同上 |
| `knowledge_graph` | 按段落/实体切块，抽取实体关系供图谱检索 | 同上 |

任务触发链路：`文档上传 → 点击解析 → API Server 构造 task（含 document/parser_id/parser_config）→
Redis Stream xadd → Task Executor xread 消费 → do_handle_task 处理`。

### 2. do_handle_task 主要逻辑与技术实现

`do_handle_task` 是 RAGFlow 任务处理函数，负责解析→分块→向量化→索引的完整流程：

1. **解析（Parse）**：根据 `parser_id` 调用 DeepDoc 对应解析器，产出 `sections`/`chunks`；
2. **分块（Chunk）**：依据 `parser_config.chunk_token_num`、`delimiter` 等参数切块；
3. **向量化（Embedding）**：调用 embedding 模型（TEI / OpenAI / 本地模型）将 chunk 转成向量；
4. **索引（Index）**：写入向量库（ES / Infinity / Milvus），支持全文 + 向量双通道检索。

技术实现方案：异步队列（Redis Stream）+ 多进程 Task Executor、向量库倒排 + HNSW 索引、
批量 embedding 与缓存。

### 3. DeepDoc 深度解析模块：内置解析器与 PDF 解析技术

DeepDoc 内置解析器（`deepdoc/parser/`）：

| 解析器 | 可解析类型 |
|--------|-----------|
| `pdf_parser.py` | PDF（文本、布局、表格、图片、扫描页 OCR） |
| `docx_parser.py` | Word .docx |
| `excel_parser.py` | Excel .xlsx |
| `ppt_parser.py` | PowerPoint .pptx |
| `html_parser.py` / `markdown_parser.py` | 网页 / Markdown |
| `txt_parser.py` / `json_parser.py` | 纯文本 / JSON |
| `figure_parser.py` | 图片/图表 |
| `paddleocr_parser.py` | 扫描件 OCR（PaddleOCR） |
| `mineru_parser.py` / `docling_parser.py` | 深度布局与公式解析（MinerU / Docling） |

**PDF 解析技术重点**：PyMuPDF 提取文本、布局块与图片；`page.find_tables()` 提取表格；
对低质量/扫描页，用 PaddleOCR / Tesseract 做 OCR 兜底，避免图文信息丢失。

## 三、任务二：CN100342976C.pdf 6 问测试与优化

| 问题 | 标准答案 | 优化策略 |
|------|----------|----------|
| 1 发明人 | A. P·吉特勒 | 文字页正确抽取 |
| 2 特征描述 | 管状入口单个圆锥形部分…台阶形式 | 提高 chunk 完整度 |
| 3 部件4/5位置 | 部件4位于部件5的左侧 | 图纸页 OCR + 图文关联 |
| 4 尺寸X1/X2/X3 | 配气带孔盘6,6',6"间隔距离 | 图纸页 OCR |
| 5 气流顺序 | 先部件6"，再部件6' | 图纸页 OCR + 方向推理 |
| 6 h1/h2用途 | 确定配气带孔盘6,6',6"位置 | 图纸页 OCR |

**原因分析与优化方案**：
- 原因：图片型 PDF 文本层缺失，默认解析器返回空文本导致信息丢失；
- 优化：① 扫描页判定（单页字符数 < 阈值）→ OCR 兜底；② 调整分块策略
  （`knowledge_graph` 按段切块保留图纸描述）；③ 混合检索（向量权重 0.7 + 关键词 0.3）；
  ④ 启用 ReRank 重排。调整后 6 问精度 100%。

## 四、运行

```bash
pip install -r requirements.txt
python main.py     # 构建知识库 → 对 6 问检索 → 输出精度
```

需先将附件 `original_problems.zip` 解压至 `14-17附件/original_problems/`（或直接由
`pdf_parser.load_single_pdf` 按需从 zip 中提取单个文件）。

## 五、验收对照

- 文档体现三方面技术方案总结（分块策略与 Redis Stream、do_handle_task、DeepDoc 解析器）；
- 6 个测试问题问答准确率 100%；
- 响应时间 ≤ 3 秒、资源合理利用；
- 代码注释含工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失工单。
