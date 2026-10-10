# 工单 14 · 代码说明与优化补丁位点

本目录归档了任务一技术分析所依据的 **RAGFlow 关键源码片段**（`RAGFlow_关键源码片段/`，文件名用 `__` 表示原路径层级，例如 `rag__svr__task_executor.py` 对应 `rag/svr/task_executor.py`），以及任务二需要在 RAGFlow 上做的**参数化调整位点**。

## 一、关键源码片段清单

| 片段 | 对应原路径 | 用途 |
|---|---|---|
| `api__db__services__task_service.py` | `api/db/services/task_service.py` | `queue_tasks()` 按 parser_id 切页任务 + `queue_product` 投 Redis Stream |
| `rag__svr__task_executor.py` | `rag/svr/task_executor.py` | 消费端 `collect()`、`do_handle_task()`、`build_chunks`、`FACTORY` 分块器映射、`embedding` |
| `rag__utils__redis_conn.py` | `rag/utils/redis_conn.py` | `queue_product`(XADD)/`queue_consumer`(XREADGROUP)/未 ACK 重放 |
| `deepdoc__parser__pdf_parser.py` | `deepdoc/parser/pdf_parser.py` | PDF 解析流水线（渲染→版面→OCR→分栏→阅读顺序→表/图提取→position tag） |
| `deepdoc__parser__figure_parser.py` | `deepdoc/parser/figure_parser.py` | `vision_figure_parser_*`：用 IMAGE2TEXT 视觉模型为图块生成描述 |
| `deepdoc__parser____init__.py` | `deepdoc/parser/__init__.py` | DeepDoc 内置解析器导出清单（Docx/Excel/Html/Json/Markdown/Pdf/Ppt/Txt） |
| `rag__app__paper.py` | `rag/app/paper.py` | `paper` 分块：摘要整块 + 标题层级聚合 |
| `rag__app__one.py` | `rag/app/one.py` | `one` 分块：整篇一个 chunk |
| `rag__app__table.py` | `rag/app/table.py` | `table` 分块：每行一个 chunk + 字段映射 |
| `rag__app__naive.py` | `rag/app/naive.py` | `naive`（`knowledge_graph` 复用）分块 + 图/表上下文折入 |
| `rag__nlp__search.py` | `rag/nlp/search.py` | `retrieval()`/`hybrid_similarity()`/`rerank_by_model()`：向量权重与重排 |
| `common__constants.py` | `common/constants.py` | `ParserType` 枚举、`SVR_QUEUE_NAME`、`SVR_CONSUMER_GROUP_NAME` |

## 二、任务二「调整后的实现代码」补丁位点（配置优先，代码为辅）

> RAGFlow 大多可**零代码**通过知识库配置（WebUI「解析方法 / chunk 配置 / 检索配置」）达成；下列同时给出配置项与对应代码位置，便于理解与二次开发。

### 1) 解析方法（解析层，根因修复）
- **配置**：知识库「解析方法」选 **DeepDOC**（勿用 Plain Text）；必要时切 **MinerU / VisionParser**。
- **代码位点**：
  - `rag/app/*.py`：`layout_recognize` 决定选 `Pdf()`(DeepDOC) 还是别的解析器；`plain text` 会走 `PlainParser`（图片型 PDF 抽取为空）。
  - 目标文档为图片型 PDF ⇒ 必须走 OCR：`deepdoc/parser/pdf_parser.py :: _layouts_rec` + `__ocr`。

### 2) 图像语义进入检索（图文题关键）
- **代码位点**：`deepdoc/parser/figure_parser.py :: vision_figure_parser_pdf_wrapper`
  - 当知识库绑定了 **IMAGE2TEXT** 视觉模型时，会为图块生成**文本描述**；`parser_config.image_context_size > 0` 时把图前后文一并拼接。
- **配置**：给知识库设置一个 IMAGE2TEXT 模型（Qwen-VL 等），并把 `image_context_size` 设为 1~3。

### 3) 分块策略
- **配置**：`chunk_token_num`（512→1024）、`delimiter`、`table_context_size`、`image_context_size`；
  对「含关键图」的文档可用 **one/整篇** 解析（`task_page_size` 大）。
- **代码位点**：`rag/svr/task_executor.py :: FACTORY` + `api/db/services/task_service.py :: queue_tasks`（page_size 规则）。

### 4) 向量相似度权重 / 混合检索
- **代码位点**：`rag/nlp/search.py :: retrieval(vector_similarity_weight=0.3, ...)`、`hybrid_similarity()`。
- **配置**：知识库「相似度阈值 / 混合权重」；WebUI 检索设置里的 `vector_similarity_weight`。

### 5) 重排（ReRank）
- **代码位点**：`rag/nlp/search.py :: rerank_by_model(rerank_mdl, ...)`、`rerank()`。
- **配置**：知识库绑定 **Rerank 模型**（如 bge-reranker），在检索时启用。

### 6) Prompt
- **代码位点**：`rag/prompts/`（各 `*.md` 模板）——命中图像上下文时追加「请结合以下技术图纸的描述进行分析」。

## 三、复现路径

见上一级 `00_提交说明与自检表.md` 第四节「复现命令」。测试题与金标准：`../03_演示与结果/工单14_6题.json`。
