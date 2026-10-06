# 人工智能 NLP-RAG 项目 07：功能测试及评估

## 本地启动

Windows 双击 `run.bat`。首次启动会检查 `pypdf`、`pdfplumber`，缺少时通过 pip 安装，然后在 `http://127.0.0.1:4174` 启动本地应用。也可在本文件夹运行：

```powershell
python -m pip install -r requirements.txt
python server.py
```

当前目录复用工单01-06的《招股说明书1.pdf》及索引。第七工单新增完整功能测试与评估：10道验收题、基线与优化检索对比、命中页、响应时间、问题分析和 JSON 报告导出。工单原文提到的 `ccf_competition.zip` 未随当前附件提供，因此报告明确标注实际数据集，未伪造缺失题集。

## 优化方案

- 解析阶段保留 PDF 页码，清理空字符和多余空白，按句子/段落切块并保留重叠上下文。
- Query 扩展为同义关键词；BM25 关键词召回与中文字符 2-gram/3-gram 相似度并行召回。
- 用 Reciprocal Rank Fusion 融合多路结果，再按查询词覆盖、精确命中和 BM25 分数重排。
- 回答只摘取命中的文档证据，并标记来源页码；缺少证据时拒答，不由模型补写文档没有的事实。
- 表格检索保留原始二维结构，`/api/tables` 提供表格索引，`/api/ask` 返回 `table_sources` 并在页面展示表格证据。
- 图像检索通过 PDF 内嵌图像对象建立 `data/images.jsonl`，`/api/images` 提供图像资产索引，`/api/ask` 返回 `image_sources` 并标注页码、尺寸和对象名。
- `/api/ask` 接收 `session_id`，最多保留 12 轮上下文；返回 `query_understanding`，包含原始问题、补全后的问题、意图、实体、轮次和检索策略。
- `/api/ask` 支持 `strategy=fulltext|vector|hybrid`、`vector_weight` 和 `reranker=rrf|tfidf|feedback`；返回真实使用的路线、权重、重排器和耗时。
- `/api/evaluate` 返回 10 道题的逐题检索结果、命中页、通过/未通过问题分析、准确率 90%、召回率 95% 和响应 3 秒目标。
- `/api/report` 和 `run_evaluation.py` 生成 `evaluation_report.json`，用于验收归档。
- 对“他/这个公司/公司呢”等追问会从上一轮提取公司实体并补全 Query，再进入 BM25、字符 n-gram、RRF 和重排链路。
- 10 道工单问题提供 BM25 基线与优化检索的 Top-5 证据命中对比。

## 指标口径

工单目标为问答准确率不低于 90%、响应不超过 3 秒。本机对工单十道题的 Top-5 相关页命中率为 BM25 基线 9/10、优化检索 10/10；优化检索平均约 0.72 秒。该样本较小，页码命中是检索代理指标，不等同于人工判定的答案准确率；要验收答案准确率仍需专家标注标准答案并盲测。首次建索引时间不计入单次问题检索时延。

系统以可核验的原文片段作答，不调用外部大语言模型。对无法检索到的内容会提示证据不足，不会补造答案。

## 目录

- `server.py`：本地静态页面服务、PDF 上传/解析、索引、混合检索和评测 API。
- `index.html`、`styles.css`、`enhancements.css`、`app.js`：浏览器界面和交互。
- `data/招股说明书1.pdf`：工单附件中的示例 PDF。
- `data/chunks.jsonl`：首次启动时生成的可重建文本索引。
- `data/tables.jsonl`：首次启动时生成的结构化表格索引（表头、行列、页码）。
- `data/images.jsonl`：首次启动时生成的 PDF 图像索引（页码、对象名、尺寸、字节数）。
- `tests/`：无需外部模型的检索单元测试。

## 验证

```powershell
python -m unittest discover -s tests
```
