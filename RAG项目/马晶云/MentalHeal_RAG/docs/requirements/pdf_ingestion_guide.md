# 心理健康 PDF 解析指南

本文说明如何解析 `data/raw/` 中的心理健康 PDF。解析文本和分块结果写入 `data/processed/`，失败记录写入 `data/failed/`。MinerU 从 PDF 结果包中提取出的图片会单独保存到 `data/processed/mineru_images/{document_id}/`，方便人工查看。

## 解析工具

本项目的 PDF 解析链路只使用 MinerU 云端 API，并在请求中开启 MinerU 内部 OCR：

| 工具 | 用途 |
|---|---|
| MinerU Cloud API | PDF 解析、OCR、版面识别、表格识别、Markdown 和内容列表输出 |
| `httpx` | 调用 MinerU API、上传 PDF、轮询任务和下载结果 |
| `python-dotenv` | 读取本地环境变量 |

代码不会使用 PyMuPDF、pdfplumber、PaddleOCR 或 PaddleOCR-VL 参与 PDF 解析。MinerU 云端解析会把 PDF 内容发送到 MinerU 服务，请只处理你有权上传且允许外部处理的文件。

## MinerU 配置

MinerU 走云端 API，因此无需执行 `pip install mineru`。在 `.env` 中检查：

```dotenv
MINERU_API_KEY=你的有效MinerU密钥
MINERU_BASE_URL=https://mineru.net/api
MINERU_API_VERSION=v4
MINERU_MODEL_VERSION=pipeline
MINERU_TIMEOUT_SECONDS=1800
```

不要把真实密钥写入 `.env.example` 或提交到 Git。MinerU 云端解析会把 PDF 内容发送到 MinerU 服务，请只处理你有权上传且允许外部处理的文件。

## 运行前依赖

只需要安装项目依赖中的 `httpx`、`python-dotenv` 等后端包；不需要本地安装 MinerU，MinerU 通过云端 API 调用。

```bash
python -m pip install -r requirements.txt
```

## 运行解析

安装依赖、确认 `.env` 后，在项目根目录执行：

```bash
PYTHONPATH=backend python -m app.ingestion.cli
```

只处理一份 PDF：

```bash
PYTHONPATH=backend python -m app.ingestion.cli --file "data/raw/你的文件.pdf"
```

每份 PDF 都直接提交 MinerU，并在请求中开启 MinerU 内部 OCR；结果中的文本、版面、表格和图片以 MinerU 输出为准。结果会清洗前写入 `data/processed/`，MinerU 提取的图片单独解压到 `data/processed/mineru_images/{document_id}/`，不会写入 JSON。超过 MinerU 官方页数限制的文件会记录失败，不会改用其他 PDF 工具。


解析完成后，保留 `data/processed/` 中的原始 MinerU JSON；清洗结果单独写入 `data/cleaned/`，不会覆盖原始结果：

```bash
PYTHONPATH=backend python -m app.ingestion.cli --clean
```

## 分块 MinerU 清洗结果

清洗完成后，将文本按段落和长度切成 RAG 分块，结果写入 `data/chunks/`，不会覆盖 `data/cleaned/`：

```bash
PYTHONPATH=backend python -m app.ingestion.cli --chunk
```

默认每块最多 1200 个字符，块之间重叠 160 个字符，过短片段会尽量与相邻正文合并。每个分块保留 `document_id`、标题、来源路径和页码，后续 BGE-m3 向量化和 Milvus 入库直接使用 `data/chunks/`。
