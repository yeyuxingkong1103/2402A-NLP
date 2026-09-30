# MentalHeal RAG 项目 20 分钟代码讲解稿

> **本版讲解顺序已经调整为：先讲离线知识库，再讲在线问答。**
>
> 你可以把整个系统想成两条流水线：
>
> 1. **离线流水线：** 先把 PDF 处理成可以搜索的知识库。这个过程不是用户每次提问时才做，而是提前做好的。
> 2. **在线流水线：** 用户在前端问问题时，从已经做好的知识库里找资料，再让 DeepSeek 参考资料生成回答。
>
> 这一版会按照“文件位置 → 关键代码行 → 用大白话解释 → 讲师可能怎么问”的方式讲。你不需要背完整代码，但一定要记住每一步的输入、输出和保存位置。

---

## 一、先记住项目的一句话介绍

### 现场可以直接这样说

> 我的项目是一个基于 RAG 的心理健康陪伴问答系统。前端使用 Vue 3 和 Vite，后端使用 FastAPI。用户提问以后，后端先做身份校验和危机表达检测；普通问题会同时经过 Milvus 向量检索和 BM25 关键词检索，再用 BGE-reranker 做精排，把最相关的心理健康资料连同聊天历史一起交给 DeepSeek 生成回答。回答通过 SSE 流式返回前端，同时把会话和引用来源保存下来。离线部分则负责把 PDF 通过 MinerU 解析、清洗、分块、向量化，最后写入 Milvus 和 MySQL，供在线问答使用。

### 这句话里面的关键词

- **RAG**：Retrieval-Augmented Generation，检索增强生成。不是让大模型只靠自己的记忆回答，而是先从自己的知识库里找相关资料，再让模型参考这些资料回答。
- **Vue 3 + Vite**：负责浏览器里的页面和交互。
- **FastAPI**：负责提供后端 HTTP 接口。
- **Milvus**：保存向量并做语义相似度检索。
- **BM25**：根据关键词匹配程度检索，弥补向量检索对专有词、关键词不敏感的问题。
- **Rerank**：对初步找出的候选资料再次精排，选出真正更相关的内容。
- **SSE**：服务器向浏览器单向持续推送事件，适合把大模型生成的内容逐字或逐片段显示出来。

---

## 二、20 分钟时间安排

| 时间 | 讲解内容 | 目标 |
|---|---|---|
| 0:00–1:30 | 项目定位和两条流水线 | 先让讲师知道离线和在线的关系 |
| 1:30–9:00 | 离线 PDF 知识库处理 | 重点讲清文件入口、保存位置、格式和入库 |
| 9:00–10:00 | 前端页面、登录和接口入口 | 把离线知识库接到在线请求上 |
| 10:00–16:30 | 在线提问、RAG 检索和 DeepSeek | 讲清用户问题如何找到资料并生成回答 |
| 16:30–18:30 | SSE 流式返回和数据保存 | 解释回答如何逐段显示并持久化 |
| 18:30–20:00 | 安全、性能、测试和追问 | 说明边界、优化和项目完成度 |

如果讲师中途提问，可以直接跳到后面“高概率追问”部分，不要慌。所有追问最终都能回到“输入、处理、输出、存储”这四步。

---

# 第一部分：开场和系统全貌（0:00–1:30）

## 1. 先讲系统解决什么问题

### 现场讲法

> 我的系统不是用户一提问才临时读取 PDF，而是先在离线阶段把心理健康 PDF 处理成可检索的知识库。然后用户在前端提问时，在线后端只需要从这个知识库里找相关片段，再交给 DeepSeek 生成回答。为了让这个流程更容易理解，我先从 PDF 进入系统开始讲，再讲用户提问时前后端如何连接。

> 整个项目可以看成两条流水线：离线流水线负责“生产知识”，在线流水线负责“使用知识”。离线流水线的输出，就是在线 RAG 检索的输入。

## 2. 两条流水线的整体关系

指着架构图或白板，按照下面顺序说：

```text
浏览器
  ↓
Vue 3 / App.vue
  ↓ POST /api/v1/chat/stream
FastAPI / api/chat.py
  ↓
get_current_user 身份校验
  ↓
ChatService.stream_answer
  ↓
危机表达检测
  ├─ 命中：直接返回固定安全提醒
  └─ 未命中：Redis 历史和缓存 → RAG 检索 → DeepSeek
  ↓
SSE 事件流
  ↓
App.vue 解析事件并更新消息气泡
  ↓
MySQL 保存会话和消息，Redis 保存短期历史
```

### 现场强调

> 这个系统可以分成两个时间维度。在线部分是用户现在提问时发生的事情；离线部分是提前处理 PDF 知识库的事情。在线问答不是临时解析 PDF，而是直接检索已经处理好的知识片段，所以响应速度更稳定。

---


---

# 第二部分：先讲离线知识库——PDF 是怎样变成可检索数据的（1:30–9:00）

> 这一部分建议你不要一开始就讲前端。先告诉讲师：在线问答能够检索资料，是因为系统提前把 PDF 处理好了。
>
> 现场可以先指着架构图说：**“我先从离线部分开始，因为在线提问时使用的知识库，都是这一条离线流水线提前生产出来的。”**

## 2.1 先把离线总流程背下来

```text
原始 PDF
  ↓
保存到 data/raw/
  ↓
管理员上传接口或命令行入口
  ↓
MinerU Cloud 解析 / OCR
  ↓
保存解析结果到 data/processed/{document_id}.json
  ↓
清洗文本
  ↓
保存清洗结果到 data/cleaned/{document_id}.json
  ↓
分块
  ↓
保存 chunk 结果到 data/chunks/{document_id}.json
  ↓
BGE-m3 生成向量
  ↓
向量写入 Milvus 的 knowledge_chunks 集合
  ↓
chunk 元数据写入 MySQL 的 knowledge_chunks 表
  ↓
在线问题可以检索这些数据
```

### 这一段现场直接这样讲

> 离线部分的输入是 PDF，输出不是一个新的 PDF，而是两类数据：第一类是可供 Milvus 搜索的向量；第二类是保存在 JSON 和 MySQL 里的文本、页码、文档 ID 等元数据。向量负责“找相似内容”，元数据负责“知道这段内容来自哪份文档、哪一页，并最终展示给用户”。

---

## 2.2 第一个问题：原始 PDF 在哪里？

### 代码位置一：默认目录配置

文件：`backend/app/ingestion/config.py:34-62`

重点代码：

```python
raw_dir=Path(os.getenv("RAW_DATA_DIR", "./data/raw")),
processed_dir=Path(os.getenv("PROCESSED_DATA_DIR", "./data/processed")),
cleaned_dir=Path(os.getenv("CLEANED_DATA_DIR", "./data/cleaned")),
chunks_dir=Path(os.getenv("CHUNKS_DATA_DIR", "./data/chunks")),
vectorized_dir=Path(os.getenv("VECTORIZED_DATA_DIR", "./data/vectorized")),
failed_dir=Path(os.getenv("FAILED_DATA_DIR", "./data/failed")),
```

### 大白话解释

- `raw_dir`：原始 PDF 放的位置，默认是 `data/raw/`。
- `processed_dir`：MinerU 解析后的结果，默认是 `data/processed/`。
- `cleaned_dir`：清洗后的结果，默认是 `data/cleaned/`。
- `chunks_dir`：最终分块后的 JSON，默认是 `data/chunks/`。
- `vectorized_dir`：向量化过程的清单文件，默认是 `data/vectorized/`。
- `failed_dir`：处理失败时保存错误 JSON，默认是 `data/failed/`。

`os.getenv("变量名", "默认值")` 的意思是：先看环境变量有没有自定义路径，没有就用代码里给的默认路径。

### 老师可能问：为什么目录不直接写死？

> 因为开发环境、服务器环境和测试环境的目录可能不同。用环境变量可以换路径，不需要改 Python 代码。代码里的默认值保证本地直接运行时也有一个合理位置。

### 代码位置二：项目中真实的原始资料

当前项目的原始 PDF 位于：

```text
data/raw/
├── NHC_国家卫生健康委_心理健康相关PDF_202105.pdf
├── UNICEF_2021-2025_工作重点_青少年心理健康.pdf
├── WHO_心理急救_现场工作者指南.pdf
├── WHO_预防自杀_全球要务.pdf
└── 其他心理健康相关 PDF
```

### 要强调的点

> `data/raw/` 里的文件就是知识库的原材料。这个目录里的 PDF 是原始文件，正常情况下不直接修改。后面处理出来的 JSON 会放到其他目录，原始 PDF 和处理结果分开保存，方便重新处理和追溯。

---

## 2.3 PDF 如果从管理接口上传，会经过什么代码？

虽然当前页面已经删掉了管理中心入口，但是后端的文档管理接口仍然保留。所以讲师如果问“新 PDF 是怎么进入系统的”，要按下面这条代码回答。

### 入口文件

文件：`backend/app/api/documents.py:64-96`

入口函数：`upload_document()`。

核心代码：

```python
if not file.filename or not file.filename.lower().endswith(".pdf"):
    raise HTTPException(status_code=400, detail="只支持 PDF 文件")
settings = get_settings()
target_dir = settings.raw_data_dir
target_dir.mkdir(parents=True, exist_ok=True)
safe_name = Path(file.filename).name
target = target_dir / safe_name
```

### 逐行解释

1. `file.filename`：上传文件的原始文件名。
2. `endswith(".pdf")`：只允许扩展名是 PDF。
3. `settings.raw_data_dir`：从配置中取出原始文件目录。
4. `mkdir(..., exist_ok=True)`：目录不存在就创建，已经存在也不会报错。
5. `Path(file.filename).name`：只保留文件名，不保留用户上传时可能带来的路径，避免把文件写到任意目录。

接着是大小和文件内容检查，位置在 `documents.py:79-95`：

```python
content = file.file.read(max_bytes + 1)
if len(content) > max_bytes:
    raise HTTPException(status_code=413, detail="文件超过上传大小限制")
if not content.startswith(b"%PDF-"):
    raise HTTPException(status_code=400, detail="文件内容不是有效的 PDF")
```

### 为什么扩展名检查还不够

> 因为用户可以把任何文件改名成 `.pdf`。所以代码又读取文件内容，检查 PDF 文件通常应该有的 `%PDF-` 文件头。它不是完整的恶意文件检测，但比只检查扩展名更可靠。

### 同名文件怎么处理

位置：`documents.py:84-95`。

- 同名且内容完全一样：直接复用已有文档记录。
- 同名但内容不同：用文件内容的 SHA-256 前 12 位生成后缀，避免静默覆盖原文件。
- 最后使用 `target.write_bytes(content)` 把二进制 PDF 写入 `data/raw/`。

```python
if not target.exists():
    target.write_bytes(content)
document = register_document(target, settings)
```

### 文档注册到 MySQL

`register_document()` 在 `backend/app/services/document_ingestion.py:33-57`。

它先通过 `document_id(path)` 生成文档 ID：

```python
document_id_value = document_id(path)
```

`document_id()` 在 `backend/app/ingestion/text_processing.py:6-8`：

```python
def document_id(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:24]
```

### 这段代码是什么意思

> 代码读取 PDF 的二进制内容，计算 SHA-256 哈希，再取前 24 位作为文档 ID。也就是说，文档 ID 和文件内容相关，同一个内容通常会得到同一个 ID；文件内容发生变化，ID 也会变化。这比单纯使用文件名更适合识别文件版本。

`register_document()` 会把这些路径保存到 MySQL 的 `documents` 表：

- 原始文件名 `source_file`。
- 原始路径 `raw_path`。
- 解析结果路径 `processed_path`。
- 清洗结果路径 `markdown_path`。
- 创建和更新时间。

对应代码：`document_ingestion.py:41-50`。

### 老师可能问：为什么路径还要存到数据库？

> 数据库保存路径后，后续任务只需要根据文档 ID 找到原始文件和中间文件，不需要重新猜文件名。数据库保存的是“这份文档在哪里、处理到哪一步”的索引信息，文件系统保存实际内容。

---

## 2.4 离线任务从哪里开始？有两个入口，要分清

当前代码有两种触发方式：**命令行入口**和**管理接口入口**。它们最终会调用同一批处理模块。

### 入口 A：命令行 `cli.py`

文件：`backend/app/ingestion/cli.py:95-130`。

命令行参数：

```python
parser.add_argument("--file", help="Process one PDF instead of the complete raw directory")
parser.add_argument("--clean", action="store_true", help="Clean MinerU JSON results")
parser.add_argument("--chunk", action="store_true", help="Chunk cleaned JSON results")
parser.add_argument("--embed", action="store_true", help="Embed chunks and insert them into Milvus")
```

### 这些参数分别做什么

- 不加参数：处理 `data/raw/` 下所有还没有处理过的 PDF。
- `--file 某个.pdf`：只解析指定的一个 PDF。
- `--clean`：把 `processed/` 里的解析结果清洗到 `cleaned/`。
- `--chunk`：把 `cleaned/` 的 JSON 分块到 `chunks/`。
- `--embed`：读取 `chunks/`，生成向量并写入 Milvus，同时写 `vectorized/manifest.json`。

当前实际命令可以这样讲：

```bash
PYTHONPATH=backend python -m app.ingestion.cli --file "data/raw/你的文件.pdf"
PYTHONPATH=backend python -m app.ingestion.cli --clean
PYTHONPATH=backend python -m app.ingestion.cli --chunk
PYTHONPATH=backend python -m app.ingestion.cli --embed
```

### 重要提醒：命令行的步骤是分开的

> `cli.py` 的代码把解析、清洗、分块、向量化设计成了几个可以单独执行的阶段。这样某一步失败时，不需要把前面所有步骤全部重做，也方便调试每个中间文件。

### 入口 B：管理员文档接口

接口文件：`backend/app/api/documents.py:109-160`。

- `POST /api/v1/documents/{document_id}/parse`：启动解析、清洗、分块任务。
- `POST /api/v1/documents/{document_id}/embed`：启动向量化和入库任务。

这两个接口使用 `Thread` 在后台执行：

```python
thread = Thread(
    target=_run_document_task,
    args=(process_document, document_id, job_id),
    daemon=True,
)
thread.start()
```

### 为什么要后台线程

> PDF 解析、清洗和模型向量化都可能耗时。如果接口一直等到任务全部结束，浏览器会长时间等待。后台线程让接口先返回“任务已排队”，任务状态再通过数据库和 Redis 更新。

### 这两个入口怎么联系起来

管理接口最终调用：

```text
upload_document()
  ↓
register_document()
  ↓
parse_document() → process_document()
  ↓
embed_document() → embed_chunks()
```

命令行则直接调用：

```text
cli.main()
  ↓
PdfIngestionPipeline / DocumentCleaner / DocumentChunker / embed_chunks
```

### 讲解时如何避免说错

> 如果讲师问“项目现在是页面上传还是命令行上传”，准确回答是：后端保留了管理员上传和文档处理接口，但当前前端已经移除了管理中心入口；我也可以通过命令行入口执行同样的离线处理步骤。两条入口最终使用相同的解析、清洗、分块和向量化模块。

---

## 2.5 第一步：MinerU 解析 PDF，保存的是什么？

### 实际解析主流程

文件：`backend/app/ingestion/pipeline.py:13-43`。

重点代码：

```python
class PdfIngestionPipeline:
    def __init__(self, config: IngestionConfig) -> None:
        self.config = config
        self.mineru_client = MineruClient(
            api_key=config.mineru_api_key,
            base_url=config.mineru_base_url,
            api_version=config.mineru_api_version,
            model_version=config.mineru_model_version,
            poll_seconds=config.mineru_poll_seconds,
            timeout_seconds=config.mineru_timeout_seconds,
            is_ocr=config.mineru_is_ocr,
        )
```

### 这段初始化在做什么

> `PdfIngestionPipeline` 是 PDF 处理流水线。它先创建 `MineruClient`，把 API Key、接口地址、版本、轮询间隔、超时时间和是否 OCR 等配置传进去。后面真正解析 PDF 时就使用这个客户端。

真正处理一个文件的是 `process_file()`，位于 `pipeline.py:26-43`：

```python
def process_file(self, path: Path) -> ParsedDocument:
    logger.info("Start MinerU-only PDF ingestion: %s", path.name)
    doc_id = document_id(path)
    image_dir = self.config.processed_dir / "mineru_images" / doc_id
    mineru_by_page = self.mineru_client.parse(path, image_dir)
    pages = self._build_pages(mineru_by_page)
    chunks = self._build_chunks(path, pages)
    result = ParsedDocument(
        document_id=doc_id,
        source_path=str(path),
        title=path.stem,
        page_count=len(pages),
        pages=pages,
        chunks=chunks,
        engines=["mineru"],
    )
    self._write_result(result)
    return result
```

### 按顺序解释每一行

1. `logger.info(...)`：写一条日志，说明现在开始处理哪一个 PDF。
2. `document_id(path)`：根据 PDF 内容生成文档 ID。
3. `image_dir`：给 MinerU 解析出来的图片准备保存目录，位置在 `data/processed/mineru_images/{document_id}/`。
4. `self.mineru_client.parse(path, image_dir)`：把 PDF 交给 MinerU，返回“页码到文本”的字典。
5. `_build_pages(...)`：把字典整理成 `ExtractedPage` 对象列表，每一页有页码和文本。
6. `_build_chunks(...)`：按页面把文本切成初步的 `TextChunk`。
7. `ParsedDocument(...)`：把文档 ID、原始路径、标题、页数、页面列表、chunk 列表和使用的引擎装在一起。
8. `_write_result(result)`：把整个解析结果写成 JSON 文件。
9. `return result`：把内存中的结构化结果返回给上层。

### 当前项目到底使用哪一种解析器？

> 当前实际主流程是 `PdfIngestionPipeline` 配合 `MineruClient`，也就是调用 MinerU Cloud API v4，并根据 `MINERU_IS_OCR` 配置决定是否让 MinerU 做 OCR。`pipeline.py:16-24` 把 MinerU 客户端初始化好，`pipeline.py:30` 真正调用 `self.mineru_client.parse(...)`。目录里的 `native_pdf.py`、`ocr_pdf.py`、`table_pdf.py` 和 `vision_ocr.py` 是保留的辅助/备用模块，但当前这条 `PdfIngestionPipeline` 主流程没有把它们串进来。讲解时应该说“当前使用 MinerU 解析和 OCR”，不要说所有 OCR 引擎都同时运行。



```python
def parse(self, path: Path, image_dir: Path | None = None) -> dict[int, str]:
    if not self.api_key:
        raise ValueError("MINERU_API_KEY is required")
    return self._parse_part(path, image_dir, 0)
```

`_parse_part()` 的顺序是：

1. `_request_upload_url()` 请求上传地址。
2. `_upload_file()` 把 PDF 上传到 MinerU 返回的地址。
3. `_poll_result()` 根据 batch ID 轮询解析状态。
4. `_download_markdown()` 下载 Markdown 或 ZIP 结果。
5. 返回每页文本。

对应代码位置：`mineru_client.py:37-54`。

请求上传地址：`mineru_client.py:56-64`。

```python
url = f"{self.base_url}/{self.api_version}/file-urls/batch"
payload = {
    "files": [{"name": path.name, "is_ocr": self.is_ocr}],
    "model_version": self.model_version,
}
response = client.post(url, headers=self._headers(), json=payload)
response.raise_for_status()
return response.json()
```

### 口语解释

> 代码不是直接把整个 PDF 内容放到 Python 里解析，而是先向 MinerU 申请上传地址，再上传文件，拿到一个批次 ID。MinerU 需要一定时间处理，所以后面通过批次 ID不断查询状态，完成后再下载解析结果。

轮询代码：`mineru_client.py:70-95`。

```python
while time.monotonic() < deadline:
    response = client.get(url, headers=self._headers())
    result = self._find_result(payload, filename)
    state = str(result.get("state", "")).lower()
    if state in {"done", "success", "succeeded"}:
        return result
    if state in {"failed", "error"}:
        raise RuntimeError(f"MinerU task failed: {result}")
    time.sleep(self.poll_seconds)
```

### 这里的 `while` 在干什么

> 在没有超过最大等待时间之前，反复查询任务状态。成功就返回，失败就抛出异常，仍在处理中就睡眠几秒再查一次。`time.monotonic()` 用来计算经过的时间，比直接比较系统时钟更适合做超时判断。

### MinerU 返回什么格式

`_download_markdown()` 位于 `mineru_client.py:113-134`，支持两类返回：

- 如果返回 `full_zip_url`，就下载 ZIP，从里面读取 `content_list.json` 或 Markdown 文件。
- 如果返回内嵌的 `markdown` 或 `content`，就直接按页面分隔符拆分。

如果 ZIP 中存在 `content_list.json`，`_content_list_pages()` 会把其中每个对象的 `page_idx` 和 `text` 重新组成：

```text
{页码: 该页文本}
```

代码位置：`mineru_client.py:193-220`。

### 解析结果保存到哪里？是什么格式？

真正写文件的是 `pipeline.py:91-93`：

```python
target = self.config.processed_dir / f"{result.document_id}.json"
target.write_text(
    json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
    encoding="utf-8",
)
```

所以结论是：

- 保存目录：`data/processed/`。
- 文件名：`{document_id}.json`。
- 格式：UTF-8 编码、格式化缩进的 JSON。
- 内容：不是单纯的一段文本，而是一个结构化文档对象。

结构大概是：

```json
{
  "document_id": "文件内容生成的24位ID",
  "source_path": "data/raw/原始文件.pdf",
  "title": "原始文件名",
  "page_count": 10,
  "pages": [
    {
      "page_number": 1,
      "native_text": "",
      "ocr_text": "",
      "vision_ocr_text": "",
      "tables": [],
      "mineru_text": "这一页解析出来的文本"
    }
  ],
  "chunks": [
    {
      "chunk_id": "文档ID-1-1",
      "page_start": 1,
      "page_end": 1,
      "text": "解析阶段按页生成的初步文本片段",
      "source": "data/raw/原始文件.pdf",
      "metadata": {
        "title": "原始文件",
        "page": 1
      }
    }
  ],
  "engines": ["mineru"],
  "warnings": []
}
```

结构定义在：`backend/app/ingestion/schemas.py:5-34`。

### 解析阶段会不会生成 chunk？

> 会。`pipeline.py:72-89` 的 `_build_chunks()` 会在解析后按页调用 `chunk_text()`，把初步 chunk 放进 `ParsedDocument`，然后一起写入 `processed` JSON。之后正式的服务流程还会经过清洗，再由 `DocumentChunker.chunk_file()` 重新生成并写入 `data/chunks/`；后续向量化和 BM25 主要读取 `data/chunks/` 这份正式结果。这样讲最准确：**解析阶段有初步 chunk，清洗和正式分块阶段会生成最终用于检索的 chunk。**



> TXT 只能保存连续文本，页码、文档 ID、来源路径、解析引擎和警告等信息不方便一起保存。JSON 可以同时保存结构化元数据和正文，后续清洗、分块、引用页码都更方便。

### 解析失败保存哪里

`pipeline.py:95-98`：

```python
target = self.config.failed_dir / f"{path.stem}.error.json"
payload = {
    "source_path": str(path),
    "error_type": type(error).__name__,
    "error": str(error),
}
target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
```

失败文件保存到 `data/failed/`，也是 JSON，里面记录原始路径、异常类型和错误信息。这样可以知道哪本 PDF 失败了、为什么失败，不会悄悄丢掉任务。

---

## 2.6 第二步：清洗文本，清洗掉什么？

### 清洗入口

文件：`backend/app/ingestion/cleaning.py:16-50`。

```python
class DocumentCleaner:
    def clean_file(self, source: Path, target: Path) -> dict[str, Any]:
        payload = json.loads(source.read_text(encoding="utf-8"))
        pages = payload.get("pages", [])
        raw_texts = [str(page.get("mineru_text", "")) for page in pages]
        basic_texts, basic_stats = self._clean_pages(raw_texts)
        noise_lines = self._find_edge_noise(basic_texts)
        cleaned_texts, edge_stats = self._remove_edge_noise(basic_texts, noise_lines)
```

### 大白话解释

1. 先读取 `processed` 的 JSON。
2. 只取每一页的 `mineru_text`。
3. 做第一轮基础清洗。
4. 找出页眉页脚等重复边缘内容。
5. 再把重复噪声删除。

### 第一轮清洗具体做什么

代码：`cleaning.py:60-87`。

它主要处理：

- 删除 Markdown 图片占位符和 HTML 图片标签。
- Unicode 规范化。
- 删除空字符。
- 合并多余空格。
- 删除单独的页码行。
- 删除同一页连续重复的行。
- 最后调用 `normalize_text()` 统一换行和空白。

比如 PDF 解析后可能出现：

```text
第 1 页
心理健康指南
心理健康指南
![image](xxx.png)

真正的正文
```

清洗后会尽量变成：

```text
心理健康指南

真正的正文
```

### `normalize_text()` 做什么

文件：`backend/app/ingestion/text_processing.py:11-16`。

```python
def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("﻿", " ")
    text = re.sub(r"[\t  ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
```

> 这不是在改变原意，而是在清理格式噪声：空字符、连续空格、行首行尾空格和过多空行。

### 重复页眉页脚怎么处理

代码：`cleaning.py:89-108` 找出出现在多页边缘的相同文本；`cleaning.py:110-132` 删除重复出现的边缘行。

口语化理解：

> 如果一本 PDF 每页顶部都重复出现同一个报告标题，或者底部都重复出现网址和页脚，直接拿去检索会让很多 chunk 都被同样的文字污染。代码只关注每页前 3 行和后 3 行，统计重复频率，再删除重复的边缘内容，同时保留第一次出现的那一份。

### 清洗后的 JSON 保存在哪里

仍然在 `clean_file()` 的后半部分：`cleaning.py:35-49`。

```python
result = dict(payload)
result["pages"] = cleaned_pages
result["chunks"] = chunks
result["cleaning"] = {
    "version": "1.0",
    "source": "mineru",
    "removed_image_placeholders": basic_stats["image_placeholders"],
    "removed_standalone_page_numbers": basic_stats["page_numbers"],
    "removed_repeated_edge_lines": edge_stats,
    "removed_duplicate_chunks": duplicate_count,
    "nonempty_pages": sum(bool(text) for text in cleaned_texts),
}
target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
```

结论：

- 输入：`data/processed/{document_id}.json`。
- 输出：`data/cleaned/{document_id}.json`。
- 格式：仍是 JSON，不是 Markdown 文件。
- 里面会多一个 `cleaning` 字段，记录清洗统计。
- 页面中的 `mineru_text` 被替换成清洗后的文本。

### 一个容易讲错的地方

> 数据库模型里有一个字段叫 `markdown_path`，但当前代码实际把清洗后的 JSON 写到了这个路径，并不是单独生成一个 `.md` Markdown 文件。讲解时准确说“清洗结果 JSON”，不要说成“保存成 Markdown 文件”。

---

## 2.7 第三步：分块，为什么一篇文章要切成很多块？

### 当前实际主流程里的分块调用

文件：`backend/app/services/document_ingestion.py:133-139`。

```python
parsed = PdfIngestionPipeline(config).process_file(source)
DocumentCleaner(config.chunk_size, config.chunk_overlap).clean_file(
    Path(document.processed_path), Path(document.markdown_path)
)
chunk_result = DocumentChunker(config.chunk_size, config.chunk_overlap).chunk_file(
    Path(document.markdown_path), config.chunks_dir / f"{document_id_value}.json"
)
```

这一段非常重要，可以现场按顺序指给讲师：

1. 第 133 行：先解析 PDF。
2. 第 134–136 行：读取 processed JSON，清洗后写到 cleaned 路径。
3. 第 137–139 行：读取 cleaned JSON，分块后写到 chunks 路径。

### 分块类和参数

文件：`backend/app/ingestion/chunking.py:9-15`。

```python
class DocumentChunker:
    def __init__(self, max_chars: int = 1200, overlap: int = 160, min_chars: int = 80):
        self.max_chars = max_chars
        self.overlap = overlap
        self.min_chars = min_chars
```

当前配置默认值来自：`backend/app/ingestion/config.py:54-55`。

```python
chunk_size=int(os.getenv("INGESTION_CHUNK_SIZE", "1200")),
chunk_overlap=int(os.getenv("INGESTION_CHUNK_OVERLAP", "160")),
```

可以这样讲：

- 一个 chunk 最多大约 1200 个字符。
- 相邻 chunk 之间保留大约 160 个字符的重叠。
- 太短的 chunk 默认少于 80 个字符，会尝试和相邻块合并。

### 为什么不把一整本 PDF 作为一个文本

> 因为用户问的是一个具体问题，不需要把整本书都交给模型。切成小块以后，检索可以更准确地定位到相关段落，传给模型的上下文也更短，速度和成本更好控制。

### `chunk_text()` 的实际逻辑

文件：`backend/app/ingestion/text_processing.py:33-69`。

它的步骤是：

1. 检查 `max_chars` 和 `overlap` 参数是否合法。
2. 先清洗文本。
3. 按空行把文本分成段落。
4. 如果多个段落合起来没有超过 1200 字，就放在同一个 chunk。
5. 如果超过了，就把当前 chunk 保存，并从上一个 chunk 尾部取 160 个字符作为下一个 chunk 的开头。
6. 如果一个段落本身超过最大长度，就按长度切成多个片段，同样保留重叠。

关键代码：

```python
paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
```

这行的意思是按空行分段，而不是简单按每一行切碎。

```python
if len(combined) <= max_chars:
    current = combined
else:
    chunks.append(current)
    tail = current[-overlap:].strip() if overlap else ""
    current = f"{tail}\n\n{paragraph}" if tail else paragraph
```

这段体现了“超长就换块，并把上一块尾巴带一点过来”。

### `DocumentChunker` 为什么还要再做一次逻辑

`DocumentCleaner` 在清洗时也会构造一份初步 chunks，但当前服务主流程接下来又明确调用了 `DocumentChunker.chunk_file()`，以 `data/chunks/` 里的结果作为后续向量化和 BM25 的正式输入。

`DocumentChunker._chunk_pages()` 位置：`chunking.py:48-80`。

它还会：

- 保留页码。
- 删除空文本页。
- 给每个块生成 `chunk_id`。
- 对重复文本做指纹去重。
- 加上 `cleaned=True` 和 `chunked=True` 元数据。

### chunk ID 是怎么来的

```python
"chunk_id": f"{document_id}-{page_number}-{index}"
```

例如：

```text
38a653429c409ee0fa12e27a-12-2
```

可以理解为：

```text
文档 ID - 第 12 页 - 本页第 2 个块
```

### chunks JSON 保存在哪里？保存什么格式？

`chunk_file()` 写文件的代码在：`chunking.py:17-37`。

```python
result = {
    "document_id": document.get("document_id", source.stem),
    "source_path": document.get("source_path", ""),
    "title": document.get("title", source.stem),
    "page_count": document.get("page_count", len(document.get("pages", []))),
    "engines": document.get("engines", ["mineru"]),
    "chunking": {
        "version": "1.0",
        "source": str(source),
        "max_chars": self.max_chars,
        "overlap": self.overlap,
        "min_chars": self.min_chars,
        "strategy": "paragraph-aware page-preserving chunks",
    },
    "chunks": chunks,
}
```

```python
target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
```

结论：

- 输入：`data/cleaned/{document_id}.json`。
- 输出：`data/chunks/{document_id}.json`。
- 格式：UTF-8 JSON。
- 顶层记录文档信息和分块策略。
- `chunks` 数组里的每一项包含 `chunk_id`、页码、正文、源文件和元数据。

一个 chunk 大概长这样：

```json
{
  "chunk_id": "文档ID-12-2",
  "page_start": 12,
  "page_end": 12,
  "text": "这一段心理健康资料正文……",
  "source": "data/raw/某份指南.pdf",
  "metadata": {
    "title": "某份指南",
    "page": 12,
    "cleaned": true,
    "chunked": true
  }
}
```

---

## 2.8 第四步：BGE-m3 把 chunk 变成向量

### 向量化入口

服务流程中的调用：`backend/app/services/document_ingestion.py:170-182`。

```python
chunk_file = config.chunks_dir / f"{document_id_value}.json"
if not chunk_file.exists():
    raise FileNotFoundError("请先完成文档解析和分块")
_update_job(job_id, status="running", started_at=datetime.utcnow())
embed_chunks(config, document_id=document_id_value)
payload = json.loads(chunk_file.read_text(encoding="utf-8"))
```

这说明向量化有一个前置条件：`data/chunks/{document_id}.json` 必须已经存在。也就是说，不能跳过解析和分块直接向量化。

### `embed_chunks()` 的关键代码

文件：`backend/app/ingestion/cli.py:24-40`。

```python
chunk_files = sorted(config.chunks_dir.glob("*.json"))
embedder = BgeEmbedder(config.embedding_model_name, config.embedding_batch_size)
if embedder.dimension != config.embedding_dim:
    raise ValueError("Embedding dimension mismatch")
store = MilvusKnowledgeStore(
    host=config.milvus_host,
    port=config.milvus_port,
    collection_name=config.milvus_collection_knowledge,
    dimension=embedder.dimension,
)
```

### 逐句解释

- 找出 `data/chunks/` 下的 JSON 文件。
- 加载 BGE-m3 模型。
- 检查模型实际输出维度和配置是否一致，当前默认是 1024 维。
- 创建 Milvus 存储客户端。

BGE 封装在：`backend/app/ingestion/embedding.py:8-27`。

```python
class BgeEmbedder:
    def __init__(self, model_path: str | Path, batch_size: int = 16):
        self.model = SentenceTransformer(str(model_path))
        self.batch_size = batch_size
```

真正编码：

```python
vectors = self.model.encode(
    list(texts),
    batch_size=self.batch_size,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
)
```

### 大白话解释“向量化”

> 向量化就是把一段文字变成一串数字。比如“最近晚上睡不着”和“失眠问题的改善方法”字面上不完全一样，但模型可能把它们映射到比较接近的位置。之后用户提问时，也把问题变成同样格式的数字，再去 Milvus 找距离近的 chunk。

### 为什么要归一化

> 归一化后不同文本的向量长度统一，后面使用余弦相似度时，比较重点是方向，也就是语义方向，而不是单纯看数字大小。

### 为什么批量编码

> 一次把多个 chunk 放在一起编码，比一个一个调用模型更高效。`batch_size=16` 表示通常每批处理 16 段文本，实际可以根据机器内存调整。

---

## 2.9 第五步：向量写入 Milvus，具体写什么？

### 生成待写入的 row

文件：`backend/app/ingestion/cli.py:44-67`。

```python
for chunk_file in chunk_files:
    document = json.loads(chunk_file.read_text(encoding="utf-8"))
    chunks = document.get("chunks", [])
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    existing = store.has_chunk_ids(chunk_ids)
    pending = [chunk for chunk in chunks if str(chunk["chunk_id"]) not in existing]
    texts = [str(chunk["text"]) for chunk in pending]
    vectors = embedder.encode(texts)
```

### 这一段是什么意思

1. 读取一个 chunks JSON。
2. 取出所有 chunk。
3. 先检查 Milvus 中哪些 `chunk_id` 已经存在。
4. 只对没写过的 chunk 继续处理，避免重复插入。
5. 把待处理文本交给 BGE-m3 生成向量。

接下来构造每一条数据：

```python
rows.append(
    {
        "chunk_id": str(chunk["chunk_id"]),
        "document_id": str(document.get("document_id", "")),
        "source_file": Path(str(document.get("source_path", ""))).name,
        "page_start": int(chunk.get("page_start", 0)),
        "page_end": int(chunk.get("page_end", 0)),
        "chunk_index": int(chunk.get("metadata", {}).get("page", index)),
        "chunk_type": "text",
        "text": str(chunk["text"]),
        "vector": vector,
    }
)
```

Milvus 中一条记录同时保存：

- `chunk_id`：分块唯一编号。
- `document_id`：属于哪篇文档。
- `source_file`：原始 PDF 文件名。
- `page_start`、`page_end`：页码。
- `chunk_index`：块序号。
- `chunk_type`：当前是 text。
- `text`：原始 chunk 正文。
- `vector`：BGE-m3 生成的 1024 维向量。

### Milvus 的表结构在哪里定义

文件：`backend/app/ingestion/milvus_store.py:57-80`。

```python
FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256)
FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256)
FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512)
FieldSchema(name="page_start", dtype=DataType.INT64)
FieldSchema(name="page_end", dtype=DataType.INT64)
FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=12000)
FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
```

### HNSW 和 COSINE 是什么

创建索引的代码：`milvus_store.py:76-79`。

```python
collection.create_index(
    field_name="vector",
    index_params={
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    },
)
```

口语解释：

- **HNSW**：一种让向量近邻搜索更快的索引结构，可以理解成提前搭好的“近似搜索路网”。
- **COSINE**：用余弦相似度比较向量。
- `M`：每个节点连接的邻居数量等索引参数。
- `efConstruction`：建索引时搜索范围相关的参数，影响索引质量和构建成本。

### 真正插入 Milvus

`milvus_store.py:23-44`：

```python
self.collection.insert([
    [row["chunk_id"] for row in batch],
    [row["document_id"] for row in batch],
    [row["source_file"] for row in batch],
    [row["page_start"] for row in batch],
    [row["page_end"] for row in batch],
    [row["chunk_index"] for row in batch],
    [row["chunk_type"] for row in batch],
    [row["text"] for row in batch],
    [row["vector"] for row in batch],
])
self.collection.flush()
```

- 代码按批次插入，默认每批最多 64 条。
- `flush()` 让数据落到 Milvus 的持久化存储中。
- `has_chunk_ids()` 会先查已有 ID，尽量避免重复。

### 老师可能问：Milvus 里只存向量吗？

> 当前不是只存向量。为了检索后能直接返回来源，Milvus 里同时保存了 chunk 文本、文档 ID、文件名和页码。这样在线检索拿到结果后就可以直接组装引用；MySQL 则进一步保存更完整的结构化管理信息。

---

## 2.10 第六步：同一批 chunk 还要写入 MySQL

向量写入 Milvus 后，`embed_document()` 会读取同一个 chunks JSON，然后把 chunk 元数据写入 MySQL：

文件：`backend/app/services/document_ingestion.py:182-212`。

```python
payload = json.loads(chunk_file.read_text(encoding="utf-8"))
db.query(KnowledgeChunk).filter(
    KnowledgeChunk.document_id == document_id_value
).delete()
for index, chunk in enumerate(payload.get("chunks", []), start=1):
    text = str(chunk.get("text", ""))
    metadata = chunk.get("metadata", {})
    db.add(KnowledgeChunk(
        chunk_id=str(chunk.get("chunk_id", "")),
        document_id=document_id_value,
        source_file=Path(str(payload.get("source_path", ""))).name,
        page_start=int(chunk.get("page_start", 0)),
        page_end=int(chunk.get("page_end", 0)),
        chunk_index=index,
        chunk_type="text",
        char_count=len(text),
        token_estimate=max(1, len(text) // 2),
        has_table=False,
        quality_score=1.0,
        milvus_collection=config.milvus_collection_knowledge,
        knowledge_version="current",
        text_preview=text[:512],
        metadata_json=metadata,
        created_at=now,
    ))
db.commit()
```

### 这里写入 MySQL 的是什么

`KnowledgeChunk` 模型在：`backend/app/models/chat.py:110-129`。

它保存：

- `chunk_id`、`document_id`。
- 原始文件名和页码。
- chunk 序号和类型。
- 字符数和估算 token 数。
- 是否包含表格、质量分数。
- 对应的 Milvus 集合名。
- 版本号。
- 前 512 个字符的预览。
- JSON 格式的元数据。

### 为什么写入前先删除同文档旧记录

> 这是为了避免同一份文档重新向量化后，MySQL 里同时留着旧版本和新版本的 chunk。当前代码先删除这个 document_id 的旧记录，再按最新 JSON 重新插入，保持 MySQL 元数据和当前文件一致。

### MySQL 和 Milvus 的分工

可以直接这样说：

> Milvus 负责“按语义找内容”，MySQL 负责“管理内容是什么、属于哪篇文档、在哪一页、什么时候入库”。Milvus 是搜索引擎，MySQL 是业务数据库，两者保存的信息有重叠，但用途不同。

---

## 2.11 第七步：向量化清单 `manifest.json` 保存在哪里？

`embed_chunks()` 的最后几行，位置：`backend/app/ingestion/cli.py:82-92`。

```python
summary = {
    "collection": config.milvus_collection_knowledge,
    "model": config.embedding_model_name,
    "dimension": embedder.dimension,
    "inserted_total": inserted_total,
    "files": manifest,
}
(config.vectorized_dir / "manifest.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
```

所以：

- 文件位置：`data/vectorized/manifest.json`。
- 文件格式：JSON。
- 保存内容：使用的 Milvus 集合、embedding 模型、向量维度、总插入数、每个文件插入和跳过的 chunk 数。

它不是向量本身，而是一次向量化任务的摘要清单。真正的向量在 Milvus 中。

---

## 2.12 第八步：任务状态保存在哪里？

如果通过管理接口触发任务，`create_ingest_job()` 在：

`backend/app/services/document_ingestion.py:60-81`。

它会在 MySQL 的 `ingest_jobs` 表创建一条任务记录，保存：

- 任务名。
- `queued`、`running`、`success`、`failed` 状态。
- 原始目录、处理目录、chunk 目录、向量目录。
- 文档数、页数、chunk 数。
- 错误或摘要信息。
- 开始和结束时间。

`_update_job()` 在：`document_ingestion.py:84-100` 更新任务状态，并在状态变化时把文档状态写入 Redis。

### 现场可以这样总结

> 文件系统保存处理结果，MySQL 保存文档和任务的结构化信息，Redis 保存短期任务状态，Milvus 保存向量检索需要的向量和部分来源字段。不同存储各自负责不同类型的数据。

---

## 2.13 离线部分最容易被问的“实际代码问题”

### 问题一：解析后的文件是 Markdown 还是 JSON？

标准回答：

> MinerU 返回的内容可能是 Markdown 或 ZIP 包里的 Markdown、`content_list.json`，但项目自己的 `PdfIngestionPipeline` 最终统一组装成 `ParsedDocument`，再通过 `json.dumps` 保存为 `data/processed/{document_id}.json`。后续清洗和分块也主要读取 JSON。数据库字段虽然叫 `markdown_path`，当前实际保存的仍然是清洗后的 JSON，不是独立的 `.md` 文件。

### 问题二：`processed`、`cleaned`、`chunks` 有什么区别？

标准回答：

> `processed` 是 MinerU 刚解析完、按页组织的原始结构化结果；`cleaned` 是删除图片占位符、页码、重复页眉页脚后的结果；`chunks` 是在清洗结果基础上，按长度和段落切成可检索的小片段。三者都是 JSON，但阶段和内容不同。

### 问题三：原始 PDF 会被覆盖吗？

标准回答：

> 上传代码会先判断同名文件内容是否一致。内容一致就复用，内容不一致就用内容哈希生成后缀，避免静默覆盖。原始 PDF 仍然保存在 `data/raw/`，处理结果写入其他目录。

### 问题四：为什么要用文档内容生成 ID？

标准回答：

> 内容哈希比文件名更可靠。同名但内容不同的文件会产生不同 ID，内容相同的文件可以识别为同一份内容，也方便处理结果文件和数据库记录对应起来。

### 问题五：如果解析失败怎么办？

标准回答：

> 命令行批处理会捕获异常并把错误类型、错误信息和源文件路径写入 `data/failed/{文件名}.error.json`；接口任务则会把 `ingest_jobs.status` 更新为 `failed`，并把错误摘要保存下来，方便重试和排查。

### 问题六：向量实际存在哪？

标准回答：

> 真正的向量存储在 Milvus 的 `knowledge_chunks` 集合里。`data/vectorized/manifest.json` 只是向量化任务的摘要，不是向量文件本身；`data/chunks/` 里的 JSON 保存的是文本和元数据，供向量化和 BM25 使用。

---

# 第三部分：前端启动、登录和身份（9:00–10:00）

## 1. `main.ts` 做什么

文件：`frontend/src/main.ts:1-4`

```ts
import { createApp } from 'vue'
import App from './App.vue'

createApp(App).mount('#app')
```

### 逐行讲解

- `import { createApp } from 'vue'`：从 Vue 引入创建应用的函数。
- `import App from './App.vue'`：引入根组件 `App.vue`。
- `createApp(App)`：创建一个 Vue 应用实例。
- `.mount('#app')`：把应用挂到 `index.html` 里 id 为 `app` 的元素上。

### 老师可能问：为什么使用 `mount`？

> 因为 Vue 组件本身只是描述页面和逻辑，必须挂载到 HTML 的一个真实节点上，浏览器才能显示出来。`#app` 就是前端应用的根节点。

## 2. Vite 端口在哪里配置

文件：`frontend/vite.config.ts:4-10`

```ts
export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
})
```

### 现场讲法

> Vite 负责前端开发服务器和构建。这里启用了 Vue 插件，开发服务器使用 5173 端口。因此浏览器访问 `http://localhost:5173` 时，拿到的是 Vue 页面。

## 3. 前端如何知道后端地址

在 `frontend/src/App.vue` 中，API 基地址类似下面的逻辑：

```ts
const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'
```

### 讲解重点

- `import.meta.env.VITE_API_BASE_URL`：读取 Vite 环境变量。
- 如果没有配置环境变量，就使用本地后端默认地址。
- `apiBase` 已经包含 `/api/v1`，所以后面只需要拼接 `/auth/login` 或 `/chat/stream`。

### 老师可能问：为什么不把完整地址散落在每个请求里？

> 统一定义基地址可以避免重复，也方便以后部署到测试或生产环境时，只修改一个环境变量，而不是修改所有请求代码。

## 4. 登录流程

前端登录逻辑在 `App.vue` 的 `submitAuth` 附近，主要步骤是：

```text
用户输入用户名和密码
  ↓
前端根据登录/注册状态选择接口
  ↓
POST /api/v1/auth/login 或 /api/v1/auth/register
  ↓
后端验证成功后返回 access_token 和 user
  ↓
前端保存 token
  ↓
后续请求在 Authorization 头中携带 Bearer token
```

### 后端对应位置

- 路由：`backend/app/api/auth.py:14`
- 注册：`backend/app/api/auth.py:26-63`
- 登录：`backend/app/api/auth.py:65-72`
- 当前用户：`backend/app/api/auth.py:75-77`
- Token 校验：`backend/app/security/auth.py:81-97`

### 后端登录具体做了什么

`backend/app/api/auth.py:65-72` 的登录流程是：

1. 根据用户名从 MySQL 查询用户。
2. 检查用户存在且处于启用状态。
3. 使用 `verify_password` 校验密码哈希。
4. 更新用户时间。
5. 调用 `create_access_token` 生成访问令牌。
6. 返回令牌和用户基本信息。

密码不是明文保存的。`backend/app/security/auth.py:22-31` 使用 PBKDF2-SHA256 加随机盐生成密码摘要，`verify_password` 再使用相同参数计算并比较摘要。

### 老师可能问：为什么密码不能直接存数据库？

> 因为数据库泄露时，明文密码会直接暴露。哈希加随机盐后，数据库保存的是不可直接还原的摘要。验证登录时不是解密，而是把用户输入的密码用保存的盐和迭代次数重新计算，再做安全比较。

### 老师可能问：这个 Token 是 JWT 吗？

> 当前代码实现的是项目自定义的签名访问令牌，不是标准 JWT。它由版本、编码后的 payload 和 HMAC-SHA256 签名组成，payload 中有用户 ID、账户角色和过期时间。后端收到令牌后重新计算签名并检查过期时间。讲解时不要把它说成标准 JWT。

### Token 校验的关键代码

文件：`backend/app/security/auth.py:81-97`

```python
if not authorization or not authorization.lower().startswith("bearer "):
    raise HTTPException(status_code=401, detail="请先登录")
payload = decode_access_token(authorization.split(" ", 1)[1].strip())
user = db.scalar(select(User).where(User.user_id == str(payload["sub"])))
```

通俗解释：

- 先看请求有没有 `Authorization` 请求头。
- 再看它是不是 `Bearer token` 格式。
- 解码并验证签名、过期时间。
- 最后根据 token 里的用户 ID 去数据库找真实用户。
- 用户不存在或被停用，就拒绝访问。

---

# 第四部分：前端发起问题，后端开始处理（10:00–11:00）

下面这部分建议你现场真的在页面输入一句话，例如：

> “最近压力比较大，晚上睡不着，有什么可以尝试的方法？”

然后严格按以下顺序讲。

## 第一步：前端收集输入（4:00–4:40）

页面中的输入框在 `App.vue` 模板里，通过 `v-model="input"` 和响应式状态绑定。

### `v-model` 是什么意思

> `v-model` 是 Vue 的双向绑定。用户在输入框里打字时，`input` 变量会同步变化；代码修改 `input` 时，输入框显示内容也会同步变化。因此点击发送时，`retrieve()` 可以直接读取 `input.value`。

用户点击发送，或者按 Enter 键时，会调用 `retrieve()`。Shift+Enter 通常保留为换行。

## 第二步：`retrieve()` 组织一次请求（4:40–5:30）

前端核心函数是 `frontend/src/App.vue` 中的 `retrieve()`，它主要做这些事情：

1. 去掉首尾空格。
2. 防止空消息。
3. 防止上一次请求还没结束时重复提交。
4. 先把用户消息加入页面消息列表。
5. 再加入一条空的 assistant 消息作为占位。
6. 清空输入框并设置 loading 状态。
7. 调用后端流式接口。

为什么要先创建空 assistant 消息？

> 因为后端的回答不是一次性返回，而是一段一段返回。前端先放一个空的 assistant 气泡，后续每收到一个 `delta` 片段，就把内容追加到这个气泡里，用户就能看到回答逐渐变长。

## 第三步：前端发出 HTTP 请求（5:30–6:00）

请求大致是：

```ts
fetch(`${apiBase}/chat/stream`, {
  method: 'POST',
  headers: {
    ...authHeaders(true),
    Accept: 'text/event-stream',
  },
  body: JSON.stringify({
    message: query,
    top_k: 5,
    session_id: sessionId.value,
    role_id: activeRoleId.value,
  }),
})
```

### 每一部分是什么意思

- `fetch(...)`：浏览器发 HTTP 请求。
- `method: 'POST'`：因为要提交一段问题和会话参数，不适合用 URL 查询参数表示。
- `Authorization: Bearer ...`：证明当前用户已经登录。
- `Accept: text/event-stream`：告诉服务器客户端希望接收 SSE 流。
- `message`：用户真正输入的问题。
- `top_k: 5`：希望最终返回几条主要引用，不等于内部所有候选数量。
- `session_id`：如果是新会话可以为空；后端创建后，前端保存它，后续继续使用同一个会话。
- `role_id`：选择当前 AI 角色。

## 第四步：请求进入 FastAPI（6:00–6:45）

后端路由在：`backend/app/api/chat.py:64-73`

```python
@router.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    return StreamingResponse(
        _sse_events(request, current_user),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

### 逐项解释

#### `@router.post("/chat/stream")`

这是路由装饰器，表示当前函数处理 POST 请求，路径是 `/api/v1/chat/stream`。因为这个路由器前面定义了 `/api/v1` 前缀，所以完整路径会自动拼起来。

#### `request: ChatRequest`

FastAPI 会根据 `ChatRequest` 自动解析和校验 JSON 请求体。

文件：`backend/app/schemas/chat.py:4-9`

```python
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)
    role_id: str = Field(default="mental-health", max_length=64)
    top_k: int | None = Field(default=None, ge=1, le=20)
```

这里的作用是：

- `message` 必须有内容，长度不能超过 4000。
- `session_id` 可以为空，因为第一次提问还没有会话。
- `role_id` 默认是心理健康角色。
- `top_k` 如果传入，范围限制在 1 到 20。

### 老师可能问：为什么要做 Pydantic 校验？

> 因为不能完全相信前端传来的数据。Schema 把接口输入格式集中定义下来，FastAPI 会在进入业务逻辑前拦截空消息、超长消息和非法的 top_k，减少异常数据进入后端。

#### `Depends(get_current_user)` 是什么意思

> 这是 FastAPI 的依赖注入。接口函数声明自己需要一个 `current_user`，FastAPI 就会先调用 `get_current_user`，从请求头取 token、验证 token、查数据库，验证通过后才把用户对象传给聊天函数。这样每个需要登录的接口都可以复用同一套鉴权逻辑。

#### `StreamingResponse` 是什么

> 普通 `JSONResponse` 要等完整答案生成后才返回；`StreamingResponse` 可以把一个迭代器或生成器产生的内容持续写到 HTTP 响应中。这里媒体类型是 `text/event-stream`，所以浏览器会把它当成 SSE 事件流处理。

## 第五步：SSE 外层包装（6:45–7:20）

`chat.py` 中的 `_sse_events()` 会调用业务服务，然后把每个事件包装成标准 SSE 文本：

```python
for event in events:
    yield f"event: {event['event']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
```

### SSE 一条消息长什么样

```text
event: delta
data: {"content":"可以先尝试调整作息"}

```

- `event:` 后面是事件类型。
- `data:` 后面是 JSON 数据。
- 最后的空行，也就是两个换行，表示一条 SSE 事件结束。
- `yield` 表示每生成一条就交给响应流，不需要等后面所有内容。

### SSE 到底是什么

> SSE 全称是 Server-Sent Events。它是建立在 HTTP 上的服务器到浏览器的单向推送机制。浏览器先发起一个请求，服务器保持这个响应连接打开，之后服务器可以不断推送文本事件。这里用户的问题通过 POST 发过去，DeepSeek 生成一小段，后端就用 SSE 推一小段回来。

### SSE 和 WebSocket 的区别

| 对比 | SSE | WebSocket |
|---|---|---|
| 通信方向 | 服务器到浏览器为主 | 双向通信 |
| 基础协议 | HTTP | WebSocket 协议 |
| 数据形式 | 文本事件 | 可传文本或二进制 |
| 本项目是否需要 | 需要后端持续推回答 | 不需要浏览器持续发实时指令 |

### 老师可能问：为什么项目不用 WebSocket？

> 本项目的主要需求是浏览器发送一次问题，服务器持续返回生成结果，通信模式是“客户端请求、服务端流式响应”，SSE 更简单，直接复用 HTTP，前端也容易使用 `ReadableStream` 解析。只有需要双向实时通信，例如多人协作、实时语音或双向状态同步时，WebSocket 才更合适。

---

# 第五部分：后端 `ChatService` 业务主线（11:00–13:00）

后端真正的业务编排在：`backend/app/services/chat_service.py:54-126` 的 `stream_answer()`。

可以把它理解成“总导演”：它不亲自实现所有底层细节，但负责决定每一步先后顺序。

## 1. 创建数据库会话和获取角色

```python
db = get_session_factory()()
history_repo = ChatHistoryRepository(db)
role = self._get_role(db, role_id)
```

- `get_session_factory()()`：拿到一个 MySQL 数据库 Session。
- `ChatHistoryRepository`：把会话和消息的数据库操作集中封装。
- `_get_role`：根据前端传来的 `role_id` 找启用的 AI 角色；如果找不到，会回退到 `mental-health`。

### 为什么需要角色

> 角色不是另一个模型，而是一组角色名称、描述和系统指令。不同角色可以影响回答风格和边界，但最终仍然通过同一个 DeepSeek 客户端生成。

## 2. 创建或恢复聊天会话

```python
chat_session = history_repo.get_or_create_session(
    session_id, message, current_user.user_id, role.role_id
)
active_session = chat_session.session_id
history_repo.add_message(chat_session, "user", message)
yield {"event": "session", "session_id": active_session, "role_id": role.role_id}
```

### 这里发生了什么

- 如果前端传来了已有 `session_id`，就尝试恢复这个会话。
- 如果没有，就创建一个新的会话 ID。
- 把用户消息先写进当前数据库事务。
- 通过 `session` 事件把后端正式生成的会话 ID告诉前端。
- 前端以后继续发送这个 ID，就能把多轮对话串起来。

### 老师可能问：为什么 session_id 要由后端确认？

> 因为会话属于后端数据，后端要保证它唯一、合法并且属于当前用户。前端可以传一个已有 ID，但最终使用的是后端查到或创建的 `active_session`，不能只信任浏览器自己生成的值。

## 3. 危机表达优先处理

```python
if is_crisis_message(message):
    answer = CRISIS_RESPONSE
    sources = []
    yield {"event": "progress", "value": 100, "stage": "已完成安全提醒"}
    yield {"event": "delta", "content": answer}
    intervention = True
```

安全检测在：`backend/app/security/safety.py:19-21`

```python
def is_crisis_message(message: str) -> bool:
    normalized = "".join(message.lower().split())
    return any(term in normalized for term in CRISIS_TERMS)
```

### 解释这几行

- `message.lower()`：把英文统一转成小写。
- `split()` 再 `join()`：去掉空白，减少用户故意插入空格导致检测失败的情况。
- `any(...)`：只要命中一个危机关键词，就返回 `True`。
- `CRISIS_RESPONSE` 是固定的安全提醒，不依赖大模型临场发挥。

### 为什么危机问题不能正常调用大模型

> 高风险场景对稳定性和可控性要求更高。固定响应可以避免网络异常、检索不相关或模型生成不合适内容。系统会建议用户联系当地急救、警方、急诊、专业人员和可信任的人，同时明确本系统不能替代紧急救助。

### 需要诚实说明的限制

> 当前安全检测是基于关键词和字符串规则的轻量检测，不是完整的临床风险评估，也不能保证识别所有隐晦表达。所以项目的定位是心理健康科普和陪伴工具，不是医生、心理咨询师或急救服务。

## 4. 普通问题读取历史

```python
history = self.memory.load(
    active_session,
    current_user.user_id,
    role.role_id,
)
history = self._load_confirmed_memories(db, current_user.user_id) + history
```

这里把两部分信息放进上下文：

- Redis 里的短期聊天历史。
- MySQL 里用户主动确认过的长期记忆。

`_load_confirmed_memories` 会筛选 `is_active=True` 且 `is_confirmed=True` 的记忆，并且限制最多 20 条。重要的是，它把这些内容标成“回答风格参考”，不是系统指令，避免普通用户内容覆盖安全规则。

### Redis 历史为什么不全部放进去

Redis 的 `load()` 会只保留最近 `max_rounds * 2` 条消息，也就是最近若干轮用户和 assistant 消息。

> 这样做是为了控制上下文长度和回答延迟。历史太长会增加请求成本，也可能让模型注意力被无关内容分散。

## 5. 查询 Redis 检索缓存

```python
sources = self.memory.get_query_cache(
    message,
    current_user.user_id,
    role.role_id,
    top_k,
)
```

Redis 缓存键在：`backend/app/memory/redis_memory.py:132-141`

```python
digest = hashlib.sha256(
    f"{query.strip().casefold()}\n{top_k or 0}".encode("utf-8")
).hexdigest()
return f"rag:cache:{user_id}:{role_id}:{digest}"
```

### 这个缓存键为什么这样设计

- `strip()`：忽略首尾空格。
- `casefold()`：尽量把大小写差异视为同一个查询。
- 加入 `top_k`：不同返回数量不能共用同一份结果。
- 加入 `user_id` 和 `role_id`：避免不同用户或不同角色混用缓存。
- SHA-256：避免把原始问题直接放进 Redis key，也让 key 长度稳定。

缓存命中时直接拿 `sources`；缓存没命中才进入完整检索：

```python
if sources is None:
    sources = self.retriever.retrieve(message, top_k)
    self.memory.set_query_cache(
        message, current_user.user_id, role.role_id, sources, top_k
    )
```

### 老师可能问：Redis 为什么适合放这里？

> Redis 是内存型键值数据库，读写速度快，适合短期历史和有过期时间的缓存。这里的检索结果不是永久事实，知识库更新后也可以通过 TTL 让它自动过期，所以不必把它当作唯一的永久存储。

---

# 第六部分：RAG 检索是怎么做的（13:00–14:30）

核心文件：`backend/app/rag/retriever.py`

## 1. 第一步：问题向量化

```python
vector = self.embedder.encode(
    query,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
).tolist()
```

### 通俗解释

> BGE-m3 把一句中文问题转换成一串数字，也就是向量。语义相近的句子在向量空间里距离更近，所以“最近压力大睡不着”可能和资料中的“睡眠问题与压力管理”匹配，即使两句话没有完全使用相同词语。

- `normalize_embeddings=True`：归一化向量，方便使用余弦相似度比较。
- `convert_to_numpy=True`：返回 NumPy 数组，便于后续处理。
- `tolist()`：转成普通 Python 列表，方便传给 Milvus 客户端。

## 2. 第二步：去 Milvus 做向量检索

```python
hits = self.client.search(
    collection_name=self.settings.milvus_collection_knowledge,
    data=[vector],
    anns_field="vector",
    search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=self.settings.rag_top_k,
    output_fields=[
        "chunk_id", "document_id", "source_file",
        "page_start", "page_end", "text",
    ],
)
```

### 每个参数怎么说

- `collection_name`：Milvus 中保存知识片段的集合，当前是 `knowledge_chunks`。
- `data=[vector]`：要搜索的用户问题向量。
- `anns_field="vector"`：告诉 Milvus 哪个字段是向量字段。
- `metric_type="COSINE"`：使用余弦相似度比较方向相似性。
- `ef=64`：HNSW 搜索时的搜索范围参数，通常搜索范围越大，准确性可能越好，但计算量也可能增加。
- `limit`：初步取回多少候选。
- `output_fields`：除了分数，还把文档 ID、页码和正文取回来，后面既要给模型看，也要展示给用户。

## 3. 第三步：BM25 关键词检索

向量检索擅长理解语义，但关键词检索对这些内容很有帮助：

- 专有名词。
- 固定术语。
- 关键词完全匹配。
- 用户问题中出现的特定风险词。

BM25 是一种经典的信息检索算法。它会根据查询词在文档中的出现情况、词频和文档长度计算相关度。

### 为什么不是只用向量检索

> 只用向量检索可能漏掉非常关键的精确词；只用关键词检索又可能理解不了同义表达。把二者合并，能同时利用语义相似和词面匹配，这就是混合检索。

## 4. 第四步：快速筛选，再做 Rerank

当前配置：

- 初步候选：大约 20 条。
- Rerank 前快速筛选：默认 8 条，配置项是 `rag_rerank_candidates`。
- 最终返回给模型和前端的主要结果：默认不超过 5 条，受 `rag_rerank_top_k` 和前端 `top_k` 影响。

### 为什么要先筛选

> BGE-reranker 比简单分数计算更准确，但 CPU 推理成本更高。如果直接对几十条候选全部精排，首字延迟会明显增加。当前先用向量分数、词面分数和 BM25 分数做快速筛选，再把前 8 条交给 reranker，减少计算量。

这也是项目解决“回答速度太慢”的关键优化之一。

## 5. 第五步：综合排序和去重

Rerank 后，系统会综合：

- 向量相似度。
- Rerank 分数。
- 词面匹配分数。
- BM25 分数。

然后限制同一个文档或同一页不能占满结果，避免最终五条资料全部来自一个重复页面。

### 老师可能问：RAG 和微调有什么区别

> RAG 不改变模型参数，而是在每次提问时临时检索外部知识并放进提示词。优点是知识更新方便、来源可追溯；微调是改变模型参数，更适合学习固定风格或任务模式，但更新知识和解释来源没有 RAG 方便。本项目需要展示心理健康资料来源，所以选择 RAG 更合适。

---

# 第七部分：DeepSeek 如何生成回答（14:30–15:30）

核心文件：`backend/app/services/deepseek_client.py`

`ChatService` 在这里调用：

```python
for part in self.deepseek.stream_answer(message, history, sources, role):
    answer_parts.append(part)
    yield {"event": "delta", "content": part, "value": 60}
```

## 1. 提交给模型的内容

`DeepSeekClient.stream_answer()` 会构造 messages：

```python
messages = [
    {"role": "system", "content": self._system_prompt(passages, role)},
    *history,
    {"role": "user", "content": question},
]
```

也就是：

1. 系统提示词。
2. 历史对话和确认过的长期记忆。
3. 本轮用户问题。

系统提示词中还会加入检索到的资料，格式大概是：

```text
[资料标题，第几页]
资料正文
```

并且通过 `rag_context_max_chars` 限制参考资料总长度，防止上下文无限增大。

## 2. 系统提示词包含哪些边界

`_system_prompt()` 明确要求模型：

- 只服务于心理健康科普资料。
- 根据参考资料回答，不编造资料中没有的事实。
- 不能进行正式医学诊断。
- 不能替代医生、心理咨询师或紧急救助。
- 超出资料范围要说明不确定性。
- 使用简体中文，语气温和、清晰、非评判。

### 老师可能问：提示词能完全防止模型胡说吗？

> 不能保证百分之百防止。提示词是约束，RAG 来源和安全分流是降低风险的措施。项目还通过展示来源、限制上下文、固定危机响应来提高可控性，但它仍然不是医疗诊断系统。

## 3. DeepSeek 流式调用

`deepseek_client.py:63-89` 使用：

```python
with httpx.stream("POST", url, ..., json={"stream": True}) as response:
    for line in response.iter_lines():
        ...
        content = payload.get("choices", [{}])[0].get("delta", {}).get("content")
        if content:
            yield str(content)
```

### 这段代码的意义

- `httpx.stream`：不等完整响应，持续读取网络数据。
- `stream=True`：告诉 DeepSeek 使用流式生成。
- `iter_lines()`：按行读取模型返回的 SSE 数据。
- `delta.content`：取这次新增的文字片段。
- `yield`：把片段交给上层 `ChatService`。

这里实际上有两层流式：

```text
DeepSeek 流式响应
  ↓ deepseek_client.py 逐片段 yield
ChatService 转成 delta 业务事件
  ↓ api/chat.py 包成项目自己的 SSE
浏览器 App.vue 逐片段显示
```

### 老师可能问：为什么要 `answer_parts.append(part)`？

> 页面需要边收边显示，所以每个片段马上 yield 出去；数据库最后又要保存完整答案，所以同时把所有片段追加到 `answer_parts`，最后用 `"".join(answer_parts)` 拼成完整文本。

---

# 第八部分：前端如何接收和展示 SSE（15:30–17:00）

前端的流式读取函数是 `App.vue` 中的 `readChatStream()`。

## 1. 获取可读流

```ts
const reader = response.body.getReader()
const decoder = new TextDecoder()
let buffer = ''
```

- `response.body`：HTTP 响应的字节流。
- `getReader()`：取得读取器。
- `TextDecoder()`：把二进制字节转成文字。
- `buffer`：暂存还没有拼成完整事件的半截内容。

## 2. 为什么需要 buffer

网络每次收到的 chunk 不一定刚好对应一条完整 SSE 事件：

```text
第一次收到：event: del
第二次收到：ta\ndata: {"content":"你好"}\n\n
```

如果第一次收到就直接解析，会失败。所以前端把每次解码结果追加到 `buffer`，只按 SSE 的空行切出完整事件，剩余半截继续留到下一轮。

### 老师可能问：`TextDecoder` 为什么要使用流式解码

> 中文 UTF-8 一个字符可能由多个字节组成，网络 chunk 可能正好从一个中文字符中间切开。流式解码会暂存不完整的字节，等下一次数据到来后再拼好，避免乱码。

## 3. 前端处理哪些事件

### `session`

保存后端创建或确认的 session ID，并让当前会话和后端一致。

### `progress`

更新页面上的进度百分比和阶段文字，例如：

- 正在生成问题向量。
- 资料检索完成，正在重排。
- 正在整理参考资料。
- 回答完成。

### `sources`

把后端返回的知识库来源挂到当前 assistant 消息上。右侧引用面板会显示标题、页码、分数和文本片段。

### `delta`

把 `data.content` 追加到当前 assistant 消息的 `content` 中，这就是回答逐步出现的原因。

### `done`

表示整条回答结束。前端保存 session ID，更新安全干预标记，停止 loading 和计时器。

### `error`

把后端错误转换成页面上的友好提示。

## 4. 前端为什么要判断有没有 `done`

> 如果网络突然中断，可能收到了一些文字但没有收到结束事件。前端可以据此判断这是一次异常中断，而不是正常完成，给用户一个明确提示。

## 5. 右侧来源为什么重要

> 右侧来源不是装饰，它体现了 RAG 的可追溯性。用户可以看到回答参考了哪份资料、哪几页和哪段文本。这样比一个只给结论、无法解释出处的黑盒回答更容易检查。

---

# 第九部分：答案保存到哪里（17:00–17:30）

`ChatService.stream_answer()` 结束生成后会执行：

```python
history_repo.add_message(chat_session, "assistant", answer, sources=sources)
history_repo.commit()
self.memory.save_turn(
    active_session,
    message,
    answer,
    current_user.user_id,
    role.role_id,
)
yield {"event": "done", ...}
```

### MySQL 保存什么

根据 `backend/app/models/chat.py`：

- `users`：用户信息和密码摘要。
- `ai_roles`：AI 角色和系统指令。
- `chat_sessions`：会话 ID、所属用户、标题、最后消息和角色。
- `chat_messages`：用户消息、assistant 回答、引用来源和时间。
- `documents`：文档元数据和处理路径。
- `knowledge_chunks`：知识分块、页码、文本摘要和 Milvus 集合信息。
- `long_term_memories`：用户主动确认的长期记忆。
- `rag_evaluation_runs`：评测任务和指标。

### Redis 保存什么

- 最近几轮聊天历史。
- RAG 检索结果缓存。
- 文档任务状态。

### 老师可能问：为什么 MySQL 和 Redis 都要保存聊天相关内容

> 两者职责不同。MySQL 是结构化、持久化存储，适合保存完整会话和审计记录；Redis 是高速、带过期时间的短期存储，适合快速读取最近上下文和检索缓存。Redis 挂掉时可以降级为空历史，但 MySQL 才是长期记录的主要来源。

---

---

# 第十部分：安全和性能总结（17:30–19:20）

## 1. 安全设计

现场可以这样说：

> 这个项目不是单纯把问题转发给大模型，而是在模型前面增加了几层保护。第一层是登录认证，防止未登录用户直接调用聊天接口；第二层是危机表达检测，高风险内容直接走固定安全提醒；第三层是系统提示词，明确不做正式医学诊断；第四层是引用来源，让回答可以追溯。真实生产环境还应该继续加强专业审核、日志脱敏和更完整的风险评估。

不要说“完全安全”或“可以诊断”。准确说法是“做了基础安全分流和边界约束”。

## 2. 性能优化

当前项目针对回答慢主要做了三件事：

1. 本地 embedding 和 rerank 固定使用 CPU，避免 Apple MPS/Metal 崩溃。
2. Rerank 前先快速筛选候选，默认从较多候选缩小到 8 条。
3. Redis 缓存相同用户、角色、问题和 top_k 下的检索结果。
4. DeepSeek 使用流式接口，让用户先看到首字，而不是等完整答案。

### 首字时间和完整时间有什么区别

> 首字时间是用户看到第一段回答所需的时间；完整时间是整段回答生成完所需的时间。SSE 不能让模型本身计算更快，但可以改善用户感知，并且让用户更早看到结果。

## 3. 后端为什么固定 CPU

> 本地 BGE-m3 和 reranker 默认可能尝试使用 Apple MPS。之前出现过 Metal command buffer 崩溃，所以当前在线检索显式指定 CPU，优先保证稳定性。代价是 CPU 推理可能比 GPU 慢，因此又通过候选筛选和 Redis 缓存降低耗时。

---

# 第十一部分：测试和现场演示（19:20–20:00）

## 建议现场演示顺序

1. 先打开项目架构图，指着离线链路说：原始 PDF 在 `data/raw/`，处理结果按阶段写入 `processed`、`cleaned` 和 `chunks`。
2. 如果条件允许，展示一个 `data/processed/{document_id}.json` 或 `data/chunks/{document_id}.json` 的结构，只展示普通文本和字段，不展示任何密钥。
3. 说明真正的向量不在 JSON 文件里，而是在 Milvus 的 `knowledge_chunks` 集合；`data/vectorized/manifest.json` 只是向量化任务摘要。
4. 打开 `http://localhost:5173`，展示登录页面并登录。
5. 选择默认心理健康角色。
6. 输入普通问题：
   - “最近压力比较大，晚上睡不着，有什么建议？”
7. 指出页面出现进度阶段，说明在线检索使用的是离线已经处理好的知识库。
8. 指出回答文字是逐步出现的，这就是 SSE。
9. 指出右侧有引用标题、页码、相关度和文本片段。
10. 最后打开后端健康检查或 OpenAPI 页面，说明 MySQL、Redis、Milvus 和知识集合已经连接。

### 现场演示时可以边操作边说

> 我现在输入的问题不会再次触发 PDF 解析。PDF 的解析、清洗、分块和向量化在离线阶段已经完成；此时后端只需要把当前问题向量化，去 Milvus 和本地 chunk 数据中检索相关内容，再把结果交给 DeepSeek。回答返回时，后端通过 SSE 一段一段推送，所以页面不是最后一次性出现全部文本。

### 如果想展示离线命令

```bash
PYTHONPATH=backend python -m app.ingestion.cli --clean
PYTHONPATH=backend python -m app.ingestion.cli --chunk
PYTHONPATH=backend python -m app.ingestion.cli --embed
```

> 说明：命令行参数最好分步执行，并先确认 `.env` 中的 MinerU、Milvus 和模型配置已经准备好。不要在讲师面前直接展示真实 API Key。

## 演示时不要做的事情

- 不要把真实 API Key、数据库密码或 `.env` 展示给讲师。
- 不要承诺系统能诊断抑郁症、焦虑症或其他疾病。
- 不要说“模型一定不会出错”。
- 不要把当前没有实现的意图路由、工具调用、RRF、MMR 说成已经存在。
- 不要说前端还有“我的偏好”和“管理中心”页面；这些入口已经删除，后端相关接口只是保留。
- 不要把自定义 access token 说成标准 JWT。

---

# 第十二部分：老师高概率追问和标准回答

## 1. 你的项目为什么要用 RAG？

> 心理健康问答需要参考相对稳定、可追溯的资料。只依赖大模型自身知识，来源不明确，也可能产生幻觉。RAG 先检索项目维护的心理健康 PDF，再把相关片段放进提示词，既能让回答参考项目知识库，也能展示标题和页码。

## 2. RAG 的完整流程是什么？

> 离线阶段把 PDF 解析、清洗、分块、向量化并写入 Milvus；在线阶段把用户问题向量化，和 Milvus 做语义检索，同时做 BM25 关键词检索，然后合并候选、Rerank 精排，把最终片段放入 DeepSeek 提示词中生成回答。

## 3. Milvus 和 MySQL 有什么区别？

> Milvus 主要解决向量相似度检索，MySQL 主要保存用户、会话、消息、文档和 chunk 元数据。Milvus 负责“找相似内容”，MySQL 负责“管理结构化业务数据”。

## 4. 为什么还要 BM25？

> 向量检索理解语义，BM25 更擅长精确关键词。心理健康资料中可能有固定术语、药物名、机构名或风险表达，混合检索可以减少只依赖单一方法带来的漏检。

## 5. Rerank 是做什么的？

> 向量检索和 BM25 得到的是初步候选，Reranker 会把用户问题和候选文本成对输入，重新判断二者相关性。它更精细，但计算成本更高，所以先筛选到少量候选再做。

## 6. `top_k` 是什么意思？

> 它表示希望最终保留多少条主要检索结果。它和内部初步候选数量不是一回事：系统可能先从 Milvus 和 BM25 取更多候选，再筛选和精排，最后返回较少的结果。

## 7. 为什么要保存页码？

> 页码是来源追踪的重要元数据。模型不仅可以参考正文，前端还可以告诉用户这段内容来自哪篇文档的哪几页，方便人工检查。

## 8. SSE 的全称和作用是什么？

> SSE 是 Server-Sent Events，服务器发送事件。它基于 HTTP，允许服务器在一个保持打开的响应里不断向浏览器推送文本事件。本项目把 DeepSeek 生成的片段转成 `delta` 事件，前端每收到一段就追加到消息气泡，所以回答可以实时显示。

## 9. SSE 和普通接口相比有什么区别？

> 普通接口通常等完整结果后返回一个 JSON；SSE 是持续返回多个事件。普通接口适合短结果，SSE 更适合大模型生成这种耗时较长、但可以逐步返回的场景。

## 10. SSE 和 WebSocket 有什么区别？

> SSE 主要是服务端到浏览器的单向推送，基于 HTTP；WebSocket 是双向长连接。这个项目是用户发一个问题、服务器持续返回回答，所以 SSE 更简单、够用。

## 11. `yield` 是什么意思？

> `yield` 让函数变成生成器。它不会一次性把所有结果算完，而是每次产生一个结果就交给调用者，下一次继续执行。这里用于把一个个 SSE 事件和一个个模型文本片段逐步输出。

## 12. `async` 没有用，为什么也能流式？

> 流式不等于一定要使用 async。当前代码通过生成器 `yield` 和 `StreamingResponse` 实现同步迭代式流式响应。后续如果并发量变大，可以进一步评估异步 HTTP 客户端和异步数据库访问，但当前实现的核心流式机制已经成立。

## 13. `Depends` 是什么意思？

> `Depends` 是 FastAPI 的依赖注入机制。接口声明依赖数据库 Session 或当前用户，FastAPI 会在执行接口前自动创建、传入，执行完后释放。这样数据库连接和认证逻辑不用在每个接口里重复写。

## 14. 为什么后端要有 Schema？

> Schema 用来定义请求和响应的结构，并自动做参数校验。比如聊天消息不能为空、不能超过 4000 字，top_k 只能在 1 到 20 之间。它把接口契约固定下来，减少前后端约定不一致。

## 15. `lru_cache` 是干什么的？

> 项目用它缓存配置、Retriever、RedisMemory 和 ChatService 的实例。这样同一个进程里不会每次请求都重新加载 BGE 模型、重新创建客户端。模型加载很重，复用单例可以减少延迟和内存浪费。

## 16. `normalize_embeddings=True` 是什么意思？

> 对向量做归一化，让向量长度变成统一尺度。这样计算余弦相似度时，重点更多落在向量方向，也方便不同文本之间比较。

## 17. 余弦相似度是什么？

> 它比较两个向量方向的接近程度，数值越接近 1 通常表示方向越相似。在文本向量检索中，可以把它理解成语义相似程度的一种衡量方式。

## 18. 为什么不把所有 PDF 直接传给大模型？

> 成本高、上下文长度不够、速度慢，而且会带入大量无关内容。分块加检索只把和当前问题相关的片段放进提示词，效率和可追溯性都更好。

## 19. 文档为什么要分块并且重叠？

> 分块是为了控制上下文大小和提高检索粒度；重叠是为了避免句子或段落在边界处被完全切断，保留一定上下文连续性。

## 20. 发生危机表达时系统怎么做？

> 后端 `is_crisis_message` 先做关键词规则检测。命中后，`ChatService` 直接使用固定的 `CRISIS_RESPONSE`，不执行普通检索和 DeepSeek 生成，并把 `safety_intervention` 标记为 true。回答会建议联系当地紧急服务、专业机构和可信任的人，同时说明系统不能替代紧急救助。

## 21. 关键词检测会不会漏掉风险？

> 会有这个可能。当前实现是基础的关键词检测，不是完整的心理风险评估。它的作用是提供一层明确、稳定的安全分流，不代表已经解决了全部安全问题。生产环境还需要专业规则、人工审核和更完整的风险模型。

## 22. 如果 Milvus 挂了会怎样？

> 健康检查会把 Milvus 和知识集合标记为不可用，返回 degraded 或 503。聊天接口的异常会被包装成友好的“在线问答暂时不可用，请稍后重试”。这是为了不把堆栈和敏感配置直接暴露给前端。

## 23. 如果 Redis 挂了会怎样？

> Redis 的历史读取和缓存读取函数会捕获 Redis 异常并记录日志，历史可以退化为空，缓存则视为未命中，系统仍可能继续走检索和数据库流程。Redis 主要是性能和短期记忆组件，不应该成为唯一的永久数据源。

## 24. 为什么以前回答慢，后来怎么优化？

> 主要瓶颈是 CPU 上的 BGE-reranker 对过多候选做精排。现在先通过向量、词面和 BM25 分数快速筛选，再只对默认 8 条候选做 Rerank；同时用 Redis 缓存重复检索结果，并使用 DeepSeek 流式输出改善首字体验。

## 25. 你做过哪些测试？

> 后端有长期记忆、在线安全和 Retriever 单元测试，之前完整测试结果是 13 个通过；前端执行过 `npm run build`，包含 TypeScript 检查和 Vite 打包；运行环境还通过健康检查验证 MySQL、Redis、Milvus 和 `knowledge_chunks` 集合。需要说明的是，完整生产级浏览器自动化、压力测试和长期监控还可以继续补充。

## 26. 你的项目现在是生产系统吗？

> 目前更准确的说法是已经达到可以演示和继续迭代的阶段，核心功能、服务连接、测试和构建都已跑通。要达到完整生产化，还需要继续完善部署守护、日志监控、浏览器自动化测试、压力测试、密钥轮换和更严格的安全审核。

## 27. 当前前端为什么没有管理中心和我的偏好？

> 根据当前产品范围，前端已经移除了这两个入口和页面逻辑，页面聚焦核心聊天、历史会话、角色选择和来源展示。后端的文档、评测和记忆 API 仍保留，是为了不破坏已有能力，也方便以后重新接管理端。

## 28. 你能指出用户输入问题后最关键的几个文件吗？

> 可以。前端首先看 `frontend/src/App.vue` 的 `retrieve()` 和 `readChatStream()`；后端入口看 `backend/app/api/chat.py`；业务编排看 `backend/app/services/chat_service.py`；检索看 `backend/app/rag/retriever.py`；大模型调用看 `backend/app/services/deepseek_client.py`；认证看 `backend/app/security/auth.py`；安全检测看 `backend/app/security/safety.py`；数据库模型看 `backend/app/models/chat.py`。

---

# 第十三部分：一页纸速记版

如果现场紧张，只记下面这段：

> 用户在 Vue 页面输入问题，`App.vue` 的 `retrieve()` 通过 `fetch` 向 `/api/v1/chat/stream` 发 POST 请求，请求里有消息、会话 ID、角色 ID 和 top_k，同时携带 Bearer token。FastAPI 的 `chat_stream` 路由通过 `Depends(get_current_user)` 先完成身份校验，再把请求交给 `ChatService.stream_answer()`。ChatService 先创建或恢复会话并发送 session 事件，然后判断是否是危机表达。命中危机就返回固定安全提醒，不调用普通检索和大模型；普通问题则先从 Redis 读取聊天历史和检索缓存，缓存没有命中时进入 `KnowledgeRetriever`。Retriever 用 BGE-m3 把问题转为向量去 Milvus 检索，同时用 BM25 做关键词检索，合并后先快速筛选，再用 BGE-reranker 对少量候选精排。检索结果和历史对话会组成 DeepSeek 的系统提示词，DeepSeek 以流式方式返回文本。后端把每个片段包装成 `delta` SSE 事件，前端通过 `ReadableStream`、`TextDecoder` 和 buffer 逐条解析，再追加到空的 assistant 消息气泡中。回答结束后，完整答案和引用来源保存到 MySQL，会话短期历史保存到 Redis。知识库中的 PDF 则提前经过 MinerU 解析、清洗、分块和 BGE-m3 向量化，向量放入 Milvus，文档和 chunk 元数据放入 MySQL，供在线检索使用。

---

# 第十四部分：最终讲解原则

1. **先讲流程，再讲函数。** 不要一上来念文件名，要先说数据正在从哪里流向哪里。
2. **每讲一个函数，都说输入、处理、输出。** 例如：`retrieve()` 输入用户文字，处理请求和流式读取，输出页面消息；`stream_answer()` 输入问题和用户，处理安全、检索和生成，输出 SSE 事件。
3. **区分在线和离线。** 在线是提问时检索和生成，离线是提前处理 PDF。
4. **区分 MySQL、Redis、Milvus。** MySQL 管业务数据，Redis 管短期和缓存，Milvus 管向量搜索。
5. **不要夸大安全能力。** 说“基础安全分流”，不要说“完全识别危机”。
6. **不要夸大模型能力。** 说“基于资料生成科普和陪伴回答”，不要说“诊断疾病”。
7. **讲 SSE 时一定讲清双向问题。** 本项目是浏览器发起请求、服务器持续推送文本，所以 SSE 足够，不必使用 WebSocket。
8. **讲性能时说真实优化。** CPU 稳定性、Rerank 候选筛选、Redis 缓存、流式首字体验，不要编造不存在的优化。
9. **被问到不会的细节时，回到文件和数据。** 先说“这个逻辑位于哪个模块”，再说明它接收什么、返回什么、为什么存在。
10. **保持诚实比背错更重要。** 可以明确说当前是演示和迭代阶段，哪些功能已实现、哪些属于后续增强。
