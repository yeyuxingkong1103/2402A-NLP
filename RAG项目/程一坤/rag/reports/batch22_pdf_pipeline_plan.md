# 批次 22 · 第二组方案：用真实 PDF 验证 PDF 解析链路（待批准）

> 状态：**方案阶段，未写任何实现代码**。批准后再实现并跑真实链路。
> 边界遵守：不改 docs/；临时脚本放项目根；新增依赖先报告；MinerU / Qwen-VL 的 key 一律不出现在代码与报告中。

---

## ⚠️ 实施后勘误（2026-09-21，实跑修正，以本节为准）

验收报告：`reports/batch22_pdf_pipeline_acceptance.md`

1. **上传 403 的根因写错了。** 原文第 2 步写「★ 这一步【不能带 Authorization】，否则 403」；
   实测真因是**不能带 `Content-Type`**（OSS 按空 Content-Type 签名，urllib 带 data 会自动补）：
   | 形态 | 结果 |
   |---|---|
   | curl 无 Content-Type | 200 |
   | urllib（自动补 Content-Type） | 403 SignatureDoesNotMatch |
   | urllib 显式 Content-Type: application/pdf | 403 |
   | http.client 不发 Content-Type | 200 |
   → 实现改为 `http_retry.request_raw`（http.client，只发给定的头）。
2. **分批不能只看页数。** 实测 `qwen-vl-ocr` 请求体超约 2MB 时**只返回第一张图的内容**
   （4 图/1.66MB→1929 字完整；6 图/2.45MB→400 字；8 图/3.28MB→399 字；逐页→3816 字）。
   新增配置 `QWEN_VL_MAX_REQUEST_BYTES`（默认 1.5MB），体积与页数双约束。
3. **页数来源要四层回退。** MinerU v4 不一定返回 `extract_progress`（本次未返回），
   vlm 后端结果包也不含页图 → 顺序改为 `extract_progress` → `layout.json.pdf_info` →
   包内 `*_origin.pdf`（PyMuPDF 数页）→ 页图数量。
4. **新增一个「方案里没写到」的修法**：PDF 分支必须先剥 Markdown 标记
   （MinerU 会把部分条文行提升为 `## 第十七条 …`，让 `^\s*第X条` 匹配不到 → 少 1 条）。
5. **cleaner 白名单实际包含「目录」**：不止章/节/条/总则/附则，PDF 的「目 录」也属结构标题。
6. 因上述第 1、3、4 条，`app/ingest/parser.py`、`app/ingest/title_normalizer.py` 也纳入了改动
   （方案里只写了 config / offline_ingest）。`embedding.py` 全程未改（哈希一致）。

---

## 0. 现状复核（先把缺口摆清）

| 事实 | 证据 |
|---|---|
| `parse_pdf` 契约已就位 | `app/ingest/pdf_parser.py`：MinerU 完整→直接用；不完整→Qwen-VL 兜底；都没有→`PdfParsingError` |
| **两个 client 都不存在** | `app/models/` 下只有 `embedding.py` / `llm.py` / `reranker.py` |
| **config 无 mineru/qwen 字段** | `grep -i "mineru\|qwen" app/core/config.py` → 空；`.env` 里 13 个键当前**无人读** |
| ⚠️ **PDF 进不了离线入库流程** | `offline_ingest.py:50` 调 `parse_document(source_path)` 只传 1 个参数（PDF 会抛 `ValueError`）；`collect_raw_files` 只收 `.html/.htm`（`offline_ingest.py:43`） |
| ⚠️ **rag 环境没有 PyMuPDF** | `fitz` / `pypdf` / `pdfplumber` 在 **anaconda base**（fitz 1.28.0）；rag env 只有 `PIL 12.2.0` |
| 清洗有一个会吃掉短行的规则 | `cleaner.py:74` 删「< 6 字符且不含标点」的孤行 —— PDF 里 `第一章`、`总则` 这类独立行有被删风险，验收 c 要专门看 |

---

## a) 已找到的真实 PDF（可公开访问、可溯源）

主目标选 **《中华人民共和国劳动合同法实施条例》**：它是国务院行政法规、条文结构规整（条/款/项齐全），且**本地法规库里已有它的 HTML 版本（documents #3）**，正好满足验收 d 的 PDF vs HTML 对照。

| # | 文件 | 来源（可溯源） | 页数 | 大小 | 文字层 |
|---|---|---|---|---|---|
| ★ | 中华人民共和国劳动合同法实施条例 | 江门市人社局「劳动法律和政策文件」栏目<br>`https://www.jiangmen.gov.cn/bmpd/jmsrlzyhshbzj/zcfg/ldgx/content/post_2982353.html`（附件 287576） | 10 | 261,817 B | **有**（4,607 字） |
| 2 | 中华人民共和国劳动合同法 | 同上（附件 287575） | 27 | 282,029 B | 有（12,348 字） |
| 3 | 中华人民共和国劳动法 | 同上（附件 287574） | 21 | 273,566 B | 有（8,931 字） |
| 4 | 劳动合同（通用）示范文本 | 南昌市人民政府 2025-07 公布（人社部编制版）<br>`https://www.nc.gov.cn/ncszf/shbz/202507/65ce5e3bd5f546fe8f446c374221f6d7.shtml` | 7 | 150,930 B | 有（3,584 字） |

- 库内 HTML 对照可用的是 #1/#2/#3（documents 表 #3 / #9 / #10）。
- 4 份都已用 PyMuPDF 检查过文字层（每页字符数非零），**没有用手造文件或项目内文件**。
- 落盘位置：`.pdf_probe/`（项目根，跑完报备删除）。

**关于验收 a 的两条分支：**

- 这 4 份**都有文字层**，MinerU 大概率直接返回完整 → 走 **mineru 分支**（这是主路径）。
- **qwen-vl 兜底分支**要真实触发，需要一份**无文字层的扫描件**。两条路，请选：
  - **R1（推荐）**：我再去找 1 份真实扫描件法规 PDF（地方人社局的老法规扫描版常见）。找到就两条分支都用真实 PDF 跑通。
  - **R2（保底）**：用注入的假 MinerU 客户端返回 `complete=False` 触发兜底，**但 Qwen-VL 那一步是对同一份真实 PDF 的真实调用**（输出真实）。触发条件人造，输出真实——报告里会明确标注是哪种。

---

## b) client 放哪、什么类名、怎么复用、怎么注入替身

### 放置位置

```
backend/app/models/mineru.py      # MineruClient
backend/app/models/qwen_vl.py     # QwenVlClient
backend/app/models/http_retry.py  # 新增：共用的 HTTP + 重试小工具
```

理由：本项目的「外部 API 适配器」一律在 `app/models/`（embedding / llm / reranker 都在），`app/ingest/` 是纯逻辑层、不做网络。与 `pdf_parser` 的注入式契约对齐（`parse_pdf` 只认「有 `parse()` 方法的对象」）。

### 类与方法（严格对齐现有契约）

```python
class MineruClient:
    def parse(self, source_path: Path) -> dict:   # {content, page_count, complete, error}

class QwenVlClient:
    def parse(self, source_path: Path) -> str     # 拼接后的页面文本
```

### MinerU 的调用流程（v4 标准 API，本地文件）

```
1) POST {base}/api/v4/file-urls/batch      Authorization: Bearer <key>
   body: {files:[{name, is_ocr, data_id}], language, enable_table, enable_formula, model_version:"vlm"}
   → {code:0, data:{batch_id, file_urls:[presigned]}}
2) PUT  <presigned_url>  上传字节   ← ★ 这一步【不能带 Authorization】，否则 403（双重鉴权）
3) GET  {base}/api/v4/extract-results/batch/{batch_id}  轮询
   → extract_result[0].state ∈ {pending, running, done, failed}
     done → full_zip_url；failed → err_msg
4) 下载 zip → 解压 → 取 full.md 作为 content，zip 内文件数/页码作 page_count
```

- 轮询间隔/上限取 `.env` 已有键：`MINERU_POLL_INTERVAL_SECONDS=3`、`MINERU_MAX_POLL_SECONDS=300`。
- `complete` 判定：`state==done` 且 `full.md` 非空且未触发截断；失败/超时 → `complete=False` 并把原因放进 `error`（**不抛异常**，交给 `parse_pdf` 决定是否兜底）。
- 限制：官方 200MB / 200 页。主目标 261KB / 10 页，远低于阈值。

### Qwen-VL 的调用方式（DashScope 兼容模式，模型 `qwen-vl-ocr`）

```
POST {base}/chat/completions      Authorization: Bearer <key>
{
  "model": "qwen-vl-ocr",
  "messages": [{"role":"user","content":[
      {"type":"image_url", "image_url":{"url":"data:image/png;base64,..."},
       "min_pixels": 3072, "max_pixels": 8388608},
      {"type":"text", "text":"Read all the text in the image."}
  ]}]
}
→ choices[0].message.content
```

- 该模型**内部固定使用 "Read all the text in the image."**，用户传入的文本不生效（官方说明）——所以我们不指望用 prompt 控制格式。
- 页面渲染：用 PyMuPDF 把每页渲染成 PNG（约 150 DPI），再 base64。按 `.env` 的 `QWEN_VL_MAX_PAGES_PER_REQUEST=8` 分批。

### 怎么复用现有 HTTP 客户端与重试

现有三种适配器各自实现了「urllib + 指数退避 + 429/5xx 才重试」的模式（`embedding.py` 最完整）。**不重构它们**（会波及 485 个既有测试），而是新增 `app/models/http_retry.py` 抽出同一套语义给新代码用：

```python
Transport = Callable[[str, dict[str,str], bytes, float], dict]   # 与 embedding/llm 同名同义

def is_retryable(error) -> bool: ...          # TimeoutError / 429 / 5xx → True
def call_with_retry(func, *, attempts, backoff, total_budget, logger, what) -> dict
def request_json(url, headers, payload, timeout) -> dict          # 默认实现（urllib）
def request_bytes(url, method, headers, payload, timeout) -> bytes  # 供 PUT 上传 / zip 下载
```

- 重试语义与 `embedding.py` 保持一致：**400/401/403 立即抛出，不浪费重试与额度**；超时/429/5xx 指数退避，受总预算约束。
- MinerU 的三种请求形态（POST json / PUT 上传 / GET 轮询+下载）都走这一套；`Authorization` 只在需要它的调用里加。

### 测试替身怎么注入

- **构造注入**，与 `embedding.py` / `llm.py` **完全同风格**：

```python
MineruClient(..., transport_json=..., upload=..., download=...)   # 三个传输点都可替换
QwenVlClient(..., transport=..., renderer=...)                    # renderer 可替换 → 测试不依赖 fitz
```

- `QwenVlClient` 的页面渲染抽成 `renderer: Callable[[Path], list[bytes]]`，默认实现才 import fitz。这样：
  - 单测可以传假 renderer（返回构造好的 PNG 字节），**不装 PyMuPDF 也能测全逻辑**；
  - 生产环境才真正渲染。
- 新增测试文件：`tests/test_mineru_client.py`、`tests/test_qwen_vl_client.py`（假 transport，覆盖：正常、轮询超时、failed 态、上传不带 Authorization、429 重试、401 不重试、返回结构异常）。

---

## c) config.py 要加的字段（键名与 .env 严格对齐）

`.env` 现有 13 个键目前无人读，本次全部接线（字段名 = 键名小写）：

```python
# ---------------- PDF 结构化解析（MinerU，批次 22）----------------
mineru_api_base_url: str = "https://mineru.net"   # MINERU_API_BASE_URL
mineru_api_key: str = ""                          # MINERU_API_KEY
mineru_poll_interval_seconds: float = 3.0         # MINERU_POLL_INTERVAL_SECONDS
mineru_max_poll_seconds: float = 300.0            # MINERU_MAX_POLL_SECONDS
mineru_language: str = "ch"                       # MINERU_LANGUAGE
mineru_enable_table: bool = True                  # MINERU_ENABLE_TABLE
mineru_enable_formula: bool = True                # MINERU_ENABLE_FORMULA
mineru_is_ocr: bool = False                       # MINERU_IS_OCR

# ---------------- PDF 兜底文字识别（Qwen-VL，批次 22）----------------
qwen_vl_api_base_url: str = ""                    # QWEN_VL_API_BASE_URL
qwen_vl_api_key: str = ""                         # QWEN_VL_API_KEY
qwen_vl_model: str = "qwen-vl-ocr"                # QWEN_VL_MODEL
qwen_vl_timeout_seconds: float = 120.0            # QWEN_VL_TIMEOUT_SECONDS
qwen_vl_max_pages_per_request: int = 8            # QWEN_VL_MAX_PAGES_PER_REQUEST
```

- bool 解析沿用现有风格：`.strip().lower() in ("1","true","yes","on")`。
- 另外可加 `build_mineru_client_from_settings()` / `build_qwen_vl_client_from_settings()` 两个装配函数（与 `build_chat_client_from_settings()` 同风格），保证「client 从哪来」只有一个出处。
- ⚠️ 这两个键**不进** `REQUIRED_IN_PRODUCTION`（PDF 解析是可选能力，不是问答链路的必需项）。

---

## 待批准的依赖（新增依赖先报告）

| 依赖 | 用途 | 现状 | 建议 |
|---|---|---|---|
| **PyMuPDF** | 把 PDF 页渲染成 PNG 供 Qwen-VL 识别 | anaconda **base** 有 1.28.0，**rag env 没有** | 在 rag env 装 `PyMuPDF==1.28.0`（与 base 对齐），并写入 `requirements.txt` |
| 其他 | zip 解压用标准库 `zipfile`；base64/urllib 均标准库 | — | **不需要新增** |

若你不想动 rag 环境，备选：`renderer` 默认实现延迟 import fitz，缺失时抛「需安装 PyMuPDF」的明确错误；**PDF 主路径（MinerU）不依赖它，仍可跑通**，只有 qwen-vl 兜底分支不可用。

---

## 验收计划（全部贴真实输出）

1. **分支证据**：真实调用后在报告里给出本次走的是 `mineru` 还是 `qwen-vl`（`ParsedDocument.parser_name`），并附 MinerU 轮询的 state 流转与耗时。
2. **正文**：打印正文前 300 字、总字符数、`page_count`。
3. **清洗 + 切块**：`chunk_document` 结果里 parent 条数 / child 数、有 `paragraph_no` 的块数、有 `item_no` 的块数；并**专门检查 `第一章`/`总则` 这类短行有没有被 `cleaner` 吃掉**（上面第 0 节的已知风险点）。
4. **PDF vs HTML 对照**：同一部《劳动合同法实施条例》，PDF 解析结果 vs 库内 HTML 版本（documents #3）——比条数、条号序列、正文一致性，列出差异并说明原因（页眉页码残留、分栏、异体字、标点差异等）。
5. **全量测试**：`485 passed` 不减；`scripts/check_services.py` 全绿。

---

## 风险与取舍（先说清）

| 风险 | 说明 | 处理 |
|---|---|---|
| 清洗吃掉章节短行 | `cleaner._is_navigation_line` 删「<6 字符且无标点」孤行，PDF 里 `第一章`、`总则` 可能中招 | 验收 3 专门检查；若确认被删，**先报告再决定**是否加白名单（不改动 cleaner 默认行为） |
| MinerU 对页码/页眉保留 | 江门 PDF 每页首行有 `—1—`/`- 1 -` 页码，可能进正文 | 观察清洗效果；必要时由 `repeated_headers/footers` 参数处理（`clean_text` 已支持） |
| Qwen-VL 输出无结构 | `qwen-vl-ocr` 只做文字识别，吐纯文本，章节号可能丢层级 | 兜底分支只保证「有文字」，切块质量如实报告，不美化 |
| 单次 LLM/解析耗时 | MinerU 异步任务 + 轮询，10 页文档预计 30~90s | 验收脚本打印各阶段耗时 |
| PDF 未接入生产入库 | `offline_ingest` 目前只吃 HTML | 本组**只做解析链路验证**，不动 `offline_ingest`；是否接入另开任务 |

---

## 请确认

1. qwen-vl 分支走 **R1（我再找真实扫描件）** 还是 **R2（假 MinerU 触发 + 真实 Qwen-VL 调用）**？
2. 是否批准在 **rag 环境安装 PyMuPDF**（以及记入 requirements.txt）？
3. `http_retry.py` 这个新文件可以加吗？（不重构 embedding.py，避免波及既有测试）
