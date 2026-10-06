# RAG_2 统一离线包（`rag2`）

本目录把 RAG 数据前置链路的各模块打包成一个统一入口的 Python 包 `rag2`，
并内置 **SQLite 离线向量库**与**端到端离线流水线**，无需 Milvus / MySQL 服务器即可跑通：

```
文件 → data_type 识别类型 → mineru_parser / pdf_table / ocr 解析 → 分块 → 向量化
     → 离线：store.py（SQLite 本地库）
     → 在线：hybrid_retriever.py（Milvus + BM25 + 重排）
```

## 目录结构

```
RAG_2/
├─ rag2/                    # 统一包：import rag2
│  ├─ __init__.py           #   聚合导出所有模块
│  ├─ data_type.py          #   文件类型识别（纯标准库）
│  ├─ mineru_parser.py      #   MinerU 文档解析（纯标准库 + mineru 环境）
│  ├─ pdf_table.py          #   PDF 表格抽取（pdfplumber）
│  ├─ ocr.py                #   图片 OCR（PaddleOCR-VL，ocr_ 环境）
│  ├─ hybrid_retriever.py   #   Milvus + BM25 混合检索 + bge 重排（在线）
│  ├─ mysql_client.py       #   MySQL 读写（pymysql）
│  ├─ store.py              #   SQLite 离线向量库（离线，纯标准库）
│  ├─ pipeline.py           #   离线端到端流水线（离线）
│  ├─ ragas_eval.py         #   RAGAS 结果评估（离线打分，接本地 Ollama + bge-m3）
│  ├─ logging_config.py     #   统一日志系统（rag2.* logger + 落盘）
│  ├─ config.py             #   在线阶段配置加载（config.yaml + RAG2_* 环境变量）
│  ├─ roles.py              #   角色定义与加载
│  ├─ llm_client.py         #   OpenAI 兼容 LLM 客户端（默认 Ollama）
│  ├─ redis_memory.py       #   Redis 短期记忆（TTL 自动过期）
│  ├─ online_chat.py        #   在线 RAG 问答编排器（多轮/多角色/多用户）
│  ├─ server.py             #   FastAPI 在线服务（SSE 流式 + 鉴权 + 评估接口）
│  └─ webui/                #   网页客户端（index.html / app.js / style.css）
├─ config.yaml              # 在线服务配置
├─ roles.yaml               # 内置角色表
├─ run.py                   # 在线服务启动入口
├─ deploy/                  # 部署脚本（install.bat / start.bat / start.sh）
├─ docs/                    # 部署文档 / 用户使用手册
├─ example.py               # 调用示例
└─ requirements.txt         # 依赖说明（仅供记录）
```

## 快速开始

```python
import rag2

# 离线：全链路无需服务器，仅一个 .sqlite 文件
rag = rag2.OfflineRAG(db_path="kb.sqlite", embed_model="D:/modelscope/bge-m3", device="cuda")
rag.add_file("文档.pdf")
rag.add_file("扫描件.png")
for h in rag.search("问题", top_k=3, mode="hybrid"):
    print(h.score, h.text)
```

所有模块同时可通过 `import rag2` 直接访问（如 `rag2.detect`、`rag2.parse_pdf`、`rag2.HybridRetriever`），或 `from rag2.data_type import detect` 按子模块访问。

---

# 一、data_type — 文件类型识别

识别文件/字节流的类型，并给出建议的 RAG 解析器（PDF→MinerU、图片→OCR 等），用于入库前路由。

```python
from rag2 import detect, detect_bytes, scan_directory, summarize

info = detect("农业知识.pdf")
print(info.kind, info.mime, info.loader)   # pdf application/pdf rag2.mineru_parser.parse_pdf

infos = scan_directory("data", recursive=True)
print(summarize(infos))                    # {'pdf': 3, 'markdown': 1, ...}
```

识别策略：空文件 → 魔数（`%PDF-` / `\x89PNG` / `PK` 等）→ 文本判定 → 内容嗅探（编码探测 + 扩展名）→ 兜底。

| 函数 | 说明 |
| --- | --- |
| `detect(path)` / `detect_bytes(data, name=None)` | 识别文件 / 字节流 → `FileInfo` |
| `scan_directory(path, recursive=True)` | 扫描目录 → `List[FileInfo]` |
| `summarize(infos)` | 统计各类别数量 → `dict` |

---

# 二、mineru_parser — MinerU 文档解析

封装 `mineru-kit parse <输入.pdf> -o <输出目录> --tier flash --format zip`，用标准库 `zipfile` + `json` **免解压**读取结果。

```python
from rag2 import parse_pdf

result = parse_pdf("输入文件.pdf", output_dir="outputs", tier="flash")
print(result.markdown)             # Markdown 正文
for name, data in result.images:   # 提取的图片 [(文件名, 字节)]
    ...
```

| 函数/类 | 说明 |
| --- | --- |
| `parse_pdf(pdf, output_dir=None, tier="flash", ...)` | 解析 PDF → `ParseResult` |
| `read_zip(zip_path)` / `start_server(*args)` | 读已有结果 zip / 启动 server |
| `MineruParser(...)` | 可复用解析器类 |

---

# 三、hybrid_retriever — Milvus + BM25 混合检索（在线）

封装 **Milvus 向量检索 ⊕ BM25 关键词检索 → 加权 RRF 融合 → 可选 bge 交叉编码器重排**。

```python
from rag2 import HybridRetriever

retriever = HybridRetriever(uri="http://localhost:19530", collection="my_kb",
                            dim=1024, embed_fn=embed,
                            rerank_model=rag2.DEFAULT_RERANK_MODEL, device="cuda")
retriever.ingest(texts=["……"], metadatas=[{"source": "a.pdf", "page": 1}])
hits = retriever.search("问题", top_k=5, mode="hybrid", rerank=True)
for h in hits:
    print(h.score, h.extra.get("pre_rerank_score"), h.text)
```

| 方法 | 说明 |
| --- | --- |
| `ingest(texts, vectors, metadatas, ids)` | 入库（写 Milvus + 重建 BM25） |
| `search(query, top_k, mode, rerank=False, rerank_top_k=None)` | 混合检索 → `List[Hit]` |
| `search_dense` / `search_bm25` / `rerank` | 单路检索 / 交叉编码器打分 |
| `rebuild_from_milvus()` / `ping()` / `count()` / `drop()` | 维护 |

重排：`rerank=True` 时先召回 `top_k*3` 候选再精排，`hit.score` 为重排分，原融合分存于 `hit.extra["pre_rerank_score"]`。

---

# 四、ocr — 图片文字识别（PaddleOCR-VL）

把「图片 → 文字 → 保存为 txt」串成一次调用。

> ⚠️ paddleocr 仅装在 `ocr_` 环境，请用 `D:/an/envs/ocr_/python.exe` 运行。

```python
from rag2 import ocr_image, ocr_images

r = ocr_image("扫描件.png")              # 识别并保存 扫描件.txt
print(r.text)                            # 识别文本
results = ocr_images(["a.png", "b.jpg"], output_dir="out_txt")  # 批量
```

---

# 五、pdf_table — PDF 表格抽取（pdfplumber）

抽取 PDF 表格并做轻量分析：识别表头、去空行/空列、转记录字典、导出 CSV/JSON。

```python
from rag2 import extract_tables, analyze_pdf, tables_to_csv

tables = extract_tables("报表.pdf")
for t in tables:
    print(t.page_number, t.header, t.to_dicts())
```

---

# 六、mysql_client — MySQL 读写（pymysql）

查询返回**字典行**，支持事务、上下文管理器、批量写入。

```python
from rag2 import MySQLClient

db = MySQLClient(host="localhost", user="root", password="xxx", database="mydb")
rows = db.query("SELECT id, name FROM user WHERE age > %s", (18,))
new_id = db.insert("user", {"name": "李四", "age": 20})
```

---

# 七、store — SQLite 离线向量库（离线）

用标准库 `sqlite3` 在**单个本地文件**里存储「文本 + 归一化向量 + 元数据」，提供向量（余弦）、关键词（BM25）、混合（RRF）三路检索，是 `hybrid_retriever` 的离线替代。

```python
from rag2 import OfflineStore

db = OfflineStore("kb.sqlite")
db.add(texts=["……"], vectors=[[0.1, 0.2, ...]], metadatas=[{"source": "a.pdf", "page": 1}])
hits = db.search("问题", top_k=3, embed_fn=embed, mode="hybrid")  # vector/keyword/hybrid
for h in hits:
    print(h.chunk_id, h.score, h.text)
```

| 方法 | 说明 |
| --- | --- |
| `add(texts, vectors, metadatas, ids)` | 写入分块 + 向量（chunk_id 相同覆盖，幂等） |
| `search(query_or_vector, top_k, embed_fn, mode)` | 向量 / 关键词 / 混合检索 → `List[Hit]` |
| `search_vector` / `search_keyword` | 单路检索 |
| `get(chunk_id)` / `count()` / `delete(ids)` / `clear()` | 维护 |

> 向量入库前自动 L2 归一化，检索时点积即余弦相似度；关键词路复用 `hybrid_retriever.BM25Index`；混合路复用 `weighted_rrf`，与在线版行为一致。

---

# 八、pipeline — 离线端到端流水线（离线）

把各模块串成一条**全离线**链路，无需任何服务器：

```
文件 → data_type 识别 → mineru / ocr / 直接读取 → 分块 → 向量化 → OfflineStore 入库 → 检索
```

```python
from rag2 import OfflineRAG

rag = OfflineRAG(db_path="kb.sqlite", embed_model="D:/modelscope/bge-m3", device="cuda")
rag.add_file("文档.pdf")     # pdf/office → MinerU（失败回退 pdfplumber）
rag.add_file("扫描件.png")   # image → PaddleOCR-VL
rag.add_texts(["……"], doc_id="notes")  # 直接入库文本
for h in rag.search("问题", top_k=3, mode="hybrid"):
    print(h.score, h.text)
```

| 方法 | 说明 |
| --- | --- |
| `add_file(path)` | 识别 + 解析 + 分块 + 向量化 + 入库 → 摘要 dict |
| `add_texts(texts, metadatas, doc_id)` | 文本分块入库 |
| `search(query, top_k, mode)` | 离线检索 |
| `embed(texts)` / `count()` | 向量化 / 计数 |

`chunk_text(text, chunk_size=500, chunk_overlap=0)` 为按段落合并的分块工具函数。

---

# 九、ragas_eval — RAGAS 结果评估（离线打分）

封装 **RAGAS 0.4.x（现代组件版）**，给检索结果 + 生成答案打分，指标 0~1，越接近 1 越好：

| 指标 | 说明 | 所需字段 | 后端 |
| --- | --- | --- | --- |
| `faithfulness` | 忠实度（答案是否忠于检索上下文） | 问题 + 答案 + 上下文 | LLM |
| `answer_relevancy` | 答案相关性 | 问题 + 答案 | LLM + 向量 |
| `context_precision` | 上下文精确率 | 问题 + 上下文 + 参考答案 | LLM |
| `context_recall` | 上下文召回率 | 问题 + 上下文 + 参考答案 | LLM |
| `context_relevance` | 上下文相关性 | 问题 + 上下文 | LLM |
| `answer_correctness` | 答案正确性 | 问题 + 答案 + 参考答案 | LLM + 向量 |
| `context_entity_recall` | 上下文实体召回 | 上下文 + 参考答案 | LLM |
| `semantic_similarity` | 语义相似度 | 答案 + 参考答案 | 向量 |

```python
from rag2 import RagasEvaluator, evaluate_rag

# 一、单条评估
ev = RagasEvaluator()                     # 默认：本地 Ollama(Qwen3.5:4B) + 本地 bge-m3
scores = ev.evaluate(
    question="什么食物富含钾元素？",
    answer="香蕉富含钾元素。",
    retrieved_contexts=["香蕉每100克含钾约256毫克。", "苹果含钾约107毫克。"],
    reference="香蕉是富含钾的水果。",
)
print(scores)                             # {'faithfulness': 1.0, 'answer_relevancy': 0.60, ...}

# 二、直接对离线流水线打分（自动取回 top_k 上下文）
rag = rag2.OfflineRAG(db_path="kb.sqlite", embed_model="D:/modelscope/bge-m3")
print(rag.evaluate("什么食物富含钾元素？", answer="香蕉富含钾元素。", reference="香蕉富含钾。"))

# 三、批量评估
rows = [
    {"question": "……", "answer": "……", "retrieved_contexts": ["……"], "reference": "……"},
    ...
]
for r in ev.evaluate_batch(rows, metrics="generation"):
    print(r)
```

| 函数/类 | 说明 |
| --- | --- |
| `RagasEvaluator(...)` | 评估器：可配 `llm_model` / `llm_base_url` / `embed_model` / `device` 等 |
| `evaluate(question, answer, contexts, reference, metrics)` | 单条打分 → `{指标: 得分}` |
| `evaluate_batch(rows, metrics)` | 批量打分 → 记录列表（追加得分列） |
| `evaluate_rag(rag, question, answer, reference, top_k)` | 对检索器/流水线打分 |
| `OfflineRAG.evaluate(...)` | 离线流水线内置方法（等价 `evaluate_rag`） |
| `ragas_available()` / `available_metrics()` | 环境探测 / 指标清单 |

- **指标分组**：`metrics="all"`（全部）、`"generation"`（生成类）、`"retrieval"`（检索类）、`"no_reference"`（无需参考答案）；也可传指标名列表。
- **缺字段自动跳过**：未给 `answer` 时跳过需要答案的指标，未给 `reference` 时跳过需要参考答案的指标。
- **LLM 后端**：默认接本地 Ollama（`http://localhost:11434/v1`）；对 Qwen3.5 等思考模型自动传 `reasoning_effort="none"` 关闭思考链，避免结构化输出被截断。接 OpenAI / vLLM / DeepSeek 等只需改 `llm_model` / `llm_base_url` / `llm_api_key`。

> 依赖 `ragas>=0.4` + `openai`，向量指标额外需要 `sentence-transformers` + 本地 bge-m3；运行环境 `rags_`。

---

# 十、在线阶段 —— 多轮 / 多角色 / 多用户 + Redis 短期记忆 + 网页端

把「检索（`HybridRetriever`）→ 生成（`LLMClient`）」串成完整的在线问答服务，
支持多轮对话、多角色、多用户，并用 Redis 做**短期记忆**（只记当前对话，TTL 自动过期）。

```bash
# 启动服务（依赖 Redis / Milvus / Ollama 已就绪）
D:/an/envs/rags_/python.exe run.py

# 浏览器打开
#   http://127.0.0.1:8000
```

| 组件 | 说明 |
| --- | --- |
| `rag2.server` | FastAPI 服务：SSE 流式问答、鉴权、会话 CRUD、RAGAS 评估接口 |
| `rag2.online_chat` | `RAGChat` 编排器：取近 N 轮记忆 → 混合检索 → 角色化提示词 → 生成 → 回写 Redis |
| `rag2.redis_memory` | `RedisMemory`：用户 / 令牌 / 会话 / 消息，前缀 `rag2:`，`history_ttl` 自动过期 |
| `rag2.llm_client` | `LLMClient`：OpenAI 兼容（默认 Ollama），`chat()` / `stream()` |
| `rag2.roles` | 角色表（农业专家 / 技术支持 / 通用助手 / 客服） |
| `rag2.config` | `load_config()`：`config.yaml` + `RAG2_*` 环境变量覆盖 |
| `rag2.logging_config` | `setup_logging()` / `get_logger()`：统一 `rag2.*` 日志，落盘 `logs/rag2.log` |

```python
from rag2 import RAGChat

chat = RAGChat()
for event in chat.stream_answer("小麦常见病虫害有哪些？", user_id="alice", role_id="agriculture_expert"):
    print(event["type"], event.get("text") or event.get("answer") or "")
```

**多用户隔离**：用户名即身份（`/api/auth/login` 仅需用户名）；会话/记忆/令牌按用户隔离。
**多角色**：每个角色有独立 `system_prompt`，检索共用同一 Milvus 集合。
**多轮对话**：`memory.short_term_turns`（默认 5 轮）的历史拼进提示词。
**网页端**：零 CDN 原生 JS，登录 → 选角色 → 会话管理 → SSE 流式聊天 → 引用展示 →「评估本条」RAGAS 打分。
**管理员后台**：`admin.usernames`（默认 `admin`）登录后出现「⚙ 管理后台」，可在线改对话模型 / 检索参数 / 记忆与 TTL / 角色，保存即热生效并写回 `config.yaml` / `roles.yaml`。

> 部署详见 `docs/部署文档.md`，使用详见 `docs/用户使用手册.md`。

---

## 依赖与运行环境

| 模块 | 依赖 | 运行环境 |
| --- | --- | --- |
| `data_type` / `mineru_parser` / `store` | 纯标准库 | 任意 |
| `mineru_parser` 实际解析 | mineru-kit | `D:\an\envs\mineru` |
| `pdf_table` | pdfplumber | `D:\an\envs\rags_` |
| `ocr` | paddleocr | `D:\an\envs\ocr_` |
| `hybrid_retriever` | pymilvus / jieba / sentence-transformers | `D:\an\envs\rags_` |
| `mysql_client` | pymysql | `D:\an\envs\rags_` |
| `store` / `pipeline` | sqlite3（标准库）+ sentence-transformers（向量化） | `D:\an\envs\rags_` |
| `ragas_eval` | ragas + openai + sentence-transformers（向量）+ 本地 Ollama | `D:\an\envs\rags_` |

## 与既有项目的关系

- `hybrid_retriever` 稠密路对应 `RAG_1/src/store.py` 的 `MilvusStore`；BM25 对应 `RAG_1/src/bm25.py`；融合对应 `RAG_/src/role_rag/retrieval/fusion.py`。
- `mineru_parser` 对应 `RAG_1/src/parse_doc.py`（升级为 `--format zip` + 免解压读取）。
- `store`（离线）与 `hybrid_retriever`（在线）接口对齐，均返回 `Hit`，可在有/无服务器场景间无缝切换。
