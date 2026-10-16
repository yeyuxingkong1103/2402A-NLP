# 研发目录说明（解析 / 分块 / 索引）

> 工单编号：**人工智能NLP-RAG-图像内容解析及检索优化**
> 本文件面向 **T5（检索生成）/ T6（评测）/ 验收人**：说明索引产物怎么加载、字段口径与**已知限制**。
> 全局交付结构见顶层 `README.md`；测试分层见 `测试/README.md`。

---

## 1. 目录速览

| 路径 | 内容 |
| --- | --- |
| `app/core/**` | 解析与索引核心（工单3 同源实现，只读复制后按工单4 契约适配）：`pdf_parser` 逐页解析、`table_parser` 表格归一化与跨页续表、`chunker` 分块（文本/表格/图像三模态）、`vector_store` FAISS 索引、`bm25_index` 自实现 BM25、`embedder` 本地 Ollama 嵌入、`milvus_store` 可选 Milvus 适配器 |
| `scripts/parse_corpus.py` | 离线解析驱动（文本+表格 → JSONL + SQLite） |
| `scripts/build_index.py` | 建索引驱动：`--incremental`（按内容 sha1 增量嵌入）、`--include-images`（图像块一等 chunk）、`--verify-only`（只读体检） |
| `scripts/extract_images.py` | 图像资产/图像块（T2 主责；`--assert-only` 可对已落盘产物跑题5/题6断言） |
| `tests/` | **研发自测层**：`test_table_defects.py`（表格三大缺陷 + 图像块契约，23 passed） |
| `data/raw/` | 两份招股说明书 PDF（只读语料） |
| `data/processed/` | 解析产物：`*.pages.jsonl`、`*.tables.jsonl`、`*.text_blocks.jsonl`、`*.parse_stats.json`、`chunks.jsonl`、`image_blocks.jsonl`（T2）、`evidence/`（T2 证据图） |
| `data/index/` | `bge-m3_1024/`（`vectors.npy`、`ids.json`、`faiss.index`、`bm25.pkl`、`bm25.meta.json`、`chunks_indexed.jsonl`）、`rag.sqlite3`、`index_manifest.json`、`q16_retrieval_recall.json` |

## 2. 索引加载（T5 入口）

```python
IDX = "研发/data/index/bge-m3_1024"          # 相对仓库根
from app.core import vector_store, embedder
from app.core.bm25_index import BM25Index
from app.core.chunker import Chunk
from app.core.text_utils import read_jsonl

idx  = vector_store.load_vector_index(IDX, expected_dim=1024, use_faiss=True)  # 读盘 faiss.index（非 ASCII 路径自动回退 deserialize）
bm   = BM25Index.load(IDX)                                                   # bm25.pkl
qvec = embedder.embed_query(question)                                        # bge-m3:latest → (1024,)

vec_hits  = vector_store.search_vectors(idx, qvec, top_k=20)                 # [(chunk_id, score)] 降序
bm25_hits = bm.search(question, top_k=20)                                    # [(chunk_id, score)] 降序

# chunk 元数据：chunks_indexed.jsonl 的**行序 == vectors.npy 行序 == ids.json 行序**（实测逐元素一致）
chunks = [Chunk.from_dict(r) for r in read_jsonl(IDX + "/chunks_indexed.jsonl")]
```

* 融合纪律：向量（bge-m3）、BM25、CLIP 图像通道属**不同语义空间** → 用 **RRF 排名融合**，**禁止把相似度直接相加**。
* 过滤/降权：`chunk.type`（= `modality`：`text`/`table`/`image`）、`chunk.modality_sub`（`diagram`/`chart`/`picture`）、`chunk.evidence_weight`（**picture = 0.4**，其余 = 1.0）。
* 只读体检：`pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py --verify-only`（不解析、不嵌入、不改索引）。

## 3. 字段口径

| 字段 | 口径 |
| --- | --- |
| `page` | **1-based 物理页码**，且带 `page_basis="1-based-physical"`（印刷页码 = 物理页码 − 1，引用一律用物理页） |
| `type` / `modality` | 同值双写（工单3 用 `type`，设计文档用 `modality`） |
| `chunk_id` | `pdf{N}-p{page}-x{i}`（文本）／`-t{i}-{part}`（表格）／`-img-{xref}-{i}`（图像） |
| `source_chunk_id` | T2 产物里的原始图像块 id（`img:<corpus>:p<page>:<sub>[:x<xref>]`） |
| `bbox` / `sub_bboxes` / `structured` | PDF pt 坐标与结构化字段（框图树、图表数值表等），用于回溯渲染 |

## 4. 已知限制（**与 `index_manifest.json` 的 `known_limitations` 同源**）

1. **`chunk_id` 与 T2 图像 id 不是一一对应**：本索引把「同页 + 同 modality_sub」的多个子图**合并成一个 chunk**（页 72 的 xref1692 条形图 + xref1693 3D 饼图 → 同一个 `pdf2-p72-img-1692-0`），且 `chunk_id` 里的 xref 只取组内第一条 → 精确回溯请用 `source_chunk_id` + `sub_blocks`（含每个子图 xref/bbox）。框图 `xref=null`，回溯依据是 `structured.bbox_by_node` / `structured.recovered_text`。
2. **faiss 1.7.4 无法在非 ASCII 路径 `write_index`**（本仓库路径含中文）→ 已改用 `faiss.serialize_index` 字节序列化落盘、`deserialize_index` 读回；numpy 精确检索为等价兜底。详见 `ids.json.faiss_note`。
3. **`picture` 是弱证据**（只有 CLIP 零样本标签）：`evidence_weight = 0.4`，检索层应降权或走单独通道，不得与 `diagram`/`chart` 等权。
4. **`q16_retrieval_recall.json` 的计数是上游自检（页级口径）**：`*_page_hit_count` = T3 三层口径的 `hit_page_level`（诊断层B）；T6 命中率主指标用权威层 `hit_complete`，**本文件计数不得当作准确率**（见 `keyword_rules.counts_definition` 与 `authoritative_judge`）。
5. **图像覆盖来自 T2 产物**：本轮两份 PDF 均有覆盖，但 CLIP 编码有 `--clip-limit`（默认每份 PDF 300 张）上限，超限位图不编码、不进索引；本索引不代为声明「全部内嵌图已解析」。
6. **位图内文字无 OCR**（本机无 OCR）：位图文字按 CLIP 零样本 + 几何标定处理，方法级别与精度见 `设计/图像解析方案.md` §9；**不得声称 OCR**。
7. **可复现性的适用边界（想用 hash 判断「索引有没有变」必须先读这条）**：
   * **稳定**：**数据文件**（`vectors.npy`、`chunks_indexed.jsonl`）跨运行**字节稳定**；`ids` 列表内容与 `bm25.pkl` 内的 ids 列表跨运行**内容稳定**（已实测幂等，见 §5 的校验片段）；
   * **不稳定**：`ids.json` 与 `index_manifest.json` 的**文件字节**每次重跑都会变，差异**仅来自 3 个 ISO 时间戳字段** —— `ids.updated_at`、`manifest.built_at` / `created_at` / `vector.updated_at`（captain 2026-10 逐一探测确认无其它易变字段）；
   * **结论**：判定「索引是否变化」请比对**数据文件**或**内容指纹**，**不要**比对元数据文件字节，否则会得出「每次都变」的错误结论。
8. **`chunk_id` 与 T2 图像 id 不是一一对应**（同 §4.1）：同页同 `modality_sub` 的多个子图会**合并成一个 chunk**，`chunk_id` 里的 xref 只取组内第一条 → 精确回溯用 `chunk.source_chunk_id` + `chunk.sub_blocks`；**框图** `xref=null`，回溯依据是 `structured.bbox_by_node` / `structured.recovered_text`。

> 第 1/4/7 条与 `index_manifest.json` 的 `known_limitations` 同源（由 `build_index.py` 生成，**非手改产物**，重跑必在）。

## 5. 复现命令

```powershell
cd E:\gao6gongdan\工单4
$env:PYTEST_ADDOPTS='-p no:cacheprovider'                      # 避免在仓库根生成 .pytest_cache 游离目录
pwsh -NoProfile -File run_py.ps1 -m pytest 研发/tests -q       # 解析/索引单测（24 passed）
pwsh -NoProfile -File run_py.ps1 研发/scripts/parse_corpus.py  # 重新解析（文本+表格）
pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py --incremental --include-images --verify-questions
pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py --verify-only   # 只读体检
```

### 校验「索引有没有变」（幂等性自检，只读）

```python
# 比对数据文件与内容指纹 —— **不要**比对 ids.json / index_manifest.json 的字节（见 §4.7）
import hashlib, json
from pathlib import Path
import numpy as np
IDX = Path("研发/data/index/bge-m3_1024")
ids = json.loads((IDX / "ids.json").read_text(encoding="utf-8"))["ids"]
print("ids 内容指纹 :", hashlib.sha1("\n".join(ids).encode()).hexdigest()[:16], "| n =", len(ids))
print("vectors 指纹 :", hashlib.sha1(np.load(IDX / "vectors.npy").tobytes()).hexdigest()[:16])
print("chunks 指纹  :", hashlib.sha1((IDX / "chunks_indexed.jsonl").read_bytes()).hexdigest()[:16])
```
