# v2 检索重排实现计划

> **面向 AI 代理的工作者：** 使用 superpowers:test-driven-development 逐任务实现此计划（先写失败测试 → 实现 → 验证）。步骤使用复选框（`- [ ]`）跟踪进度。**注意：当前目录未初始化 git，所有 Commit 步骤跳过**（v1 计划中的 commit 命令不再出现）。
>
> **规格：** `docs/v2/spec.md`；**设计：** `docs/superpowers/specs/2026-09-01-v2-retrieval-rerank-design.md`

**目标：** 在 v1 检索链路上新增「粗排 + 精排」两阶段重排：RRF 召回扩大到 `retrieval_candidate_k`(30) → 粗排（BGE-M3 dense 余弦，可配置 scorer）取 `coarse_top_k`(10) → 精排（bge-reranker-v2-m3 交叉编码器）取 `final_top_k`(6) → 按 `min_retrieval_score` 过滤 → 引用。`rerank_enabled=false` 时完全退化为 v1 行为。评测脚本支持 `--variant v1|v2|both`，产出 recall/MRR/延迟对比。

**架构：** 新增 `backend/app/rerank.py`（CoarseScorer/CoarseReranker/FineReranker/RetrievalChain）；`vector_store.search` 增加 `with_vectors` 返回 dense 供粗排；`routes_chat` 改为经 `RetrievalChain` 编排；`run_sf6_eval.py` 增加 variant 与延迟采集。

**技术栈：** Python 3.12（`D:\an\envs\mineru`）、FastAPI、Qdrant、本地 bge-m3、本地 bge-reranker-v2-m3（FlagReranker）、pytest。精排模型路径 `D:\八维学习\bge-reranker-v2-m3`（用户确认已下载）。

---

## 宪法检查

- **技术栈合规**：精排用 bge-reranker 家族（bge-reranker-v2-m3），本地离线加载，**禁止走 Ollama** ✅
- **目录边界**：代码只落 `backend/`、`scripts/`；文档只落 `docs/`；评测产物落 `eval/` ✅
- **依赖锁定**：`requirements.txt` 精确版本；若 FlagEmbedding 1.3.3 无法加载 v2-m3，升级锁定新版本并在任务 6 记录原因 ✅
- **规格链**：specify → clarify → plan（本文件）→ tasks → checklist ✅
- **强制 TDD**：每个任务先写失败测试再实现 ✅
- **指标比对**：完整规格链迭代必出，v2 评测结果追加到 `docs/指标对比/优化指标对比.md` ✅

## 文件结构

### 修改

- `backend/app/config.py` — 新增重排配置，`top_k` 由 `retrieval_candidate_k` + `final_top_k` 取代
- `backend/app/vector_store.py` — `SearchResult.dense` 字段 + `search(with_vectors)` 参数
- `backend/app/routes_chat.py` — 改经 `RetrievalChain` 编排
- `.env.example` — 新增重排配置占位
- `scripts/run_sf6_eval.py` — `--variant` + 延迟采集 + 对比报告
- `README.md` — 补充 v2 重排说明
- `docs/需求说明.md` — 补版本头/版本迭代表（v1、v2），新增重排 FR，把「reranker 精排」从范围外移除
- `docs/指标对比/优化指标对比.md` — 追加 v2 记录
- 测试：`backend/tests/test_config.py`、`test_vector_store.py`、`test_routes_chat.py`、`test_sf6_eval_runner.py`

### 新增

- `backend/app/rerank.py` — 粗排/精排/链路
- `backend/tests/test_rerank.py` — 单测
- `backend/tests/test_rerank_real.py` — 真实 v2-m3 验证
- `docs/版本迭代.md` — 版本记录（v1 缺失需补 + v2）
- `docs/架构/架构图-v2.md` — 六层架构图（含重排链路）

---

## 实现任务

### 任务 1：配置扩展

**文件：**
- 修改：`backend/app/config.py`
- 修改：`.env.example`
- 测试：`backend/tests/test_config.py`

- [ ] **步骤 1：编写失败测试**

在 `backend/tests/test_config.py` 追加：

```python
def test_settings_rerank_defaults():
    settings = AppSettings()

    assert settings.rerank_enabled is True
    assert settings.retrieval_candidate_k == 30
    assert settings.coarse_top_k == 10
    assert settings.final_top_k == 6
    assert settings.coarse_scorer == "dense_cosine"
    assert settings.min_retrieval_score == 0.2
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest backend/tests/test_config.py -v`

预期：FAIL，报错 `AppSettings` 无 `rerank_enabled` 属性。

- [ ] **步骤 3：实现配置**

修改 `backend/app/config.py`，`top_k` 替换为：

```python
rerank_enabled: bool = True
reranker_model_path: Path = Path(r"D:\八维学习\bge-reranker-v2-m3")
reranker_device: str | None = None          # None → 自动检测 CUDA
rerank_batch_size: int = Field(default=8, gt=0)
retrieval_candidate_k: int = Field(default=30, gt=0)
coarse_top_k: int = Field(default=10, gt=0)
final_top_k: int = Field(default=6, gt=0)
coarse_scorer: str = "dense_cosine"         # dense_cosine | rrf_score | hybrid
coarse_hybrid_weight: float = Field(default=0.5, ge=0.0, le=1.0)
```

删除 `top_k` 字段。保留 `min_retrieval_score`。

修改 `.env.example`，把 `TOP_K=6` 替换为：

```env
RERANK_ENABLED=true
RERANKER_MODEL_PATH=D:\八维学习\bge-reranker-v2-m3
RERANK_BATCH_SIZE=8
RETRIEVAL_CANDIDATE_K=30
COARSE_TOP_K=10
FINAL_TOP_K=6
COARSE_SCORER=dense_cosine
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest backend/tests/test_config.py -v`

预期：新增测试通过，原有 3 个测试仍通过。

### 任务 2：SearchResult.dense + search(with_vectors)

**文件：**
- 修改：`backend/app/vector_store.py`
- 测试：`backend/tests/test_vector_store.py`

- [ ] **步骤 1：编写失败测试**

在 `backend/tests/test_vector_store.py` 追加（验证 QdrantVectorStore 在 `with_vectors=True` 时把 dense 解析进结果）：

```python
class ReturningQueryClient(FailingQueryClient):
    def query_points(self, *args, **kwargs):
        assert kwargs.get("with_vector") is True
        point = type(
            "Point",
            (),
            {
                "payload": {
                    "chunk_id": "c1",
                    "document_id": "doc-1",
                    "page": 1,
                    "category": "正文",
                    "text": "第一段",
                    "source_span": "page=1:block=1",
                },
                "score": 0.8,
                "vector": {"dense": [0.1, 0.2, 0.3]},
            },
        )()
        return type("Result", (), {"points": [point]})()


def test_qdrant_search_returns_dense_when_with_vectors(tmp_path, monkeypatch):
    store = QdrantVectorStore(tmp_path / "qdrant", "rag_documents", dense_size=3)
    store.client = ReturningQueryClient()

    results = store.search("问题", DummyEmbedder(), limit=5, with_vectors=True)

    assert results[0].dense == [0.1, 0.2, 0.3]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest backend/tests/test_vector_store.py -v`

预期：FAIL，`search()` 不接受 `with_vectors` 关键字。

- [ ] **步骤 3：实现**

修改 `backend/app/vector_store.py`：

1. `SearchResult` 增加字段：

```python
class SearchResult(BaseModel):
    """检索结果。"""

    chunk: Chunk
    score: float
    dense: list[float] | None = None
```

2. `VectorStore` 协议 `search` 增加 `with_vectors: bool = False` 参数。
3. `InMemoryVectorStore.search` 增加 `with_vectors: bool = False` 参数（dense 保持 None）。
4. `QdrantVectorStore.search` 增加 `with_vectors: bool = False` 参数；`query_points(..., with_vector=with_vectors)`；解析时：

```python
dense = None
if with_vectors and point.vector:
    dense = [float(value) for value in point.vector.get("dense", [])] if isinstance(point.vector, dict) else None
return SearchResult(chunk=Chunk.model_validate(point.payload), score=float(point.score or 0.0), dense=dense)
```

5. `_fallback_search` 返回 `SearchResult(..., dense=None)`（默认值即可）。

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest backend/tests/test_vector_store.py -v`

预期：新增测试通过，原测试通过。

### 任务 3：rerank.py 核心（粗排/精排/链路）

**文件：**
- 新增：`backend/app/rerank.py`
- 测试：`backend/tests/test_rerank.py`

- [ ] **步骤 1：编写失败测试**

新增 `backend/tests/test_rerank.py`：

```python
from pathlib import Path

from backend.app.models import Chunk
from backend.app.rerank import (
    CoarseReranker,
    DenseCosineScorer,
    FineReranker,
    HybridScorer,
    RetrievalChain,
    RrfScoreScorer,
)
from backend.app.vector_store import SearchResult


def make_result(chunk_id: str, text: str, score: float, dense: list[float] | None) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc-1",
            page=1,
            category="正文",
            text=text,
            source_span="page=1:block=1",
        ),
        score=score,
        dense=dense,
    )


def test_dense_cosine_scorer_uses_query_dense():
    scorer = DenseCosineScorer()
    results = [
        make_result("a", "第一段", 0.3, [1.0, 0.0]),
        make_result("b", "第二段", 0.5, [0.0, 1.0]),
    ]

    scores = scorer.score([1.0, 0.0], results)

    assert scores[0] > scores[1]


def test_dense_cosine_scorer_falls_back_to_rrf_when_no_dense():
    scorer = DenseCosineScorer()
    results = [make_result("a", "第一段", 0.42, None)]

    scores = scorer.score(None, results)

    assert scores == [0.42]


def test_hybrid_scorer_weighted():
    scorer = HybridScorer(weight=0.5)
    results = [
        make_result("a", "第一段", 1.0, [1.0, 0.0]),
        make_result("b", "第二段", 0.0, [0.0, 1.0]),
    ]

    scores = scorer.score([1.0, 0.0], results)

    assert scores[0] > scores[1]


def test_coarse_reranker_sorts_and_truncates():
    reranker = CoarseReranker(DenseCosineScorer(), top_k=1)
    results = [
        make_result("a", "第一段", 0.1, [1.0, 0.0]),
        make_result("b", "第二段", 0.9, [0.0, 1.0]),
    ]

    kept = reranker.rerank([1.0, 0.0], results)

    assert [item.chunk.chunk_id for item in kept] == ["a"]


class FakeFlagReranker:
    def __init__(self, *args, **kwargs) -> None:
        self.scores = [[0.9], [0.2], [0.5]]

    def compute_score(self, pairs, normalize: bool = False):
        return [pair_score[0] for pair_score in self.scores[: len(pairs)]]


def test_fine_reranker_scores_and_takes_top_k(monkeypatch):
    import backend.app.rerank as rerank_module

    monkeypatch.setattr(rerank_module, "_load_flag_reranker", lambda path, device: FakeFlagReranker())

    fine = FineReranker(Path("models/bge-reranker-v2-m3"), batch_size=2, top_k=2)
    results = [
        make_result("a", "第一段", 1.0, None),
        make_result("b", "第二段", 1.0, None),
        make_result("c", "第三段", 1.0, None),
    ]

    kept = fine.rerank("问题", results)

    assert [item.chunk.chunk_id for item in kept] == ["a", "c"]
    assert kept[0].score == 0.9


class FakeSettingsForChain:
    rerank_enabled = True
    retrieval_candidate_k = 30
    coarse_top_k = 10
    final_top_k = 6
    min_retrieval_score = 0.2


class FakeChainStore:
    def __init__(self) -> None:
        self.results = [
            make_result("a", "第一段", 0.9, [1.0, 0.0]),
            make_result("b", "第二段", 0.1, [0.0, 1.0]),
        ]

    def search(self, query, embedder, limit, with_vectors=False):
        return list(self.results)


class FakeChainEmbedder:
    def embed_texts(self, texts):
        from backend.app.embeddings import EmbeddingResult

        return [EmbeddingResult(dense=[1.0, 0.0], sparse_indices=[0], sparse_values=[1.0]) for _ in texts]


def test_retrieval_chain_runs_v2_path():
    store = FakeChainStore()
    chain = RetrievalChain(
        store,
        FakeChainEmbedder(),
        FakeSettingsForChain(),
        CoarseReranker(DenseCosineScorer(), top_k=10),
        FineReranker(Path("models/bge-reranker-v2-m3"), batch_size=2, top_k=6),
    )

    results = chain.retrieve("问题")

    assert len(results) == 1  # b 被过滤
    assert results[0].chunk.chunk_id == "a"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest backend/tests/test_rerank.py -v`

预期：FAIL，`No module named 'backend.app.rerank'`。

- [ ] **步骤 3：实现 rerank.py**

写入 `backend/app/rerank.py`（核心实现，见下）：

```python
"""粗排与精排两阶段重排。

- CoarseScorer / CoarseReranker：复用 BGE-M3 稠密信号做廉价粗排。
- FineReranker：封装 bge-reranker-v2-m3 交叉编码器做精排。
- RetrievalChain：编排召回 → 粗排 → 精排 → 阈值过滤。
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from backend.app.config import AppSettings
from backend.app.embeddings import TextEmbedder
from backend.app.vector_store import SearchResult, VectorStore


def _cosine(a: list[float], b: list[float]) -> float:
    """计算两个稠密向量的余弦相似度。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _min_max(values: list[float]) -> list[float]:
    """min-max 归一化到 [0, 1]；全等输入返回全 0。"""
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


class CoarseScorer(Protocol):
    """粗排打分器协议。"""

    def score(self, query_dense: list[float] | None, results: list[SearchResult]) -> list[float]:
        """返回与 results 一一对应的粗排分。"""


class DenseCosineScorer:
    """用 BGE-M3 dense 余弦相似度打分；query 或候选 dense 缺失时回退 RRF 分。"""

    def score(self, query_dense: list[float] | None, results: list[SearchResult]) -> list[float]:
        scores: list[float] = []
        for result in results:
            if query_dense is not None and result.dense is not None:
                scores.append(_cosine(query_dense, result.dense))
            else:
                scores.append(float(result.score))
        return scores


class RrfScoreScorer:
    """直接用 RRF 融合分作为粗排分（零模型成本）。"""

    def score(self, query_dense: list[float] | None, results: list[SearchResult]) -> list[float]:
        return [float(result.score) for result in results]


class HybridScorer:
    """dense 余弦与 RRF 分归一化后加权。"""

    def __init__(self, weight: float = 0.5) -> None:
        self.weight = weight

    def score(self, query_dense: list[float] | None, results: list[SearchResult]) -> list[float]:
        dense_scores = DenseCosineScorer().score(query_dense, results)
        rrf_scores = [float(result.score) for result in results]
        dense_norm = _min_max(dense_scores)
        rrf_norm = _min_max(rrf_scores)
        return [
            self.weight * dense + (1.0 - self.weight) * rrf
            for dense, rrf in zip(dense_norm, rrf_norm, strict=True)
        ]


class CoarseReranker:
    """粗排：打分 + 降序排序 + 截断到 top_k。"""

    def __init__(self, scorer: CoarseScorer, top_k: int) -> None:
        self.scorer = scorer
        self.top_k = top_k

    def rerank(self, query_dense: list[float] | None, results: list[SearchResult]) -> list[SearchResult]:
        scores = self.scorer.score(query_dense, results)
        ranked = sorted(zip(results, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            SearchResult(chunk=result.chunk, score=score, dense=result.dense)
            for result, score in ranked[: self.top_k]
        ]


_FLAGRANKER_CACHE: dict[str, object] = {}


def _load_flag_reranker(model_path: Path, device: str | None) -> object:
    """懒加载并缓存 FlagReranker（chat 与 eval 共享同一实例）。"""
    cache_key = str(model_path.resolve())
    reranker = _FLAGRANKER_CACHE.get(cache_key)
    if reranker is None:
        if not model_path.exists():
            raise FileNotFoundError(f"bge-reranker 模型路径不存在: {model_path}")
        from FlagEmbedding import FlagReranker

        reranker = FlagReranker(str(model_path), use_fp16=True)
        if device == "cpu":
            reranker.model.to("cpu")
        _FLAGRANKER_CACHE[cache_key] = reranker
    return reranker


class FineReranker:
    """精排：bge-reranker-v2-m3 交叉编码器，normalize 打分覆盖 result.score。"""

    def __init__(self, model_path: Path, batch_size: int, top_k: int, device: str | None = None) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.top_k = top_k
        self.device = device

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return []
        model = _load_flag_reranker(self.model_path, self.device)
        pairs = [[query, result.chunk.text] for result in results]
        all_scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch_scores = model.compute_score(pairs[start : start + self.batch_size], normalize=True)
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            all_scores.extend(float(score) for score in batch_scores)
        ranked = sorted(zip(results, all_scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            SearchResult(chunk=result.chunk, score=score, dense=result.dense)
            for result, score in ranked[: self.top_k]
        ]


def build_coarse_scorer(settings: AppSettings) -> CoarseScorer:
    """按配置构造粗排打分器。"""
    if settings.coarse_scorer == "hybrid":
        return HybridScorer(settings.coarse_hybrid_weight)
    if settings.coarse_scorer == "dense_cosine":
        return DenseCosineScorer()
    return RrfScoreScorer()


class RetrievalChain:
    """编排召回 → 粗排 → 精排 → 阈值过滤；rerank_enabled=false 退化为 v1。"""

    def __init__(
        self,
        store: VectorStore,
        embedder: TextEmbedder,
        settings: AppSettings,
        coarse: CoarseReranker,
        fine: FineReranker,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.coarse = coarse
        self.fine = fine

    def retrieve(self, question: str) -> list[SearchResult]:
        if not self.settings.rerank_enabled:
            results = self.store.search(question, self.embedder, limit=self.settings.final_top_k)
            return [result for result in results if result.score >= self.settings.min_retrieval_score]

        query_dense = self.embedder.embed_texts([question])[0].dense
        candidates = self.store.search(
            question,
            self.embedder,
            limit=self.settings.retrieval_candidate_k,
            with_vectors=True,
        )
        if not candidates:
            return []
        coarse_results = self.coarse.rerank(query_dense, candidates)
        if not coarse_results:
            return []
        fine_results = self.fine.rerank(question, coarse_results)
        return [result for result in fine_results if result.score >= self.settings.min_retrieval_score]
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest backend/tests/test_rerank.py -v`

预期：全部通过。

### 任务 4：routes_chat 接入 RetrievalChain

**文件：**
- 修改：`backend/app/routes_chat.py`
- 测试：`backend/tests/test_routes_chat.py`

- [ ] **步骤 1：编写失败测试**

更新 `backend/tests/test_routes_chat.py`：`FakeSettings` 增加 `rerank_enabled = True`、`rerank_batch_size = 8`、`reranker_model_path = Path("models/bge-reranker-v2-m3")`、`reranker_device = None`、`retrieval_candidate_k = 3`、`coarse_top_k = 3`、`final_top_k = 3`，删除 `top_k`。

新增测试（monkeypatch `build_retrieval_chain` 返回假 chain）：

```python
class FakeChain:
    def __init__(self, results) -> None:
        self.results = results

    def retrieve(self, question: str):
        return list(self.results)


def test_chat_route_uses_retrieval_chain(monkeypatch):
    fake_service = FakeQaService("deepseek-r1:7b", "http://localhost:11434")
    chain = FakeChain([SearchResult(chunk=Chunk(...), score=0.42, dense=None)])

    monkeypatch.setattr(routes_chat, "AppSettings", FakeSettings)
    monkeypatch.setattr(routes_chat, "BgeM3Embedder", FakeEmbedder)
    monkeypatch.setattr(routes_chat, "JsonStateStore", FakeStateStore)
    monkeypatch.setattr(routes_chat, "QaService", lambda model, base_url: fake_service)
    monkeypatch.setattr(routes_chat, "build_retrieval_chain", lambda settings, store, embedder: chain)

    response = routes_chat.chat(routes_chat.ChatRequest(question="怎么处理？"))

    assert response.citations[0].score == 0.42
```

（具体 `Chunk(...)` 字段沿用现有测试中的 make_chunk 内容。）

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest backend/tests/test_routes_chat.py -v`

预期：FAIL，`routes_chat` 无 `build_retrieval_chain`。

- [ ] **步骤 3：实现**

修改 `backend/app/routes_chat.py`：

```python
from backend.app.rerank import (
    RetrievalChain,
    build_coarse_scorer,
    CoarseReranker,
    FineReranker,
)


def build_retrieval_chain(settings, store, embedder) -> RetrievalChain:
    """构建检索重排链路。"""
    coarse = CoarseReranker(build_coarse_scorer(settings), settings.coarse_top_k)
    fine = FineReranker(
        settings.reranker_model_path,
        settings.rerank_batch_size,
        settings.final_top_k,
        settings.reranker_device,
    )
    return RetrievalChain(store, embedder, settings, coarse, fine)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """执行问答。"""
    question = request.question.strip()
    if not question:
        return build_fallback_response()

    settings = AppSettings()
    embedder = BgeM3Embedder(settings.bge_m3_model_path)
    store = QdrantVectorStore(settings.qdrant_path, settings.qdrant_collection)
    chain = build_retrieval_chain(settings, store, embedder)
    results = chain.retrieve(question)
    if not results:
        return build_fallback_response()

    document_store = JsonStateStore(Path("data/state.json"))
    document_names = {document.document_id: document.file_name for document in document_store.list_documents()}
    citations = build_citations(results, document_names)
    if not citations:
        return build_fallback_response()

    service = QaService(settings.ollama_model, settings.ollama_base_url)
    return service.answer(question, citations)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest backend/tests/test_routes_chat.py -v`

预期：全部通过。

### 任务 5：eval 脚本 --variant + 延迟 + 对比

**文件：**
- 修改：`scripts/run_sf6_eval.py`
- 测试：`backend/tests/test_sf6_eval_runner.py`

- [ ] **步骤 1：编写失败测试**

在 `backend/tests/test_sf6_eval_runner.py`：
- `install_fake_backend` 中 `FakeAppSettings` 增加重排配置字段，并新增 fake `backend.app.rerank` 模块（`build_coarse_scorer`、`CoarseReranker`、`FineReranker`、`RetrievalChain`），`RetrievalChain.retrieve` 返回固定结果。
- `run_eval_set` 改为接收 evaluator 返回 `(answer, citations, latency_s)`，`summary` 增加 `avg_latency_s`。
- 新增 `test_build_and_run_eval_v2_writes_comparison`：以 `variant="v2"` 调用，断言 `v2_metrics.json` 与 `v1_vs_v2_comparison.json` 存在。

（具体断言按实现后的数据形态微调。）

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest backend/tests/test_sf6_eval_runner.py -v`

预期：FAIL（新参数/新模块缺失）。

- [ ] **步骤 3：实现**

修改 `scripts/run_sf6_eval.py`：

1. `run_eval_set` 的 evaluator 约定改为返回 `(answer, citations, latency_s)`；每个 item 记录 `latency_s`；summary 增加 `avg_latency_s`。
2. `_build_item_evaluator(settings, store, embedder, vector_store, variant)`：v2 走 `RetrievalChain.retrieve`，v1 走 `store.search(limit=final_top_k)` + 过滤；用 `time.perf_counter()` 测检索耗时。
3. `build_and_run_eval(pdf_path, eval_sets_dir, variant="v2")`：输出 `eval/results/{variant}_metrics.json`；当 `variant in ("v2", "both")` 且 v1 基线存在时，产出 `eval/results/v1_vs_v2_comparison.json`（recall/MRR delta；延迟取两版本均值对比）。支持 `variant="both"`：先跑 v1 再跑 v2。
4. `main()` 增加 `--variant`（choices `v1|v2|both`，默认 `v2`），并兼容环境变量 `SF6_EVAL_VARIANT`。

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest backend/tests/test_sf6_eval_runner.py -v`

预期：全部通过。

### 任务 6：真实模型验证

**文件：**
- 新增：`backend/tests/test_rerank_real.py`

- [ ] **步骤 1：确认 FlagEmbedding 兼容性**

运行：`D:\an\envs\mineru\python.exe -c "import FlagEmbedding; from FlagEmbedding import FlagReranker; print(FlagEmbedding.__version__)"`

预期：可导入。若报错或 `FlagReranker` 不存在，升级 `requirements.txt` 中 `FlagEmbedding` 到可加载 v2-m3 的锁定版本并记录原因（宪法要求精确锁定）。

- [ ] **步骤 2：编写真实模型测试**

新增 `backend/tests/test_rerank_real.py`（仿 `test_embeddings_real.py`）：

```python
from pathlib import Path

import pytest

from backend.app.config import AppSettings
from backend.app.rerank import FineReranker


@pytest.mark.real
def test_fine_reranker_scores_real_model():
    settings = AppSettings()
    if not settings.reranker_model_path.exists():
        pytest.skip("v2-m3 模型不存在，跳过真实模型测试")

    fine = FineReranker(settings.reranker_model_path, batch_size=8, top_k=1)
    from backend.app.vector_store import SearchResult
    from backend.app.models import Chunk

    result = SearchResult(
        chunk=Chunk(
            chunk_id="c1",
            document_id="doc-1",
            page=7,
            category="正文",
            text="本标准规定了电气设备中六氟化硫（SF6）气体的现场检测、回收、净化、回充全流程循环再利用方法。",
            source_span="page=7:block=1",
        ),
        score=0.0,
        dense=None,
    )

    kept = fine.rerank("本标准适用于什么场景？", [result])

    assert len(kept) == 1
    assert 0.0 <= kept[0].score <= 1.0
```

- [ ] **步骤 3：运行真实模型测试**

运行：`D:\an\envs\mineru\python.exe -m pytest backend/tests/test_rerank_real.py -v`

预期：通过（模型已下载）。同时记录 v2-m3 在 GPU 上的单次批打分延迟，供评测参考。

### 任务 7：全量验证

- [ ] **步骤 1：运行全部单元测试**

运行：`D:\an\envs\mineru\python.exe -m pytest backend/tests -v`

预期：全部通过（含真实模型测试）。

- [ ] **步骤 2：API 导入检查**

运行：`D:\an\envs\mineru\python.exe -c "from backend.app.main import app; print(app.title)"`

预期：输出 `RAG PDF 问答系统`。

- [ ] **步骤 3：规格覆盖核对**

对照 `docs/v2/spec.md` 的 FR-001~FR-009、NFR、边界条件逐一勾选，确认实现覆盖。

### 任务 8：宪法交付物

**文件：**
- 修改：`docs/需求说明.md`
- 新增：`docs/版本迭代.md`
- 新增：`docs/架构/架构图-v2.md`
- 修改：`README.md`

- [ ] **步骤 1：需求文档升级 v2**

在 `docs/需求说明.md` 顶部补版本头：

```markdown
**Version**: v2 | **日期**: 2026-09-01
```

补「版本迭代表」（含 v1 基线行与 v2 行），并在 FR 表中追加：

| FR15 | 粗排 | 高 | RRF 召回后先用 BGE-M3 dense 余弦信号粗排，从候选集中保留 coarse_top_k。 |
| FR16 | 精排 | 高 | 用本地 bge-reranker-v2-m3 交叉编码器对粗排结果精排，取 final_top_k 作为引用。 |
| FR17 | 重排可配置 | 高 | 候选规模、各层 top_k、阈值、粗排打分方式、模型路径、批大小均可通过配置调整。 |
| FR18 | 检索重排评测 | 中 | 评测脚本支持 v1/v2 两档对比，产出 recall/MRR/延迟。 |

在「范围外」列表中移除「reranker 精排」。

- [ ] **步骤 2：新建版本迭代记录**

新增 `docs/版本迭代.md`，记录 v1（基线）与 v2（重排）两行，含日期/类型/改动摘要/影响范围。

- [ ] **步骤 3：新建架构图-v2**

新增 `docs/架构/架构图-v2.md`，按宪法六层（呈现层 → 接口层 → 服务层 → 管线层 → 数据层 → 外部）绘制 Mermaid 图，在服务层展示「RRF 召回 → 粗排 → 精排 → 阈值过滤」链路，外部层含 Ollama 与本地模型路径。

- [ ] **步骤 4：更新 README**

在 README 检索部分补充 v2 重排链路与配置项说明。

### 任务 9：评测 + 指标对比 + 汇报

- [ ] **步骤 1：运行 v1/v2 对比评测**

运行（PDF 路径可用环境变量 `SF6_EVAL_PDF_PATH` 覆盖）：

```powershell
$env:SF6_EVAL_PDF_PATH="C:\Users\tirito\Downloads\GB_T_44653-2024_六氟化硫气体现场循环再利用导则.pdf"
D:\an\envs\mineru\python.exe scripts/run_sf6_eval.py --variant both
```

预期：产出 `eval/results/v1_metrics.json`、`eval/results/v2_metrics.json`、`eval/results/v1_vs_v2_comparison.json`。

- [ ] **步骤 2：记录指标对比**

在 `docs/指标对比/优化指标对比.md` 追加 v2 记录：recall/MRR/延迟对比 v1 基线，注明改动内容（评测集未变；safety 集 5 条低于宪法 ≥10 下限，本轮沿用保可比，下轮扩充）。

- [ ] **步骤 3：向用户汇报**

汇报：本次改了什么（改动清单：涉及文件、影响范围）、v2 相对 v1 的指标变化、下一步迭代建议。

---

## 自检结果

### 规格覆盖度

- FR-001/FR-002（粗排/精排链路）：任务 3、任务 4
- FR-003（粗排可配置 scorer）：任务 3
- FR-004（精排分 0~1 与阈值过滤）：任务 3、任务 4
- FR-005（v1 退化开关）：任务 3、任务 4
- FR-006（配置项）：任务 1
- FR-007（评测 --variant/延迟/对比）：任务 5
- FR-008（本地 bge-reranker、禁止 Ollama）：任务 3、任务 6
- FR-009（失败不静默降级）：任务 3
- NFR-001（延迟纳入对比）：任务 5
- NFR-002（依赖锁定）：任务 6
- SC-001~SC-005：任务 7、任务 9
- 宪法交付物：任务 8、任务 9

### 占位符扫描

- 计划无未完成章节；无待定参数（默认值见任务 1）。

### 类型一致性

- `SearchResult.dense` 可选，粗排 fallback 路径已覆盖。
- `RetrievalChain.retrieve` 统一 v1/v2 出口，`routes_chat` 与 eval 共用。
- `run_eval_set` evaluator 约定升级为三返回值，相关测试同步更新。
