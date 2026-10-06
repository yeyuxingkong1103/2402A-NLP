# v2 检索重排设计：粗排 + 精排

> **范围**：在 v1 RAG 项目代码上直接修改（覆盖原代码），在 `RRF 融合召回` 之后新增「粗排 + 精排」两阶段重排链路。MinerU 空结果修复不在本轮范围。

## 1. 目标

1. 在 v1 检索链路（`BGE-M3 dense+sparse → Qdrant RRF → 过滤 → top_k → 引用`）基础上，新增可迭代的粗排与精排，提升最终引用质量。
2. 粗排复用 BGE-M3 稠密信号（不引入第二个模型），精排使用本地 `bge-reranker-v2-m3` 交叉编码器。
3. 粗排、精排设计为独立可替换模块，关键规模与阈值全部进入配置，支持后续按 recall/MRR/延迟迭代优化。
4. 直接修改 v1 代码，不另开分支；`rerank_enabled` 开关保留 v1 行为用于 A/B 对比。

## 2. 现状

v1 检索链路：`BgeM3Embedder → QdrantVectorStore.search(RRF, limit=top_k=6) → score>=min_retrieval_score(0.2) → build_citations → QaService(Ollama deepseek-r1:7b)`。

评测基线（`eval/baseline/v1_metrics.json`）：general recall=1.0 / mrr=0.6583，另有 safety、retrieval 两套。

问题：RRF 之后没有额外重排层，最终引用直接受 `top_k` 限制，召回空间偏小；低质结果可能占用引用名额。

## 3. 设计原则

- **召回优先**：先扩大候选（30），再做重排，避免相关内容过早被截断。
- **分层清晰**：召回、粗排、精排职责单一，各自可替换、可测试。
- **可配置**：候选规模、各层 top_k、阈值、模型路径、scorer 类型全部进 `AppSettings`。
- **可评测**：同一评测集可跑 v1/v2 两档，对比 recall/MRR/延迟。
- **直接迭代**：在 v1 代码上原地修改，不保留并行分支。

## 4. 数据流

```text
question
  → Qdrant dense+sparse RRF 召回（retrieval_candidate_k=30，带 dense 向量返回）
  → 粗排 CoarseReranker（BGE-M3 dense 余弦，取 coarse_top_k=10）
  → 精排 FineReranker（bge-reranker-v2-m3 交叉编码器 normalize 0~1，取 final_top_k=6）
  → 按 min_retrieval_score 过滤
  → build_citations → QaService
```

`rerank_enabled=false` 时绕过粗排/精排，退化为 v1：`search(limit=final_top_k=6) → 过滤 → 引用`，用于 A/B 基线对比。

## 5. 模块划分

### 5.1 新增 `backend/app/rerank.py`

| 组件 | 职责 |
|---|---|
| `CoarseScorer(Protocol)` | 输入 query 与候选，输出每个候选的粗排分。实现 `DenseCosineScorer`（复用 BGE-M3 dense 余弦）、`RrfScoreScorer`（直接用 RRF 融合分）、`HybridScorer`（dense 余弦与 RRF 分加权） |
| `CoarseReranker` | 调用 `CoarseScorer` 给候选打分，按分排序取 top `coarse_top_k` |
| `FineReranker` | 封装 `FlagReranker(model_path, use_fp16=True)`，`compute_score([[query, chunk.text]...], normalize=True)` 按 `rerank_batch_size` 批处理打分，取 top `final_top_k`，**用精排分数覆盖 `result.score`** |
| `RetrievalChain` | 编排「召回 → 粗排 → 精排 → 阈值过滤」，暴露 `retrieve(question) -> list[SearchResult]`，chat 路由与 eval 共用 |

### 5.2 修改 `backend/app/vector_store.py`

- `SearchResult` 增加可选字段 `dense: list[float] | None = None`，供粗排使用。
- `QdrantVectorStore.search(query, embedder, limit, with_vectors=False)`：`with_vectors=True` 时 `query_points(with_vector=True)`，从返回的 named vector 中提取 dense 填充到结果。
- `_fallback_search`（MemoryError 兜底）与 `InMemoryVectorStore.search` 的 dense 均为 `None`。

### 5.3 修改 `backend/app/routes_chat.py`

- `chat()` 改为：`settings.rerank_enabled` 为真时走 `RetrievalChain.retrieve()`；为假时走 v1 路径（`search(limit=final_top_k)` → 过滤）。
- 任一环节结果为空的处理沿用现有 `build_fallback_response()`。

### 5.4 模型加载

- `FineReranker` 懒加载，模块级缓存（仿 `_QDRANT_CLIENT_CACHE`，按模型路径做 key），chat 与 eval 共享同一实例。
- `reranker_device=None` 时自动检测 CUDA；GPU 可用时 `use_fp16=True`。

## 6. 配置（`AppSettings`）

新增字段，`top_k` 由 `retrieval_candidate_k` + `final_top_k` 取代（v1 模式下搜索 limit 用 `final_top_k`，行为与 v1 完全一致）：

```python
rerank_enabled: bool = True
reranker_model_path: Path = Path(r"D:\八维学习\bge-reranker-v2-m3")
reranker_device: str | None = None        # None → 自动检测 CUDA
rerank_batch_size: int = 8
retrieval_candidate_k: int = 30
coarse_top_k: int = 10
final_top_k: int = 6
coarse_scorer: str = "dense_cosine"       # dense_cosine | rrf_score | hybrid
coarse_hybrid_weight: float = 0.5
```

保留 `min_retrieval_score`（默认 0.2）：v2 下对精排分数（0~1 sigmoid）兜底过滤，v1 下对 RRF 分数过滤。

## 7. 错误处理

- **精排模型缺失/加载失败/打分报错**：抛可诊断错误，不允许静默降级成错误排序的引用（与 `2026-09-01-retrieval-rerank-mineru-design.md` 一致）。
- **候选为空 / 粗排后为空 / 过滤后为空**：走现有 fallback answer。
- **v2-m3 与 FlagEmbedding 1.3.3 不兼容**：实现时验证；若报错则升级 `requirements.txt` 中 FlagEmbedding 版本并记录。

## 8. 评测改造

- `scripts/run_sf6_eval.py` 增加 `--variant v1|v2`（环境变量 `SF6_EVAL_VARIANT`，默认 v2）。v1 走原路径，v2 走 `RetrievalChain.retrieve()`。
- 每个 item 记录检索耗时 `latency_s`（召回 + 重排，不含 Ollama 回答），summary 增加平均延迟。
- 产出：
  - `eval/results/v2_metrics.json`（v2 各套指标 + 延迟）
  - `eval/results/v1_vs_v2_comparison.json`（recall/MRR/延迟 delta，用于迭代对比）

## 9. 测试方案

- **`backend/tests/test_rerank.py`（新增）**：
  - `CoarseReranker`：dense 余弦排序与截断；三种 scorer 行为；dense 为 None 时 fallback 到 `rrf_score`。
  - `FineReranker`：monkeypatch `FlagReranker`，验证批处理、排序、top_k、分数覆盖。
  - `RetrievalChain`：完整编排；空候选 → 空；`rerank_enabled=false` → v1 路径。
- **`backend/tests/test_routes_chat.py`（更新）**：FakeSettings 换成新配置字段；monkeypatch 粗排/精排与开关。
- **`backend/tests/test_rerank_real.py`（新增，仿 `test_embeddings_real.py`）**：模型已下载完成，真实加载 v2-m3 对一条 question-passage 打分，断言分数落在合理范围。
- **`backend/tests/test_sf6_eval_runner.py`（更新）**：覆盖 `--variant` 选择。

## 10. 非目标

- 不在本轮修复 MinerU 空结果（另开一轮）。
- 不在本轮改变 Qdrant 存储结构、替换问答模型或改写 prompt 体系。
- 不在本轮引入第二个粗排模型。

## 11. 验收标准

- `/api/chat` 走粗排 + 精排链路；`rerank_enabled=false` 可复现 v1 行为。
- 粗排/精排规模、scorer、阈值、模型路径均可通过 `AppSettings` 调整。
- 同一评测集可分别输出 v1/v2 指标与对比报告（recall/MRR/延迟）。
- `test_rerank_real.py` 在真实 v2-m3 上通过；其余测试全部通过。
