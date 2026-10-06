# -*- coding: utf-8 -*-
"""T4 离线回归：嵌入 / 向量库 / 自实现 BM25 / 索引产物一致性。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

断言范围（对应设计 §3.8~§3.10、§5.4、§6.1 与验收标准 §1）：
    1. 索引目录存在且自洽：`vectors.npy` 行数 == `ids.json` 的 ids 数 == 块数；
    2. 维度隔离：`expected_dim` 不匹配必须抛 `IndexDimensionError`（禁止静默截断）；
    3. numpy 精确检索 vs faiss 结果一致（容差 1e-5）；faiss 不可用时跳过并如实标注；
    4. BM25 自实现：倒排优化前后打分**完全一致**（对照暴力实现），且 `save`/`load` 往返一致；
    5. `allowed_ids` 硬过滤生效（返回的 chunk 必须全部来自所选文件）；
    6. `index_manifest.json` 的块数与 SQLite/JSONL 实际件数一致（不得编造）；
    7. 诊断（非门槛）：golden 题 33/260 的正文证据块已入索引且 BM25 可召回。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/离线/test_index.py
退出码：0 全通过；1 存在断言失败。
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "研发"))

import numpy as np  # noqa: E402

from app.core import chunker, embedder, text_utils, vector_store  # noqa: E402
from app.core.bm25_index import BM25Index, PureBM25  # noqa: E402
from app.core.config import get_config  # noqa: E402
from app.core.errors import IndexDimensionError  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    """记录并打印一条断言结果。"""
    RESULTS.append((bool(ok), name, detail))
    print(f"  {'✅' if ok else '❌'} {name}" + (f" —— {detail}" if detail else ""))


def brute_scores(bm: PureBM25, query_tokens: list[str]) -> list[float]:
    """优化前的暴力实现（对照用，仅测试内部）。"""
    scores = [0.0] * bm.doc_count
    weighted = Counter(t for t in query_tokens if t)
    for term, qtf in weighted.items():
        idf = bm.idf.get(term)
        if idf is None:
            continue
        for row, tf_map in enumerate(bm.term_freqs):
            tf = tf_map.get(term)
            if not tf:
                continue
            denom = tf + bm.k1 * (1.0 - bm.b + bm.b * bm.doc_lengths[row] / bm.avgdl)
            scores[row] += idf * (tf * (bm.k1 + 1.0) / denom) * float(qtf)
    return scores


def main() -> int:
    """执行 T4 离线断言。"""
    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("test_index")
    index_dir = cfg.paths.index_dir / "bge-m3_1024"
    chunks_path = cfg.paths.processed_dir / "chunks.jsonl"
    db_path = cfg.paths.index_dir / "rag.sqlite3"

    print(f"\n索引目录：{index_dir}")
    if not (index_dir / "vectors.npy").is_file():
        print("❌ 索引不存在，请先跑：pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py --force")
        return 1

    chunks = [chunker.Chunk.from_dict(r) for r in text_utils.read_jsonl(chunks_path)]
    chunks.sort(key=lambda c: c.order)
    print(f"分块文件：{chunks_path} → {len(chunks)} 块")

    print("\n=== [1] 索引自洽性与件数一致 ===")
    meta = json.loads((index_dir / "ids.json").read_text(encoding="utf-8"))
    vectors = np.load(index_dir / "vectors.npy")
    check(vectors.shape[0] == len(meta["ids"]) == len(chunks),
          "vectors 行数 == ids 数 == 分块数",
          f"{vectors.shape[0]} / {len(meta['ids'])} / {len(chunks)}")
    check(vectors.shape[1] == 1024 and meta["dim"] == 1024, "向量维度 1024", f"{vectors.shape}")
    check(vectors.dtype == np.float32, "向量 dtype 为 float32", str(vectors.dtype))
    norms = np.linalg.norm(vectors, axis=1)
    check(float(np.min(norms)) > 0.99 and float(np.max(norms)) < 1.01,
          "向量已 L2 归一化（模长≈1）", f"[{norms.min():.4f}, {norms.max():.4f}]")
    conn = sqlite3.connect(str(db_path))
    db_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    db_ids = {r[0] for r in conn.execute("SELECT chunk_id FROM chunks")}
    conn.close()
    check(db_chunks == len(chunks), "SQLite chunks 行数 == 分块数", f"{db_chunks} vs {len(chunks)}")
    check(set(meta["ids"]) == db_ids, "索引 ids 与 SQLite chunk_id 完全一致")

    print("\n=== [2] 维度隔离（禁止静默截断）===")
    try:
        vector_store.load_vector_index(index_dir, expected_dim=768)
        check(False, "expected_dim 不匹配应抛 IndexDimensionError")
    except IndexDimensionError as exc:
        check(True, "expected_dim 不匹配抛 IndexDimensionError", str(exc)[:52])

    print("\n=== [3] numpy vs faiss 一致性（容差 1e-5）===")
    index = vector_store.load_vector_index(index_dir, expected_dim=1024)
    query = embedder.embed_query("武汉兴图新科电子来自军用领域的收入占比是多少？", cfg=cfg)
    numpy_hits = vector_store.search_vectors(index, query, 10)
    faiss_index = vector_store.try_build_faiss(index.vectors, enabled=True)
    if faiss_index is None:
        check(True, "faiss 不可用 → 已显式降级为纯 numpy（如实标注，不算失败）",
              "try_build_faiss 返回 None 且已写 vector.faiss_skipped 日志")
    else:
        scores, rows = faiss_index.search(np.asarray(query, dtype=np.float32).reshape(1, -1), 10)
        faiss_map = {index.ids[int(r)]: float(s) for s, r in zip(scores[0], rows[0])}
        worst = 0.0
        for cid, score in numpy_hits:
            worst = max(worst, abs(score - faiss_map.get(cid, -1.0)))
        check(worst <= 1e-5, "faiss 与 numpy 的 top-10 分数一致（≤1e-5）", f"最大偏差 {worst:.2e}")
        check(set(faiss_map) == {cid for cid, _ in numpy_hits}, "faiss 与 numpy 的 top-10 集合一致")

    print("\n=== [4] BM25 自实现：等价性与往返 ===")
    bm25 = BM25Index.load(index_dir)
    check(bm25.size() == len(chunks), "BM25 块数 == 分块数", f"{bm25.size()}")
    check(bm25.vocab_size() > 10000, "BM25 词表规模合理（>10000）", str(bm25.vocab_size()))
    question = "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"
    qt = text_utils.tokenize(question)
    fast = bm25.bm25.get_scores(qt)
    slow = brute_scores(bm25.bm25, qt)
    check(all(abs(a - b) < 1e-12 for a, b in zip(fast, slow)),
          "倒排打分与暴力实现完全一致（优化未改语义）")
    t0 = time.perf_counter()
    hits = bm25.search(question, 20)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    check(bool(hits), "BM25 检索有结果", f"top1={hits[0][0] if hits else '-'}，{elapsed_ms:.1f} ms（含分词）")
    check(elapsed_ms <= 300.0, "BM25 单次检索 ≤ 300 ms（预算内）", f"{elapsed_ms:.1f} ms")
    check(not any("rank_bm25" in str(m) for m in sys.modules), "未 import rank_bm25（自实现）")

    print("\n=== [5] allowed_ids 硬过滤 ===")
    file_names = sorted({c.file_name for c in chunks})
    check(len(file_names) == 2, "索引含两份 PDF 的块", str(file_names))
    allowed = {c.chunk_id for c in chunks if c.file_name == file_names[0]}
    filtered = vector_store.search_vectors(index, query, 10, allowed_ids=allowed)
    check(bool(filtered) and all(cid in allowed for cid, _ in filtered),
          "向量检索按 allowed_ids 过滤（结果全属所选文件）", f"{len(filtered)} 条")
    bm_filtered = bm25.search(question, 10, allowed_ids=allowed)
    check(bool(bm_filtered) and all(cid in allowed for cid, _ in bm_filtered),
          "BM25 检索按 allowed_ids 过滤", f"{len(bm_filtered)} 条")
    empty_filter = vector_store.search_vectors(index, query, 10, allowed_ids={"不存在"})
    check(empty_filter == [], "过滤后为空 → 返回空列表（不报错、不返回未过滤结果）")

    print("\n=== [6] manifest 与实测件数一致（不得编造）===")
    manifest = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))
    check(manifest["total_chunks"] == len(chunks), "manifest.total_chunks == 分块数",
          f"{manifest['total_chunks']} vs {len(chunks)}")
    check(manifest["embedding"]["count"] == vectors.shape[0],
          "manifest.embedding.count == 向量行数", f"{manifest['embedding']['count']}")
    check(manifest["bm25"]["count"] == bm25.size(), "manifest.bm25.count == BM25 块数")
    check(manifest["bm25"]["vocab_size"] == bm25.vocab_size(), "manifest BM25 词表规模一致")
    per_file_chunks = {f["file_name"]: f["text_chunks"] + f["table_chunks"] for f in manifest["files"]}
    real_per_file = {name: sum(1 for c in chunks if c.file_name == name) for name in file_names}
    check(per_file_chunks == real_per_file, "manifest 逐文件块数 == 实际分块数",
          f"{per_file_chunks} vs {real_per_file}")
    check(manifest["deps"]["rank_bm25"] == "unavailable", "manifest 如实标注 rank_bm25 不可用")
    check(manifest["files"][0]["page_count"] in (548, 350), "manifest 页数来自实测",
          str([f["page_count"] for f in manifest["files"]]))

    print("\n=== [7] 诊断（非门槛）：golden 题 33/260 证据可达性 ===")
    evidence = "报告期内，公司来自军用领域的收入分别为6,464.51 万元、14,414.16 万元、18,780.67 万元和4,627.14 万元"
    holders = [c for c in chunks if text_utils.evidence_contains(c.content, evidence)]
    check(bool(holders), "含 golden 证据的块已进入索引",
          str([c.chunk_id for c in holders]))
    ranks = []
    short_hits = [cid for cid, _ in bm25.search("来自军用领域的收入", 20)]
    for holder in holders:
        ranks.append((holder.chunk_id, short_hits.index(holder.chunk_id) + 1 if holder.chunk_id in short_hits else None))
    print(f"     BM25 top-20 排名：{ranks}（供 T5 融合/重排调试，不作为验收门槛）")

    failed = [name for ok, name, _ in RESULTS if not ok]
    print("\n" + "─" * 68)
    print(f"T4 断言：✅{len(RESULTS) - len(failed)} / {len(RESULTS)}"
          + (f"，失败：{failed}" if failed else ""))
    shutdown_logging()
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 —— 顶层兜底：打印完整堆栈并以 1 退出
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
