# -*- coding: utf-8 -*-
"""离线级：索引与 BM25 确定性断言（本地可跑，不需要嵌入服务）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

覆盖：
    * ``index_manifest.json`` 与产物一致（页数 548/350、总块数、维度 1024、模型 bge-m3:latest）；
    * ``chunk`` 元数据不变量（唯一 id、页码 1-based 且在范围内、正文非空、类型合法）；
    * 退化表不进索引（与 §3.2 互为佐证）；
    * BM25：规模一致 + **可复现**（同题两次结果完全相同）+ 14 题均能召回候选；
    * 逐题记录「最佳命中块在 BM25 中的名次」到留痕（诊断用，**不设自造门槛**）。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from common import assertions, paths

pytestmark = [pytest.mark.offline, pytest.mark.linkage]


def test_manifest_matches_artifacts(index_manifest: dict[str, Any], chunks_all: list[dict[str, Any]],
                                    page_counts: dict[str, int], discovered_pdfs: list[Any]) -> None:
    """manifest 的页数/总块数/维度/模型与真实产物一致。"""
    manifest_files = {row["file_name"]: row for row in index_manifest.get("files", [])}
    assert set(manifest_files) == set(page_counts), \
        f"manifest 文件集合与自动发现不一致：{sorted(manifest_files)} vs {sorted(page_counts)}"
    for name, row in manifest_files.items():
        assert int(row["page_count"]) == int(page_counts[name]), f"{name} 页数不一致"
    assert int(index_manifest["total_chunks"]) == len(chunks_all), \
        f"总块数不一致：manifest {index_manifest['total_chunks']} vs chunks.jsonl {len(chunks_all)}"
    assert int(index_manifest["embedding"]["dim"]) == 1024
    assert index_manifest["embedding"]["model"] == "bge-m3:latest"
    assert int(index_manifest["bm25"]["count"]) == len(chunks_all)


def test_chunk_metadata_invariants(chunks_all: list[dict[str, Any]], page_counts: dict[str, int]) -> None:
    """块元数据不变量：id 唯一、页码 1-based 合法、正文非空、类型与 table_id 一致。"""
    ids = [str(c["chunk_id"]) for c in chunks_all]
    assert len(set(ids)) == len(ids), "chunk_id 出现重复"

    problems: list[str] = []
    for chunk in chunks_all:
        name, page = str(chunk["file_name"]), chunk["page"]
        if name not in page_counts:
            problems.append(f"{chunk['chunk_id']} 文件名未知：{name}")
            continue
        if not isinstance(page, int) or not (1 <= page <= page_counts[name]):
            problems.append(f"{chunk['chunk_id']} 页码非法：{page}（应为 1..{page_counts[name]}）")
        if not str(chunk.get("content") or "").strip():
            problems.append(f"{chunk['chunk_id']} 正文为空")
        if str(chunk.get("type")) not in ("text", "table"):
            problems.append(f"{chunk['chunk_id']} 类型非法：{chunk.get('type')!r}")
        if str(chunk.get("type")) == "table" and not str(chunk.get("table_id") or "").strip():
            problems.append(f"{chunk['chunk_id']} 表格块缺 table_id")
        if not isinstance(chunk.get("keywords"), list):
            problems.append(f"{chunk['chunk_id']} keywords 不是列表")
    assert not problems, "\n".join(problems[:20])


def test_degenerate_tables_not_indexed(chunks_all: list[dict[str, Any]],
                                       tables_pdf1: list[dict[str, Any]],
                                       tables_pdf2: list[dict[str, Any]]) -> None:
    """§3.2：退化表 ``table_id`` 在 ``chunks.jsonl`` 命中 0 个。"""
    degenerate = {str(b["table_id"]) for b in list(tables_pdf1) + list(tables_pdf2)
                  if bool(b.get("degenerate"))}
    indexed = {str(c.get("table_id")) for c in chunks_all if c.get("table_id")}
    assert not (degenerate & indexed), f"退化表进入索引：{sorted(degenerate & indexed)}"


def test_bm25_size_and_manifest_consistent(bm25_index: Any, index_manifest: dict[str, Any],
                                           discovered_pdfs: list[Any]) -> None:
    """BM25 索引规模与 manifest 一致，且元数据只覆盖自动发现的语料。"""
    assert bm25_index.size() == int(index_manifest["bm25"]["count"])
    names = {str(row["file_name"]) for row in bm25_index.meta.values()}
    assert names, "BM25 元数据为空"
    assert names <= {p.name for p in discovered_pdfs}, f"BM25 含未知语料：{sorted(names)}"


def test_bm25_reproducible_for_all_questions(bm25_index: Any, golden_items: list[Any]) -> None:
    """BM25 结果可复现（同题两次完全一致）且 14 题都能召回候选。"""
    failures: list[str] = []
    reruns: list[str] = []
    for item in golden_items:
        first = [cid for cid, _ in bm25_index.search(item.question, top_k=20)]
        second = [cid for cid, _ in bm25_index.search(item.question, top_k=20)]
        if not first:
            failures.append(f"题 {item.id} BM25 无候选")
        if first != second:
            reruns.append(f"题 {item.id} 两次结果不一致")
    assert not failures, "\n".join(failures)
    assert not reruns, "\n".join(reruns)


def test_bm25_rank_diagnostic_written(golden_items: list[Any], chunks_all: list[dict[str, Any]],
                                      bm25_index: Any) -> None:
    """把「最佳命中块在 BM25 中的名次」写入留痕（诊断用；不在本层设门槛）。

    说明：命中判据一律走 ``assertions.evidence_hit``（``evidence_contains``），
    题 207 优先 ``evidence_verbatim``；此处只做**名次记录**，供 T9 优化前后对比与
    t12 修正（题 207 召回、题 34 排序）定位使用。
    """
    content_map = {str(c["chunk_id"]): c for c in chunks_all}
    rows: list[dict[str, Any]] = []
    for item in golden_items:
        ranked = bm25_index.search(item.question, top_k=200)
        best = None
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            chunk = content_map.get(chunk_id)
            if chunk is None:
                continue
            outcome = assertions.evidence_hit([chunk], evidence=item.evidence,
                                              verbatim=item.evidence_verbatim)
            if outcome.hit:
                best = {"rank": rank, "chunk_id": chunk_id, "score": round(float(score), 6)}
                break
        rows.append({
            "id": item.id, "corpus": item.corpus, "question": item.question,
            "bm25_candidates": len(ranked), "best_hit": best,
            "hit_in_top20": bool(best and best["rank"] <= 20),
            "hit_in_top5": bool(best and best["rank"] <= 5),
        })

    paths.ensure_trace_dir()
    target = paths.TRACE_DIR / "bm25_rank_diagnostic.json"
    target.write_text(json.dumps({
        "work_order": assertions.WORK_ORDER,
        "note": ("BM25 单路名次诊断（非门槛）。命中判据 = evidence_contains；"
                 "题 207 用 evidence_verbatim。最终检索门槛在 在线 层的 top-5 混合检索用例。"),
        "ragas": assertions.RAGAS_BANNER,
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    assert len(rows) == len(golden_items)
    assert all(row["bm25_candidates"] > 0 for row in rows)
