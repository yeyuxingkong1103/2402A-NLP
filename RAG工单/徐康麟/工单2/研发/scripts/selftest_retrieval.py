"""检索自测：10 个工单问题的证据排名（重排前 vs 重排后）与最终结果命中数。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 自测脚本（T2 初步自测；**正式判分由 T5/T9 负责**）

口径说明（重要，避免歧义）：
1. **证据块现场求得**：不硬编码 chunk_id（chunk_id 跨索引不稳定），一律用 golden
   ``evidence`` 原文（空白归一）在新索引里**严格匹配**定位；匹配到多块时构成"证据集合"，
   排名取**证据集合中最靠前者**（这就是"证据是否进了最终结果"的判据）。
2. ``evidence`` 含「……」或被合成（Q95/Q207）时退回**定位串 + 允许页码**规则
   （Q95 用「全军第一个视频指挥系统技术标准」；Q207 用「补充流动资金 + 15,000.00」且页 ∈ {479,490}）。
3. **未命中用 ``null``**（``pre_rerank_rank``/``post_rerank_rank`` 为 ``null``），不使用 0，
   避免"未命中"与"第 0 名"混淆。
4. 检索链路与线上完全一致：查询变体（原句/去主体/关键词/同义）+ 块级/子块级向量 + BM25
   + 相对化加权 + 规则重排；``pre`` 用同一候选集但**关闭重排**（reranker=None）。

用法::

    pwsh -NoProfile -File run_py.ps1 研发/scripts/selftest_retrieval.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from app.core.chunker import Chunker  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.embedder import get_embedder  # noqa: E402
from app.core.logging_conf import flush_logs, logger, setup_logging  # noqa: E402
from app.core.query_understanding import get_query_understanding  # noqa: E402
from app.core.reranker import get_reranker  # noqa: E402
from app.core.retriever import Retriever  # noqa: E402
from app.core.vector_store import VectorStore  # noqa: E402
from app.models.schemas import Chunk, GoldenQA  # noqa: E402

#: 替代定位规则（应对 evidence 含「……」/为合成文本的题目）：定位串 + 允许页码
FALLBACK_LOCATORS: dict[int, dict[str, object]] = {
    95: {"needles": ["全军第一个视频指挥系统技术标准"], "pages": [160]},
    957: {"needles": ["视频指挥领域的重要供应商"], "pages": [26, 154, 332]},
    795: {"needles": ["荣获国家科技进步一等奖"], "pages": [27, 155, 181, 182, 241]},
    207: {"needles": ["补充流动资金", "15,000.00"], "pages": [479, 490]},
}


def squeeze(text: str) -> str:
    """去掉所有空白，便于跨排版差异匹配。"""
    return "".join((text or "").split())


def load_golden(path: Path) -> list[GoldenQA]:
    """读取金标准问答。"""
    items: list[GoldenQA] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(GoldenQA(**json.loads(line)))
    return items


def locate_evidence(chunks: list[Chunk], item: GoldenQA) -> list[Chunk]:
    """定位证据块集合（严格整段匹配优先；含「……」/合成串/标点差异时用替代规则收敛。

    说明：``evidence_pages`` 存在错标（环境事实 §5.1），因此只用它做**收敛偏好**，
    不把它当作"命中"的硬条件；替代规则来自 tester/captain 对新索引的独立勘探。
    """
    full = squeeze(item.evidence or "")
    rule = FALLBACK_LOCATORS.get(item.id) or {}
    needles = [squeeze(str(value)) for value in (rule.get("needles") or [])]
    pages = set(rule.get("pages") or [])
    matches: list[Chunk] = []
    if len(full) >= 20:
        matches = [chunk for chunk in chunks if full in squeeze(chunk.content)]
    if not matches:
        segments = [squeeze(seg) for seg in (item.evidence or "").split("……") if len(squeeze(seg)) >= 8]
        if segments:
            matches = [chunk for chunk in chunks if all(seg in squeeze(chunk.content) for seg in segments)]
    if not matches and needles:
        matches = [chunk for chunk in chunks if all(needle in squeeze(chunk.content) for needle in needles)]
    if not matches:
        return []
    if pages:
        narrowed = [chunk for chunk in matches if chunk.page in pages]
        if narrowed:
            matches = narrowed
    return matches


def best_rank(hits: list, evidence_ids: set[str]) -> int | None:
    """返回证据集合在结果列表中最靠前的名次（1 起）；未命中返回 ``None``。"""
    if not evidence_ids:
        return None
    for index, hit in enumerate(hits, start=1):
        if hit.chunk.chunk_id in evidence_ids:
            return index
    return None


def main() -> int:
    """自测入口。"""
    setup_logging()
    settings = get_settings()
    chunks = Chunker.load(settings.paths.data_processed / "chunks.jsonl")
    golden = load_golden(settings.paths.test_data / "golden_qa.jsonl")
    embedder = get_embedder()
    store = VectorStore(index_dir=settings.index_dir(embedder.slug, embedder.dimension))
    if not store.load():
        print("业务失败：索引未就绪，请先执行 build_index.py", file=sys.stderr)
        return 2
    understanding = get_query_understanding()
    reranker = get_reranker()

    pre = Retriever(vector_store=store, embedder=embedder, reranker=None)  # 重排前（融合排序）
    post = Retriever(vector_store=store, embedder=embedder, reranker=reranker)  # 重排后（线上链路）
    for retriever in (pre, post):
        if not retriever.load_index(chunks):
            print("业务失败：索引加载失败（维度不一致或缺失）", file=sys.stderr)
            return 2

    print("=" * 108)
    print("检索自测（工单2）：证据在重排前 top-20 / 重排后 top-5 中的名次（null = 未命中）")
    print(
        f"索引：{store.index_dir} | 分块 {len(chunks)} | 向量 {store.dimension} 维 | "
        f"子块 {len(post._segment_owner)} | 重排 {reranker.mode}"  # noqa: SLF001（自测取证）
    )
    print("=" * 108)
    print(f"{'题号':<6}{'主证据块':<11}{'页':<5}{'证据集':<7}{'重排前':<8}{'重排后':<8}{'进top5':<8}{'最终页码':<26}{'耗时ms':<8}")
    print("-" * 108)

    post_hits = 0
    pre_hits = 0
    top20_hits = 0
    rows: list[dict[str, object]] = []
    for item in golden:
        started = time.perf_counter()
        evidence_chunks = locate_evidence(chunks, item)
        evidence_ids = {chunk.chunk_id for chunk in evidence_chunks}
        analysis = understanding.analyze(item.question, [])
        variants = understanding.search_queries(analysis)
        pre_list = pre.retrieve_multi(variants, analysis=analysis, top_k=20)
        pre_rank = best_rank(pre_list, evidence_ids)
        post_list = post.retrieve_multi(variants, analysis=analysis, top_k=settings.retrieval.rerank_top_n)
        post_rank = best_rank(post_list, evidence_ids)
        post_hits += 1 if post_rank is not None else 0
        pre_hits += 1 if (pre_rank is not None and pre_rank <= settings.retrieval.rerank_top_n) else 0
        top20_hits += 1 if pre_rank is not None else 0
        elapsed = (time.perf_counter() - started) * 1000
        primary = evidence_chunks[0] if evidence_chunks else None
        print(
            f"{item.id:<6}{(primary.chunk_id if primary else '未定位'):<11}"
            f"{(primary.page if primary else '-'):<5}{len(evidence_ids):<7}"
            f"{str(pre_rank if pre_rank is not None else 'null'):<8}"
            f"{str(post_rank if post_rank is not None else 'null'):<8}"
            f"{('✅' if post_rank is not None else '❌'):<8}"
            f"{str([h.chunk.page for h in post_list]):<26}{elapsed:<8.0f}"
        )
        rows.append(
            {
                "question_id": item.id,
                "evidence_chunk_ids": sorted(evidence_ids),
                "evidence_pages": sorted({chunk.page for chunk in evidence_chunks}),
                "pre_rerank_rank": pre_rank,
                "post_rerank_rank": post_rank,
                "pre_rerank_pages": [hit.chunk.page for hit in pre_list],
                "final_pages": [hit.chunk.page for hit in post_list],
                "final_chunk_ids": [hit.chunk.chunk_id for hit in post_list],
                "final_scores": [round(hit.score, 4) for hit in post_list],
                "elapsed_ms": round(elapsed, 1),
            }
        )
    print("-" * 108)
    print(f"重排前 top-20 含证据：{top20_hits}/{len(golden)}")
    print(f"重排前 top-5  含证据：{pre_hits}/{len(golden)}")
    print(f"重排后 top-5  含证据：{post_hits}/{len(golden)}（自测口径；正式判分由 T5/T9 负责）")
    out = settings.paths.data_processed / "selftest_retrieval_ranks.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写入：{out}")
    flush_logs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
