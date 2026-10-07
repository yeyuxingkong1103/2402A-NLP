"""工单 05：Query 理解优化入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.index import DocumentIndex
from common.llm import answer_with_backend, chunks_to_results
from common.query import QueryAnalysis, analyze_query, expand_query, rewrite_query
from 工单02_系统优化.main import optimize_results


def build_query_context(
    query: str,
    recent_turns: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    limit: int = 3,
    max_chars: int = 2000,
) -> str:
    """按最新轮次优先整理上下文，并限制轮数与字符数。"""
    if not query.strip():
        raise ValueError("query 不能为空")
    current = f"当前问题：{query.strip()}"
    if limit <= 0 or max_chars <= 0:
        return current
    selected = list(recent_turns)[-limit:]
    remaining = max_chars - len(current)
    if remaining <= 0:
        return current
    history_parts = []
    for question, answer in reversed(selected):
        if remaining <= 1:
            break
        history = f"问题：{question}\n答案：{answer}"
        addition = "\n" + history
        if len(addition) > remaining:
            addition = addition[:remaining]
        history_parts.insert(0, addition.lstrip("\n"))
        remaining -= len(addition)
    return "\n".join(history_parts + [current])


def process_query(query: str, recent_turns=(), *, llm=None) -> tuple[QueryAnalysis, str, list[str]]:
    rewritten = rewrite_query(query, recent_turns, llm=llm)
    analysis = analyze_query(rewritten)
    return analysis, rewritten, expand_query(rewritten)


def retrieve_query(query: str, index: DocumentIndex, top_k: int = 5, recent_turns=()):
    """把扩展 Query 交给工单 02 风格检索并合并结果。"""
    _, rewritten, variants = process_query(query, recent_turns)
    candidates = []
    seen = set()
    for variant in variants:
        for result in index.search(variant, top_k=top_k):
            identity = result.metadata.get("chunk_id") or (result.source, result.page, result.content)
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(next(
                chunk for chunk in index.chunks
                if chunk.source == result.source and chunk.page == result.page and chunk.content == result.content
            ))
    ranked = optimize_results(rewritten, candidates, top_k=top_k)
    return rewritten, ranked


def main(argv: list[str] | None = None) -> int:
    def positive_int(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--top-k 必须是正整数") from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError("--top-k 必须是正整数")
        return parsed

    parser = argparse.ArgumentParser(description="Query 理解与检索")
    parser.add_argument("query", nargs="?", help="用户问题")
    parser.add_argument("--index", type=Path, help="DocumentIndex JSON 文件")
    parser.add_argument("--top-k", type=positive_int, default=5)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    if not isinstance(args.query, str) or not args.query.strip():
        print("错误：query 不能为空", file=sys.stderr)
        return 2
    if not args.index:
        analysis, rewritten, variants = process_query(args.query)
        print(f"意图：{analysis.intent}\n改写：{rewritten}\n扩展：{' | '.join(variants)}")
        return 0
    try:
        index = DocumentIndex.load(args.index)
        rewritten, results = retrieve_query(args.query, index, args.top_k)
        answer = answer_with_backend(rewritten, chunks_to_results([result.chunk for result in results]))
        print(answer.text)
    except (OSError, ValueError, StopIteration) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
