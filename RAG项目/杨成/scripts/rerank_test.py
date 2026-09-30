import argparse
import json
import time
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from pymilvus import MilvusClient

from hybrid_search import compact_hit, embed_query, hybrid_search


DEFAULT_COLLECTION = "hypertension_rag"
DEFAULT_URI = "http://localhost:19530"
DEFAULT_EMBEDDING_MODEL_PATH = r"D:\桌面缓存\bge-m3"
DEFAULT_RERANKER_MODEL_PATH = r"D:\桌面缓存\bge-reranker-v2-m3\bge-reranker-v2-m3"
DEFAULT_QUESTION = "氨氯地平的禁忌证是什么？"
VECTOR_DIMENSION = 1024


def rerank_hits(reranker, question, hits):
    pairs = [[question, hit["text"]] for hit in hits]
    scores = reranker.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [float(scores)]
    reranked = []
    for hit, score in zip(hits, scores):
        item = dict(hit)
        item["rerank_score"] = float(score)
        reranked.append(item)
    return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)


def preview(text, limit):
    return " ".join((text or "").split())[:limit]


def print_hits(title, hits, score_field):
    print(title)
    for index, hit in enumerate(hits, start=1):
        print(f"#{index}")
        print(f"id: {hit['id']}")
        print(f"section_path: {hit['section_path']}")
        print(f"chunk_type: {hit['chunk_type']}")
        print(f"{score_field}: {hit.get(score_field)}")
        print(f"text: {preview(hit['text'], 80)}")


def main():
    parser = argparse.ArgumentParser(description="Dense + BM25 RRF retrieval plus BGE reranker test.")
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL_PATH)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL_PATH)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--dense-top-k", type=int, default=20)
    parser.add_argument("--bm25-top-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/runs/rerank/rerank_test_result.json"))
    args = parser.parse_args()

    start = time.perf_counter()
    embedding_model = BGEM3FlagModel(args.embedding_model, use_fp16=False)
    query_vector = embed_query(embedding_model, args.question)

    client = MilvusClient(uri=args.uri)
    client.load_collection(collection_name=args.collection)
    dense_hits, bm25_hits, fused_hits = hybrid_search(
        client=client,
        collection=args.collection,
        question=args.question,
        query_vector=query_vector,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        rrf_k=args.rrf_k,
    )

    reranker = FlagReranker(args.reranker_model, use_fp16=False)
    reranked_hits = rerank_hits(reranker, args.question, fused_hits[: args.dense_top_k])
    elapsed = time.perf_counter() - start

    dense_top5 = dense_hits[:5]
    rerank_top5 = reranked_hits[: args.rerank_top_k]
    top1 = rerank_top5[0] if rerank_top5 else None
    top1_is_amlodipine = bool(top1 and "名称: 氨氯地平" in top1["text"])

    result = {
        "question": args.question,
        "query_vector_dimension": len(query_vector),
        "dense_top_k": args.dense_top_k,
        "bm25_top_k": args.bm25_top_k,
        "rrf_k": args.rrf_k,
        "rerank_top_k": args.rerank_top_k,
        "elapsed_seconds": round(elapsed, 2),
        "rerank_top1_is_amlodipine": top1_is_amlodipine,
        "rerank_top1_name": preview(top1["text"], 80) if top1 else None,
        "dense_top5": [compact_hit(hit) for hit in dense_top5],
        "bm25_top5": [compact_hit(hit) for hit in bm25_hits[:5]],
        "fused_top20": [compact_hit(hit) for hit in fused_hits[:20]],
        "rerank_top5": [compact_hit(hit) for hit in rerank_top5],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"question: {args.question}")
    print(f"query_vector_dimension: {len(query_vector)}")
    print(f"dense_candidates: {len(dense_hits)}")
    print(f"bm25_candidates: {len(bm25_hits)}")
    print(f"fused_candidates: {len(fused_hits)}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print(f"rerank_top1_is_amlodipine: {top1_is_amlodipine}")
    print()
    print_hits("Dense top-5", dense_top5, "dense_score")
    print()
    print_hits("BM25 top-5", bm25_hits[:5], "bm25_score")
    print()
    print_hits("Fused top-5", fused_hits[:5], "fusion_score")
    print()
    print_hits("Rerank top-5", rerank_top5, "rerank_score")
    print(f"\nresult_json: {args.output.as_posix()}")


if __name__ == "__main__":
    main()
