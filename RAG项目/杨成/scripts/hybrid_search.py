import argparse
import json
import time
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel
from pymilvus import MilvusClient


DEFAULT_COLLECTION = "hypertension_rag"
DEFAULT_URI = "http://localhost:19530"
DEFAULT_MODEL_PATH = r"D:\桌面缓存\bge-m3"
DEFAULT_QUESTION = "氨氯地平的禁忌证是什么？"
DEFAULT_OUTPUT = Path("data/runs/search/hybrid_search_result.json")
VECTOR_DIMENSION = 1024
OUTPUT_FIELDS = ["text", "section_path", "chunk_type", "source_file", "document_id", "document_version"]


def normalize_vector(vector):
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def embed_query(model, question):
    output = model.encode(
        [question],
        batch_size=1,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense_vectors = output["dense_vecs"] if isinstance(output, dict) else output
    query_vector = normalize_vector(dense_vectors[0])
    if len(query_vector) != VECTOR_DIMENSION:
        raise ValueError(f"query vector dimension mismatch: {len(query_vector)} != {VECTOR_DIMENSION}")
    return query_vector


def hit_score(hit):
    return hit.get("distance", hit.get("score"))


def normalize_hit(hit, rank, score_field):
    entity = hit.get("entity", {})
    item = {
        "id": hit.get("id"),
        "text": entity.get("text", ""),
        "section_path": entity.get("section_path", ""),
        "chunk_type": entity.get("chunk_type", ""),
        "source_file": entity.get("source_file", ""),
        "document_id": entity.get("document_id", ""),
        "document_version": entity.get("document_version", ""),
    }
    item[f"{score_field}_rank"] = rank
    item[f"{score_field}_score"] = float(hit_score(hit))
    return item


def dense_search(client, collection, query_vector, top_k):
    results = client.search(
        collection_name=collection,
        data=[query_vector],
        anns_field="vector",
        limit=top_k,
        output_fields=OUTPUT_FIELDS,
        search_params={"metric_type": "COSINE", "params": {"ef": 128}},
    )
    return [normalize_hit(hit, rank, "dense") for rank, hit in enumerate(results[0] if results else [], start=1)]


def bm25_search(client, collection, question, top_k):
    results = client.search(
        collection_name=collection,
        data=[question],
        anns_field="sparse_vector",
        limit=top_k,
        output_fields=OUTPUT_FIELDS,
        search_params={"metric_type": "BM25", "params": {}},
    )
    return [normalize_hit(hit, rank, "bm25") for rank, hit in enumerate(results[0] if results else [], start=1)]


def reciprocal_rank_fusion(result_sets, k=60):
    fused_by_id = {}
    order = 0
    rank_field_names = ["dense_rank", "bm25_rank"]

    for retriever_index, results in enumerate(result_sets):
        rank_field = rank_field_names[retriever_index] if retriever_index < len(rank_field_names) else f"retriever_{retriever_index + 1}_rank"
        seen_in_result = set()
        for rank, hit in enumerate(results, start=1):
            hit_id = hit.get("id")
            if not hit_id or hit_id in seen_in_result:
                continue
            seen_in_result.add(hit_id)
            if hit_id not in fused_by_id:
                order += 1
                fused_by_id[hit_id] = {"id": hit_id, "fusion_score": 0.0, "_order": order}
            fused = fused_by_id[hit_id]
            fused["fusion_score"] += 1.0 / (k + rank)
            fused[rank_field] = rank
            for key, value in hit.items():
                if key not in fused or fused.get(key) in (None, ""):
                    fused[key] = value

    fused = list(fused_by_id.values())
    fused.sort(key=lambda item: (-item["fusion_score"], item["_order"]))
    for item in fused:
        item.pop("_order", None)
    return fused


def hybrid_search(client, collection, question, query_vector, dense_top_k=20, bm25_top_k=20, rrf_k=60):
    dense_hits = dense_search(client, collection, query_vector, dense_top_k)
    bm25_hits = bm25_search(client, collection, question, bm25_top_k)
    fused_hits = reciprocal_rank_fusion([dense_hits, bm25_hits], k=rrf_k)
    return dense_hits, bm25_hits, fused_hits


def preview(text, limit=100):
    return " ".join((text or "").split())[:limit]


def compact_hit(hit, text_limit=100):
    return {
        "id": hit.get("id"),
        "section_path": hit.get("section_path"),
        "chunk_type": hit.get("chunk_type"),
        "source_file": hit.get("source_file"),
        "document_id": hit.get("document_id"),
        "document_version": hit.get("document_version"),
        "dense_score": hit.get("dense_score"),
        "bm25_score": hit.get("bm25_score"),
        "fusion_score": hit.get("fusion_score"),
        "rerank_score": hit.get("rerank_score"),
        "text_preview": preview(hit.get("text"), text_limit),
    }


def main():
    parser = argparse.ArgumentParser(description="Dense + BM25 + RRF hybrid search test.")
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--dense-top-k", type=int, default=20)
    parser.add_argument("--bm25-top-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    start = time.perf_counter()
    model = BGEM3FlagModel(args.model_path, use_fp16=False)
    query_vector = embed_query(model, args.question)
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
    elapsed = time.perf_counter() - start

    result = {
        "question": args.question,
        "query_vector_dimension": len(query_vector),
        "dense_top_k": args.dense_top_k,
        "bm25_top_k": args.bm25_top_k,
        "rrf_k": args.rrf_k,
        "elapsed_seconds": round(elapsed, 2),
        "dense_hits": [compact_hit(hit) for hit in dense_hits[:5]],
        "bm25_hits": [compact_hit(hit) for hit in bm25_hits[:5]],
        "fused_hits": [compact_hit(hit) for hit in fused_hits[:10]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"question: {args.question}")
    print(f"query_vector_dimension: {len(query_vector)}")
    print(f"dense_candidates: {len(dense_hits)}")
    print(f"bm25_candidates: {len(bm25_hits)}")
    print(f"fused_candidates: {len(fused_hits)}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    for rank, hit in enumerate(fused_hits[:5], start=1):
        print(f"\n#{rank} fusion={hit.get('fusion_score'):.6f} id={hit.get('id')}")
        print(f"document_id: {hit.get('document_id')}")
        print(f"section_path: {hit.get('section_path')}")
        print(f"text: {preview(hit.get('text'))}")
    print(f"\nresult_json: {args.output.as_posix()}")


if __name__ == "__main__":
    main()
