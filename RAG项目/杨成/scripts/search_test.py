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
DEFAULT_OUTPUT = Path("data/runs/search/search_test_result.json")
VECTOR_DIMENSION = 1024


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


def text_preview(text, limit=100):
    text = " ".join((text or "").split())
    return text[:limit]


def main():
    parser = argparse.ArgumentParser(description="Dense search test for hypertension RAG Milvus collection.")
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    start = time.perf_counter()
    model = BGEM3FlagModel(args.model_path, use_fp16=False)
    query_vector = embed_query(model, args.question)

    client = MilvusClient(uri=args.uri)
    client.load_collection(collection_name=args.collection)
    results = client.search(
        collection_name=args.collection,
        data=[query_vector],
        anns_field="vector",
        limit=args.top_k,
        output_fields=["text", "section_path", "chunk_type", "source_file"],
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    )

    elapsed = time.perf_counter() - start
    print(f"question: {args.question}")
    print(f"collection: {args.collection}")
    print(f"top_k: {args.top_k}")
    print(f"query_vector_dimension: {len(query_vector)}")
    print(f"elapsed_seconds: {elapsed:.2f}")

    hits = results[0] if results else []
    output_hits = []
    for rank, hit in enumerate(hits, start=1):
        entity = hit.get("entity", {})
        score = hit.get("distance", hit.get("score"))
        output_hits.append(
            {
                "rank": rank,
                "id": hit.get("id"),
                "score": score,
                "text": entity.get("text"),
                "section_path": entity.get("section_path"),
                "chunk_type": entity.get("chunk_type"),
                "source_file": entity.get("source_file"),
            }
        )
        print(f"\n#{rank}")
        print(f"id: {hit.get('id')}")
        print(f"section_path: {entity.get('section_path')}")
        print(f"chunk_type: {entity.get('chunk_type')}")
        print(f"score: {score}")
        print(f"text: {text_preview(entity.get('text'))}")

    output = {
        "question": args.question,
        "collection": args.collection,
        "top_k": args.top_k,
        "query_vector_dimension": len(query_vector),
        "elapsed_seconds": elapsed,
        "hits": output_hits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\noutput: {args.output}")


if __name__ == "__main__":
    main()
