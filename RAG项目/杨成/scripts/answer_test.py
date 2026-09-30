import argparse
import json
import os
import re
import time
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from pymilvus import MilvusClient

from hybrid_search import embed_query, hybrid_search


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


def build_context(contexts):
    blocks = []
    for index, item in enumerate(contexts, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{index}] id: {item.get('id')}",
                    f"section_path: {item.get('section_path')}",
                    f"chunk_type: {item.get('chunk_type')}",
                    f"rerank_score: {item.get('rerank_score')}",
                    f"text: {item.get('text')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def extract_field(text, field_name):
    pattern = rf"{re.escape(field_name)}: (.*?)(?: \| [^|]+:|$)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def extract_drug_name(text):
    return extract_field(text, "名称")


def extract_fallback_answer(question, contexts):
    if not contexts:
        return "未检索到可用依据，无法回答。"

    top = contexts[0]
    text = top.get("text", "")
    drug_name = extract_drug_name(text)
    section_path = top.get("section_path", "")

    if "禁忌" in question:
        contraindication = extract_field(text, "禁忌证")
        if contraindication:
            return f"{drug_name}的禁忌证为：{contraindication}。依据：{section_path}。"

    return f"根据检索到的资料：{text}。依据：{section_path}。"


def build_prompt(question, context):
    return f"""你是一名社区高血压健康管理助手。请只根据给定资料回答问题，不要编造资料外内容。

问题：{question}

资料：
{context}

回答要求：
1. 先直接回答结论。
2. 如果资料来自表格，保留药物名称和字段名称。
3. 最后列出引用编号，例如 [1]。
"""


def call_openai_compatible_llm(prompt):
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL")
    if not api_key or not model:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def preview(text, limit=120):
    return " ".join((text or "").split())[:limit]


def main():
    parser = argparse.ArgumentParser(description="End-to-end RAG QA smoke test with hybrid retrieval and reranking.")
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL_PATH)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL_PATH)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--dense-top-k", type=int, default=20)
    parser.add_argument("--bm25-top-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/runs/answer/answer_test_result.json"))
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
    contexts = reranked_hits[: args.rerank_top_k]
    context_text = build_context(contexts)
    prompt = build_prompt(args.question, context_text)
    answer = call_openai_compatible_llm(prompt) or extract_fallback_answer(args.question, contexts)
    elapsed = time.perf_counter() - start

    result = {
        "question": args.question,
        "query_vector_dimension": len(query_vector),
        "dense_top_k": args.dense_top_k,
        "bm25_top_k": args.bm25_top_k,
        "rrf_k": args.rrf_k,
        "rerank_top_k": args.rerank_top_k,
        "elapsed_seconds": round(elapsed, 2),
        "answer": answer,
        "contexts": [
            {
                "rank": index,
                "id": item["id"],
                "section_path": item["section_path"],
                "chunk_type": item["chunk_type"],
                "dense_score": item.get("dense_score"),
                "bm25_score": item.get("bm25_score"),
                "fusion_score": item.get("fusion_score"),
                "rerank_score": item["rerank_score"],
                "source_file": item.get("source_file"),
                "document_id": item.get("document_id"),
                "document_version": item.get("document_version"),
                "text_preview": preview(item["text"]),
            }
            for index, item in enumerate(contexts, start=1)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"question: {args.question}")
    print(f"query_vector_dimension: {len(query_vector)}")
    print(f"dense_candidates: {len(dense_hits)}")
    print(f"bm25_candidates: {len(bm25_hits)}")
    print(f"fused_candidates: {len(fused_hits)}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print("\ncontexts:")
    for context in result["contexts"]:
        print(
            f"[{context['rank']}] {context['chunk_type']} rerank={context['rerank_score']:.4f} "
            f"section={context['section_path']} text={context['text_preview']}"
        )
    print("\nanswer:")
    print(answer)
    print(f"\nresult_json: {args.output.as_posix()}")


if __name__ == "__main__":
    main()
