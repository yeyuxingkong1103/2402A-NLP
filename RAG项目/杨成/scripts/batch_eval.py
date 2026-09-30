import argparse
import json
import time
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from pymilvus import MilvusClient

import rag_answer


DEFAULT_DATASET = Path("data/processed/eval_dataset.json")
DEFAULT_OUTPUT = Path("data/processed/eval_results.json")


def load_dataset(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def retrieved_contexts(chunks):
    return [str(chunk.get("text", "")) for chunk in chunks]


def retrieved_sources(chunks):
    return [str(chunk.get("section_path", "")) for chunk in chunks]


def build_eval_record(dataset_item, result, retrieved):
    return {
        "question": dataset_item["question"],
        "ground_truth": dataset_item["ground_truth"],
        "answer": result.get("answer", ""),
        "retrieved_contexts": retrieved_contexts(retrieved),
        "retrieved_sources": retrieved_sources(retrieved),
        "latency_ms": result.get("latency", {}).get("total_ms"),
        "has_citation": result.get("has_citation"),
    }


def build_error_record(dataset_item, error, latency_ms):
    return {
        "question": dataset_item.get("question", ""),
        "ground_truth": dataset_item.get("ground_truth", ""),
        "answer": "",
        "retrieved_contexts": [],
        "retrieved_sources": [],
        "latency_ms": latency_ms,
        "has_citation": False,
        "error": str(error),
    }


def evaluate_item(dataset_item, client, collection, embedding_model, reranker, model, timeout):
    retrieval_start = time.perf_counter()
    top5 = rag_answer.retrieve_top5(client, collection, embedding_model, reranker, dataset_item["question"])
    retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000)

    prompt = rag_answer.build_prompt(dataset_item["question"], rag_answer.build_retrieved_context(top5))
    generation_start = time.perf_counter()
    answer = rag_answer.call_deepseek_api(prompt, model=model, timeout=timeout)
    generation_ms = round((time.perf_counter() - generation_start) * 1000)

    top1_section_path = top5[0].get("section_path", "") if top5 else ""
    answer, citation_status = rag_answer.process_answer(answer, top1_section_path)
    result = {
        "answer": answer,
        "has_citation": citation_status,
        "latency": {
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": retrieval_ms + generation_ms,
        },
    }
    return build_eval_record(dataset_item, result, top5)


def run_batch(args):
    dataset = load_dataset(args.dataset)
    total = len(dataset)

    embedding_model = BGEM3FlagModel(args.embedding_model, use_fp16=False)
    reranker = FlagReranker(args.reranker_model, use_fp16=False)
    client = MilvusClient(uri=args.uri)
    client.load_collection(collection_name=args.collection)

    results = []
    for index, item in enumerate(dataset, start=1):
        start = time.perf_counter()
        try:
            record = evaluate_item(
                dataset_item=item,
                client=client,
                collection=args.collection,
                embedding_model=embedding_model,
                reranker=reranker,
                model=args.model,
                timeout=args.timeout,
            )
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            results.append(record)
            print(f"第 {index}/{total} 条：成功，耗时 {elapsed_ms} ms")
        except Exception as error:
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            record = build_error_record(item, error, elapsed_ms)
            results.append(record)
            print(f"第 {index}/{total} 条：失败，耗时 {elapsed_ms} ms，错误：{error}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Batch RAG evaluation over eval_dataset.json.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--uri", default=rag_answer.DEFAULT_URI)
    parser.add_argument("--collection", default=rag_answer.DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=rag_answer.DEFAULT_EMBEDDING_MODEL_PATH)
    parser.add_argument("--reranker-model", default=rag_answer.DEFAULT_RERANKER_MODEL_PATH)
    parser.add_argument("--model", default=rag_answer.DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=rag_answer.DEFAULT_TIMEOUT)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    results = run_batch(args)
    success_count = sum(1 for item in results if not item.get("error"))
    failure_count = len(results) - success_count
    total_latency = sum(item.get("latency_ms") or 0 for item in results)
    avg_latency = round(total_latency / len(results)) if results else 0
    print(f"完成：成功 {success_count} 条，失败 {failure_count} 条，平均耗时 {avg_latency} ms")
    print(f"result_json: {args.output.as_posix()}")


if __name__ == "__main__":
    main()
