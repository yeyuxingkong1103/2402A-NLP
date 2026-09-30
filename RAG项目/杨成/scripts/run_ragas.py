import argparse
import copy
import json
import math
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness


DEFAULT_INPUT = Path("data/processed/eval_results.json")
DEFAULT_OUTPUT = Path("data/processed/ragas_result.json")
DEFAULT_EMBEDDING_MODEL_PATH = r"D:\桌面缓存\bge-m3"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall"]


load_dotenv()


def load_eval_results(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_ragas_rows(results):
    rows = []
    for item in results:
        if item.get("error"):
            continue
        rows.append(
            {
                "user_input": item.get("question", ""),
                "response": item.get("answer", ""),
                "retrieved_contexts": item.get("retrieved_contexts", []),
                "reference": item.get("ground_truth", ""),
            }
        )
    return rows


def as_float(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def averages(scores):
    output = {}
    for metric in METRIC_NAMES:
        values = [as_float(row.get(metric)) for row in scores]
        values = [value for value in values if value is not None]
        output[metric] = round(sum(values) / len(values), 4) if values else None
    return output


def build_output(source_results, scores):
    successful_source = [item for item in source_results if not item.get("error")]
    rows = []
    for source, score in zip(successful_source, scores):
        rows.append(
            {
                "question": source.get("question", ""),
                "faithfulness": as_float(score.get("faithfulness")),
                "answer_relevancy": as_float(score.get("answer_relevancy")),
                "context_recall": as_float(score.get("context_recall")),
            }
        )
    return {"averages": averages(rows), "results": rows}


def make_llm(model, timeout):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=DEFAULT_BASE_URL,
        temperature=0.1,
        timeout=timeout,
        max_retries=1,
    )


def make_embeddings(model_path):
    return HuggingFaceEmbeddings(
        model=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def make_metrics():
    answer_relevancy_metric = copy.deepcopy(answer_relevancy)
    answer_relevancy_metric.strictness = 1
    return [faithfulness, answer_relevancy_metric, context_recall]


def run_ragas(args):
    eval_results = load_eval_results(args.input)
    rows = to_ragas_rows(eval_results)
    dataset = Dataset.from_list(rows)
    llm = make_llm(args.model, args.timeout)
    embeddings = make_embeddings(args.embedding_model)
    result = evaluate(
        dataset=dataset,
        metrics=make_metrics(),
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=True,
    )
    score_rows = result.to_pandas()[METRIC_NAMES].to_dict(orient="records")
    output = build_output(eval_results, score_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run RAGAS metrics over eval_results.json.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL_PATH)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    output = run_ragas(args)
    print("averages:")
    for metric, value in output["averages"].items():
        print(f"  {metric}: {value}")
    print("per_question:")
    for index, item in enumerate(output["results"], start=1):
        print(
            f"  {index}. faithfulness={item['faithfulness']} "
            f"answer_relevancy={item['answer_relevancy']} "
            f"context_recall={item['context_recall']} | {item['question']}"
        )
    print(f"result_json: {args.output.as_posix()}")


if __name__ == "__main__":
    main()
