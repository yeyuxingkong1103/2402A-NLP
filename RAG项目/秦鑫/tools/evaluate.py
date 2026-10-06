import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import evaluate_dataset


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except (OSError, ValueError):
                pass


def parse_ks(value: str) -> list[int]:
    values: list[int] = []
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values or [1, 3, 5, 10]


def parse_thresholds(items: list[str], max_latency_ms: float | None = None) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"阈值格式应为 metric=value，收到：{item}")
        key, value = item.split("=", 1)
        thresholds[key.strip()] = float(value)
    if max_latency_ms is not None:
        thresholds["max:latency.total_ms.p95_ms"] = float(max_latency_ms)
    return thresholds


def load_dataset(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "items", "cases", "dataset"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("评测数据集必须是列表，或包含 records/items/cases/dataset 列表字段")


def call_live_endpoint(rows: list[dict[str, Any]], url: str, include_web: bool, timeout: float) -> list[dict[str, Any]]:
    import httpx

    evaluated_rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        for index, row in enumerate(rows, start=1):
            question = str(row.get("question") or row.get("query") or "").strip()
            if not question:
                evaluated_rows.append(row)
                continue
            payload = {
                "question": question,
                "include_web": bool(row.get("include_web", include_web)),
                "session_id": row.get("session_id") or f"eval-{index}",
            }
            start = perf_counter()
            response = client.post(url, json=payload)
            elapsed_ms = round((perf_counter() - start) * 1000, 3)
            response.raise_for_status()
            body = response.json()
            data = body.get("data", body) if isinstance(body, dict) else {}
            meta = data.get("meta", {}) if isinstance(data, dict) else {}
            sources = data.get("sources", []) if isinstance(data, dict) else []
            latency = {"total_ms": elapsed_ms}
            if isinstance(meta, dict):
                latency["total_ms"] = meta.get("elapsed_ms", elapsed_ms)
                for key, value in meta.get("stage_timings", {}).items():
                    latency[f"{key}_ms"] = value
            evaluated_rows.append({
                **row,
                "answer": data.get("answer", {}) if isinstance(data, dict) else {},
                "returned": sources,
                "sources": sources,
                "meta": meta,
                "latency_ms": latency,
            })
    return evaluated_rows


def write_report(report: dict[str, Any], output: Path | None, report_dir: Path) -> Path | None:
    if output is None:
        return None
    if output.is_dir():
        report_dir = output
        output = None
    report_dir.mkdir(parents=True, exist_ok=True)
    path = output or report_dir / f"law-rag-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(path)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def print_text_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    retrieval = summary.get("retrieval", {})
    answer = summary.get("answer", {})
    citation = summary.get("citation", {})
    latency = summary.get("latency", {})
    data_status = summary.get("data_status", {})
    total_latency = latency.get("total_ms", {})
    lines = [
        f"status: {report['status']}",
        f"mode: {report.get('mode', 'evaluated')}",
        f"count: {report['count']}",
        f"retrieval recall@5: {retrieval.get('recall@5', 'n/a')}",
        f"retrieval precision@5: {retrieval.get('precision@5', 'n/a')}",
        f"retrieval mrr@5: {retrieval.get('mrr@5', 'n/a')}",
        f"answer keyword_coverage: {answer.get('keyword_coverage', 'n/a')}",
        f"citation groundedness: {citation.get('groundedness', 'n/a')}",
        f"latency total p95_ms: {total_latency.get('p95_ms', 'n/a')}",
    ]
    if data_status.get("golden_only_count") == report.get("count"):
        lines.append("note: golden_only 数据集尚未包含实际 returned/answer；请加 --ask-url 进行真实 RAG 评估")
    print("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评测 Law-RAG 检索、回答、引用和性能")
    parser.add_argument("dataset", nargs="?", help="JSON 数据集，建议放在 evaluation/datasets 下")
    parser.add_argument("--k", default="1,3,5,10", help="逗号分隔的 TopK 列表，例如 1,3,5,10")
    parser.add_argument("--ask-url", help="可选：调用本地问答接口生成结果，例如 http://127.0.0.1:7294/api/v1/legal/ask")
    parser.add_argument("--include-web", action="store_true", help="调用 --ask-url 时打开 Web 检索")
    parser.add_argument("--timeout", type=float, default=120.0, help="调用 --ask-url 的单题超时时间，单位秒")
    parser.add_argument("--fail-under", action="append", default=[], help="汇总指标下限，例如 retrieval.recall@5=0.8")
    parser.add_argument("--max-latency-ms", type=float, help="total_ms p95 延迟上限")
    parser.add_argument("--output", type=Path, help="报告输出路径；如果是目录则自动生成文件名")
    parser.add_argument("--report-dir", type=Path, default=Path("evaluation/reports"), help="默认报告目录")
    parser.add_argument("--format", choices=("json", "text"), default="json", help="终端输出格式")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.dataset:
        print(json.dumps({
            "status": "no_dataset",
            "message": "请传入 evaluation/datasets 下的 JSON 数据集",
            "schema": {
                "question": "问题文本",
                "expected": ["期望命中的 source_id"],
                "returned": ["实际返回的 source_id，或使用 sources/results/evidence 字段"],
                "answer": "答案文本，或包含 answer/citations 的对象",
                "expected_answer_keywords": ["答案应该覆盖的关键词"],
                "expected_citations": ["答案应引用的 source_id"],
                "forbidden_answer_terms": ["答案不应出现的词"],
                "latency_ms": {"total_ms": 1000, "retrieval_ms": 100},
            },
        }, ensure_ascii=False, indent=2))
        return 0

    dataset_path = Path(args.dataset)
    rows = load_dataset(dataset_path)
    if args.ask_url:
        rows = call_live_endpoint(rows, args.ask_url, args.include_web, args.timeout)
    thresholds = parse_thresholds(args.fail_under, args.max_latency_ms)
    report = evaluate_dataset(rows, parse_ks(args.k), thresholds or None)
    report["dataset"] = str(dataset_path)
    write_report(report, args.output, args.report_dir)

    if args.format == "text":
        print_text_summary(report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
