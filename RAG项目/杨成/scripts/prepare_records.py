import argparse
import json
from pathlib import Path

from document_identity import make_primary_key, normalize_document_id


VECTOR_DIMENSION = 1024


class PrepareRecordsError(ValueError):
    pass


def normalize_source_file(source_file):
    value = str(source_file or "").strip().replace("\\", "/")
    if not value:
        return ""
    name = Path(value).name
    if name.endswith(".full.md"):
        return name[: -len(".full.md")]
    if name.endswith(".md"):
        return name[: -len(".md")]
    return Path(name).stem or name


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise PrepareRecordsError(f"{path} 第 {line_number} 行 JSON 解析失败：{exc}") from exc


def validate_vector(record):
    vector = record.get("vector")
    child_id = record.get("child_id", "<unknown>")
    if not isinstance(vector, list) or len(vector) != VECTOR_DIMENSION:
        raise PrepareRecordsError(f"{child_id} 向量维度异常：{len(vector) if isinstance(vector, list) else 'missing'}")
    try:
        return [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise PrepareRecordsError(f"{child_id} 向量包含非数值元素") from exc


def prepare_record(record, document_id, document_version):
    child_id = str(record.get("child_id") or "").strip()
    text = str(record.get("text") or "").strip()
    if not child_id:
        raise PrepareRecordsError("缺少 child_id")
    if not text:
        raise PrepareRecordsError(f"{child_id} 缺少 text")

    output_record = dict(record)
    output_record["vector"] = validate_vector(record)
    output_record["source_file"] = normalize_source_file(record.get("source_file"))
    output_record["document_id"] = normalize_document_id(document_id)
    output_record["document_version"] = str(document_version)
    output_record["id"] = make_primary_key(document_id, child_id)
    return output_record


def prepare_jsonl(input_path, output_path, document_id, document_version, seen_ids=None):
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise PrepareRecordsError(f"输入文件不存在：{input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids = seen_ids if seen_ids is not None else set()
    count = 0

    with output_path.open("a" if output_path.exists() and output_path.stat().st_size else "w", encoding="utf-8") as output_file:
        for _, record in iter_jsonl(input_path):
            output_record = prepare_record(record, document_id, document_version)
            record_id = output_record["id"]
            if record_id in seen_ids:
                raise PrepareRecordsError(f"重复 id：{record_id}")
            seen_ids.add(record_id)
            output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            count += 1

    return count


def prepare_inputs(input_paths, output_path, document_id, document_version):
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()
    seen_ids = set()
    total_count = 0
    for input_path in input_paths:
        total_count += prepare_jsonl(input_path, output_path, document_id, document_version, seen_ids=seen_ids)
    return total_count


def main():
    parser = argparse.ArgumentParser(description="Prepare vector JSONL records with stable document metadata.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        total_count = prepare_inputs(args.inputs, args.output, args.document_id, args.version)
    except PrepareRecordsError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print(
        json.dumps(
            {
                "document_id": normalize_document_id(args.document_id),
                "document_version": str(args.version),
                "output": args.output.as_posix(),
                "record_count": total_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
