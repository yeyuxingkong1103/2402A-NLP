import argparse
import json
from pathlib import Path

from pymilvus import MilvusClient

from prepare_records import normalize_source_file


DEFAULT_COLLECTION = "hypertension_rag"
DEFAULT_URI = "http://localhost:19530"
DEFAULT_INPUTS = [
    Path("data/processed/vectors/prepared_guideline_2025.jsonl"),
    Path("data/processed/vectors/prepared_nutrition_exercise_2024.jsonl"),
]
BATCH_SIZE = 50


class InsertError(RuntimeError):
    pass


def get_prepared_id(record):
    record_id = str(record.get("id") or "").strip()
    if not record_id:
        raise InsertError(f"缺少 prepared record id：{record}")
    return record_id


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise InsertError(f"{path} 第 {line_number} 行 JSON 解析失败：{exc}") from exc


def load_records(paths):
    records = []
    for path in paths:
        if not path.exists():
            raise InsertError(f"输入文件不存在：{path}")
        file_count = 0
        for record in iter_jsonl(path):
            records.append(to_milvus_row(record))
            file_count += 1
        print(f"loaded {file_count} records from {path.as_posix()}")
    return records


def to_milvus_row(record):
    vector = record.get("vector")
    if not isinstance(vector, list) or len(vector) != 1024:
        raise InsertError(f"{record.get('child_id', '<unknown>')} 向量维度异常")
    return {
        "id": get_prepared_id(record),
        "vector": [float(value) for value in vector],
        "text": record.get("text", ""),
        "section_path": record.get("section_path", ""),
        "chunk_type": record.get("chunk_type", ""),
        "source_file": normalize_source_file(record.get("source_file", "")),
        "document_id": record.get("document_id", ""),
        "document_version": record.get("document_version", ""),
    }


def insert_batches(client, collection_name, records, batch_size, upsert=False):
    total_count = len(records)
    success_count = 0
    failed_count = 0
    operation = client.upsert if upsert else client.insert
    operation_name = "upserted" if upsert else "inserted"

    for start in range(0, total_count, batch_size):
        batch = records[start : start + batch_size]
        batch_number = start // batch_size + 1
        try:
            result = operation(collection_name=collection_name, data=batch)
            result_key = "upsert_count" if upsert else "insert_count"
            insert_count = result.get(result_key, len(batch)) if isinstance(result, dict) else len(batch)
            success_count += insert_count
            print(f"batch {batch_number}: {operation_name} {insert_count}/{len(batch)} ({success_count}/{total_count})")
        except Exception:
            failed_count += len(batch)
            print(f"batch {batch_number}: failed {len(batch)} records")
            raise

    client.flush(collection_name=collection_name)
    return total_count, success_count, failed_count


def get_entity_count(client, collection_name):
    stats = client.get_collection_stats(collection_name=collection_name)
    row_count = stats.get("row_count", 0)
    return int(row_count)


def main():
    parser = argparse.ArgumentParser(description="Insert vectorized hypertension RAG chunks into Milvus.")
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--upsert", action="store_true", help="Use explicit upsert for re-importing stable IDs.")
    parser.add_argument("--inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    args = parser.parse_args()

    client = MilvusClient(uri=args.uri)
    client.load_collection(collection_name=args.collection)

    records = load_records(args.inputs)
    total_count, success_count, failed_count = insert_batches(
        client=client,
        collection_name=args.collection,
        records=records,
        batch_size=args.batch_size,
        upsert=args.upsert,
    )
    entity_count = get_entity_count(client, args.collection)

    print(f"total_count: {total_count}")
    print(f"success_count: {success_count}")
    print(f"failed_count: {failed_count}")
    print(f"collection_entity_count: {entity_count}")


if __name__ == "__main__":
    main()
