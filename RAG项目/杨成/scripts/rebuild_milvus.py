import argparse
import json
from pathlib import Path

from pymilvus import Collection, MilvusClient, connections, utility

from init_milvus import build_index_params, build_schema
from prepare_records import VECTOR_DIMENSION


class RebuildMilvusError(RuntimeError):
    pass


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RebuildMilvusError(f"{path} 第 {line_number} 行 JSON 解析失败：{exc}") from exc


def to_milvus_row(record):
    required = ("id", "text", "vector", "document_id", "document_version", "source_file", "section_path", "chunk_type")
    missing = [field for field in required if field not in record]
    if missing:
        raise RebuildMilvusError(f"记录缺少字段：{missing}")
    vector = record["vector"]
    if not isinstance(vector, list) or len(vector) != VECTOR_DIMENSION:
        raise RebuildMilvusError(f"{record.get('id', '<unknown>')} 向量维度异常")
    return {
        "id": record["id"],
        "vector": [float(value) for value in vector],
        "text": record["text"],
        "section_path": record.get("section_path", ""),
        "chunk_type": record.get("chunk_type", ""),
        "source_file": record.get("source_file", ""),
        "document_id": record.get("document_id", ""),
        "document_version": record.get("document_version", ""),
    }


def load_prepared_records(paths):
    records = []
    seen_ids = set()
    for path in paths:
        if not path.exists():
            raise RebuildMilvusError(f"输入文件不存在：{path}")
        for record in iter_jsonl(path):
            row = to_milvus_row(record)
            if row["id"] in seen_ids:
                raise RebuildMilvusError(f"重复 id：{row['id']}")
            seen_ids.add(row["id"])
            records.append(row)
    return records


def connect_orm(uri):
    if uri.startswith("http://"):
        uri = uri[len("http://") :]
    if uri.startswith("https://"):
        uri = uri[len("https://") :]
    host, _, port = uri.partition(":")
    connections.connect(alias="default", host=host or "localhost", port=port or "19530")


def recreate_collection(client, uri, collection_name):
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
    connect_orm(uri)
    collection = Collection(name=collection_name, schema=build_schema())
    for index_config in build_index_params():
        collection.create_index(field_name=index_config["field_name"], index_params=index_config["index_params"])
    collection.load()
    return collection


def insert_batches(client, collection_name, records, batch_size):
    inserted_count = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        result = client.insert(collection_name=collection_name, data=batch)
        inserted_count += result.get("insert_count", len(batch)) if isinstance(result, dict) else len(batch)
    client.flush(collection_name=collection_name)
    return inserted_count


def get_entity_count(client, collection_name):
    stats = client.get_collection_stats(collection_name=collection_name)
    return int(stats.get("row_count", 0))


def rebuild_milvus(client, collection_name, records, drop_existing=False, uri="http://localhost:19530", batch_size=50):
    if not drop_existing:
        raise RebuildMilvusError("拒绝重建 collection：请显式传入 --drop-existing")
    recreate_collection(client, uri, collection_name)
    inserted_count = insert_batches(client, collection_name, records, batch_size)
    entity_count = get_entity_count(client, collection_name)
    return {
        "collection": collection_name,
        "expected_count": len(records),
        "inserted_count": inserted_count,
        "collection_entity_count": entity_count,
        "document_ids": sorted({record["document_id"] for record in records}),
    }


def main():
    parser = argparse.ArgumentParser(description="Drop, rebuild, and reload the hypertension RAG Milvus collection.")
    parser.add_argument("--uri", default="http://localhost:19530")
    parser.add_argument("--collection", default="hypertension_rag")
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    args = parser.parse_args()

    if not args.drop_existing:
        raise SystemExit("ERROR: 拒绝重建 collection：请显式传入 --drop-existing")

    try:
        records = load_prepared_records(args.inputs)
        client = MilvusClient(uri=args.uri)
        result = rebuild_milvus(
            client=client,
            collection_name=args.collection,
            records=records,
            drop_existing=args.drop_existing,
            uri=args.uri,
            batch_size=args.batch_size,
        )
    except RebuildMilvusError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
