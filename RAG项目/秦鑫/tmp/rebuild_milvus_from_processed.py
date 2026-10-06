import json
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from backend.app.config import get_settings
from backend.app.models import ModelGateway
from backend.app.storage.vector import MilvusStore
from data_pipeline.embedding import embed_public_batch, make_public_batches


COLLECTIONS = (
    "civil_code_articles",
    "civil_interpretations",
    "civil_cases",
    "civil_elements",
    "civil_evidence",
    "civil_processes",
    "civil_questions",
    "civil_citations",
)

FIELD_LIMITS = {
    "civil_code_articles": {"id": 128, "law_name": 512, "chapter": 1024, "article_content": 65535},
    "civil_interpretations": {"id": 128, "law_name": 512, "part_name": 256, "document_number": 128, "content": 65535},
    "civil_cases": {"case_id": 128, "title": 512, "case_number": 128, "summary": 65535},
    "civil_elements": {"serial_number": 128, "title": 512, "case_number": 128, "case_summary": 65535},
    "civil_evidence": {"evidence_id": 128, "source_case_id": 128, "title": 512, "rule_name": 512, "content": 65535},
    "civil_processes": {"process_id": 128, "title": 512, "content": 65535},
    "civil_questions": {"question_id": 128, "question": 65535, "answer": 65535, "content": 65535},
    "civil_citations": {"citation_id": 128, "source_collection": 128, "source_id": 128, "target_collection": 128, "target_id": 128, "relation_type": 64, "content": 65535},
}

ALLOWED_FIELDS = {
    "civil_code_articles": {"id", "embedding", "article_number", "law_name", "chapter", "article_content"},
    "civil_interpretations": {"id", "embedding", "law_name", "part_name", "document_number", "article_count", "content"},
    "civil_cases": {"case_id", "embedding", "title", "case_number", "summary"},
    "civil_elements": {"serial_number", "embedding", "title", "case_number", "case_summary"},
    "civil_evidence": {"evidence_id", "embedding", "source_case_id", "title", "rule_name", "content"},
    "civil_processes": {"process_id", "embedding", "title", "content"},
    "civil_questions": {"question_id", "embedding", "question", "answer", "content"},
    "civil_citations": {"citation_id", "embedding", "source_collection", "source_id", "target_collection", "target_id", "article_number", "relation_type", "content"},
}

INTEGER_FIELDS = {
    "article_number",
    "article_count",
}


def normalize_row(collection: str, source: dict) -> dict:
    row = {key: value for key, value in source.items() if key in ALLOWED_FIELDS[collection]}
    for field, limit in FIELD_LIMITS[collection].items():
        value = row.get(field)
        if isinstance(value, str) and len(value.encode("utf-8")) > limit:
            row[field] = value.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
    for field in ALLOWED_FIELDS[collection] - {"embedding"}:
        if field not in row or row[field] is None:
            row[field] = 0 if field in INTEGER_FIELDS else ""
    return row


def main() -> None:
    processed = Path("/app/data/processed")
    collections = {}
    for name in COLLECTIONS:
        path = processed / f"{name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"{path} 不是数组")
        collections[name] = [normalize_row(name, row) for row in payload if isinstance(row, dict)]
        print(f"loaded {name}: {len(payload)}", flush=True)

    settings = get_settings()
    model = ModelGateway(settings)
    store = MilvusStore(settings)
    result = {}
    for collection, rows in collections.items():
        if store.public.has_collection(collection):
            store.public.drop_collection(collection)
        store.ensure_public_collection(collection, dimension=settings.embedding_dim)
        inserted = 0
        for batch in make_public_batches(collection, rows, max_records=16, max_characters=30000):
            payload = embed_public_batch(
                collection,
                batch,
                model,
                expected_dimension=settings.embedding_dim,
                retry_delay=2,
            )
            payload = [normalize_row(collection, row) for row in payload]
            store.public.insert(collection, payload, timeout=120)
            inserted += len(payload)
            print(f"embedded {collection}: {inserted}/{len(rows)}", flush=True)
        store.public.flush(collection, timeout=120)
        count = int((store.public.get_collection_stats(collection) or {}).get("row_count", 0))
        if count != len(rows):
            raise RuntimeError(f"{collection} 行数不一致：应为 {len(rows)}，实际为 {count}")
        result[collection] = count
        print(f"completed {collection}: {count}", flush=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
