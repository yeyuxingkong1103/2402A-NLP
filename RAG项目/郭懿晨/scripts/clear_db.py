"""清空数据库：state.json 与 Qdrant rag_documents 的所有点。"""
import json
from pathlib import Path

from qdrant_client import QdrantClient


def main() -> None:
    state_path = Path("data/state.json")
    state_path.write_text(
        json.dumps({"documents": {}, "tasks": {}, "chunks": {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("state.json 已重置为空")

    client = QdrantClient(path=str(Path("data/qdrant")))
    collections = [c.name for c in client.get_collections().collections]
    for col in collections:
        points, _ = client.scroll(collection_name=col, limit=100000, with_payload=False)
        ids = [p.id for p in points]
        if ids:
            client.delete(collection_name=col, points_selector=ids)
        print(f"qdrant {col}: 删除 {len(ids)} 个点")


if __name__ == "__main__":
    main()
