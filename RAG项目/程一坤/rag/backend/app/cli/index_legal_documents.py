import argparse
import json
import os
import sys
from pathlib import Path

from sqlalchemy import text

from app.db import sql_models  # noqa: F401
from app.db.base import Base
from app.db.engine import create_database_engine, create_session_factory
from app.db.vector_index_service import index_awaiting_embeddings
from app.models.embedding import SiliconFlowEmbeddingClient


class FakeEmbeddingClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 0.0, 1.0] for index, _ in enumerate(texts)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def existing_chunk_keys(self, chunk_keys: list[str]) -> set[str]:
        # 返回 fake 向量库中已经存在的候选 chunk key，模拟 Milvus 查询。
        existing_keys = {row["chunk_key"] for row in self.rows}
        return existing_keys.intersection(chunk_keys)

    def upsert(self, rows: list[dict]) -> int:
        self.rows.extend(rows)
        return len(rows)

    def get_all_chunk_keys(self) -> set[str]:
        return {row["chunk_key"] for row in self.rows}

    def delete_by_chunk_keys(self, chunk_keys: list[str]) -> int:
        keys_to_delete = set(chunk_keys)
        before_count = len(self.rows)
        self.rows = [row for row in self.rows if row["chunk_key"] not in keys_to_delete]
        return before_count - len(self.rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="索引 awaiting_embedding 法律文档到 Milvus。")
    parser.add_argument("--limit", type=int, default=100, help="本次最多处理的版本数")
    # 控制单次 Embedding 请求包含的 chunk 数，降低请求超时概率。
    parser.add_argument("--embedding-batch-size", type=int, default=32, help="每次 Embedding 请求的 chunk 数")
    parser.add_argument("--database-url", help="数据库连接串；未提供时读取 DATABASE_URL")
    parser.add_argument("--env-file", help="可选 dotenv 文件路径")
    parser.add_argument("--fake-services", action="store_true", help="仅用于测试，不连接外部 Embedding 或 Milvus")
    parser.add_argument(
        "--prune-orphans",
        action="store_true",
        help="清理孤儿向量：删除 Milvus 中存在但 MySQL document_chunks 中不存在的向量",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="重建 Milvus collection（删除现有向量并重新创建 schema）",
    )
    args = parser.parse_args(argv)
    if args.env_file:
        load_env_file(Path(args.env_file))

    database_url = args.database_url or os.getenv("DATABASE_URL") or build_mysql_url_from_env()
    if not database_url:
        parser.exit(2, "database url is required\n")

    try:
        engine = create_database_engine(database_url)
        Base.metadata.create_all(engine)
        session_factory = create_session_factory(engine)
        vector_store = build_vector_store(args.fake_services)

        # 如果启用了重建 collection，先执行重建
        if args.recreate_collection:
            if not args.fake_services:
                print("警告：即将删除现有 Milvus collection 并重新创建", file=sys.stderr)
                vector_store.recreate_collection()
                print("Milvus collection 已重建", file=sys.stderr)
                # 重建后需要将所有版本标记为 awaiting_embedding
                with session_factory() as session:
                    session.execute(
                        text("UPDATE document_versions SET processing_status = 'awaiting_embedding' WHERE processing_status = 'indexed'")
                    )
                    session.commit()
                print("已将所有版本标记为待嵌入", file=sys.stderr)

        # 如果启用了清理孤儿向量，先执行清理
        if args.prune_orphans:
            from app.db.vector_index_service import prune_orphan_vectors
            with session_factory() as session:
                prune_result = prune_orphan_vectors(session, vector_store)
            _print_index_result(
                {
                    "status": "pruned",
                    "orphan_vectors_deleted": prune_result.deleted_count,
                    "orphan_keys_sample": prune_result.deleted_keys_sample,
                }
            )

        with session_factory() as session:
            result = index_awaiting_embeddings(
                session,
                embedding_client=build_embedding_client(args.fake_services),
                vector_store=vector_store,
                batch_size=args.limit,
                # 将 CLI 指定的 Embedding 批大小传递给索引服务。
                embedding_batch_size=args.embedding_batch_size,
            )
    except Exception as error:
        _print_index_result({"status": "failed", "stage": "index", "error": error.__class__.__name__, "error_summary": str(error)[:200]})
        return 1

    _print_index_result(
        {
            "status": result.status,
            "indexed_versions": result.indexed_versions,
            "indexed_chunks": result.indexed_chunks,
        }
    )
    return 0


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def build_mysql_url_from_env() -> str | None:
    required = ["MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_HOST", "MYSQL_PORT", "MYSQL_DATABASE"]
    if any(not os.getenv(key) for key in required):
        return None
    from urllib.parse import quote_plus

    return (
        f"mysql+pymysql://{quote_plus(os.environ['MYSQL_USER'])}:"
        f"{quote_plus(os.environ['MYSQL_PASSWORD'])}@"
        f"{os.environ['MYSQL_HOST']}:{os.environ['MYSQL_PORT']}/{os.environ['MYSQL_DATABASE']}"
    )


def build_embedding_client(fake_services: bool):
    if fake_services:
        return FakeEmbeddingClient()
    return SiliconFlowEmbeddingClient(
        api_url=os.environ["EMBEDDING_API_BASE_URL"],
        api_key=os.environ["EMBEDDING_API_KEY"],
        model=os.environ["EMBEDDING_MODEL"],
        dimension=int(os.environ["EMBEDDING_DIMENSION"]),
        timeout=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60")),
    )


def build_vector_store(fake_services: bool):
    if fake_services:
        return FakeVectorStore()
    from pymilvus import MilvusClient

    from app.db.milvus_store import LegalMilvusStore

    client = MilvusClient(uri=f"http://{os.environ['MILVUS_HOST']}:{os.environ['MILVUS_PORT']}")
    store = LegalMilvusStore(
        client=client,
        collection_name=os.getenv("MILVUS_COLLECTION_NAME", "legal_documents"),
        dimension=int(os.environ["EMBEDDING_DIMENSION"]),
    )
    store.ensure_collection()
    return store


def _print_index_result(payload: dict[str, str | int]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
