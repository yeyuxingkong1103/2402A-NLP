"""清理旧 fallback 数据并重新用 MinerU 入库 SF6 标准 PDF。

- 删除文档 19365c61 的 state.json 记录（document/task/chunks）与 Qdrant 向量点
- 为同一 PDF 新建 document/task 并跑完整流水线（MinerU 解析 → 分块 → 嵌入 → 入库）
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OLD_DOC_ID = "19365c61-0acd-4707-819a-21f3a13208a3"
PDF = Path(r"C:\Users\tirito\Downloads\GB_T_44653-2024_六氟化硫气体现场循环再利用导则.pdf")
STATE_FILE = Path("data/state.json")


def main() -> None:
    from backend.app.config import AppSettings
    from backend.app.embeddings import BgeM3Embedder
    from backend.app.mineru import MinerUParser
    from backend.app.models import BuildTask, Document
    from backend.app.pipeline import BuildPipeline
    from backend.app.storage import JsonStateStore
    from backend.app.vector_store import QdrantVectorStore

    settings = AppSettings()

    if not PDF.exists():
        print(f"!! PDF 不存在: {PDF}")
        sys.exit(1)

    # 0. 备份
    backup = STATE_FILE.with_suffix(".json.bak")
    shutil.copy2(STATE_FILE, backup)
    print(f"已备份 state.json -> {backup}")

    store = JsonStateStore(STATE_FILE)
    qdrant = QdrantVectorStore(settings.qdrant_path, settings.qdrant_collection)

    # 1. 收集旧文档的 chunk_ids
    old_chunks = [c for c in store.list_chunks() if c.document_id == OLD_DOC_ID]
    old_chunk_ids = [c.chunk_id for c in old_chunks]
    print(f"旧文档 chunks: {len(old_chunk_ids)}")

    # 2. 删除 state.json 里的记录
    data = store._read()
    removed_docs = data["documents"].pop(OLD_DOC_ID, None)
    removed_tasks = {
        tid: t for tid, t in data["tasks"].items() if t.get("document_id") == OLD_DOC_ID
    }
    for tid in removed_tasks:
        data["tasks"].pop(tid, None)
    removed_chunks = {
        cid: c for cid, c in data["chunks"].items() if c.get("document_id") == OLD_DOC_ID
    }
    for cid in removed_chunks:
        data["chunks"].pop(cid, None)
    store._write(data)
    print(f"state.json 已清理: doc={bool(removed_docs)} tasks={len(removed_tasks)} chunks={len(removed_chunks)}")

    # 3. 删除 Qdrant 向量点
    if old_chunk_ids:
        qdrant.client.delete(collection_name="rag_documents", points_selector=old_chunk_ids)
        print(f"Qdrant 已删除 {len(old_chunk_ids)} 个点")
    else:
        # 兜底：按 document_id payload 过滤删除
        from qdrant_client import models
        qdrant.client.delete(
            collection_name="rag_documents",
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=OLD_DOC_ID))]
                )
            ),
        )
        print("Qdrant 按 document_id 过滤删除完成")

    # 4. 新建 document/task 并跑流水线
    document = Document.new(file_name=PDF.name, file_path=str(PDF))
    task = BuildTask.new(document.document_id)
    store.save_document(document)
    store.save_task(task)

    parser = MinerUParser()
    embedder = BgeM3Embedder(settings.bge_m3_model_path)
    pipeline = BuildPipeline(store, parser, embedder, qdrant)

    print("\n=== 重新入库 ===")
    t0 = time.time()
    pipeline.run(task.task_id)
    print(f"pipeline done in {time.time() - t0:.1f}s")

    task_after = store.get_task(task.task_id)
    print(f"task.status={task_after.status} error={task_after.error_message!r}")
    chunks = store.list_chunks(document.document_id)
    spans = [c.source_span for c in chunks]
    fb = sum(1 for s in spans if "fallback" in s)
    print(f"新文档 chunks: {len(chunks)}, fallback: {fb}")
    print(f"新 document_id: {document.document_id}")

    points, _ = qdrant.client.scroll(collection_name="rag_documents", limit=10000, with_payload=True)
    print(f"Qdrant 总点数: {len(points)}")
    docs = {}
    for p in points:
        did = (p.payload or {}).get("document_id")
        docs[did] = docs.get(did, 0) + 1
    print(f"Qdrant 按文档分布: {docs}")


if __name__ == "__main__":
    main()
