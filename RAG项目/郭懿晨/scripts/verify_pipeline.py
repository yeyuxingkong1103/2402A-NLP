"""端到端验证：MinerU 解析 → 分块 → 嵌入 → 入库，检查数据质量。"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PDF = Path(r"C:\Users\tirito\Downloads\GB_T_44653-2024_六氟化硫气体现场循环再利用导则.pdf")


def main() -> None:
    from backend.app.config import AppSettings
    from backend.app.mineru import MinerUParser
    from backend.app.embeddings import BgeM3Embedder
    from backend.app.vector_store import QdrantVectorStore
    from backend.app.storage import JsonStateStore
    from backend.app.models import Document, BuildTask, DocumentStatus
    from backend.app.pipeline import BuildPipeline

    settings = AppSettings()
    tmp = Path(tempfile.mkdtemp(prefix="pipeline-verify-"))
    store = JsonStateStore(tmp / "state.json")
    qdrant = QdrantVectorStore(tmp / "qdrant", "rag_documents")
    embedder = BgeM3Embedder(settings.bge_m3_model_path)

    document = Document.new(file_name=PDF.name, file_path=str(PDF))
    task = BuildTask.new(document.document_id)
    store.save_document(document)
    store.save_task(task)

    parser = MinerUParser()
    pipeline = BuildPipeline(store, parser, embedder, qdrant)

    print("=== 运行流水线 ===")
    t0 = time.time()
    pipeline.run(task.task_id)
    print(f"pipeline done in {time.time() - t0:.1f}s")

    task_after = store.get_task(task.task_id)
    doc_after = store.get_document(document.document_id)
    print(f"task.status={task_after.status} step={task_after.step} progress={task_after.progress}")
    print(f"task.error={task_after.error_message!r}")
    print(f"doc.status={doc_after.status}")

    chunks = store.list_chunks(document.document_id)
    print(f"\n=== chunks: {len(chunks)} ===")
    if chunks:
        spans = [c.source_span for c in chunks]
        fallback_count = sum(1 for s in spans if "fallback" in s)
        categories = {}
        for c in chunks:
            categories[c.category] = categories.get(c.category, 0) + 1
        print(f"fallback 块数: {fallback_count}/{len(chunks)}")
        print(f"类别分布: {categories}")
        print(f"前5 source_spans: {spans[:5]}")
        print(f"示例文本: {chunks[0].text[:80]!r}")

    print("\n=== qdrant 入库检查 ===")
    points, _ = qdrant.client.scroll(collection_name="rag_documents", limit=10000, with_payload=True, with_vectors=False)
    print(f"qdrant points: {len(points)}")
    if points:
        fb = sum(1 for p in points if "fallback" in str((p.payload or {}).get("source_span", "")))
        print(f"  fallback payloads: {fb}/{len(points)}")
        sample = points[0].payload
        print(f"  样例 payload: page={sample.get('page')} category={sample.get('category')} span={sample.get('source_span')}")
        print(f"  样例文本: {str(sample.get('text'))[:60]!r}")


if __name__ == "__main__":
    main()
