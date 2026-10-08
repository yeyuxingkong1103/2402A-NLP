# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
scripts/init_milvus_v3.py —— 工单三 Milvus 表格库初始化 + 批量入库

职责：
  1. 创建（或重建）rag_tables collection
  2. 扫描 data/tables/*.json，对每份表格 JSON：
     - 嵌入（若 JSON 内已有 embedding 则复用）
     - 删除该 doc_id 旧记录（幂等）
     - 批量 insert
  3. 打印 collection 统计
  4. 对"发行股数"做 demo 检索 top5

用法：
  python scripts/init_milvus_v3.py
  python scripts/init_milvus_v3.py --rebuild          # 删 collection 重建
  python scripts/init_milvus_v3.py --demo-query "发行股数"
"""
import argparse
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.table_parser.table_store import TableStore  # noqa: E402
from src.table_parser.table_embedding import (  # noqa: E402
    load_table_chunks_with_vectors,
)
from src.embedding import get_embedder  # noqa: E402


def init_and_ingest(
    tables_dir: Path,
    rebuild: bool = False,
) -> dict:
    """工单三：初始化 collection 并批量入库所有表格 JSON"""
    store = TableStore()
    if rebuild:
        store.drop_collection()
    store.ensure_collection()

    json_files = sorted(tables_dir.glob("*_tables.json"))
    if not json_files:
        logger.error(f"[init_milvus_v3] {tables_dir} 下无 *_tables.json")
        return {"ingested": [], "total": 0, "stats": store.get_stats()}

    logger.info(f"[init_milvus_v3] 扫描到 {len(json_files)} 份表格 JSON")
    ingested = []
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        doc_id = data.get("doc_name") or data.get("doc_id") or jf.stem
        chunks = data.get("table_texts") or []
        if not chunks:
            logger.warning(f"[init_milvus_v3] {jf.name} 无 table_texts，跳过")
            continue
        # 嵌入（JSON 内有则复用，否则现算）
        chunks_loaded, vecs = load_table_chunks_with_vectors(str(jf))
        for c in chunks_loaded:
            c.setdefault("doc_id", doc_id)
        # 幂等：先删旧
        store.delete_by_doc_id(doc_id)
        n = store.insert_tables(chunks_loaded, vecs)
        ingested.append({
            "doc_id": doc_id,
            "file": jf.name,
            "inserted": n,
            "table_count": len(chunks_loaded),
        })
        logger.info(f"[init_milvus_v3] {doc_id}: 入库 {n} 条")

    stats = store.get_stats()
    return {"ingested": ingested, "total": sum(d["inserted"] for d in ingested),
            "stats": stats, "store": store}


def demo_search(store: TableStore, query: str, top_k: int = 5) -> list:
    """工单三：对查询做向量检索，返回 top_k 表格"""
    embedder = get_embedder()
    vec = embedder.encode([query], show_progress_bar=False)[0]
    return store.search_tables(vec, top_k=top_k)


def _main() -> int:
    parser = argparse.ArgumentParser(description="工单三：Milvus 表格库初始化 CLI")
    parser.add_argument("--tables-dir", default=str(PROJECT_ROOT / "data" / "tables"),
                        help="表格 JSON 目录")
    parser.add_argument("--rebuild", action="store_true",
                        help="删 collection 重建（清空旧数据）")
    parser.add_argument("--demo-query", default="发行股数",
                        help="入库后做 demo 检索的查询（默认'发行股数'）")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    result = init_and_ingest(Path(args.tables_dir), rebuild=args.rebuild)
    print("\n========== 入库结果 ==========")
    print(json.dumps({
        "ingested": result["ingested"],
        "total_inserted": result["total"],
        "stats": result["stats"],
    }, ensure_ascii=False, indent=2))

    # demo 检索
    store: TableStore = result["store"]
    print(f"\n========== demo 检索: '{args.demo_query}' top{args.top_k} ==========")
    hits = demo_search(store, args.demo_query, top_k=args.top_k)
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] doc_id={h['doc_id']} table_id={h['table_id']} "
              f"page={h['page']} score={h['score']:.4f}")
        print(f"    metadata: caption={h['metadata'].get('caption','')}")
        text = (h.get("table_text") or "")[:200]
        print(f"    table_text: {text}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
