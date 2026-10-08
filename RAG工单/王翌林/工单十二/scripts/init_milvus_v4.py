# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
scripts/init_milvus_v4.py —— 工单四 rag_images 初始化与图像入库脚本（新增文件）

用法：
  python scripts/init_milvus_v4.py                 # 建表 + 统计 + 演示检索
  python scripts/init_milvus_v4.py --ingest        # 解析两册 parsed JSON → 全量入库
  python scripts/init_milvus_v4.py --rebuild-招股说明书2   # 重建单册
说明：不影响 rag_chunks / rag_tables（独立 collection 与 Lite 降级库）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"
DOCS = ["招股说明书1", "招股说明书2"]
PARSED_DIR = "data/image_descriptions"


def ingest_doc(store, embedder, clipper, doc_name: str) -> int:
    """工单四：单册 parsed JSON → rag_images 入库（缺 parsed 则跳过并提示）"""
    parsed_path = Path(PARSED_DIR) / f"{doc_name}_images_parsed.json"
    if not parsed_path.exists():
        print(f"[ingest] {doc_name} 缺少解析结果 {parsed_path}，跳过（先跑 image_parser）")
        return 0
    data = json.loads(parsed_path.read_text(encoding="utf-8"))
    from src.image_parser.image_embedding import build_image_text
    records = []
    for img in data.get("images", []):
        text = build_image_text(img.get("caption", ""), img.get("ocr_text", ""),
                                json.dumps(img.get("vqa_qa", []), ensure_ascii=False))
        records.append({
            "doc_id": doc_name, "image_id": img["image_id"],
            "page": img.get("page", 0), "path": img["path"],
            "caption": img.get("caption", ""), "ocr_text": img.get("ocr_text", ""),
            "vqa_text": "\n".join(f"问:{q['q']} 答:{q['a']}"
                                  for q in img.get("vqa_qa", []) if q.get("a")),
            "embedding": embedder.embed_text(text) if text else None,
            "clip_embedding": clipper.embed_image_file(img["path"]),
            "metadata": {"page": img.get("page"), "width": img.get("width"),
                         "height": img.get("height"),
                         "extract_source": img.get("extract_source"),
                         "work_order": WORK_ORDER},
        })
    # 工单四：embedding 为 None 的记录用零向量占位（仍可走 CLIP 通道）
    for r in records:
        if r["embedding"] is None:
            r["embedding"] = [0.0] * 1024
    n = store.insert(records)
    print(f"[ingest] {doc_name}: 入库 {n} 条")
    return n


def main() -> None:
    """工单四：建表 + 可选入库 + 统计 + 演示检索（组织结构图 销售部 top5）"""
    parser = argparse.ArgumentParser(
        description="工单四 rag_images 初始化（人工智能NLP-RAG-图像内容解析及检索优化）")
    parser.add_argument("--ingest", action="store_true", help="解析结果全量入库")
    parser.add_argument("--recreate", action="store_true",
                        help="先删除旧 collection 再建表（schema 变更时用）")
    parser.add_argument("--rebuild", nargs="*", default=None,
                        help="重建指定 doc（先删后插），缺省全部")
    args = parser.parse_args()

    from src.image_parser.image_store import ImageStore
    store = ImageStore()
    if args.recreate and store.client.has_collection(store.collection):
        store.client.drop_collection(store.collection)   # 工单四：schema 变更后重建
        print("[init] 已删除旧 collection")
    store.ensure_collection()
    print(f"[init] collection={store.collection} rows={store.count()}")

    if args.ingest:
        from src.image_parser.image_embedding import (ImageClipEmbedder,
                                                      ImageTextEmbedder)
        embedder, clipper = ImageTextEmbedder(), ImageClipEmbedder()
        docs = args.rebuild or DOCS
        for d in docs:
            if args.rebuild:
                store.delete_by_doc_id(d)          # 工单四：重建先删旧数据
            ingest_doc(store, embedder, clipper, d)
        print(f"[init] 入库完成 rows={store.count()}")

    # 工单四：验收演示——"组织结构图 销售部" top5
    demo = "组织结构图 销售部"
    try:
        from src.image_parser.image_retriever import ImageRetriever
        hits = ImageRetriever(store=store).retrieve(demo, top_k=5)
        print(f"\n=== 检索演示 '{demo}' top5 ===")
        for i, h in enumerate(hits, 1):
            print(f"{i}. [{h['image_id']}] doc={h['doc_id']} p{h['page']} "
                  f"score={h['final_score']} kw={h['kw_hits']}")
            print(f"   path: {h['path']}")
            print(f"   caption: {h.get('caption', '')[:60]}")
    except Exception as e:
        print(f"[demo] 检索演示跳过（{e}）")      # 工单四：空库/无数据时容错


if __name__ == "__main__":
    main()
