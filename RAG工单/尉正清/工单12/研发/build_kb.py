# 工单编号：人工智能NLP-RAG项目-LightRAG优化
"""建 RAG 向量知识库（对比的基线那一侧）

两份招股说明书建**一个**库而不是两个：测试问题横跨两家公司，而 RAG 那边
没有图结构做实体路由，分库就得先猜问题属于哪家 —— 那是另一套逻辑，会
把「检索能力」和「路由准确率」两件事混在一起。合一库 + 每条 chunk 带 doc
字段，报告里再按 doc 统计召回落在哪一份上。

与 LightRAG 那一路共用 `corpus.parse` 的解析结果（见 corpus.py 的说明）。

⚠️ **本脚本跑在 `rag_gd` 环境**（FlagEmbedding + BGE-M3 那一套）。
LightRAG/对比/RAGAS 跑在 `rag_gd1`。为什么要分两个环境见
优化/过程问题记录.md 问题 2。

用法：
    python build_kb.py                 # 全量（含图表解析，耗时主要在图表）
    python build_kb.py --no-images     # 跳过图表解析，快很多
    python build_kb.py --no-cache      # 忽略已有缓存重建
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import CACHE_ROOT                                  # noqa: E402
from corpus import parse                                       # noqa: E402
from prospectus_docs import available, label                    # noqa: E402
from vector_store import BGEM3VectorStore                      # noqa: E402

KB_NAME = "prospectus12"
MANIFEST = "docs.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true", help="忽略已有知识库，强制重建")
    ap.add_argument("--no-images", action="store_true",
                    help="跳过图表解析（第 5、6 题会因此答不出来）")
    args = ap.parse_args()

    docs = available()
    if not docs:
        print("[错误] 没找到招股说明书 PDF，检查 prospectus_docs.CORPUS_DIR")
        return 1

    out_dir = Path(CACHE_ROOT) / KB_NAME
    if out_dir.exists() and not args.no_cache:
        store = BGEM3VectorStore()
        if store.load(out_dir):
            print(f"[跳过] 知识库已存在且可用：{out_dir}（{len(store.chunks)} 块）。"
                  f"要重建请加 --no-cache")
            return 0

    cache_dir = Path(CACHE_ROOT) / "images"
    all_chunks = []
    for doc in docs:
        chunks, _ = parse(doc, with_images=not args.no_images, cache_dir=cache_dir)
        for c in chunks:
            c["doc"] = doc["key"]
            c["company"] = doc["company"]
        all_chunks += chunks

    print(f"[编码] 共 {len(all_chunks)} 块，交给 BGE-M3 ...", flush=True)
    t0 = time.time()
    store = BGEM3VectorStore()
    store.build(all_chunks)
    print(f"[编码] 完成，耗时 {time.time() - t0:.1f}s", flush=True)

    store.save(out_dir)
    (Path(out_dir) / MANIFEST).write_text(json.dumps({
        "name": KB_NAME,
        "embedding_model": store.model_name,
        "with_images": not args.no_images,
        "docs": [{"key": d["key"], "file": d["file"], "company": d["company"],
                  "pages": d["pages"]} for d in docs],
        "chunks": len(all_chunks),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 知识库已写入 {out_dir}（{len(all_chunks)} 块）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
