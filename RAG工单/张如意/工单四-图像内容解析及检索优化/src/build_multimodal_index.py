# -*- coding: utf-8 -*-
"""
多模态知识库构建（文本块 + 表格块 + 图像语义块 合并入库）
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

把工单01 的文本块、工单03 的表格块、工单04 的**图像语义块**统一分块后，
写入向量库（Chroma / NumPy）与 BM25 倒排索引，供检索问答使用。

两个索引（对比实验的两条腿）：
    wo04_multimodal —— 文本 + 表格 + 图像语义描述（工单04 完整版）
    wo04_text       —— 文本 + 表格（工单03 水平，不含图像解析）

关键实现点：
  1. 图像语义块来自 image_semantic_parse.py 的 retrieval_text，
     按 (文档, 页码) 回填到 parse_pdf 抽出的 image 块上；
  2. 图像块整体作为一个 chunk（chunk.py 对非 text 块不切分），
     保证层级树/表格不会被切碎；
  3. BM25 索引**必须按 collection 分文件保存**：
     rag_core.bm25.BM25Retriever.save() 默认写 data/index/bm25.pkl，
     两个索引共用会互相覆盖，这里显式传 bm25_path。

用法：
    python src/build_multimodal_index.py                  # 同时构建两个索引
    python src/build_multimodal_index.py --mode with_images
    python src/build_multimodal_index.py --docs 2         # 只索引招股说明书2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_extractor import WORKDIR, RESULTS_DIR, DOCS  # noqa: E402
from rag_core import config  # noqa: E402
from rag_core.bm25 import BM25Retriever  # noqa: E402
from rag_core.chunk import chunk_blocks  # noqa: E402
from rag_core.pdf_parse import parse_pdf  # noqa: E402
from rag_core.retriever import Retriever  # noqa: E402
from rag_core.vectorstore import VectorStore  # noqa: E402

# 两个索引的 collection 名（Chroma/NumPy 通用，仅含字母数字与下划线）
COLLECTIONS = {"with_images": "wo04_multimodal", "text_only": "wo04_text"}
INDEX_STATS_JSON = RESULTS_DIR / "index_stats.json"


def bm25_path(collection: str) -> Path:
    """BM25 索引按 collection 分文件，避免两个索引互相覆盖。"""
    return config.INDEX_DIR / f"bm25_{collection}.pkl"


def load_retriever(collection: str) -> Retriever:
    """装载指定 collection 的检索器（向量库 + 对应 BM25）。"""
    r = Retriever(collection, bm25_path=bm25_path(collection))
    r.load_bm25()
    return r


# ---------------------------------------------------------------------------
# 图像语义文本回填
# ---------------------------------------------------------------------------
def enrich_image_blocks(blocks, doc_name: str, desc_map: dict,
                        keep_undescribed: bool = False) -> tuple[list, int, int]:
    """
    把图像语义描述回填到 image 块的 content 字段。

    Args:
        desc_map: {(文档名, 页码): [检索文本, ...]}，来自 image_semantic_parse
        keep_undescribed: 无描述的图像块是否保留（默认丢弃，避免噪声块）

    Returns:
        (处理后的 blocks, 回填数, 丢弃数)
    """
    filled = dropped = 0
    out = []
    for b in blocks:
        if b.type != "image":
            out.append(b)
            continue
        texts = desc_map.get((doc_name, b.page)) or []
        if texts:
            # 同一页多张图：按出现顺序取第 k 条文本
            k = sum(1 for x in out if x.type == "image" and x.page == b.page)
            b.content = texts[k] if k < len(texts) else texts[-1]
            b.extra["semantic_parsed"] = True
            b.extra["texts_on_page"] = len(texts)
            filled += 1
            out.append(b)
        elif keep_undescribed:
            b.content = (f"[图像] 《{doc_name}》第{b.page}页包含一张图，"
                         f"未做多模态解析，图中信息不可检索。")
            b.extra["semantic_parsed"] = False
            out.append(b)
        else:
            dropped += 1
    return out, filled, dropped


# ---------------------------------------------------------------------------
# 建索引
# ---------------------------------------------------------------------------
def build_index(mode: str, doc_keys: list[str] | None = None,
                rebuild: bool = True, verbose: bool = True,
                keep_undescribed: bool = False) -> dict:
    """
    构建一个索引。

    Args:
        mode: "with_images"（多模态完整版）或 "text_only"（不含图像解析）
        doc_keys: 要索引的文档编号（"1"/"2"），默认两份都索引
                   —— 16 个问题里 10 个兴图问题需要《招股说明书1》
    """
    if mode not in COLLECTIONS:
        raise ValueError(f"未知模式 {mode}，可选 {list(COLLECTIONS)}")
    with_images = (mode == "with_images")
    collection = COLLECTIONS[mode]
    doc_keys = doc_keys or ["1", "2"]

    desc_map: dict = {}
    if with_images:
        try:
            from image_semantic_parse import text_by_page
            desc_map = text_by_page()
            if verbose:
                print(f"[图像语义] 载入 {sum(len(v) for v in desc_map.values())} "
                      f"段图像描述，覆盖 {len(desc_map)} 个 (文档,页) 组合")
        except FileNotFoundError as e:
            print(f"[warn] 未找到图像语义解析结果：{e}")
            print("       请先运行 python src/image_semantic_parse.py；"
                  "本次将退化为纯文本+表格索引")

    t0 = time.perf_counter()
    all_chunks = []
    doc_stats = []
    n_filled = n_dropped = 0

    for key in doc_keys:
        spec = DOCS[key]
        path = Path(spec["path"])
        if not path.exists():
            print(f"[warn] 跳过不存在的文档：{path}")
            continue
        t = time.perf_counter()
        if verbose:
            print(f"[解析] 《{spec['name']}》…")
        parsed = parse_pdf(path, spec["name"], with_tables=True,
                           with_images=with_images)
        blocks = parsed.blocks
        if with_images:
            blocks, f, d = enrich_image_blocks(
                blocks, spec["name"], desc_map, keep_undescribed)
            n_filled += f
            n_dropped += d
        chunks = chunk_blocks(blocks, strategy="structure",
                              size=config.CHUNK_SIZE,
                              overlap=config.CHUNK_OVERLAP)
        all_chunks.extend(chunks)
        n_img_chunks = sum(1 for c in chunks if c.type == "image")
        doc_stats.append({
            "name": spec["name"], "pages": parsed.n_pages,
            "blocks": len(parsed.blocks), "chunks": len(chunks),
            "image_chunks": n_img_chunks,
            "seconds": round(time.perf_counter() - t, 1),
        })
        if verbose:
            print(f"       {parsed.n_pages} 页 → {len(parsed.blocks)} 块 → "
                  f"{len(chunks)} chunks（图像块 {n_img_chunks}）"
                  f"  {time.perf_counter() - t:.1f}s")

    # --- 向量库 ---
    vs = VectorStore(collection)
    if rebuild:
        vs.reset()
    vs.add_chunks(all_chunks, show_progress=verbose)

    # --- BM25（按 collection 分文件，避免覆盖）---
    bm25 = BM25Retriever()
    bm25.build(all_chunks)
    bp = bm25.save(bm25_path(collection))

    def _count_types(chunks):
        out: dict[str, int] = {}
        for c in chunks:
            out[c.type] = out.get(c.type, 0) + 1
        return out

    stats = {
        "mode": mode,
        "collection": collection,
        "with_images": with_images,
        "docs": doc_stats,
        "n_chunks": len(all_chunks),
        "chunk_types": _count_types(all_chunks),
        "n_vectors": vs.count(),
        "n_bm25_docs": len(bm25.doc_ids),
        "bm25_path": str(bp),
        "image_blocks_filled": n_filled,
        "image_blocks_dropped": n_dropped,
        "build_seconds": round(time.perf_counter() - t0, 2),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 汇总落盘
    all_stats = {}
    if INDEX_STATS_JSON.exists():
        try:
            all_stats = json.loads(INDEX_STATS_JSON.read_text(encoding="utf-8"))
        except Exception:
            all_stats = {}
    all_stats[mode] = stats
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_STATS_JSON.write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        print(f"[完成] {collection}：{stats['n_chunks']} chunks"
              f"（图像块 {stats['chunk_types'].get('image', 0)}）｜"
              f"向量 {stats['n_vectors']} 条｜BM25 {stats['n_bm25_docs']} 篇｜"
              f"{stats['build_seconds']}s")
        print(f"        统计：{INDEX_STATS_JSON}")
    return stats


def ensure_index(mode: str, rebuild: bool = False, verbose: bool = True) -> str:
    """
    确保索引存在，返回 collection 名（评测/对比脚本调用）。
    已构建过且 rebuild=False 时直接复用，避免重复编码（BGE 编码很慢）。
    """
    collection = COLLECTIONS[mode]
    if not rebuild and INDEX_STATS_JSON.exists():
        try:
            stats = json.loads(INDEX_STATS_JSON.read_text(encoding="utf-8"))
            if mode in stats and VectorStore(collection).count() > 0:
                if verbose:
                    print(f"[复用] {collection} 索引已存在"
                          f"（{stats[mode]['n_chunks']} chunks，"
                          f"构建于 {stats[mode]['built_at']}）")
                return collection
        except Exception:
            pass
    build_index(mode, verbose=verbose)
    return collection


def main() -> None:
    ap = argparse.ArgumentParser(description="构建多模态知识库索引")
    ap.add_argument("--mode", default="both",
                    choices=["with_images", "text_only", "both"],
                    help="with_images=含图像语义；text_only=不含图像解析；both=两个都建")
    ap.add_argument("--docs", default="1,2", help="索引哪些文档，逗号分隔（1/2）")
    ap.add_argument("--keep-undescribed", action="store_true",
                    help="保留没有语义描述的图像块（默认丢弃）")
    args = ap.parse_args()

    doc_keys = [d.strip() for d in args.docs.split(",") if d.strip()]
    modes = ["with_images", "text_only"] if args.mode == "both" else [args.mode]
    for m in modes:
        print(f"\n{'=' * 66}\n构建索引：{m} → {COLLECTIONS[m]}\n{'=' * 66}")
        build_index(m, doc_keys=doc_keys, keep_undescribed=args.keep_undescribed)


if __name__ == "__main__":
    main()
