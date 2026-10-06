# -*- coding: utf-8 -*-
"""
建立「含表格」的知识库索引
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

本模块把工单01/02 的「纯文本索引」升级为「文本块 + 表格块」双类型索引：

    文本层：rag_core.pdf_parse.parse_pdf(with_tables=False)   -> PageBlock(text)
    表格层：table_extractor.extract_all()  质量校验 + 跨页合并 -> PageBlock(table, Markdown)
                    ↓ 合并后交给共享分块器
    rag_core.chunk.chunk_blocks(strategy="structure")
                    ↓ 表格块走 _block_to_single_chunk，整表不切碎
    向量库 VectorStore(collection) + BM25 倒排索引

另外提供两个检索侧增强（不改 rag_core，只在调用侧包装）：
  1. boost_table_docs()  表格块优先级提升 —— 数值/清单类问题命中表格时加权；
  2. 对 TF-IDF 重排器补一次 fit()，让它真正用上全量语料的 IDF。

用法：
    # 含表格索引（工单03 主方案）
    python build_index_with_tables.py
    # 不含表格索引（消融实验对照组）
    python build_index_with_tables.py --no-tables --collection wo03_no_table
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_core import config, generator       # noqa: E402
from rag_core.bm25 import BM25Retriever      # noqa: E402
from rag_core.chunk import chunk_blocks      # noqa: E402
from rag_core.pdf_parse import parse_pdf     # noqa: E402
from rag_core.retriever import Retriever     # noqa: E402
from rag_core.vectorstore import VectorStore, backend as vs_backend  # noqa: E402

# 本工单模块（同目录直接 import）
sys.path.insert(0, str(Path(__file__).resolve().parent))
import table_extractor as te                 # noqa: E402

WO03_DIR = ROOT / "工单03-表格解析及检索优化"
RESULTS_DIR = WO03_DIR / "results"

# 两个对照索引的集合名（消融实验用）
COLLECTION_WITH_TABLES = "wo03_with_table"
COLLECTION_NO_TABLES = "wo03_no_table"

# 表格块优先级提升系数
TABLE_BOOST_DEFAULT = 1.15       # 一般问题：轻微提升
TABLE_BOOST_NUMERIC = 1.35       # 数值/清单类问题：明显提升
# 触发强提升的问法特征（招股书问题里大量出现）
_NUMERIC_CUES = ("多少", "比例", "占比", "哪些", "分别", "金额", "几", "列表", "投资项目")


# ---------------------------------------------------------------------------
# 表格记录缓存（pdfplumber 解析慢，消融实验要跑两遍索引，必须缓存）
# ---------------------------------------------------------------------------
def load_table_records(pdf_path: Path, doc_name: str,
                       out_dir: Path | None = None,
                       use_cache: bool = True,
                       verbose: bool = True) -> list[te.TableRecord]:
    """抽取（或从缓存读取）一份 PDF 的表格记录。"""
    out_dir = Path(out_dir or RESULTS_DIR)
    cache = out_dir / "tables" / f"{doc_name}_records.json"
    if use_cache and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if verbose:
            print(f"    [缓存] 表格记录 {len(data)} 张 <- {cache.name}")
        return [te.TableRecord(**d) for d in data]

    records, events = te.extract_all(pdf_path, doc_name, out_dir=out_dir,
                                     verbose=verbose)
    te.save_inventory(records, events, out_dir / "table_inventory.json", doc_name)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps([r.to_dict() for r in records], ensure_ascii=False, indent=1),
        encoding="utf-8")
    return records


# ---------------------------------------------------------------------------
# 建索引
# ---------------------------------------------------------------------------
def build_index(pdf_paths: list[Path],
                doc_names: list[str] | None = None,
                collection: str = COLLECTION_WITH_TABLES,
                with_tables: bool = True,
                out_dir: Path | None = None,
                use_enhanced_tables: bool = True,
                use_cache: bool = True,
                verbose: bool = True) -> dict:
    """
    建立索引（向量库 + BM25），返回索引统计信息。

    Args:
        with_tables: 是否把表格块并入索引（工单03 的核心开关，
                     消融实验中 False 即「不含表格」对照组）
        use_enhanced_tables: True = 用 table_extractor 的增强表格
                             （质量校验 + 列对齐修复 + 跨页续表合并）；
                             False = 退回 rag_core.parse_pdf(with_tables=True)
                             的原始表格，用于对比「表格质量」的价值
    """
    doc_names = doc_names or [Path(p).stem for p in pdf_paths]
    out_dir = Path(out_dir or RESULTS_DIR)
    t0 = time.perf_counter()

    all_chunks, doc_infos, table_records = [], [], []
    n_text_blocks = 0

    for path, name in zip(pdf_paths, doc_names):
        path = Path(path)
        t = time.perf_counter()
        if verbose:
            print(f"[1/3] 解析《{name}》（with_tables={with_tables}）…")

        # ---- 文本层 ----
        parsed = parse_pdf(path, name, with_tables=False, with_images=False)
        blocks = list(parsed.blocks)
        n_text_blocks += len(blocks)

        # ---- 表格层 ----
        n_tb = 0
        if with_tables:
            if use_enhanced_tables:
                records = load_table_records(path, name, out_dir=out_dir,
                                             use_cache=use_cache, verbose=verbose)
                table_records.extend(records)
                blocks += te.records_to_page_blocks(records)
                n_tb = len(records)
            else:
                parsed_tb = parse_pdf(path, name, with_tables=True, with_images=False)
                tb_blocks = [b for b in parsed_tb.blocks if b.type == "table"]
                blocks += tb_blocks
                n_tb = len(tb_blocks)

        # 与 parse_pdf 内部一致的排序：按页，同页 文本 -> 表格
        blocks.sort(key=lambda b: (b.page, {"text": 0, "table": 1, "image": 2}[b.type]))

        chunks = chunk_blocks(blocks, strategy="structure",
                              size=config.CHUNK_SIZE,
                              overlap=config.CHUNK_OVERLAP)
        all_chunks.extend(chunks)
        doc_infos.append({
            "name": name, "pages": parsed.n_pages,
            "text_blocks": len(parsed.blocks), "table_blocks": n_tb,
            "chunks": len(chunks),
            "table_chunks": sum(1 for c in chunks if c.type == "table"),
        })
        if verbose:
            print(f"      {parsed.n_pages} 页 / 文本块 {len(parsed.blocks)} / "
                  f"表格块 {n_tb} → {len(chunks)} 个 chunk "
                  f"({time.perf_counter() - t:.1f}s)")

    # ---- 写向量库 ----
    if verbose:
        print(f"[2/3] 写入向量库（后端 {vs_backend()}）…")
    VectorStore(collection).reset()
    vs = VectorStore(collection)
    vs.add_chunks(all_chunks, show_progress=verbose)

    # ---- 写 BM25 ----
    if verbose:
        print("[3/3] 构建 BM25 倒排索引…")
    bm25 = BM25Retriever()
    bm25.build(all_chunks)
    bm25_path = bm25_path_of(collection)
    bm25.save(bm25_path)

    stats = {
        "collection": collection,
        "with_tables": with_tables,
        "use_enhanced_tables": use_enhanced_tables,
        "vector_backend": vs_backend(),
        "embed_model": config.EMBED_MODEL_NAME,
        "chunk_strategy": "structure",
        "chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP,
        "docs": doc_infos,
        "n_text_blocks": n_text_blocks,
        "n_table_records": len(table_records),
        "n_merged_tables": sum(1 for r in table_records if r.merged_pages),
        "n_chunks": len(all_chunks),
        "chunk_types": _count_types(all_chunks),
        "n_vectors": vs.count(),
        "n_bm25_docs": len(bm25.doc_ids),
        "bm25_path": str(bm25_path),
        "top_table_chunks": [
            {"chunk_id": c.chunk_id, "doc": c.doc, "page": c.page,
             "rows": c.meta.get("rows"), "cols": c.meta.get("cols"),
             "caption": c.meta.get("caption", ""),
             "char_len": len(c.text)}
            for c in all_chunks if c.type == "table"
        ][:50],
        "build_seconds": round(time.perf_counter() - t0, 2),
    }

    out = out_dir / "index_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"完成：{len(all_chunks)} 个 chunk（表格 {stats['chunk_types'].get('table', 0)} 个），"
              f"向量 {vs.count()} 条，耗时 {stats['build_seconds']}s → {out}")
    return stats


def bm25_path_of(collection: str) -> Path:
    """每个集合一份独立的 BM25 索引文件（避免与其它工单/集合互相覆盖）。"""
    return config.INDEX_DIR / f"{collection}.bm25.pkl"


def _count_types(chunks) -> dict:
    out: dict = {}
    for c in chunks:
        out[c.type] = out.get(c.type, 0) + 1
    return out


# ---------------------------------------------------------------------------
# 检索侧：装载 + 表格块优先级提升
# ---------------------------------------------------------------------------
def load_retriever(collection: str, fit_reranker: bool = True) -> Retriever:
    """
    装载指定集合的检索器（向量库 + BM25）。
    fit_reranker: 给 TF-IDF 重排器补一次 fit，让它拿到全量语料 IDF
                  （rag_core 只在级联重排时自动 fit，单用 tfidf 时需要调用方补）。
    """
    r = Retriever(collection, bm25_path=bm25_path_of(collection))
    r.load_bm25()
    if fit_reranker:
        try:
            r.get_reranker("tfidf").fit(r.bm25.docs)
        except Exception as e:                       # 重排器不可用时不影响主流程
            print(f"  [warn] TF-IDF 重排器 fit 失败，将退化为普通 TF 排序：{e}")
    return r


def hydrate_docs(docs: list[dict], retriever: Retriever) -> list[dict]:
    """
    用 BM25 索引里的完整 chunk 记录补全向量检索结果的元数据。

    为什么需要：rag_core 的向量库只存 chunk_id/doc/page/type/section 五个字段
    （见 VectorStore.add_chunks 的 metas），表格块特有的 caption / rows / cols
    会丢失，导致界面与报告里无法显示「这是哪张表」。
    而 BM25 索引里保存的是 Chunk.to_dict() 的完整记录（含 meta 展开），
    因此按 chunk_id 回填一次即可，成本是 O(1) 的字典查找。

    只在调用侧做，不修改 rag_core。
    """
    try:
        full = {d.get("chunk_id"): d for d in retriever.bm25.docs}
    except Exception:
        return docs
    out = []
    for d in docs:
        meta = full.get(d.get("chunk_id"))
        if meta:
            merged = {**meta, **{k: v for k, v in d.items() if v is not None}}
            merged["score"] = d.get("score", merged.get("score"))
            if "final_score" in d:
                merged["final_score"] = d["final_score"]
            out.append(merged)
        else:
            out.append(d)
    return out


def table_boost_of(question: str) -> float:
    """按问题类型决定表格块提升系数：数值/清单类问题提升更明显。"""
    return TABLE_BOOST_NUMERIC if any(c in question for c in _NUMERIC_CUES) \
        else TABLE_BOOST_DEFAULT


def boost_table_docs(docs: list[dict], question: str = "",
                     boost: float | None = None) -> list[dict]:
    """
    表格块优先级提升（工单03 关键技术点之一）。

    动机：招股书的表格 chunk 文本里密密麻麻全是数字与专有名词，
    与问题的字面重合度天然低于「整段叙述性文字」，纯向量/BM25 排序时
    常被泛泛而谈的正文压到后面。而 id 1~4 这类问题的答案恰恰只在表里。

    做法：对 type=="table" 的片段，在 final_score 上乘一个 >1 的系数
    （数值/清单类问题 1.35，其它 1.15），再做一次稳定排序。
    只在调用侧包装，不改动 rag_core 的检索实现。
    """
    boost = boost if boost is not None else table_boost_of(question)
    out = []
    for d in docs:
        nd = dict(d)
        base = float(nd.get("final_score", nd.get("score", 0.0)) or 0.0)
        if nd.get("type") == "table":
            nd["table_boost"] = boost
            nd["final_score"] = base * boost
        else:
            nd["final_score"] = base
        out.append(nd)
    out.sort(key=lambda x: -float(x.get("final_score", 0.0)))
    return out


# ---------------------------------------------------------------------------
# 检索 + 生成（带离线抽取式兜底，保证无 API Key 也能跑通全流程）
# ---------------------------------------------------------------------------
def answer_with_tables(question: str,
                       retriever: Retriever,
                       top_k: int = 5,
                       recall_k: int = config.TOP_K_RECALL,
                       strategy: str = "vector",
                       reranker: str = "tfidf",
                       use_table_boost: bool = True,
                       use_llm: bool = True,
                       return_trace: bool = False) -> dict:
    """
    单问完整流程：检索 → 表格提升 → 拼上下文 → 生成。

    Returns:
        {question, answer, mode, docs, timings, context}
    """
    t_all = time.perf_counter()

    t = time.perf_counter()
    res = retriever.retrieve(question, strategy=strategy, top_k=top_k,
                             recall_k=recall_k, reranker=reranker)
    timings = dict(res.timings)
    timings["retrieve"] = time.perf_counter() - t
    docs = hydrate_docs(res.docs, retriever)     # 回填 caption/rows/cols 等表格元数据

    if use_table_boost:
        docs = boost_table_docs(docs, question)

    ctx = retriever.format_context(docs)

    answer, mode, refused = "", "rag", False
    if use_llm:
        try:
            gen = generator.generate_rag(question, docs, ctx)
            answer, refused, mode = gen.answer, gen.refused, "rag"
            timings["generate"] = gen.latency
        except Exception as e:                       # 无 Key / 网络异常 -> 抽取式兜底
            answer = extractive_answer(question, docs)
            mode = f"extractive(离线兜底：{type(e).__name__})"
    else:
        answer = extractive_answer(question, docs)
        mode = "extractive(--no-llm)"

    timings["total"] = time.perf_counter() - t_all
    out = {
        "question": question, "answer": answer, "mode": mode,
        "refused": refused, "docs": docs, "context": ctx,
        "timings": {k: round(v, 4) for k, v in timings.items()},
    }
    return out


def extractive_answer(question: str, docs: list[dict], max_chars: int = 1200) -> str:
    """
    离线抽取式作答（无 LLM 时的兜底，保证脚本在无网/无 Key 环境仍可跑完整流程）。

    策略：命中片段若是表格，则回填整张 Markdown 表（行与行之间的对应关系完整保留），
    若是正文则回填该段原文；并标注来源页，方便人工核对。
    """
    if not docs:
        return "根据提供的文档内容，未能找到该问题的答案。"
    parts = []
    for i, d in enumerate(docs[:3], 1):
        dt = d.get("doc", ""); pg = d.get("page", "?")
        text = (d.get("text") or "").strip()
        if d.get("type") == "table":
            parts.append(f"[片段{i}]《{dt}》第{pg}页（表格原样回填）：\n{text[:max_chars]}")
        else:
            parts.append(f"[片段{i}]《{dt}》第{pg}页：{text[:max_chars]}")
    return "（离线抽取式作答，未调用生成模型）\n\n" + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="工单03 建立含表格的知识库索引（文本块 + 表格块）")
    ap.add_argument("--collection", default=COLLECTION_WITH_TABLES)
    ap.add_argument("--pdf", action="append", default=None,
                    help="可多次指定；默认《招股说明书1》《招股说明书2》")
    ap.add_argument("--no-tables", action="store_true",
                    help="不含表格（消融实验对照组）")
    ap.add_argument("--raw-tables", action="store_true",
                    help="用 rag_core 原始表格，不做质量校验/跨页合并（对比表格质量）")
    ap.add_argument("--out", default=str(RESULTS_DIR))
    ap.add_argument("--no-cache", action="store_true", help="忽略表格记录缓存重新抽取")
    args = ap.parse_args()

    pdfs = [Path(p) for p in args.pdf] if args.pdf else \
        [config.PDF_PROSPECTUS_1, config.PDF_PROSPECTUS_2]
    names = [p.stem for p in pdfs]
    for p in pdfs:
        if not p.exists():
            print(f"[error] 找不到 PDF：{p}")
            sys.exit(1)

    build_index(
        pdfs, names,
        collection=args.collection,
        with_tables=not args.no_tables,
        out_dir=Path(args.out),
        use_enhanced_tables=not args.raw_tables,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
