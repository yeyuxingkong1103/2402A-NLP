# -*- coding: utf-8 -*-
"""
工单06 索引构建：向量索引 + BM25 倒排索引
工单编号：人工智能NLP-RAG-混合检索任务

职责（工单「功能详细需求」的索引侧落地）：
  1. 解析《招股说明书1》（武汉兴图新科）与《招股说明书2》（武汉力源信息），
     采用 structure 章节结构感知分块（工单02 验证的最优策略），
     表格块整体保留（工单03 能力），保证金额/比例类问题可检索。
  2. 建立【多嵌入模型】向量索引（工单要求支持 bge、m3e 等多种模型）：
       bge-large-zh-v1.5 -> 集合 prospectus      （1024 维，默认模型）
       m3e-base          -> 集合 prospectus_m3e  （768 维，轻量对比模型）
     集合名与 rag_core.api / pipeline 的默认集合 "prospectus" 对齐，
     因此 Web 服务（serve.py / api.py）无需任何改动即可直接用上本索引。
  3. 建立 BM25 倒排索引（自研，不依赖 Elasticsearch），
     【多字段加权】：正文 text 权重 1.0、章节路径 section 权重 1.6。
     BM25 落盘到 rag_core 默认路径 data/index/bm25.pkl，全工单共享。

运行：
    python 工单06-混合检索/src/build_index.py            # 增量：缺什么建什么
    python 工单06-混合检索/src/build_index.py --rebuild  # 全量重建

产出：
    data/index/chroma/            向量库（chromadb 持久化）
    data/index/bm25.pkl           BM25 倒排索引
    工单06-混合检索/results/index_stats.json  索引统计（含多字段信息）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 让脚本能直接 import 项目根目录下的 rag_core
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag_core import config                                    # noqa: E402
from rag_core.bm25 import BM25Retriever, tokenize              # noqa: E402
from rag_core.chunk import Chunk, chunk_blocks                 # noqa: E402
from rag_core.pdf_parse import parse_pdf                       # noqa: E402
from rag_core.vectorstore import VectorStore                   # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STATS_PATH = RESULTS_DIR / "index_stats.json"

# 多字段权重（工单06 明确要求：正文 text 1.0 / 章节路径 section 1.6）
# 章节路径在 structure 分块时会以【第五节 … > 一、主营业务】形式注入正文头部，
# 权重 1.6 让「章节名里的关键词」获得更高 BM25 贡献。
FIELD_WEIGHTS = {"text": 1.0, "section": 1.6}

# 嵌入模型 -> 向量集合名（不同模型维度不同，必须分集合存放）
COLLECTIONS = {
    "bge-large-zh-v1.5": "prospectus",
    "m3e-base": "prospectus_m3e",
}
DEFAULT_MODEL = "bge-large-zh-v1.5"

# BM25 落盘路径（与 rag_core.bm25.BM25Retriever.save 的默认路径一致）
BM25_PATH = config.INDEX_DIR / "bm25.pkl"

# 待入库的两份招股说明书
PDFS = [
    (config.PDF_PROSPECTUS_1, "招股说明书1"),
    (config.PDF_PROSPECTUS_2, "招股说明书2"),
]


# ---------------------------------------------------------------------------
# 解析 + 分块
# ---------------------------------------------------------------------------
def load_chunks(verbose: bool = True) -> list[Chunk]:
    """解析两份招股说明书并按章节结构分块（表格入库，图像留给工单04）。"""
    missing = [str(p) for p, _ in PDFS if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "未找到招股说明书 PDF：\n  " + "\n  ".join(missing) +
            "\n请确认 rag_core/config.py 中 SRC_ATTACH 路径，或把 PDF 放到该目录。"
        )

    chunks: list[Chunk] = []
    for path, name in PDFS:
        t0 = time.perf_counter()
        if verbose:
            print(f"[解析] 《{name}》…")
        parsed = parse_pdf(path, name, with_tables=True, with_images=False)
        doc_chunks = chunk_blocks(parsed.blocks, strategy="structure",
                                  size=config.CHUNK_SIZE,
                                  overlap=config.CHUNK_OVERLAP)
        chunks.extend(doc_chunks)
        if verbose:
            print(f"       {parsed.n_pages} 页 -> {len(doc_chunks)} 块 "
                  f"({time.perf_counter() - t0:.1f}s)")
    return chunks


def _field_stats(chunks: list[Chunk]) -> dict:
    """
    统计多字段信息（工单要求索引统计包含「多字段」）。
    逐字段统计：参与索引的块数、总词数、去重词数、平均字段长度。
    """
    stats: dict[str, dict] = {}
    for field, weight in FIELD_WEIGHTS.items():
        n_present, total_tokens, terms = 0, 0, set()
        for c in chunks:
            val = getattr(c, field, "") or ""
            if not val.strip():
                continue
            toks = tokenize(val)
            n_present += 1
            total_tokens += len(toks)
            terms.update(toks)
        stats[field] = {
            "字段": "章节路径" if field == "section" else "正文",
            "权重": weight,
            "覆盖块数": n_present,
            "总词数": total_tokens,
            "去重词数": len(terms),
            "平均字段长度": round(total_tokens / n_present, 1) if n_present else 0.0,
        }
    return stats


# ---------------------------------------------------------------------------
# 建索引主流程
# ---------------------------------------------------------------------------
def build_all(models: list[str] | None = None,
              rebuild: bool = False,
              verbose: bool = True) -> dict:
    """
    建立全部索引并输出统计。

    Args:
        models:  需要建向量索引的嵌入模型列表，默认 bge + m3e
        rebuild: True 时先清空同名集合再建（改分块参数后必须重建）
    """
    from rag_core.embed import set_model

    models = models or list(COLLECTIONS)
    for m in models:
        if m not in COLLECTIONS:
            raise ValueError(f"未知嵌入模型：{m}，可选 {list(COLLECTIONS)}")

    t_all = time.perf_counter()
    chunks = load_chunks(verbose=verbose)
    if not chunks:
        raise RuntimeError("解析后没有得到任何文本块，请检查 PDF 解析结果。")

    # ---- 1. 向量索引（每个模型一个集合）----
    # 注意：每次构建前先 reset，保证重复运行不会追加出重复向量
    # （集合是「全量重建」语义，rebuild 参数保留给命令行语义兼容）。
    vector_counts: dict[str, int] = {}
    for model in models:
        t0 = time.perf_counter()
        set_model(model)                       # 切换嵌入模型（工单06 多模型要求）
        vs = VectorStore(COLLECTIONS[model])
        vs.reset()
        n = vs.add_chunks(chunks, show_progress=verbose)
        vector_counts[model] = n
        if verbose:
            print(f"[向量] {model} -> 集合 {COLLECTIONS[model]}：{n} 条 "
                  f"({time.perf_counter() - t0:.1f}s)")

    # ---- 2. BM25 倒排索引 ----
    t0 = time.perf_counter()
    bm25 = BM25Retriever(field_weights=FIELD_WEIGHTS)
    n_docs = bm25.build(chunks)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    bm25.save(BM25_PATH)
    if verbose:
        print(f"[BM25] 倒排索引：{n_docs} 篇文档 / {len(bm25.inverted)} 个词条，"
              f"avgdl={bm25.avgdl:.1f} ({time.perf_counter() - t0:.1f}s)")

    # ---- 3. 索引统计（含多字段信息）----
    type_count: dict[str, int] = {}
    for c in chunks:
        type_count[c.type] = type_count.get(c.type, 0) + 1

    stats = {
        "工单编号": "人工智能NLP-RAG-混合检索任务",
        "collection_map": COLLECTIONS,
        "bm25_path": str(BM25_PATH),
        "embed_models": models,
        "n_chunks": len(chunks),
        "n_vectors": vector_counts,
        "n_bm25_docs": n_docs,
        "n_bm25_terms": len(bm25.inverted),
        "bm25_avgdl": round(bm25.avgdl, 2),
        "bm25_params": {"k1": bm25.k1, "b": bm25.b},
        "chunk_types": type_count,
        "打分参数": {"k1": bm25.k1, "b": bm25.b,
                     "字段权重": FIELD_WEIGHTS},
        "docs": [{"name": name, "path": str(p)} for p, name in PDFS],
        "multi_field": _field_stats(chunks),
        "build_seconds": round(time.perf_counter() - t_all, 2),
    }
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    if verbose:
        print(f"[完成] 索引统计 -> {STATS_PATH}")
    return stats


def ensure_index(models: list[str] | None = None, verbose: bool = True) -> dict:
    """
    确保索引就绪：缺失则自动构建，已存在则直接返回统计。
    实验脚本统一调用本函数，保证「开箱即跑」。
    """
    models = models or list(COLLECTIONS)
    missing = [m for m in models if VectorStore(COLLECTIONS[m]).count() == 0]
    if not BM25_PATH.exists():
        missing.append("bm25")
    if missing:
        if verbose:
            print(f"[索引] 缺少 {missing}，开始构建…")
        return build_all(models=models, rebuild=False, verbose=verbose)
    if STATS_PATH.exists():
        return json.loads(STATS_PATH.read_text(encoding="utf-8"))
    return {"n_vectors": {m: VectorStore(COLLECTIONS[m]).count() for m in models}}


def get_retriever(model: str = DEFAULT_MODEL, verbose: bool = False):
    """
    获取绑定了指定嵌入模型的统一检索器。
    注意：必须先 set_model 再检索，保证 query 向量与库内向量同模型同维度。
    """
    from rag_core.embed import set_model
    from rag_core.retriever import Retriever

    ensure_index([model], verbose=verbose)
    set_model(model)
    return Retriever(COLLECTIONS[model], bm25_path=BM25_PATH)


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="工单06 混合检索索引构建")
    parser.add_argument("--rebuild", action="store_true", help="全量重建向量库")
    parser.add_argument("--models", nargs="*", default=list(COLLECTIONS),
                        help=f"要建索引的嵌入模型，可选 {list(COLLECTIONS)}")
    args = parser.parse_args()

    stats = build_all(models=args.models, rebuild=args.rebuild, verbose=True)

    print("\n========== 索引统计 ==========")
    print(f"文本块总数   : {stats['n_chunks']}")
    print(f"向量索引     : " + "，".join(
        f"{m}={n}" for m, n in stats["n_vectors"].items()))
    print(f"BM25 文档数  : {stats['n_bm25_docs']}，词条数 {stats['n_bm25_terms']}")
    print("多字段权重   : " + "，".join(
        f"{k}({v['字段']})={v['权重']}" for k, v in stats["multi_field"].items()))
    for k, v in stats["multi_field"].items():
        print(f"  - {v['字段']} {k}: 覆盖 {v['覆盖块数']} 块，"
              f"总词数 {v['总词数']}，去重词数 {v['去重词数']}，"
              f"平均长度 {v['平均字段长度']}")
    print(f"总耗时       : {stats['build_seconds']}s")


if __name__ == "__main__":
    main()
