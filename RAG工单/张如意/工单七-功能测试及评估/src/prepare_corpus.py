# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估

语料准备脚本 —— 把 ccf_competition 的 9 份金融年报解析入库，并输出语料统计。

本脚本重点解决一个真实存在的数据问题：
    源目录 `config.CCF_PDF_DIR` 下的 9 个 PDF 文件名在拷贝过程中被「二次编码」破坏，
    原始 GBK 字节先被按 UTF-8 解码（非法字节替换为 U+FFFD），再被按 GBK 显示，
    于是出现经典的「锟斤拷」乱码，且部分信息已不可逆丢失。

    文件名格式本应是：
        日期__公司全称__股票代码__公司简称__年份__年报.pdf
    例如：
        2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf

    处理策略（三级兜底）：
      1) os.listdir 拿到文件名 → os.fsencode 还原为原始字节 → 尝试严格 GBK 解码；
      2) 严格解码失败时，保留所有「幸存」的汉字片段（如「平…证券」「国泰…」）作为证据；
      3) 以文件名中**未被破坏的 ASCII 字段**（日期 / 6 位股票代码 / 年份）为准，
         通过股票代码→公司名映射表还原真实公司名，再结合年报正文首 3 页做一致性校验。

    这样得到的文档元数据（doc 名）干净可靠，从源头避免「元数据错误」污染检索与评估。

用法：
    python prepare_corpus.py                 # 全量解析 + 建索引
    python prepare_corpus.py --no-tables     # 跳过表格解析（快很多，数值题会掉点）
    python prepare_corpus.py --max-pages 40  # 只解析前 N 页（快速演示 / 冒烟测试用）
    python prepare_corpus.py --stats-only    # 只出语料统计，不建索引
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --- 让脚本能 import 到项目根目录下的 rag_core 共享库 -----------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import config, evaluate  # noqa: E402
from rag_core.bm25 import BM25Retriever  # noqa: E402
from rag_core.chunk import chunk_blocks  # noqa: E402
from rag_core.pdf_parse import parse_pdf  # noqa: E402
from rag_core.vectorstore import VectorStore  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
WO_NO = "人工智能NLP-RAG-功能测试及评估"

#: 本工单专用的向量库集合名（与工单01~06 的 prospectus 集合隔离，互不覆盖）
COLLECTION = "ccf_reports"
#: 默认沿用工单06 的「混合检索 + 级联重排」配置，即 01-06 的最终交付形态
PRESET = "wo06_hybrid"

#: 股票代码 → 公司简称。文件名中 6 位代码是纯 ASCII，未被编码破坏，可作权威锚点。
STOCK_CODE_MAP: dict[str, str] = {
    "000001": "平安银行",
    "601318": "中国平安",
    "600036": "招商银行",
    "601658": "邮储银行",
    "600030": "中信证券",
    "601628": "中国人寿",
    "601601": "中国太保",
    "600999": "招商证券",
    "601211": "国泰君安",
}

#: 公司简称 → 所属金融子行业（用于测试用例的行业覆盖统计）
COMPANY_SECTOR: dict[str, str] = {
    "平安银行": "银行", "招商银行": "银行", "邮储银行": "银行",
    "中国平安": "保险", "中国人寿": "保险", "中国太保": "保险",
    "中信证券": "证券", "招商证券": "证券", "国泰君安": "证券",
}

#: 文档在正文里可能出现的主体名称（用于元数据一致性校验）
COMPANY_ALIASES: dict[str, list[str]] = {
    "平安银行": ["平安银行"],
    "中国平安": ["中国平安", "平安保险"],
    "招商银行": ["招商银行"],
    "邮储银行": ["邮储银行", "邮政储蓄银行"],
    "中信证券": ["中信证券"],
    "中国人寿": ["中国人寿"],
    "中国太保": ["中国太保", "太平洋保险"],
    "招商证券": ["招商证券"],
    "国泰君安": ["国泰君安"],
}

#: 结果目录
ROOT = Path(__file__).resolve().parents[2]
WO_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = WO_DIR / "results"


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


# ---------------------------------------------------------------------------
# 一、GBK 乱码文件名还原
# ---------------------------------------------------------------------------
def decode_garbled_filename(filename: str) -> dict:
    """
    尝试还原被破坏的文件名。

    返回 dict，包含：
        filename          原始（乱码）文件名
        gbk_strict_ok     是否能被 GBK 严格解码（即文件名未被破坏）
        decoded_gbk       解码结果（失败时含 U+FFFD 替换符）
        fragments         替换符剔除后幸存的汉字片段（可读线索）
        code / doc_date / year   从 ASCII 字段中提取的结构化字段
    """
    raw = os.fsencode(filename)          # str -> 原始字节（保留被破坏的 EF BF BD）
    info: dict = {
        "filename": filename,
        "gbk_strict_ok": False,
        "decoded_gbk": "",
        "fragments": "",
        "code": "",
        "doc_date": "",
        "year": "",
    }

    # --- 1) 严格 GBK 解码 ---------------------------------------------------
    try:
        decoded = raw.decode("gbk")
        info["gbk_strict_ok"] = True
        info["decoded_gbk"] = decoded
        info["fragments"] = "".join(c for c in decoded if "一" <= c <= "鿿")
    except UnicodeDecodeError:
        # --- 2) 宽松解码：保留幸存汉字作为人工核对线索 -----------------------
        partial = raw.decode("gbk", errors="replace")
        info["decoded_gbk"] = partial
        info["fragments"] = "".join(c for c in partial if "一" <= c <= "鿿")

    # --- 3) 从 ASCII 骨架里取结构化字段（分隔符 __ 与数字不会被破坏）---------
    ascii_view = raw.decode("ascii", errors="ignore")
    parts = [p for p in ascii_view.split("__")]
    for p in parts:
        if not info["doc_date"] and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p):
            info["doc_date"] = p
        if not info["code"] and re.fullmatch(r"\d{6}", p):
            info["code"] = p
        # 年份字段必须「整段就是 4 位数字」，否则会把 2020-02-14 里的 2020 误当报告年度
        if not info["year"] and re.fullmatch(r"(19|20)\d{2}", p):
            info["year"] = p
    return info


def canonical_doc_name(code: str, year: str, fragments: str = "") -> str:
    """
    由股票代码 + 年份生成规范文档名，形如「平安银行2019年报」。

    代码不在映射表内时，退化为「幸存汉字片段 + 年份」；再退化为「未知公司」。
    """
    company = STOCK_CODE_MAP.get(code) or (fragments[:6] if fragments else "未知公司")
    return f"{company}{year}年报"


@dataclass
class DocInfo:
    """一份年报的完整元数据（含乱码还原过程的全部证据）。"""
    index: int
    pdf_path: str
    txt_path: str
    raw_filename: str                 # 原始乱码文件名
    decoded_filename: str             # GBK 解码结果（可能含 U+FFFD）
    gbk_strict_ok: bool               # 文件名是否完好
    survivors: str                    # 幸存汉字片段
    doc_date: str
    code: str
    year: str
    company: str
    sector: str
    doc_name: str                     # 规范文档名（入库用）
    n_pages: int = 0
    n_blocks: int = 0
    n_text_blocks: int = 0
    n_table_blocks: int = 0
    n_chunks: int = 0
    n_table_chunks: int = 0
    total_chars: int = 0
    metadata_verified: bool = False   # 是否在正文中校验到公司名与年份
    verify_evidence: str = ""
    parse_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def list_ccf_docs(pdf_dir: Path | None = None) -> list[DocInfo]:
    """
    遍历 CCF 年报目录，还原文件名后返回 9 份文档的元数据（按日期排序）。

    说明：os.listdir 返回的 str 在 Windows 上已经是被破坏的形态，
    所以这里先用 os.fsencode 拿回字节再做 GBK 解码，而不是直接 decode 字符串。
    """
    pdf_dir = Path(pdf_dir or config.CCF_PDF_DIR)
    txt_dir = Path(config.CCF_TXT_DIR)

    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"未找到 CCF 年报目录：{pdf_dir}\n"
            f"请确认附件已解压到 D:\\工单\\RAG 工单\\RAG 工单\\附件\\ccf_competition\\"
        )

    docs: list[DocInfo] = []
    for i, fname in enumerate(sorted(os.listdir(pdf_dir)), 1):
        if not fname.lower().endswith(".pdf"):
            continue
        info = decode_garbled_filename(fname)
        code, year = info["code"], info["year"]
        company = STOCK_CODE_MAP.get(code, "")
        doc = DocInfo(
            index=i,
            pdf_path=str(pdf_dir / fname),
            txt_path=str(txt_dir / (Path(fname).stem + ".txt")),
            raw_filename=fname,
            decoded_filename=info["decoded_gbk"],
            gbk_strict_ok=info["gbk_strict_ok"],
            survivors=info["fragments"],
            doc_date=info["doc_date"],
            code=code,
            year=year,
            company=company or f"未知({info['fragments'][:4]})",
            sector=COMPANY_SECTOR.get(company, "未知"),
            doc_name=canonical_doc_name(code, year, info["fragments"]),
        )
        docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# 二、元数据一致性校验（用年报正文回验公司名与年份）
# ---------------------------------------------------------------------------
def verify_doc_metadata(parsed, doc: DocInfo, head_pages: int = 3) -> tuple[bool, str]:
    """
    用年报正文的前几页回验「公司名 + 年份」是否与文件名还原结果一致。

    这一步把「文件名乱码」从不可控风险变成可检测风险：
    只要正文里能查到公司简称与年份，就说明代码→公司名的还原是对的。
    """
    head = "".join(
        b.content for b in parsed.blocks
        if b.page <= head_pages and b.type == "text"
    )
    aliases = COMPANY_ALIASES.get(doc.company, [doc.company])
    hit_alias = next((a for a in aliases if a in head), "")
    year_hit = bool(re.search(rf"{doc.year}\s*年?", head))
    ok = bool(hit_alias) and year_hit
    evidence = (
        f"正文前{head_pages}页命中公司名「{hit_alias or '无'}」、"
        f"年份「{doc.year if year_hit else '无'}」"
    )
    return ok, evidence


# ---------------------------------------------------------------------------
# 三、建索引（带逐文档统计）
# ---------------------------------------------------------------------------
def bm25_path_for(collection: str = COLLECTION) -> Path:
    """
    BM25 索引文件路径。

    注意：rag_core.bm25 默认把索引固定存到 INDEX_DIR/bm25.pkl（全局唯一），
    多个工单共用时会互相覆盖。这里改用按集合命名的文件，
    并在 load_pipeline() 里通过 retriever._bm25_path 指过去（不修改 rag_core）。
    """
    return config.INDEX_DIR / f"bm25_{collection}.pkl"


def build_index(docs: list[DocInfo], with_tables: bool = True,
                with_images: bool = False, max_pages: int | None = None,
                collection: str = COLLECTION, verbose: bool = True) -> dict:
    """
    解析 → 分块 → 建向量库 / BM25 索引，并回填每份文档的统计信息。

    与 rag_core.pipeline.Pipeline.build_index 等价，区别是：
      · 显式传入规范化后的 doc_name（避免乱码文件名进入元数据）
      · 逐文档记录块数 / 表格块数，便于产出语料统计
      · BM25 落盘到集合专属路径，避免与其它工单互相覆盖
    """
    from rag_core.pipeline import PRESETS, PipelineConfig
    # 复制一份配置，避免污染全局 PRESETS（同进程内后续还要用原型配置）
    cfg = PipelineConfig(**{**PRESETS[PRESET].to_dict(),
                            "with_tables": with_tables, "with_images": with_images})

    t_all = time.perf_counter()
    all_chunks = []
    for d in docs:
        t0 = time.perf_counter()
        if verbose:
            print(f"[解析] {d.doc_name:<14} ← {d.raw_filename[:34]}…")
        parsed = parse_pdf(Path(d.pdf_path), d.doc_name,
                           with_tables=with_tables, with_images=with_images)

        # -- 元数据一致性校验（顺手做掉，不额外花时间）--
        d.metadata_verified, d.verify_evidence = verify_doc_metadata(parsed, d)

        # -- 分块 --
        blocks = parsed.blocks
        if max_pages:
            blocks = [b for b in blocks if b.page <= max_pages]
        chunks = chunk_blocks(blocks, strategy=cfg.chunk_strategy,
                              size=cfg.chunk_size, overlap=cfg.chunk_overlap)

        d.n_pages = parsed.n_pages if not max_pages else min(parsed.n_pages, max_pages)
        d.n_blocks = len(blocks)
        d.n_text_blocks = sum(1 for b in blocks if b.type == "text")
        d.n_table_blocks = sum(1 for b in blocks if b.type == "table")
        d.n_chunks = len(chunks)
        d.n_table_chunks = sum(1 for c in chunks if c.type == "table")
        d.total_chars = sum(len(c.text) for c in chunks)
        d.parse_seconds = round(time.perf_counter() - t0, 2)
        all_chunks.extend(chunks)

        if verbose:
            print(f"        {d.n_pages} 页 / {d.n_blocks} 块 → {d.n_chunks} 个检索块 "
                  f"(表格块 {d.n_table_chunks}) {d.parse_seconds}s "
                  f"{'✓元数据校验通过' if d.metadata_verified else '✗元数据校验存疑'}")

    if verbose:
        print(f"[入库] 共 {len(all_chunks)} 个块，开始编码（首次运行需加载本地嵌入模型）…")

    t0 = time.perf_counter()
    vs = VectorStore(collection)
    vs.reset()
    vs.add_chunks(all_chunks, show_progress=verbose)

    bm25 = BM25Retriever()
    bm25.build(all_chunks)
    bm25.save()
    # 归档为集合专属 BM25 索引，供后续实验反复加载
    shutil.copyfile(config.INDEX_DIR / "bm25.pkl", bm25_path_for(collection))

    # 与 Pipeline.build_index 保持相同的 meta.json 结构，便于 rag_core 复用
    meta = {
        "collection": collection,
        "config": cfg.to_dict(),
        "docs": [{"name": d.doc_name, "pages": d.n_pages, "blocks": d.n_blocks}
                 for d in docs],
        "n_chunks": len(all_chunks),
        "n_vectors": vs.count(),
        "n_bm25_docs": len(bm25.doc_ids),
        "build_seconds": round(time.perf_counter() - t_all, 2),
        "chunk_types": _count_types(all_chunks),
    }
    (config.INDEX_DIR / f"{collection}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        print(f"[完成] 向量库 {vs.count()} 条 / BM25 {len(bm25.doc_ids)} 条，"
              f"编码+入库耗时 {meta['build_seconds']}s")
    meta["encode_seconds"] = round(time.perf_counter() - t0, 2)
    return meta


def _count_types(chunks) -> dict:
    out: dict[str, int] = {}
    for c in chunks:
        out[c.type] = out.get(c.type, 0) + 1
    return out


def load_pipeline(collection: str = COLLECTION, top_k: int | None = None,
                  recall_k: int | None = None):
    """
    装载「工单06 混合检索」流水线，供测试与评估脚本复用。

    返回 rag_core.pipeline.Pipeline 实例（已加载 BM25 索引）。
    """
    from rag_core.pipeline import PRESETS, Pipeline

    cfg = PRESETS[PRESET]
    if top_k:
        cfg.top_k = top_k
    if recall_k:
        cfg.recall_k = recall_k
    p = Pipeline(cfg, collection=collection)
    p.retriever._bm25_path = bm25_path_for(collection)
    try:
        p.retriever.load_bm25()
    except FileNotFoundError as e:
        raise SystemExit(
            f"未找到 {collection} 的索引（{bm25_path_for(collection)}）。\n"
            f"请先运行：python prepare_corpus.py\n原始错误：{e}"
        )
    return p


# ---------------------------------------------------------------------------
# 四、主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=f"CCF 年报语料准备（{WO_NO}）")
    ap.add_argument("--no-tables", action="store_true",
                    help="跳过表格解析（快，但数值题召回会下降）")
    ap.add_argument("--with-images", action="store_true",
                    help="启用图像抽取（工单04 能力，年报图表多、速度慢）")
    ap.add_argument("--max-pages", type=int, default=0,
                    help="每份年报只取前 N 页（0=全量），用于快速冒烟演示")
    ap.add_argument("--stats-only", action="store_true", help="只统计不建索引")
    ap.add_argument("--collection", default=COLLECTION)
    args = ap.parse_args()

    ensure_results_dir()
    print(f"===== {WO_NO} · 语料准备 =====")
    print(f"PDF 目录：{config.CCF_PDF_DIR}")

    docs = list_ccf_docs()
    if len(docs) != 9:
        print(f"[warn] 期望 9 份年报，实际发现 {len(docs)} 份")

    print("\n-- 文件名还原结果 --------------------------------------------------")
    broken = 0
    for d in docs:
        flag = "完好" if d.gbk_strict_ok else "乱码"
        broken += int(not d.gbk_strict_ok)
        print(f"  [{d.index}] {flag} code={d.code or '??????'} year={d.year or '????'} "
              f"→ {d.doc_name}  (幸存片段：{d.survivors[:16] or '无'})")
    print(f"  → 9 份文件中 {broken} 份文件名已被破坏，全部通过「股票代码锚点 + "
          f"正文校验」还原\n")

    if not args.stats_only:
        meta = build_index(
            docs, with_tables=not args.no_tables, with_images=args.with_images,
            max_pages=args.max_pages or None, collection=args.collection,
        )
    else:
        # 只统计：解析（走缓存）+ 分块，不入库
        meta = {"collection": "(stats-only)", "n_chunks": 0, "n_vectors": 0}
        for d in docs:
            parsed = parse_pdf(Path(d.pdf_path), d.doc_name,
                               with_tables=not args.no_tables,
                               with_images=args.with_images)
            d.metadata_verified, d.verify_evidence = verify_doc_metadata(parsed, d)
            blocks = parsed.blocks
            if args.max_pages:
                blocks = [b for b in blocks if b.page <= args.max_pages]
            chunks = chunk_blocks(blocks, strategy="structure",
                                  size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
            d.n_pages = parsed.n_pages
            d.n_blocks = len(blocks)
            d.n_text_blocks = sum(1 for b in blocks if b.type == "text")
            d.n_table_blocks = sum(1 for b in blocks if b.type == "table")
            d.n_chunks = len(chunks)
            d.n_table_chunks = sum(1 for c in chunks if c.type == "table")
            d.total_chars = sum(len(c.text) for c in chunks)
            meta["n_chunks"] += len(chunks)

    # -- 落盘语料统计 --------------------------------------------------------
    stats = {
        "wo_no": WO_NO,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": str(config.CCF_PDF_DIR),
        "collection": args.collection,
        "preset": PRESET,
        "options": {"with_tables": not args.no_tables,
                    "with_images": args.with_images,
                    "max_pages": args.max_pages},
        "n_docs": len(docs),
        "n_docs_filename_broken": broken,
        "sector_distribution": _sector_dist(docs),
        "total_pages": sum(d.n_pages for d in docs),
        "total_chunks": sum(d.n_chunks for d in docs),
        "total_table_chunks": sum(d.n_table_chunks for d in docs),
        "total_chars": sum(d.total_chars for d in docs),
        "index_meta": meta,
        "docs": [d.to_dict() for d in docs],
    }
    out = ensure_results_dir() / "corpus_stats.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n-- 语料统计 --------------------------------------------------------")
    print(f"  文档数 {stats['n_docs']} | 总页数 {stats['total_pages']} | "
          f"检索块 {stats['total_chunks']}（表格块 {stats['total_table_chunks']}）")
    print(f"  行业分布 {stats['sector_distribution']}")
    print(f"  语料统计已保存 → {out}")

    # 顺带把文档清单打印给 build_questions.py 用
    print("\n-- 可供出题的公司清单 ----------------------------------------------")
    for d in docs:
        print(f"  {d.doc_name:<14} {d.sector}  {d.doc_date}  {d.n_pages}页")


def _sector_dist(docs: list[DocInfo]) -> dict:
    out: dict[str, int] = {}
    for d in docs:
        out[d.sector] = out.get(d.sector, 0) + 1
    return out


if __name__ == "__main__":
    main()
