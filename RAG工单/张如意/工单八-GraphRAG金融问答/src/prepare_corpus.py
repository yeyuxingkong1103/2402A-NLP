# -*- coding: utf-8 -*-
"""
语料准备：CCF 金融年报 PDF → 解析 → 分块 → 语料清单
工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

本模块是工单08 流水线的第一步，同时作为其余脚本的「公共依赖」，
对外提供四类能力：

  1. 语料准备  prepare_corpus.py  —— 遍历 9 份年报 PDF（还原 GBK 乱码文件名）
                                   → rag_core.pdf_parse 解析 → rag_core.chunk 分块
                                   → results/corpus_stats.json + data/corpus/chunks.json
  2. 公共路径  RESULTS / DATA / ROOT 等常量，供 build_graph / visualize_graph /
              graph_qa / compare_with_wo07 / research_report / serve 复用
  3. 图谱载入  load_graph()      —— 把 data/graph/kg.json 还原成 KnowledgeGraph 对象
  4. 问题解析  load_questions()  —— 解析 questions/eval_question.md 的问题表

设计要点：
  · 文件名是 GBK 乱码（附件压缩包在 Windows 下解压时编码错乱，部分字节已丢失为
    U+FFFD），代码先尝试 latin-1/cp1252 → gbk/gb18030 的还原，还原失败则回退到
    文件名中的 6 位股票代码（ASCII，永远可靠）查表得到规范文档名。
  · PDF 解析结果由 rag_core.pdf_parse 自带磁盘缓存，重复运行不会重复解析。
  · 分块结果落盘到 data/corpus/chunks.json，后续脚本无需再次解析 PDF。

运行：
    python 工单08-GraphRAG金融问答/src/prepare_corpus.py
    python .../prepare_corpus.py --max-docs 2 --no-tables     # 快速试跑
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

# --- 把项目根目录（工单作业/）加入 import 路径，从而 import rag_core -----------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import config                      # noqa: E402
from rag_core.chunk import Chunk, chunk_blocks   # noqa: E402
from rag_core.pdf_parse import parse_pdf         # noqa: E402


# ---------------------------------------------------------------------------
# 路径常量（工单08 自有的 results/ 与 data/）
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]           # 工单作业/
WO08_DIR = Path(__file__).resolve().parents[1]       # 工单08-GraphRAG金融问答/
SRC_DIR = WO08_DIR / "src"
RESULTS_DIR = WO08_DIR / "results"
DATA_DIR = WO08_DIR / "data"
REPORT_DIR = RESULTS_DIR / "graph"                   # 可视化产物目录
CORPUS_DIR = DATA_DIR / "corpus"
CORPUS_CHUNKS = CORPUS_DIR / "chunks.json"           # 分块结果（下游复用）
UPLOAD_DIR = CORPUS_DIR / "uploads"                  # Web 界面上传的 PDF 分块结果
CORPUS_STATS = RESULTS_DIR / "corpus_stats.json"     # 语料统计
QUESTIONS_MD = WO08_DIR / "questions" / "eval_question.md"

for _d in (RESULTS_DIR, REPORT_DIR, CORPUS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 一、GBK 乱码文件名还原
# ---------------------------------------------------------------------------
# 附件 ccf_competition.zip 中的文件名形如：
#   2020-02-14__ƽ�����йɷ����޹�˾__000001__ƽ������__2019��__��ȱ���.pdf
# 原始文件名为 GBK 编码的「2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf」，
# 但在解压/传输过程中被按单字节编码解码过一次，且部分字节被替换为 U+FFFD（不可逆丢失）。
#
# 因此采用「先尝试编码还原、失败再用股票代码查表」的两级策略。
_ENCODINGS_IN = ("latin-1", "cp1252", "utf-8")       # 可能的错误解码方式
_ENCODINGS_OUT = ("gbk", "gb18030", "big5")          # 真实编码（中文 Windows 通常为 GBK）

# 文件名中嵌入的 6 位股票代码 —— 纯 ASCII，不受编码错乱影响，是最可靠的锚点
STOCK_CODE_INFO: dict[str, dict] = {
    "000001": {"name": "平安银行股份有限公司", "short": "平安银行", "period": "2019",
               "sector": "银行"},
    "601318": {"name": "中国平安保险（集团）股份有限公司", "short": "中国平安",
               "period": "2019", "sector": "保险"},
    "600036": {"name": "招商银行股份有限公司", "short": "招商银行", "period": "2019",
               "sector": "银行"},
    "601658": {"name": "中国邮政储蓄银行股份有限公司", "short": "邮储银行",
               "period": "2019", "sector": "银行"},
    "600030": {"name": "中信证券股份有限公司", "short": "中信证券", "period": "2020",
               "sector": "证券"},
    "601628": {"name": "中国人寿保险股份有限公司", "short": "中国人寿", "period": "2020",
               "sector": "保险"},
    "601601": {"name": "中国太平洋保险（集团）股份有限公司", "short": "中国太保",
               "period": "2021", "sector": "保险"},
    "600999": {"name": "招商证券股份有限公司", "short": "招商证券", "period": "2021",
               "sector": "证券"},
    "601211": {"name": "国泰君安证券股份有限公司", "short": "国泰君安", "period": "2021",
               "sector": "证券"},
}

_CN_RE = re.compile(r"[一-鿿]")
_STOCK_RE = re.compile(r"(\d{6})")


def _score_candidate(text: str) -> int:
    """给还原结果打分：中文字符越多越好，出现替换符/「锟斤拷」重罚。"""
    if not text:
        return -999
    score = len(_CN_RE.findall(text))
    score -= 50 * text.count("�")
    score -= 50 * text.count("锟")
    score -= 20 * text.count("?")
    return score


def repair_gbk_filename(filename: str) -> str:
    """
    尝试把乱码文件名还原成中文。

    做法：把当前字符串按「错误的解码方式」重新编码回字节，再按「真实编码」解码。
    例如 latin-1 编码 → gbk 解码，可以还原出大部分被错解的 GBK 字节；
    对已经变成 U+FFFD 的字节则无能为力（信息已丢失），此时返回原文。
    """
    stem = Path(filename).stem
    best, best_score = stem, _score_candidate(stem)
    for enc_in in _ENCODINGS_IN:
        try:
            raw = stem.encode(enc_in)
        except UnicodeEncodeError:
            continue                       # 当前字符串无法用该编码表示，跳过
        for enc_out in _ENCODINGS_OUT:
            try:
                cand = raw.decode(enc_out)
            except UnicodeDecodeError:
                continue
            s = _score_candidate(cand)
            if s > best_score:
                best, best_score = cand, s
    return best


def resolve_doc_name(pdf_path: Path) -> tuple[str, str, str]:
    """
    由 PDF 文件名推断规范文档名。

    Returns:
        (文档名, 还原方式, 文件名原文)

    还原优先级：
        ① GBK 还原成功（中文占比高）→ 用还原后的文件名
        ② 还原失败 → 用文件名里的 6 位股票代码查 STOCK_CODE_INFO 表
        ③ 再失败 → 直接用 path.stem（保底，不报错）
    """
    raw = pdf_path.name
    repaired = repair_gbk_filename(raw)

    if _score_candidate(repaired) >= 6:                 # ① 还原足够干净
        name = repaired.replace("_", "").replace(".pdf", "")
        return name, "gbk_repair", raw

    m = _STOCK_RE.search(raw)                           # ② 股票代码查表
    if m and m.group(1) in STOCK_CODE_INFO:
        info = STOCK_CODE_INFO[m.group(1)]
        name = f"{info['name']}{info['period']}年年度报告"
        return name, f"stock_code:{m.group(1)}", raw

    return repaired, "fallback", raw                    # ③ 保底


def collect_pdfs(pdf_dir: Path | None = None) -> list[Path]:
    """收集 CCF 年报 PDF（默认取 rag_core.config.CCF_PDF_DIR）。"""
    pdf_dir = Path(pdf_dir or config.CCF_PDF_DIR)
    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"未找到年报 PDF 目录：{pdf_dir}\n"
            f"请确认附件 ccf_competition 已解压到该位置，或用 --pdf-dir 指定目录。")
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"目录中没有 PDF 文件：{pdf_dir}")
    return pdfs


# ---------------------------------------------------------------------------
# 二、分块结果的读写（下游脚本复用，避免重复解析 PDF）
# ---------------------------------------------------------------------------
def chunk_from_dict(d: dict) -> Chunk:
    """把 chunks.json 里的一条记录还原成 Chunk 对象。"""
    meta = dict(d.get("meta") or {})
    return Chunk(
        text=d["text"], doc=d.get("doc", ""), page=int(d.get("page", 0)),
        type=d.get("type", "text"), chunk_id=d.get("chunk_id", ""),
        section=d.get("section", ""), meta=meta,
    )


def save_chunks(chunks: list[Chunk], path: Path = CORPUS_CHUNKS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([c.to_dict() for c in chunks], ensure_ascii=False),
        encoding="utf-8")
    return path


def load_chunks(path: Path = CORPUS_CHUNKS,
                include_uploads: bool = True) -> list[Chunk]:
    """
    载入分块结果；不存在时提示先运行 prepare_corpus.py。

    include_uploads=True 时，会一并并入通过 Web 界面（serve.py /api/upload）
    上传并解析的 PDF 分块（存放在 data/corpus/uploads/ 下），
    这样重新运行 build_graph.py 就能把新上传的文档也抽进图谱。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到分块结果：{path}\n"
            f"请先运行：python 工单08-GraphRAG金融问答/src/prepare_corpus.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = [chunk_from_dict(d) for d in data]

    if include_uploads and UPLOAD_DIR.exists():
        seen = {c.chunk_id for c in chunks}
        for f in sorted(UPLOAD_DIR.glob("*.json")):
            try:
                for d in json.loads(f.read_text(encoding="utf-8")):
                    c = chunk_from_dict(d)
                    if c.chunk_id not in seen:
                        seen.add(c.chunk_id)
                        chunks.append(c)
            except Exception:
                continue
    return chunks


def save_upload_chunks(doc_name: str, chunks: list[Chunk]) -> Path:
    """把 Web 界面上传文档的分块结果单独落盘（供 build_graph 增量并入）。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w一-鿿.-]", "_", doc_name)[:80] or "uploaded"
    path = UPLOAD_DIR / f"{safe}.json"
    path.write_text(json.dumps([c.to_dict() for c in chunks], ensure_ascii=False),
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 三、语料构建主流程
# ---------------------------------------------------------------------------
def build_corpus(pdf_dir: Path | None = None,
                 max_docs: int | None = None,
                 strategy: str = "structure",
                 size: int = 900,
                 overlap: int = 120,
                 with_tables: bool = True,
                 verbose: bool = True) -> tuple[list[Chunk], dict]:
    """
    遍历年报 PDF → 解析 → 分块。

    Args:
        strategy: 分块策略，默认 structure（章节结构感知，工单02 起沿用；
                  年报的「第一节 重要提示」等结构同样适用）
        size:     块大小（年报正文密度高，比招股书用更大的块，减少跨块丢信息）
        with_tables: 是否解析表格（数值型问题依赖表格，建议开启）
    """
    pdfs = collect_pdfs(pdf_dir)
    if max_docs:
        pdfs = pdfs[:max_docs]

    all_chunks: list[Chunk] = []
    doc_stats: list[dict] = []
    t_all = time.perf_counter()

    for i, pdf in enumerate(pdfs, 1):
        doc_name, how, raw = resolve_doc_name(pdf)
        if verbose:
            print(f"[{i}/{len(pdfs)}] 解析《{doc_name}》"
                  f"（文件名还原方式：{how}）…")
        t0 = time.perf_counter()
        parsed = parse_pdf(pdf, doc_name, with_tables=with_tables,
                           with_images=False, use_cache=True)
        chunks = chunk_blocks(parsed.blocks, strategy=strategy,
                              size=size, overlap=overlap)
        all_chunks.extend(chunks)
        types = Counter(c.type for c in chunks)
        doc_stats.append({
            "doc": doc_name,
            "pdf": str(pdf),
            "raw_filename": raw,
            "name_resolution": how,
            "n_pages": parsed.n_pages,
            "n_blocks": len(parsed.blocks),
            "n_chunks": len(chunks),
            "chunk_types": dict(types),
            "avg_chunk_chars": round(
                sum(len(c.text) for c in chunks) / max(len(chunks), 1), 1),
            "seconds": round(time.perf_counter() - t0, 2),
        })
        if verbose:
            print(f"      {parsed.n_pages} 页 / {len(parsed.blocks)} 块 → "
                  f"{len(chunks)} 个 chunk（{doc_stats[-1]['seconds']}s）")

    stats = {
        "work_order": "人工智能NLP-RAG-基于Graph RAG 实现金融问答",
        "pdf_dir": str(pdf_dir or config.CCF_PDF_DIR),
        "chunk_strategy": strategy,
        "chunk_size": size,
        "chunk_overlap": overlap,
        "with_tables": with_tables,
        "n_docs": len(pdfs),
        "n_chunks": len(all_chunks),
        "chunk_type_total": dict(Counter(c.type for c in all_chunks)),
        "total_chars": sum(len(c.text) for c in all_chunks),
        "avg_chunk_chars": round(
            sum(len(c.text) for c in all_chunks) / max(len(all_chunks), 1), 1),
        "build_seconds": round(time.perf_counter() - t_all, 2),
        "docs": doc_stats,
    }
    return all_chunks, stats


# ---------------------------------------------------------------------------
# 四、问题集解析（questions/eval_question.md）
# ---------------------------------------------------------------------------
_QID_RE = re.compile(r"^Q\d+$", re.IGNORECASE)


def parse_eval_questions(path: Path = QUESTIONS_MD) -> list[dict]:
    """
    解析 eval_question.md 中的 Markdown 表格。

    表格列约定：| 编号 | 问题 | 类型 | 期望答案要点 |
    只用「编号」列形如 Q01 的行，避免把语料表误读成问题。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"未找到问题集：{path}")
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4 or not _QID_RE.match(cells[0]):
            continue
        out.append({
            "id": cells[0],
            "question": cells[1],
            "type": cells[2],
            "ground_truth": cells[3],
        })
    return out


def load_questions(path: Path = QUESTIONS_MD) -> list[dict]:
    qs = parse_eval_questions(path)
    if not qs:
        raise ValueError(
            f"未能从 {path} 解析出任何问题，请检查表格格式"
            f"（应为 | 编号 | 问题 | 类型 | 期望答案要点 |）。")
    return qs


def questions_by_id(path: Path = QUESTIONS_MD) -> dict[str, dict]:
    return {q["id"]: q for q in load_questions(path)}


# ---------------------------------------------------------------------------
# 五、知识图谱载入（把 data/graph/kg.json 还原成 KnowledgeGraph）
# ---------------------------------------------------------------------------
def load_graph(path: Path | None = None):
    """
    从 kg.json 还原 KnowledgeGraph 对象。

    说明：rag_core.graph_rag.KnowledgeGraph 只提供 save()，没有 load()，
    因此这里按落盘结构逆向重建：实体/关系到内存 → 重建 networkx 图 →
    恢复社区与社区摘要。权重直接沿用（不再做二次合并，避免权重被重复累加）。
    """
    import networkx as nx
    from rag_core.graph_rag import Entity, KnowledgeGraph, Relation

    path = Path(path or (config.GRAPH_DIR / "kg.json"))
    if not path.exists():
        raise FileNotFoundError(
            f"未找到知识图谱文件：{path}\n"
            f"请先运行：python 工单08-GraphRAG金融问答/src/build_graph.py")

    data = json.loads(path.read_text(encoding="utf-8"))
    kg = KnowledgeGraph()

    for e in data.get("entities", []):
        ent = Entity(name=e["name"], type=e.get("type", "其他"),
                     description=e.get("description", ""))
        ent.aliases = set(e.get("aliases", []) or [])
        kg.entities[ent.name] = ent

    kg.relations = [
        Relation(source=r["source"], target=r["target"],
                 type=r.get("type", "相关"),
                 description=r.get("description", ""),
                 weight=float(r.get("weight", 1.0)))
        for r in data.get("relations", [])
        if r.get("source") in kg.entities and r.get("target") in kg.entities
    ]

    # 重建 networkx 图（与原图结构一致）
    kg.g = nx.MultiDiGraph()
    for name, ent in kg.entities.items():
        kg.g.add_node(name, type=ent.type, description=ent.description)
        for a in ent.aliases:
            kg._alias_index[a] = name
    for r in kg.relations:
        kg.g.add_edge(r.source, r.target, type=r.type,
                      description=r.description, weight=r.weight)
    kg.relations_merged = list(kg.relations)
    for name in kg.entities:
        kg.entities[name].degree = kg.g.degree(name)

    kg.communities = {int(k): v for k, v in (data.get("communities") or {}).items()}
    kg.community_summaries = {int(k): v for k, v
                              in (data.get("community_summaries") or {}).items()}
    return kg


def graph_path() -> Path:
    """知识图谱落盘路径（与 rag_core.api 的 /api/graph 保持一致）。"""
    return config.GRAPH_DIR / "kg.json"


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="工单08 语料准备：CCF 金融年报 PDF → 解析 → 分块")
    ap.add_argument("--pdf-dir", type=Path, default=None,
                    help="年报 PDF 目录（默认 rag_core.config.CCF_PDF_DIR）")
    ap.add_argument("--max-docs", type=int, default=None, help="只处理前 N 份（调试用）")
    ap.add_argument("--strategy", default="structure",
                    choices=["fixed", "recursive", "semantic", "structure"],
                    help="分块策略，默认 structure")
    ap.add_argument("--size", type=int, default=900, help="块大小（字符数）")
    ap.add_argument("--overlap", type=int, default=120, help="块间重叠（字符数）")
    ap.add_argument("--no-tables", action="store_true", help="跳过表格解析（更快）")
    args = ap.parse_args()

    chunks, stats = build_corpus(
        pdf_dir=args.pdf_dir, max_docs=args.max_docs,
        strategy=args.strategy, size=args.size, overlap=args.overlap,
        with_tables=not args.no_tables,
    )

    save_chunks(chunks)
    CORPUS_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"语料准备完成：{stats['n_docs']} 份文档 / {stats['n_chunks']} 个 chunk "
          f"/ {stats['total_chars']:,} 字")
    print(f"  分块结果：{CORPUS_CHUNKS}")
    print(f"  统计结果：{CORPUS_STATS}")
    print("  下一步：python 工单08-GraphRAG金融问答/src/build_graph.py")


if __name__ == "__main__":
    main()
