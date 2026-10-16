# -*- coding: utf-8 -*-
"""T4 建索引驱动：解析 → 分块（文本/表格/图像）→ 嵌入（增量）→ FAISS + BM25 → manifest。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化（工单3 同源实现，只读复制后按工单4 契约适配）

相对 工单3 的三个工单4 改动（其余全量沿用，保证既有资产可复用）：
    1. ``--include-images``：把 T2（图像工程师）产出的图像块
       （框图结构树 / 图表数值表 / CLIP 语义描述）作为 ``type="image"`` 的**一等 chunk**
       并入同一索引，带 1-based 物理页码 + xref + bbox；同页同模态子图按 captain 复核结论
       合并成 1 块（第 6 题的页 72 是左右并列的两张子图）；
    2. ``--incremental``：按**内容 sha1** 复用既有向量（工单4 旧索引 + 工单3 只读索引种子），
       只对新增/变化的 chunk 调本地 Ollama 嵌入；复用正确性用「抽样重嵌入 + 余弦比对」实测断言；
    3. FAISS 为主用（``IndexFlatIP``，落盘 ``faiss.index``），另提供 ``milvus_store.py``
       适配器接口（本机不强制导入，见该模块 docstring）。

用法（工作目录 = E:\\gao6gongdan\\工单4）：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py --incremental --include-images
    pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py --rebuild --include-images
    pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py --no-embed      # 仅 BM25（调试）

索引目录：``{index-dir}/{model_slug}/``（如 ``研发/data/index/bge-m3_1024/``），内含
``vectors.npy``、``ids.json``、``faiss.index``、``bm25.pkl``、``bm25.meta.json``、
``index_manifest.json``；另在 ``{index-dir}/index_manifest.json`` 落一份（设计/接口设计.md §5.1 口径）。

退出码：0 成功；1 校验失败（无 PDF / 页数不符 / 条数不一致 / 第5、6题证据块召回失败）；
2 环境或入参错误；3 图像块缺失且显式禁止降级。
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "研发"))

import numpy as np  # noqa: E402

from app.core import chunker, embedder, pdf_parser, vector_store  # noqa: E402
from app.core.bm25_index import BM25Index  # noqa: E402
from app.core.config import (  # noqa: E402
    DEFAULT_SEED_CHUNKS,
    DEFAULT_SEED_INDEX_DIR,
    WORK_ORDER,
    describe_runtime,
    discover_pdfs,
    get_config,
    model_slug,
)
from app.core.errors import RagError  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.text_utils import read_jsonl  # noqa: E402

# 两份语料的期望页数（用于「页数与预期不符」校验；按发现顺序核对，不硬编码文件名）
EXPECTED_PAGES = {548, 350}
MANIFEST_FILE = "index_manifest.json"

# 图像题（第 5、6 题）的证据页 + 关键词召回自检口径。
# 来源：设计/验收标准.md §3.1（题 5 证据页 39、题 6 证据页 72，均为 1-based 物理页码）。
#
# ⚠️ 关键字严格度纪律（T3 复核指出、已据此修正）：**禁止把裸数字当关键字**。
# 裸 "4"/"6" 在任何含页码/金额/年份的长文本里几乎必然命中 → keyword_hit 虚高、且与其他题不可比。
# 数值类关键字一律**带语境**（"4个部门" / "6个销售处"）；匹配前会归一化空白与千分位逗号，
# 因此 "由 4 个部门构成" 能命中 "4个部门"（见 _norm_for_match）。
IMAGE_EVIDENCE_CHECKS: tuple[dict[str, Any], ...] = (
    {
        "qid": 5,
        "corpus": "招股说明书2.pdf",
        "page": 39,
        # 第 5 题的**证据页**（设计/验收标准.md §3.1）：39 是框图页，40/113 是正文互证页；
        # 召回判定用「证据页集合」，而「图像块是否存在」仍只看图像所在页 39。
        "evidence_pages": (39, 40, 113),
        "query": "组织结构图 销售部有几个部门构成 大客户销售部有几个销售处构成",
        # 计数类关键字**带语境**（"4个部门"/"6个销售处"），城市类保持与 设计/验收标准.md §3.1 同源的**裸城市名**
        # （正文写的是「在北京、广州、…各设有 1 家销售处」，写成「北京销售处」会漏掉正文互证页）。
        # 城市名偏弱（"武汉" 会被公司名「武汉力源…」误命中）→ 在 KEYWORD_RULES 里如实标注为弱判据。
        "answer_keywords": ["4个部门", "电话及网络销售部", "渠道销售部", "大客户销售部", "国际贸易部",
                            "6个销售处", "北京", "广州", "成都", "深圳", "武汉", "珠海"],
    },
    {
        "qid": 6,
        "corpus": "招股说明书2.pdf",
        "page": 72,
        "evidence_pages": (72,),
        "query": "2008年中国IC市场应用结构与增长图 增长率最快的是哪个行业 负增长的是哪个行业",
        "answer_keywords": ["汽车", "IC卡", "负增长"],
    },
)

# 关键字严格度与计数口径的**自描述**（写进自检产物，避免下游误读：T3 指出不同题严格度不一致会让计数不可比）。
KEYWORD_RULES: dict[str, Any] = {
    "numeric_keywords_contextual": True,
    "place_names_bare_by_design": True,
    "rule": "数值类关键字一律带语境（如 '4个部门' / '6个销售处'）；**裸数字不作为关键字** —— "
            "裸 '4'/'6' 会被页码、金额、年份误命中，使 keyword_hit 虚高。",
    "place_name_note": "城市/地名类关键字保持裸城市名（与 设计/验收标准.md §3.1 同源），因为正文互证页写的是"
                       "「在北京、广州、成都、深圳、武汉、珠海各设有 1 家销售处」；代价是城市名**偏弱**"
                       "（'武汉' 会被公司名「武汉力源…」误命中）→ 城市类只作弱判据，计数类必须带语境。",
    "normalization": "匹配前统一：去空白、去千分位逗号、全角转半角（见 _norm_for_match）；"
                     "两个自检（图像题自检 / 16 题自检）**使用同一个归一化函数**（T3 复核指出的口径不一致已修）",
    "counts_definition": {
        "bm25_page_hit_count": "BM25 Top-10 内存在「该题证据页」上的块（粗判，仅定位）",
        "bm25_keyword_hit_count": "BM25 Top-10 命中块的**合并正文**含该题全部关键字（细判）",
        "vector_page_hit_count": "向量 Top-10 内存在「该题证据页」上的块",
        "vector_keyword_hit_count": "向量 Top-10 命中块的**合并正文**含该题全部关键字",
        "no_hit_ids": "BM25 与向量两路的证据页命中**都**为空的题号（比上面四个计数更严格）",
    },
    "authoritative_judge": "最终判对/判错以 T6 的 golden 判分为准（测试/common/golden.py）；"
                           "本文件的计数只用于检索层调参（如 T5 定 RRF 权重），不得当作准确率。",
    # 与 T3 的三层命中口径对齐（T3 2026-10 通知的命名）：避免读者把这里的数字当成 T6 的命中率主指标。
    "layer_mapping_to_golden": {
        "本文件 bm25_page_hit_count/vector_page_hit_count": "= golden.check_retrieval_hit_layers 的 "
                                                           "hit_page_level（诊断层B，页级口径）",
        "本文件 bm25_keyword_hit_count/vector_keyword_hit_count": "关键字级细判；不等于 T6 的命中率主指标，"
                                                                 "T6 主指标用 hit_complete（权威层）",
        "golden.check_retrieval_hit_layers 三层": ["hit_complete（权威层，命中率主指标）",
                                                   "hit_any_evidence（诊断层A，含子句片段）",
                                                   "hit_page_level（诊断层B，页级）"],
    },
}

# 16 道工单问题的「检索层证据」清单（题面与证据页逐字取自 设计/验收标准.md §3）。
# 用途：`--verify-questions` 在建完索引后，逐题做**块级**检索命中检查
# （命中判据 = 检索到的 chunk 内容含答案关键字，**不是**「引用页 == 证据页」，避免虚高）。
QUESTIONS_16: tuple[dict[str, Any], ...] = (
    {"id": 5, "corpus": "招股说明书2.pdf", "pages": (39, 40, 113),
     "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？",
     # 严格度：计数类用带语境的 "4个部门"/"6个销售处"（裸 "4"/"6" 会被页码/金额误命中）；
     # 城市类保持与 设计/验收标准.md §3.1 同源的**裸城市名**（正文互证页写的是「在北京、广州、…
     # 各设有 1 家销售处」；写成「北京销售处」会漏掉正文互证页）。城市名是**弱判据**，见 KEYWORD_RULES。
     "keywords": ("4个部门", "6个销售处", "电话及网络销售部", "渠道销售部", "大客户销售部",
                  "国际贸易部", "北京", "广州", "成都", "深圳", "武汉", "珠海")},
    {"id": 6, "corpus": "招股说明书2.pdf", "pages": (72,),
     "question": "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？",
     "keywords": ("汽车", "IC卡")},
    {"id": 1, "corpus": "招股说明书2.pdf", "pages": (22, 24),
     "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？",
     "keywords": ("1,670", "25.04")},
    {"id": 2, "corpus": "招股说明书2.pdf", "pages": (22, 30, 306, 144),
     "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？",
     "keywords": ("3,393.40", "1,526.38", "2,492.78", "9,000.00", "其他与主营业务相关的营运资金")},
    {"id": 3, "corpus": "招股说明书2.pdf", "pages": (157, 21, 158),
     "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？",
     "keywords": ("赵马克", "42.35")},
    {"id": 4, "corpus": "招股说明书2.pdf", "pages": (157, 158),
     "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？",
     "keywords": ("融冰投资", "武汉博润", "上海博润", "听音投资", "联众聚源", "力源贸易", "普芯达")},
    {"id": 260, "corpus": "招股说明书1.pdf", "pages": (129,),
     "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
     "keywords": ("6,464.51", "14,414.16", "18,780.67", "4,627.14")},
    {"id": 95, "corpus": "招股说明书1.pdf", "pages": (160,),
     "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
     "keywords": ("视频指挥系统技术标准", "军事科学院")},
    {"id": 33, "corpus": "招股说明书1.pdf", "pages": (129,),
     "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
     "keywords": ("82.10", "97.31", "94.84", "94.34")},
    {"id": 34, "corpus": "招股说明书1.pdf", "pages": (152,),
     "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
     "keywords": ("电子元器件制造企业", "金属壳体制造企业")},
    {"id": 957, "corpus": "招股说明书1.pdf", "pages": (154, 332),
     "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
     "keywords": ("视频指挥",)},
    {"id": 793, "corpus": "招股说明书1.pdf", "pages": (152,),
     "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
     "keywords": ("军队", "政府机关", "能源")},
    {"id": 795, "corpus": "招股说明书1.pdf", "pages": (27, 157),
     "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
     "keywords": ("国家科技进步一等奖",)},
    {"id": 543, "corpus": "招股说明书1.pdf", "pages": (22,),
     "question": "武汉兴图新科电子股份有限公司注册资本是多少？",
     "keywords": ("5,520",)},
    {"id": 531, "corpus": "招股说明书1.pdf", "pages": (22,),
     "question": "武汉兴图新科电子股份有限公司法定代表人是谁？",
     "keywords": ("程家明",)},
    {"id": 207, "corpus": "招股说明书1.pdf", "pages": (30, 479),
     "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
     "keywords": ("15,000.00",)},
)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha1(text: str) -> str:
    """chunk 内容指纹（增量复用的键；只对 content 取指纹，与 chunk_id 解耦）。"""
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def _repo_rel(path: Path) -> str:
    """相对仓库根的 POSIX 路径（不在仓库内则返回绝对路径，不抛异常）。"""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(Path(path).resolve()).replace("\\", "/")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数（工单3 原有参数全部保留，新增工单4 的图像/增量参数）。"""
    parser = argparse.ArgumentParser(
        description="工单4 建索引（人工智能NLP-RAG-图像内容解析及检索优化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--raw-dir", default=None, help="语料目录，默认 研发/data/raw")
    parser.add_argument("--out-dir", default=None, help="JSONL 产物目录，默认 研发/data/processed")
    parser.add_argument("--index-dir", default=None, help="索引根，默认 研发/data/index")
    parser.add_argument("--incremental", action="store_true",
                        help="增量：复用既有 chunks 与既有向量（按 content sha1），只嵌入新增块")
    parser.add_argument("--include-images", action="store_true",
                        help="把图像块（T2 产出）作为 type=image 的一等 chunk 并入索引")
    parser.add_argument("--image-fallback-pages", default="招股说明书2.pdf:39,72",
                        help="图像块文件缺失时的降级页（格式 文件名:页,页；默认 招股说明书2.pdf:39,72）")
    parser.add_argument("--no-image-fallback", action="store_true",
                        help="禁止降级：图像块文件缺失时直接退出码 3 失败（严格验收用）")
    parser.add_argument("--no-embed", action="store_true", help="只建 BM25（manifest.embedding.backend=none）")
    parser.add_argument("--no-table", action="store_true", help="关表格扫描（仅调试，会打 warnings）")
    parser.add_argument("--fast-table", action="store_true", help="跳过「无绘制线且文本极短」页的表扫描")
    parser.add_argument("--force", action="store_true", help="强制重建（覆盖产物与索引）")
    parser.add_argument("--rebuild", action="store_true", help="强制全量重建（= --force 且不复用 chunks）")
    parser.add_argument("--limit-pages", type=int, default=None, help="只跑前 N 页（调试用，会打 warnings）")
    parser.add_argument("--reuse-chunks", action="store_true", help="无条件复用 out-dir/chunks.jsonl，跳过解析")
    parser.add_argument("--seed-index-dir", default=str(DEFAULT_SEED_INDEX_DIR),
                        help="增量向量的只读种子索引目录（默认 工单3 的 bge-m3_1024）")
    parser.add_argument("--seed-chunks", default=str(DEFAULT_SEED_CHUNKS),
                        help="种子索引对应的 chunks.jsonl（默认 工单3 的 chunks.jsonl）")
    parser.add_argument("--no-seed", action="store_true", help="不使用外部种子，只用本仓库既有索引")
    parser.add_argument("--reuse-verify-samples", type=int, default=5,
                        help="增量复用向量的抽样重嵌入校验条数（0 = 跳过校验，会在 manifest 标注）")
    parser.add_argument("--batch-size", type=int, default=None, help="嵌入批大小（默认 RAG_EMBEDDING__BATCH_SIZE=16）")
    parser.add_argument("--log-level", default=None, help="RAG_LOG__LEVEL 覆盖")
    parser.add_argument("--verify-questions", action="store_true",
                        help="建完后追加 16 题块级检索覆盖检查（写 研发/data/index/q16_retrieval_recall.json）")
    parser.add_argument("--verify-only", action="store_true",
                        help="只校验（不解析/不嵌入）：读既有 chunks.jsonl + 索引跑 16 题与图像题检查")
    return parser


# ---------------------------------------------------------------------------
# chunks 获取（复用 / 重新解析）
# ---------------------------------------------------------------------------
def _load_existing_chunks(out_dir: Path) -> list[chunker.Chunk]:
    """从 ``chunks.jsonl`` 复用已有分块（按 order 排序）。"""
    path = out_dir / "chunks.jsonl"
    if not path.is_file():
        raise RagError(f"复用分块失败：找不到 {path}", code="RAG-2200", stage="chunk")
    chunks = [chunker.Chunk.from_dict(row) for row in read_jsonl(path)]
    chunks.sort(key=lambda c: c.order)
    return chunks


def _fast_table_pages(source: Any, limit_pages: int | None, log: Any) -> set[int]:
    """``--fast-table``：返回「仍需扫描表格」的页清单（跳过无绘制线且文本 < 40 字符的页）。"""
    skip: set[int] = set()
    import pymupdf

    doc = pymupdf.open(str(source.path))
    try:
        total = int(doc.page_count)
        for page_no in range(1, (limit_pages or total) + 1):
            page = doc.load_page(page_no - 1)
            text = "".join((page.get_text("text") or "").split())
            if len(text) < 40:
                try:
                    has_drawings = len(page.get_drawings()) > 0
                except Exception:  # noqa: BLE001 —— 取不到绘制线时保守处理：不跳过（留痕）
                    has_drawings = True
                    log.log_event("fast_table.drawings_error", level="WARNING", page=page_no,
                                  degrade="保留该页表扫描")
                if not has_drawings:
                    skip.add(page_no)
    finally:
        doc.close()
    log.log_event("fast_table.scan", file_name=source.file_name, skipped=len(skip),
                  pages=sorted(skip)[:30])
    return {p for p in range(1, int(total) + 1) if p not in skip}


def _source_hashes_from_manifest(index_root: Path) -> dict[str, str]:
    """读上次 manifest 的 ``source_hashes``（判断语料是否变化 → 能否复用 chunks）。"""
    for path in (index_root / MANIFEST_FILE, index_root / "bge-m3_1024" / MANIFEST_FILE):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        hashes = payload.get("source_hashes")
        if isinstance(hashes, dict) and hashes:
            return {str(k): str(v) for k, v in hashes.items()}
    return {}


def _parse_and_chunk(args: argparse.Namespace, cfg: Any, raw_dir: Path, out_dir: Path,
                     index_root: Path, log: Any) -> tuple[list[chunker.Chunk], list[Any], list[Any]]:
    """解析 PDF → 落盘产物 → 分块 → 落 SQLite（逐页扫描 + 续表合并 + 表头继承，禁止抽样）。"""
    sources = discover_pdfs(raw_dir, logger=log)
    if not sources:
        raise RagError(f"目录下没有任何 PDF：{raw_dir}", code="RAG-1001", stage="parse")
    chunks_path = out_dir / "chunks.jsonl"
    if chunks_path.exists() and not args.force:
        chunks_path.unlink()
    parse_results: list[Any] = []
    chunks: list[chunker.Chunk] = []
    pages_arg = list(range(1, args.limit_pages + 1)) if args.limit_pages else None
    for source in sources:
        if not source.readable:
            log.log_event("build_index.skip_unreadable", level="ERROR", file_name=source.file_name)
            print(f"  ⚠️ {source.file_name} 不可读，已跳过")
            continue
        table_pages = None
        if args.fast_table and not args.no_table:
            table_pages = _fast_table_pages(source, args.limit_pages, log)
        result = pdf_parser.parse_pdf(
            source, cfg=cfg, scan_tables=not args.no_table,
            pages=pages_arg, table_pages=table_pages, logger=log,
        )
        pdf_parser.write_parse_artifacts(result, out_dir, overwrite=True)
        # 解析统计落盘：复用 chunks 的那一轮（--incremental 第二次起）必须能读到**实测值**，
        # 不得用「从 JSONL 反推」的近似值冒充（实测踩坑：反推的 continued_tables/table_pages 与
        # 解析路径不一致，manifest 会写出假数字）。
        stats_path = out_dir / f"{source.file_name.rsplit('.', 1)[0]}.parse_stats.json"
        stats_path.write_text(json.dumps({
            "file_name": result.file_name, "page_count": result.page_count,
            "text_blocks": len(result.text_blocks), "tables": len(result.tables),
            "elapsed_ms": result.elapsed_ms, "stats": dict(result.stats),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pdf_parser.persist_parse_result(result, db_path=index_root / "rag.sqlite3",
                                        source=source, logger=log)
        file_chunks = chunker.build_chunks(result, cfg=cfg, logger=log)
        chunker.persist_chunks(file_chunks, db_path=index_root / "rag.sqlite3", logger=log)
        chunks.extend(file_chunks)
        parse_results.append(result)
        stats = result.stats
        print(f"  ✅ {result.file_name}：{result.page_count} 页 → {len(file_chunks)} 块"
              f"（文本 {sum(1 for c in file_chunks if c.type == 'text')}"
              f" / 表格 {sum(1 for c in file_chunks if c.type == 'table')}）"
              f"，表 {len(result.tables)} 个（跨页续表 {stats.get('continued_tables')}"
              f"、重复表头 {stats.get('repeated_headers_removed')}、退化表 {stats.get('degenerate_tables')}）"
              f"，{result.elapsed_ms / 1000:.1f}s")
    return chunks, sources, parse_results


def _collect_image_chunks(args: argparse.Namespace, out_dir: Path, sources: Sequence[Any],
                          log: Any) -> tuple[list[chunker.Chunk], dict[str, Any]]:
    """图像块 → ``type="image"`` chunk（T2 产出优先；缺失时按参数走**如实标注**的降级路径）。"""
    if not args.include_images:
        return [], {"enabled": False, "reason": "未指定 --include-images"}
    records, diagnosis = chunker.load_image_blocks(out_dir, logger=log)
    info: dict[str, Any] = {"enabled": True, "source_file": diagnosis.get("used"),
                            "records": len(records), "skipped_files": diagnosis.get("skipped"),
                            "fallback": False}
    if records:
        merged = chunker.merge_image_blocks(records, logger=log)
        info["merged_groups"] = len(merged)
        info["merged_from_records"] = len(records)
        chunks = chunker.build_image_chunks(merged, logger=log)
        info["chunks"] = len(chunks)
        # —— 覆盖范围声明（captain 2026-10 抽检要求：不得让读者误以为「两份 PDF 的图像资产都解析了」）——
        by_corpus: dict[str, int] = {}
        for chunk in chunks:
            by_corpus[chunk.file_name] = by_corpus.get(chunk.file_name, 0) + 1
        missing = [s.file_name for s in sources if s.file_name not in by_corpus]
        info["by_corpus"] = by_corpus
        info["corpora_without_image_blocks"] = missing
        info["coverage_note"] = (
            f"图像内容解析范围 = {'、'.join(sorted(by_corpus))}（图像题证据页所在语料）；"
            f"未纳入图像块 = {'、'.join(missing) if missing else '无'}。"
            "说明：招股说明书1 的 image_blocks/charts/diagrams 均为 0（T2 未提供该语料的图像块），"
            "本索引只收录 T2 实际产出的图像块，**不代为声明 PDF1 的图像已被解析**；"
            "对 16 题无影响（第 5、6 题的证据页 39/72 均在招股说明书2）。"
            "如需全语料图像解析，属 T2 的 extract_images.py --all 范畴，代价须由 T2 实测给出。"
        )
        info["evidence_weight_policy"] = (
            "文本/表格块 evidence_weight=1.0；图像块 diagram/chart=1.0、picture=0.4。"
            "依据：picture（页眉 banner、签章页大图等）只有 CLIP 零样本标签，属**弱证据**（T2 实测提醒），"
            "与 diagram/chart 同权会挤掉正文证据。该字段只是**建议权重**，是否使用由检索层（T5）决定；"
            "检索层也可改用 modality=true 的硬过滤或单独通道（设计/接口设计.md §3.4 的 modality 参数）。"
        )
        print(f"图像块（T2 产出）：{diagnosis.get('used')} → {len(records)} 条记录，"
              f"按「同页 + 同 modality_sub」合并为 {len(merged)} 组 → {len(chunks)} 个 type=image chunk"
              f"（覆盖 {by_corpus}；未覆盖 {missing or '无'}）")
        return chunks, info
    if args.no_image_fallback:
        raise RagError(
            f"--include-images 且 --no-image-fallback：在 {out_dir} 找不到 T2 的图像块文件"
            f"（已试 image_blocks.jsonl / image_candidates.jsonl / image_blocks.json）",
            code="RAG-2203", stage="chunk",
        )
    # 降级：只对「图像题证据页」用页自身文本层生成图像块，并如实标注（不产生任何图表数值）
    wanted: dict[str, list[int]] = {}
    for item in str(args.image_fallback_pages or "").split(";"):
        item = item.strip()
        if not item:
            continue
        name, _, pages = item.partition(":")
        if not pages:
            continue
        wanted.setdefault(name.strip(), []).extend(
            int(p) for p in pages.replace("，", ",").split(",") if p.strip())
    fallback_chunks: list[chunker.Chunk] = []
    for source in sources:
        pages = wanted.get(source.file_name) or []
        if not pages:
            continue
        fallback_chunks.extend(chunker.build_image_chunks_from_text_layer(source, pages, logger=log))
    info.update({"fallback": True,
                 "fallback_pages": {k: sorted(set(v)) for k, v in wanted.items()},
                 "reason": "T2 图像块文件缺失（image_blocks.jsonl / image_candidates.jsonl）",
                 "chunks": len(fallback_chunks),
                 "coverage_note": "降级路径：仅覆盖 --image-fallback-pages 指定的证据页，"
                                  "**未做**图像语义编码，不得视为图像解析已覆盖全部语料"})
    print(f"⚠️ 图像块文件缺失 → 降级路径（extraction_method=text_layer_anchor）："
          f"{len(fallback_chunks)} 个 type=image chunk，页 = {info['fallback_pages']}")
    log.log_event("build_index.image_fallback", level="WARNING", **info)
    return fallback_chunks, info


# ---------------------------------------------------------------------------
# 增量嵌入
# ---------------------------------------------------------------------------
def _seed_map_from_index(index_dir: Path, chunks_path: Path, *, model: str, dim: int,
                         log: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """从（旧索引目录, 该索引自己的内容快照）构造 ``content_sha1 → 向量`` 映射（只读，不改种子）。

    ⚠️ 实测教训（本轮自查发现并修复）：早期实现把种子内容取自 ``研发/data/processed/chunks.jsonl``，
    但那个文件在本轮已**先被新内容覆盖**，于是「新内容 → 旧向量」被错误映射（图像块内容改了却复用了旧向量，
    抽样校验刚好没抽到）。因此每个索引目录都必须自带 ``chunks_indexed.jsonl``（= 该索引真正嵌入过的内容快照），
    复用以它为准；没有快照就**拒绝复用**并如实告警，绝不猜。
    """
    info: dict[str, Any] = {"index_dir": str(index_dir), "chunks": str(chunks_path), "usable": False}
    if not index_dir.is_dir() or not chunks_path.is_file():
        info["reason"] = "索引目录或 chunks.jsonl 不存在"
        return {}, info
    try:
        meta = json.loads((index_dir / "ids.json").read_text(encoding="utf-8"))
        vectors = np.load(index_dir / "vectors.npy")
    except (OSError, ValueError) as exc:
        info["reason"] = f"读取种子索引失败：{type(exc).__name__}: {exc}"
        return {}, info
    ids = list(meta.get("ids") or [])
    src_model = str(meta.get("model") or "")
    src_dim = int(meta.get("dim") or 0)
    info.update({"model": src_model, "dim": src_dim, "count": len(ids)})
    if src_model and model and src_model != model:
        info["reason"] = f"嵌入模型不一致（种子 {src_model} vs 当前 {model}），拒绝复用"
        log.log_event("build_index.seed_rejected", level="WARNING", **info)
        return {}, info
    if src_dim and dim and src_dim != int(dim):
        info["reason"] = f"向量维度不一致（种子 {src_dim} vs 当前 {dim}），拒绝复用"
        log.log_event("build_index.seed_rejected", level="WARNING", **info)
        return {}, info
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        info["reason"] = f"种子索引自洽性失败：vectors={vectors.shape} ids={len(ids)}"
        log.log_event("build_index.seed_rejected", level="WARNING", **info)
        return {}, info
    content_by_id: dict[str, str] = {}
    for row in read_jsonl(chunks_path):
        cid = str(row.get("chunk_id") or "")
        if cid:
            content_by_id[cid] = str(row.get("content") or "")
    mapping: dict[str, np.ndarray] = {}
    for row_index, cid in enumerate(ids):
        content = content_by_id.get(cid)
        if content is None:
            continue
        mapping.setdefault(_sha1(content), vectors[row_index])
    info.update({"usable": bool(mapping), "mapped": len(mapping), "seed_ids": len(ids),
                 "ids_matched_in_chunks": len(content_by_id)})
    log.log_event("build_index.seed_loaded", index_dir=str(index_dir), mapped=len(mapping),
                  vectors=int(vectors.shape[0]), ids_matched=len(content_by_id))
    return mapping, info


def _embed_with_reuse(chunks: Sequence[chunker.Chunk], *, cfg: Any, batch_size: int, log: Any,
                      seed_maps: Sequence[tuple[str, dict[str, np.ndarray]]],
                      verify_samples: int) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """增量嵌入：命中种子即复用向量，其余调 Ollama 批量嵌入；并做抽样重嵌入校验。"""
    started = time.perf_counter()
    dim_seed = 0
    for _name, mapping in seed_maps:
        for vector in mapping.values():
            dim_seed = int(np.shape(vector)[0])
            break
        if dim_seed:
            break
    vectors_by_row: list[np.ndarray | None] = [None] * len(chunks)
    reused_from: list[str | None] = [None] * len(chunks)
    for row, chunk in enumerate(chunks):
        digest = _sha1(chunk.content)
        for name, mapping in seed_maps:
            vector = mapping.get(digest)
            if vector is not None:
                vectors_by_row[row] = np.asarray(vector, dtype=np.float32)
                reused_from[row] = name
                break
    todo = [row for row, vector in enumerate(vectors_by_row) if vector is None]
    reused = len(chunks) - len(todo)
    print(f"增量嵌入：块 {len(chunks)}｜复用 {reused}｜需新嵌入 {len(todo)}"
          f"（种子：{'、'.join(name for name, _ in seed_maps) or '无'}）")
    log.log_event("build_index.embed_plan", chunks=len(chunks), reused=reused, to_embed=len(todo))

    embed_info: dict[str, Any] = {"backend": "none", "model": cfg.llm.ollama_embed_model,
                                  "dim": dim_seed, "count": 0, "elapsed_ms": 0.0}
    if todo:
        result = embedder.embed_texts([chunks[row].content for row in todo], cfg=cfg,
                                      batch_size=batch_size, logger=log)
        if result.count != len(todo):
            raise RagError(f"嵌入条数 {result.count} 与待嵌入块数 {len(todo)} 不一致",
                           code="RAG-3200", stage="index")
        for offset, row in enumerate(todo):
            vectors_by_row[row] = np.asarray(result.vectors[offset], dtype=np.float32)
        embed_info = {"backend": result.backend, "model": result.model, "dim": result.dim,
                      "count": result.count, "elapsed_ms": result.elapsed_ms}
        print(f"新嵌入完成：{result.count} 条 × {result.dim} 维，{result.elapsed_ms / 1000:.1f}s"
              f"（backend={result.backend}）")
    elif reused:
        # 本轮全部复用：必须**如实**标注 backend，不能让 manifest 出现「backend=none 却有 2601 条向量」
        embed_info["backend"] = "reuse"
        embed_info["note"] = "本轮 0 条新增块，全部复用既有向量（未调用嵌入后端）"
        embed_info["dim"] = int(embed_info.get("dim") or dim_seed or 0)
    dim = int(embed_info.get("dim") or dim_seed or 0)
    if dim <= 0:
        raise RagError("无法确定向量维度（既无种子也无嵌入结果）", code="RAG-3200", stage="index")
    for row, vector in enumerate(vectors_by_row):
        if vector is None or int(np.shape(vector)[0]) != dim:
            raise RagError(f"第 {row} 行向量缺失或维度不符（期望 {dim}）", code="RAG-3200",
                           stage="index", detail={"row": row})
    matrix = np.vstack([np.asarray(vector, dtype=np.float32) for vector in vectors_by_row])
    reuse_info: dict[str, Any] = {
        "enabled": True, "reused": reused, "embedded": len(todo), "chunks": len(chunks),
        "reuse_ratio": round(reused / len(chunks), 4) if chunks else 0.0,
        "sources": [{"name": name, "mapped": len(mapping)} for name, mapping in seed_maps],
        "verify": None,
    }
    # 抽样重嵌入校验：复用向量必须与「现在重新跑一次嵌入」高度一致（余弦 ≥ 0.99）
    if reused and verify_samples > 0:
        row_candidates = [row for row, name in enumerate(reused_from) if name is not None]
        step = max(len(row_candidates) // int(verify_samples), 1)
        sample_rows = row_candidates[::step][: int(verify_samples)]
        result = embedder.embed_texts([chunks[row].content for row in sample_rows], cfg=cfg,
                                      batch_size=len(sample_rows), logger=log)
        cosines: list[float] = []
        for offset, row in enumerate(sample_rows):
            fresh = np.asarray(result.vectors[offset], dtype=np.float32)
            old = np.asarray(vectors_by_row[row], dtype=np.float32)
            denom = float(np.linalg.norm(fresh) * np.linalg.norm(old))
            cosines.append(float(fresh @ old / denom) if denom else 0.0)
        reuse_info["verify"] = {
            "samples": len(sample_rows), "rows": [int(r) for r in sample_rows],
            "min_cosine": round(min(cosines), 6), "max_cosine": round(max(cosines), 6),
            "method": "抽样重嵌入 + 余弦比对（bge-m3 本地 Ollama，实测确定性）",
        }
        print(f"复用向量抽样校验：{len(sample_rows)} 条，余弦 min={min(cosines):.6f} / max={max(cosines):.6f}")
        if min(cosines) < 0.90:
            raise RagError(
                f"增量复用校验失败：抽样最小余弦 {min(cosines):.4f} < 0.90（复用向量与现算不一致）；"
                f"请改用 --rebuild 全量重建", code="RAG-3201", stage="index",
                detail={"min_cosine": min(cosines)},
            )
        if min(cosines) < 0.99:
            log.log_event("build_index.reuse_verify_soft", level="WARNING",
                          min_cosine=min(cosines), samples=len(sample_rows))
    elif reused:
        reuse_info["verify"] = {"samples": 0, "note": "--reuse-verify-samples=0：未做抽样校验（已在此标注）"}
    reuse_info["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return matrix, embed_info, reuse_info


# ---------------------------------------------------------------------------
# 图像题证据块召回自检（验收项 2 的直接证据）
# ---------------------------------------------------------------------------
def _verify_image_recall(chunks: Sequence[chunker.Chunk], vectors: np.ndarray | None,
                         bm25: BM25Index, *, cfg: Any, log: Any) -> dict[str, Any]:
    """第 5、6 题：证据块是否入索引 + BM25/向量两路能否召回（返回结构化结果）。

    判据（**两层，分别报告，不混为一谈**）：
        ① 该图像页是否存在 ``type="image"`` chunk（结构性证据：证据块确实入索引）；
        ② 该题的**证据页集合**（Q5: 39/40/113；Q6: 72）在 BM25 Top-10 或向量 Top-10 内是否有块被召回
           ——按 设计/验收标准.md §3.4「不得以引用页判命中」的反面用法：这里只做**召回覆盖**判定，
           答案正确性由 T6 的块级关键字判定负责；
        ③ 一并报出「图像块本身是否被召回」与答案关键字命中数（降级路径下会明显偏低，**原样报出**）。
    """
    report: dict[str, Any] = {"checks": [], "passed": True, "vector_path": vectors is not None,
                              "keyword_rules": KEYWORD_RULES}
    image_chunks = [c for c in chunks if c.type == "image"]
    row_of = {c.chunk_id: index for index, c in enumerate(chunks)}
    by_file_page: dict[tuple[str, int], list[chunker.Chunk]] = {}
    for chunk in chunks:
        by_file_page.setdefault((chunk.file_name, chunk.page), []).append(chunk)
    query_vector: np.ndarray | None = None
    for check in IMAGE_EVIDENCE_CHECKS:
        evidence_pages = tuple(int(p) for p in (check.get("evidence_pages") or (check["page"],)))
        image_ids = {c.chunk_id for c in image_chunks
                     if c.file_name == check["corpus"] and c.page == int(check["page"])}
        entry: dict[str, Any] = {
            "qid": check["qid"], "corpus": check["corpus"], "page": check["page"],
            "evidence_pages": list(evidence_pages), "query": check["query"],
            "answer_keywords": list(check["answer_keywords"]),
        }
        if not image_ids:
            entry["problem"] = "该页没有任何 type=image chunk（证据块未入索引）"
            entry["passed"] = False
            report["passed"] = False
            report["checks"].append(entry)
            continue
        entry["chunk_ids"] = sorted(image_ids)
        entry["modality_sub"] = sorted({str(c.modality_sub) for c in image_chunks if c.chunk_id in image_ids})
        entry["extraction_methods"] = sorted({str(c.extraction_method) for c in image_chunks
                                              if c.chunk_id in image_ids})
        joined = "\n".join(c.content for c in image_chunks if c.chunk_id in image_ids)
        entry["answer_keywords_present"] = [kw for kw in check["answer_keywords"]
                                            if _norm_for_match(kw) in _norm_for_match(joined)]
        entry["answer_keywords_missing"] = [kw for kw in check["answer_keywords"]
                                            if _norm_for_match(kw) not in _norm_for_match(joined)]
        # ① BM25 / 关键词路
        hits = bm25.search(check["query"], top_k=10, logger=log)
        entry["bm25_top10"] = [{"rank": rank, "chunk_id": cid, "score": round(score, 4),
                                "corpus": (chunks[row_of[cid]].file_name if cid in row_of else None),
                                "page": (int(chunks[row_of[cid]].page) if cid in row_of else None),
                                "type": (chunks[row_of[cid]].type if cid in row_of else None)}
                               for rank, (cid, score) in enumerate(hits, 1)]
        entry["bm25_recalled"] = any(cid in image_ids for cid, _ in hits)          # 图像块本身是否召回
        bm25_evidence = [cid for cid, _ in hits
                         if cid in {c.chunk_id for p in evidence_pages
                                    for c in by_file_page.get((check["corpus"], p), [])}]
        entry["bm25_evidence_chunks"] = bm25_evidence
        entry["bm25_evidence_recalled"] = bool(bm25_evidence)
        # ② 向量路（bge-m3；与 BM25 分属不同语义空间，只分别判定、**不把分数相加**）
        if vectors is not None:
            if query_vector is None:
                query_vector = np.asarray(embedder.embed_query(check["query"], cfg=cfg, logger=log),
                                          dtype=np.float32)
            scores = vectors @ query_vector
            order = np.argsort(-scores)[:10]
            top_ids = [chunks[int(i)].chunk_id for i in order]
            entry["vector_recalled"] = any(cid in image_ids for cid in top_ids)
            # ⚠️ 字段口径（captain 2026-10 抽检发现并已修）：``rank`` = 检索名次，``page`` = **真实物理页码**。
            # 早期实现把 enumerate 的位置当成了 chunk 下标，导致 page 写出 1,2,3…（名次冒充页码），
            # 会被报告误读为「第 1 页」并让「引用页码可回溯」判错。此处一律从 chunk 元数据取页码。
            entry["vector_top10"] = [{"rank": rank, "chunk_id": chunks[int(row)].chunk_id,
                                      "score": round(float(scores[int(row)]), 4),
                                      "corpus": chunks[int(row)].file_name,
                                      "page": int(chunks[int(row)].page),
                                      "type": chunks[int(row)].type}
                                     for rank, row in enumerate(order, 1)]
            evidence_keys = {(check["corpus"], p) for p in evidence_pages}
            entry["vector_evidence_chunks"] = [chunks[int(i)].chunk_id for i in order
                                               if (chunks[int(i)].file_name, chunks[int(i)].page)
                                               in evidence_keys]
            entry["vector_evidence_recalled"] = bool(entry["vector_evidence_chunks"])
        else:
            entry["vector_top10"] = None
            entry["vector_recalled"] = None
            entry["vector_evidence_recalled"] = None
        entry["recall_basis"] = "证据页集合在 BM25 Top-10 或向量 Top-10 内被召回（含图像块或正文互证块）"
        entry["passed"] = bool(entry["bm25_evidence_recalled"] or (entry.get("vector_evidence_recalled") or False))
        if not entry["passed"]:
            entry["problem"] = "证据页集合在关键词路与向量路 Top-10 内均无任何块被召回"
            report["passed"] = False
        report["checks"].append(entry)
    report["image_chunk_total"] = len(image_chunks)
    return report


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def _dep_version(name: str) -> str:
    """依赖版本（**不可用即返回 "unavailable"**；元数据存在必须再真实 import 一次）。"""
    aliases = {"pymupdf": ("pymupdf", "PyMuPDF"), "faiss": ("faiss-cpu", "faiss"),
               "pymilvus": ("pymilvus", "milvus")}
    version = "unavailable"
    try:
        import importlib.metadata as md

        for candidate in aliases.get(name, (name,)):
            try:
                version = md.version(candidate)
                break
            except md.PackageNotFoundError:
                continue
    except Exception:  # noqa: BLE001 —— 版本探测失败不阻断建索引，但如实标注
        version = "unavailable"
    if version == "unavailable":
        return version
    try:
        import importlib

        importlib.import_module(name)
        return version
    except Exception as exc:  # noqa: BLE001 —— 半残包必须显式标注，不静默
        return f"{version}(present_but_unimportable:{type(exc).__name__})"


def _deps_report() -> tuple[dict[str, str], list[str]]:
    """依赖清单 + 由其诚实标注衍生的 warnings。"""
    names = ("pymupdf", "numpy", "jieba", "faiss", "pymilvus")
    deps = {name: _dep_version(name) for name in names}
    warnings: list[str] = []
    for name, value in deps.items():
        if "present_but_unimportable" in value:
            warnings.append(f"依赖 {name} 元数据存在但真实 import 失败：{value}（按不可用对待）")
        elif value == "unavailable" and name == "pymilvus":
            warnings.append("依赖 pymilvus 不可用 → Milvus 适配器仅提供接口桩，不作为运行依赖（已如实登记）")
    return deps, warnings


def _known_limitations() -> list[dict[str, Any]]:
    """索引侧**已知限制与口径**（结构化登记，供 T5/T6/验收人直接引用，避免误读或重复解释）。

    纪律：每条只写「可验证的事实 + 影响 + 处置」，不写猜测；与 `设计/验收标准.md` §10 的全局已知限制互补。
    """
    return [
        {
            "id": "image_chunk_id_vs_source_id",
            "statement": "本索引的 `chunk_id`（如 `pdf2-p72-img-1692-0`）与 T2（图像工程师）产物里的图像块 id "
                         "（如 `img:招股说明书2.pdf:p72:chart:x1692`）**不是一一对应关系**，禁止互相推导或直接字符串替换。",
            "reason": "本索引按「同页 + 同 modality_sub」把同一张组合图的多个子图**合并成一个 chunk**"
                      "（页 72 的 xref1692 条形图 + xref1693 3D 饼图即合并为 `pdf2-p72-img-1692-0`），"
                      "且 `chunk_id` 里的 xref 片段只取「合并组内第一条记录」的 xref。",
            "how_to_trace_back": "精确回溯请用 chunk 的 `source_chunk_id`（T2 原始 id）与 `sub_blocks`"
                                 "（每个子图带 xref/bbox/image_sha1）；`sub_bboxes` 给出合并后的各子图 bbox。",
            "extra_semantics": {
                "page": "一律 1-based 物理页码，且 chunk 带 `page_basis=\"1-based-physical\"`",
                "diagram": "框图无位图，故 `xref=null`；回溯依据是 `structured.bbox_by_node` 与 "
                           "`structured.recovered_text`（**不是** sub_blocks 的 xref）",
                "chart/picture": "带位图 xref，可用 `xref` + `bbox` + 页码渲染回原区域",
            },
            "impact": "低：只影响「回溯」路径，不影响检索与引用页码（引用页码取 `page`，口径唯一）。",
        },
        {
            "id": "faiss_write_index_non_ascii_path",
            "statement": "本机 faiss **1.7.4** 的 `faiss.write_index` 在**非 ASCII 路径**（本仓库路径含中文「工单4」）"
                         "上必失败：`FileIOWriter ... could not open ...: No such file or directory`（C++ 侧窄字符 fopen）。",
            "mitigation": "已改用 `faiss.serialize_index` 取字节 + Python（支持 Unicode 路径）写 `faiss.index`；"
                          "读取端优先 `faiss.read_index`，失败则回退 `faiss.deserialize_index`（同一序列化格式）。",
            "evidence": "ids.json 的 `faiss_note` 字段 + 日志事件 `vector.faiss_write_fallback` / "
                        "`vector.faiss_written(method=faiss.serialize_index)` / `vector.faiss_loaded`",
            "impact": "无功能影响：索引由 FAISS IndexFlatIP 承载，numpy 精确检索为等价兜底（结果一致）。",
        },
        {
            "id": "picture_is_weak_evidence",
            "statement": "图像块里 `modality_sub=picture` 的文本只有 CLIP 零样本标签（页眉 banner、签章页大图等），"
                         "属**弱证据**，其 `evidence_weight = 0.4`（diagram/chart 与文本/表格均为 1.0）。",
            "how_to_use": "检索层应据此降权，或按 `type`/`modality_sub` 硬过滤 / 走单独通道；"
                          "**不得**与 diagram/chart 等权，否则会挤掉正文证据。",
            "impact": "只提供建议权重；是否使用由检索层（T5）决定。",
        },
        {
            "id": "q16_counts_are_page_level_selfcheck",
            "statement": "manifest 的 `q16_retrieval_recall` 与 `q05_q6_image_recall` 是**上游自检**，其"
                         "「证据页命中」为**页级口径**（= T3 三层口径里的 `hit_page_level` 诊断层B）。",
            "how_to_use": "T6 报告的命中率主指标必须用权威层 `hit_complete`（完整证据）；"
                          "本自检计数只用于检索层调参，**不得当作准确率**。详见 `keyword_rules`。",
            "impact": "口径不同会给出不同数字（如 id5：页级 True、权威层需另算），两者不可混用。",
        },
        {
            "id": "image_coverage_from_t2",
            "statement": "索引只收录 T2（图像工程师）实际产出的图像块；本轮两份 PDF 均有覆盖，但 T2 的 CLIP 编码"
                         "有 `--clip-limit`（默认每份 PDF 300 张）上限，**超限的位图不会被编码**，因而不进本索引。",
            "how_to_use": "需要全量图像语义时，属 T2 的 `extract_images.py --all` 范畴，代价须由其实测给出；"
                          "本索引不代为声明「全部内嵌图已解析」。",
            "evidence": "manifest.images.records / by_corpus / corpora_without_image_blocks + "
                        "研发/data/index/images_clip_manifest.json",
            "impact": "对 16 题无影响（两道图像题证据页 39/72 均在覆盖范围内）。",
        },
        {
            "id": "reproducibility_scope_metadata_files_not_byte_stable",
            "statement": "**数据文件**（`vectors.npy` / `chunks_indexed.jsonl`）与 **`ids` 列表内容**跨运行"
                         "字节/内容稳定（已实测幂等）；但 **`ids.json` / `index_manifest.json` 的文件字节不稳定**，"
                         "差异**仅来自时间戳字段**（`ids.updated_at`；`manifest.built_at` / `created_at` / "
                         "`vector.updated_at` —— 共 3 个 ISO 时间戳，captain 2026-10 逐一探测确认无其它易变字段）。",
            "stable_across_runs": ["vectors.npy（字节）", "chunks_indexed.jsonl（字节）",
                                   "bm25.pkl 内的 ids 列表内容", "ids.json 的 ids 列表内容（语义指纹）"],
            "not_byte_stable_and_why": {
                "研发/data/index/bge-m3_1024/ids.json": "字段 updated_at",
                "研发/data/index/index_manifest.json": "字段 built_at / created_at / vector.updated_at",
                "研发/data/index/bge-m3_1024/index_manifest.json": "同上（两份 manifest 内容相同）",
            },
            "how_to_check_change": "判定「索引是否变化」请比对**数据文件**（`vectors.npy` / `chunks_indexed.jsonl`）"
                                   "或**内容指纹**（`ids` 列表）；**不要**比对元数据文件字节 —— 否则会得出"
                                   "「每次都变」的错误结论（正是工单4 修掉的那类问题）。",
            "impact": "无功能影响；只影响「用文件 hash 判断索引是否变化」的做法。",
        },
    ]


def _clip_info(index_root: Path) -> dict[str, Any]:
    """CLIP 通道状态（读 T2 的 images_clip_manifest.json；没有就如实写 false + reason）。"""
    candidates = (index_root / "images_clip_manifest.json",
                  index_root / "bge-m3_1024" / "images_clip_manifest.json",
                  index_root / "images_clip" / "images_clip_manifest.json")
    for path in candidates:
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                return {"enabled": False, "reason": f"images_clip_manifest.json 无法解析：{type(exc).__name__}"}
            return {"enabled": True, "manifest": _repo_rel(path), "model": payload.get("model"),
                    "dim": payload.get("dim"), "images": payload.get("count") or payload.get("images"),
                    "device": payload.get("device"), "failures": payload.get("failures")}
    return {"enabled": False, "model": None, "dim": None, "images": 0,
            "reason": "未发现 研发/data/index/images_clip_manifest.json（T2 的 CLIP 编码产物）"}


def _norm_for_match(text: str) -> str:
    """归一化（去空白/千分位逗号、全角转半角）——用于块级关键字命中判定，避免格式差异误判。"""
    body = str(text or "")
    table = str.maketrans("０１２３４５６７８９％（），．－", "0123456789%(),.-")
    body = body.translate(table)
    return "".join(body.split()).replace(",", "").replace("，", "")


def _verify_questions(chunks: Sequence[chunker.Chunk], vectors: np.ndarray | None, bm25: BM25Index,
                      *, cfg: Any, log: Any) -> dict[str, Any]:
    """16 题**块级**检索覆盖检查（T4 objective 的直接证据）。

    判据（严格按 设计/验收标准.md §3.4 口径，**不用「引用页 == 证据页」判命中**）：
        * ``bm25_page_hit`` / ``vector_page_hit``：Top-10 里是否存在「文件+证据页」的块（粗判，仅供定位）；
        * ``keyword_hit_bm25`` / ``keyword_hit_vector``：检索到的块**内容**是否含该题答案关键字（细判）；
        * 每题给出检索到的证据页块与其关键字命中数，缺口原样报出，不做「命中率虚高」。
    """
    row_of = {c.chunk_id: i for i, c in enumerate(chunks)}
    by_file_page: dict[tuple[str, int], list[chunker.Chunk]] = {}
    for chunk in chunks:
        by_file_page.setdefault((chunk.file_name, chunk.page), []).append(chunk)
    report: dict[str, Any] = {"total": len(QUESTIONS_16), "vector_path": vectors is not None,
                              "questions": [], "bm25_page_hit_count": 0, "bm25_keyword_hit_count": 0,
                              "vector_page_hit_count": 0, "vector_keyword_hit_count": 0,
                              "no_hit_ids": [], "keyword_rules": KEYWORD_RULES}
    for item in QUESTIONS_16:
        entry: dict[str, Any] = {"id": item["id"], "corpus": item["corpus"],
                                "evidence_pages": list(item["pages"]), "question": item["question"],
                                "keywords": list(item["keywords"])}
        bm25_hits = bm25.search(item["question"], top_k=10, logger=log)
        # 口径同 _verify_image_recall：rank = 名次，page = 真实物理页码（从 chunk 元数据取，绝不用名次冒充）
        entry["bm25_top10"] = [{"rank": rank, "chunk_id": cid, "score": round(score, 4),
                                "corpus": chunks[row_of[cid]].file_name if cid in row_of else None,
                                "page": int(chunks[row_of[cid]].page) if cid in row_of else None,
                                "type": chunks[row_of[cid]].type if cid in row_of else None}
                               for rank, (cid, score) in enumerate(bm25_hits, 1)]
        bm25_pages = {(chunks[row_of[cid]].file_name, chunks[row_of[cid]].page)
                      for cid, _ in bm25_hits if cid in row_of}
        entry["bm25_page_hit"] = any((item["corpus"], int(p)) in bm25_pages for p in item["pages"])
        bm25_text = _norm_for_match("\n".join(chunks[row_of[cid]].content for cid, _ in bm25_hits
                                             if cid in row_of))
        entry["bm25_answers"] = [kw for kw in item["keywords"] if _norm_for_match(kw) in bm25_text]
        entry["keyword_hit_bm25"] = len(entry["bm25_answers"]) == len(item["keywords"])
        if vectors is not None:
            query = np.asarray(embedder.embed_query(item["question"], cfg=cfg, logger=log),
                               dtype=np.float32)
            scores = vectors @ query
            order = np.argsort(-scores)[:10]
            entry["vector_top10"] = [{"rank": rank, "chunk_id": chunks[int(row)].chunk_id,
                                      "score": round(float(scores[int(row)]), 4),
                                      "corpus": chunks[int(row)].file_name,
                                      "page": int(chunks[int(row)].page),
                                      "type": chunks[int(row)].type}
                                     for rank, row in enumerate(order, 1)]
            vector_pages = {(chunks[int(i)].file_name, chunks[int(i)].page) for i in order}
            entry["vector_page_hit"] = any((item["corpus"], int(p)) in vector_pages for p in item["pages"])
            vector_text = _norm_for_match("\n".join(chunks[int(i)].content for i in order))
            entry["vector_answers"] = [kw for kw in item["keywords"]
                                       if _norm_for_match(kw) in vector_text]
            entry["keyword_hit_vector"] = len(entry["vector_answers"]) == len(item["keywords"])
        else:
            entry["vector_page_hit"] = None
            entry["keyword_hit_vector"] = None
            entry["vector_answers"] = []
        entry["chunks_on_evidence_pages"] = sum(
            len(by_file_page.get((item["corpus"], int(p)), [])) for p in item["pages"])
        entry["passed"] = bool(entry["bm25_page_hit"] or (entry.get("vector_page_hit") or False))
        report["questions"].append(entry)
        if entry["bm25_page_hit"]:
            report["bm25_page_hit_count"] += 1
        if entry["keyword_hit_bm25"]:
            report["bm25_keyword_hit_count"] += 1
        if entry.get("vector_page_hit"):
            report["vector_page_hit_count"] += 1
        if entry.get("keyword_hit_vector"):
            report["vector_keyword_hit_count"] += 1
        if not entry["passed"]:
            report["no_hit_ids"].append(item["id"])
    report["all_questions_have_page_hit"] = not report["no_hit_ids"]
    return report


def _build_manifest(*, cfg: Any, sources: Sequence[Any], parse_results: Sequence[Any],
                    chunks: Sequence[chunker.Chunk], embedding_info: dict[str, Any],
                    vector_info: dict[str, Any], bm25: BM25Index, model_dir: Path, index_root: Path,
                    warnings: Sequence[str], elapsed_ms: float, reused_chunks: bool,
                    reuse_info: dict[str, Any], image_info: dict[str, Any],
                    recall: dict[str, Any] | None, source_hashes: dict[str, str],
                    q16_recall: dict[str, Any] | None = None) -> dict[str, Any]:
    """按 设计/接口设计.md §5.1 组装 index_manifest.json（数字全部来自实测，不照抄示例）。"""
    by_modality: dict[str, int] = {"text": 0, "table": 0, "image": 0}
    by_sub: dict[str, int] = {}
    by_corpus: dict[str, int] = {}
    for chunk in chunks:
        by_modality[chunk.type] = by_modality.get(chunk.type, 0) + 1
        if chunk.modality_sub:
            by_sub[chunk.modality_sub] = by_sub.get(chunk.modality_sub, 0) + 1
        by_corpus[chunk.file_name] = by_corpus.get(chunk.file_name, 0) + 1

    per_file: list[dict[str, Any]] = []
    for source in sources:
        result = next((r for r in parse_results if r.file_name == source.file_name), None)
        file_chunks = [c for c in chunks if c.file_name == source.file_name]
        entry: dict[str, Any] = {
            "file_name": source.file_name,
            "sha256_16": source.sha256_16,
            "page_count": (result.page_count if result is not None else source.page_count),
            "text_chunks": sum(1 for c in file_chunks if c.type == "text"),
            "table_chunks": sum(1 for c in file_chunks if c.type == "table"),
            "image_chunks": sum(1 for c in file_chunks if c.type == "image"),
        }
        if result is not None:
            stats = result.stats
            entry.update({
                "tables": len(result.tables), "raw_tables": stats.get("raw_tables"),
                "continued_tables": stats.get("continued_tables"),
                "repeated_headers_removed": stats.get("repeated_headers_removed"),
                "degenerate_tables": stats.get("degenerate_tables"),
                "text_blocks": len(result.text_blocks), "max_table_run": stats.get("max_run"),
                "table_pages": stats.get("table_pages"),
            })
        else:
            entry["note"] = "reuse_chunks：未重新解析，逐文件解析统计见 data/processed/*.jsonl"
        per_file.append(entry)

    deps, dep_warnings = _deps_report()
    built_at = _now_iso()
    return {
        "run_id": cfg.run_id,
        "built_at": built_at,
        "created_at": built_at,
        "work_order": WORK_ORDER,
        # —— 设计/接口设计.md §5.1 冻结键（值全部来自本次实测）——
        "embed_model": embedding_info.get("model") or cfg.llm.ollama_embed_model,
        "embed_dim": int(vector_info.get("dim") or embedding_info.get("dim") or 0),
        "vector_backend": (f"faiss:{vector_info.get('faiss_index_type')}"
                           if vector_info.get("faiss") else "numpy:flat-ip"),
        "chunk_total": len(chunks),
        "chunk_by_modality": by_modality,
        "chunk_by_corpus": by_corpus,
        "page_basis": "1-based-physical",
        "source_hashes": dict(source_hashes),
        "clip": _clip_info(index_root),
        # —— 已知限制（captain 2026-10 要求：把「chunk_id 与 T2 图像 id 非一一对应」等写成交付物里的显式限制）——
        "known_limitations": _known_limitations(),
        # —— 工单4 扩展键（全部实测值）——
        "files": per_file,
        "embedding": embedding_info,
        "bm25": {"count": bm25.size(), "vocab_size": bm25.vocab_size(), "k1": bm25.k1, "b": bm25.b,
                 "table_chunks": sum(1 for m in bm25.meta.values() if m.get("type") == "table"),
                 "image_chunks": sum(1 for m in bm25.meta.values() if m.get("type") == "image")},
        "vector": {k: v for k, v in vector_info.items() if k != "ids"},
        "chunk_by_modality_sub": by_sub,
        "index_dir": _repo_rel(model_dir),
        "index_root": _repo_rel(index_root),
        "incremental": reuse_info,
        "images": image_info,
        "q5_q6_image_recall": recall,
        "q16_retrieval_recall": (None if q16_recall is None else {
            "total": q16_recall["total"],
            "bm25_page_hit_count": q16_recall["bm25_page_hit_count"],
            "bm25_keyword_hit_count": q16_recall["bm25_keyword_hit_count"],
            "vector_page_hit_count": q16_recall.get("vector_page_hit_count"),
            "vector_keyword_hit_count": q16_recall.get("vector_keyword_hit_count"),
            "no_hit_ids": q16_recall["no_hit_ids"],
            "detail_file": "研发/data/index/q16_retrieval_recall.json",
        }),
        "reused_chunks": reused_chunks,
        "deps": deps,
        "elapsed_ms": elapsed_ms,
        "warnings": list(warnings) + dep_warnings,
    }


def _verify_only(args: argparse.Namespace, cfg: Any, out_dir: Path, index_root: Path, log: Any) -> int:
    """只校验模式：读既有 ``chunks.jsonl`` + 索引，跑第 5/6 题与 16 题检查（不解析、不嵌入新块）。"""
    chunks_path = out_dir / "chunks.jsonl"
    if not chunks_path.is_file():
        print(f"❌ --verify-only 需要既有分块：{chunks_path} 不存在")
        return 3
    chunks = _load_existing_chunks(out_dir)
    model_dir = index_root / model_slug(cfg.llm.ollama_embed_model, 1024)
    if not model_dir.is_dir():
        print(f"❌ --verify-only 需要既有索引目录：{model_dir} 不存在")
        return 3
    print(f"只校验模式：分块 {len(chunks)}（文本 {sum(1 for c in chunks if c.type == 'text')}"
          f" / 表格 {sum(1 for c in chunks if c.type == 'table')}"
          f" / 图像 {sum(1 for c in chunks if c.type == 'image')}）｜索引 {model_dir}")
    load = vector_store.load_vector_index(model_dir, expected_dim=1024, logger=log,
                                          use_faiss=cfg.vector_store.use_faiss)
    bm25 = BM25Index.load(model_dir, logger=log)
    vectors = load.vectors if load.count == len(chunks) else None
    if vectors is None:
        print(f"⚠️ 索引条数 {load.count} 与分块数 {len(chunks)} 不一致 → 向量路判定改为跳过（如实标注）")
    image_count = sum(1 for c in chunks if c.type == "image")
    recall = _verify_image_recall(chunks, vectors, bm25, cfg=cfg, log=log) if image_count else None
    questions = _verify_questions(chunks, vectors, bm25, cfg=cfg, log=log)
    exit_code = 0
    if recall is not None and not recall.get("passed"):
        print("❌ 第 5/6 题证据块召回自检未通过")
        exit_code = 1
    if not questions["all_questions_have_page_hit"]:
        print(f"❌ 16 题中这些题的证据页在前 10 名内没有任何块：{questions['no_hit_ids']}")
        exit_code = 1
    path = index_root / "q16_retrieval_recall.json"
    payload = {"q5_q6_image_recall": recall, "q16_retrieval_recall": questions,
               "keyword_rules": KEYWORD_RULES,
               "refreshed_by": "build_index.py --verify-only（只重算自检证据，**不改动任何索引文件**）",
               "refreshed_at": _now_iso()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 同步刷新两份 manifest 里的**证据块**（q5_q6_image_recall / q16_retrieval_recall 摘要），
    # 使 manifest 与详情文件一致；索引本体（vectors.npy/ids.json/faiss.index/bm25.pkl/快照）一律不碰，
    # 因此不会影响正在读取索引的 T5。
    for manifest_path in (index_root / MANIFEST_FILE, model_dir / MANIFEST_FILE):
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.log_event("build_index.verify_only_manifest_skip", level="WARNING",
                          path=str(manifest_path), error_type=type(exc).__name__, message=str(exc))
            continue
        manifest["q5_q6_image_recall"] = recall
        manifest["q16_retrieval_recall"] = {
            "total": questions["total"],
            "bm25_page_hit_count": questions["bm25_page_hit_count"],
            "bm25_keyword_hit_count": questions["bm25_keyword_hit_count"],
            "vector_page_hit_count": questions.get("vector_page_hit_count"),
            "vector_keyword_hit_count": questions.get("vector_keyword_hit_count"),
            "no_hit_ids": questions["no_hit_ids"],
            "detail_file": "研发/data/index/q16_retrieval_recall.json",
            "refreshed_by": "--verify-only",
            "refreshed_at": payload["refreshed_at"],
        }
        manifest["keyword_rules"] = KEYWORD_RULES
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(manifest_path)
        log.log_event("build_index.verify_only_manifest_refreshed", path=str(manifest_path))
    print(f"16 题块级检索覆盖：BM25 证据页命中 {questions['bm25_page_hit_count']}/16"
          f"｜向量证据页命中 {questions.get('vector_page_hit_count')}/16"
          f"｜BM25 全关键字命中 {questions['bm25_keyword_hit_count']}/16"
          f"｜向量全关键字命中 {questions.get('vector_keyword_hit_count')}/16 → {path}")
    return exit_code


def _read_parse_summaries(out_dir: Path, sources: Sequence[Any], log: Any) -> list[Any]:
    """复用 chunks 时，从 ``{stem}.pages.jsonl`` / ``tables.jsonl`` 回读解析统计（不写假 0）。"""

    class _Summary:
        """最小只读摘要（只含 manifest 需要的字段）。"""

        def __init__(self, file_name: str, page_count: int, text_blocks: int,
                     tables: int, stats: dict[str, Any]) -> None:
            self.file_name = file_name
            self.page_count = page_count
            self.text_blocks = [None] * text_blocks
            self.tables = [None] * tables
            self.stats = stats
            self.elapsed_ms = 0.0

    summaries: list[Any] = []
    for source in sources:
        stem = source.file_name.rsplit(".", 1)[0]
        pages_path = out_dir / f"{stem}.pages.jsonl"
        tables_path = out_dir / f"{stem}.tables.jsonl"
        blocks_path = out_dir / f"{stem}.text_blocks.jsonl"
        stats_path = out_dir / f"{stem}.parse_stats.json"
        if not pages_path.is_file():
            log.log_event("build_index.summary_missing", level="WARNING", file=str(pages_path))
            continue
        pages = list(read_jsonl(pages_path))
        blocks = list(read_jsonl(blocks_path)) if blocks_path.is_file() else []
        if stats_path.is_file():
            # 首选：解析那一轮落盘的**实测统计**（build_index 解析路径写入，见 _parse_and_chunk）
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            stats = dict(payload.get("stats") or {})
            summaries.append(_Summary(source.file_name, int(payload.get("page_count") or len(pages)),
                                      int(payload.get("text_blocks") or len(blocks)),
                                      int(payload.get("tables") or 0), stats))
            log.log_event("build_index.summary_from_stats", file_name=source.file_name,
                          file=str(stats_path))
            continue
        # 兜底：没有 stats 文件时从产物按**与解析路径同口径**的公式反推
        tables = list(read_jsonl(tables_path)) if tables_path.is_file() else []
        table_pages = sorted({int(p["page"]) for p in pages if int(p.get("table_count") or 0) > 0})
        best = run = 0
        prev: int | None = None
        for page in table_pages:
            run = run + 1 if prev is not None and page == prev + 1 else 1
            best = max(best, run)
            prev = page
        stats = {
            "raw_tables": sum(int(p.get("table_count") or 0) for p in pages),
            "continued_tables": sum(len(t.get("absorbed_pages") or []) for t in tables),
            "repeated_headers_removed": sum(len([n for n in (t.get("notes") or []) if "重复表头" in str(n)])
                                            for t in tables),
            "degenerate_tables": sum(1 for t in tables if t.get("degenerate")),
            "table_pages": len(table_pages),
            "max_run": best,
        }
        log.log_event("build_index.summary_derived", level="WARNING", file_name=source.file_name,
                      reason=f"缺少 {stats_path.name}，已按同口径反推")
        summaries.append(_Summary(source.file_name, len(pages), len(blocks), len(tables), stats))
    return summaries


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """执行建索引全流程并打印实测摘要。"""
    args = build_parser().parse_args(argv)
    if args.rebuild:
        args.force = True
    cfg = get_config()
    if args.log_level:
        cfg = get_config(refresh=True)
    setup_logging(cfg, run_id=cfg.run_id, force=True)
    atexit.register(shutdown_logging)
    log = get_logger("build_index")
    raw_dir = Path(args.raw_dir) if args.raw_dir else cfg.paths.raw_dir
    out_dir = Path(args.out_dir) if args.out_dir else cfg.paths.processed_dir
    index_root = (Path(args.index_dir) if args.index_dir else cfg.paths.index_dir).resolve()
    batch_size = int(args.batch_size or cfg.embedding.batch_size)

    with log.enter("build_index", {"raw_dir": str(raw_dir), "out_dir": str(out_dir),
                                   "index_dir": str(index_root), "incremental": args.incremental,
                                   "include_images": args.include_images, "no_embed": args.no_embed,
                                   "no_table": args.no_table, "fast_table": args.fast_table,
                                   "reuse_chunks": args.reuse_chunks, "rebuild": args.rebuild,
                                   "limit_pages": args.limit_pages, "batch_size": batch_size}) as span:
        print("=" * 78)
        print(f"T4 建索引｜{describe_runtime()['work_order']}")
        print(f"解释器：{sys.executable}（{sys.version.split()[0]}）")
        print(f"语料  ：{raw_dir}　产物：{out_dir}　索引根：{index_root}")
        print(f"模式  ：{'增量' if args.incremental else '全量'}｜图像块："
              f"{'并入' if args.include_images else '不并入'}")
        print("=" * 78)
        t_all = time.perf_counter()
        warnings: list[str] = []
        if args.no_table:
            warnings.append("no_table：关闭了表格扫描（仅调试）")
        if args.limit_pages:
            warnings.append(f"limit_pages={args.limit_pages}（仅调试，不得用于验收）")
        if args.no_embed:
            warnings.append("no_embed：仅构建 BM25 索引（embedding.backend=none）")

        if not raw_dir.is_dir():
            print(f"❌ 语料目录不存在：{raw_dir}")
            return 2
        if args.verify_only:
            return _verify_only(args, cfg, out_dir, index_root, log)
        sources = discover_pdfs(raw_dir, logger=log)
        if not sources:
            print(f"❌ 目录下没有任何 PDF：{raw_dir}")
            return 1
        source_hashes = {s.file_name: s.sha256_16 for s in sources}
        print(f"发现 {len(sources)} 份 PDF：" + "、".join(f"{s.file_name}({s.page_count}页)" for s in sources))

        # ---- 1) chunks：增量复用 / 重新解析 ----
        chunks_path = out_dir / "chunks.jsonl"
        prev_hashes = _source_hashes_from_manifest(index_root) if args.incremental else {}
        hashes_same = bool(prev_hashes) and prev_hashes == source_hashes
        reuse_chunks = False
        if chunks_path.is_file() and not args.force and not args.rebuild:
            if args.reuse_chunks:
                reuse_chunks = True
                warnings.append("reuse_chunks：无条件复用既有 chunks.jsonl（未重新解析 PDF）")
            elif args.incremental:
                if hashes_same:
                    reuse_chunks = True
                    print("增量：PDF 哈希与上次 manifest 一致 → 复用既有 chunks.jsonl（不重新解析）")
                else:
                    reason = ("上次 manifest 无 source_hashes" if not prev_hashes
                              else "PDF 哈希与上次不一致（语料已变）")
                    print(f"增量：{reason} → 重新解析全部 PDF")
        parse_results: list[Any] = []
        if reuse_chunks:
            chunks = _load_existing_chunks(out_dir)
            print(f"复用已有分块：{chunks_path} → {len(chunks)} 块（跳过解析）")
            log.log_event("build_index.reuse_chunks", path=str(chunks_path), chunks=len(chunks))
            parse_results = _read_parse_summaries(out_dir, sources, log)
        else:
            chunks, sources, parse_results = _parse_and_chunk(args, cfg, raw_dir, out_dir, index_root, log)

        # ---- 2) 图像块并入（工单4 新增）----
        # 关键：图像块**每次都由 T2 产物（或降级路径）重新推导**，因此复用 chunks.jsonl 时必须先
        # 丢掉旧的 type=image 块，否则会与新版重复（实测坑：增量重跑时旧降级块留在索引里）。
        stale = [c for c in chunks if c.type == "image"]
        if stale:
            chunks = [c for c in chunks if c.type != "image"]
            print(f"丢弃旧图像块 {len(stale)} 个（每次按 T2 产物/降级路径重新生成）")
            log.log_event("build_index.drop_stale_image_chunks", count=len(stale),
                          chunk_ids=[c.chunk_id for c in stale][:20])
        image_chunks, image_info = _collect_image_chunks(args, out_dir, sources, log)
        chunks.extend(image_chunks)
        # 全局 order 重排（图像块按 (文件, 页码, 表>图>文) 归位，保证 order 与物理顺序一致）
        rank = {"table": 0, "image": 1, "text": 2}
        chunks.sort(key=lambda c: (c.file_name, c.page_start, rank.get(c.type, 9), c.chunk_id))
        for order, chunk in enumerate(chunks):
            chunk.order = order
        chunker.write_chunks(chunks, chunks_path, overwrite=True)
        if image_chunks:
            chunker.persist_chunks(image_chunks, db_path=index_root / "rag.sqlite3", logger=log)
        text_chunks = sum(1 for c in chunks if c.type == "text")
        table_chunks = sum(1 for c in chunks if c.type == "table")
        image_count = sum(1 for c in chunks if c.type == "image")
        print(f"分块合计：{len(chunks)}（文本 {text_chunks} / 表格 {table_chunks} / 图像 {image_count}）")
        if args.include_images and image_count == 0:
            print("❌ --include-images 但没有任何图像块入索引（验收要求 chunk_by_modality.image > 0）")
            return 1

        # ---- 3) 嵌入（增量）----
        vector_info: dict[str, Any] = {"backend": "none", "count": 0, "dim": 0, "faiss": False}
        embedding_info: dict[str, Any] = {"backend": "none", "model": cfg.llm.ollama_embed_model,
                                          "dim": 0, "count": 0, "elapsed_ms": 0.0}
        reuse_info: dict[str, Any] = {"enabled": False, "reason": "未启用 --incremental"}
        slug = "bm25_only"
        vectors: np.ndarray | None = None
        if args.no_embed:
            print("⚠️ --no-embed：跳过嵌入，仅建 BM25")
        else:
            seed_maps: list[tuple[str, dict[str, np.ndarray]]] = []
            if args.incremental:
                own_snapshot = index_root / "bge-m3_1024" / "chunks_indexed.jsonl"
                own_map, own_info = _seed_map_from_index(index_root / "bge-m3_1024", own_snapshot,
                                                         model=cfg.llm.ollama_embed_model, dim=1024,
                                                         log=log)
                if own_map:
                    seed_maps.append(("工单4既有索引", own_map))
                elif (index_root / "bge-m3_1024" / "ids.json").is_file():
                    warnings.append(f"未启用 工单4 既有索引作为种子：{own_info.get('reason')}")
                if not args.no_seed:
                    seed_map, seed_info = _seed_map_from_index(Path(args.seed_index_dir),
                                                               Path(args.seed_chunks),
                                                               model=cfg.llm.ollama_embed_model,
                                                               dim=1024, log=log)
                    if seed_map:
                        seed_maps.append(("工单3只读种子", seed_map))
                    else:
                        warnings.append(f"未启用 工单3 种子：{seed_info.get('reason')}")
                if not seed_maps:
                    warnings.append("--incremental 但没有任何可用向量种子 → 本次等价全量嵌入")
            vectors, embedding_info, reuse_info = _embed_with_reuse(
                chunks, cfg=cfg, batch_size=batch_size, log=log, seed_maps=seed_maps,
                verify_samples=int(args.reuse_verify_samples),
            )
            slug = model_slug(embedding_info["model"] or cfg.llm.ollama_embed_model,
                              embedding_info["dim"])
        model_dir = index_root / slug
        model_dir.mkdir(parents=True, exist_ok=True)

        # ---- 4) 向量索引（FAISS 主用）----
        if vectors is not None:
            # 内容快照：与 vectors.npy 行序一一对应，供下次 --incremental 精确复用（按 content sha1）。
            # 必须写在**本目录内**，避免被 out_dir/chunks.jsonl 的后续覆盖带偏（见 _seed_map_from_index 注释）。
            snapshot_path = model_dir / "chunks_indexed.jsonl"
            chunker.write_chunks(chunks, snapshot_path, overwrite=True)
            snapshot_hashes = {c.chunk_id: _sha1(c.content) for c in chunks}
            vector_info = vector_store.build_vector_index(
                chunks, vectors, index_dir=model_dir, model=cfg.llm.ollama_embed_model,
                logger=log, use_faiss=cfg.vector_store.use_faiss,
                meta_extra={
                    "by_modality": {"text": text_chunks, "table": table_chunks, "image": image_count},
                    "chunk_count": len(chunks),
                    "embed_backend": embedding_info.get("backend"),
                    "incremental": {k: v for k, v in reuse_info.items() if k != "sources"},
                    "images": image_info,
                    "content_snapshot": snapshot_path.name,
                    "content_sha1_sample": list(snapshot_hashes.items())[:3],
                },
            )
            print(f"向量索引：{model_dir}（{vector_info['count']} × {vector_info['dim']}"
                  f"，faiss={'是' if vector_info.get('faiss') else '否'}"
                  f"{'，faiss.index 已落盘' if vector_info.get('faiss_file') else ''}）")

        # ---- 5) BM25 ----
        bm25 = BM25Index(k1=1.5, b=0.75, use_jieba=True)
        bm25.build(chunks, logger=log)
        dim_model = (f"{embedding_info.get('model')}#{embedding_info.get('dim')}"
                     if vectors is not None else None)
        bm25.save(model_dir, dim_model=dim_model, logger=log)
        print(f"BM25 索引：{bm25.size()} 块，词表 {bm25.vocab_size()} 词 → {model_dir / 'bm25.pkl'}")

        # ---- 6) 第 5、6 题证据块召回自检（验收项 2）----
        recall: dict[str, Any] | None = None
        if image_count:
            recall = _verify_image_recall(chunks, vectors, bm25, cfg=cfg, log=log)
            print("-" * 78)
            print("第 5、6 题证据块召回自检：")
            for entry in recall["checks"]:
                status = "✅" if entry.get("passed") else "❌"
                print(f"  {status} 题 {entry['qid']}（{entry['corpus']} 第 {entry['page']} 页）："
                      f"图像块 {len(entry.get('chunk_ids') or [])} 个"
                      f"｜BM25证据页召回={entry.get('bm25_evidence_recalled')}"
                      f"｜向量证据页召回={entry.get('vector_evidence_recalled')}"
                      f"｜图像块自身召回 BM25={entry.get('bm25_recalled')}/向量={entry.get('vector_recalled')}"
                      f"｜答案关键词命中 {len(entry.get('answer_keywords_present') or [])}"
                      f"/{len(entry.get('answer_keywords') or [])}")
                if entry.get("answer_keywords_missing"):
                    print(f"      图像块缺失关键词：{'、'.join(entry['answer_keywords_missing'])}")
                if entry.get("bm25_evidence_chunks"):
                    print(f"      命中的证据页块：{'、'.join(entry['bm25_evidence_chunks'][:6])}")
                if entry.get("problem"):
                    print(f"      问题：{entry['problem']}")

        # ---- 7) manifest（模型目录 + 索引根各一份，后者为 设计 §5.1 口径）----
        questions: dict[str, Any] | None = None
        if args.verify_questions:
            questions = _verify_questions(chunks, vectors, bm25, cfg=cfg, log=log)
            q_path = index_root / "q16_retrieval_recall.json"
            q_path.write_text(json.dumps({"q5_q6_image_recall": recall, "q16_retrieval_recall": questions},
                                         ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("-" * 78)
            print("16 题块级检索覆盖（T4 objective 证据）：")
            for entry in questions["questions"]:
                mark = "✅" if entry["passed"] else "❌"
                print(f"  {mark} 题 {entry['id']:>3}｜证据页 {entry['evidence_pages']}"
                      f"｜BM25页命中={entry['bm25_page_hit']}｜向量页命中={entry.get('vector_page_hit')}"
                      f"｜BM25关键字 {len(entry['bm25_answers'])}/{len(entry['keywords'])}"
                      f"｜向量关键字 {len(entry.get('vector_answers') or [])}/{len(entry['keywords'])}")
            print(f"  合计：BM25 证据页命中 {questions['bm25_page_hit_count']}/16"
                  f"｜向量证据页命中 {questions.get('vector_page_hit_count')}/16"
                  f"｜BM25 全关键字 {questions['bm25_keyword_hit_count']}/16"
                  f"｜向量全关键字 {questions.get('vector_keyword_hit_count')}/16"
                  f" → {q_path}")
        manifest = _build_manifest(
            cfg=cfg, sources=sources, parse_results=parse_results, chunks=chunks,
            embedding_info=embedding_info, vector_info=vector_info, bm25=bm25, model_dir=model_dir,
            index_root=index_root, warnings=warnings,
            elapsed_ms=round((time.perf_counter() - t_all) * 1000, 2),
            reused_chunks=bool(reuse_chunks), reuse_info=reuse_info, image_info=image_info,
            recall=recall, source_hashes=source_hashes, q16_recall=questions,
        )
        for path in (model_dir / MANIFEST_FILE, index_root / MANIFEST_FILE):
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"manifest：{model_dir / MANIFEST_FILE}")
        print(f"manifest：{index_root / MANIFEST_FILE}（设计/接口设计.md §5.1 口径）")

        # ---- 8) 校验 ----
        exit_code = 0
        if not args.limit_pages and parse_results:
            found = {r.page_count for r in parse_results}
            if not found <= EXPECTED_PAGES:
                print(f"❌ 页数与预期不符：实际 {sorted(found)}，预期 ⊆ {sorted(EXPECTED_PAGES)}")
                exit_code = 1
            else:
                print(f"✅ 页数校验通过：{sorted(found)} ⊆ {sorted(EXPECTED_PAGES)}")
        if recall is not None and not recall.get("passed"):
            print("❌ 第 5/6 题证据块召回自检未通过（详见上方与 manifest.q5_q6_image_recall）")
            exit_code = 1
        print("-" * 78)
        print(f"总耗时：{time.perf_counter() - t_all:.1f}s｜块数 {len(chunks)}"
              f"（文本 {text_chunks} / 表格 {table_chunks} / 图像 {image_count}）｜"
              f"向量 {vector_info.get('count', 0)}×{vector_info.get('dim', 0)}｜BM25 词表 {bm25.vocab_size()}")
        if warnings:
            print("⚠️ warnings：" + "；".join(warnings))
        span.set_output({"files": len(sources), "chunks": len(chunks), "text_chunks": text_chunks,
                         "table_chunks": table_chunks, "image_chunks": image_count,
                         "embedding": embedding_info, "incremental": reuse_info,
                         "index_dir": str(model_dir), "warnings": warnings,
                         "q5_q6_passed": (recall or {}).get("passed")})
        return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RagError as exc:
        print(f"\n❌ 建索引失败：{exc}", file=sys.stderr)
        print(f"   detail={exc.detail}", file=sys.stderr)
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 —— 顶层兜底：完整堆栈 + 退出码 2
        import traceback

        traceback.print_exc()
        raise SystemExit(2)
