# -*- coding: utf-8 -*-
"""工单4 分块、关键词与元数据（工单3 同源实现，只读复制后按工单4 契约适配；§号沿用 工单3 设计文档编号，工单4 契约见 设计/系统架构.md §5.2）。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化

策略：
    * **表格**：一个逻辑表块 = 1 个 chunk（含跨页合并块），``content`` 按架构 §6.4 组装
      （表标题 + 来源行 + Markdown + 关键数字）；超过 ``max_table_chars`` 才按行批次拆，
      每部分都带同一 ``table_id`` 与「第 k/n 部分」标注；
    * **文本**：逐页聚合文本块 → 清洗页眉页脚 → 按句切分（size=500 / overlap=80），
      尾巴 < ``min_size`` 并入前一块；
    * **图像（工单4 新增）**：把 T2 产出的图像块（框图结构树 / 图表数值表 / CLIP 语义描述）
      作为**一等 chunk** 并入同一索引，``type="image"``（= ``modality``，见 ``Chunk.to_dict``），
      带 1-based 物理页码、``xref`` 与 ``bbox`` 以便回溯渲染；
    * 页码一律 1-based 物理页，越界抛 ``ChunkError``（不静默钳制）。

chunk_id 口径（工单4 冻结，见 设计/接口设计.md §3.3/§5.1/§5.2 示例）：
    ``{corpus_key}-p{page}-x{idx}``（文本）／``{corpus_key}-p{page}-t{idx}``（表格）／
    ``{corpus_key}-p{page}-img-{xref}-{idx}``（图像），其中 ``corpus_key`` = pdf1/pdf2。
    工单3 原 id 保留在 ``legacy_chunk_id`` 字段，便于与只读基线对照。
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field, fields as dc_fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import AppConfig, get_config
from .errors import ChunkError, wrap
from .pdf_parser import PdfParseResult, TextBlock, TableBlock, connect_sqlite
from .text_utils import (
    collapse_whitespace,
    extract_keywords as _text_extract_keywords,
    read_jsonl,
    split_sentences,
    strip_printed_page,
    write_jsonl,
)

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

# 关键字段加权词典（工单4 的 16 道题面关键词；命中即作为关键词候选并加权）
# 来源：设计/验收标准.md §3 的「判对关键字」逐题汇总（图像题 5/6 的词也在内，
# 这样「销售部/大客户销售部/增长率/IC卡」这类词在图像块与正文块里同权参与抽取）。
KEYWORD_WEIGHTS: dict[str, float] = {
    # —— 图像题（5、6）——
    "组织结构图": 1.9,
    "销售部": 1.9,
    "大客户销售部": 1.9,
    "销售处": 1.7,
    "电话及网络销售部": 1.6,
    "渠道销售部": 1.6,
    "国际贸易部": 1.6,
    "上海分公司": 1.4,
    "市场开发部": 1.5,
    "信息系统部": 1.5,
    "行政人事部": 1.5,
    "物流部": 1.5,
    "研发中心": 1.5,
    "增长率": 1.9,
    "应用结构": 1.6,
    "IC卡": 1.7,
    "汽车电子": 1.6,
    "负增长": 1.6,
    "工业控制": 1.4,
    "赛迪顾问": 1.3,
    # —— 文本/表格题（1、2、3、4、260、95、33、34、957、793、795、543、531、207）——
    "发行股数": 1.8,
    "发行后总股本": 1.8,
    "募集资金": 1.8,
    "补充流动资金": 1.8,
    "营运资金": 1.6,
    "投资项目": 1.5,
    "关联方": 1.7,
    "实际控制人": 1.7,
    "控股股东": 1.7,
    "持股比例": 1.7,
    "本公司关系": 1.6,
    "注册资本": 1.8,
    "法定代表人": 1.8,
    "军用领域": 1.7,
    "主营业务收入": 1.6,
    "上游": 1.6,
    "下游": 1.6,
    "供应商": 1.5,
    "客户": 1.4,
    "技术标准": 1.7,
    "视频指挥": 1.6,
    "国家科技进步一等奖": 1.9,
    "重要供应商": 1.5,
    "电子信息行业": 1.5,
    "收入": 1.5,
    "占比": 1.5,
}

_SECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^第[一二三四五六七八九十百零〇0-9]{1,4}[节章]\s*\S{1,30}$"),
    re.compile(r"^[一二三四五六七八九十]{1,3}、\S{1,30}$"),
    re.compile(r"^（[一二三四五六七八九十]{1,3}）\S{1,30}$"),
)

# 编号/小标题标记：用于「段落内再分小点」时的**优先切分点**（设计 §7「按段落/小标题聚合」）。
# 实测依据（T5）：PDF1 物理 129 的段落把「经营模式…」与「①行业集中度高的影响 发行人产品主要应用于军队的信息化建设…
# 82.10%…」粘在同一个 chunk 里，导致答案句被前文主题稀释（BM25 仅第 5、向量前 100 都不进）→ 题 33/260 丢命中。
# 把该句切出来后：**BM25 排名第 2/2392**（实测），命中恢复。
_ENUM_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]"),
    re.compile(r"（[一二三四五六七八九十]{1,3}）(?=[^\s])"),
    re.compile(r"(?<=[。；;！？!?])\s*[0-9]{1,2}、(?=[^\s])"),
)


@dataclass(slots=True)
class Chunk:
    """检索/生成共用的最小单元。

    基础字段沿用 工单3 契约（保证既有向量/B​M25 种子可复用）；末段为**工单4 新增**的
    统一 chunk 契约字段（设计/系统架构.md §5.2：``modality``/``modality_sub``/``xref``/
    ``bbox``/``structured``/``image_sha1``），仅图像块必填，文本/表格块保持 ``type`` 语义。
    """

    chunk_id: str
    file_name: str
    page: int                       # = page_start（对外引用页，1-based 物理页）
    page_start: int
    page_end: int
    type: str                       # "text" | "table" | "image"（图像块为工单4 新增）
    section: str
    content: str
    keywords: list[str]
    table_id: str | None
    char_count: int
    order: int
    source_block_ids: list[str] = field(default_factory=list)
    # ---- 工单4 新增（统一 chunk 契约）----
    modality_sub: str | None = None          # diagram | chart | picture
    xref: int | None = None                  # 位图 xref；框图无位图则为 None
    bbox: list[float] | None = None          # [x0,y0,x1,y1]，PDF pt
    structured: dict[str, Any] | None = None  # 结构树 / 图表数值表（T2 原样透传）
    image_sha1: str | None = None
    extraction_method: str | None = None     # clip_zero_shot | geometry+anchor | geometry_only | text_layer_anchor
    source_chunk_id: str | None = None       # T2 原始 id（如 img:招股说明书2.pdf:p39:diagram:x1692）
    legacy_chunk_id: str | None = None       # 工单3 风格 id（对照只读基线用）
    sub_bboxes: list[list[float]] | None = None  # 组合图各子图 bbox（页72 左右并列两图）
    # 证据强度（检索侧建议权重；**不是**最终分数，T5 决定是否使用）：
    #   文本/表格 = 1.0；图像块中 diagram（几何+连线解析）/chart（几何+正文锚点）= 1.0；
    #   picture（页眉 banner、签章页大图等，只有 CLIP 零样本标签）= 0.4 —— 依据 T2 实测提醒：
    #   picture 类是**弱证据**，与 diagram/chart 同权会挤掉正文证据（T2 全量运行会产出大量 picture）。
    evidence_weight: float = 1.0

    @property
    def modality(self) -> str:
        """模态别名（= ``type``）：设计文档用 ``modality``，工单3 索引用 ``type``，两者同值。"""
        return self.type

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # 同时给出 modality 别名：满足 设计/系统架构.md §5.2 与 manifest 的 modality 口径，
        # 又不破坏工单3 既有的 type 字段（captain 2026-10 复核：不要另造字段名，加别名即可）。
        data["modality"] = self.type
        data["page_basis"] = "1-based-physical"
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Chunk":
        names = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def _lazy_logger(logger: Any, module: str) -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def corpus_key(file_name: str) -> str:
    """语料短名（chunk_id 前缀）：``招股说明书1.pdf`` → ``pdf1``、``招股说明书2.pdf`` → ``pdf2``。

    通用规则：文件名里若含「说明书N.pdf」取 ``pdfN``；否则用 stem 的安全化短名
    （**不硬编码文件名**，换成别的语料也能产出稳定 id）。
    """
    stem = str(file_name or "").rsplit(".", 1)[0]
    match = re.search(r"(\d+)\s*$", stem)
    if match and ("说明书" in stem or "pdf" in stem.lower()):
        return f"pdf{int(match.group(1))}"
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", stem)
    return (slug[:8] or "corpus").lower()


def chunk_id_for(file_name: str, page: int, kind: str, idx: int, *,
                 xref: int | None = None, part: int = 0) -> str:
    """按工单4 冻结口径生成 chunk_id（见模块 docstring）。

    ``kind`` ∈ ``{"x"(文本), "t"(表格), "img"(图像)}``；非法值抛 ``ChunkError``（不静默兜底）。
    表格的 ``part``：单块表 = 0（与 设计/接口设计.md 的 ``pdf2-p22-t3-0`` 完全一致），
    超长表拆多部分时递增。
    """
    if kind not in {"t", "x", "img"}:
        raise ChunkError(f"非法 chunk 类型：{kind}", detail={"kind": kind})
    ck = corpus_key(file_name)
    page_no = int(page)
    if kind == "img":
        xref_text = str(int(xref)) if xref is not None else "x"
        return f"{ck}-p{page_no}-img-{xref_text}-{int(idx)}"
    if kind == "t":
        return f"{ck}-p{page_no}-t{int(idx)}-{int(part)}"
    return f"{ck}-p{page_no}-x{int(idx)}"


def legacy_chunk_id_for(file_name: str, page: int, kind: str, idx: int) -> str:
    """工单3 风格 id（``招股说明书1_p0001_x00``）——仅用于与只读基线对照，不入 chunk_id。"""
    stem = file_name.rsplit(".", 1)[0]
    return f"{stem}_p{int(page):04d}_{kind}{int(idx):02d}"


def split_segments(text: str, *, min_segment: int = 60) -> list[str]:
    """把一段正文按「编号/小标题标记」切成若干语义小点（供分块时优先在这些位置切开）。

    返回的片段保持原文顺序；长度 < ``min_segment`` 的碎片会并回前一片段（避免切得过碎）。
    """
    body = str(text or "")
    if not body:
        return []
    cuts: set[int] = set()
    for pattern in _ENUM_MARKERS:
        for match in pattern.finditer(body):
            cuts.add(match.start())
    if not cuts:
        return [body]
    pieces: list[str] = []
    last = 0
    for position in sorted(cuts):
        if position - last < min_segment:
            continue                          # 与上一片段太近 → 不切（防碎片化）
        pieces.append(body[last:position])
        last = position
    pieces.append(body[last:])
    merged: list[str] = []
    for piece in pieces:
        if merged and len(piece.strip()) < min_segment:
            merged[-1] = merged[-1] + piece
        elif piece.strip():
            merged.append(piece)
    return [p.strip() for p in merged if p.strip()]


def detect_section(page_text: str, *, default: str = "") -> str:
    """从页文本中取页内最后一个「节/章/条」级标题；取不到回退 ``default``。"""
    found = ""
    for line in (str(page_text or "").splitlines()):
        text = line.strip()
        if not text or len(text) > 40:
            continue
        if any(pattern.match(text) for pattern in _SECTION_PATTERNS):
            found = text
    return found or default


def split_text(text: str, *, size: int = 500, overlap: int = 80, min_size: int = 120) -> list[str]:
    """按句切分文本：目标块长 ``size``、相邻块共享 ``overlap`` 字符、尾块 < ``min_size`` 并入前块。"""
    body = collapse_whitespace(text)
    if not body:
        return []
    size = max(int(size), 50)
    overlap = max(min(int(overlap), size - 1), 0)
    pieces: list[str] = []
    current = ""
    for sentence in split_sentences(body):
        if len(sentence) > size:
            # 单句超长：先冲掉当前块，再对超长句硬切
            if current:
                pieces.append(current)
                current = ""
            step = max(size - overlap, 1)
            for start in range(0, len(sentence), step):
                pieces.append(sentence[start:start + size])
            current = ""
            continue
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= size:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            # overlap 前缀必须**按剩余空间裁剪**：否则 80 字重叠 + 接近 size 的句子会产出
            # 超过上限的块（实测出现 606 字文本块，正是此处无上限判断所致）。
            keep = max(size - len(sentence) - 1, 0)
            tail = current[-min(overlap, keep):] if keep > 0 else ""
            current = f"{tail} {sentence}".strip() if tail else sentence
    if current:
        pieces.append(current)
    if len(pieces) > 1 and len(pieces[-1]) < int(min_size):
        merged = f"{pieces[-2]} {pieces[-1]}".strip()
        pieces[-2] = merged
        pieces.pop()
    return [p for p in pieces if p.strip()]


def extract_keywords(text: str, *, topk: int = 8, weight_dict: Mapping[str, float] | None = None) -> list[str]:
    """关键词（委托 ``text_utils.extract_keywords``）：默认使用工单的关键字段加权词典。"""
    weights = KEYWORD_WEIGHTS if weight_dict is None else weight_dict
    return _text_extract_keywords(text, topk=topk, weight_dict=weights)


def _keyword_source(content: str) -> str:
    """关键词投影：去掉 content 模板标签（``[表标题]`` 等）与 Markdown 竖线/分隔行。

    实测教训：直接用 content 抽关键词会得到 ``---``、``1``、``2`` 这类噪声。
    """
    kept: list[str] = []
    for line in str(content or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "]" in stripped:
            continue
        if stripped and set(stripped) <= set("|-: "):
            continue
        kept.append(line.replace("|", " "))
    return " ".join(kept)


# ---------------------------------------------------------------------------
# 表格 chunk
# ---------------------------------------------------------------------------
def _table_source_line(block: TableBlock) -> str:
    """来源行（架构 §6.4）：单页写「第 N 页」，跨页写「第 A–B 页（表头页 A）」。"""
    if block.page_start == block.page_end:
        return f"{block.file_name} 第 {block.page_start} 页"
    return f"{block.file_name} 第 {block.page_start}-{block.page_end} 页（表头页 {block.page_start}）"


def _table_content(block: TableBlock, *, part: int | None = None, parts: int | None = None,
                   markdown: str | None = None, numbers: Sequence[str] | None = None) -> str:
    """按 §6.4 组装表格块 content（表标题 + 来源 + Markdown + 关键数字）。"""
    source = _table_source_line(block)
    if part is not None and parts is not None and parts > 1:
        source = f"{source}（第 {part}/{parts} 部分）"
    body = markdown if markdown is not None else block.markdown
    nums = list(numbers if numbers is not None else block.key_numbers)
    return "\n".join([
        f"[表标题] {block.title}",
        f"[来源] {source}",
        "[Markdown]",
        body,
        f"[关键数字] {' '.join(nums)}",
    ])


def _split_table_parts(block: TableBlock, cfg: AppConfig) -> list[tuple[str, list[str]]]:
    """超长逻辑表 → 按行批次拆（每部分保留表标题与来源行，共享 table_id）。"""
    limit = int(cfg.chunk.max_table_chars)
    full = _table_content(block)
    if len(full) <= limit or not block.rows:
        return [(full, list(block.key_numbers))]
    batches: list[list[list[str]]] = []
    current: list[list[str]] = []
    current_len = len(_table_content(block, markdown="", numbers=[]))
    for row in block.rows:
        row_len = len(" | ".join(str(c) for c in row)) + 4
        if current and current_len + row_len > limit:
            batches.append(current)
            current = []
            current_len = len(_table_content(block, markdown="", numbers=[]))
        current.append(row)
        current_len += row_len
    if current:
        batches.append(current)
    total = len(batches)
    parts: list[tuple[str, list[str]]] = []
    for index, rows in enumerate(batches, 1):
        markdown = _markdown_for(block.header, rows)
        numbers = _numbers_of(markdown, block.header)
        parts.append((_table_content(block, part=index, parts=total, markdown=markdown, numbers=numbers), numbers))
    return parts


def _markdown_for(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    from .table_parser import table_to_markdown

    return table_to_markdown(header, rows)


def _numbers_of(markdown: str, header: Sequence[str]) -> list[str]:
    from .text_utils import extract_numbers

    return extract_numbers(markdown + " " + " ".join(header))[:24]


# ---------------------------------------------------------------------------
# 图像块 chunk（工单4 新增：把 T2 的图像产物接入同一索引）
# ---------------------------------------------------------------------------
def _merge_structured(items: Sequence[Any]) -> Any:
    """递归合并多个 ``structured``：dict 逐键合并、list 拼接、标量取首个非空值。"""
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    if all(isinstance(item, dict) for item in items):
        merged: dict[str, Any] = {}
        for item in items:
            for key, value in item.items():
                if key in merged and isinstance(merged[key], (dict, list)) and isinstance(value, (dict, list)):
                    merged[key] = _merge_structured([merged[key], value])
                elif key not in merged or merged[key] in (None, "", [], {}):
                    merged[key] = value
        return merged
    if all(isinstance(item, list) for item in items):
        joined: list[Any] = []
        for item in items:
            joined.extend(item)
        return joined
    for item in items:
        if item not in (None, "", [], {}):
            return item
    return items[0]


def merge_image_blocks(records: Sequence[Mapping[str, Any]], *, logger: Any = None) -> list[dict[str, Any]]:
    """把「同页 + 同 modality_sub」的多个图像块合成 **1 个** 图像 chunk 记录。

    依据（captain 2026-10 复核 + T2 实测）：招股书2 第 72 页是**左右并列的两张子图**
    （左=3D 饼图 xref1693、右=条形图 xref1692，bbox 仅相接 0.9pt），属同一张组合图；
    第 6 题要同时用到「汽车 14.0% 最快」与「IC卡 -2.0% 负增长」，**分开存会导致单块证据不全**。
    合并后保留每个子图的 xref/bbox 到 ``sub_blocks`` 与 ``sub_bboxes``，仍可回溯渲染。

    合并键 = ``(corpus, page, modality_sub)``；不同 ``modality_sub``（图框/图表/其它图）不混并。
    """
    log = _lazy_logger(logger, "chunker")
    groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    order: list[tuple[str, int, str]] = []
    for record in records:
        page = int(record.get("page") or 0)
        if page <= 0:
            raise ChunkError("图像块缺少 1-based 物理页码 page，拒绝入索引",
                             code="RAG-2202", detail={"record_keys": sorted(record.keys())[:12]})
        key = (str(record.get("corpus") or ""), page, str(record.get("modality_sub") or "picture"))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)

    merged_records: list[dict[str, Any]] = []
    for key in order:
        items = groups[key]
        head = dict(items[0])
        if len(items) > 1:
            head["text"] = "\n".join(str(item.get("text") or "").strip() for item in items if str(item.get("text") or "").strip())
            head["structured"] = _merge_structured([item.get("structured") for item in items])
            head["sub_blocks"] = [
                {"source_chunk_id": item.get("chunk_id"), "sub_index": index,
                 "xref": item.get("xref"), "bbox": item.get("bbox"), "image_sha1": item.get("image_sha1"),
                 "modality_sub": item.get("modality_sub"), "extraction_method": item.get("extraction_method")}
                for index, item in enumerate(items)
            ]
            head["merged_from"] = [item.get("chunk_id") for item in items]
            log.log_event("chunker.image_merged", corpus=key[0], page=key[1], modality_sub=key[2],
                          parts=len(items), xrefs=[item.get("xref") for item in items])
        merged_records.append(head)
    return merged_records


def load_image_blocks(processed_dir: Path | str, *, logger: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """自动发现并读取图像块文件（T2 产物）。返回 ``(records, 诊断)``；找不到返回 ``([], 诊断)``。

    只读 ``image_blocks.jsonl`` / ``image_candidates.jsonl`` / ``image_blocks.json``；
    **显式拒绝** ``images.jsonl``（含水印/装饰图，T2 标记 keep=false，混入会污染检索）。
    """
    log = _lazy_logger(logger, "chunker")
    from .config import IMAGE_BLOCK_DENYLIST, IMAGE_BLOCK_FILES

    directory = Path(processed_dir)
    diagnosis: dict[str, Any] = {"dir": str(directory), "candidates": [], "used": None, "skipped": []}
    for name in IMAGE_BLOCK_DENYLIST:
        path = directory / name
        if path.is_file():
            diagnosis["skipped"].append(name)
            log.log_event("chunker.image_denylist_skipped", level="WARNING", path=str(path),
                          reason="该清单含 keep=false 的水印/装饰图，禁止直接入索引")
    for name in IMAGE_BLOCK_FILES:
        path = directory / name
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows = payload.get("records") if isinstance(payload, dict) else payload
                records = [dict(row) for row in (rows or [])]
            else:
                records = [dict(row) for row in read_jsonl(path)]
        except (OSError, ValueError) as exc:
            log.log_event("chunker.image_blocks_unreadable", level="ERROR", path=str(path),
                          error_type=type(exc).__name__, message=str(exc))
            raise ChunkError(f"图像块文件无法读取：{path}（{type(exc).__name__}: {exc}）",
                             code="RAG-2203", detail={"path": str(path)}) from exc
        diagnosis["candidates"].append({"file": str(path), "records": len(records)})
        diagnosis["used"] = str(path)
        log.log_event("chunker.image_blocks_loaded", path=str(path), records=len(records))
        return records, diagnosis
    log.log_event("chunker.image_blocks_missing", level="WARNING", dir=str(directory),
                  tried=list(IMAGE_BLOCK_FILES))
    return [], diagnosis


def build_image_chunks(records: Sequence[Mapping[str, Any]], *, logger: Any = None) -> list[Chunk]:
    """把（已合并的）图像块记录转成 ``type="image"`` 的 :class:`Chunk`。"""
    log = _lazy_logger(logger, "chunker")
    chunks: list[Chunk] = []
    per_page_idx: dict[tuple[str, int], int] = {}
    for record in records:
        file_name = str(record.get("corpus") or record.get("file_name") or "").strip()
        if not file_name:
            raise ChunkError("图像块缺少 corpus/file_name，无法归属语料", code="RAG-2202",
                             detail={"chunk_id": record.get("chunk_id")})
        page = int(record.get("page"))
        modality_sub = str(record.get("modality_sub") or "picture")
        xref = record.get("xref")
        if isinstance(xref, (list, tuple)):
            xref_first = int(xref[0]) if xref else None
        else:
            xref_first = int(xref) if xref is not None else None
        text = str(record.get("text") or "").strip()
        if not text:
            # 不伪造内容：text 为空则用结构化字段拼一个**如实**的最小检索文本
            text = f"（图像块）{file_name} 第 {page} 页 {modality_sub} 图像（无文本描述：T2 未提供 text 字段）"
            log.log_event("chunker.image_empty_text", level="WARNING", page=page,
                          modality_sub=modality_sub, source_chunk_id=record.get("chunk_id"))
        bbox_raw = record.get("bbox")
        bbox = [float(v) for v in bbox_raw] if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4 else None
        sub_bboxes: list[list[float]] = []
        for sub in (record.get("sub_blocks") or []):
            box = sub.get("bbox") if isinstance(sub, dict) else None
            if isinstance(box, (list, tuple)) and len(box) == 4:
                sub_bboxes.append([float(v) for v in box])
        if bbox is None and sub_bboxes:
            bbox = [min(b[0] for b in sub_bboxes), min(b[1] for b in sub_bboxes),
                    max(b[2] for b in sub_bboxes), max(b[3] for b in sub_bboxes)]
        key = (file_name, page)
        seq = per_page_idx.get(key, 0)
        per_page_idx[key] = seq + 1
        chunk = Chunk(
            chunk_id=chunk_id_for(file_name, page, "img", seq, xref=xref_first),
            file_name=file_name, page=page, page_start=page, page_end=page, type="image",
            section=str(record.get("section") or ""), content=text,
            keywords=extract_keywords(_keyword_source(text), topk=10, weight_dict=KEYWORD_WEIGHTS),
            table_id=None, char_count=len(text), order=-1,
            source_block_ids=[str(record.get("chunk_id") or "")],
            modality_sub=modality_sub, xref=xref_first, bbox=bbox,
            structured=record.get("structured") if isinstance(record.get("structured"), dict) else None,
            image_sha1=(record.get("image_sha1") if isinstance(record.get("image_sha1"), str) else None),
            extraction_method=str(record.get("extraction_method") or "") or None,
            source_chunk_id=str(record.get("chunk_id") or "") or None,
            sub_bboxes=sub_bboxes or None,
            evidence_weight=1.0 if modality_sub in {"diagram", "chart"} else 0.4,
        )
        chunks.append(chunk)
        log.log_event("chunker.image_chunk", chunk_id=chunk.chunk_id, page=page,
                      modality_sub=modality_sub, xref=xref_first, chars=len(text))
    log.log_event("chunker.image_chunks_done", chunks=len(chunks))
    return chunks


def descramble_vertical_text(page_text: str) -> str:
    """把「竖排单字各占一行」的页文本复原为连续词（工单4 图像题降级路径）。

    规则（保守，只做可验证的事）：把**连续的单字符行**按出现顺序拼成一个词，
    长度限制 2~12；空行/多字符行原样保留。实测依据（T1）：招股书2 第 39 页的
    财务部/市场开发部/…/大客户销售部/珠海销售处等竖直标签，其字符在 ``get_text("text")``
    里正是**连续**出现的（按 y 聚类反而会串成「财场息销政物」乱码，故不能按 y 聚类）。
    """
    lines = [line.strip() for line in str(page_text or "").splitlines()]
    out: list[str] = []
    run: list[str] = []
    for line in lines:
        if len(line) == 1 and not line.isspace():
            run.append(line)
            continue
        if run:
            word = "".join(run)
            if 2 <= len(word) <= 12:
                out.append(word)
            else:
                out.extend(run)
            run = []
        if line:
            out.append(line)
    if run:
        word = "".join(run)
        out.append(word) if 2 <= len(word) <= 12 else out.extend(run)
    return "\n".join(out)


def build_image_chunks_from_text_layer(source: Any, pages: Sequence[int], *, logger: Any = None) -> list[Chunk]:
    """**降级路径**（T2 的 image_blocks.jsonl 尚未落盘时）：用页面自身文本层生成图像块。

    严格边界（不得越界，也不得伪造）：
        * 只读 PDF 文本层，**不做** CLIP 语义编码、**不做**图表像素反算 → 不产生任何
          「汽车 14.0%」这类图表数值；文本里有什么就是什么；
        * ``extraction_method = "text_layer_anchor"``、``structured = {"degraded": True, ...}``
          如实标注降级，检索层与报告都能看出来；
        * 一旦 T2 的 ``image_blocks.jsonl`` 落盘，重跑 ``--incremental --include-images``
          即自动改用 T2 版本（本降级块不再产生）。
    """
    log = _lazy_logger(logger, "chunker")
    records: list[dict[str, Any]] = []
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover —— 环境缺 pymupdf 时无法降级，显式抛出
        raise ChunkError(f"降级路径需要 pymupdf：{exc}", code="RAG-2204") from exc
    file_name = getattr(source, "file_name", None) or Path(str(source)).name
    path = getattr(source, "path", source)
    doc = pymupdf.open(str(path))
    try:
        for page_no in pages:
            page = doc.load_page(int(page_no) - 1)
            raw = page.get_text("text") or ""
            lines = [line.strip() for line in descramble_vertical_text(raw).splitlines() if line.strip()]
            kept: list[str] = []
            for line in lines:
                if _is_page_noise(line, page_no):
                    continue
                if line not in kept:
                    kept.append(line)
            # 竖排标签（按 x 列聚类还原）+ 横向短行（节点名/图题/资料来源）+ 长段正文（含正文锚点句）
            labels = extract_vertical_labels(page)
            short_lines = [line for line in kept if len(line) <= 12]
            long_lines = [line for line in kept if len(line) > 12][:12]
            node_names = list(dict.fromkeys(labels + short_lines))
            try:
                images = page.get_images(full=True) or []
            except Exception as exc:  # noqa: BLE001 —— 取图失败不阻断，但如实记录
                images = []
                log.log_event("chunker.fallback_images_error", level="WARNING", page=page_no,
                              error_type=type(exc).__name__, message=str(exc))
            text = (f"{file_name} 第 {page_no} 页 图像（本页图片资源 {len(images)} 个）。"
                    f"图中文字/节点标签（竖排单字已按 x 列聚类 + y 升序复原）：{'、'.join(node_names)}。"
                    f"本页正文（含图题与资料来源）：{' '.join(long_lines)}")
            records.append({
                "chunk_id": f"fallback:{file_name}:p{page_no}",
                "corpus": file_name, "page": int(page_no),
                # 降级路径**只读文本层**，无从判断图是框图还是图表 → 一律按最保守的 "picture" 记，
                # 绝不冒充 T2 的 diagram/chart 结果。
                "modality_sub": "picture",
                "xref": None, "bbox": None, "image_sha1": None,
                "text": text,
                "structured": {"degraded": True, "reason": "T2 的 image_blocks.jsonl 尚未落盘",
                               "image_count": len(images), "vertical_labels": labels,
                               "node_names": node_names, "text_layer": " ".join(kept)},
                "extraction_method": "text_layer_anchor",
            })
    finally:
        doc.close()
    log.log_event("chunker.image_fallback_built", level="WARNING", file_name=file_name,
                  pages=list(pages), records=len(records))
    return build_image_chunks(merge_image_blocks(records, logger=log), logger=log)


def extract_vertical_labels(page: Any, *, x_tol: float = 6.0, min_len: int = 2,
                            max_len: int = 12) -> list[str]:
    """从页面 span 坐标还原**竖排标签**（工单4 降级路径用；方法同 T1 实测口径）。

    做法（T1 在 设计/环境事实.md §5 实测的方法）：
        ① 取 ``page.get_text("dict")`` 的全部 span，筛出**单字符** span；
        ② 按 x0 聚类（容差 ``x_tol`` = 6 pt）成「列」；
        ③ 列内按 y0 升序拼接 → 该列的竖排标签文本；
        ④ 长度落在 [``min_len``, ``max_len``] 的列才算标签。

    为什么不能只按行拼接（反面示例，T1 已实测）：页 39 的 17 个部门/销售处标签**首尾相接成一个
    长达 85 字的单字流**（「财 务 部 市 场 开 发 部 … 成 都 销 售 处」），任何「按长度截断」的启发式都会串错；
    按 x 列聚类才能还原成 财务部 / 市场开发部 / … / 大客户销售部 / 珠海销售处。
    """
    try:
        data = page.get_text("dict")
    except Exception:  # noqa: BLE001 —— 取不到结构化文本就返回空表（调用方会如实降级）
        return []
    singles: list[tuple[float, float, str]] = []
    for block in data.get("blocks", []) or []:
        for line in block.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                text = str(span.get("text") or "").strip()
                if len(text) != 1 or text.isspace():
                    continue
                bbox = span.get("bbox") or (0.0, 0.0, 0.0, 0.0)
                singles.append((float(bbox[0]), float(bbox[1]), text))
    singles.sort(key=lambda item: (round(item[0], 1), item[1]))
    columns: list[list[tuple[float, float, str]]] = []
    for item in singles:
        for column in columns:
            if abs(column[-1][0] - item[0]) <= float(x_tol):
                column.append(item)
                break
        else:
            columns.append([item])
    labels: list[str] = []
    for column in columns:
        column.sort(key=lambda item: item[1])
        word = "".join(item[2] for item in column)
        if min_len <= len(word) <= max_len:
            labels.append(word)
    return labels


def _is_page_noise(line: str, page_no: int) -> bool:
    """降级路径去噪：页眉（含「招股意向书/招股说明书」）、裸页码（= 物理页 − 1）、超短行。"""
    text = str(line or "").strip()
    if not text or len(text) <= 1:
        return True
    if "招股意向书" in text or "招股说明书" in text:
        return True
    if text.isdigit() and int(text) in {int(page_no), int(page_no) - 1}:
        return True
    return False


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_chunks(
    parse_result: PdfParseResult,
    *,
    cfg: AppConfig | None = None,
    logger: Any = None,
) -> list[Chunk]:
    """把解析结果转成 chunk 列表（表格整体成块，文本按 400~600 字切分）。"""
    config = cfg or get_config()
    log = _lazy_logger(logger, "chunker")
    file_name = parse_result.file_name
    stem = file_name.rsplit(".", 1)[0]
    with log.enter("build_chunks", {"file_name": file_name, "pages": parse_result.page_count,
                                    "tables": len(parse_result.tables),
                                    "text_blocks": len(parse_result.text_blocks)}) as span:
        t0 = time.perf_counter()
        chunks: list[Chunk] = []
        seen_ids: set[str] = set()
        table_idx = 0
        page_table_idx: dict[int, int] = {}
        page_text_idx: dict[int, int] = {}
        text_chunks = 0
        table_chunks = 0
        total_chars = 0

        # ---- 1) 表格块：逻辑表块整块成 1 chunk（超长才拆部分）----
        for block in sorted(parse_result.tables, key=lambda b: (b.page_start, b.table_id)):
            if block.degenerate:
                log.log_event("chunker.skip_degenerate", level="WARNING", table_id=block.table_id,
                              page=block.page_start, reason="退化表不进正文 chunk")
                continue
            parts = _split_table_parts(block, config)
            if len(parts) > 1:
                log.log_event("chunker.table_split", level="WARNING", table_id=block.table_id,
                              page_start=block.page_start, page_end=block.page_end, parts=len(parts),
                              chars=len(_table_content(block)))
            section = detect_section(parse_result.pages[block.page_start - 1].text if
                                     block.page_start - 1 < len(parse_result.pages) else "", default="")
            for part_idx, (content, _numbers) in enumerate(parts):
                page_seq = page_table_idx.get(block.page_start, 0)
                cid = chunk_id_for(file_name, block.page_start, "t", page_seq, part=part_idx)
                if cid in seen_ids:
                    raise ChunkError(f"chunk_id 重复：{cid}", code="RAG-2201", detail={"chunk_id": cid})
                seen_ids.add(cid)
                chunk = Chunk(
                    chunk_id=cid, file_name=file_name, page=block.page_start, page_start=block.page_start,
                    page_end=block.page_end, type="table", section=section, content=content,
                    keywords=extract_keywords(_keyword_source(content), topk=8, weight_dict=KEYWORD_WEIGHTS),
                    table_id=block.table_id, char_count=len(content), order=-1,
                    source_block_ids=[block.table_id],
                    legacy_chunk_id=legacy_chunk_id_for(file_name, block.page_start, "t", table_idx),
                )
                if block.page_start != block.page_end:
                    log.log_event("chunker.continued_chunk", chunk_id=cid, table_id=block.table_id,
                                  page_start=block.page_start, page_end=block.page_end,
                                  absorbed_pages=block.absorbed_pages)
                chunks.append(chunk)
                page_table_idx[block.page_start] = page_seq + 1
                table_idx += 1
                table_chunks += 1
                total_chars += len(content)

        # ---- 2) 文本块：**按段落（TextBlock）边界聚合** → 清洗页眉页脚 → 仅在段落超长时按句切分 ----
        # 设计 §7 要求「按段落/小标题聚合并按 chunk_size 切分」。
        # 实测教训（T5）：早期实现只按字数累积、忽略段落边界，会把「答案句」粘在
        # 另一个主题段落的长块尾部（如 PDF1 物理 129：477 字块前 380 字讲民品客户，
        # 真正的「军用收入占比 82.10%…」只在最后 95 字）→ 该块语义被前文主导，检索排到第 8~21 名，题 33/260 丢命中。
        by_page: dict[int, list[TextBlock]] = {}
        for block in parse_result.text_blocks:
            by_page.setdefault(block.page, []).append(block)
        last_section = ""
        for page in sorted(by_page):
            blocks = sorted(by_page[page], key=lambda b: b.block_index)
            raw_text = "\n".join(b.text for b in blocks)
            page_text = strip_printed_page(raw_text, page=page)
            if not page_text.strip():
                continue
            section = detect_section(page_text, default=last_section)
            last_section = section
            pieces: list[str] = []
            buffer = ""
            used_blocks: list[str] = []
            min_size = int(config.chunk.min_size)
            for block in blocks:
                # 逐块清洗（页眉页脚只在块首/块尾出现，按页清洗更稳 → 用整页清洗结果兜底）
                block_text = strip_printed_page(block.text, page=page)
                block_text = collapse_whitespace(block_text)
                if not block_text:
                    continue
                if len(block_text) > int(config.chunk.size):
                    # 超长单段：先把已有缓冲冲掉，再按句切分该段
                    if buffer:
                        pieces.append(buffer)
                        buffer, used_blocks = "", []
                    for piece in split_text(block_text, size=config.chunk.size,
                                            overlap=config.chunk.overlap,
                                            min_size=config.chunk.min_size):
                        pieces.append(piece)
                    continue
                # **小标题/编号点优先切分**：让每个语义小点尽量独立成块，避免答案句被前文主题稀释
                for segment in split_segments(block_text):
                    starts_with_marker = bool(
                        any(pattern.match(segment[:2]) for pattern in _SECTION_PATTERNS)
                        or any(marker.match(segment) for marker in _ENUM_MARKERS)
                    )
                    if buffer and starts_with_marker and len(buffer) >= min_size:
                        pieces.append(buffer)
                        buffer, used_blocks = "", []
                    if buffer and len(buffer) + 1 + len(segment) > int(config.chunk.size):
                        pieces.append(buffer)                    # **在段落边界切开**
                        buffer, used_blocks = "", []
                    buffer = f"{buffer} {segment}".strip() if buffer else segment
                    if block.block_id not in used_blocks:
                        used_blocks.append(block.block_id)
            if buffer:
                pieces.append(buffer)
            # 最终硬上限兜底：任何仍超过 size+100 的块再按句切一刀
            # （覆盖 overlap 前缀、页内再平衡、尾块并入等所有路径；实测修复后 >600 的块 14 → 0）
            max_chars = int(config.chunk.size) + 100
            if any(len(piece) > max_chars for piece in pieces):
                normalized: list[str] = []
                for piece in pieces:
                    if len(piece) > max_chars:
                        normalized.extend(split_text(piece, size=config.chunk.size,
                                                    overlap=config.chunk.overlap,
                                                    min_size=config.chunk.min_size))
                    else:
                        normalized.append(piece)
                pieces = normalized
            # 页内再平衡：末块 < min_size 且前块有余量时，从句边界把前块的尾句搬给末块，
            # 使**两侧**都落在 [min_size, size+100]。这样既避免「末块过小」，也避免「并入前块后越界」
            # （设计 §3.7 的 400~600 与「尾块并入前块」两条在边界上会冲突，本函数同时满足两者的可行区间）。
            if len(pieces) >= 2 and len(pieces[-1]) < min_size:
                max_chars = int(config.chunk.size) + 100
                prev_text, tail_text = pieces[-2], pieces[-1]
                guard = 0
                while len(tail_text) < min_size and len(prev_text) > min_size and guard < 8:
                    cut = max(prev_text.rfind("。"), prev_text.rfind("；"), prev_text.rfind("！"))
                    if cut <= 0 or (len(prev_text) - cut) > 150:
                        break                       # 找不到合适的句边界或移出量过大 → 停止
                    moved = prev_text[cut + 1:]
                    if len(tail_text) + len(moved) > max_chars:
                        break                       # 搬过来会越上限 → 停止
                    tail_text = moved + tail_text
                    prev_text = prev_text[:cut + 1]
                    guard += 1
                pieces[-2], pieces[-1] = prev_text.strip(), tail_text.strip()
                log.log_event("chunker.page_rebalance", file_name=file_name, page=page,
                              moved_sentences=guard, prev_chars=len(pieces[-2]),
                              tail_chars=len(pieces[-1]))
            for index, piece in enumerate(pieces):
                is_tail = index == len(pieces) - 1
                # 尾块（< min_size）并入**本文件最近的前一个文本块**，但必须守住设计 §3.7 的文本块上限
                # （size=500 → 上限 600）。实测教训（captain 复核发现）：早期实现无条件并入，
                # 导致 14 个文本块越界（最大 697，特征 = 块内含尾块并入留下的换行符）。
                # 越界时宁可为独立小尾块，也不破坏尺寸契约；并写 chunker.tail_standalone 留痕。
                if is_tail and len(piece) < min_size and chunks:
                    max_chars = int(config.chunk.size) + 100
                    target = None
                    for candidate in reversed(chunks):
                        if candidate.file_name != file_name:
                            break
                        if candidate.type == "text":
                            target = candidate
                            break
                    if target is not None and len(target.content) + 1 + len(piece) <= max_chars:
                        # 尾巴并入前块（设计 §3.7）：延长前块的 page_end，不新开块
                        target.content = f"{target.content}\n{piece}".strip()
                        target.char_count = len(target.content)
                        target.page_end = max(target.page_end, page)
                        target.keywords = extract_keywords(_keyword_source(target.content), topk=8,
                                                           weight_dict=KEYWORD_WEIGHTS)
                        target.source_block_ids.append(f"{stem}#p{page:04d}#tail")
                        total_chars = total_chars + len(piece)
                        continue
                    log.log_event("chunker.tail_standalone", level="WARNING",
                                  file_name=file_name, page=page, chars=len(piece),
                                  reason=("前块并入后超上限" if target is not None else "无同类前块可并入"),
                                  prev_chars=(len(target.content) if target is not None else 0),
                                  max_chars=max_chars)
                cid = chunk_id_for(file_name, page, "x", page_text_idx.get(page, 0))
                if cid in seen_ids:
                    raise ChunkError(f"chunk_id 重复：{cid}", code="RAG-2201", detail={"chunk_id": cid})
                seen_ids.add(cid)
                chunks.append(Chunk(
                    chunk_id=cid, file_name=file_name, page=page, page_start=page, page_end=page,
                    type="text", section=section, content=piece,
                    keywords=extract_keywords(_keyword_source(piece), topk=8, weight_dict=KEYWORD_WEIGHTS),
                    table_id=None, char_count=len(piece), order=-1,
                    source_block_ids=used_blocks or [b.block_id for b in blocks],
                    legacy_chunk_id=legacy_chunk_id_for(file_name, page, "x", text_chunks),
                ))
                page_text_idx[page] = page_text_idx.get(page, 0) + 1
                text_chunks += 1
                total_chars += len(piece)

        # ---- 3) 全局 order：按 (物理页, 表格 > 图像 > 文本, chunk_id) 排序 ----
        rank = {"table": 0, "image": 1, "text": 2}
        chunks.sort(key=lambda c: (c.page_start, rank.get(c.type, 9), c.chunk_id))
        for order, chunk in enumerate(chunks):
            chunk.order = order
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.log_event("chunker.done", file_name=file_name, text_chunks=text_chunks,
                      table_chunks=table_chunks, total_chunks=len(chunks), total_chars=total_chars,
                      elapsed_ms=elapsed_ms)
        span.set_output({"file_name": file_name, "text_chunks": text_chunks,
                         "table_chunks": table_chunks, "total_chunks": len(chunks),
                         "total_chars": total_chars})
        return chunks


def write_chunks(chunks: Sequence[Chunk], path: Path | str, *, overwrite: bool = False) -> int:
    """写 ``chunks.jsonl``（按 order 递增），返回行数。"""
    ordered = sorted(chunks, key=lambda c: c.order)
    return write_jsonl(path, (c.to_dict() for c in ordered), overwrite=overwrite)


def persist_chunks(
    chunks: Sequence[Chunk],
    *,
    db_path: Path | str | None = None,
    logger: Any = None,
) -> int:
    """把 chunk 写入 SQLite ``chunks`` 表（DDL 复用设计 §5.3；``order`` → ``ord``）。"""
    log = _lazy_logger(logger, "chunker")
    target = db_path if db_path is not None else get_config().paths.index_dir / "rag.sqlite3"
    with log.enter("persist_chunks", {"db": str(target), "chunks": len(chunks)}) as span:
        t0 = time.perf_counter()
        conn = connect_sqlite(target)
        try:
            # 先删除这批 chunk 所属文件的**旧行**：重建（重新分块）后 chunk_id 会变化，
            # 只做 INSERT OR REPLACE 会把上一版的分块永久留在表里（实测出现 4565 行 vs 实际 2391）。
            files = sorted({c.file_name for c in chunks})
            with conn:
                if files:
                    conn.executemany("DELETE FROM chunks WHERE file_name = ?", [(name,) for name in files])
                conn.executemany(
                    "INSERT OR REPLACE INTO chunks(chunk_id,file_name,page,page_start,page_end,type,section,"
                    "content,keywords,table_id,char_count,ord,source_block_ids) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            c.chunk_id, c.file_name, c.page, c.page_start, c.page_end, c.type, c.section,
                            c.content, json.dumps(c.keywords, ensure_ascii=False), c.table_id, c.char_count,
                            c.order, json.dumps(c.source_block_ids, ensure_ascii=False),
                        )
                        for c in chunks
                    ],
                )
                # 工单4 新增：主表 columns 冻结（不动 工单3 schema），把**完整** chunk（含
                # modality_modality_sub/xref/bbox/structured/image_sha1）另存 side table，
                # 供检索层与评测直接取结构化字段（SQLite 侧无需 ALTER 主表，零破坏）。
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS chunk_payloads("
                    "chunk_id TEXT PRIMARY KEY, file_name TEXT NOT NULL, page INTEGER NOT NULL,"
                    "type TEXT NOT NULL, payload TEXT NOT NULL)"
                )
                if files:
                    conn.executemany("DELETE FROM chunk_payloads WHERE file_name = ?", [(name,) for name in files])
                conn.executemany(
                    "INSERT OR REPLACE INTO chunk_payloads(chunk_id,file_name,page,type,payload)"
                    " VALUES(?,?,?,?,?)",
                    [
                        (c.chunk_id, c.file_name, c.page, c.type,
                         json.dumps(c.to_dict(), ensure_ascii=False))
                        for c in chunks
                    ],
                )
        except sqlite3.Error as exc:
            log.log_event("sqlite.error", level="ERROR", db=str(target), table="chunks",
                          error_type=type(exc).__name__, message=str(exc))
            raise wrap(exc, code="RAG-7000", stage="storage", db_path=str(target)) from exc
        finally:
            conn.close()
        log.log_event("sqlite.upsert", db=str(target), chunks=len(chunks),
                      elapsed_ms=round((time.perf_counter() - t0) * 1000, 2))
        span.set_output({"db": str(target), "chunks": len(chunks)})
        return len(chunks)


def iter_chunk_dicts(chunks: Iterable[Chunk]) -> Iterable[dict[str, Any]]:
    """便捷：chunk → dict 迭代。"""
    for chunk in chunks:
        yield chunk.to_dict()
