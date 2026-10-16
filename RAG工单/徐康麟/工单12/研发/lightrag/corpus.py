# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""语料构建：PDF → 文本块 / 表格块 / 图像块。

处理口径（与 设计/LightRAG架构设计.md「索引构建流程」一一对应）：
  1. 逐页抽文本，按「段落边界 → 句末标点 → 硬切」三级回退分块，块间保留重叠；
  2. 表格：pymupdf ``find_tables()`` 优先，失败时用「多行等距空白」启发式兜底；
  3. 图像：**本机无 OCR、Ollama 无视觉模型**，因此图像只做
     「图页/图注关联 + 文本描述」—— 即抽取图片元信息 + 抓取图附近正文作为 caption，
     生成一条可检索的图像块。**不做也不声称任何 OCR**。

产物（``研发/data/processed/``）：
  chunks.jsonl      统一块表（text/table/image 三类，同一 schema）
  images.jsonl      图像块明细（含 bbox、尺寸、caption 关联）
  corpus_stats.json 语料统计（真实计数）
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pymupdf

from .config import Cfg

# 句末标点（中文优先）
SENT_END = "。！？；\n"
# 图注前缀
CAPTION_PREFIX = re.compile(r"^\s*(图|表|图表|附图|资料来源|数据来源|注[:：])")
# 表格启发式：一行里出现 2 个以上 2+ 连续空格
TABLE_ROW_HINT = re.compile(r"\S {2,}\S")


@dataclass
class Chunk:
    """统一块结构（文本/表格/图像同类）。"""

    chunk_id: str
    file_name: str
    doc_key: str
    page: int
    page_end: int
    type: str            # text | table | image
    section: str
    content: str
    keywords: list[str] = field(default_factory=list)
    char_count: int = 0
    order: int = 0
    image_sha1: str | None = None
    bbox: list[float] | None = None
    width: int | None = None
    height: int | None = None
    caption: str = ""
    page_basis: str = "1-based-physical"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _dhash(image_bytes: bytes, *, size: int = 8) -> str:
    """感知哈希（dHash）：把图缩到 (size+1)×size 灰度图后比较相邻像素。

    用途：同一公司 logo / 页眉装饰图会在几百页里重复出现，
    仅靠二进制 sha1 去不掉（水印每页重新编码），感知哈希可以。
    """
    try:
        import numpy as np
        import pymupdf

        pix = pymupdf.Pixmap(image_bytes)
        if pix.n - pix.alpha >= 4:                      # CMYK → RGB
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        pix = pix.scale(size + 1, size)                 # 面积平均缩放
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        gray = arr[:, :, :3].mean(axis=2)
        bits = (gray[:, 1:] > gray[:, :-1]).flatten()
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return f"{value:0{size * size // 4}x}"
    except Exception:
        return ""


def hamming_hex(a: str, b: str) -> int:
    """两个十六进制感知哈希的汉明距离；任一为空返回 64（视为完全不同）。"""
    if not a or not b:
        return 64
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 64


def _slug(text: str) -> str:
    """文件名 → 短标签，用于 chunk_id。"""
    keep = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)
    return keep[:8] or "doc"


def split_text_blocks(text: str, *, size: int, overlap: int, min_size: int) -> list[str]:
    """把一页（或一段）文本切成块：段落 → 标点 → 硬切的降级策略。"""
    text = text.strip()
    if not text:
        return []
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
            continue
        buf = ""
        for sent in re.split(r"(?<=[。！？；])", para):
            if not sent:
                continue
            if len(buf) + len(sent) > size and buf:
                units.append(buf)
                buf = buf[-overlap:] if overlap > 0 else ""
            buf += sent
            while len(buf) > size:  # 单句超长：硬切
                units.append(buf[:size])
                buf = buf[size - overlap:] if overlap > 0 else buf[size:]
        if buf.strip():
            units.append(buf)

    out: list[str] = []
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if out and len(out[-1]) < size * 0.6 and len(out[-1]) + len(unit) <= size:
            out[-1] = (out[-1] + "\n" + unit).strip()
        else:
            out.append(unit)
    return [u for u in out if len(u) >= min_size]


def find_section(text: str) -> str:
    """粗提取章节标题（第X节 / 一、二、…），用于块元信息。"""
    m = re.search(r"第[一二三四五六七八九十百]+节\s*([^\n]{2,30})", text)
    if m:
        return m.group(0).strip()
    m = re.search(r"^\s*([一二三四五六七八九十]+、[^\n]{2,30})", text, re.M)
    return m.group(1).strip() if m else ""


def extract_keywords(text: str, *, top: int = 24) -> list[str]:
    """无 jieba 依赖的关键词兜底：数字 + 连续中文/英文词。"""
    tokens: list[str] = []
    tokens.extend(re.findall(r"\d[\d,\.]*%?", text))
    tokens.extend(re.findall(r"[A-Za-z]{2,}", text))
    for seg in re.findall(r"[\u4e00-\u9fff]{2,8}", text):
        tokens.append(seg)
    seen: list[str] = []
    for tok in tokens:
        if tok not in seen:
            seen.append(tok)
        if len(seen) >= top:
            break
    return seen


def _looks_like_table(lines: Sequence[str]) -> bool:
    """启发式判断：至少 3 行、且多数行含等距空白分隔的多个字段。"""
    if len(lines) < 3:
        return False
    hits = sum(1 for ln in lines if TABLE_ROW_HINT.search(ln))
    return hits >= max(3, int(len(lines) * 0.6))


def _page_content_blocks(page: Any, page_text: str, *, cfg: Cfg, log: Any) -> list[tuple[str, str]]:
    """返回 [(type, content)]：先用 pymupdf ``find_tables`` 抽真实表格，剩余文本再分块。

    LightRAG 的图谱质量强依赖表格里的「金额/比例/日期」三元组，
    所以表格必须结构化保留，不能当普通文本切碎。
    """
    c = cfg.chunk
    size = int(c.size)
    overlap = int(c.overlap)
    min_size = int(c.min_size)
    max_table = int(c.max_table_chars)

    results: list[tuple[str, str]] = []
    covered: list[Any] = []          # 已被表格覆盖的 bbox，用于扣减正文
    try:
        found = page.find_tables()
    except Exception as exc:
        log.warning("find_tables 失败：%s: %s", type(exc).__name__, exc)
        found = None

    for table in (getattr(found, "tables", None) or []):
        try:
            rows = table.extract()
        except Exception as exc:
            log.warning("表格抽取失败：%s: %s", type(exc).__name__, exc)
            continue
        lines: list[str] = []
        for row in rows:
            cells = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
            if cells:
                lines.append(" | ".join(cells))
        content = "\n".join(lines)
        if len(content) < min_size:
            continue
        covered.append(table.bbox)
        if len(content) <= max_table:
            results.append(("table", content))
        else:
            for i in range(0, len(content), max_table - overlap):
                piece = content[i:i + max_table]
                if len(piece) >= min_size:
                    results.append(("table", piece))

    # 扣掉表格区域后剩下的正文
    rest_text = page_text
    if covered:
        try:
            blocks = page.get_text("blocks")     # (x0,y0,x1,y1,text,block_no,block_type)
            keep = []
            for x0, y0, x1, y1, text, *_ in blocks:
                box = pymupdf.Rect(x0, y0, x1, y1)
                if any(box.intersects(pymupdf.Rect(cb)) for cb in covered):
                    continue
                keep.append(text)
            rest_text = "\n".join(keep)
        except Exception as exc:
            log.warning("按 bbox 扣减表格失败，回退整页文本：%s: %s", type(exc).__name__, exc)
            rest_text = page_text

    results.extend(("text", blk) for blk in split_text_blocks(rest_text, size=size, overlap=overlap, min_size=min_size))
    if not results:  # 极端情况兜底：整页当一段
        results.extend(("text", blk) for blk in split_text_blocks(page_text, size=size, overlap=overlap, min_size=min_size))
    return results


def _caption_near(page_text: str, bbox: Sequence[float], *, radius: int) -> str:
    """图注关联：在图 bbox 附近取「图/表/资料来源」开头的行，或最近一段正文。"""
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if not lines:
        return ""
    y0 = float(bbox[1]) if len(bbox) > 1 else 0.0
    page_h = max((float(bbox[3]) if len(bbox) > 3 else 800.0), 1.0)
    # 按行在页面的相对位置近似排序（pymupdf 文本无坐标，这里用行序近似）
    approx_line_idx = int(min(max(y0 / max(page_h, 1.0), 0.0), 1.0) * max(len(lines) - 1, 0))
    window = lines[max(0, approx_line_idx - 3): approx_line_idx + 8]
    captions = [ln for ln in window if CAPTION_PREFIX.match(ln)]
    if captions:
        return " ".join(captions)[:radius]
    return " ".join(window)[:radius]


def build_documents(raw_dir: Path, *, cfg: Cfg, files: Sequence[str] | None = None,
                    logger: Any = None) -> tuple[list[Chunk], dict[str, Any]]:
    """解析 raw_dir 下所有 PDF，返回 (chunks, stats)。单文件失败不影响其余。"""
    from .logging_util import get_logger
    log = logger or get_logger("corpus", cfg)
    ic = cfg.image
    chunks: list[Chunk] = []
    stats: dict[str, Any] = {"files": [], "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    names = files or sorted(f for f in os.listdir(raw_dir) if f.lower().endswith(".pdf"))
    seen_phash: list[str] = []          # 已入库的感知哈希（跨文件去重）
    for name in names:
        path = Path(raw_dir) / name
        doc_key = _slug(name)
        t0 = time.perf_counter()
        n_text = n_table = n_img = 0
        order = 0
        img_total = 0
        try:
            doc = pymupdf.open(path)
        except Exception as exc:  # 容错：损坏 PDF
            log.warning("PDF 打开失败：%s (%s: %s)", name, type(exc).__name__, exc)
            stats["files"].append({"file": name, "error": f"{type(exc).__name__}: {exc}"})
            continue

        with doc:
            page_count = doc.page_count
            for page_index in range(page_count):
                page_no = page_index + 1
                try:
                    page = doc[page_index]
                    page_text = page.get_text()
                except Exception as exc:  # 容错：单页失败
                    log.warning("第 %d 页解析失败（%s）：%s", page_no, name, exc)
                    continue

                section = find_section(page_text)
                for ctype, content in _page_content_blocks(page, page_text, cfg=cfg, log=log):
                    cid = f"{doc_key}-p{page_no}-{ctype}-{order}"
                    chunks.append(Chunk(
                        chunk_id=cid, file_name=name, doc_key=doc_key, page=page_no, page_end=page_no,
                        type=ctype, section=section, content=content,
                        keywords=extract_keywords(content), char_count=len(content), order=order,
                    ))
                    order += 1
                    if ctype == "table":
                        n_table += 1
                    else:
                        n_text += 1

                if bool(ic.enable) and img_total < int(ic.max_images):
                    img_total, added = _collect_page_images(
                        page, page_text, name=name, doc_key=doc_key, page_no=page_no,
                        order_start=order, cfg=cfg, chunks=chunks, log=log, seen_phash=seen_phash,
                    )
                    order += added
                    n_img += added

        el = time.perf_counter() - t0
        stats["files"].append({
            "file": name, "pages": page_count, "chunks_text": n_text,
            "chunks_table": n_table, "chunks_image": n_img, "seconds": round(el, 3),
        })
        log.info("解析完成 %s：文本块=%d 表格块=%d 图像块=%d 耗时=%.2fs", name, n_text, n_table, n_img, el)

    stats["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["chunk_total"] = len(chunks)
    stats["chunk_by_type"] = {
        t: sum(1 for c in chunks if c.type == t) for t in sorted({c.type for c in chunks})
    }
    stats["char_total"] = sum(c.char_count for c in chunks)
    return chunks, stats


def _collect_page_images(page: Any, page_text: str, *, name: str, doc_key: str, page_no: int,
                         order_start: int, cfg: Cfg, chunks: list[Chunk], log: Any,
                         seen_phash: list[str] | None = None) -> tuple[int, int]:
    """抽取本页有效图像块（水印/装饰/重复图全部过滤）。

    返回 (本页候选图像数, 本页实际新增块数)。
    本机无 OCR、Ollama 无视觉模型，因此图像块内容 = 图片元信息 + 图附近正文，
    绝不声称做过 OCR 或图像识别。
    """
    ic = cfg.image
    added = 0
    seen_phash = seen_phash if seen_phash is not None else []
    rows: list[dict[str, Any]] = []
    try:
        raw = page.get_images(full=True)
    except Exception as exc:
        log.warning("第 %d 页取图失败（%s）：%s", page_no, name, exc)
        return 0, 0

    candidates = 0
    for info in raw:
        xref = int(info[0])
        w, h = int(info[2]), int(info[3])
        if w < int(ic.min_width) or h < int(ic.min_height):
            continue  # 水印瓦片（143×127）与线条
        if w * h < int(ic.min_pixels):
            continue
        if max(w, h) / max(min(w, h), 1) > float(ic.max_aspect):
            continue  # 1×914 之类的分隔条
        candidates += 1
        try:
            blob = page.get_image_data(xref)[-1]
        except Exception:
            blob = b""
        phash = _dhash(blob) if blob else ""
        if any(hamming_hex(phash, prev) <= int(ic.phash_hamming) for prev in seen_phash):
            continue  # 重复图（水印/logo/同一张图重复出现）
        seen_phash.append(phash)
        rows.append({"xref": xref, "width": w, "height": h, "phash": phash, "blob": blob})
        if len(rows) >= int(ic.max_per_page):
            break

    for row in rows:
        xref = row["xref"]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        bbox = [float(v) for v in (rects[0] if rects else pymupdf.Rect(0, 0, row["width"], row["height"]))]
        caption = _caption_near(page_text, bbox, radius=int(ic.caption_radius_chars))
        desc = (
            f"{name} 第{page_no}页 插图（{row['width']}×{row['height']}px，文字描述口径，未做 OCR）\n"
            f"图页关联正文：{caption}"
        )
        chunks.append(Chunk(
            chunk_id=f"{doc_key}-p{page_no}-image-{order_start + added}",
            file_name=name, doc_key=doc_key, page=page_no, page_end=page_no, type="image",
            section=find_section(page_text), content=desc, keywords=extract_keywords(desc),
            char_count=len(desc), order=order_start + added,
            image_sha1=_sha1(row["blob"]) if row["blob"] else f"{doc_key}{page_no}{xref}",
            bbox=bbox, width=row["width"], height=row["height"], caption=caption,
        ))
        added += 1
    return candidates, added


def save_corpus(chunks: Sequence[Chunk], stats: dict[str, Any], *, cfg: Cfg) -> dict[str, str]:
    """落盘 chunks.jsonl / images.jsonl / corpus_stats.json。"""
    out_dir = cfg.path("processed_dir")
    chunks_path = Path(out_dir) / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    images_path = Path(out_dir) / "images.jsonl"
    with images_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            if c.type == "image":
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    stats_path = Path(out_dir) / "corpus_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"chunks": str(chunks_path), "images": str(images_path), "stats": str(stats_path)}


def load_chunks(path: Path | str) -> list[Chunk]:
    """读回 chunks.jsonl。"""
    out: list[Chunk] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Chunk(**json.loads(line)))
    return out


def iter_pages(pages_jsonl: Path | str) -> Iterable[dict[str, Any]]:
    """遍历 <doc>.pages.jsonl。"""
    with Path(pages_jsonl).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
