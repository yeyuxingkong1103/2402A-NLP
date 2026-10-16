# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill 并集成至智能体工作流工单

模块：document_quality_assessment.extractors
职责：统一的多格式文本抽取层，供五大功能共用。

支持：pdf(pymupdf) / docx(标准库 zipfile 自解析，本机无 python-docx) /
      md / txt / csv / json / jsonl / html。
约束：任何抽取失败都不许抛到调用方，统一返回带 error 的 ExtractResult。
"""
from __future__ import annotations

import html as html_mod
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import pymupdf  # PyMuPDF >= 1.24 的新包名
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # type: ignore

# ---------------------------------------------------------------------------
# 静态正则（与阈值无关的语法解析，不属于"判断阈值"，故不放进配置）
# ---------------------------------------------------------------------------
_DOCX_TEXT_RE = re.compile(r"<w:t[^>]*>(.*?)</w:t>", re.S)
_DOCX_PARA_RE = re.compile(r"</w:p>", re.S)
_XML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_CODE_FENCE_RE = re.compile(r"^```.*?^```", re.S | re.M)
_MD_INLINE_RE = re.compile(r"[`*_>#~\[\]\(\)!]|https?://\S+")
_WS_RE = re.compile(r"[ \t\u00a0\u3000]+")


@dataclass
class PageStat:
    """PDF 单页统计（功能 2 的输入）。"""

    index: int          # 0-based 页号
    char_count: int     # 抽到的可见字符数（去空白）
    image_area_ratio: float  # 图像覆盖面积 / 页面面积
    is_low_text: bool = False  # 字符数 < page_char_threshold（工单原文口径的"扫描页"）
    is_scan: bool = False      # 最终判定的扫描页（低文本 + 大图铺满，见配置 require_image_for_scan）
    has_text_layer: bool = False  # 是否抽到任何文本


@dataclass
class ExtractResult:
    """统一抽取结果。"""

    path: str
    suffix: str = ""
    size_bytes: int = 0
    text: str = ""
    char_count: int = 0
    page_count: int = 0
    pages: list[PageStat] = field(default_factory=list)
    is_pdf: bool = False
    encrypted: bool = False
    corrupted: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "suffix": self.suffix,
            "size_bytes": self.size_bytes,
            "char_count": self.char_count,
            "page_count": self.page_count,
            "is_pdf": self.is_pdf,
            "encrypted": self.encrypted,
            "corrupted": self.corrupted,
            "error": self.error,
        }


def normalize_text(text: str) -> str:
    """统一空白，保证"字符数"口径一致。"""
    if not text:
        return ""
    return _WS_RE.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def extract_pdf(path: Path, cfg: Any) -> ExtractResult:
    """抽取 PDF：逐页文本 + 逐页图像覆盖率。任何异常都转成 corrupted 标记。"""
    page_char_threshold = cfg.number("scan_detection.page_char_threshold")
    image_area_ratio_threshold = cfg.number("scan_detection.image_area_ratio_threshold")
    require_image_for_scan = cfg.boolean("scan_detection.require_image_for_scan", False)
    max_pages_sampled = cfg.integer("scan_detection.max_pages_sampled")
    try_empty_password = cfg.boolean("scan_detection.try_empty_password", True)
    max_chars_per_page = cfg.integer("scan_detection.max_chars_per_page", 200000)

    res = ExtractResult(path=str(path), suffix=".pdf", is_pdf=True)
    try:
        res.size_bytes = path.stat().st_size
    except OSError as exc:
        res.corrupted = True
        res.error = f"stat_failed: {exc}"
        return res

    if res.size_bytes == 0:
        res.corrupted = True
        res.error = "zero_byte_file"
        return res

    doc = None
    try:
        doc = pymupdf.open(str(path))
        if doc.needs_pass:
            res.encrypted = True
            opened = False
            if try_empty_password:
                try:
                    opened = bool(doc.authenticate(""))
                except Exception:  # noqa: BLE001 - 加密 PDF 的认证异常一律兜底
                    opened = False
            if not opened:
                res.corrupted = True
                res.error = "encrypted_pdf_cannot_open"
                return res
        res.page_count = int(doc.page_count)
        if res.page_count <= 0:
            res.corrupted = True
            res.error = "pdf_with_zero_pages"
            return res

        limit = res.page_count if max_pages_sampled <= 0 else min(res.page_count, max_pages_sampled)
        chunks: list[str] = []
        for i in range(limit):
            clean = ""
            ratio = 0.0
            try:
                page = doc.load_page(i)
                raw = page.get_text("text") or ""
                if len(raw) > max_chars_per_page:
                    raw = raw[:max_chars_per_page]
                clean = normalize_text(raw)
            except MemoryError:
                # 单页文本抽取内存不足：本页记为扫描页，整册继续（不放弃已抽到的页）
                res.error = (res.error + "; " if res.error else "") + f"page_{i}_memory_error"
            except Exception as exc:  # noqa: BLE001 - 单页坏页不影响整册
                res.error = (res.error + "; " if res.error else "") + f"page_{i}_text_failed: {exc}"
            try:
                page = doc.load_page(i)
                page_area = float(page.rect.width * page.rect.height) or 1.0
                img_area = 0.0
                for info in page.get_image_info():
                    bbox = info.get("bbox")
                    if not bbox:
                        continue
                    w = max(0.0, float(bbox[2]) - float(bbox[0]))
                    h = max(0.0, float(bbox[3]) - float(bbox[1]))
                    img_area += w * h
                ratio = min(1.0, img_area / page_area)
            except MemoryError:
                ratio = 0.0
                res.error = (res.error + "; " if res.error else "") + f"page_{i}_image_memory_error"
            except Exception:  # noqa: BLE001 - 图像信息缺失不影响主流程
                ratio = 0.0

            is_low_text = len(clean) < page_char_threshold
            if not require_image_for_scan:
                # 工单原文口径：字符数低于阈值即扫描页
                is_scan = is_low_text
            else:
                # 加严口径：低文本 + （抽不到任何文本 或 被大图铺满）
                is_scan = is_low_text and (len(clean) == 0 or ratio >= image_area_ratio_threshold)
            res.pages.append(
                PageStat(
                    index=i,
                    char_count=len(clean),
                    image_area_ratio=round(ratio, 4),
                    is_low_text=is_low_text,
                    is_scan=is_scan,
                    has_text_layer=len(clean) > 0,
                )
            )
            chunks.append(clean)
        res.text = "\n".join(chunks)
        res.char_count = sum(p.char_count for p in res.pages)
        return res
    except Exception as exc:  # noqa: BLE001 - 打开失败（损坏 / 非 PDF / 权限）
        res.corrupted = True
        res.error = f"open_failed: {type(exc).__name__}: {exc}"
        return res
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# DOCX（标准库自解析：本机无 python-docx）
# ---------------------------------------------------------------------------
def extract_docx(path: Path) -> ExtractResult:
    res = ExtractResult(path=str(path), suffix=path.suffix.lower())
    try:
        res.size_bytes = path.stat().st_size
    except OSError as exc:
        res.corrupted = True
        res.error = f"stat_failed: {exc}"
        return res
    if res.size_bytes == 0:
        res.corrupted = True
        res.error = "zero_byte_file"
        return res
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            parts = [n for n in ("word/document.xml", "word/header1.xml", "word/footer1.xml") if n in names]
            if not parts:
                res.corrupted = True
                res.error = "docx_missing_word_document_xml"
                return res
            buf: list[str] = []
            for name in parts:
                xml = zf.read(name).decode("utf-8", errors="ignore")
                for para in _DOCX_PARA_RE.split(xml):
                    text = "".join(_DOCX_TEXT_RE.findall(para))
                    text = html_mod.unescape(_XML_TAG_RE.sub("", text))
                    if text.strip():
                        buf.append(text)
        res.text = normalize_text("\n".join(buf))
        res.char_count = len(res.text)
        res.page_count = 0
        return res
    except (zipfile.BadZipFile, KeyError, OSError, RuntimeError) as exc:
        res.corrupted = True
        res.error = f"docx_parse_failed: {type(exc).__name__}: {exc}"
        return res


# ---------------------------------------------------------------------------
# 纯文本类
# ---------------------------------------------------------------------------
def extract_plain(path: Path, *, strip_markup: str = "none") -> ExtractResult:
    """读取文本文件。``strip_markup``: none / markdown / html。"""
    res = ExtractResult(path=str(path), suffix=path.suffix.lower())
    try:
        res.size_bytes = path.stat().st_size
    except OSError as exc:
        res.corrupted = True
        res.error = f"stat_failed: {exc}"
        return res
    if res.size_bytes == 0:
        res.corrupted = True
        res.error = "zero_byte_file"
        return res
    raw = ""
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            raw = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            res.corrupted = True
            res.error = f"read_failed: {exc}"
            return res
    if strip_markup == "markdown":
        raw = _MD_CODE_FENCE_RE.sub(" ", raw)
        raw = _MD_INLINE_RE.sub(" ", raw)
    elif strip_markup == "html":
        raw = _HTML_SCRIPT_RE.sub(" ", raw)
        raw = html_mod.unescape(_HTML_TAG_RE.sub(" ", raw))
    res.text = normalize_text(raw)
    res.char_count = len(res.text)
    return res


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def extract(path: str | Path, cfg: Any) -> ExtractResult:
    """按扩展名分派抽取器。``cfg`` 为 Config 实例（阈值来源）。"""
    p = Path(path)
    suffix = p.suffix.lower()
    group_map = cfg.get("format_stats.extension_groups", {}) or {}
    group = str(group_map.get(suffix, cfg.get("format_stats.unknown_group", "Other")))
    try:
        if suffix == ".pdf":
            return extract_pdf(p, cfg)
        if suffix in (".docx", ".doc", ".pptx", ".xlsx"):
            return extract_docx(p)
        if group == "Markdown" or suffix in (".md", ".markdown"):
            return extract_plain(p, strip_markup="markdown")
        if group == "Web" or suffix in (".html", ".htm"):
            return extract_plain(p, strip_markup="html")
        return extract_plain(p, strip_markup="none")
    except Exception as exc:  # noqa: BLE001 - 最后一道兜底，Skill 绝不允许裸崩
        res = ExtractResult(path=str(p), suffix=suffix)
        res.corrupted = True
        res.error = f"extractor_unhandled: {type(exc).__name__}: {exc}"
        return res


def is_parseable_suffix(suffix: str, cfg: Any) -> bool:
    """该扩展名是否属于本 Skill 支持解析的格式。"""
    suffix = suffix.lower()
    group_map = cfg.get("format_stats.extension_groups", {}) or {}
    return suffix in group_map


def quick_json_loads(text: str) -> Any:  # 供测试/脚本复用小工具
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
