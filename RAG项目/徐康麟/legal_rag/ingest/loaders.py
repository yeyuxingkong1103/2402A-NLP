# -*- coding: utf-8 -*-
"""文档解析。

格式 → 解析器的对应关系放在**注册表** ``_PARSERS`` 里（下表的 ``register_parser``）：
新增格式只要 ``register_parser(".docx", load_docx)``，**不需要改** :func:`load_document`。
（此前文件头声称有 ``_PARSERS`` 可注册，但代码里其实是一条 ``if/elif`` 链 ——
注释与实现不一致，本轮把它兑现。）

PDF 走两级：

1. **文字层**：``pypdf`` 抽正文；
2. **表格**：``pdfplumber``（可选依赖）用**表格线**识别表格并转成 Markdown，
   **整表放在一个段落里**（行间只用单个换行、不留空行）⇒ 默认的段落/法条分块策略
   会把整张表留在**同一个块**里，不会出现"表头在一片、数据在另一片"。

拿不到文字、或者没有 ``pdfplumber`` 时**都不许静默**：
前者由调用方（上传链路）按"抽出 0 块"告警，后者在 ``Document.parse`` 里
写明 ``tables_engine="unavailable"`` 并打 WARNING。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..utils import md5_of_text

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text"}
JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | JSON_SUFFIXES | PDF_SUFFIXES

#: **格式 → 解析器**的注册表（后缀一律小写、带点）。``load_document`` 只查这张表。
_PARSERS: dict[str, Callable[[Path], str]] = {}


def register_parser(suffix: str, loader: Callable[[Path], str]) -> None:
    """注册（或覆盖）某个后缀的解析器。``suffix`` 形如 ``".docx"``。"""
    key = str(suffix).strip().lower()
    if not key.startswith("."):
        raise ValueError(f"后缀要以点开头：{suffix!r}")
    _PARSERS[key] = loader


def registered_suffixes() -> tuple[str, ...]:
    """已注册的后缀（排序后返回，便于错误信息与文档保持一致）。"""
    return tuple(sorted(_PARSERS))

#: ``doc_id`` 落进 Milvus 的 varchar(256)，而且上限**按 UTF-8 字节算**
#: （2026-09-17 实测：300 个汉字的文件名主干 = 900 字节 → 整篇拒收）。
#: 这里留 56 字节余量：截断到 ≤200 字节 + 8 位内容哈希（合计 ≤209 字节）。
MAX_DOC_ID_BYTES = 200


def safe_doc_id(stem: str) -> str:
    """把文件名主干压到 ``doc_id`` 字节上限内：超长时按字符边界截断 + 8 位哈希后缀。"""
    stem = (stem or "").strip()
    if len(stem.encode("utf-8")) <= MAX_DOC_ID_BYTES:
        return stem
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8]
    head = stem.encode("utf-8")[: MAX_DOC_ID_BYTES - 9].decode("utf-8", errors="ignore")
    return f"{head}_{digest}"

_TEXT_KEYS = ("text", "content", "body", "passage", "article")


@dataclass
class Document:
    doc_id: str
    source: str
    text: str
    suffix: str = ""
    created_at: float = field(default_factory=time.time)
    md5: str = ""
    #: 解析报告（**给用户/运维看的**）：用了哪个引擎、抽到几张表、字符数多少。
    #: 解析降级（例如没装 pdfplumber）必须在这里留下痕迹，不能只写日志。
    parse: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.md5:
            self.md5 = md5_of_text(self.text)


def _flatten_json(obj) -> str:
    """从 JSON 结构里尽力抽取正文。"""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return "\n\n".join(filter(None, (_flatten_json(x) for x in obj)))
    if isinstance(obj, dict):
        for key in _TEXT_KEYS:
            if isinstance(obj.get(key), str):
                return obj[key]
        # 法律 FAQ 语料常见 input/output 结构
        if isinstance(obj.get("input"), str) or isinstance(obj.get("output"), str):
            question = obj.get("input") or obj.get("instruction") or ""
            answer = obj.get("output") or obj.get("response") or ""
            return f"{question}\n{answer}".strip()
        return "\n".join(
            filter(None, (_flatten_json(v) for v in obj.values()))
        )
    return str(obj)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        parts = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                parts.append(_flatten_json(json.loads(line)))
        return "\n\n".join(parts)
    return _flatten_json(json.loads(raw))


#: 「疑似扫描件」判定阈值：文字层**平均每页**字符数低于它就尝试 OCR。
#: 为什么按"每页"而不是总量：一份 30 页的扫描件里偶尔混着几个页眉字，
#: 按总量会误判成"有文字层"。
OCR_MIN_CHARS_PER_PAGE = 20

#: OCR 引擎优先级：paddleocr（中文更强，但依赖重）> rapidocr-onnxruntime（轻量 CPU）。
OCR_ENGINE_PRIORITY = ("paddleocr", "rapidocr")

#: 引擎实例缓存（加载 ONNX/paddle 模型要 1~2 秒，逐文件重建会拖垮批量入库）
_OCR_ENGINE_CACHE: dict[str, object] = {}


def _pypdf_text(path: Path) -> str:
    """文字层抽取（``pypdf``）。单页失败不影响整篇，但会记 debug。"""
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - 单页解析失败不影响整体
            logger.debug("pypdf 单页抽取失败（%s）：%s: %s", path.name, type(exc).__name__, exc)
            pages.append("")
    return "\n\n".join(pages)


def _table_to_markdown(rows: list[list[str]]) -> str:
    """把 ``pdfplumber`` 的表格转成 Markdown。

    **行间只用单个换行、不留空行**：这样在默认的段落/法条分块策略下，
    整张表会落在**同一个块**里（`ingest/chunker.py` 的段落切分以空行为界）。
    """
    cleaned = [[str(cell or "").replace("\n", " ").replace("|", "/").strip()
                for cell in row] for row in (rows or [])]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    header, body = cleaned[0], cleaned[1:]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in range(width)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _row_keys(tables: list[list[list[str]]]) -> set[str]:
    """把每张表的每一行压成"去空白后的整行文本"，作为**精确匹配**用的键。"""
    keys: set[str] = set()
    for table in tables or []:
        for row in table or []:
            cells = [str(cell or "").strip() for cell in (row or [])]
            cells = [cell for cell in cells if cell]
            if len(cells) >= 2:               # 单格行太容易与正文撞车，不参与匹配
                keys.add(re.sub(r"\s+", "", "".join(cells)))
    return keys


def _load_ocr_engine() -> tuple[str, object | None]:
    """惰性加载并**缓存** OCR 引擎；返回 ``(引擎名, 实例)``，都没有则 ``("", None)``。

    两个后端（都是"可选依赖"，没装就明确降级，**绝不静默**）：

    * ``paddleocr`` —— 中文效果更好，但依赖重（paddlepaddle + 模型，几百 MB~1 GB）；
    * ``rapidocr``（包名 ``rapidocr-onnxruntime``）—— 轻量 CPU ONNX 版 PP-OCR，模型随包带。

    ⚠️ 诚实边界：本机（Python 3.14 / Windows）**装不了 paddlepaddle**（无可用发行版），
    所以 ``paddleocr`` 分支**未在本机实测**（只按它的 API 写）；``rapidocr`` 分支实测可用。
    """
    if "engine" in _OCR_ENGINE_CACHE:
        name = str(_OCR_ENGINE_CACHE.get("name") or "")
        return name, _OCR_ENGINE_CACHE.get("engine")
    name, engine = "", None
    for candidate in OCR_ENGINE_PRIORITY:
        try:
            if candidate == "paddleocr":
                from paddleocr import PaddleOCR  # type: ignore
                engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            else:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore
                engine = RapidOCR()
        except Exception as exc:  # noqa: BLE001 - 没装/加载失败都算"不可用"，但要看得见
            logger.info("OCR 后端 %s 不可用（%s: %s）", candidate, type(exc).__name__, exc)
            continue
        name = candidate
        logger.info("OCR 后端已就绪：%s", name)
        break
    if engine is None:
        logger.warning("没有可用的 OCR 后端（paddleocr / rapidocr-onnxruntime 都没装）："
                       "扫描件 PDF 将**抽不出文字**。需要时：pip install rapidocr-onnxruntime")
    _OCR_ENGINE_CACHE["name"] = name
    _OCR_ENGINE_CACHE["engine"] = engine
    return name, engine


def _ocr_lines_from_image(name: str, engine: object, array) -> list[str]:
    """把一个页面位图交给某个后端识别，返回文本行（按后端归一化输出）。

    * ``rapidocr``：``engine(array) -> ([[box, text, score], ...], elapse)``
      —— 注意 **score 是字符串**，别按 float 格式化（本机实测踩过）；
    * ``paddleocr``：``engine.ocr(array) -> [[[box, (text, score)], ...]]``（未本机实测）。
    """
    if name == "rapidocr":
        result, _elapse = engine(array)          # type: ignore[operator]
        lines: list[str] = []
        for item in result or []:
            if len(item) >= 2 and str(item[1]).strip():
                lines.append(str(item[1]).strip())
        return lines
    if name == "paddleocr":
        result = engine.ocr(array, cls=True)     # type: ignore[attr-defined]
        lines = []
        for page in result or []:
            for entry in page or []:
                if len(entry) >= 2 and isinstance(entry[1], (list, tuple)) and entry[1]:
                    text = str(entry[1][0]).strip()
                    if text:
                        lines.append(text)
        return lines
    return []


def _ocr_pdf(path: Path, *, dpi: int = 150, max_pages: int = 30) -> tuple[str, dict]:
    """对 PDF **逐页渲染后 OCR**，返回 ``(文本, 报告)``。

    渲染用 ``pdfplumber`` 的 ``page.to_image()``（它底下的 pypdfium2 已在依赖里）。
    报告里的 ``ocr_status`` 取值：

    * ``ok`` —— 至少识别出一些文字；
    * ``empty`` —— 引擎可用，但一页也没识别出文字（图片太糊/确实没字）；
    * ``unavailable`` —— 没有可用引擎（没装），或缺少渲染依赖；
    * ``failed`` —— 渲染/识别过程抛异常（**不吞**：WARNING + 状态带回）。
    """
    name, engine = _load_ocr_engine()
    if engine is None:
        # 在这里也留一条 WARNING：**"这份文件被判为扫描件、但没有 OCR 可用"** 与
        # "引擎不可用"是两件事，排障时想知道的是前者（哪份文件因此没进库）。
        logger.warning("判定为疑似扫描件，但没有可用的 OCR 后端，该 PDF 将抽不出文字：%s"
                       "（要 OCR 请 pip install rapidocr-onnxruntime）", path.name)
        return "", {"ocr_status": "unavailable", "ocr_engine": "", "ocr_chars": 0}
    try:
        import numpy as np  # type: ignore
        import pdfplumber  # type: ignore

        page_texts: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages[:max_pages]:
                image = page.to_image(resolution=dpi).original
                lines = _ocr_lines_from_image(name, engine, np.asarray(image))
                if lines:
                    page_texts.append("\n".join(lines))
        text = "\n\n".join(page_texts)
        status = "ok" if text.strip() else "empty"
        if status == "empty":
            logger.warning("OCR 跑完了但没识别出文字：%s（可能图片太糊/不是文字）", path.name)
        return text, {"ocr_status": status, "ocr_engine": name, "ocr_chars": len(text.strip())}
    except Exception as exc:  # noqa: BLE001 - OCR 是增强项，失败不能把整篇 PDF 搞挂
        logger.warning("OCR 失败（%s，引擎=%s）：%s: %s —— 该 PDF 只按文字层入库",
                       path.name, name, type(exc).__name__, exc)
        return "", {"ocr_status": "failed", "ocr_engine": name, "ocr_chars": 0}


def _pdf_page_count(path: Path) -> int:
    """PDF 页数（读不到就返回 0，调用方据此跳过"每页字数"判定）。"""
    try:
        from pypdf import PdfReader  # type: ignore
        return len(PdfReader(str(path)).pages)
    except Exception as exc:  # noqa: BLE001
        logger.debug("读 PDF 页数失败（%s）：%s: %s", path.name, type(exc).__name__, exc)
        return 0


def drop_table_rows(text: str, tables: list[list[list[str]]]) -> tuple[str, int]:
    """从**扁平文本**里删掉"已经被表格还原覆盖"的行，返回 ``(新文本, 删了几行)``。

    为什么：`pypdf` 抽出来的扁平文本里也有表格里的字（只是没有结构），
    而 Markdown 表格又写了一遍 ⇒ **同一份数据被索引两次**（检索可能重复命中）。

    安全性（都写成了测试）：

    * **只做精确整行匹配**：把一行去掉所有空白后，正好等于某张表某一行的
      "所有单元格拼接"才删 ⇒ pypdf 把宽行折成两行时**匹配不上**，
      最坏情况只是"没去重"，**绝不会删掉正文**；
    * 单格行不参与匹配（正文里"第一条"这种行太常见）；
    * 没有表格的文件 `keys` 为空 ⇒ **正文逐字不变**。
    """
    keys = _row_keys(tables)
    if not keys:
        return text, 0
    kept: list[str] = []
    dropped = 0
    for line in (text or "").splitlines():
        compact = re.sub(r"\s+", "", line)
        if compact and compact in keys:
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept) if dropped else text, dropped


def _pdfplumber_tables(path: Path) -> tuple[list[list[list[str]]], list[str], str]:
    """用 ``pdfplumber`` 抽表格；返回 ``(原始表格, Markdown 块, 状态)``。

    状态取值（**调用方据此决定要不要告警**）：

    * ``"ok"`` —— 真的抽到了表；
    * ``"no_tables"`` —— 装了 pdfplumber，但这份 PDF 里没有识别到表格；
    * ``"unavailable"`` —— **没装 pdfplumber**（可选依赖）⇒ 必须 WARNING，不许静默降级；
    * ``"failed"`` —— 抽表过程中抛异常 ⇒ 同上，WARNING + 状态带回。
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        logger.warning("未安装 pdfplumber，PDF 表格不会还原（只抽文字层）：%s —— "
                       "需要表格还原时请 pip install pdfplumber", path.name)
        return [], [], "unavailable"
    tables: list[list[list[str]]] = []
    blocks: list[str] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables() or []:
                    markdown = _table_to_markdown(table)
                    if markdown:
                        tables.append(table)
                        blocks.append(markdown)
    except Exception as exc:  # noqa: BLE001 - 表格是增强项，失败不能把整篇 PDF 搞挂
        logger.warning("pdfplumber 抽表失败（%s）：%s: %s —— 该 PDF 只按文字层入库",
                       path.name, type(exc).__name__, exc)
        return [], [], "failed"
    return tables, blocks, ("ok" if blocks else "no_tables")


def load_pdf_with_report(path: Path) -> tuple[str, dict]:
    """PDF 解析：文字层 + 表格 +（疑似扫描件时）OCR，返回 ``(正文, 解析报告)``。

    三级递进，每级都**可见**：

    1. ``pypdf`` 抽文字层；
    2. ``pdfplumber`` 抽表格 → Markdown，并把扁平文本里"已被表格覆盖"的行去掉；
    3. **疑似扫描件**（文字层平均每页 < :data:`OCR_MIN_CHARS_PER_PAGE`）时走 OCR；
       引擎没装 ⇒ ``ocr_status="unavailable"``（WARNING），引擎报错 ⇒ ``"failed"``（WARNING），
       **都不影响文字层那一份照常入库**。
    """
    text = _pypdf_text(path)
    pages = _pdf_page_count(path)
    tables, blocks, tables_status = _pdfplumber_tables(path)
    dropped = 0
    if blocks:
        text, dropped = drop_table_rows(text, tables)

    ocr_status, ocr_engine, ocr_chars = "not_needed", "", 0
    per_page = (len(text.strip()) / pages) if pages else None
    if per_page is not None and per_page < OCR_MIN_CHARS_PER_PAGE:
        logger.info("文字层偏薄（%.1f 字/页 < %d），判定为**疑似扫描件**，尝试 OCR：%s",
                    per_page, OCR_MIN_CHARS_PER_PAGE, path.name)
        ocr_text, ocr_report = _ocr_pdf(path)
        ocr_status = str(ocr_report.get("ocr_status") or "")
        ocr_engine = str(ocr_report.get("ocr_engine") or "")
        ocr_chars = int(ocr_report.get("ocr_chars") or 0)
        if ocr_text.strip():
            # 文字层若真有一点残留，保留 + 追加 OCR（都是这份文件的内容，不丢）
            text = ocr_text.strip() if not text.strip() else f"{text.strip()}\n\n{ocr_text.strip()}"

    if blocks:
        # 表格与正文之间**空行**分段；表格内部不留空行（保证整表同块）
        text = "\n\n".join([part for part in (text, *blocks) if part.strip()])

    engines = ["pypdf"]
    if blocks:
        engines.append("pdfplumber")
    if ocr_status == "ok":
        engines.append(f"ocr:{ocr_engine}" if ocr_engine else "ocr")
    report = {
        "engine": "+".join(engines),
        "pages": pages,
        "tables": len(blocks),
        "tables_engine": tables_status,
        "dropped_flat_rows": dropped,
        "ocr_used": ocr_status == "ok",
        "ocr_status": ocr_status,
        "ocr_engine": ocr_engine,
        "ocr_chars": ocr_chars,
        "chars": len(text),
    }
    return text, report


def load_pdf(path: Path) -> str:
    """只取正文（**向后兼容**：老调用方/测试直接用这个）。"""
    text, _report = load_pdf_with_report(path)
    return text


def load_document(path) -> Document:
    """把一个文件解析成 Document（**查注册表**，不再是一条 if/elif 链）。"""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    suffix = path.suffix.lower()
    loader = _PARSERS.get(suffix)
    if loader is None:
        raise ValueError(
            f"暂不支持的格式: {suffix}（已注册 {' '.join(registered_suffixes())}）")

    report: dict = {"engine": suffix.lstrip(".") or "unknown", "tables": 0,
                    "tables_engine": "n/a"}
    if suffix in PDF_SUFFIXES:
        text, report = load_pdf_with_report(path)
    else:
        text = loader(path)
        report["chars"] = len(text)

    return Document(
        doc_id=safe_doc_id(path.stem),
        source=path.name,
        text=text,
        suffix=suffix,
        parse=report,
    )


def iter_document_paths(targets) -> list[Path]:
    """把文件/目录混合的目标展开成待入库文件列表。"""
    found: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                    found.append(child)
        elif path.is_file():
            found.append(path)
    # 去重且保持稳定顺序
    seen: set[str] = set()
    unique: list[Path] = []
    for path in found:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


# --------------------------------------------------------------------------- 注册
#: 内置解析器：文本类 / JSON 类走各自函数；PDF 走 `load_pdf`（内部再分文字层与表格）。
for _suffix in sorted(TEXT_SUFFIXES):
    register_parser(_suffix, load_text)
for _suffix in sorted(JSON_SUFFIXES):
    register_parser(_suffix, load_json)
for _suffix in sorted(PDF_SUFFIXES):
    register_parser(_suffix, load_pdf)
del _suffix
