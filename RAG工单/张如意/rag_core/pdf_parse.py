# -*- coding: utf-8 -*-
"""
PDF 解析模块（文字 + 表格 + 图像）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
          人工智能NLP-RAG-PDF文档的表格解析及检索优化
          人工智能NLP-RAG-图像内容解析及检索优化

三层解析能力：
  L1 文字层：PyMuPDF 提取正文，保留页码，用于向量检索
  L2 表格层：pdfplumber 提取表格 -> Markdown，解决「数字答不准」问题
  L3 图像层：PyMuPDF 抽取内嵌图像 -> 落盘，供多模态模型生成语义描述

解析结果统一为 PageBlock 列表并缓存为 JSON，避免重复解析。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import fitz  # PyMuPDF

from . import config

# 解析器版本号：清洗/抽取逻辑有变更时递增，使旧缓存自动失效
PARSER_VERSION = "2"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class PageBlock:
    """PDF 中一个可检索的基本单元。"""
    doc: str                 # 文档名
    page: int                # 页码（1 起）
    type: str                # text | table | image
    content: str             # 文本内容（表格为 Markdown，图像为语义描述）
    extra: dict = field(default_factory=dict)   # 表格行列数、图片路径等

    @property
    def uid(self) -> str:
        h = hashlib.md5(
            f"{self.doc}|{self.page}|{self.type}|{self.content[:64]}".encode("utf-8")
        ).hexdigest()[:12]
        return f"{self.doc}#p{self.page}#{self.type}#{h}"


@dataclass
class ParsedDoc:
    name: str
    path: str
    n_pages: int
    blocks: list[PageBlock] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "n_pages": self.n_pages,
            "blocks": [asdict(b) for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ParsedDoc":
        return cls(
            name=d["name"], path=d["path"], n_pages=d["n_pages"],
            blocks=[PageBlock(**b) for b in d["blocks"]],
        )


# ---------------------------------------------------------------------------
# 文本清洗
# ---------------------------------------------------------------------------
_HEADER_FOOTER_PAT = re.compile(
    r"^\s*(北京八维信息集团|武汉兴图新科电子股份有限公司|武汉力源信息技术股份有限公司)?\s*\d*\s*$"
)
_PAGENUM_PAT = re.compile(r"^\s*[-—–]?\s*\d{1,3}\s*[-—–]?\s*$")


# 句子结束标点：上一行以此结尾说明语义已完整，换行应保留
_SENT_END_PAT = re.compile(r"[。！？；：!?;:]$")
# 行首特征：编号条目 / 章节标题 / 项目符号，遇到这些必须保留换行
# 注意 `\d{1,2}[、.．](?!\d)` 的负向断言：避免把 "1.00 元" 这类小数误判成编号条目
_LINE_START_PAT = re.compile(
    r"^(第[一二三四五六七八九十百]+[节章]|[一二三四五六七八九十]{1,3}[、.．]"
    r"|（[一二三四五六七八九十]{1,3}）|\([一二三四五六七八九十]{1,3}\)"
    r"|\d{1,2}[、.．](?!\d)|[（(]\d{1,2}[)）]"
    r"|[·•▪◆▲●○]\s|附表|附件|资料来源|注[:：])"
)


def clean_text(t: str, keep_line_breaks: bool = True) -> str:
    """
    清洗 PDF 提取出的正文：去页眉页脚、拼接被硬换行切断的句子。

    Args:
        keep_line_breaks: 是否保留「语义完整的行」之间的换行。
            **必须为 True 才能让 chunk_structure 的章节标题识别生效**——
            章节标题（如「第五节 业务与技术」「（一）主要产品」）依赖行首特征，
            若把所有行强行合并成一行，结构感知分块就退化成固定窗口分块了。
    """
    lines = []
    for ln in t.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if _PAGENUM_PAT.match(s):
            continue
        if _HEADER_FOOTER_PAT.match(s):
            continue
        lines.append(s)

    parts: list[str] = []
    for ln in lines:
        if not parts:
            parts.append(ln)
            continue

        prev = parts[-1]
        # 判定是否该在此处断行：
        #   1) 上一行是完整句子（以句末标点结尾）
        #   2) 当前行是编号条目 / 章节标题 / 项目符号
        #   3) 上一行本身就是标题特征行
        break_here = (
            _SENT_END_PAT.search(prev)
            or _LINE_START_PAT.match(ln)
            or _LINE_START_PAT.match(prev)
            or len(prev) < 20
        ) if keep_line_breaks else False

        if break_here:
            parts.append(ln)
        else:
            # 中文跨行直接拼接；英文单词间补空格
            if re.search(r"[A-Za-z0-9]$", prev) and re.search(r"^[A-Za-z0-9]", ln):
                parts[-1] = prev + " " + ln
            else:
                parts[-1] = prev + ln

    return re.sub(r"[ \t]+", " ", "\n".join(parts)).strip()


# ---------------------------------------------------------------------------
# L1 文字层
# ---------------------------------------------------------------------------
def extract_text_blocks(pdf_path: Path, doc_name: str) -> list[PageBlock]:
    doc = fitz.open(pdf_path)
    blocks: list[PageBlock] = []
    for i, page in enumerate(doc):
        raw = page.get_text("text")
        txt = clean_text(raw)
        if not txt:
            continue
        blocks.append(
            PageBlock(doc=doc_name, page=i + 1, type="text", content=txt,
                      extra={"char_len": len(txt)})
        )
    doc.close()
    return blocks


# ---------------------------------------------------------------------------
# L2 表格层
# ---------------------------------------------------------------------------
def _table_to_markdown(rows: list[list[str | None]]) -> str:
    """把 pdfplumber 提取的二维表转成 Markdown，便于 LLM 读懂行列关系。"""
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows if r]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    head, body = rows[0], rows[1:]
    md = ["| " + " | ".join(head) + " |",
          "| " + " | ".join(["---"] * width) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(md)


def extract_table_blocks(pdf_path: Path, doc_name: str,
                         max_pages: int | None = None) -> list[PageBlock]:
    """
    用 pdfplumber 抽取表格并转 Markdown。
    注意：pdfplumber 逐页解析较慢，招股书首页/目录页可跳过。
    """
    import pdfplumber

    blocks: list[PageBlock] = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        for i, page in enumerate(pages):
            try:
                tables = page.extract_tables()
            except Exception as e:                      # 容错：单页失败不影响整体
                print(f"  [warn] 第{i+1}页表格解析失败: {e}")
                continue
            for ti, tb in enumerate(tables):
                md = _table_to_markdown(tb)
                # 过滤噪声：少于 2 行或纯空表
                if md.count("\n") < 1 or len(md) < 20:
                    continue
                blocks.append(
                    PageBlock(
                        doc=doc_name, page=i + 1, type="table", content=md,
                        extra={"rows": len(tb), "cols": len(tb[0]) if tb else 0,
                               "table_index": ti},
                    )
                )
    return blocks


# ---------------------------------------------------------------------------
# L3 图像层
# ---------------------------------------------------------------------------
def extract_images(pdf_path: Path, doc_name: str,
                   out_dir: Path | None = None,
                   min_size: int = 120) -> list[PageBlock]:
    """
    抽取 PDF 内嵌位图并落盘。
    向量图（组织结构图、柱状图这类）通常是矢量绘制，
    PyMuPDF 的 get_images() 取不到，此时回退为「整页渲染成图」。
    """
    out_dir = out_dir or (config.IMAGE_DIR / doc_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    blocks: list[PageBlock] = []
    for i, page in enumerate(doc):
        page_no = i + 1
        imgs = page.get_images(full=True)
        got = False
        for j, img in enumerate(imgs):
            try:
                pix = fitz.Pixmap(doc, img[0])
                if pix.width < min_size or pix.height < min_size:
                    continue
                if pix.n - pix.alpha >= 4:              # CMYK -> RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                fp = out_dir / f"p{page_no}_img{j}.png"
                pix.save(fp)
                blocks.append(
                    PageBlock(doc=doc_name, page=page_no, type="image",
                              content="",   # 语义描述由 image_parse 模块回填
                              extra={"image_path": str(fp), "source": "embedded",
                                     "w": pix.width, "h": pix.height})
                )
                got = True
            except Exception as e:
                print(f"  [warn] 第{page_no}页图像{j}抽取失败: {e}")

        # 矢量图回退：整页渲染（招股书的图表多为矢量绘制）
        if not got and _page_looks_like_figure(page):
            fp = out_dir / f"p{page_no}_fullpage.png"
            page.get_pixmap(dpi=150).save(fp)
            blocks.append(
                PageBlock(doc=doc_name, page=page_no, type="image", content="",
                          extra={"image_path": str(fp), "source": "page_render",
                                 "w": 0, "h": 0})
            )
    doc.close()
    return blocks


def _page_looks_like_figure(page) -> bool:
    """
    判断页面是否以图为主（用于矢量图回退策略）。
    启发式：正文文字很少 且 存在绘图对象。
    """
    txt = page.get_text("text").strip()
    has_drawings = len(page.get_drawings()) > 20
    return len(txt) < 200 and has_drawings


# ---------------------------------------------------------------------------
# 统一入口 + 缓存
# ---------------------------------------------------------------------------
def parse_pdf(
    pdf_path: Path,
    doc_name: str | None = None,
    with_tables: bool = True,
    with_images: bool = False,
    use_cache: bool = True,
) -> ParsedDoc:
    """
    解析 PDF 为结构化 PageBlock 列表（带磁盘缓存）。

    Args:
        with_tables: 是否做表格解析（较慢，工单03 起开启）
        with_images: 是否抽取图像（工单04 起开启）
    """
    pdf_path = Path(pdf_path)
    doc_name = doc_name or pdf_path.stem

    # 缓存键包含 PARSER_VERSION：解析/清洗逻辑变更后自动失效旧缓存
    cache_key = hashlib.md5(
        f"{pdf_path}|{pdf_path.stat().st_mtime}|{with_tables}|{with_images}"
        f"|{PARSER_VERSION}".encode("utf-8")
    ).hexdigest()[:16]
    cache_file = config.CACHE_DIR / f"parsed_{doc_name}_{cache_key}.json"

    if use_cache and cache_file.exists():
        return ParsedDoc.from_dict(json.loads(cache_file.read_text(encoding="utf-8")))

    n_pages = fitz.open(pdf_path).page_count
    blocks = extract_text_blocks(pdf_path, doc_name)
    if with_tables:
        blocks += extract_table_blocks(pdf_path, doc_name)
    if with_images:
        blocks += extract_images(pdf_path, doc_name)
    blocks.sort(key=lambda b: (b.page, {"text": 0, "table": 1, "image": 2}[b.type]))

    parsed = ParsedDoc(name=doc_name, path=str(pdf_path),
                       n_pages=n_pages, blocks=blocks)

    cache_file.write_text(
        json.dumps(parsed.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return parsed
