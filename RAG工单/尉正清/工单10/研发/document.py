# 工单编号：人工智能NLP-RAG-金融问答系统部署
"""文档处理：PDF 解析（文字 + 表格）+ 文本分块"""
import re
from collections import Counter

from pathlib import Path

import pdfplumber
import pymupdf

from config import CHUNK_SIZE, CHUNK_OVERLAP

# 纯页码样式的行，如 "1-1-62"、"12"、"– 8 –"
_COMPANY = re.compile(r"[一-龥]{2,12}(?:股份)?有限公司")
_PAGE_NO = re.compile(r"^[\s\-–—_0-9]{1,12}$")


# ---------------- PDF 解析 ----------------
def _norm(text):
    """去掉所有空白。比对页眉时必须归一化：pymupdf 与 pdfplumber 抽同一行时，
    词之间的空格数不一样（前者保留版面空白，后者压成单空格），精确匹配会漏。"""
    return re.sub(r"\s+", "", text)


def _repeated_lines(pages, min_ratio=0.3, min_pages=5):
    """统计出现在 30% 以上页面里的行 —— 这些就是页眉页脚。"""
    counter = Counter()
    for page in pages:
        for line in {_norm(ln) for ln in page["text"].splitlines()}:
            if line:
                counter[line] += 1
    threshold = max(min_pages, int(len(pages) * min_ratio))
    return {ln for ln, n in counter.items() if n >= threshold}


def _parse_text_and_junk(pdf_path):
    """逐页取文字、去掉页眉页脚，返回 (已清理的页面, 页眉页脚行集合)。

    招股书每页页眉都是「公司名 + 招股意向书」，不去掉的话近四成的块都会吃到
    同等匹配加分，检索排序被拉平。页眉行集合要一并返回：表格标题是从版面文字
    里裁出来的，同样会裁到页眉，必须用同一套规则滤掉。
    """
    doc = pymupdf.open(pdf_path)
    try:
        pages = [{"page": i, "text": page.get_text("text")}
                 for i, page in enumerate(doc, start=1)]
    finally:
        doc.close()

    pages = [p for p in pages if p["text"].strip()]
    if not pages:
        return pages, set()

    junk = _repeated_lines(pages)
    for page in pages:
        page["text"] = "\n".join(
            ln for ln in page["text"].splitlines()
            if ln.strip() and _norm(ln) not in junk and not _PAGE_NO.match(ln)
        )
    return pages, junk


def parse_pdf_tables(pdf_path, junk_lines=()):
    """抽取表格并转成 Markdown，同时带上表格上方的标题。

    工单3 的核心：表格本身只有「赵马克 | 42.35% | 公司控股股东」这样的单元格，
    而「关联方」这个关键词在表格上方的标题行里。不带标题的话，这张表在检索时
    会被文档里其它一大堆「持股比例」表格淹没 —— 实测正确表格块融合名次只有 43，
    排不进 Top-8。带上标题后「存在控制关系的关联方」进入表格块，才能被召回。
    """
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            prev_bottom = 0
            # 按纵向位置排序，保证 prev_bottom 始终是紧邻上方那张表的底边
            for found in sorted(page.find_tables(), key=lambda t: t.bbox[1]):
                md = _table_to_markdown(found.extract())
                if md:
                    caption = _caption_above(page, found.bbox, prev_bottom, junk_lines)
                    tables.append({"page": i, "table": md, "caption": caption})
                prev_bottom = found.bbox[3]
    return tables


def _caption_above(page, bbox, top_limit=0, junk_lines=(), max_lines=2):
    """取表格上方最近的若干行文字作为标题。

    top_limit 是上一张表格的底边 —— 一页里常有多张表挨着排，不设下界的话
    上一张表的数据行会被当成标题的一部分（实测把「赵马克 42.35% 公司控股股东」
    混进了下一张表的标题里）。

    junk_lines 是页眉页脚行，同样要滤掉：不用同一套规则过滤的话，裁出来的
    标题会变成「武汉力源信息技术股份有限公司 招股意向书 3、报告期内曾为关联方…」。
    """
    top = max(bbox[1] - 1, top_limit + 1)
    if top <= top_limit:
        return ""
    try:
        region = page.crop((0, top_limit, page.width, top))
        lines = [ln.strip() for ln in (region.extract_text() or "").splitlines()]
    except Exception:                                          # noqa: BLE001
        return ""
    lines = [ln for ln in lines
             if ln and _norm(ln) not in junk_lines and not _PAGE_NO.match(ln)]
    return " ".join(lines[-max_lines:])


def _drop_empty(table):
    """删掉整行/整列都为空的单元格。

    pdfplumber 会因为版面里的空白间隙切出大量空列。实测第 157 页的关联方表
    被切成 9 列，其中 6 列全空：
        |  | 关联方名称 |  |  | 持股比例 |  |  | 与本公司关系 |  |
        | 赵马克 |  |  | 42.35% |  |  | 公司控股股东 |  |
    清理后压成 3 列，表格块体积平均减少约 16%，检索时也不再被 `|  |  |` 稀释。
    """
    rows = [r for r in table if any(_cell(c) for c in r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    keep = [j for j in range(width)
            if any(j < len(r) and _cell(r[j]) for r in rows)]
    return [[r[j] if j < len(r) else "" for j in keep] for r in rows]


def _table_to_markdown(table):
    if not table or not table[0]:
        return ""
    table = _drop_empty(table)
    if not table:
        return ""
    rows = ["| " + " | ".join(_cell(c) for c in row) + " |" for row in table]
    width = max(len(row) for row in table)          # 按最宽的一行定列数，防止错位
    sep = "| " + " | ".join(["---"] * width) + " |"
    return "\n".join([rows[0], sep] + rows[1:])


def _cell(value):
    return "" if value is None else str(value).replace("\n", " ").strip()


# ---------------- 文本分块 ----------------
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """固定长度滑窗切分，尽量在换行处断开，避免把财务数字切成两半。"""
    text = text.strip()
    if not text:
        return []
    overlap = min(overlap, chunk_size - 1)          # 防止步长 <=0 造成死循环
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            cut = text.rfind("\n", start + chunk_size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap
    return chunks


def build_chunks(text_pages, table_blocks):
    """把文字页和表格切成带页码、带类型的块。"""
    chunks = [
        {"text": piece, "page": page["page"], "type": "text"}
        for page in text_pages for piece in split_text(page["text"])
    ]
    # 表格块带上标题，否则「关联方」这类关键词只存在于标题里，表格块检索不到
    chunks += [
        {"text": (f"【{t['caption']}】\n" if t.get("caption") else "") + t["table"],
         "page": t["page"], "type": "table"}
        for t in table_blocks
    ]
    return chunks


def load_pdf(pdf_path):
    """返回 (文字页列表, 表格块列表)"""
    pages, junk = _parse_text_and_junk(pdf_path)
    return pages, parse_pdf_tables(pdf_path, junk)


def company_name(pdf_path, head_pages=3):
    """从 PDF 开头几页提取发行人公司名（出现次数最多的「XX股份有限公司」）。

    多文档路由要用它：问题里出现哪家公司名，就把检索交给对应的知识库。
    """
    doc = pymupdf.open(pdf_path)
    try:
        text = "".join(doc[i].get_text() for i in range(min(head_pages, doc.page_count)))
    finally:
        doc.close()
    names = _COMPANY.findall(text)
    return Counter(names).most_common(1)[0][0] if names else Path(pdf_path).stem
