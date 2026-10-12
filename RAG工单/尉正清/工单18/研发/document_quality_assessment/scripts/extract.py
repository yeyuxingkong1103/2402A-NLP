# 工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流
"""正文抽取：把各种格式的文件读成文本

从 `common.py` 拆出来的。拆的边界不是按行数硬切，是按职责：

- `common.py` —— 管线（读配置、遍历目录、断点缓存、杂项）
- `extract.py` —— 抽取（把文件读成文本，以及判扫描页要用的物理量）

所有函数**异常全吞**，返回错误字符串而不是抛异常 —— 知识库里损坏文件是常态，
不能让一份坏 PDF 掀掉整轮质检（这条是工单验收标准点名的要求）。
"""
from pathlib import Path


# ---------------------------------------------------------------- 抽正文

def pdf_parse(path, sample_pages=0):
    """打开一次 PDF，同时拿到**全文**、**逐页字符数**、**逐页图片覆盖率**。

    ## 为什么要一起拿

    判类型要「每页字符数」，而长度分布、SimHash、敏感信息都要全文。
    分两次 open 的话 1700 份 PDF 要白解析一遍 —— 打开并解析大 PDF 的开销
    远大于把文本拼起来。图片覆盖率顺手一起算，见下。

    ## 为什么还要「图片覆盖率」

    光看字符数判不了扫描页。**专利的附图页**字符数也很少（整页只有「图1」
    「图2」几个字），但它不是扫描件，是数字排版的原生页，不需要 OCR。
    实测两者的物理区别很明显：

        扫描页：一整张位图铺满整页，图片覆盖率 ≈ 1.00
        附图页：图只占版面中间一块，图片覆盖率 0.2 ~ 0.45（实测 CN106926565B）

    所以每页记两个物理量：字符数 + 图片面积占比。**两个都是事实**，
    阈值怎么定是出报告时的事（见 `pdf_type.py`）。

    每页只取前 `sample_pages` 页（0 = 全取），全文始终是全部页。

    返回 `(全文, 每页字符数, 每页图片覆盖率, 总页数)`；
    打不开返回 `(None, None, None, 0)`。
    """
    try:
        import fitz
    except ImportError:
        return None, None, None, 0
    try:
        with fitz.open(path) as doc:
            n = doc.page_count
            limit = n if not sample_pages else min(n, sample_pages)
            texts, counts, covers = [], [], []
            for i in range(n):
                page = doc[i]
                try:
                    t = page.get_text("text")
                except Exception:
                    t = ""
                texts.append(t)
                if i < limit:
                    counts.append(len(t.strip()))
                    covers.append(_image_cover(page))
            return "\n".join(texts), counts, covers, n
    except Exception:
        return None, None, None, 0


def _image_cover(page):
    """这一页被图片盖住了多大比例（0~1）。

    整页扫描件接近 1.0，带插图的文字页只有零点几。

    ## 必须用 `get_image_info()`，不能用 `get_text("dict")` 的图片块

    起初用的是后者（图片块的 bbox 面积）。**实测会漏**：有 4 份
    （`CN2926129Y.pdf` 等）是货真价实的扫描件 —— 渲染出来看就是拍下来的
    专利说明书首页 —— 但 `get_text("dict")` 返回的 blocks 是**空的**，
    面积算出来 0.00，于是被判成「文字型」，送去抽文字只会得到一篇空白。

    原因大概是这几份的内容流用 Form XObject 之类的方式画图，
    pymupdf 不在 dict blocks 里报。换 `get_image_info()` 就正常：

        文件                 dict块   get_image_info
        CN2926129Y（扫描件）   0.00    1.00   ← dict 漏了
        CN100408987C（扫描件） 0.93    0.93
        CN111318330A（文字型） 0.09    0.10
        CN106926565B（附图页） 0.10    0.11

    所以以 `get_image_info()` 为准，dict 块只作兜底。
    """
    area = abs(page.rect.width * page.rect.height) or 1
    try:
        info = page.get_image_info()
        got = sum(abs((i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1]))
                  for i in info if i.get("bbox"))
        if got:
            return min(got / area, 1.0)
    except Exception:
        pass
    try:
        got = 0.0
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") == 1:
                x0, y0, x1, y1 = b["bbox"]
                got += abs((x1 - x0) * (y1 - y0))
        return min(got / area, 1.0)
    except Exception:
        return 0.0


def extract_text(path, max_parse_mb=200):
    """抽正文。返回 (文本, 错误信息)。

    按扩展名分派。**任何异常都吞掉并返回错误字符串** ——
    知识库里损坏文件是常态，不能因为一个坏文件让整轮质检挂掉。
    """
    path = Path(path)
    try:
        size_mb = path.stat().st_size / 1048576
    except OSError as e:
        return "", f"读不到文件：{e}"
    if size_mb > max_parse_mb:
        return "", f"文件过大（{size_mb:.0f}MB），跳过正文抽取"

    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            return pdf_text(path), ""
        if ext in (".md", ".markdown", ".txt", ".html"):
            return path.read_text(encoding="utf-8", errors="replace"), ""
        if ext == ".docx":
            return _docx_text(path), ""
        if ext in (".xlsx", ".xls", ".csv"):
            return _sheet_text(path), ""
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"
    return "", f"不支持的扩展名 {ext}"


def pdf_text(path):
    text = pdf_parse(path)[0]
    if text is None:
        raise RuntimeError("PDF 打不开（损坏或加密）")
    return text


def _docx_text(path):
    import docx
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts.append("\t".join(c.text for c in row.cells))
    return "\n".join(parts)


def _sheet_text(path):
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        out = []
        for ws in wb.worksheets:
            out.append(f"[sheet] {ws.title}")
            for row in ws.iter_rows(values_only=True):
                out.append("\t".join("" if v is None else str(v) for v in row))
        return "\n".join(out)
    except ImportError:
        return path.read_text(encoding="utf-8", errors="replace") \
            if path.suffix.lower() == ".csv" else ""
