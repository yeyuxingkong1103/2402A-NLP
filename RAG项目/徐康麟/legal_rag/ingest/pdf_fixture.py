# -*- coding: utf-8 -*-
"""生成「含中文的小 PDF」测试夹具（零第三方依赖）。

为什么要自己造 PDF
------------------
验收要求「``legal_rag/ingest/loaders.py`` 的 PDF 路径（pypdf）在真实 PDF 上验证可用」。
沙箱里没有 reportlab/fpdf，也不该为此引入新依赖，于是这里**手写 PDF 语法**：

* 字体用 ``Type0 + Identity-H``，正文按 UTF-16BE 逐字 ``<XXXX>`` 写出；
* 附带一份 **ToUnicode CMap**，让 pypdf 能把 CID 还原成真正的 Unicode 汉字
  （否则抽出来是乱码，中文检索就废了）；
* 生成结果只依赖标准库，任何 Python 环境都能复现。

用法::

    from legal_rag.ingest.pdf_fixture import write_chinese_pdf
    write_chinese_pdf("tests/fixtures/sample_cn.pdf", ["第一页", "第二页"])
"""
from __future__ import annotations

import io
import zlib
from pathlib import Path
from typing import Sequence

__all__ = ["build_chinese_pdf", "write_chinese_pdf", "SAMPLE_PAGES", "ensure_sample_pdf",
           "build_table_pdf", "write_table_pdf", "build_scanned_pdf", "write_scanned_pdf"]

#: 默认示例内容（法律场景的中文，含可被检索命中的关键词）
SAMPLE_PAGES: tuple[str, ...] = (
    "中华人民共和国劳动合同法要点",
    "第一条 用人单位自用工之日起即与劳动者建立劳动关系。",
    "用人单位应当自用工之日起一个月内订立书面劳动合同，"
    "超过一个月不满一年未订立的，应当向劳动者每月支付二倍工资。",
    "民间借贷利率司法保护上限：合同成立时一年期贷款市场报价利率（LPR）的四倍。",
    "劳动争议申请仲裁的时效期间为一年，自当事人知道或者应当知道其权利被侵害之日起计算。",
)

# 页面版式（A4 点单位）
_PAGE_WIDTH = 595
_PAGE_HEIGHT = 842
_FONT_SIZE = 13
_LINE_HEIGHT = 24
_MARGIN_X = 56
_START_Y = 780
_MAX_CHARS_PER_LINE = 34


def _cmap_stream(chars: str) -> bytes:
    """构造 Identity 字体的 ToUnicode CMap（CID == Unicode 码点）。"""
    codes = sorted({ch for ch in chars if not ch.isspace()})
    entries = [f"<{ord(ch):04X}> <{ord(ch):04X}>" for ch in codes]
    body = "\n".join(entries) if entries else ""
    count = len(entries)
    text = (
        "/CIDInit /ProcSet findresource begin\n"
        "12 dict begin\n"
        "begincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def\n"
        "/CMapType 2 def\n"
        "1 begincodespacerange\n"
        "<0000> <FFFF>\n"
        "endcodespacerange\n"
        f"{count} beginbfchar\n{body}\nendbfchar\n"
        "endcmap\n"
        "CMapName currentdict /CMap defineresource pop\n"
        "end\nend\n"
    )
    return text.encode("ascii")


def _wrap(text: str) -> list[str]:
    """按字符宽度粗暴折行（中文字符宽度近似等宽）。"""
    lines: list[str] = []
    for raw in text.splitlines() or [""]:
        current = ""
        for ch in raw:
            if len(current) >= _MAX_CHARS_PER_LINE:
                lines.append(current)
                current = ""
            current += ch
        lines.append(current)
    return [line for line in lines if line]


def _page_stream(lines: Sequence[str], y0: float) -> bytes:
    """**页面内容流的裸内容**（只有绘图/文字算子）。

    ⚠️ 这里**不包含** ``<< /Length ... >> stream ... endstream`` 外壳 ——
    外壳由 :func:`_build_document` 统一加。早先把整段对象塞进来、外层又包一次，
    结果是"流里嵌了对象头"，`pypdf` 抽不出任何文字（正文变成空串）——
    而那个 bug 会被"两边都为空"的相等断言**掩盖**，所以测试必须断言**具体内容**。
    """
    ops = [f"BT /F1 {_FONT_SIZE} Tf {_MARGIN_X} {y0:.0f} Td {_LINE_HEIGHT} TL"]
    for line in lines:
        hex_text = line.encode("utf-16-be").hex().upper()
        # 长行分片，避免单个 Tj 的 hex 字符串过长
        for start in range(0, len(hex_text), 72):
            ops.append(f"<{hex_text[start:start + 72]}> Tj")
        ops.append("T*")
    ops.append("ET")
    return ("\n".join(ops) + "\n").encode("ascii")


def _cell_text(x: float, y: float, text: str, size: float = 11) -> bytes:
    """把一段文本放在页面绝对坐标 (x, y)（PDF 坐标：原点在左下）。"""
    hex_text = str(text).encode("utf-16-be").hex().upper()
    pieces = [f"BT /F1 {size} Tf {x:.1f} {y:.1f} Td"]
    for start in range(0, len(hex_text), 72):
        pieces.append(f"<{hex_text[start:start + 72]}> Tj")
    pieces.append("ET")
    return ("\n".join(pieces) + "\n").encode("ascii")


def _grid_lines(xs: Sequence[float], ys: Sequence[float]) -> bytes:
    """画一张**完整网格**（横线 + 竖线）。

    为什么必须画线：`pdfplumber` 默认（``strategy="lines"``）就是靠**表格线**判定表格的；
    真实的法律文书表格也是这种"有框线"的形态，所以夹具要和真实形态一致。
    """
    ops = ["0.7 w"]
    for y in ys:                                   # 横线
        ops.append(f"{xs[0]:.1f} {y:.1f} m {xs[-1]:.1f} {y:.1f} l S")
    for x in xs:                                   # 竖线
        ops.append(f"{x:.1f} {ys[0]:.1f} m {x:.1f} {ys[-1]:.1f} l S")
    return ("\n".join(ops) + "\n").encode("ascii")


def _build_document(page_streams: Sequence[bytes], all_chars: str, *,
                    resources_extra: str = "",
                    extra_objects: dict[int, bytes] | None = None) -> bytes:
    """把若干**页面内容流**装配成一个 PDF（字体 / ToUnicode / xref 都在这里）。

    ``build_chinese_pdf`` / ``build_table_pdf`` / ``build_scanned_pdf`` 共用它 ——
    否则几套 xref 代码迟早各自长歪。

    ``resources_extra`` 会原样插进每个页面的 ``/Resources``（例如扫描件的
    ``/XObject << /Im0 9 0 R >>``）；``extra_objects`` 是**调用方自己定好对象号**的
    附加对象（例如图像对象）。
    """
    streams = list(page_streams) or [b""]
    # 对象号分配：1=Catalog 2=Pages 3=Font 4=CIDFont 5=FontDescriptor
    # 6=ToUnicode，之后每页两个对象（Contents, Page）
    to_unicode_id = 6
    first_page_id = 7
    page_ids: list[int] = []
    content_ids: list[int] = []
    cursor = first_page_id
    for _ in streams:
        content_ids.append(cursor)
        page_ids.append(cursor + 1)
        cursor += 2

    objects: dict[int, bytes] = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[2] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode()
    objects[3] = (
        b"<< /Type /Font /Subtype /Type0 /BaseFont /SimSun /Encoding /Identity-H "
        b"/DescendantFonts [4 0 R] /ToUnicode 6 0 R >>"
    )
    objects[4] = (
        b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /SimSun "
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
        b"/FontDescriptor 5 0 R /DW 1000 >>"
    )
    objects[5] = (
        b"<< /Type /FontDescriptor /FontName /SimSun /Flags 4 "
        b"/FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 900 /Descent -200 "
        b"/CapHeight 700 /StemV 80 >>"
    )
    cmap_data = _cmap_stream(all_chars)
    objects[to_unicode_id] = (
        b"<< /Length " + str(len(cmap_data)).encode() + b" >>\nstream\n"
        + cmap_data + b"\nendstream"
    )
    for stream, content_id, page_id in zip(streams, content_ids, page_ids):
        data = stream if stream.endswith(b"\n") else stream + b"\n"
        objects[content_id] = (
            b"<< /Length " + str(len(data)).encode() + b" >>\nstream\n"
            + data + b"endstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_PAGE_WIDTH} {_PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 3 0 R >>{resources_extra} >> "
            f"/Contents {content_id} 0 R >>"
        ).encode()
    for object_id, body in (extra_objects or {}).items():
        objects[int(object_id)] = body

    buffer = io.BytesIO()
    buffer.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for object_id in sorted(objects):
        offsets[object_id] = buffer.tell()
        buffer.write(f"{object_id} 0 obj\n".encode())
        buffer.write(objects[object_id])
        buffer.write(b"\nendobj\n")

    xref_offset = buffer.tell()
    max_id = max(objects)
    buffer.write(f"xref\n0 {max_id + 1}\n".encode())
    buffer.write(b"0000000000 65535 f \n")
    for object_id in range(1, max_id + 1):
        if object_id in offsets:
            buffer.write(f"{offsets[object_id]:010d} 00000 n \n".encode())
        else:
            buffer.write(b"0000000000 65535 f \n")
    buffer.write(
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return buffer.getvalue()


def build_table_pdf(headers: Sequence[str], rows: Sequence[Sequence[str]], *,
                    title: str = "", col_width: float = 120.0,
                    row_height: float = 28.0) -> bytes:
    """生成一张**带框线的表格 PDF**（表头 + 若干行），供"表格还原"验收用。

    * 单元格文本按**绝对坐标**放置、框线是完整网格 ⇒ `pdfplumber` 的 lines 策略能识别；
    * ``title`` 在表格上方单独写一行（用来验证"正文与表格都要在"）。
    """
    header_cells = [str(h) for h in headers]
    body = [[str(cell) for cell in row] for row in rows]
    cols = max([len(header_cells)] + [len(row) for row in body]) if (header_cells or body) else 0
    if cols == 0:
        raise ValueError("表格至少要有一列")
    left, top = float(_MARGIN_X), float(_START_Y)
    xs = [left + index * float(col_width) for index in range(cols + 1)]
    grid = ([header_cells] if header_cells else []) + body
    ys = [top - index * float(row_height) for index in range(len(grid) + 1)]

    parts: list[bytes] = []
    chars: list[str] = []
    if title:
        parts.append(_cell_text(left, top + 8, title, size=_FONT_SIZE))
        chars.append(title)
    parts.append(_grid_lines(xs, ys))
    for row_index, row in enumerate(grid):
        baseline = top - (row_index + 1) * float(row_height) + 8
        for col_index in range(cols):
            text = row[col_index] if col_index < len(row) else ""
            if not text:
                continue
            parts.append(_cell_text(xs[col_index] + 6, baseline, text))
            chars.append(text)
    return _build_document([b"".join(parts)], "".join(chars))


def write_table_pdf(path, headers: Sequence[str], rows: Sequence[Sequence[str]],
                    **kwargs) -> Path:
    """把表格 PDF 写到 ``path``，返回该路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(build_table_pdf(headers, rows, **kwargs))
    return target


#: 渲染"扫描件"用的中文字体（Windows 自带；没有就退回 Pillow 默认字体，但中文会变方框）
_CJK_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)

#: 找不到中文字体时给调用方看的一句话（**同一句**同时给日志和测试的 skip 理由，别写两遍）。
CJK_FONT_MISSING_NOTICE = (
    "未找到中文字体（%s）—— 夹具画不出可读的中文，"
    "OCR 类的用例应当 **skip** 而不是失败（这是『这台机器构造不出被测场景』，不是功能坏了）。"
    "装一个即可：apt-get install -y fonts-noto-cjk"
    % " / ".join(_CJK_FONT_CANDIDATES)
)


def cjk_font_path() -> str:
    """本机有没有能画中文的字体？返回路径，没有返回空串。

    ⚠️ 为什么要有这个**公开**函数（2026-09-27 在部署机上踩到）：
    干净的 Linux 服务器**一个 CJK 字体都没有**，而扫描件用的是"画中文位图"的夹具
    ⇒ 中文被画成"豆腐块" ⇒ OCR 当然读不出东西 ⇒ 用例**假失败**（我一开始还把它
    误读成"OCR 在 Linux 上不工作"）。测试需要能**先问一句**再决定跑还是跳。
    """
    for candidate in _CJK_FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return ""


def _cjk_font(size: int):
    """找一个能画中文的字体；找不到返回 Pillow 默认字体（**并明确告知调用方**）。"""
    from PIL import ImageFont

    found = cjk_font_path()
    if found:
        return ImageFont.truetype(found, size), found
    print(f"[注意] {CJK_FONT_MISSING_NOTICE}")
    return ImageFont.load_default(), ""


def build_scanned_pdf(lines: Sequence[str], *, font_size: int = 44,
                      padding: int = 40, line_gap: int = 24) -> bytes:
    """生成一份**纯图片 PDF（没有文字层）** —— 模拟扫描件，供 OCR 验收用。

    实现要点：

    * 用 Pillow 把文字画成位图（中文字体从系统字体里找，找不到会**明确提示**）；
    * 位图按 **RGB + FlateDecode** 作为 ``/XObject`` 嵌进手写 PDF；
    * 页面内容流只做一件事：``q W 0 0 H 0 x y cm /Im0 Do Q``（**不写任何文字**）
      ⇒ `pypdf` 抽出来必然为空 —— 这正是"扫描件"的判定特征。

    对象号：单页文档的 Contents=7 / Page=8，所以图像对象固定用 **9**。
    """
    from PIL import Image, ImageDraw

    if not lines:
        raise ValueError("build_scanned_pdf 至少需要一行文字")
    font, _font_path = _cjk_font(font_size)
    # 先用一张足够大的画布量文本尺寸，再裁到实际大小（避免留大片空白）
    probe = Image.new("RGB", (10, 10), "white")
    probe_draw = ImageDraw.Draw(probe)
    widths, heights = [], []
    for line in lines:
        box = probe_draw.textbbox((0, 0), str(line), font=font)
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    width = max(widths) + padding * 2
    height = sum(heights) + line_gap * (len(lines) - 1) + padding * 2

    image = Image.new("RGB", (max(width, 1), max(height, 1)), "white")
    draw = ImageDraw.Draw(image)
    y = padding
    for line, line_height in zip(lines, heights):
        draw.text((padding, y), str(line), fill="black", font=font)
        y += line_height + line_gap

    raw = image.tobytes()
    compressed = zlib.compress(raw)
    image_object = (
        f"<< /Type /XObject /Subtype /Image /Width {image.width} /Height {image.height} "
        f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
        f"/Length {len(compressed)} >>\nstream\n".encode()
        + compressed + b"\nendstream"
    )
    # 把图按页面尺寸摆放：A4 内留边（_MARGIN_X），等比缩放到可用宽度
    usable = _PAGE_WIDTH - _MARGIN_X * 2
    scale = min(usable / image.width, 1.0)
    draw_width = image.width * scale
    draw_height = image.height * scale
    x = _MARGIN_X
    y_top = _PAGE_HEIGHT - _MARGIN_X - draw_height
    content = (
        f"q {draw_width:.1f} 0 0 {draw_height:.1f} {x:.1f} {y_top:.1f} cm /Im0 Do Q\n"
    ).encode("ascii")
    return _build_document([content], "", resources_extra=" /XObject << /Im0 9 0 R >>",
                           extra_objects={9: image_object})


def write_scanned_pdf(path, lines: Sequence[str], **kwargs) -> Path:
    """把"纯图片 PDF"写到 ``path``，返回该路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(build_scanned_pdf(lines, **kwargs))
    return target


def build_chinese_pdf(pages: Sequence[str]) -> bytes:
    """把若干段中文文本构造为一个多页 PDF，返回字节。"""
    texts = list(pages) or [""]
    streams = [_page_stream(_wrap(text), _START_Y) for text in texts]
    return _build_document(streams, "".join(texts))


def write_chinese_pdf(path, pages: Sequence[str] | None = None) -> Path:
    """把中文 PDF 写到 ``path``，返回该路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(build_chinese_pdf(list(pages) if pages else SAMPLE_PAGES))
    return target


def ensure_sample_pdf(path=None, pages: Sequence[str] | None = None) -> Path:
    """幂等生成示例 PDF：存在且非空就直接复用。"""
    target = Path(path) if path else (
        Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "sample_cn.pdf")
    if target.is_file() and target.stat().st_size > 0:
        return target
    return write_chinese_pdf(target, pages)
