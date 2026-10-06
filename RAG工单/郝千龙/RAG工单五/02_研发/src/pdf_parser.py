# -*- coding: utf-8 -*-
# 【PDF解析优化模块 · pdf_parser.py】双引擎解析、页眉页脚清洗、表格保留、组织结构图几何层级重建
# 工单编号：人工智能NLP-RAG-Query理解优化任务

"""PDF 解析层：输出带标题路径与页码的结构化 Document。

解析策略：
1. PyMuPDF 按文本块提取正文，快速识别标题并拼接硬断行；
2. 对文本稀疏页回退 pdfplumber 提取表格，序列化为 Markdown；
3. 清洗页眉页脚、纯页码、硬回车；识别章节标题层级；
4. 对“组织结构图”所在页，用矢量矩形与线段重建部门→销售处的层级关系，
   生成带父子归属的结构化文本，保证“哪个销售部的销售处最多”类问题可检索。
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pymupdf

try:
    import pdfplumber
except Exception:  # pragma: no cover - pdfplumber 缺失时仅禁用表格回退
    pdfplumber = None

from config import CONFIG


class PDFParseError(Exception):
    """文档解析失败时抛出的自定义异常。"""


@dataclass
class Block:
    """结构化文档块：一段正文或一张表格。"""

    type: str               # text / table / orgchart
    text: str               # 清洗后的文本
    page_no: int            # 1 起始页码
    heading_path: str = ""
    table_title: str = ""
    meta: dict = field(default_factory=dict)


_END_PUNCT = tuple("。！？；：”’）)】」！？.!?;")
# 公司全称正则（用于从页眉识别发行人）
_COMPANY_RE = re.compile(r"[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限责任公司|有限公司)")


def _clean_line(line: str) -> str:
    """清洗单行文本：行内剔除页眉页脚标记与噪声词。"""
    text = line.replace("\u3000", " ").replace("\xa0", " ").strip()
    text = re.sub(r"[\u200b\ufeff\u2000-\u200a\u202f\u205f]", " ", text)
    for pattern in CONFIG.header_footer_patterns:
        text = re.sub(pattern, "", text)
    for word in CONFIG.noise_words:
        text = text.replace(word, "")
    text = re.sub(r"^[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限责任公司)\s*", "", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _looks_tabular(page_text: str) -> bool:
    """启发式判断页面是否为表格页。"""
    if "单位：" not in page_text:
        return False
    return len(re.findall(r"\d[\d,\.]{2,}", page_text)) >= 8


def _is_heading(text: str) -> bool:
    """判断文本行是否为章节标题。

    含冒号（：或:）的“数字、属性：值”行（如“4、法定代表人：赵马克”）
    属于公司概况属性，不应判为标题，否则会被分块阶段跳过导致信息丢失。
    """
    if "：" in text or ":" in text:
        return False
    return bool(re.match(CONFIG.heading_pattern, text)) and len(text) <= 40


def _extract_tables(page_no: int, pdf_page) -> List[Block]:
    """使用 pdfplumber 提取当前页全部表格并序列化为 Markdown。"""
    blocks: List[Block] = []
    for table in pdf_page.extract_tables() or []:
        rows = [[(cell or "").replace("\n", " ").strip() for cell in row]
                for row in table if row]
        if len(rows) < 2:
            continue
        md_lines = ["| " + " | ".join(rows[0]) + " |",
                    "| " + " | ".join(["---"] * len(rows[0])) + " |"]
        md_lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        blocks.append(Block(type="table", text="\n".join(md_lines), page_no=page_no))
    return blocks


def _detect_orgchart(page) -> Optional[str]:
    """识别当前页是否为组织结构图页，并重建部门→销售处的层级文本。

    方法：
    - 先检查页面文本是否同时含“组织结构”与“销售处”；
    - 若是，按 y 坐标分三层识别矩形框（部门层 y~330、子部门层 y~440、销售处层 y~570），
      用矢量连线（水平线+竖直线）重建父子关系；
    - 对每个销售处按 x 坐标找到最近的上层“销售部”作为父部门。
    """
    # 注意：page.get_text() 在竖排文字页可能截断，故用 dict 方式提取全文判断
    d = page.get_text("dict")
    full_text = "".join(
        sp.get("text", "")
        for b in d.get("blocks", []) if b.get("type") == 0
        for line in b.get("lines", [])
        for sp in line.get("spans", [])
    )
    # 组织结构图页特征：同时含“销售处”与“销售部”（图页本身不含“组织结构”字样）
    if "销售处" not in full_text or "销售部" not in full_text:
        return None

    # 收集所有矩形框（排除超大的页面边框）
    boxes: List[Tuple[float, float, float, float, str]] = []
    span_boxes: Dict[int, List] = {}
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for sp in line.get("spans", []):
                t = sp["text"].strip()
                if not t:
                    continue
                x0, y0, x1, y1 = sp["bbox"]
                # 竖排文字：x 近似固定，按 x 桶聚合
                key = round(x0 / 6) * 6
                span_boxes.setdefault(key, []).append((y0, t))
    # 重建竖排/横排文字所在的框（用其 bbox 近似）
    # 同一 x 桶内若有大段 y 间隔（>40），拆分为多个 text_unit，
    # 避免“行政人事部”与“武汉销售处”因 x 接近而被拼成一个单元
    text_units: List[Tuple[float, float, float, float, str]] = []
    for key, lst in span_boxes.items():
        lst.sort()
        # 按 y 间隔分段
        segments = []
        cur = [lst[0]]
        for item in lst[1:]:
            if item[0] - cur[-1][0] > 40:
                segments.append(cur)
                cur = [item]
            else:
                cur.append(item)
        segments.append(cur)
        for seg in segments:
            chars = "".join(t for _, t in seg)
            if not any("一" <= c <= "龥" for c in chars):
                continue
            y0 = seg[0][0]
            y1 = seg[-1][0] + 12
            x0 = key
            x1 = x0 + 30
            text_units.append((x0, y0, x1, y1, chars))

    # 按 y 分层：第三层（最下，销售处）y0 >= 550
    sales_offices = [u for u in text_units if "销售处" in u[4] and u[1] >= 540]
    # 第二层（销售部/子部门）y 在 430~535
    sales_depts = [u for u in text_units if "销售部" in u[4] and 430 <= u[1] <= 540]
    # 第一层（职能部门）y 在 310~400
    top_depts = [u for u in text_units if "部" in u[4] and 310 <= u[1] <= 400
                 and "销售" not in u[4]]

    if not sales_offices:
        return None

    # 销售处归属：按水平连通线重建。线段中 y~543 的水平线为公共母线，
    # 销售处竖线 (x, 543) -> (x, 566)；销售部中只有“大客户销售部”的竖线
    # 向下连到 y=543（坐标 x=350.8）。其余销售部不与该母线相连。
    # 此处按几何 x 距离并结合连线判定：取有竖线连到 y=543 水平线的销售部。
    draws = page.get_drawings()
    vertical_touches_bus: List[float] = []
    horizontal_bus_y = None
    for dd in draws:
        for item in dd["items"]:
            if item[0] == "l":
                a, b = item[1], item[2]
                if abs(a.x - b.x) < 1.5 and abs(a.y - b.y) > 10:
                    # 竖线：若下端 y 接近某水平线（母线）则记录 x
                    pass
                elif abs(a.y - b.y) < 1.5 and abs(a.x - b.x) > 200:
                    # 水平线且跨度大 → 母线
                    if 535 <= a.y <= 550:
                        horizontal_bus_y = a.y
    if horizontal_bus_y is not None:
        for dd in draws:
            for item in dd["items"]:
                if item[0] == "l":
                    a, b = item[1], item[2]
                    if abs(a.x - b.x) < 1.5:
                        ys = [a.y, b.y]
                        # 只保留“母线上方”的竖线（销售部向下连到母线），
                        # 排除母线下方的销售处竖线，避免销售处 x 与销售部 x 混淆
                        if min(ys) <= horizontal_bus_y + 2 <= max(ys) + 2 \
                                and max(ys) <= horizontal_bus_y + 5:
                            vertical_touches_bus.append(a.x)

    # 归属逻辑：只有“有竖线触达母线”的销售部才是销售处的父部门；
    # 所有销售处的竖线都连到同一母线，故全部归属到该销售部。
    parent_dept = None
    for sd in sales_depts:
        sdx = sd[0]
        if any(abs(x - sdx) < 12 for x in vertical_touches_bus):
            parent_dept = sd
            break
    parent_map: Dict[str, List[str]] = {}
    if parent_dept is not None:
        parent_map[parent_dept[4]] = [so[4] for so in sales_offices]

    if not parent_map:
        return None

    # 从页面页眉提取发行公司全称，注入到组织结构图文本中（多文档主语约束用）
    page_company = ""
    for m in _COMPANY_RE.finditer(full_text):
        page_company = m.group(0)
        break  # 页眉通常是第一个公司名

    # 统计哪个销售部的销售处最多
    ranked = sorted(parent_map.items(), key=lambda kv: len(kv[1]), reverse=True)
    top_dept, top_offices = ranked[0]
    lines = []
    if page_company:
        lines.append(f"{page_company}组织结构图：销售部门与销售处归属")
    else:
        lines.append("组织结构图：销售部门与销售处归属")
    lines.append(f"销售处最多的销售部：{top_dept}（共 {len(top_offices)} 个销售处）。")
    lines.append(f"{top_dept}下设销售处：{'、'.join(top_offices)}。")
    if len(ranked) > 1:
        for dept, offices in ranked[1:]:
            lines.append(f"{dept}下设销售处：{'、'.join(offices)}。")
    all_offices = []
    for _, offices in ranked:
        all_offices.extend(offices)
    lines.append(f"公司全部销售处共 {len(all_offices)} 个：{'、'.join(all_offices)}。")
    return "\n".join(lines)


def parse_pdf(path: str) -> List[Block]:
    """解析 PDF 为结构化 Block 列表。

    :param path: PDF 文件绝对路径
    :return: 有序 Block 列表
    :raises PDFParseError: 文件无法打开或解析失败时抛出
    """
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise PDFParseError(f"无法打开PDF文件: {path}，原因: {exc}") from exc

    plumber = None
    if pdfplumber is not None:
        try:
            plumber = pdfplumber.open(path)
        except Exception:
            plumber = None

    blocks: List[Block] = []
    heading_stack: List[str] = []
    try:
        for index, page in enumerate(doc):
            page_no = index + 1

            # ① 组织结构图几何重建（若该页是组织结构图）
            org_text = _detect_orgchart(page)
            if org_text:
                current_heading = " > ".join(heading_stack) or "发行人组织结构"
                blocks.append(Block(
                    type="orgchart", text=org_text, page_no=page_no,
                    heading_path=current_heading,
                ))

            raw_dict = page.get_text("dict")
            page_h = page.rect.height
            page_lines: List[str] = []
            for item in raw_dict.get("blocks", []):
                if item.get("type") != 0:
                    continue
                line_text = "".join(
                    "".join(span.get("text", "") for span in line.get("spans", []))
                    for line in item.get("lines", [])
                )
                x0, y0, _x1, y1 = item.get("bbox", (0, 0, 0, 0))
                in_margin = (y1 < page_h * 0.06) or (y0 > page_h * 0.95)
                if in_margin and len(line_text.strip()) < 40:
                    continue
                cleaned = _clean_line(line_text)
                if not cleaned:
                    continue
                if _is_heading(cleaned):
                    if cleaned.startswith("第") and ("章" in cleaned or "节" in cleaned):
                        heading_stack = [cleaned]
                    elif re.match(r"^[一二三四五六七八九十]+、", cleaned):
                        heading_stack = heading_stack[:1] + [cleaned]
                page_lines.append(cleaned)

            merged, buf = [], ""
            for line in page_lines:
                if _is_heading(line):
                    if buf:
                        merged.append(buf)
                        buf = ""
                    merged.append(line)
                elif buf and not buf.endswith(_END_PUNCT):
                    buf += line
                else:
                    if buf:
                        merged.append(buf)
                    buf = line
            if buf:
                merged.append(buf)

            page_text = "".join(merged)
            page_text_len = len(page_text)
            current_heading = " > ".join(heading_stack)

            table_blocks: List[Block] = []
            if plumber is not None:
                need_table = (page_text_len < CONFIG.table_fallback_chars
                              or _looks_tabular(page_text))
                if need_table:
                    table_blocks = _extract_tables(page_no, plumber.pages[index])
            table_chars = sum(len(b.text) for b in table_blocks)
            if table_blocks and table_chars > page_text_len * 0.6:
                blocks.extend(table_blocks)
                continue

            for seg in merged:
                if _is_heading(seg):
                    continue
                blocks.append(Block(
                    type="text", text=seg, page_no=page_no,
                    heading_path=current_heading,
                ))
            blocks.extend(table_blocks)
    except PDFParseError:
        raise
    except Exception as exc:
        raise PDFParseError(f"解析PDF失败: {path}，原因: {exc}") from exc
    finally:
        doc.close()
        if plumber is not None:
            plumber.close()
    return blocks
