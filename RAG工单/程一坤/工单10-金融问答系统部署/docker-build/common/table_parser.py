# -*- coding: utf-8 -*-
"""
表格解析模块（工单03）
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
说明：基于 PyMuPDF find_tables 提取表格，序列化为"列名: 值"文本行，
     使表格数据可被向量化检索（金融招股书的核心数字多在表格中）。
"""
import fitz  # PyMuPDF


def _cell(c):
    return str(c or "").replace("\n", "").strip()


def _table_to_text(rows):
    """把二维表序列化为 '列名: 值 | 列名: 值' 行文本。
    处理合并单元格：空表头向前继承最近非空表头（如"2019年1-6月/金额/占比"
    三列结构），避免序列化后列名丢失导致 LLM 误读行列。"""
    if not rows:
        return ""
    header = [_cell(c) for c in rows[0]]
    filled, last = [], ""
    for h in header:
        if h:
            last = h
            filled.append(h)
        else:
            filled.append(last + "(续)" if last else "")
    header = filled

    lines = []
    for r in rows[1:]:
        cells = [_cell(c) for c in r]
        pairs = [f"{header[j]}: {cells[j]}" for j in range(min(len(header), len(cells)))
                 if cells[j]]
        # 无表头列名时退化为原始行拼接
        if not pairs:
            pairs = [c for c in cells if c]
        if pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(lines)


def _table_caption(page, tab, max_chars=60):
    """提取表格正上方的文字作为标题（表格标题在网格外，find_tables 抓不到，
    不补标题会导致"不存在控制关系的关联方"这类标题型检索词失配）"""
    try:
        x0, y0, x1, y1 = tab.bbox
        words = [w for w in page.get_text("words")
                 if y0 - 55 <= w[3] <= y0 - 2 and w[0] >= x0 - 80]
        words.sort(key=lambda w: (round(w[1], 1), w[0]))
        return "".join(w[4] for w in words)[:max_chars]
    except Exception:
        return ""


def parse_pdf_tables(pdf_path, min_rows=2):
    """提取 PDF 全部表格
    返回: [{"page": 页码, "text": 序列化表格文本}, ...]
    """
    doc = fitz.open(pdf_path)
    out = []
    for i, page in enumerate(doc):
        try:
            tabs = page.find_tables()
        except Exception:
            continue
        for tab in tabs.tables:
            rows = tab.extract()
            if not rows or len(rows) < min_rows:
                continue
            body = _table_to_text(rows)
            # 过短/无数字的表格对问答价值低，过滤
            if len(body) < 20 or not any(ch.isdigit() for ch in body):
                continue
            caption = _table_caption(page, tab)
            head = f"【表格数据 | 第{i+1}页】"
            if caption:
                head += f" 表格标题：{caption}"
            # 超长表格按行切分为多个分块（避免超出嵌入模型上下文导致500）
            body_lines = body.split("\n")
            cur = head
            for ln in body_lines:
                if len(cur) + len(ln) > 2200 and cur != head:
                    out.append({"page": i + 1, "text": cur})
                    cur = head + "\n(续)" + ln
                else:
                    cur += "\n" + ln
            if cur.strip() and cur != head:
                out.append({"page": i + 1, "text": cur})
    doc.close()
    return out
