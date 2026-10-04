# -*- coding: utf-8 -*-
"""
表格解析模块（工单03）
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
说明：基于 PyMuPDF find_tables 提取表格，序列化为"列名: 值"文本行，
     使表格数据可被向量化检索（金融招股书的核心数字多在表格中）。
"""
import fitz  # PyMuPDF：提供 find_tables 表格识别能力


def _cell(c):
    """清洗单元格：None/换行处理。表格单元格常含内部换行，会破坏行序列化格式"""
    # c or ""：None 单元格（空格）转空串；replace 去掉单元格内换行防串行
    return str(c or "").replace("\n", "").strip()


def _table_to_text(rows):
    """把二维表序列化为 '列名: 值 | 列名: 值' 行文本。
    处理合并单元格：空表头向前继承最近非空表头（如"2019年1-6月/金额/占比"
    三列结构），避免序列化后列名丢失导致 LLM 误读行列。"""
    if not rows:
        return ""  # 空表直接返回空串
    header = [_cell(c) for c in rows[0]]  # 第一行作表头
    filled, last = [], ""
    for h in header:
        if h:
            last = h  # 非空表头：更新"最近有效表头"
            filled.append(h)
        else:
            # 空表头（合并单元格被拆开后遗留）：继承左侧最近表头并加"(续)"标记
            filled.append(last + "(续)" if last else "")
    header = filled  # 补全后的表头替换原表头

    lines = []
    for r in rows[1:]:
        cells = [_cell(c) for c in r]  # 清洗数据行各单元格
        # 逐列拼 "表头: 值"，min() 防表头与数据行列数不一致时越界；
        # if cells[j] 过滤空单元格（合并单元格的续行只有部分列有值）
        pairs = [f"{header[j]}: {cells[j]}" for j in range(min(len(header), len(cells)))
                 if cells[j]]
        # 无表头列名时退化为原始行拼接
        if not pairs:
            pairs = [c for c in cells if c]
        if pairs:
            # 用 " | " 分隔各列，LLM 读起来像键值对，数字与列名强关联
            lines.append(" | ".join(pairs))
    return "\n".join(lines)


def _table_caption(page, tab, max_chars=60):
    """提取表格正上方的文字作为标题（表格标题在网格外，find_tables 抓不到，
    不补标题会导致"不存在控制关系的关联方"这类标题型检索词失配）"""
    try:
        x0, y0, x1, y1 = tab.bbox  # 表格在页面上的包围盒（左上/右下坐标）
        # 取"表格上方 2~55pt 范围内、横向不左偏超 80pt"的文字词块：
        # 即紧贴表格上沿的标题区域
        words = [w for w in page.get_text("words")
                 if y0 - 55 <= w[3] <= y0 - 2 and w[0] >= x0 - 80]
        # words 元素为 (x0,y0,x1,y1,text,...)：先按 y（行）再按 x（列内顺序）排序，
        # round 到 0.1 防同一行文字 y 值有微小抖动导致排序错乱
        words.sort(key=lambda w: (round(w[1], 1), w[0]))
        # 拼接词的文本（w[4]），截断到 max_chars 防超长标题污染向量
        return "".join(w[4] for w in words)[:max_chars]
    except Exception:
        # bbox 异常/无文字时返回空串，主流程跳过标题拼接即可
        return ""


def parse_pdf_tables(pdf_path, min_rows=2):
    """提取 PDF 全部表格
    返回: [{"page": 页码, "text": 序列化表格文本}, ...]
    """
    doc = fitz.open(pdf_path)
    out = []
    for i, page in enumerate(doc):
        try:
            # find_tables 返回 TableFinder，表格在 .tables 属性中
            tabs = page.find_tables()
        except Exception:
            # 个别损坏页/加密页解析抛异常，跳过该页不影响整体
            continue
        for tab in tabs.tables:
            rows = tab.extract()  # 提取二维数组（首行为表头）
            if not rows or len(rows) < min_rows:
                # 少于 min_rows 行的"表格"多是装饰性线条框，无信息量
                continue
            body = _table_to_text(rows)
            # 过短/无数字的表格对问答价值低，过滤
            # len<20：几乎空表；无数字：金融招股书的表格核心是数字，纯文字框价值低
            if len(body) < 20 or not any(ch.isdigit() for ch in body):
                continue
            caption = _table_caption(page, tab)  # 补表格标题，提升标题型查询的召回
            head = f"【表格数据 | 第{i+1}页】"
            if caption:
                head += f" 表格标题：{caption}"
            # 超长表格按行切分为多个分块（避免超出嵌入模型上下文导致500）
            body_lines = body.split("\n")
            cur = head  # cur 是当前累积的表格分块
            for ln in body_lines:
                # 阈值 2200 字符：为 bge-m3 的 4000 字符截断留余量；
                # cur != head 保证至少装一行数据再切
                if len(cur) + len(ln) > 2200 and cur != head:
                    out.append({"page": i + 1, "text": cur})
                    # 新块重打表头并标"(续)"：续块单独被检索到时也知来源
                    cur = head + "\n(续)" + ln
                else:
                    cur += "\n" + ln
            # 收尾：把最后未满阈值的剩余部分入库（cur != head 防空块）
            if cur.strip() and cur != head:
                out.append({"page": i + 1, "text": cur})
    doc.close()
    return out
