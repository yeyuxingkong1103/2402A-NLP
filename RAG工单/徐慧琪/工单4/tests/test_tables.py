# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
import pytest
from rag04.config import PROJECT_ROOT
from rag04.ingest.tables import (
    rows_to_markdown, extract_tables, iter_all_tables, label_value_hints,
)

GY2 = PROJECT_ROOT / "招股说明书2.pdf"
needs_corpus = pytest.mark.skipif(not GY2.exists(), reason="语料缺失")


def test_rows_to_markdown_shape():
    md = rows_to_markdown([["项目", "金额"], ["A", "1"], ["B", "2"]])
    assert md.startswith("| 项目 | 金额 |")
    assert "| --- | --- |" in md
    assert "| A | 1 |" in md
    assert "| B | 2 |" in md


def test_rows_to_markdown_escapes_pipes_and_newlines():
    md = rows_to_markdown([["a|b", "c\nd"]])
    assert "a\\|b" in md
    assert "c d" in md


def test_rows_to_markdown_pads_ragged_rows():
    md = rows_to_markdown([["a", "b", "c"], ["d"]])
    assert "| d | " in md


def test_rows_to_markdown_empty():
    assert rows_to_markdown([]) == ""


@needs_corpus
def test_extract_tables_returns_blocks_with_bbox():
    blocks = extract_tables(GY2, "招股说明书2", 22)
    assert blocks, "p22 应含表格"
    b = blocks[0]
    assert b.doc_id == "招股说明书2" and b.page == 22
    assert b.block_type == "table"
    assert b.n_rows >= 2 and b.n_cols >= 2
    assert "|" in b.markdown


@needs_corpus
def test_iter_all_tables_finds_many():
    n = 0
    for _ in iter_all_tables(GY2, "招股说明书2"):
        n += 1
        if n >= 50:
            break
    assert n >= 50, f"表格数量过少：{n}"


def test_rows_to_markdown_escapes_backslash_before_pipe():
    # 输入 a\|b → 输出 a\\\|b：先转义反斜杠，`|` 才保持为活分隔符。
    md = rows_to_markdown([["a\\|b", "c"]])
    assert md.splitlines()[0] == "| a" + "\\" * 3 + "|b | c |"


def test_rows_to_markdown_zero_columns_returns_empty():
    assert rows_to_markdown([[]]) == ""
    assert rows_to_markdown([[], []]) == ""


class _FakePage:
    """_tables_from_page 的最小 Page 替身：只需 extract_tables() 与 bbox。"""

    bbox = (0.0, 0.0, 100.0, 100.0)

    def __init__(self, tables):
        self._tables = tables

    def extract_tables(self):
        return self._tables


def test_tables_from_page_skips_all_blank_table():
    from rag04.ingest.tables import _tables_from_page

    blank = _FakePage([[["", " "], [None, "\n"]]])
    assert _tables_from_page(blank, "d", 1) == []


def test_tables_from_page_keeps_table_with_content():
    from rag04.ingest.tables import _tables_from_page

    blocks = _tables_from_page(_FakePage([[["a", ""], ["", "b"]]]), "d", 7)
    assert len(blocks) == 1
    assert blocks[0].page == 7 and blocks[0].n_cols == 2


def test_extract_tables_missing_file_returns_empty():
    assert extract_tables(PROJECT_ROOT / "definitely-missing.pdf", "d", 1) == []


def test_iter_all_tables_missing_file_yields_nothing():
    assert list(iter_all_tables(PROJECT_ROOT / "definitely-missing.pdf", "d")) == []


@needs_corpus
def test_extract_tables_out_of_range_page_returns_empty():
    assert extract_tables(GY2, "招股说明书2", 10**9) == []


def _two_page_pdf(tmp_path):
    """生成 2 页空白 PDF（非语料，解析负担可忽略）。"""
    import fitz

    path = tmp_path / "two-pages.pdf"
    doc = fitz.open()
    for _ in range(2):
        doc.new_page()
    doc.save(str(path))
    doc.close()
    return path


def test_iter_all_tables_opens_pdf_once(monkeypatch, tmp_path):
    import rag04.ingest.tables as tables_mod

    path = _two_page_pdf(tmp_path)
    calls = []
    real_open = tables_mod.pdfplumber.open

    def counting_open(*args, **kwargs):
        calls.append(1)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(tables_mod.pdfplumber, "open", counting_open)
    assert list(iter_all_tables(path, "d")) == []
    assert len(calls) == 1, "全量遍历只应打开一次文档"


def test_iter_all_tables_survives_page_failure(monkeypatch, tmp_path):
    import rag04.ingest.tables as tables_mod

    path = _two_page_pdf(tmp_path)
    seen = []
    real = tables_mod._tables_from_page

    def flaky(page, doc_id, page_no):
        seen.append(page_no)
        if page_no == 1:
            raise RuntimeError("boom")
        return real(page, doc_id, page_no)

    monkeypatch.setattr(tables_mod, "_tables_from_page", flaky)
    assert list(iter_all_tables(path, "d")) == []
    assert seen == [1, 2], "第 1 页失败后应继续处理第 2 页"


# ---------- RC4：表格行标签 ↔ 数值对应关系保真（重建第二阶段 / 复核收紧） ----------

class _FakeKVRow:
    def __init__(self, bbox):
        self.bbox = bbox


class _FakeKVTable:
    """find_tables() 返回的最小 Table 替身：只需 bbox 与 rows[].bbox（4 元组）。"""

    def __init__(self, bbox, rows):
        self.bbox = bbox
        self.rows = [_FakeKVRow(r) for r in rows]


class _FakeKVPage:
    """RC4 伪页：仿真实 p22 的版面——网格只识别出中部两栏（值列/标签列），
    行带正确；折行单元格、报告期表头行、纯文字表头行都按真实形态布置。

    伪页词 x 范围越出网格 bbox（真实 p22 同形），故「完整网格首行是表头」
    这条规则在此不生效——它由真实语料测试覆盖。
    """

    bbox = (0.0, 0.0, 520.0, 800.0)
    TABLE_BBOX = (140.0, 235.0, 395.0, 478.0)
    # 行带用 4 元组（x0, top, x1, bottom），与 pdfplumber Table.rows[].bbox 同形
    ROWS = [(140.0, 235.0, 395.0, 267.0), (140.0, 299.0, 395.0, 319.2),
            (140.0, 319.2, 395.0, 382.0), (140.0, 382.0, 395.0, 450.8),
            (140.0, 450.8, 395.0, 478.0)]

    def __init__(self):
        self._tables = [[["5,520.00万元", "法定代表人"],
                         ["程家明", "实际控制人"]]]
        self._words = [
            # 数据行：中文名称 = 武汉…（同视觉行另有成立日期 → 整行有数值）
            self._w(90, 251.0, "中文名称"),
            self._w(145, 251.0, "武汉兴图新科电子股份有限公司"),
            self._w(400, 251.0, "2004年6月17日"),
            # 注册资本 / 法定代表人 行（id543 的目标）
            self._w(90, 309.0, "注册资本"), self._w(145, 309.0, "5,520.00万元"),
            self._w(308, 309.0, "法定代表人"), self._w(400, 309.0, "程家明"),
            # 折行单元格：值被切成片段（真实 p22 的注册地址 / 主要生产经营地）
            self._w(308, 327.0, "湖北省武汉市东湖新技"),
            self._w(145, 335.0, "湖北省武汉市东湖新技术开发区"),
            self._w(308, 342.0, "主要生产经营地"),
            self._w(400, 342.0, "术开发区关山大道1号"),
            self._w(90, 350.0, "注册地址"),
            self._w(145, 350.0, "关山大道1号软件产业三期A3栋"),
            self._w(308, 358.0, "址"), self._w(400, 358.0, "软件产业三期A3栋8"),
            self._w(145, 366.0, "8层"), self._w(400, 374.0, "层"),
            # 报告期表头行（同行 ≥2 个期间）与纯文字表头行（整行无数字）
            self._w(90, 400.0, "项目"), self._w(145, 400.0, "2019年1-6月"),
            self._w(250, 400.0, "2018年度"),
            self._w(90, 430.0, "项目"), self._w(145, 430.0, "类型"),
        ]

    @staticmethod
    def _w(x0, top, text, height=12.0):
        return {"x0": float(x0), "x1": float(x0 + 10 * len(text)),
                "top": float(top), "bottom": float(top + height), "text": text}

    def extract_tables(self):
        return self._tables

    def find_tables(self):
        return [_FakeKVTable(self.TABLE_BBOX, self.ROWS)]

    def extract_words(self, **kwargs):
        return list(self._words)


def _hints(md: str) -> str:
    """取表块 Markdown 里的行标签对照段（没有则空串）。"""
    return md.split("[行标签对照", 1)[-1] if "[行标签对照" in md else ""


def _fake_bands(page):
    from rag04.ingest.tables import row_bands

    return row_bands(page.find_tables()[0])


def _fake_pairs():
    page = _FakeKVPage()
    return label_value_hints(page, page.TABLE_BBOX, _fake_bands(page))


def test_label_value_hints_pairs_label_with_same_line_value():
    pairs = _fake_pairs()
    assert ("中文名称", "武汉兴图新科电子股份有限公司") in pairs
    assert ("注册资本", "5,520.00万元") in pairs
    assert ("法定代表人", "程家明") in pairs


def test_label_value_hints_drops_folded_fragment_values():
    """折行单元格的值只是片段（真实 p22 注册地址/主要生产经营地）：不得产出。"""
    pairs = _fake_pairs()
    assert not any(lbl.startswith("注册地址") for lbl, _ in pairs)
    assert not any("主要生产经营" in lbl for lbl, _ in pairs)
    assert all("术开发区关山大道1号" != v for _, v in pairs)
    assert all("关山大道1号软件产业三期A3栋" != v for _, v in pairs)


def test_label_value_hints_skips_header_lines():
    """报告期表头行（项目=2019年1-6月）与纯文字表头行（项目=类型）都不是键值对。"""
    pairs = _fake_pairs()
    assert not any(lbl == "项目" for lbl, _ in pairs)


def test_label_value_hints_rejects_numeric_labels():
    """带阿拉伯数字的段不能当行标签（数值对 `1,840万股 / 不低于25.00%` 跳过）。"""
    page = _FakeKVPage()
    page._words += [page._w(90, 462.0, "发行股数"),
                    page._w(145, 462.0, "1,840万股"),
                    page._w(308, 462.0, "占发行后总股本比例"),
                    page._w(420, 462.0, "不低于25.00%")]
    pairs = label_value_hints(page, page.TABLE_BBOX, _fake_bands(page))
    assert ("占发行后总股本比例", "不低于25.00%") in pairs
    assert not any(lbl.startswith("1,840") for lbl, _ in pairs)
    assert not any(lbl == "不低于25.00%" for lbl, _ in pairs)


def test_label_value_hints_rejects_far_cross_column_pair():
    """标签与相邻栏位隔 260pt（跨列）时不得硬凑成键值对。"""
    page = _FakeKVPage()
    page._words += [page._w(90, 283.0, "英文名称"),
                    page._w(400, 283.0, "2011年1月26日")]
    pairs = label_value_hints(page, page.TABLE_BBOX, _fake_bands(page))
    assert ("英文名称", "2011年1月26日") not in pairs


def test_label_value_hints_without_row_bands_makes_no_claims():
    """取不到行带就无法验证整格完整性：返回空（表头声称权威，宁缺勿错）。"""
    page = _FakeKVPage()
    assert label_value_hints(page, page.TABLE_BBOX) == []


def test_tables_from_page_appends_row_label_cross_reference():
    """表网格错行时，Markdown 必须补上行标签↔数值对照（id543 的 5,520↔注册资本）。"""
    from rag04.ingest.tables import _tables_from_page

    blocks = _tables_from_page(_FakeKVPage(), "d", 22)
    assert len(blocks) == 1
    md = blocks[0].markdown
    assert "| 注册资本 | 5,520.00万元 |" in md
    assert "| 法定代表人 | 程家明 |" in md
    assert "注册地址" not in _hints(md)


@needs_corpus
def test_real_p22_table_keeps_registered_capital_row_mapping():
    """真实语料回归（id543）：p22 表块必须能读出 注册资本=5,520.00万元。"""
    md = "\n".join(t.markdown for t in extract_tables(PROJECT_ROOT / "招股说明书1.pdf",
                                                      "招股说明书1", 22))
    assert "5,520.00万元" in md
    assert "| 注册资本 | 5,520.00万元 |" in md


@needs_corpus
def test_real_p22_hints_have_no_fragment_values():
    """真实语料回归（复核 Important-1）：行标签对照不得出现折行片段值。

    修复前的真实输出含 `主要生产经营地 = 术开发区关山大道1号`、
    `注册地址 = 关山大道1号软件产业三期A3栋`（真实值分别是
    `湖北省武汉市东湖新技术开发区关山大道1号软件产业三期A3栋8层` 的中间/前段）。
    """
    p22 = extract_tables(PROJECT_ROOT / "招股说明书1.pdf", "招股说明书1", 22)
    hints = "\n".join(_hints(t.markdown) for t in p22)
    assert "| 注册资本 | 5,520.00万元 |" in hints
    assert "| 法定代表人 | 程家明 |" in hints
    assert "主要生产经营地" not in hints, "折行单元格的片段值不得入对照"
    assert "注册地址" not in hints
    assert "术开发区关山大道1号" not in hints


@needs_corpus
def test_real_p434_hints_skip_category_and_period_headers():
    """真实语料回归：分类列头（项目=存货类别）与报告期列头（项目=2019年1-6月）不得产出。"""
    hints = "\n".join(_hints(t.markdown) for t in extract_tables(
        PROJECT_ROOT / "招股说明书1.pdf", "招股说明书1", 434))
    assert "| 项目 | 存货类别 |" not in hints
    assert "| 项目 | 2019年1-6月 |" not in hints
    assert "| 原材料 | 685.23 |" in hints, "同表的正常数据行仍应产出"


@needs_corpus
def test_real_p54_hints_skip_complete_grid_header_row():
    """真实语料回归：网格覆盖整行的表，首行带是表头（序号/股东名称…）不得产出。"""
    hints = "\n".join(_hints(t.markdown) for t in extract_tables(
        PROJECT_ROOT / "招股说明书1.pdf", "招股说明书1", 54))
    assert "| 序号 | 股东名称 |" not in hints
    assert "| 持股数量（万股） | 持股比例（%） |" not in hints
