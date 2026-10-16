# -*- coding: utf-8 -*-
"""T4 单测：表格三大真实缺陷 + 图像块入索引契约（工单4 验收项 3 的直接证据）。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化

覆盖的**真实缺陷**（工单3 实测、工单4 必须保留处理逻辑）：
    缺陷① 合并单元格 → ``find_tables().extract()`` 用 ``None`` 表示「合并后的空格 / 本行没有的格」；
           处理：表头相邻 ``None`` 合并为一列（``merge_none_header``）、数据行 ``None`` 置空保持列位
           （``forward_fill_merged_rows``）、全空列删除（``collapse_none_columns``）；
    缺陷② 格内换行 → 单元格含 ``\\n``（招股书里极常见），必须归一化空白后再转 Markdown；
    缺陷③ 跨页表格是常态（PDF1 312/548 页含表、PDF2 134/350 页含表，最长 28 连页）→
           续表检测（``looks_like_continuation`` + ``first_row_is_data`` + 列数一致）+ 表头继承
           （``merge_continued_tables``）+ 重复表头去重（``is_repeated_header`` / ``dedupe_repeated_header``）。

另覆盖工单4 新增：``chunk_id`` 口径、图像块合并（同页同 modality_sub → 1 块）、
竖排单字复原、``Chunk.to_dict`` 的 ``modality``/``page_basis``、BM25 对图像块注入数字。

运行：
    pwsh -NoProfile -File run_py.ps1 -m pytest 研发/tests/test_table_defects.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import chunker  # noqa: E402
from app.core.bm25_index import BM25Index  # noqa: E402
from app.core.table_parser import (  # noqa: E402
    build_table_block,
    build_table_blocks,
    collapse_none_columns,
    dedupe_repeated_header,
    forward_fill_merged_rows,
    is_repeated_header,
    looks_like_continuation,
    merge_continued_tables,
    merge_none_header,
    normalize_table,
    table_to_markdown,
)
from app.core.text_utils import normalize_cell  # noqa: E402

PROCESSED = REPO_ROOT / "研发" / "data" / "processed"


# ---------------------------------------------------------------------------
# 缺陷①：合并单元格产生 None
# ---------------------------------------------------------------------------
def test_缺陷1_表头相邻None合并为一列():
    """表头 ``['项目', None, None, '金额']`` → 合并组只留 1 列（列数 4 → 2）。"""
    assert merge_none_header(["项目", None, None, "金额"]) == ["项目", "金额"]
    assert merge_none_header(["", None]) == [""]        # 首个即空：保留 1 个空列名
    assert startswith_is_degenerate([""])               # 但 is_degenerate 会把它判为退化（列不足）
    assert merge_none_header(["A", "B"]) == ["A", "B"]  # 正常表头不动


def startswith_is_degenerate(header: list[str]) -> bool:
    """辅助：空表头必须被 ``is_degenerate`` 判为退化表（否则会产出空列名的假表）。"""
    from app.core.table_parser import is_degenerate

    return is_degenerate(header, [["x", "y"]])


def test_缺陷1_数据行None置空且不左填充():
    """``None`` 必须**置空**而不是左填充：左填充会把同一逻辑值复制到多列（BM25 词频虚增 3 倍）。"""
    rows = forward_fill_merged_rows([["合计", None, None, "305,800.00"]])
    assert rows == [["合计", "", "", "305,800.00"]]
    assert rows[0].count("305,800.00") == 1


def test_缺陷1_全空列被删除():
    rows = collapse_none_columns([["A", "", "C"], ["1", "", "3"]])
    assert rows == [["A", "C"], ["1", "3"]]


def test_缺陷1_归一化后矩阵无None且列对齐():
    """``normalize_table`` 产出的 rows 里**不得残留 None**（cells 全部是 str）。"""
    raw = [
        ["项目", None, "金额（万元）"],
        ["合计", None, "1,234.00"],
        ["其中：母公司", None, "1,000.00"],
    ]
    block = normalize_table(raw, page=22, table_id="t#p0022#t00")
    assert block.rows, "应至少有一行数据"
    for row in block.rows:
        assert all(isinstance(cell, str) for cell in row), f"残留非 str 单元格：{row}"
        assert None not in row
    assert block.logical_cols <= 2          # 相邻 None 表头已合并
    assert block.markdown.count("|") > 0


# ---------------------------------------------------------------------------
# 缺陷②：格内换行 → 归一化空白后转 Markdown
# ---------------------------------------------------------------------------
def test_缺陷2_格内换行被归一化为空格():
    assert normalize_cell("有限公司成立日\n期") == "有限公司成立日期"   # 中文之间的换行直接拼接
    assert normalize_cell("2019 年度\n营业收入") == "2019 年度营业收入"
    assert " " in normalize_cell("Video\nCommand")                     # 英文之间留一个空格
    assert "\n" not in normalize_cell("a\nb\nc")


def test_缺陷2_Markdown单元格不含换行也不含字面None():
    raw = [
        ["项目\n名称", "金额\n（万元）"],
        ["军用\n领域收入", "6,464.51"],
    ]
    block = normalize_table(raw, page=129, table_id="t#p0129#t00")
    assert block.markdown
    for line in block.markdown.splitlines():
        assert "\n" not in line
    assert "None" not in block.markdown
    assert "|" in block.markdown
    assert block.key_numbers, "关键数字应被抽取（BM25 数字路依赖它）"


def test_缺陷2_竖线被转义不破坏Markdown表格():
    md = table_to_markdown(["a|b"], [["c|d"]])
    body_lines = md.splitlines()
    assert body_lines[0].count("|") == 3      # 首尾竖线 + 单元格内被转义的那条不再算分隔符
    assert "\\|" in md


# ---------------------------------------------------------------------------
# 缺陷③：跨页续表 / 重复表头 / 表头继承
# ---------------------------------------------------------------------------
def _mk_block(page: int, raw: list[list[str | None]], idx: int = 0):
    return build_table_block(raw, file_name="招股说明书1.pdf", page=page, idx=idx, title=f"{page} 页表格")


def test_缺陷3_续表判据_首格为空才算疑似续表():
    assert looks_like_continuation([None, "1,000.00"]) is True
    assert looks_like_continuation(["", "1,000.00"]) is True
    assert looks_like_continuation(["合计", "1,000.00"]) is False


def test_缺陷3_重复表头判据为精确比较_不给容差():
    """早期版本容忍 1 个格子差异 → 把数据行误判为重复表头并整行丢弃（数据丢失），此处锁死。"""
    prev = _mk_block(10, [["项目", "金额"], ["A", "1"]])
    assert is_repeated_header(prev, list(prev.raw_first_row)) is True
    assert is_repeated_header(prev, ["项目", "金额"]) is True
    assert is_repeated_header(prev, ["项目 ", "金额"]) is True       # 归一化后等价
    assert is_repeated_header(prev, ["项目", "金额合计"]) is False    # 1 格不同 → 必须判 False
    assert is_repeated_header(prev, [None, "1,000.00"]) is False


def test_缺陷3_合并续表_表头继承且页范围扩展():
    prev = _mk_block(10, [["项目", "金额"], ["A", "1"], ["B", "2"]])
    cont = _mk_block(11, [[None, "3"], ["D", "4"]])           # 首格为空的续表
    header_before = list(prev.header)
    merged = merge_continued_tables(prev, cont)
    assert merged.header == header_before, "续表必须继承前块表头（不得自己乱造表头）"
    assert merged.page_start == 10 and merged.page_end == 11
    assert 11 in merged.absorbed_pages
    assert len(merged.rows) >= 4
    assert any("合并续表" in note for note in merged.notes)


def test_缺陷3_重复表头去重只丢表头不丢数据行():
    prev = _mk_block(10, [["项目", "金额"], ["A", "1"]])
    dup = _mk_block(11, [["项目", "金额"], ["B", "2"], ["C", "3"]])
    rows_before = len(prev.rows)
    merged = dedupe_repeated_header(prev, dup)
    assert len(merged.rows) == rows_before + 2, "重复表头行必须被丢弃，数据行必须保留"
    assert any("重复表头已去重" in note for note in merged.notes)


def test_缺陷3_跨页三连页端到端统计():
    """模拟「页10 表头 → 页11 续表 → 页12 续表（带重复表头）→ 页14 独立新表」的常见形态。"""
    raw_tables = [
        _mk_block(10, [["项目", "金额"], ["A", "1"], ["B", "2"]]),
        _mk_block(11, [[None, "3"], ["D", "4"]]),
        _mk_block(12, [["项目", "金额"], ["E", "5"]]),        # 重复表头
        _mk_block(14, [["项目", "金额"], ["F", "6"]]),
    ]
    blocks, stats = build_table_blocks(raw_tables)
    assert stats["continued_tables"] == 1, "页11 应被判为续表并合并"
    assert stats["repeated_headers_removed"] == 1, "页12 的重复表头应被去重"
    ids = [b.table_id for b in blocks]
    # 页10/11/12 同属一个逻辑块（页12 的重复表头被去重后并入），页14 是新块
    assert len(blocks) == 2, f"应合成为 2 个逻辑表块，实际 {ids}"
    first = next(b for b in blocks if b.page_start == 10)
    assert first.page_end == 12, "逻辑块应跨到第 12 页"
    merged_rows = "\n".join(" ".join(r) for r in first.rows)
    for value in ("1", "2", "3", "4", "5"):
        assert value in merged_rows, f"合并后不应丢数据：缺 {value}"


def test_缺陷3_列数不匹配时拒绝合并并留痕():
    prev = _mk_block(10, [["项目", "金额"], ["A", "1"]])
    other = _mk_block(11, [[None, "3", "x"]])
    if prev.flat_cols == other.flat_cols:
        pytest.skip("该构造下两表列数相同，跳过（列数不匹配判据需要不同列宽）")
    blocks, stats = build_table_blocks([prev, other])
    assert stats["merge_rejected"] >= 1 or stats["continued_tables"] == 0


# ---------------------------------------------------------------------------
# 真实语料集成断言（解析产物存在时才跑；不存在则 skip，不制造假绿）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stem", ["招股说明书1", "招股说明书2"])
def test_真实产物_表格无None残留且统计与基线同量级(stem: str):
    path = PROCESSED / f"{stem}.tables.jsonl"
    if not path.is_file():
        pytest.skip(f"解析产物不存在（先跑 build_index.py）：{path}")
    blocks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert blocks, "表格产物不应为空"
    continued = 0
    for block in blocks:
        if block.get("degenerate"):
            # 退化表（列不足 / 无有效数据行）按契约写空 markdown 且不进正文 chunk —— 不算缺陷
            assert block["markdown"] == "", f"{block['table_id']} 退化表不应有 markdown"
            continue
        for row in block["rows"]:
            assert all(isinstance(cell, str) for cell in row), f"{block['table_id']} 残留非 str 单元格"
            # 缺陷②：**单元格内**不得残留换行（Markdown 文档本身当然是多行的）
            assert all("\n" not in cell for cell in row), f"{block['table_id']} 单元格残留换行：{row}"
        assert block["markdown"].count("\n") == len(block["rows"]) + 1, \
            f"{block['table_id']} Markdown 行数与数据行数不匹配（疑似格内换行未归一）"
        if (block.get("page_end") or 0) > (block.get("page_start") or 0):
            continued += 1
    assert continued >= (5 if stem == "招股说明书1" else 1), f"{stem} 跨页续表数量异常：{continued}"
    # PDF1 实测：312/548 页含表（严禁抽样）；这里是产物级断言，只锁「表块存在」与「无 None/无格内换行」
    table_pages = len({b["page_start"] for b in blocks})
    assert table_pages >= (50 if stem == "招股说明书1" else 30), f"{stem} 含表页过少：{table_pages}"


# ---------------------------------------------------------------------------
# 工单4 新增：chunk_id 口径 / 图像块 / 竖排复原
# ---------------------------------------------------------------------------
def test_chunkid_口径与设计文档示例一致():
    assert chunker.chunk_id_for("招股说明书2.pdf", 22, "t", 3, part=0) == "pdf2-p22-t3-0"
    assert chunker.chunk_id_for("招股说明书2.pdf", 39, "img", 0, xref=1693) == "pdf2-p39-img-1693-0"
    assert chunker.chunk_id_for("招股说明书1.pdf", 40, "x", 12) == "pdf1-p40-x12"
    assert chunker.corpus_key("招股说明书1.pdf") == "pdf1"
    assert chunker.corpus_key("招股说明书2.pdf") == "pdf2"
    with pytest.raises(Exception):
        chunker.chunk_id_for("招股说明书2.pdf", 39, "zzz", 0)


def test_图像块_同页同模态子图合并为一块并保留子图元数据():
    """第 6 题：页 72 是左右并列的两张子图，必须合成 1 块（captain 2026-10 复核结论）。"""
    records = [
        {"chunk_id": "img:c:p72:chart:x1693", "corpus": "招股说明书2.pdf", "page": 72,
         "modality_sub": "chart", "xref": 1693, "bbox": [10, 10, 100, 200],
         "text": "IC卡 -2.0%（负增长）", "extraction_method": "geometry+anchor",
         "structured": {"series": [{"label": "IC卡", "value": -2.0}]}},
        {"chunk_id": "img:c:p72:chart:x1692", "corpus": "招股说明书2.pdf", "page": 72,
         "modality_sub": "chart", "xref": 1692, "bbox": [100.9, 10, 200, 200],
         "text": "汽车 14.0%（增长率最快）", "extraction_method": "geometry+anchor",
         "structured": {"series": [{"label": "汽车", "value": 14.0}]}},
    ]
    merged = chunker.merge_image_blocks(records)
    assert len(merged) == 1, "同页同 modality_sub 必须合并成 1 组"
    chunks = chunker.build_image_chunks(merged)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.type == "image" and chunk.modality == "image"
    assert chunk.page == 72 and 72 >= 1                    # 1-based 物理页码
    assert "IC卡" in chunk.content and "汽车" in chunk.content
    assert chunk.structured and len(chunk.structured.get("series") or []) == 2
    assert chunk.sub_bboxes and len(chunk.sub_bboxes) == 2
    assert chunk.chunk_id.startswith("pdf2-p72-img-")
    payload = chunk.to_dict()
    assert payload["modality"] == "image" and payload["page_basis"] == "1-based-physical"


def test_图像块_缺页码即拒收不静默():
    with pytest.raises(Exception):
        chunker.merge_image_blocks([{"chunk_id": "x", "corpus": "招股说明书2.pdf", "page": 0}])


def test_竖排单字复原_第5题框图标签():
    raw = "武汉力源信息技术股份有限公司\n38\n股东大会\n财\n务\n部\n销\n售\n部\n大\n客\n户\n销\n售\n部\n董事会\n"
    text = chunker.descramble_vertical_text(raw)
    assert "财务部" in text
    assert "销售部" in text
    assert "大客户销售部" in text
    assert "股东大会" in text and "董事会" in text


def test_真实PDF_页39竖排标签按x列聚类可还原():
    """真实语料断言（工单4 第 5 题的降级路径）：页 39 的竖排标签必须能被还原成完整词。

    若本机缺少 PDF/依赖则 skip（**不制造假绿**）。这条测试锁死「按 x 列聚类」这一正确做法：
    页 39 的 17 个标签在纯文本里是一段 85 字的单字流，任何按长度截断的启发式都会串错。
    """
    pdf = REPO_ROOT / "研发" / "data" / "raw" / "招股说明书2.pdf"
    if not pdf.is_file():
        pytest.skip(f"语料不存在：{pdf}")
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open(str(pdf))
    try:
        labels = chunker.extract_vertical_labels(doc.load_page(38))   # 0-based 38 = 物理第 39 页
    finally:
        doc.close()
    joined = "、".join(labels)
    for expected in ("销售部", "大客户销售部", "电话及网络销售部", "渠道销售部", "国际贸易部",
                     "珠海销售处", "成都销售处", "财务部", "市场开发部"):
        assert expected in joined, f"页 39 竖排标签缺失 {expected}：{joined}"


def test_图像块在BM25里注入数字():
    chunks = chunker.build_image_chunks([
        {"chunk_id": "i1", "corpus": "招股说明书2.pdf", "page": 39, "modality_sub": "diagram",
         "text": "销售部由 4 个部门构成，大客户销售部由 6 个销售处构成。", "structured": {}},
    ])
    index = BM25Index(use_jieba=True)
    index.build(chunks)
    tokens = index.tokens_for(chunks[0])
    assert "4" in tokens and "6" in tokens
    hits = index.search("大客户销售部有几个销售处构成", top_k=5)
    assert hits and hits[0][0] == chunks[0].chunk_id
    assert index.meta[chunks[0].chunk_id]["type"] == "image"


def test_图像块证据权重_弱证据picture被降权():
    """T2 实测提醒的落地：picture（只有 CLIP 零样本标签）必须与 diagram/chart 不同权。"""
    chunks = chunker.build_image_chunks([
        {"chunk_id": "i-d", "corpus": "招股说明书2.pdf", "page": 39, "modality_sub": "diagram", "text": "框图"},
        {"chunk_id": "i-c", "corpus": "招股说明书2.pdf", "page": 72, "modality_sub": "chart", "text": "图表"},
        {"chunk_id": "i-p", "corpus": "招股说明书2.pdf", "page": 1, "modality_sub": "picture", "text": "页眉"},
    ])
    weights = {c.modality_sub: c.evidence_weight for c in chunks}
    assert weights["diagram"] == 1.0 and weights["chart"] == 1.0
    assert weights["picture"] == 0.4, "picture 是弱证据，必须降权，否则会挤掉正文证据"
    assert all("evidence_weight" in c.to_dict() for c in chunks)


def test_向量归一化对单位向量幂等_保证重跑不改索引():
    """T4 实测驱动的修正：单位向量必须原值返回，否则每次重跑都会漂最低位。

    背景（实测）：同一份 chunks 重跑 `build_index.py`，`ids.json` 完全一致（ids_sha1 相同），
    但 `vectors.npy` 的 sha1 每次都变 —— 根因是重复 `l2_normalize` 的浮点舍入。
    该回归锁死「已是单位向量 → 不做除法」，从而保证「重跑不改索引」（字节级可复现）。
    """
    np = pytest.importorskip("numpy")
    from app.core.vector_store import l2_normalize

    vector = np.random.RandomState(20261006).rand(1024).astype(np.float32)
    unit = vector / np.linalg.norm(vector)                      # 已是单位向量
    matrix = unit.reshape(1, -1)
    assert np.array_equal(l2_normalize(matrix), matrix), "单位向量行被改动了最低位（幂等性被破坏）"
    assert np.array_equal(l2_normalize(l2_normalize(matrix)), matrix)
    # 非单位向量仍必须被归一化（语义不放松）
    loose = (unit * np.float32(3.0)).reshape(1, -1)
    fixed = l2_normalize(loose)
    assert abs(float(np.linalg.norm(fixed[0])) - 1.0) < 1e-6
    assert np.array_equal(l2_normalize(fixed), fixed)


def test_图像块降级路径不伪造数值():
    """降级块必须显式标注 extraction_method 与 degraded，且不得凭空造出图表数值。"""
    chunk = chunker.build_image_chunks([{
        "chunk_id": "fallback:招股说明书2.pdf:p72", "corpus": "招股说明书2.pdf", "page": 72,
        "modality_sub": "diagram", "text": "（文本层）2008 年中国IC 市场应用结构与增长 资料来源：赛迪顾问",
        "extraction_method": "text_layer_anchor", "structured": {"degraded": True},
    }])[0]
    assert chunk.extraction_method == "text_layer_anchor"
    assert chunk.structured.get("degraded") is True
    assert "14.0" not in chunk.content, "降级路径不得出现未经算法解出的图表数值"
