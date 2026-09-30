"""ingest.py 单元测试：normalize_whitespace / split_chunks / clean_chunks / make_summary。"""
import ingest


# ---- normalize_whitespace ----

def test_normalize_collapses_horizontal_whitespace():
    """多个空格与制表符压缩为单个空格。"""
    assert ingest.normalize_whitespace("a  b\t\tc") == "a b c"


def test_normalize_merges_blank_lines():
    """连续换行（含空白行）合并为单个换行。"""
    assert ingest.normalize_whitespace("a\n\n\nb") == "a\nb"


def test_normalize_replaces_fullwidth_space():
    """全角空格替换为半角空格。"""
    assert ingest.normalize_whitespace("a　b") == "a b"


def test_normalize_strips_edges():
    """去掉文本首尾的空白（含换行）。"""
    assert ingest.normalize_whitespace("  \n内容\n  ") == "内容"


def test_normalize_keeps_clean_text():
    """已规范文本保持不变。"""
    assert ingest.normalize_whitespace("正常的 一句话") == "正常的 一句话"


# ---- split_chunks ----

def test_split_chunks_normal_count_and_size():
    """6×100 字句 → 2 块（400/200 字），每块 ≤ CHUNK_SIZE。"""
    text = "\n".join(["字" * 100] * 6)
    chunks = ingest.split_chunks(text, 1)
    assert len(chunks) == 2
    assert [len(c["content"]) for c in chunks] == [400, 200]
    assert all(len(c["content"]) <= ingest.CHUNK_SIZE for c in chunks)


def test_split_chunks_ids_and_page():
    """id 为 p{page}_c{序号}，page 字段与入参一致。"""
    text = "\n".join(["字" * 300] * 2)
    chunks = ingest.split_chunks(text, 3)
    assert [c["id"] for c in chunks] == ["p3_c0", "p3_c1"]
    assert all(c["page"] == 3 for c in chunks)


def test_split_chunks_long_sentence_hard_cut():
    """超长单句按 step=340 硬切为 3 块，每块 ≤ CHUNK_SIZE。"""
    chunks = ingest.split_chunks("字" * 1000, 1)
    assert [len(c["content"]) for c in chunks] == [400, 400, 320]
    assert all(len(c["content"]) <= ingest.CHUNK_SIZE for c in chunks)


def test_split_chunks_long_sentence_overlap():
    """硬切相邻块重叠 CHUNK_OVERLAP 字符。"""
    chunks = ingest.split_chunks("字" * 1000, 1)
    assert chunks[0]["content"][340:] == chunks[1]["content"][:60]
    assert chunks[1]["content"][340:] == chunks[2]["content"][:60]


def test_split_chunks_empty_returns_none():
    """空串或纯空白不产生 chunk。"""
    assert ingest.split_chunks("", 1) == []
    assert ingest.split_chunks("   \n  ", 1) == []


# ---- split_parents ----

def test_split_parents_accumulates_until_size():
    """段落累加到接近 size 就切一块（分隔符也计入长度）。"""
    text = "\n\n".join(["字" * 800] * 3)
    parents = ingest.split_parents(text, size=1500)
    # 800+2+800=1602>1500 切，800+2+800=1602>1500 再切，剩 800
    assert [len(p) for p in parents] == [800, 800, 800]


def test_split_parents_counts_separator():
    """两段恰好 1500（含分隔符）合并为一个父块，不超 size。"""
    text = "\n\n".join(["字" * 800, "字" * 698])  # 800 + 2 + 698 = 1500
    parents = ingest.split_parents(text, size=1500)
    assert [len(p) for p in parents] == [1500]


def test_split_parents_long_para_hard_cut():
    """超长单段（>size）按 800 字硬切。"""
    parents = ingest.split_parents("字" * 2000, size=1500)
    assert [len(p) for p in parents] == [800, 800, 400]


def test_split_parents_empty():
    """空串或纯空白不产生父块。"""
    assert ingest.split_parents("") == []
    assert ingest.split_parents("   \n\n  ") == []


def test_split_parents_newline_fallback():
    """无空行时按单 \\n 行累加，仍切出合理父块（parser 输出的平铺行）。"""
    text = "\n".join(["字" * 800] * 3)
    parents = ingest.split_parents(text, size=1500)
    assert [len(p) for p in parents] == [800, 800, 800]


def test_split_parents_single_short_para():
    """≤ size 的单段作为一个父块，不切。"""
    assert ingest.split_parents("字" * 500, size=1500) == ["字" * 500]


# ---- clean_chunks / make_summary ----

def _chunk(content, cid="c0", page=1):
    return {"id": cid, "content": content, "page": page}


def test_clean_chunks_dedups_by_md5():
    """相同 content 的 chunk 只保留首个。"""
    chunks = [_chunk("内容甲" * 10), _chunk("内容乙" * 10), _chunk("内容甲" * 10)]
    cleaned, stats = ingest.clean_chunks(chunks)
    assert [c["content"] for c in cleaned] == ["内容甲" * 10, "内容乙" * 10]
    assert stats["before"] == 3 and stats["after"] == 2 and stats["dedup_removed"] == 1


def test_clean_chunks_drops_too_short():
    """长度 < 20 的 chunk 丢弃。"""
    cleaned, stats = ingest.clean_chunks([_chunk("短"), _chunk("足够长的正文" * 5)])
    assert [c["content"] for c in cleaned] == ["足够长的正文" * 5]
    assert stats["low_quality_removed"] == 1


def test_clean_chunks_drops_pure_symbols():
    """纯符号（无 CJK/字母/数字）丢弃。"""
    cleaned, stats = ingest.clean_chunks([_chunk("！？——……" * 5), _chunk("正常内容" * 5)])
    assert [c["content"] for c in cleaned] == ["正常内容" * 5]
    assert stats["low_quality_removed"] == 1


def test_clean_chunks_drops_garbled():
    """含替换符的乱码丢弃。"""
    cleaned, _ = ingest.clean_chunks([_chunk("乱码�文本" * 10), _chunk("正常内容" * 5)])
    assert [c["content"] for c in cleaned] == ["正常内容" * 5]


def test_make_summary_takes_first_100_chars():
    assert ingest.make_summary("字" * 200) == "字" * 100
    assert ingest.make_summary("短") == "短"
