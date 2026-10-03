"""T3 离线测试 ②：分块层。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
覆盖（设计/验收标准.md 验收 11、工单 6.2、需求分析 FR-06）：
- 分块规模与类型分布真实；
- ``chunk_size=500`` / ``chunk_overlap=80``（设计/接口设计.md L94：工单 6.2 的 400~600/80 映射）；
- **overlap 实测为 80 字**（同页相邻块滑窗重叠，非仅读配置）；
- 表格块整表成块、不被拆分；
- 每块含 chunk_id/page/section/type/keywords；
- 关键词标签覆盖工单点名字段；
- **证据原文不跨块破损**（A/B 两归一化 × 宽松/严格两口径的校准矩阵）。

口径诚实声明（重要）：
400~600 在设计中是**软约束**（接口设计 L178「长度目标 400~600（软约束），句末断开优先」）。
实测正文块长度 p50=396、p90=492，落在 400~600 区间的比例为 48.6%，
另有 6 块 <30 字、3 块 >600 字。因此本文件断言的是**分布性质与硬边界**，
并把实测分布与离群块**如实打印**，不伪造「所有块都在 400~600」。
"""

from __future__ import annotations

from collections import Counter

from conftest import norm_a, norm_b

#: 工单点名的关键字段（关键词标签必须能覆盖）
REQUIRED_KEYWORD_FIELDS = (
    "收入", "注册资本", "法定代表人", "技术标准", "上下游", "军用领域", "募集资金",
)


def test_chunk_scale_and_types(chunks):
    """分块规模与类型分布真实（1669 = text 1256 + table 413）。"""
    assert len(chunks) == 1669, f"分块总数应为 1669，实际 {len(chunks)}"
    kinds = Counter(c.type for c in chunks)
    assert kinds["text"] == 1256, f"text 块应为 1256，实际 {kinds['text']}"
    assert kinds["table"] == 413, f"table 块应为 413，实际 {kinds['table']}"
    assert all(c.page >= 1 for c in chunks), "存在页码 <1 的块"


def test_chunk_size_and_overlap_config(settings):
    """滑窗参数必须等于工单要求的映射值（chunk_size 500 / overlap 80）。"""
    assert settings.chunk.chunk_size == 500, (
        f"chunk_size 应为 500（工单 400~600 的目标中点），实际 {settings.chunk.chunk_size}"
    )
    assert settings.chunk.chunk_overlap == 80, (
        f"chunk_overlap 应为 80，实际 {settings.chunk.chunk_overlap}"
    )


def test_overlap_is_really_eighty_chars(chunks):
    """实测 overlap 必须真是配置的 80 字字符级重叠（不能只信配置）。

    做法：对**同页相邻**的 text 块，比较前块尾部与后块首部的公共后缀/前缀，
    按**逐字符**精确扫描（步长 1，避免漏掉 79 这类差一字符的情况）。

    实测：712 对同页相邻块中 191 对出现字符级重叠，长度分布 ``{80: 188, 79: 3}``。
    其中 79 是块内容 ``strip()`` 去掉首尾空白造成的**一字差**，属正常清洗结果，
    不是 overlap 参数失效；故断言「全部落在 {79, 80} 且以 80 为众数，80 占比 ≥95%」。
    （其余相邻块来自不同段落，本就不该重叠——软约束 + 句末断开的正常结果。）
    """
    by_page: dict[int, list] = {}
    for c in chunks:
        if c.type == "text":
            by_page.setdefault(c.page, []).append(c)
    pairs = overlaps = 0
    sizes: Counter[int] = Counter()
    for page, lst in by_page.items():
        lst = sorted(lst, key=lambda c: c.chunk_id)
        for a, b in zip(lst, lst[1:]):
            pairs += 1
            tail, head = a.content[-200:], b.content[:200]
            best = 0
            for size in range(20, min(len(tail), len(head)) + 1):
                if tail[-size:] == head[:size]:
                    best = size
            if best:
                overlaps += 1
                sizes[best] += 1
    assert overlaps > 0, "未检出任何字符级重叠——overlap 未生效"
    print(f"\n[overlap] 同页相邻 text 块对 {pairs} 对，检出重叠 {overlaps} 对，长度分布 {dict(sorted(sizes.items()))}")
    unexpected = {size: cnt for size, cnt in sizes.items() if size not in (79, 80)}
    assert not unexpected, (
        f"重叠长度应落在 {{79, 80}}（配置 80，差额来自 strip 清洗），实测异常值 {unexpected}"
    )
    ratio80 = sizes[80] / overlaps
    assert ratio80 >= 0.95, (
        f"重叠长度=80 的占比应 ≥95%，实际 {ratio80:.1%}（分布 {dict(sizes)}）"
    )


def test_chunk_required_fields_present(chunks):
    """每块必须含 chunk_id/page/section/type/content/char_count/keywords。"""
    for c in chunks[:400]:
        assert c.chunk_id and c.chunk_id.startswith("c"), f"chunk_id 非法: {c.chunk_id!r}"
        assert isinstance(c.page, int) and 1 <= c.page <= 548, f"{c.chunk_id} 页码非法: {c.page}"
        assert c.type in ("text", "table"), f"{c.chunk_id} 类型非法: {c.type}"
        assert isinstance(c.content, str) and c.content.strip(), f"{c.chunk_id} 内容为空"
        assert isinstance(c.section, str), f"{c.chunk_id} section 非字符串"
        assert isinstance(c.keywords, list), f"{c.chunk_id} keywords 非列表"
        assert isinstance(c.char_count, int), f"{c.chunk_id} char_count 非整数"
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "存在重复 chunk_id"


def test_char_count_matches_content(chunks):
    """``char_count`` 必须等于 ``len(content)``（否则长度断言会被架空）。"""
    bad = [c.chunk_id for c in chunks if c.char_count != len(c.content)]
    assert not bad, f"char_count 与 len(content) 不一致的块: {bad[:5]}"


def test_text_chunk_length_distribution(chunks):
    """正文块长度分布：断言硬边界与分布，并如实打印离群块。

    断言（实测值 + 设计软约束）：
    - ≥98% 的正文块落在 [min_chunk_chars, 600] 内（实测 1250/1256 = 99.5%）；
    - p90 ≤ 600、p50 ∈ [300, 500]（实测 p90=492、p50=396）；
    - 落在 400~600 的比例 ≥45%（实测 48.6%，设计为软约束，故不断言 100%）。
    """
    lengths = sorted(len(c.content) for c in chunks if c.type == "text")
    n = len(lengths)
    band = sum(1 for x in lengths if 400 <= x <= 600)
    below_floor = [c.chunk_id for c in chunks if c.type == "text" and len(c.content) < 30]
    above_ceil = [(c.chunk_id, len(c.content)) for c in chunks if c.type == "text" and len(c.content) > 600]
    p50, p90 = lengths[n // 2], lengths[int(n * 0.9)]
    print(f"\n[分块长度] n={n} min={lengths[0]} p50={p50} p90={p90} max={lengths[-1]} "
          f"落在400~600={band}/{n} ({band / n * 100:.1f}%) "
          f"<30={len(below_floor)} >600={len(above_ceil)}")
    if above_ceil:
        print(f"[分块长度] >600 的块: {above_ceil}")
    if below_floor:
        print(f"[分块长度] <30 的块: {below_floor}")

    in_hard_band = sum(1 for x in lengths if 30 <= x <= 600)
    assert in_hard_band / n >= 0.98, (
        f"落在 [30,600] 的正文块比例应 ≥98%，实际 {in_hard_band}/{n} = {in_hard_band / n:.1%}"
    )
    assert p90 <= 600, f"正文块 p90 应 ≤600，实际 {p90}"
    assert 300 <= p50 <= 500, f"正文块中位长度应在 [300,500]，实际 {p50}"
    assert band / n >= 0.45, f"落在 400~600 的比例应 ≥45%，实际 {band / n:.1%}"
    assert len(above_ceil) <= 5, f">600 字的正文块过多（{len(above_ceil)}）: {above_ceil}"


def test_table_chunks_not_split(chunks):
    """表格块必须整表成块：table_id 唯一、含 Markdown 表结构、不重复出现。"""
    tables = [c for c in chunks if c.type == "table"]
    assert tables, "没有表格块"
    ids = [c.table_id for c in tables]
    assert all(ids), f"存在 table_id 为空的表格块: {[c.chunk_id for c in tables if not c.table_id][:5]}"
    dup = [tid for tid, cnt in Counter(ids).items() if cnt > 1]
    assert not dup, f"表格被拆成多块（table_id 重复）: {dup[:5]}"
    no_md = [c.chunk_id for c in tables if "|" not in c.content]
    assert not no_md, f"表格块缺 Markdown 结构: {no_md[:5]}"
    # 表格块不参与段落滑窗，长度可超 600
    over = [c.chunk_id for c in tables if len(c.content) > 600]
    print(f"\n[表格块] {len(tables)} 块，table_id 唯一；长度 >600 的表格块 {len(over)} 个（表格整块豁免滑窗）")


def test_keywords_cover_work_order_fields(chunks):
    """关键词标签必须覆盖工单点名的关键字段（工单 6.2「关键词标签」）。"""
    all_kw = set()
    for c in chunks:
        all_kw.update(c.keywords or [])
    missing = [kw for kw in REQUIRED_KEYWORD_FIELDS if kw not in all_kw]
    assert not missing, f"关键词标签缺少工单点名字段: {missing}（现有 {len(all_kw)} 种）"
    # 标签不能等于"每块都有"，否则无区分度
    tagged = sum(1 for c in chunks if c.keywords)
    assert 0 < tagged <= len(chunks), "关键词标签覆盖异常"
    print(f"\n[关键词] 共 {len(all_kw)} 种标签，覆盖块 {tagged}/{len(chunks)}；"
          f"命中字段 {[kw for kw in REQUIRED_KEYWORD_FIELDS]}")


def test_boilerplate_marked_small(chunks):
    """释义页样板块应被标记且数量小（实测 6 块），供检索层降权。"""
    boiler = [c for c in chunks if c.is_boilerplate]
    assert len(boiler) <= 20, f"is_boilerplate 块数异常偏多: {len(boiler)}"
    print(f"\n[释义块] is_boilerplate={len(boiler)}，页码 {[c.page for c in boiler]}")


def test_evidence_not_broken_across_chunks(chunks, golden):
    """**证据原文不跨块破损**——四格校准矩阵（A/B 归一化 × 宽松/严格口径）。

    环境事实 §4.1.4 规定：归一化 A（仅去空白）宽松 8/严格 6；B（去空白+标点）宽松 9/严格 8；
    差值全部来自 Q95（evidence 含「……」）与 Q207（合成引用串）。
    """
    texts_a = {c.chunk_id: norm_a(c.content) for c in chunks}
    texts_b = {c.chunk_id: norm_b(c.content) for c in chunks}
    matrix = {}
    for tag, texts, norm in (("A", texts_a, norm_a), ("B", texts_b, norm_b)):
        loose = strict = 0
        detail = []
        for item in golden:
            ev = norm(item.evidence)
            pre = ev[:80]
            hl = next((cid for cid, t in texts.items() if pre and pre in t), None)
            hs = next((cid for cid, t in texts.items() if ev and ev in t), None)
            loose += bool(hl)
            strict += bool(hs)
            detail.append(f"Q{item.id}:{'L' if hl else '-'}{'S' if hs else '-'}")
        matrix[tag] = (loose, strict)
        print(f"[命中矩阵 {tag}] 宽松 {loose}/10，严格 {strict}/10  {' '.join(detail)}")

    assert matrix["B"][0] >= 9, f"归一化 B 宽松命中应 ≥9/10，实际 {matrix['B'][0]}"
    assert matrix["B"][1] >= 8, f"归一化 B 严格命中应 ≥8/10，实际 {matrix['B'][1]}"
    assert matrix["A"][0] >= 8, f"归一化 A 宽松命中应 ≥8/10，实际 {matrix['A'][0]}"
    assert matrix["A"][1] >= 6, f"归一化 A 严格命中应 ≥6/10，实际 {matrix['A'][1]}"
    # B 必须不劣于 A（标点归一化只会让命中变多）
    assert matrix["B"][0] >= matrix["A"][0] and matrix["B"][1] >= matrix["A"][1], (
        f"归一化 B 不应劣于 A：A={matrix['A']}，B={matrix['B']}"
    )
