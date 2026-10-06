# -*- coding: utf-8 -*-
"""t22 根因探针（只读，不改产品代码）：英文降级题（EN4/EN5/EN6）的取值与逐字核验边界。

回答三个问题：
1. EN4/EN5 取到的**证据块原文**长什么样（是否真为源文本残缺，还是取值边界问题）；
2. `_extract_field_value()` 在这些证据上取到什么值；
3. 用该值构造的**确定性英文框架句**能否通过 `citation.answer_support_check()`（逐字核验）。
"""
import sqlite3
import sys

sys.path.insert(0, r"E:\gao6gongdan\工单3\研发")
from app.core import citation as citation_mod
from app.core.generator import ENGLISH_RATIO_MIN, _extract_field_value  # noqa: E402
from app.core.language import english_frame, english_ratio  # noqa: E402

DB = r"E:\gao6gongdan\工单3\研发\data\index\rag.sqlite3"
TARGETS = [
    ("EN4 注册资本 力源", "招股说明书2", 15, "注册资本"),
    ("EN5 发行股数 力源", "招股说明书2", 2, "发行股数"),
    ("EN6 注册地址 兴图", "招股说明书1", 22, "注册地址"),
    ("EN1 注册资本 兴图", "招股说明书1", 22, "注册资本"),
]

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cols = [str(r[1]) for r in conn.execute("PRAGMA table_info(chunks)")]
print("chunks 列：", cols)
type_col = "chunk_type" if "chunk_type" in cols else ("type" if "type" in cols else "''")
for label, pdf_like, page, keyword in TARGETS:
    rows = conn.execute(
        f"SELECT chunk_id, file_name, page, {type_col} AS chunk_type, content FROM chunks "
        "WHERE file_name LIKE ? AND page = ? AND content LIKE ? ORDER BY chunk_id",
        (f"%{pdf_like}%", page, f"%{keyword}%"),
    ).fetchall()
    print(f"\n===== {label} | 命中 {len(rows)} 块（{pdf_like} p{page} 含「{keyword}」）=====")
    for row in rows[:3]:
        content = str(row["content"] or "")
        value_any, style_any = _extract_field_value(content, keyword, expects_numeric=False)
        value_num, _ = _extract_field_value(content, keyword, expects_numeric=True)
        frame = english_frame(keyword, value_any) if value_any else ""
        check_ok = None
        if frame:
            body = f"{frame} (verbatim source: 「{keyword}：{value_any}」)"
            try:
                res = citation_mod.answer_support_check(body, content)
                check_ok = getattr(res, "supported", res)
            except Exception as exc:                                        # pragma: no cover
                check_ok = f"EXC {type(exc).__name__}: {exc}"
        print(f"  -- {row['chunk_id']} type={row['chunk_type']} len={len(content)}")
        print(f"     RAW     {content[:300]!r}")
        print(f"     VALUE   any={value_any!r}({style_any}) num={value_num!r}")
        print(f"     FRAME   {frame!r} ratio={english_ratio(body) if frame else None} "
              f"(阈值 {ENGLISH_RATIO_MIN}) 逐字核验={check_ok}")
conn.close()
