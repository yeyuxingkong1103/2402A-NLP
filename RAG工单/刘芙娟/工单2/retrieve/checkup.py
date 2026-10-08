"""语料一致性检查：`corpus` 子命令的判断逻辑。

与 `report.py`（纯输出）分开：本模块是**判断**，那个是**呈现**。
与 `selfcheck.py`（纯函数断言）分开：那个什么前置都不要，本模块需要 Milvus 起着。

---

## 为什么要检查这三件事

一份**被截断的**或**与清单不符的**语料看起来完全正常：BM25 照样返回结果，
只是少了些文档；语义路完全不受影响（向量检索由 Milvus 自己管）。两路从此
不一致，而没有任何报错。

下面三项是这件事**唯一**能被观测到的地方。
"""

from __future__ import annotations

from . import QUERY_LIMIT, REQUIRED_CORPUS_FIELDS
from .bundle import IndexBundle

__all__ = ["check_corpus"]


def check_corpus(bundle: IndexBundle) -> tuple[int, list[str]]:
    """返回 `(失败项数, 输出行)`。

    三件事任一不符即失败（contracts/cli.md §2）：

    1. 拉回条数 == `index_manifest.json` 的 `total_chunks` —— 不符说明库被改过
       而清单未更新，或反之。
    2. 拉回条数 **<** `QUERY_LIMIT` —— 相等则可能被截断（research R3）。
    3. `chunk_id` 唯一 —— 不唯一会让"去重"这一步失去意义。

    另有四项**字段完整性**检查，它们是 FR-004 / constitution 原则 II 的前提：
    引用的 `file_name` / `page_start` / `page_end` 缺失时，"可回原文核对"就落空了。
    """

    failures = 0
    lines: list[str] = []

    size = bundle.size
    matched = bundle.matches_manifest

    if matched is None:
        lines.append("manifest total_chunks: 无法比对（清单缺失或格式不符）")
        failures += 1
    elif not matched:
        lines.append(
            "manifest total_chunks: %s           [不一致] ← 库被改过？"
            % bundle.manifest_total
        )
        failures += 1
    else:
        lines.append("manifest total_chunks: %s           [一致]" % bundle.manifest_total)

    if size >= QUERY_LIMIT:
        lines.append(
            "拉回条数 %d 已达 QUERY_LIMIT 上限           [可能被截断] ← 语料不完整"
            % size
        )
        failures += 1

    unique = len(bundle.chunks_by_id)
    if unique != size:
        lines.append(
            "去重后 chunk_id: %d（原始 %d）           [chunk_id 不唯一]"
            % (unique, size)
        )
        failures += 1
    else:
        lines.append("去重后 chunk_id: %d           [一致]" % unique)

    lines.insert(0, "语料条数: %d" % size)
    lines.append("平均文档长度(tokens): %.1f" % bundle.lexical.avgdl)

    for name in REQUIRED_CORPUS_FIELDS:
        empty = sum(1 for c in bundle.chunks if not getattr(c, name))
        lines.append("%s 为空: %d" % (name, empty))
        if empty:
            failures += 1

    bad_pages = sum(
        1 for c in bundle.chunks if c.page_start > c.page_end or c.page_start < 1
    )
    lines.append("page_start > page_end 或 < 1: %d" % bad_pages)
    if bad_pages:
        failures += 1

    lines.append("高频词 top10: %s" % "、".join(_top_terms(bundle, 10)))
    return failures, lines


def _top_terms(bundle: IndexBundle, limit: int) -> list[str]:
    """按文档频次取前 N 个词。

    这不是装饰：US1 的验证需要一个**确认出现在原文里**的术语，而这里是
    找到它的最快途径（`quickstart.md` §3.2 的第一步）。
    """

    ranked = sorted(bundle.lexical.df.items(), key=lambda kv: (-kv[1], kv[0]))
    return ["%s(%d)" % (term, df) for term, df in ranked[:limit]]
