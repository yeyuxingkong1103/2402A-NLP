"""CLI 的报告渲染。**纯输出，不做任何判断。**

判断（自检、语料一致性）在 `checkup.py`。

---

## 为什么 `search` 必须显示两路各自的原始名次

`render_search` 的表格有 `余弦` / `BM25` / `语义名次` / `关键词名次` / `RRF` /
`来源` 六列。只显示最终排名的话，CLI 就退化成一个"看起来对不对"的工具 ——
而融合规则的**有效性**只能靠回看两路名次来验证（data-model.md §2.1）。

这几列不是调试便利，是 CLI 存在的理由之一。

## 为什么原文全文打印，不做截断

CLI 的用途之一是核对"命中的究竟是不是原文"。截断会让"命中了但看着不对"
变成无法判断 —— 而用户拿着的正是纸质原文，他需要逐字比对。

前端不同：界面上同时有几十条卡片，必须先给预览再展开。终端上一次只有几条，
这个取舍不存在。
"""

from __future__ import annotations

import json

from .bundle import IndexBundle, SearchTrace
from .models import Candidate

__all__ = ["render_search", "render_search_json", "render_corpus"]


def render_search(
    trace: SearchTrace, *, question: str, threshold: float, source_note: str
) -> None:
    """`search` 子命令的人类可读输出。"""

    print("问题: %s" % question)
    print("语料: %s" % source_note)
    print()

    print("语义路   命中 %d 条" % trace.semantic_n)
    print("关键词路  命中 %d 条" % trace.lexical_n)
    print("融合后   %d 条" % len(trace.fused))
    print()

    if not trace.fused:
        print("（两路均无候选）")
    else:
        print(
            "%-3s %-22s %-8s %-8s %-9s %-9s %-8s %-7s %s"
            % (
                "#", "chunk_id", "余弦", "BM25", "语义名次", "关键词名次",
                "覆盖率", "RRF", "来源",
            )
        )
        for rank, cand in enumerate(trace.fused, start=1):
            print(_row(rank, cand))

    print()
    print(_verdict(trace, threshold))

    if trace.result.passages:
        print()
        print("---- 最终返回的 %d 条原文 ----" % len(trace.result.passages))
        for index, passage in enumerate(trace.result.passages, start=1):
            print()
            print(
                "[%d] %s 第 %s 页  余弦 %.4f"
                % (
                    index,
                    passage.file_name,
                    passage.page_start
                    if passage.page_start == passage.page_end
                    else "%d–%d" % (passage.page_start, passage.page_end),
                    passage.score,
                )
            )
            print(passage.text)


def _row(rank: int, cand: Candidate) -> str:
    """一行候选。

    `覆盖率` 一列是**关键词准入的第二个条件**（2026-09-28 裁决）。标定
    `LEXICAL_MIN_COVERAGE` 时看的就是这一列 —— 它同时解释了"这条为什么被
    放行/拦下"，而只看最终排名是看不出来的。
    """

    return "%-3d %-22s %-8s %-8s %-9s %-9s %-8.2f %-7.5f %s" % (
        rank,
        cand.chunk.chunk_id,
        "-" if cand.cosine is None else "%.4f" % cand.cosine,
        "-" if cand.bm25 is None else "%.4f" % cand.bm25,
        "-" if cand.semantic_rank is None else cand.semantic_rank,
        "-" if cand.lexical_rank is None else cand.lexical_rank,
        cand.matched_ratio,
        cand.rrf_score,
        cand.source_label,
    )


def _verdict(trace: SearchTrace, threshold: float) -> str:
    """判定行。

    ⚠️ `below_threshold=True` 在混合检索下**是正确行为，不是故障** ——
    它表示"结果全部靠关键词纳入"，即知识库里有这个词而语义模型没理解它
    （Q4 裁决）。CLI 必须把这层意思说清楚：使用者第一次看到它会以为检索坏了，
    而真相恰恰是关键词路在正确地干活。
    """

    result = trace.result

    if result.is_empty:
        note = (
            "有候选但全部未过阈值（余弦 < %.4f 且关键词名次 > 准入线）" % threshold
            if trace.fused
            else "两路均无候选"
        )
        return "判定: is_empty=True  below_threshold=%s\n       （%s）" % (
            result.below_threshold,
            note,
        )

    note = (
        "结果全部靠关键词纳入 —— 语义模型没能理解这些问题词。"
        "这是**正常结论**，不是故障（Q4 裁决）"
        if result.below_threshold
        else "至少一条余弦 >= 阈值"
    )

    return (
        "判定: is_empty=False  below_threshold=%s\n"
        "       （%s）\n"
        "耗时: %d ms   state=%s"
        % (result.below_threshold, note, trace.elapsed_ms, trace.state)
    )


def render_search_json(trace: SearchTrace) -> None:
    """`--json` 输出。

    ⚠️ **只输出契约模型**（`RetrievalResult`），不夹带内部结构。融合名次在
    人类可读输出里已经能看到；把它塞进 JSON 会让消费方误以为它是契约的一部分，
    而它随时可能随实现调整。
    """

    print(json.dumps(trace.result.model_dump(), ensure_ascii=False, indent=2))


def render_corpus(bundle: IndexBundle, failures: int, lines: list[str]) -> None:
    """`corpus` 子命令的输出。判断已由 `checkup.check_corpus` 做完。"""

    print("collection: %s" % bundle.collection)
    print()
    for line in lines:
        print(line)

    if failures:
        print("\n语料自检未通过（%d 项）。" % failures)
    else:
        print("\n语料自检通过。")
