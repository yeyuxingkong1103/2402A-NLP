"""自检用的构造语料。

与 `selftest_cases.py`（断言）和 `selfcheck.py`（执行器）分开，是因为三者变化的
原因不同：语料在"想测一个新的边界情况"时变，断言在"发现公式理解错了"时变，
执行器几乎不变。

---

## 为什么用构造语料而不是真实数据

依赖真实数据的话，数据一变这些断言就跟着失去意义 —— 而它们要验证的是**公式**。
`SEED` 是四条手工写死的语料，全文可见、可推演、不随库内容变化。

四条语料刻意覆盖了三类关系：主题词重复（高血压出现三次）、罕见词只出现在一条
（盐酸氨氯地平、糖尿病各一次）、以及一条与查询词完全无关的对照。
"""

from __future__ import annotations

from .models import CorpusChunk

__all__ = ["SEED", "seed_corpus", "seed_map"]


SEED: tuple[str, ...] = (
    "高血压患者的血压控制目标",
    "盐酸氨氯地平片用于高血压治疗",
    "高血压合并糖尿病的用药建议",
    "糖尿病患者应定期监测血糖",
)


def seed_corpus() -> list[CorpusChunk]:
    return [
        CorpusChunk(
            chunk_id="t:%04d" % i,
            text=text,
            file_name="t.pdf",
            page_start=1,
            page_end=1,
            block_type="text",
        )
        for i, text in enumerate(SEED)
    ]


def seed_map() -> dict[str, CorpusChunk]:
    return {c.chunk_id: c for c in seed_corpus()}


