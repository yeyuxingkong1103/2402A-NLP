"""生成结果的模型、引用校验与派生字段。

拆出来的理由：`client.py` 管**出网**，本模块管**模型说了什么、以及它说的能采信多少**
—— 两者的失效方式完全不同（网络超时 vs 引用了不存在的出处），混在一起时
排查要先分辨是哪一类。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from backend.api import DISCLAIMER

from . import DEFAULT_MODE, DEFAULT_MODEL

__all__ = [
    "GenerationResult",
    "parse_answer",
    "build_sources",
    "confidence_of",
    "INSUFFICIENT_MARKER",
]

# 提示词要求模型在"资料不足"时独占一行输出的哨兵。
#
# ⚠️ 与 `backend/api/__init__.py` 的 `REFUSAL_FALLBACK` **不是同一个东西**：
#    那是给用户看的固定话术（由装配层拼），这只是一个**控制信号**，
#    见到它就把整条生成结果判为失败、转走拒答路径 —— 它本身不展示给用户。
INSUFFICIENT_MARKER = "【资料不足】"

# 正文里的引用标记。只匹配「【数字】」，不匹配 `【1,2】` 之类的变体 ——
# 提示词明确要求一句只标一个编号，变体一律按"模型没照做"处理（当成普通文本留着）。
_CITATION_RE = re.compile(r"【(\d+)】")


@dataclass
class GenerationResult:
    """`docs/05` §4.3 的 `GenerationResult`，四个契约字段逐一致。

    契约字段之外的部分只服务于**落盘与展示**，不改变契约。
    """

    answer_text: str
    used_citation_ids: list[int]
    is_citation_failure: bool
    raw_model_output: str

    # ---- 契约之外 ----
    answer_id: str = ""
    question: str = ""
    mode: str = DEFAULT_MODE
    sources: list[dict] = field(default_factory=list)
    confidence: float | None = None
    model: str = DEFAULT_MODEL

    @property
    def disclaimer(self) -> str:
        """免责声明。**引用装配层的那一份，不在这里重写。**

        `docs/05` §4.4 要求全仓检索那两句逐字文本只应命中 `assembly_service.build`。
        本属性只是把它**读出来放进返回值**，不构成第二处定义 —— 因此不会让
        「只应命中一处」的检索多出命中。
        """

        return DISCLAIMER

    def to_record(self) -> dict:
        """落盘记录。字段顺序固定，便于 grep。"""

        return {
            "answer_id": self.answer_id,
            "question": self.question,
            "mode": self.mode,
            "answered_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
            "model": self.model,
            "status": "failed" if self.is_citation_failure else "ok",
            "answer_text": self.answer_text,
            "sources": self.sources,
            "disclaimer": self.disclaimer,
            "confidence": self.confidence,
            "used_citation_ids": self.used_citation_ids,
            "is_citation_failure": self.is_citation_failure,
        }


def parse_answer(raw: str, passages: list) -> tuple[str, list[int], bool]:
    """引用白名单校验（`docs/05` §3.1.7）。

    返回 `(已剥离无效标记的正文, 有效引用编号, 是否引用失败)`。

    三件事：

    1. **抽取**正文里的 `【n】`，只保留 `1 ≤ n ≤ len(passages)` 的。
    2. **剥离**无效标记 —— `docs/05` §4.3 要求 `answer_text` 是"已剥离无效标记的
       正文"。留着它们会让用户看到一个指向不存在出处的编号，而用户点不到它，
       只能认为界面坏了。
    3. `is_citation_failure` = 正文里一个有效标记都没有 —— 调用方**MUST**据此
       转拒答，不得把这段正文当作有依据的回答展示（§4.3 的约束）。
    """

    valid_max = len(passages)

    # 去重但保留首次出现的顺序 —— 正文里的引用顺序是有意义的。
    used: list[int] = []
    for token in _CITATION_RE.findall(raw):
        number = int(token)
        if 1 <= number <= valid_max and number not in used:
            used.append(number)

    stripped = _CITATION_RE.sub(
        lambda m: m.group(0) if 1 <= int(m.group(1)) <= valid_max else "", raw
    ).strip()

    insufficient = INSUFFICIENT_MARKER in stripped
    if insufficient:
        # 哨兵是给调用方看的控制信号，不该出现在用户看到的正文里。
        stripped = stripped.replace(INSUFFICIENT_MARKER, "").strip()

    return stripped, used, insufficient or not used


def build_sources(passages: list, used_ids: list[int]) -> list[dict]:
    """来源列表。

    ⚠️ **来自检索片段，不是来自模型输出。**

    让模型吐一份来源清单会引入两个问题：它与正文里的 `【n】` 标记可能不一致，
    且给了模型一个编造来源元数据的机会。这里按**实际被引用的片段**生成，字段
    全部取自库里真实存在的数据，因此不可能与正文矛盾。
    """

    return [
        {
            "citation_id": index,
            "file_name": passages[index - 1].file_name,
            "page_start": passages[index - 1].page_start,
            "page_end": passages[index - 1].page_end,
            "section": passages[index - 1].section,
        }
        for index in used_ids
        if 1 <= index <= len(passages)
    ]


def confidence_of(passages: list) -> float | None:
    """置信度 = 首条引用的余弦相似度。

    ⚠️ **它不是概率，不要当概率用。** 它只是"检索到的原文与问题有多接近"这个
    已经算出来的数字（`docs/05` §4.2 的 `score`），在这里换个名字透出。

    刻意**不**让模型自报置信度：模型对自己置信度的报告与它的实际准确率几乎不相关，
    而这个数至少是真实测量出来的。
    """

    for passage in passages:
        score = getattr(passage, "score", None)
        if isinstance(score, (int, float)):
            return float(score)
    return None
