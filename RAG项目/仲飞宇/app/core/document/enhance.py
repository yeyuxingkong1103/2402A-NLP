"""入库侧的数据增强 / 去重 / 低质量过滤。

供离线 CLI（scripts/ingest.py）与上传接口（app/api/knowledge.py）共用，避免两边各写一份。
对应需求里的「知识库数据增强（摘要）」「知识库优化（去重、删除低质量文档）」。

链路上的位置：chunk 切好之后、写库之前。低质量过滤由调用方先做，然后
ingest.build_chunks 会调这里的 summarize_chunks 拿摘要、并用 chunk_hash 做同批去重。
"""
from __future__ import annotations

import hashlib

from ..llm import LLMClient
from ..logging_config import get_logger

log = get_logger("ingest.enhance")

# 30 字上限与「只输出摘要本身」都是刻意的：摘要写进 Milvus 的 summary 字段
# （VARCHAR 2048），当前召回只 output text/title/source，所以它既不参与检索也不进 prompt，
# 是入库侧的增强留档。限长只是防止它膨胀成第二份正文——真要长摘要应当先改检索那一侧。
# 「只输出摘要本身」是挡掉模型习惯加的前言（"好的，以下是摘要："），那会原样入库。
_SUMMARY_SYSTEM = "你是摘要器。用一句不超过 30 字的中文概括用户给的文本要点，只输出摘要本身。"


def chunk_hash(text: str) -> str:
    """内容去重的稳定指纹：空白归一化后取 md5。

    先做空白归一化，是因为同一段内容在不同来源下空白不同（PDF 抽出的换行、手打的多空格），
    不归一就等于两份不同指纹、去重直接失效（tests/test_enhance.py 盯着这条）。
    md5 在这里只当快速内容指纹用，不涉及安全，碰撞概率对去重场景足够低。
    注意它的作用域是**同一批**（build_chunks 内的 seen 集合）；跨批次的去重靠关系库登记的
    source（见 store_chunks）。
    """
    return hashlib.md5(" ".join(text.split()).encode("utf-8")).hexdigest()


def filter_low_quality(texts: list[str], min_chars: int) -> list[str]:
    """低质量过滤：丢弃过短（无效）的 chunk。

    判据只有字符数（按 strip 后的长度），不看语义——OCR 抽出的零碎字符、页眉残渣都是
    靠这一步挡掉的。min_chars 由调用方传 settings.min_chunk_chars（默认 10）。
    这一步在上游（上传接口 / scripts/ingest.py）做，**不在 build_chunks 里**：所以直接
    调 build_chunks 的调用方（scripts/seed.py）不受这个阈值约束。
    """
    return [t for t in texts if len(t.strip()) >= min_chars]


def summarize_chunks(llm: LLMClient, texts: list[str]) -> list[str]:
    """给每个 chunk 生成一句摘要（知识库数据增强），LLM 失败则该 chunk 摘要留空。

    返回列表与入参**等长且按位对齐**：build_chunks 用下标取摘要，少一条摘要就会整体错位。
    逐条串行调用（没有批量/并发）：单条失败只影响该条，不会连累整批；代价是逐条一次
    LLM 往返，summary=True 时入库会明显变慢（所以 SUMMARY_ENABLED 默认关）。
    注意 llm=None 时不会抛给调用方——`llm.chat` 的 AttributeError 也被下面吞掉，
    表现是每个 chunk 一条 warning、摘要全空。要摘要就必须给 llm。
    """
    out: list[str] = []
    for t in texts:
        try:
            s = llm.chat(
                [
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user", "content": t},
                ]
            ).strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("摘要生成失败，留空: %s", exc)
            s = ""
        out.append(s)
    return out
