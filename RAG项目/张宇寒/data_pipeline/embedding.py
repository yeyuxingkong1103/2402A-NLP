"""把文本分块转换为向量，并在入库前检查向量是否有效。

本文件不直接连接 Milvus。它只负责：选择每条记录中要嵌入的文字、控制模型批次、
调用 BGE-M3 等嵌入模型，以及检查返回数量、维度、NaN 和无穷大。
"""

# 延迟解析类型注解，减少运行时类型依赖。
from __future__ import annotations

# time 只用于模型调用失败后的可选等待重试。
import time
# Iterable 表示 embedding_batches 返回一个可以逐批遍历的生成器。
from collections.abc import Iterable
# isfinite 用来排除向量中的 NaN 和正负无穷大。
from math import isfinite
# Real 表示整数、浮点数等真实数值类型。
from numbers import Real


def embedding_text(row: dict) -> str:
    """按优先级找出一条记录真正用于生成向量的文本。"""

    # embedding_text 是专门准备的检索文本；没有时依次兼容通用正文和摘要字段。
    # 所有字段都没有内容时返回空字符串，保证模型输入类型始终为 str。
    return str(row.get("embedding_text") or row.get("text") or row.get("content") or row.get("summary") or row.get("case_summary") or "")


def embedding_batches(rows: list[dict], batch_size: int, max_characters: int | None = None) -> Iterable[list[dict]]:
    """按记录数和总字符数把数据分成多个模型请求批次。"""

    # batch 保存当前尚未提交给模型的记录。
    batch: list[dict] = []
    # characters 保存当前批次全部嵌入文本的字符总数。
    characters = 0
    # 逐条考虑记录应放入当前批次还是新批次。
    for row in rows:
        # 只统计真正送给嵌入模型的文字长度。
        length = len(embedding_text(row))
        # 当前批次非空，并达到记录数上限或加入本条后超过字符上限时，先提交旧批次。
        if batch and (len(batch) >= batch_size or bool(max_characters and characters + length > max_characters)):
            # yield 返回一批后暂停函数，下次迭代会从这里继续。
            yield batch
            # 开始新批次，同时把计数归零。
            batch, characters = [], 0
        # 把当前记录加入批次。
        batch.append(row)
        # 更新本批字符数。
        characters += length
    # 循环结束后可能还有不足一整批的记录，也必须提交。
    if batch:
        yield batch


def embed(rows: list[dict], model, *, batch_size: int = 32, max_characters: int | None = None, max_batch_chars: int | None = None, expected_dimension: int | None = None, max_retries: int = 1, retry_delay: float = 0.0) -> list[dict]:
    """批量生成向量并返回附带 ``embedding`` 字段的新记录列表。

    ``expected_dimension`` 正式运行时应等于数据库向量维度；``max_retries`` 包含
    第一次调用，例如值为 2 表示最多调用两次。
    """

    # 测试或只处理文本时可以不传模型；此时复制记录但不增加向量。
    if model is None:
        return [dict(row) for row in rows]
    # 两个参数是新旧名称兼容入口；同时指定不同值会产生歧义，因此报错。
    if max_characters is not None and max_batch_chars is not None and max_characters != max_batch_chars:
        raise ValueError("max_characters 与 max_batch_chars 不能同时指定不同值")
    # 批次大小和尝试次数不能为零或负数，否则循环无法正确执行。
    if batch_size <= 0 or max_retries <= 0:
        raise ValueError("batch_size 和 max_retries 必须大于 0")
    # 优先使用新名称 max_characters，没有时兼容旧名称 max_batch_chars。
    character_limit = max_characters if max_characters is not None else max_batch_chars
    # output 收集所有带向量的记录。
    output: list[dict] = []
    # 按限制逐批调用模型，避免一次输入过多导致超时或超过接口限制。
    for batch in embedding_batches(rows, batch_size, character_limit):
        # None 表示当前批次还没有得到成功的模型返回。
        vectors = None
        # attempt 从 0 开始，最多执行 max_retries 次。
        for attempt in range(max_retries):
            try:
                # 提取本批文本列表并调用模型；返回顺序必须与输入顺序一致。
                vectors = model.embed([embedding_text(row) for row in batch])
                # 调用成功就退出重试循环。
                break
            except Exception:
                # 最后一次仍失败就保留原异常交给上层，不能伪造空向量继续入库。
                if attempt + 1 == max_retries:
                    raise
                # 配置了等待时间才暂停；0 表示立即重试。
                if retry_delay > 0:
                    time.sleep(retry_delay)
        # 一条输入必须对应一条向量，数量不一致会造成记录和向量错位。
        if vectors is None or len(vectors) != len(batch):
            raise ValueError("向量数量与记录数量不一致")
        # strict=True 再次保证 zip 不会静默截断任意一边。
        for row, vector in zip(batch, vectors, strict=True):
            # 向量必须是非空列表或元组。
            if not isinstance(vector, (list, tuple)) or not vector:
                raise ValueError("向量必须是非空的数字列表")
            # 正式入库时，模型向量维度必须与 Milvus schema 完全相同。
            if expected_dimension is not None and len(vector) != expected_dimension:
                raise ValueError(f"向量维度错误：应为 {expected_dimension}，实际为 {len(vector)}")
            # bool 虽是 int 子类但不是合理向量值；同时拒绝非数字、NaN 和无穷大。
            if any(isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) for value in vector):
                raise ValueError("向量含非数字、NaN或无穷大")
            # 复制原记录并增加 embedding，避免直接修改调用者传入的字典。
            output.append({**dict(row), "embedding": vector})
    # 返回全部带向量记录，下一步由 index.py 写入 Milvus。
    return output


def embed_chunks(rows: list[dict], model, **kwargs) -> list[dict]:
    """兼容旧名称：分块向量化实际复用统一的 embed 函数。"""

    # **kwargs 继续传递批次、维度和重试等可选参数。
    return embed(rows, model, **kwargs)


def _estimate_tokens(value: str) -> int:
    """用“约四个字符一个 token”的简单规则估算输入长度。"""

    # 非空文本至少算一个 token；这是快速限长，不是精确分词器计数。
    return max(1, len(value) // 4) if value else 0


def _truncate_tokens(value: str, limit: int) -> str:
    """按估算 token 上限截短用于嵌入的副本文本。"""

    # 统一输入类型并处理 None。
    text = str(value or "")
    # 非正上限代表不允许任何内容。
    if limit <= 0:
        return ""
    # 未超过限制保持原文；超过时按估算比例截到 limit * 4 个字符。
    return text if _estimate_tokens(text) <= limit else text[:limit * 4]


def make_public_batches(collection: str, rows: list[dict], max_records: int = 32, max_characters: int | None = None):
    """为公共集合生成嵌入批次；collection 参数保留给未来差异化策略。"""

    # 当前所有集合使用同一批次规则，del 明确说明此参数暂未参与计算。
    del collection
    # 直接把 embedding_batches 产生的每一批向上返回。
    yield from embedding_batches(rows, max_records, max_characters)


def embed_public_batch(collection: str, rows: list[dict], model, expected_dimension: int | None = None, retry_delay: float = 0.0) -> list[dict]:
    """向量化一批公共法律记录，并移除临时的 embedding_text 字段。"""

    # 当前不同集合使用相同处理方式，参数暂不参与逻辑。
    del collection
    # prepared 保存只对嵌入文本做过限长的记录副本。
    prepared: list[dict] = []
    # 逐条复制，不能截短原记录中的法律正文。
    for source in rows:
        row = dict(source)
        # BGE-M3 输入使用最多约 1500 token，控制延迟和接口负担。
        text = _truncate_tokens(embedding_text(row), 1500)
        # embedding_text 是模型专用副本，不会替换 content/summary 等原文。
        row["embedding_text"] = text
        prepared.append(row)
    # 这一批已经由上层控制大小，因此 batch_size 设为本批数量；空批次至少传 1。
    # 公共数据最多尝试两次，仍失败则终止入库。
    embedded = embed(prepared, model, batch_size=max(1, len(prepared)), expected_dimension=expected_dimension, max_retries=2, retry_delay=retry_delay)
    # embedding_text 只是临时模型输入，入库前删除；法律原文和 embedding 保留。
    return [{key: value for key, value in row.items() if key != "embedding_text"} for row in embedded]


# 限制本模块对外暴露的公共函数名称。
__all__ = ["embed", "embed_chunks", "embed_public_batch", "embedding_batches", "embedding_text", "make_public_batches"]
