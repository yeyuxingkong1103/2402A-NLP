"""根据资料类型把长文本切成适合检索和生成向量的小块。

公共法条、案例等已经是结构化记录时，通常“一条记录就是一块”；普通长文档则先按
句子语义切分，较长文档进一步建立父子块：子块负责准确召回，父块提供完整上下文。
"""

# 延迟解析类型注解。
from __future__ import annotations

# sha1 为旧版公共记录分块生成稳定短 ID。
from hashlib import sha1
# re 用于按句号、换行等边界切分文本。
import re
# uuid5 根据固定输入生成稳定 ID，同一文件同一块重复处理仍得到相同编号。
import uuid
# Any 用于表示公共记录中不同类型的字段值。
from typing import Any


# 这些公共集合已经由 parse.py 解析成结构化记录，不应再次随意打断一条法条或案例。
LAW_COLLECTIONS = {
    "civil_code_articles", "civil_interpretations", "civil_cases", "civil_elements",
    "civil_evidence", "civil_processes", "civil_questions", "civil_citations",
}
# 指定了集合且属于这些文档类型时，也按单条结构化记录处理。
SINGLE_RECORD_TYPES = {"law", "case", "evidence"}
# 非贪婪地匹配一句话，直到常见中英文句末标点或文本结尾。
SENTENCE_RE = re.compile(r".+?(?:[。！？!?；;]|$)")
# 单句话超过块长度时，优先在这些标点处截断。
BREAK_MARKS = ("。", "！", "？", "!", "?", "；", ";")
# 文档达到 5000 字时启用父子分块。
LONG_DOCUMENT_LENGTH = 5000
# 即使总字符不足，语义块达到 4 个也启用父子分块。
LONG_DOCUMENT_CHUNKS = 4
# 多个子块合并成父块时，目标长度约 3000 字。
PARENT_TARGET_LENGTH = 3000


def fixed_chunks(text: str, size: int) -> list[str]:
    """按固定字符数直接切块，不考虑句子边界。"""

    # size 非正会让 range 步长无效，因此立即报错。
    if size <= 0:
        raise ValueError("chunk size 必须大于 0")
    # 统一输入并去除首尾空白。
    value = str(text or "").strip()
    # offset 从 0 开始每次前进 size；空文本直接返回空列表。
    return [value[offset:offset + size] for offset in range(0, len(value), size)] if value else []


def _split_long_unit(text: str, size: int) -> list[str]:
    """把单个超长句切小，并尽量在句末标点后断开。"""

    # remaining 保存尚未切出的文本。
    remaining = str(text or "").strip()
    # pieces 收集切分结果。
    pieces: list[str] = []
    # 只要剩余文本仍超过上限，就继续截取一块。
    while len(remaining) > size:
        # 先查看最多 size 个字符的窗口。
        window = remaining[:size]
        # 找到窗口内最后一个句末标点的位置；都没有时为 -1。
        boundary = max((window.rfind(mark) for mark in BREAK_MARKS), default=-1)
        # 没有合适标点就硬切 size；找到时把标点本身包含进前一块。
        boundary = size if boundary <= 0 else boundary + 1
        # 保存前一块并清除首尾空白。
        pieces.append(remaining[:boundary].strip())
        # 从边界之后继续处理剩余文字。
        remaining = remaining[boundary:].strip()
    # 最后一段即使不足 size 也要保留。
    if remaining:
        pieces.append(remaining)
    return pieces


def _sentence_units(text: str) -> list[str]:
    """先按段落、再按句末标点，把文本拆成语义句单元。"""

    # units 按原文顺序保存非空句子。
    units: list[str] = []
    # 一个或多个换行都视为段落边界。
    for paragraph in re.split(r"\n+", str(text or "").strip()):
        value = paragraph.strip()
        # 空段落不参与分块。
        if value:
            # SENTENCE_RE 找到每句话；推导式清理空白并排除空结果。
            units.extend(part.strip() for part in SENTENCE_RE.findall(value) if part.strip())
    return units


def _merge_short_tail(chunks: list[str], minimum: int) -> list[str]:
    """把过短的最后一块并回前一块，减少缺少语义的碎片。"""

    # 少于两块无法合并；最后一块已达到最小长度也保持原样。
    if len(chunks) < 2 or len(chunks[-1]) >= minimum:
        return chunks
    # 保留前面的块，并用换行合并最后两块。
    return [*chunks[:-2], f"{chunks[-2]}\n{chunks[-1]}".strip()]


def _semantic_chunks(text: str, size: int, minimum: int) -> list[str]:
    """按句子边界组合语义块，并控制最大长度和末块最小长度。"""

    # chunks 保存已经确定的块。
    chunks: list[str] = []
    # current 保存正在累积、还未提交的当前块。
    current = ""
    # 遍历按段落和句号拆出的语义单元。
    for unit in _sentence_units(text):
        # 单个句子仍可能超长，因此先用标点或固定长度进一步拆分。
        for piece in _split_long_unit(unit, size):
            # 当前块非空时用换行连接，保留句子边界。
            candidate = f"{current}\n{piece}".strip() if current else piece
            # 当前为空或合并后没有超长，就继续累积。
            if not current or len(candidate) <= size:
                current = candidate
            else:
                # 合并会超长时，先提交旧块，再用当前句开始新块。
                chunks.append(current)
                current = piece
    # 循环后保存尚未提交的最后一块。
    if current:
        chunks.append(current)
    # minimum 至少为 1 且不超过最大块长，然后合并过短尾块。
    return _merge_short_tail(chunks, min(size, max(1, minimum)))


def _group_parents(chunks: list[str], size: int = PARENT_TARGET_LENGTH) -> list[list[str]]:
    """把相邻子块分组，每组后续组成一个较完整的父块。"""

    # groups 保存全部父块分组。
    groups: list[list[str]] = []
    # current 保存当前父组中的子块。
    current: list[str] = []
    # length 保存当前父组估算字符数。
    length = 0
    # 保持子块原顺序逐个分组。
    for chunk in chunks:
        # bool(current) 在已有内容时加 1，估算连接换行符长度。
        candidate_length = length + len(chunk) + bool(current)
        # 当前组非空且再加入会超过父块目标长度时，先提交当前组。
        if current and candidate_length > size:
            groups.append(current)
            current, length = [chunk], len(chunk)
        else:
            # 未超长就把子块加入当前父组。
            current.append(chunk)
            length = int(candidate_length)
    # 保存最后一个未满组。
    if current:
        groups.append(current)
    return groups


def make_chunk_id(file_id: str, index: int, text: str, role: str = "chunk") -> str:
    """根据文件、块角色、顺序和正文生成可重复的分块 ID。"""

    # 文件 ID 为空时使用 file 占位，避免生成空前缀。
    base = str(file_id or "file").strip() or "file"
    # uuid5 对相同字符串总会生成相同值，截取前 12 位兼顾可读性和区分度。
    digest = uuid.uuid5(uuid.NAMESPACE_URL, f"{base}|{role}|{index}|{str(text or '').strip()}").hex[:12]
    # ID 中同时保留文件、角色和三位顺序，便于人工定位。
    return f"{base}_{role}_{index:03d}_{digest}"


def _chunk_row(metadata: dict, parsed: dict, index: int, text: str, chunk_id: str | None = None, **fields) -> dict:
    """把一段文本包装成包含来源、用户和层级信息的标准分块记录。"""

    # 按优先级取得稳定文件 ID，并兼容旧字段名。
    file_id = str(metadata.get("file_id") or metadata.get("document_id") or metadata.get("file_name") or "file")
    # 建立所有普通、父、子块共用的基础字段。
    row = {
        # 调用者可传父块 ID；否则自动生成普通块 ID。
        "chunk_id": chunk_id or make_chunk_id(file_id, index, text),
        # chunk_index 是本次输出列表中的全局顺序。
        "chunk_index": index,
        # 默认层级为 chunk；父子分块会通过 fields 覆盖。
        "chunk_level": fields.pop("chunk_level", "chunk"),
        # text 是真正用于检索和嵌入的块正文。
        "text": text,
        # 用户与会话字段用于私有资料检索隔离。
        "user_id": metadata.get("user_id", ""),
        "session_id": metadata.get("session_id", ""),
        # 文件标识和文件名用于追踪证据来源。
        "file_id": file_id,
        "file_name": metadata.get("file_name", ""),
        # 文档类型和集合决定后续解析、入库和检索展示。
        "document_type": parsed.get("document_type", "general"),
        "collection": parsed.get("collection", ""),
        # 当前解析器尚未提供章节/页码时使用安全默认值。
        "section": "",
        "page_start": 1,
        "page_end": 1,
    }
    # 添加父子关系等额外字段，同时过滤 None 和空字符串。
    row.update({key: value for key, value in fields.items() if value not in (None, "")})
    return row


def _structured_rows(parsed: dict, metadata: dict) -> list[dict]:
    """把法条、案例等结构化记录直接转换成检索块，不再次切断正文。"""

    # output 保存最终结构化分块。
    output: list[dict] = []
    # 文件 ID 用于无法从业务主键取得 chunk_id 时生成后备 ID。
    file_id = str(metadata.get("file_id") or metadata.get("document_id") or metadata.get("file_name") or "file")
    # records 是当前规范字段，rows 用于兼容旧解析结果。
    for index, source in enumerate(parsed.get("records") or parsed.get("rows") or []):
        # 非字典记录无法安全读取字段，跳过。
        if not isinstance(source, dict):
            continue
        # 依次从模型专用文本、正文、案例摘要或要素摘要中选择检索文本。
        text = str(source.get("embedding_text") or source.get("content") or source.get("summary") or source.get("case_summary") or "").strip()
        # 没有任何正文的记录没有检索价值。
        if not text:
            continue
        # 复制源记录，避免分块阶段修改解析结果。
        row = dict(source)
        # content 统一为展示和生成答案使用的正文。
        row.setdefault("content", str(source.get("content") or source.get("summary") or source.get("case_summary") or text))
        # embedding_text 可包含标题等检索增强文字。
        row.setdefault("embedding_text", text)
        # 补齐集合、文档类型和通用 text 字段。
        row.setdefault("collection", parsed.get("collection", ""))
        row.setdefault("document_type", parsed.get("document_type", "general"))
        row.setdefault("text", text)
        # 优先使用法律记录自己的稳定主键作为 chunk_id；都没有时才自动生成。
        row.setdefault("chunk_id", str(source.get("id") or source.get("case_id") or source.get("evidence_id") or source.get("process_id") or source.get("question_id") or source.get("citation_id") or source.get("serial_number") or make_chunk_id(file_id, index, text)))
        # 加入用户、会话和文件来源，公共数据这些值通常为空。
        row.update(user_id=metadata.get("user_id", ""), session_id=metadata.get("session_id", ""), file_id=metadata.get("file_id", ""), file_name=metadata.get("file_name", ""))
        output.append(row)
    return output


def _parent_child_rows(parsed: dict, metadata: dict, size: int, minimum: int) -> list[dict]:
    """为普通长文生成父子块：子块精确召回，父块补充完整上下文。"""

    # 首先按句子生成较小子块。
    children = _semantic_chunks(parsed.get("text", ""), size, minimum)
    # 取得文件 ID，用于生成稳定父块编号。
    file_id = str(metadata.get("file_id") or metadata.get("document_id") or metadata.get("file_name") or "file")
    # rows 会依次保存父块及其所有子块。
    rows: list[dict] = []
    # 把相邻子块按父块目标长度分组。
    for parent_index, group in enumerate(_group_parents(children)):
        # 父块正文是同组子块按原顺序拼接。
        parent_text = "\n".join(group).strip()
        # len(rows) 是父块在最终结果中的位置，用于稳定 ID。
        parent_id = make_chunk_id(file_id, len(rows), parent_text, "parent")
        # 父块的 parent_chunk_id 指向自身，child_count 记录其子块数量。
        rows.append(_chunk_row(metadata, parsed, len(rows), parent_text, parent_id, chunk_level="parent", parent_chunk_id=parent_id, parent_index=parent_index, child_count=len(group)))
        # 随后加入当前父块下的每个子块。
        for child_index, child_text in enumerate(group):
            # 子块保存父块 ID 和父块全文，命中后可以恢复更完整语境。
            rows.append(_chunk_row(metadata, parsed, len(rows), child_text, chunk_level="child", parent_chunk_id=parent_id, parent_text=parent_text, parent_index=parent_index, child_index=child_index))
    return rows


def chunk(parsed: dict, metadata: dict | None = None, size: int = 1000, minimum: int = 120, *, document_type: str | None = None, collection: str | None = None) -> list[dict]:
    """根据解析结果自动选择结构化分块、普通语义分块或父子分块。"""

    # 最大块长必须为正数。
    if size <= 0:
        raise ValueError("chunk size 必须大于 0")
    # 没有 metadata 时使用空字典，避免后续反复判断 None。
    base = metadata or {}
    # 显式参数优先，其次使用 parsed 中解析出的类型和集合。
    resolved_type = document_type or parsed.get("document_type", "general")
    resolved_collection = collection or parsed.get("collection", "")
    # 公共法律集合已经按法条/案例解析，一条记录保持一块，避免破坏法律边界。
    if resolved_collection in LAW_COLLECTIONS or (resolved_collection and resolved_type in SINGLE_RECORD_TYPES):
        return _structured_rows(parsed, base)
    # 普通文档先按句子语义边界切块。
    chunks = _semantic_chunks(parsed.get("text", ""), size, minimum)
    # 没有有效文字时返回空列表。
    if not chunks:
        return []
    # 长文或块数较多时使用父子结构，兼顾召回精度和上下文完整性。
    if len(str(parsed.get("text", ""))) >= LONG_DOCUMENT_LENGTH or len(chunks) >= LONG_DOCUMENT_CHUNKS:
        return _parent_child_rows(parsed, base, size, minimum)
    # 短文直接包装每个语义块，不增加父子层级。
    return [_chunk_row(base, parsed, index, text) for index, text in enumerate(chunks)]


def chunk_document(parsed: dict, metadata: dict | None = None, chunk_size: int = 1000, minimum: int = 120) -> list[dict]:
    """面向上层流程的易懂入口，把 chunk_size 转交给通用 chunk 函数。"""

    return chunk(parsed, metadata, size=chunk_size, minimum=minimum)


# 旧版公共记录分块使用的“主键字段、正文字段”映射。
PUBLIC_RECORD_FIELDS = {
    "civil_code_articles": ("id", "article_content"),
    "civil_interpretations": ("id", "content"),
    "civil_cases": ("case_id", "summary"),
    "civil_elements": ("serial_number", "case_summary"),
    "civil_evidence": ("evidence_id", "content"),
    "civil_processes": ("process_id", "content"),
    "civil_questions": ("question_id", "content"),
    "civil_citations": ("citation_id", "content"),
}


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """旧版通用文本切分：按句子组合，并让相邻块保留指定重叠文字。"""

    # 块大小必须为正。
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    # overlap 不能为负，也必须小于块大小，否则滑动步长会失效。
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")
    # 统一文本并清理首尾空白。
    content = str(text or "").strip()
    # 空文本没有分块。
    if not content:
        return []
    # 短文本直接作为一块，不需要引入重叠。
    if len(content) <= chunk_size:
        return [content]

    # 按中文句末标点或换行切成句子单元，并过滤空内容。
    units = [item.strip() for item in re.findall(r".*?[。！？；\n]|.+$", content, re.S) if item.strip()]
    # chunks 保存已经确定的块。
    chunks: list[str] = []
    # current 保存正在累积的块。
    current = ""
    # 超长句滑动切片时每次前进“块长-重叠长度”。
    step = chunk_size - overlap
    # 依次处理句子单元。
    for unit in units:
        # 单句本身超过上限时必须单独滑动切分。
        if len(unit) > chunk_size:
            # 先提交此前已累积的正常句子块。
            if current:
                chunks.append(current)
                current = ""
            # 对超长句按 step 滑动，连续块之间保留 overlap 个字符。
            for start in range(0, len(unit), step):
                piece = unit[start:start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                # 当前切片已经覆盖句尾时结束，避免多生成无意义循环。
                if start + chunk_size >= len(unit):
                    break
            continue
        # 当前为空时用本句开始。
        if not current:
            current = unit
        # 加入本句仍未超长，就直接拼接以保持上下文。
        elif len(current) + len(unit) <= chunk_size:
            current += unit
        else:
            # 会超长时先提交当前块。
            chunks.append(current)
            # 新块从旧块尾部保留 overlap 个字符，再接当前句。
            current = f"{current[-overlap:] if overlap else ''}{unit}"
            # 极端情况下重叠加本句仍超长，则放弃重叠，确保块不超过限制。
            if len(current) > chunk_size:
                current = unit
    # 保存最后一个未提交块。
    if current:
        chunks.append(current)
    return chunks


def chunk_records(
    collection: str,
    rows: list[dict[str, Any]],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[dict[str, Any]]:
    """旧版公共数据入口：把每条结构化记录的正文继续切成带来源的子块。"""

    # 只有已定义主键和正文字段的公共集合才能使用。
    if collection not in PUBLIC_RECORD_FIELDS:
        raise ValueError(f"unknown public collection: {collection}")
    # 取得当前集合的业务主键和正文名称。
    id_field, content_field = PUBLIC_RECORD_FIELDS[collection]
    # output 收集所有记录产生的分块。
    output: list[dict[str, Any]] = []
    # 逐条处理结构化记录。
    for row in rows:
        # record_id 指回原始法条、案例或证据主键。
        record_id = str(row.get(id_field, "")).strip()
        # content 是需要切分的正文。
        content = str(row.get(content_field, "")).strip()
        # 其他有意义字段放入 metadata，供检索结果展示和过滤。
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {id_field, content_field, "embedding"} and value not in (None, "", [])
        }
        # 按句子和重叠策略切分当前记录正文。
        for index, piece in enumerate(split_text(content, chunk_size, overlap)):
            # SHA-1 输入包含集合、原记录、顺序和正文，因此同一块可重复生成稳定 ID。
            digest = sha1(f"{collection}|{record_id}|{index}|{piece}".encode("utf-8")).hexdigest()[:32]
            # 建立一条可入向量库的标准分块记录。
            output.append({
                "chunk_id": f"chunk_{digest}",
                "collection": collection,
                "record_id": record_id,
                "parent_id": record_id,
                "chunk_index": index,
                "content": piece,
                "metadata": metadata,
            })
    # 返回所有原记录的全部分块。
    return output


# 明确本模块提供给其他文件的公共函数。
__all__ = [
    "chunk", "chunk_document", "chunk_records", "fixed_chunks", "make_chunk_id",
    "split_text",
]
