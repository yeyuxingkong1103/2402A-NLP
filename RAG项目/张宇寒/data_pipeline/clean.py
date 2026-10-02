"""清理公共法律资料，并在向量化前检查数据质量和集合关联。

清洗只处理空白、页码、装饰线、空记录、重复记录和已知占位内容，不改写法律原意。
校验还会确认案例、法条、证据和引用之间的 ID 真实存在，防止检索到无法追溯的资料。
"""

# 延迟解析类型注解。
from __future__ import annotations

# re 用正则表达式识别页码、装饰线和多余空格。
import re
# Any 表示清理函数可能接收字符串、列表、字典或其他基础值。
from typing import Any


# 这些文字表示案例只有占位说明而没有真实案情，不能作为法律问答依据。
PLACEHOLDER_CASE_PHRASES = ("案情内容暂缺", "本案案情内容暂缺", "详见原文链接", "详见原文")
# 匹配“— 12 —”一类单独页码行。
PAGE_MARKER_RE = re.compile(r"^[－—–-]\s*\d+\s*[－—–-]$")
# 匹配由等号、横线等组成的长装饰分隔线。
SEPARATOR_RE = re.compile(r"^[=_*#-]{8,}$")
# 匹配连续半角空格、制表符和中文全角空格。
HORIZONTAL_SPACE_RE = re.compile(r"[\t \u3000]+")
# 定义每个公共集合的“主键字段、正文字段”。清洗和校验都以此为统一标准。
RECORD_FIELDS = {
    "civil_code_articles": ("id", "article_content"),
    "civil_interpretations": ("id", "content"),
    "civil_cases": ("case_id", "summary"),
    "civil_elements": ("serial_number", "case_summary"),
    "civil_evidence": ("evidence_id", "content"),
    "civil_processes": ("process_id", "content"),
    "civil_questions": ("question_id", "content"),
    "civil_citations": ("citation_id", "content"),
}


def clean_line(value: Any) -> str:
    """清理一行文字：删除隐藏字符、合并水平空格并去掉首尾空白。"""

    # \ufeff 是 BOM，\x00 是空字符；两者都可能来自文件编码或 OCR。
    # sub(" ", ...) 将多种连续空白统一成一个普通空格。
    return HORIZONTAL_SPACE_RE.sub(" ", str(value or "").replace("\ufeff", "").replace("\x00", "")).strip()


def clean_text(value: Any) -> str:
    """逐行清理多行文本，并移除空行、单独页码和装饰分隔线。"""

    # lines 收集最终保留的有效文本行。
    lines: list[str] = []
    # 先统一 Windows/Mac 换行符，再按 \n 分行处理。
    for raw_line in str(value or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        # 清理当前行中的隐藏字符和重复空格。
        line = clean_line(raw_line)
        # 去除空格后的版本只用于识别页码和装饰线，不改变真正保存的正文。
        compact = line.replace(" ", "")
        # 空行、页码行和装饰线没有检索价值，直接跳过。
        if not line or PAGE_MARKER_RE.fullmatch(compact) or SEPARATOR_RE.fullmatch(compact):
            continue
        # 其他内容保持原顺序加入结果。
        lines.append(line)
    # 用换行重新连接，保留正文段落结构。
    return "\n".join(lines)


def is_placeholder_case(title: str, summary: str) -> bool:
    """判断案例标题或摘要是否只是“详见原文”等空内容占位符。"""

    # 标题和摘要一起检查，任一位置出现已知占位短语都视为占位案例。
    content = f"{title}\n{summary}"
    return any(phrase in content for phrase in PLACEHOLDER_CASE_PHRASES)


def _normalize_identifier(value: Any) -> str:
    """删除编号中的所有空白，使格式不同的同一案号能够匹配。"""

    # 例如“（2024） 民初 1号”和“（2024）民初1号”会得到相同标准值。
    return re.sub(r"\s+", "", str(value or ""))


def _clean_value(value: Any) -> Any:
    """递归清理记录中的字符串、列表和字典，同时保持原数据结构。"""

    # 字符串包含换行时按多行正文清理，否则只按单行清理。
    if isinstance(value, str):
        return clean_text(value) if "\n" in value or "\r" in value else clean_line(value)
    # 列表中的每个元素继续调用本函数，支持嵌套结构。
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    # 字典保留键，只递归清理每个值。
    if isinstance(value, dict):
        return {key: _clean_value(item) for key, item in value.items()}
    # 数字、布尔值等非文本类型保持原样。
    return value


def _case_indexes(cases: list[dict]) -> tuple[set[str], dict[str, str]]:
    """为案例建立两个索引：有效 case_id 集合，以及“案号 -> case_id”映射。"""

    # identifiers 用于快速判断某个 case_id 是否存在。
    identifiers: set[str] = set()
    # numbers 用于旧数据只保存案号时反查规范 case_id。
    numbers: dict[str, str] = {}
    # 遍历全部有效案例。
    for row in cases:
        # 读取并清理案例主键。
        case_id = str(row.get("case_id") or "").strip()
        # 没有主键的案例无法建立关联，跳过。
        if not case_id:
            continue
        identifiers.add(case_id)
        # 将案号移除空白后作为索引键。
        case_number = _normalize_identifier(row.get("case_number"))
        if case_number:
            # setdefault 保留首次出现的映射，不让后续重复案号静默覆盖。
            numbers.setdefault(case_number, case_id)
    # 同时返回两个索引供关联修复使用。
    return identifiers, numbers


def _resolve_case_id(row: dict, valid_ids: set[str], ids_by_number: dict[str, str]) -> str:
    """优先保留有效 case_id，否则尝试按案号找到正确案例主键。"""

    # 元素记录使用 case_id，证据记录使用 source_case_id，因此依次兼容两者。
    current = str(row.get("case_id") or row.get("source_case_id") or "").strip()
    # 当前 ID 已在案例表中就直接使用。
    if current in valid_ids:
        return current
    # ID 无效时尝试根据普通案号或来源案号反查。
    for field in ("case_number", "source_case_number"):
        match = ids_by_number.get(_normalize_identifier(row.get(field)))
        if match:
            return match
    # 两种方式都找不到时返回空字符串，避免保留悬空关联。
    return ""


def clean_public_collections(
    collections: dict[str, list[dict]],
) -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
    """清理所有公共集合，并返回清理结果与每个集合的统计报告。"""

    # cleaned 保存可以继续处理的有效记录。
    cleaned: dict[str, list[dict]] = {}
    # report 保存原始、有效和各类丢弃记录数量。
    report: dict[str, dict[str, int]] = {}
    # 逐集合处理，collection 是名称，source_rows 是原始记录列表。
    for collection, source_rows in collections.items():
        # 未登记集合没有明确主键和正文字段，不能猜测处理。
        if collection not in RECORD_FIELDS:
            raise ValueError(f"unknown public collection: {collection}")
        # 取得当前集合的主键字段和正文字段。
        primary_field, content_field = RECORD_FIELDS[collection]
        # output 收集当前集合通过清理的记录。
        output: list[dict] = []
        # seen 用于检测同一集合内重复主键。
        seen: set[str] = set()
        # 三个计数器分别记录空记录、重复记录和占位案例。
        empty_records = duplicate_records = placeholder_records = 0
        # replacement_characters 统计“�”乱码替换字符数量。
        replacement_characters = 0
        # 逐条清理原始记录。
        for source_row in source_rows:
            # 非字典值不符合结构化记录要求，按空记录丢弃。
            if not isinstance(source_row, dict):
                empty_records += 1
                continue
            # 对记录每个字段递归清理，并创建新字典避免修改输入。
            row = {key: _clean_value(value) for key, value in source_row.items()}
            # 提取清理后的主键和正文，用于有效性判断。
            key = str(row.get(primary_field, "")).strip()
            content = str(row.get(content_field, "")).strip()
            # 案例集合额外过滤“详见原文”等无真实案情占位记录。
            if collection == "civil_cases" and is_placeholder_case(
                str(row.get("title", "")), str(row.get("summary", ""))
            ):
                placeholder_records += 1
                continue
            # 主键或正文任一为空都不能入库。
            if not key or not content:
                empty_records += 1
                continue
            # 相同主键只保留第一次出现的记录，避免 Milvus 主键冲突。
            if key in seen:
                duplicate_records += 1
                continue
            # 记录未自动删除的乱码数量，后续严格校验会阻止其入库。
            replacement_characters += str(row).count("�")
            # 标记主键已见并保存有效记录。
            seen.add(key)
            output.append(row)
        # 当前集合清理完毕后写入总结果。
        cleaned[collection] = output
        # 保存可读的质量统计；dropped_records 是所有丢弃类型的总数。
        report[collection] = {
            "source_records": len(source_rows),
            "valid_records": len(output),
            "dropped_records": len(source_rows) - len(output),
            "empty_records": empty_records,
            "duplicate_records": duplicate_records,
            "placeholder_records": placeholder_records,
            "replacement_characters": replacement_characters,
        }

    # 建立“民法典条号 -> 条文 ID”映射，供民事要素记录建立准确关联。
    article_ids = {
        int(row["article_number"]): str(row["id"])
        for row in cleaned.get("civil_code_articles", [])
        if str(row.get("article_number", "")).isdigit()
    }
    # 建立案例主键和案号索引，供要素与证据修复案例关联。
    valid_cases, cases_by_number = _case_indexes(cleaned.get("civil_cases", []))
    # 为每条民事要素补充法条和案例关联。
    for row in cleaned.get("civil_elements", []):
        # serial_number 在该数据结构中对应民法典条号。
        article_number = str(row.get("serial_number", "")).strip()
        # 条号有效且存在时，rag_doc_id 指向真实法条主键。
        if article_number.isdigit() and int(article_number) in article_ids:
            row["rag_doc_id"] = article_ids[int(article_number)]
        # 保存修复前 ID，便于数据追踪。
        previous = str(row.get("case_id") or "").strip()
        # 按已有 ID 或案号解析出真实 case_id。
        row["case_id"] = _resolve_case_id(row, valid_cases, cases_by_number)
        # 只有原 ID 非空且确实被改动时，才记录 original_case_id。
        if previous and previous != row["case_id"]:
            row["original_case_id"] = previous
        # 修正旧数据中的已知错别字“发条”。
        if "发条" in str(row.get("document_type", "")):
            row["document_type"] = "民法典法律要件"
    # 证据记录也需要指向真实存在的案例。
    for row in cleaned.get("civil_evidence", []):
        # 先保留旧来源案例 ID。
        previous = str(row.get("source_case_id") or "").strip()
        # 优先按 ID，必要时按案号修复。
        row["source_case_id"] = _resolve_case_id(row, valid_cases, cases_by_number)
        # 发生修复时保留原值，便于审计来源数据。
        if previous and previous != row["source_case_id"]:
            row["original_source_case_id"] = previous
    # 清洗后的数据继续进入校验，report 可保存为质量报告。
    return cleaned, report


def _article_ids(rows: list[dict]) -> set[str]:
    """提取全部非空法条主键，形成便于快速查询的集合。"""

    # 集合自动去重；条件保证不会把空字符串当成合法 ID。
    return {str(row.get("id") or "").strip() for row in rows if str(row.get("id") or "").strip()}


def validate_public_records(collections: dict[str, list[dict]]) -> dict[str, dict]:
    """严格校验清洗后公共记录；发现问题立即报错，全部通过才返回报告。"""

    # report 最终记录每个集合通过了哪些检查。
    report: dict[str, dict] = {}
    # 第一轮检查所有集合自身的主键、正文和字符质量。
    for collection, rows in collections.items():
        # 未登记集合无法安全校验和建表。
        if collection not in RECORD_FIELDS:
            raise ValueError(f"未知知识库：{collection}")
        # 获取当前集合主键和正文名称。
        primary_field, content_field = RECORD_FIELDS[collection]
        # 提取全部主键并清理首尾空白。
        keys = [str(row.get(primary_field) or "").strip() for row in rows]
        # any(not key) 检测至少一个空主键。
        if any(not key for key in keys):
            raise ValueError(f"{collection} 存在空主键")
        # 列表长度与去重集合长度不同，说明有重复主键。
        if len(keys) != len(set(keys)):
            raise ValueError(f"{collection} 存在主键重复")
        # 分别统计空正文、乱码替换字符和残留装饰线。
        empty = sum(not str(row.get(content_field) or "").strip() for row in rows)
        corrupted = sum(str(row).count("�") for row in rows)
        separators = sum(bool(SEPARATOR_RE.search(line.replace(" ", ""))) for row in rows for line in str(row.get(content_field) or "").splitlines())
        # 任一质量问题都阻止入库，避免大模型引用残缺或乱码资料。
        if empty:
            raise ValueError(f"{collection} 存在 {empty} 条空正文")
        if corrupted:
            raise ValueError(f"{collection} 存在乱码替换字符")
        if separators:
            raise ValueError(f"{collection} 存在 {separators} 条装饰分隔线")
        # 能运行到这里代表这些基础指标全部为零。
        report[collection] = {"record_count": len(rows), "empty_content": 0, "duplicate_primary_keys": 0, "replacement_characters": 0, "separator_lines": 0}

    # 第二轮检查跨集合关联：法条、案例、证据和引用必须指向真实记录。
    # 提取有效法条和案例主键。
    articles = _article_ids(collections.get("civil_code_articles", []))
    cases = {str(row.get("case_id") or "").strip() for row in collections.get("civil_cases", [])} - {""}
    # 即使清洗阶段已经过滤占位案例，严格校验仍再次确认。
    if "civil_cases" in report:
        placeholders = sum(is_placeholder_case(str(row.get("title", "")), str(row.get("summary", ""))) for row in collections["civil_cases"])
        report["civil_cases"]["placeholder_cases"] = placeholders
        if placeholders:
            raise ValueError(f"civil_cases 存在 {placeholders} 条占位案例")
    # 民事要素中的 rag_doc_id 和 case_id 必须分别存在于法条与案例集合。
    if "civil_elements" in report:
        bad_articles = sum(bool(row.get("rag_doc_id")) and str(row["rag_doc_id"]).strip() not in articles for row in collections["civil_elements"])
        bad_cases = sum(bool(row.get("case_id")) and str(row["case_id"]).strip() not in cases for row in collections["civil_elements"])
        report["civil_elements"].update(dangling_rag_doc_id=bad_articles, dangling_case_id=bad_cases)
        if bad_articles or bad_cases:
            raise ValueError(f"civil_elements 存在悬空关联：rag_doc_id={bad_articles}, case_id={bad_cases}")
    # 证据的 source_case_id 必须能找到来源案例。
    if "civil_evidence" in report:
        bad_cases = sum(bool(row.get("source_case_id")) and str(row["source_case_id"]).strip() not in cases for row in collections["civil_evidence"])
        report["civil_evidence"]["dangling_source_case_id"] = bad_cases
        if bad_cases:
            raise ValueError(f"civil_evidence 存在 {bad_cases} 条悬空 source_case_id")
    # 引用集合的源和目标可能属于不同集合，需要构建“集合 -> 全部合法主键”。
    if "civil_citations" in report:
        all_ids = {
            "civil_code_articles": articles,
            "civil_cases": cases,
            "civil_interpretations": {str(row.get("id") or "").strip() for row in collections.get("civil_interpretations", [])},
            "civil_elements": {str(row.get("serial_number") or "").strip() for row in collections.get("civil_elements", [])},
            "civil_evidence": {str(row.get("evidence_id") or "").strip() for row in collections.get("civil_evidence", [])},
            "civil_processes": {str(row.get("process_id") or "").strip() for row in collections.get("civil_processes", [])},
            "civil_questions": {str(row.get("question_id") or "").strip() for row in collections.get("civil_questions", [])},
        }
        # 统计 source_id 不存在于其 source_collection 的引用。
        bad_source = sum(str(row.get("source_id") or "").strip() not in all_ids.get(str(row.get("source_collection") or ""), set()) for row in collections["civil_citations"])
        # 统计 target_id 不存在于其 target_collection 的引用。
        bad_target = sum(str(row.get("target_id") or "").strip() not in all_ids.get(str(row.get("target_collection") or ""), set()) for row in collections["civil_citations"])
        report["civil_citations"].update(bad_source_ids=bad_source, bad_target_ids=bad_target)
        if bad_source or bad_target:
            raise ValueError(f"civil_citations 存在无效引用：source_id={bad_source}, target_id={bad_target}")
    # 全部校验通过后返回质量报告，供保存到 quality_report.json。
    return report


# 明确本模块对外可导入的公共常量与函数。
__all__ = ["RECORD_FIELDS", "clean_line", "clean_public_collections", "clean_text", "is_placeholder_case", "validate_public_records"]
