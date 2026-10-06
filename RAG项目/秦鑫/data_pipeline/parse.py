from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .clean import clean_text, is_placeholder_case

NUMBER_TEXT = "一二三四五六七八九十百千万零〇两0-9"
ARTICLE_RE = re.compile(rf"^第\s*([{NUMBER_TEXT}]+)\s*条")
PAGE_NUMBER_RE = re.compile(r"^[－—-]\s*\d+\s*[－—-]$")
PART_RE = re.compile(rf"^(?:[一二三四五六七八九十]+、.+|第\s*[{NUMBER_TEXT}]+\s*[编章节].*|附\s*则)$")

SOURCE_FILES = {
    "civil_code_articles": "civil_code_articles/civil_code_articles.txt",
    "civil_interpretations": "civil_interpretations/civil_interpretations.docx",
    "civil_cases": "civil_cases/civil_cases.txt",
    "civil_elements": "civil_elements/civil_elements.json",
    "civil_evidence": "civil_evidence/civil_evidence.txt",
    "civil_processes": "civil_processes/civil_processes.txt",
    "civil_questions": "civil_questions/civil_questions.txt",
}

COLLECTION_DOCUMENT_TYPE = {
    "civil_code_articles": "law",
    "civil_interpretations": "law",
    "civil_cases": "case",
    "civil_elements": "law",
    "civil_evidence": "evidence",
    "civil_processes": "general",
    "civil_questions": "general",
    "civil_citations": "general",
}

PRIVATE_SINGLE_RECORD_DOCUMENT_TYPES = {"law", "case", "evidence"}


def chinese_number_to_int(value: str) -> int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    total = 0
    current = 0
    for char in value:
        if char in digits:
            current = digits[char]
        elif char in units:
            unit = units[char]
            if unit == 10000:
                total = (total + current) * unit
            else:
                total += (current or 1) * unit
            current = 0
    return total + current


def _line(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("　", " ")).strip()


def _article_number(line: str) -> int:
    match = ARTICLE_RE.match(line)
    return chinese_number_to_int(match.group(1)) if match else 0


def related_civil_code_articles(text: str) -> list[int]:
    numbers: set[int] = set()
    for sentence in re.split(r"[。；;\n]", text):
        if "民法典" not in sentence:
            continue
        civil_code_part = sentence[sentence.find("民法典"):]
        for number_text in re.findall(rf"第\s*([{NUMBER_TEXT}]+)\s*条", civil_code_part):
            number = chinese_number_to_int(number_text)
            if 1 <= number <= 1260:
                numbers.add(number)
    return sorted(numbers)


def extract_articles(text: str) -> list[dict]:
    rows: list[dict] = []
    hierarchy = {"book": "", "subpart": "", "chapter": "", "section": "", "appendix": ""}
    current_number = 0
    current_content = ""
    current_chapter = ""

    def finish_article() -> None:
        nonlocal current_number, current_content
        if not current_number or not current_content:
            return
        rows.append({
            "id": f"civil_code_articles_article_{current_number}",
            "article_number": current_number,
            "law_name": "中华人民共和国民法典",
            "chapter": current_chapter,
            "article_content": current_content,
            "embedding_text": f"中华人民共和国民法典 {current_chapter} {current_content}".strip(),
        })
        current_number = 0
        current_content = ""

    for raw_line in text.splitlines():
        line = _line(raw_line)
        if not line or PAGE_NUMBER_RE.match(line):
            continue
        compact = line.replace(" ", "")
        if re.match(rf"^第[{NUMBER_TEXT}]+编", compact):
            finish_article()
            hierarchy.update(book=line, subpart="", chapter="", section="", appendix="")
            continue
        if re.match(rf"^第[{NUMBER_TEXT}]+分编", compact):
            finish_article()
            hierarchy.update(subpart=line, chapter="", section="", appendix="")
            continue
        if re.match(rf"^第[{NUMBER_TEXT}]+章", compact):
            finish_article()
            hierarchy.update(chapter=line, section="", appendix="")
            continue
        if re.match(rf"^第[{NUMBER_TEXT}]+节", compact):
            finish_article()
            hierarchy.update(section=line, appendix="")
            continue
        if compact == "附则":
            finish_article()
            hierarchy.update(book="", subpart="", chapter="", section="", appendix="附则")
            continue
        number = _article_number(line)
        if number and current_number and number <= current_number:
            current_content += line
            continue
        if number:
            finish_article()
            current_number = number
            current_content = line
            current_chapter = " > ".join(value for value in hierarchy.values() if value)
        elif current_number:
            current_content += line
    finish_article()
    return rows


def _interpretation_header_start(lines: list[str], first_article_index: int) -> int:
    lower = max(0, first_article_index - 20)
    for index in range(first_article_index - 1, lower - 1, -1):
        line = lines[index]
        if line.startswith("最高人民法院") and "审判委员会" not in line:
            return index
    return lower


def _interpretation_title(header: list[str]) -> str:
    title: list[str] = []
    for line in header:
        if re.search(r"法释\s*[〔\[]", line):
            break
        if line.startswith(("（", "(")) or line.startswith(("为正确", "为了", "根据")):
            break
        if PART_RE.match(line):
            continue
        title.append(line)
    return "".join(title).replace("最高人民法院最高人民法院", "最高人民法院").strip()


def extract_interpretations(text: str) -> list[dict]:
    lines = [_line(item) for item in text.splitlines()]
    lines = [item for item in lines if item and not PAGE_NUMBER_RE.match(item)]
    first_articles = [index for index, line in enumerate(lines) if _article_number(line) == 1]
    header_starts = [_interpretation_header_start(lines, index) for index in first_articles]
    all_rows: list[dict] = []

    for document_index, first_index in enumerate(first_articles, start=1):
        header_start = header_starts[document_index - 1]
        end = header_starts[document_index] if document_index < len(header_starts) else len(lines)
        header = lines[header_start:first_index]
        law_name = _interpretation_title(header) or f"司法解释 {document_index}"
        document_number = next((line for line in header if re.search(r"法释\s*[〔\[]", line)), "")
        initial_part = next((line for line in reversed(header) if PART_RE.match(line)), "")
        articles: list[dict] = []
        current_part = initial_part
        current_number = 0
        current_lines: list[str] = []

        def finish_article() -> None:
            nonlocal current_number, current_lines
            if not current_number or not current_lines:
                return
            articles.append({
                "article_number": current_number,
                "part_name": current_part,
                "content": "\n".join(current_lines),
            })
            current_number = 0
            current_lines = []

        for line in lines[first_index:end]:
            number = _article_number(line)
            if number:
                finish_article()
                current_number = number
                current_lines = [line]
            elif PART_RE.match(line):
                finish_article()
                current_part = line
            elif current_number:
                current_lines.append(line)
        finish_article()

        for sequence, article in enumerate(articles, start=1):
            content = article["content"]
            all_rows.append({
                "id": f"civil_interpretations_{document_index}_{article['article_number']}_{sequence}",
                "law_name": law_name,
                "part_name": article["part_name"],
                "document_number": document_number,
                "article_count": len(articles),
                "content": content,
                "related_article_numbers_text": [str(number) for number in related_civil_code_articles(content)],
                "embedding_text": "\n".join(value for value in (law_name, document_number, article["part_name"], content) if value),
            })
    return all_rows


def _case_fields(lines: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    current_key = ""
    for line in lines:
        match = re.match(r"^([^：]{1,20})：\s*(.*)$", line)
        if match:
            current_key = match.group(1).strip()
            fields[current_key] = match.group(2).strip()
        elif current_key:
            fields[current_key] = (fields[current_key] + line).strip()
    return fields


def extract_cases(text: str) -> list[dict]:
    parts = re.split(r"(?m)^[ \t]*【案例\s*\d+】[ \t]*$", text)
    rows: list[dict] = []
    seen_ids: set[str] = set()
    markers = {"【案情内容】", "【基本案情】"}

    for block in parts[1:]:
        lines = [_line(item) for item in block.splitlines()]
        lines = [item for item in lines if item]
        if not lines:
            continue
        marker_index = next((index for index, line in enumerate(lines) if line in markers), -1)
        metadata_lines = lines[:marker_index] if marker_index >= 0 else lines
        body_lines = lines[marker_index + 1:] if marker_index >= 0 else []
        fields = _case_fields(metadata_lines)
        title = fields.get("标题", "")
        subtitle = ""
        if not title:
            title = metadata_lines[0] if metadata_lines else ""
            subtitle = metadata_lines[1] if len(metadata_lines) > 1 else ""
        summary = "\n".join(body_lines).strip() or "\n".join(metadata_lines).strip()
        document_type = fields.get("文书种类", "")
        material_type = fields.get("材料类型", "")
        if material_type == "司法解释/意见稿" or any(word in document_type for word in ("司法解释", "刑事", "行政")):
            continue
        if not title or not summary:
            continue
        if is_placeholder_case(title, summary):
            continue
        case_number = fields.get("案号", "")
        digest = hashlib.sha1(f"{title}|{case_number}|{summary}".encode("utf-8")).hexdigest()[:24]
        case_id = f"civil_case_{digest}"
        if case_id in seen_ids:
            continue
        seen_ids.add(case_id)
        references = related_civil_code_articles(f"{title}\n{summary}")
        core_issue = fields.get("核心法律问题") or subtitle or summary.split("\n", 1)[0][:500]
        rows.append({
            "case_id": case_id,
            "title": title,
            "case_number": case_number,
            "judgment_date": fields.get("裁判日期", ""),
            "court": fields.get("审理法院", ""),
            "cause_of_action": fields.get("案由", ""),
            "core_legal_issue": core_issue,
            "summary": summary,
            "related_article_numbers": references,
            "related_article_count": len(references),
            "trial_procedure": fields.get("审判程序", ""),
            "region": fields.get("地区", ""),
            "court_level": fields.get("法院层级", ""),
            "document_type": document_type,
            "publishing_authority": fields.get("发布机关", ""),
            "date_type": fields.get("日期类型", ""),
            "body_status": fields.get("正文状态", "有正文"),
            "embedding_text": _embedding_text(title, case_number, fields.get("案由", ""), core_issue, summary, max_tokens=1500),
        })
    return rows


def _value(text: str, field: str) -> str:
    match = re.search(rf"(?m)^{re.escape(field)}：\s*(.*)$", text)
    return match.group(1).strip() if match else ""


def _section(text: str, field: str, next_fields: list[str]) -> str:
    endings = "|".join(re.escape(name) for name in next_fields)
    pattern = rf"(?ms)^{re.escape(field)}：\s*\n?(.*?)(?=^(?:{endings})：|\Z)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def extract_evidence(text: str) -> list[dict]:
    blocks = [block for block in re.split(r"(?m)(?=^案例编号：)", text) if block.strip()]
    rows: list[dict] = []
    seen: set[str] = set()
    for block in blocks:
        number = _value(block, "案例编号")
        detail = _section(block, "详细内容", ["案例编号"])
        title = next((line.strip() for line in detail.splitlines() if line.strip()), "")
        if not number or not detail:
            continue
        evidence_id = f"civil_evidence_{number}"
        if evidence_id in seen:
            digest = hashlib.sha1(detail.encode("utf-8")).hexdigest()[:10]
            evidence_id = f"{evidence_id}_{digest}"
        seen.add(evidence_id)
        rule_name = _value(detail, "规则名称")
        rule_summary = _value(detail, "规则摘要")
        evidence_rule = _section(detail, "证据规则", ["源案例序号", "源子案例序号", "源标题", "源案号"])
        references = related_civil_code_articles(detail)
        rows.append({
            "evidence_id": evidence_id,
            "source_case_id": _value(block, "源案例编号"),
            "title": title,
            "rule_name": rule_name,
            "rule_summary": rule_summary,
            "evidence_rule": evidence_rule,
            "source_case_number": _value(detail, "源案号"),
            "related_article_numbers": references,
            "content": detail,
            "embedding_text": _embedding_text(title, rule_name, rule_summary, evidence_rule, detail, max_tokens=1500),
        })
    return rows


def _numbered_items(text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"(?m)^\d+\.\s*(.+)$", text)]


def _bullet_items(text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"(?m)^-\s*(.+)$", text)]


def extract_processes(text: str) -> list[dict]:
    blocks = [block for block in re.split(r"(?m)(?=^# CPC-\d+)", text) if block.strip()]
    rows: list[dict] = []
    fields = ["适用场景", "处理路径", "所需材料", "交付成果", "来源文件", "依据数量", "法律依据"]
    for block in blocks:
        header = re.search(r"(?m)^#\s+(CPC-\d+)\s+(.+)$", block)
        if not header:
            continue
        process_id, title = header.group(1), header.group(2).strip()
        stages_text = _value(block, "适用阶段")
        scenario = _section(block, "适用场景", fields[1:])
        steps_text = _section(block, "处理路径", fields[2:])
        materials_text = _section(block, "所需材料", fields[3:])
        deliverables_text = _section(block, "交付成果", fields[4:])
        legal_basis = _section(block, "法律依据", [])
        references = related_civil_code_articles(legal_basis)
        rows.append({
            "process_id": process_id,
            "title": title,
            "stages": [item.strip() for item in re.split(r"[、,，]", stages_text) if item.strip()],
            "scenario": scenario,
            "steps": _numbered_items(steps_text),
            "required_materials": _bullet_items(materials_text),
            "deliverables": _bullet_items(deliverables_text),
            "source_files": _bullet_items(_section(block, "来源文件", fields[5:])),
            "legal_basis_count": int(_value(block, "依据数量") or 0),
            "legal_basis": legal_basis,
            "source_ids": re.findall(r"来源ID：\s*([^\s]+)", legal_basis),
            "related_article_numbers": references,
            "content": block.strip(),
            "embedding_text": _embedding_text(title, stages_text, scenario, steps_text, materials_text, legal_basis, max_tokens=1500),
        })
    return rows


def extract_questions(text: str) -> list[dict]:
    blocks = [block for block in re.split(r"(?m)(?=^#\s+\d+\s*$)", text) if block.strip()]
    rows: list[dict] = []
    for block in blocks:
        header = re.search(r"(?m)^#\s+(\d+)\s*$", block)
        question = _section(block, "问题", ["回答", "来源", "complexity", "clarity", "informativeness"])
        answer = _section(block, "回答", ["来源", "complexity", "clarity", "informativeness"])
        if not header or not question or not answer:
            continue
        question_id = f"civil_question_{header.group(1)}"
        source = _value(block, "来源")
        references = related_civil_code_articles(f"{question}\n{answer}")
        rows.append({
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "source": source,
            "complexity": int(_value(block, "complexity") or 0),
            "clarity": int(_value(block, "clarity") or 0),
            "informativeness": int(_value(block, "informativeness") or 0),
            "related_article_numbers": references,
            "content": f"问题：{question}\n回答：{answer}",
            "embedding_text": f"问题：{question}\n回答：{answer}",
        })
    return rows


def _load_elements_payload(payload: object) -> list[dict]:
    source_rows = payload.get("civil_elements", []) if isinstance(payload, dict) else payload
    rows: list[dict] = []
    for source in source_rows or []:
        if not isinstance(source, dict):
            continue
        row = dict(source)
        row["serial_number"] = str(row.get("serial_number", "")).strip()
        row["title"] = str(row.get("title", "")).strip()
        row["case_summary"] = str(row.get("case_summary", "")).strip()
        if not row["serial_number"] or not row["case_summary"]:
            continue
        row["embedding_text"] = _embedding_text(row["title"], row["case_summary"], max_tokens=1500)
        rows.append(row)
    return rows


def load_elements(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return _load_elements_payload(payload)


def _article_ids_by_number(rows: list[dict]) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in rows:
        value = row.get("article_number")
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        result[number] = str(row.get("id", "")).strip()
    return result


def build_citations(collections: dict[str, list[dict]]) -> list[dict]:
    article_ids = _article_ids_by_number(collections.get("civil_code_articles", []))
    source_fields = {
        "civil_interpretations": ("id", "law_name", "related_article_numbers_text"),
        "civil_cases": ("case_id", "title", "related_article_numbers"),
        "civil_evidence": ("evidence_id", "title", "related_article_numbers"),
        "civil_processes": ("process_id", "title", "related_article_numbers"),
        "civil_questions": ("question_id", "question", "related_article_numbers"),
    }
    citations: list[dict] = []
    seen: set[str] = set()

    def add(source_collection: str, source_id: str, title: str, number: int, relation_type: str) -> None:
        target_id = article_ids.get(number)
        if not source_id or not target_id:
            return
        citation_id = hashlib.sha1(f"{source_collection}|{source_id}|{target_id}|{relation_type}".encode("utf-8")).hexdigest()[:32]
        if citation_id in seen:
            return
        seen.add(citation_id)
        content = f"{title or source_id} 引用民法典第{number}条"
        citations.append({
            "citation_id": citation_id,
            "source_collection": source_collection,
            "source_id": source_id,
            "target_collection": "civil_code_articles",
            "target_id": target_id,
            "article_number": number,
            "relation_type": relation_type,
            "content": content,
            "embedding_text": content,
        })

    for collection, (id_field, title_field, reference_field) in source_fields.items():
        for row in collections.get(collection, []):
            for value in row.get(reference_field, []):
                add(collection, str(row.get(id_field, "")), str(row.get(title_field, "")), int(value), "references_article")

    for row in collections.get("civil_elements", []):
        value = str(row.get("serial_number", ""))
        if value.isdigit():
            add("civil_elements", value, str(row.get("title", "")), int(value), "based_on_article")
    return citations


def build_elements(articles: list[dict], interpretations: list[dict], cases: list[dict]) -> list[dict]:
    interpretation_index: dict[int, list[dict]] = {}
    case_index: dict[int, list[dict]] = {}
    for row in interpretations:
        for value in row.get("related_article_numbers_text", []):
            interpretation_index.setdefault(int(value), []).append(row)
    for row in cases:
        for value in row.get("related_article_numbers", []):
            case_index.setdefault(int(value), []).append(row)

    elements: list[dict] = []
    for article in articles:
        number = int(article["article_number"])
        matched_interpretations = interpretation_index.get(number, [])[:3]
        matched_cases = case_index.get(number, [])[:3]
        first_case = matched_cases[0] if matched_cases else {}
        interpretation_texts = [
            _embedding_text(row["law_name"], row.get("document_number", ""), row["content"], max_tokens=500)
            for row in matched_interpretations
        ]
        case_texts = [
            _embedding_text(row["title"], row.get("case_number", ""), row["summary"], max_tokens=500)
            for row in matched_cases
        ]
        summary = _embedding_text(
            f"【法条】\n{article['law_name']} {article['chapter']}\n{article['article_content']}",
            "【司法解释】" if interpretation_texts else "",
            *[f"【司法解释 {index + 1}】\n{text}" for index, text in enumerate(interpretation_texts) if text],
            "【相关案例】" if case_texts else "",
            *[f"【相关案例 {index + 1}】\n{text}" for index, text in enumerate(case_texts) if text],
            max_tokens=1800,
        )
        elements.append({
            "serial_number": str(number),
            "title": f"《中华人民共和国民法典》第{number}条法律要件",
            "case_number": first_case.get("case_number", ""),
            "decision_date": first_case.get("judgment_date", ""),
            "court": first_case.get("court", ""),
            "cause_of_action": first_case.get("cause_of_action", ""),
            "document_type": "民法典法律要件",
            "trial_procedure": first_case.get("trial_procedure", ""),
            "region": first_case.get("region", ""),
            "court_level": first_case.get("court_level", ""),
            "rag_doc_id": article["id"],
            "case_id": first_case.get("case_id", ""),
            "publishing_authority": "全国人民代表大会",
            "date_type": "法条与关联材料",
            "body_status": f"司法解释{len(matched_interpretations)}条；案例{len(matched_cases)}条",
            "case_summary": summary,
            "embedding_text": _embedding_text(article["law_name"], article["chapter"], article["article_content"], summary, max_tokens=1800),
        })
    return elements


def detect_type(text: str) -> str:
    if re.search(r"第[一二三四五六七八九十百千万零〇两\d]+条", text) and any(word in text for word in ("中华人民共和国", "条例", "办法")):
        return "law"
    if any(word in text for word in ("法院认为", "案号", "判决如下", "裁定如下")):
        return "case"
    if any(word in text for word in ("甲方", "乙方", "违约责任", "争议解决", "合同")):
        return "contract"
    if any(word in text for word in ("证据", "转账记录", "聊天记录", "录音", "照片")):
        return "evidence"
    return "general"


def detect_document_type(text: str) -> str:
    """兼容旧调用方的文档类型检测名称。"""
    return detect_type(text)


def extract_sections(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    sections: dict[str, str] = {}
    for line in text.splitlines():
        for label in labels:
            match = re.match(rf"^{re.escape(label)}\s*[:：]\s*(.+)$", line)
            if match:
                sections[label] = match.group(1).strip()
    return sections


def build_metadata(text: str, document_type: str, collection: str | None = None) -> dict:
    if document_type == "contract":
        labels = ("合同主体", "合同期限", "标的", "付款", "工资待遇", "解除条件", "违约责任", "争议解决")
    elif document_type == "case":
        labels = ("法院", "案号", "原告", "被告", "诉讼请求", "案件事实", "法院认为", "引用法条", "判决结果")
    else:
        labels = ()
    return {
        "document_type": document_type,
        "collection": collection or "",
        "char_count": len(text),
        "sections": extract_sections(text, labels),
    }


def _safe_truncate_utf8(text: str, max_bytes: int) -> str:
    value = str(text or "")
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _estimate_tokens(value: str) -> int:
    return max(1, len(value) // 4) if value else 0


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    value = str(text or "")
    if max_tokens <= 0 or not value:
        return ""
    if _estimate_tokens(value) <= max_tokens:
        return value
    truncated = value[: max_tokens * 4]
    while truncated and _estimate_tokens(truncated) > max_tokens:
        truncated = truncated[:-1]
    return truncated


def _embedding_text(*parts: object, max_tokens: int = 1500) -> str:
    segments: list[str] = []
    remaining_tokens = max_tokens
    for part in parts:
        value = str(part or "").strip()
        if not value or remaining_tokens <= 0:
            continue
        tokens = _estimate_tokens(value)
        if tokens <= remaining_tokens:
            segments.append(value)
            remaining_tokens -= tokens
            continue
        truncated = _truncate_to_tokens(value, remaining_tokens)
        if truncated:
            segments.append(truncated)
        break
    return "\n".join(segments).strip()


def _hash_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _single_record(document_type: str, text: str, *, collection: str | None = None) -> list[dict]:
    value = str(text or "").strip()
    if not value:
        return []
    first_line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    prefix = collection or document_type or "record"
    return [{"id": _hash_id(prefix, f"{prefix}|{value}"), "title": first_line or document_type, "content": value, "embedding_text": _embedding_text(value, max_tokens=1500)}]


def _public_records(collection: str, raw_text: str, source_path: Path | None = None) -> list[dict]:
    if collection == "civil_code_articles":
        return extract_articles(raw_text)
    if collection == "civil_interpretations":
        return extract_interpretations(raw_text)
    if collection == "civil_cases":
        return extract_cases(raw_text)
    if collection == "civil_elements":
        return load_elements(source_path) if source_path is not None else _load_elements_payload(json.loads(raw_text))
    if collection == "civil_evidence":
        return extract_evidence(raw_text)
    if collection == "civil_processes":
        return extract_processes(raw_text)
    if collection == "civil_questions":
        return extract_questions(raw_text)
    if collection == "civil_citations":
        return []
    return []


def parse(text: str, document_type: str | None = None, *, collection: str | None = None, source_path: Path | None = None) -> dict:
    raw_text = str(text or "")
    cleaned = clean_text(raw_text)
    if collection:
        resolved_type = COLLECTION_DOCUMENT_TYPE.get(collection, document_type or detect_type(cleaned))
        records = _public_records(collection, raw_text, source_path)
    else:
        resolved_type = document_type or detect_type(cleaned)
        if resolved_type == "law":
            records = extract_articles(cleaned) or _single_record(resolved_type, cleaned)
        elif resolved_type in PRIVATE_SINGLE_RECORD_DOCUMENT_TYPES:
            records = _single_record(resolved_type, cleaned)
        else:
            records = []

    metadata = build_metadata(cleaned, resolved_type, collection)
    metadata["record_count"] = len(records)
    return {
        "text": cleaned,
        "document_type": resolved_type,
        "collection": collection or "",
        "records": records,
        "metadata": metadata,
    }


def parse_document(text: str, document_type: str | None = None, *, collection: str | None = None, source_path: Path | None = None) -> dict:
    return parse(text, document_type=document_type, collection=collection, source_path=source_path)


__all__ = [
    "SOURCE_FILES",
    "build_citations",
    "build_elements",
    "build_metadata",
    "clean_text",
    "chinese_number_to_int",
    "detect_type",
    "detect_document_type",
    "extract_articles",
    "extract_cases",
    "extract_evidence",
    "extract_interpretations",
    "extract_processes",
    "extract_questions",
    "extract_sections",
    "is_placeholder_case",
    "load_elements",
    "parse",
    "parse_document",
    "related_civil_code_articles",
]
