"""把清理后的文字解析成法条、司法解释、案例、证据、流程和问答记录。

读取器只知道“文件里有什么文字”，本文件进一步理解资料的固定格式。例如民法典按
“第×条”拆记录，案例按“【案例×】”拆记录，证据按“案例编号”拆记录。解析后的
字典字段会直接影响分块、向量化、Milvus schema 和在线检索展示。
"""

# 延迟解析类型注解。
from __future__ import annotations

# hashlib 用法律记录内容生成稳定主键。
import hashlib
# json 解析民事要素 JSON 数据。
import json
# re 用正则识别条号、标题、字段和不同记录边界。
import re
# Path 表示原始文件路径，民事要素可直接从 JSON 文件加载。
from pathlib import Path

# clean_text 清理通用噪声；is_placeholder_case 过滤没有真实案情的案例。
from .clean import clean_text, is_placeholder_case

# 法条编号可能出现的中文数字、阿拉伯数字和“两”。
NUMBER_TEXT = "一二三四五六七八九十百千万零〇两0-9"
# 从行首识别“第×条”，括号内捕获条号文字。
ARTICLE_RE = re.compile(rf"^第\s*([{NUMBER_TEXT}]+)\s*条")
# 识别“— 12 —”一类单独页码。
PAGE_NUMBER_RE = re.compile(r"^[－—-]\s*\d+\s*[－—-]$")
# 识别司法解释或法律中的大标题：编号标题、编/章/节和附则。
PART_RE = re.compile(rf"^(?:[一二三四五六七八九十]+、.+|第\s*[{NUMBER_TEXT}]+\s*[编章节].*|附\s*则)$")

# 约定每个公共集合相对于 data/public 的原始文件位置。
SOURCE_FILES = {
    "civil_code_articles": "civil_code_articles/civil_code_articles.txt",
    "civil_interpretations": "civil_interpretations/civil_interpretations.docx",
    "civil_cases": "civil_cases/civil_cases.txt",
    "civil_elements": "civil_elements/civil_elements.json",
    "civil_evidence": "civil_evidence/civil_evidence.txt",
    "civil_processes": "civil_processes/civil_processes.txt",
    "civil_questions": "civil_questions/civil_questions.txt",
}

# 不同公共集合对应的通用文档类型，用于分块和元数据。
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

# 用户上传这些类型时通常整份作为一条结构化记录，再由分块器决定后续处理。
PRIVATE_SINGLE_RECORD_DOCUMENT_TYPES = {"law", "case", "evidence"}


def chinese_number_to_int(value: str) -> int:
    """把“一百二十六”或“126”转换成整数 126。"""

    # 清除条号两侧空白。
    value = value.strip()
    # 已经是阿拉伯数字时直接转换。
    if value.isdigit():
        return int(value)
    # 中文数字字符对应的个位值。
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    # 中文数位对应倍率。
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    # total 保存已经完成的数位累加结果。
    total = 0
    # current 保存最近读到的数字，例如“二十”中的二。
    current = 0
    # 从左到右读取中文数字。
    for char in value:
        if char in digits:
            current = digits[char]
        elif char in units:
            unit = units[char]
            # “万”需要把前面整体乘万。
            if unit == 10000:
                total = (total + current) * unit
            else:
                # “十”前省略“一”时按 1 处理，例如“十二”。
                total += (current or 1) * unit
            current = 0
    # 加上最后未遇到单位的个位数字。
    return total + current


def _line(value: str) -> str:
    """把一行中的全角空格和连续空白统一为单个普通空格。"""

    return re.sub(r"\s+", " ", value.replace("　", " ")).strip()


def _article_number(line: str) -> int:
    """从行首提取法条数字，无法识别时返回 0。"""

    match = ARTICLE_RE.match(line)
    # group(1) 是“第”和“条”之间的编号文字。
    return chinese_number_to_int(match.group(1)) if match else 0


def related_civil_code_articles(text: str) -> list[int]:
    """提取文字中明确写在“民法典”之后的第×条引用。"""

    # 集合自动去除同一段中重复出现的条号。
    numbers: set[int] = set()
    # 先按句号、分号和换行分句，避免其他法律条号被误当成民法典条号。
    for sentence in re.split(r"[。；;\n]", text):
        # 句子没有出现“民法典”就不提取。
        if "民法典" not in sentence:
            continue
        # 只查看“民法典”之后的内容。
        civil_code_part = sentence[sentence.find("民法典"):]
        # 一句话可能引用多条法条，全部找出。
        for number_text in re.findall(rf"第\s*([{NUMBER_TEXT}]+)\s*条", civil_code_part):
            number = chinese_number_to_int(number_text)
            # 现行民法典为 1—1260 条，范围外数字不作为有效引用。
            if 1 <= number <= 1260:
                numbers.add(number)
    # 排序后返回，确保每次处理结果稳定。
    return sorted(numbers)


def extract_articles(text: str) -> list[dict]:
    """按“一条法律为一条记录”解析民法典，并保留编、章、节层级。"""

    # rows 保存全部法条记录。
    rows: list[dict] = []
    # hierarchy 随标题变化更新，用于记录当前法条所在层级。
    hierarchy = {"book": "", "subpart": "", "chapter": "", "section": "", "appendix": ""}
    # 以下三个变量保存正在累积、尚未提交的当前法条。
    current_number = 0
    current_content = ""
    current_chapter = ""

    def finish_article() -> None:
        """把当前累积法条加入 rows，然后清空状态。"""

        # 内层函数需要修改外层变量，因此声明 nonlocal。
        nonlocal current_number, current_content
        # 没有条号或正文时没有可提交记录。
        if not current_number or not current_content:
            return
        # 生成结构化法条记录。
        rows.append({
            "id": f"civil_code_articles_article_{current_number}",
            "article_number": current_number,
            "law_name": "中华人民共和国民法典",
            "chapter": current_chapter,
            "article_content": current_content,
            "embedding_text": f"中华人民共和国民法典 {current_chapter} {current_content}".strip(),
        })
        # 清空当前条，等待下一条开始。
        current_number = 0
        current_content = ""

    # 逐行判断标题、条文开始或条文续行。
    for raw_line in text.splitlines():
        line = _line(raw_line)
        # 空行和单独页码不属于正文。
        if not line or PAGE_NUMBER_RE.match(line):
            continue
        # 去掉空格便于匹配“第 一 编”等不同排版。
        compact = line.replace(" ", "")
        # 遇到新的编标题时先提交上一条，再重置下级层次。
        if re.match(rf"^第[{NUMBER_TEXT}]+编", compact):
            finish_article()
            hierarchy.update(book=line, subpart="", chapter="", section="", appendix="")
            continue
        # 分编、章、节同样先结束上一条，并只重置它们下面的层级。
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
        # 尝试把当前行识别为一条新法条。
        number = _article_number(line)
        # 某些正文中可能提到更小条号；如果当前条尚未结束，不把它误判成下一条。
        if number and current_number and number <= current_number:
            current_content += line
            continue
        # 正常的新条号先提交上一条，再开始累积本条。
        if number:
            finish_article()
            current_number = number
            current_content = line
            # 用 > 连接当前非空编、分编、章、节或附则标题。
            current_chapter = " > ".join(value for value in hierarchy.values() if value)
        # 没有新条号但已有当前法条时，视为该条的续行。
        elif current_number:
            current_content += line
    # 文件结束后还要提交最后一条。
    finish_article()
    return rows


def _interpretation_header_start(lines: list[str], first_article_index: int) -> int:
    """从司法解释第一条向上寻找该文件标题的起点。"""

    # 最多向前查看 20 行，防止把上一份文件大量内容并入标题。
    lower = max(0, first_article_index - 20)
    # 从第一条的上一行反向搜索。
    for index in range(first_article_index - 1, lower - 1, -1):
        line = lines[index]
        # 最高人民法院标题常是司法解释开头；审判委员会说明不算新标题。
        if line.startswith("最高人民法院") and "审判委员会" not in line:
            return index
    # 没找到明显标题时使用最多 20 行前的位置。
    return lower


def _interpretation_title(header: list[str]) -> str:
    """从司法解释头部提取标题，排除文号、说明和章节名。"""

    # title 收集可能跨多行的标题文字。
    title: list[str] = []
    # 按原顺序读取头部行。
    for line in header:
        # 遇到“法释〔…〕”文号后，标题通常已经结束。
        if re.search(r"法释\s*[〔\[]", line):
            break
        # 日期括号或“为正确/根据”等正文引导语也表示标题结束。
        if line.startswith(("（", "(")) or line.startswith(("为正确", "为了", "根据")):
            break
        # 编、章、节标题不是整份司法解释名称。
        if PART_RE.match(line):
            continue
        title.append(line)
    # 拼接跨行标题，并修复已知重复机构名。
    return "".join(title).replace("最高人民法院最高人民法院", "最高人民法院").strip()


def extract_interpretations(text: str) -> list[dict]:
    """把包含多份司法解释的文本按文件和条文解析成独立记录。"""

    # 先逐行统一空白。
    lines = [_line(item) for item in text.splitlines()]
    # 删除空行和页码行。
    lines = [item for item in lines if item and not PAGE_NUMBER_RE.match(item)]
    # 每份司法解释一般从“第一条”开始，因此所有第一条位置就是文件候选边界。
    first_articles = [index for index, line in enumerate(lines) if _article_number(line) == 1]
    # 为每个第一条向上定位其标题头部起点。
    header_starts = [_interpretation_header_start(lines, index) for index in first_articles]
    # all_rows 汇总全部文件中的所有条文。
    all_rows: list[dict] = []

    # document_index 从 1 开始，为不同司法解释生成稳定 ID。
    for document_index, first_index in enumerate(first_articles, start=1):
        # 当前文件头部起点。
        header_start = header_starts[document_index - 1]
        # 下一份文件头部就是当前文件结尾；最后一份延伸到全部行结束。
        end = header_starts[document_index] if document_index < len(header_starts) else len(lines)
        # header 位于标题起点和第一条之间。
        header = lines[header_start:first_index]
        # 提取法律名称；失败时使用带序号的明确后备名称。
        law_name = _interpretation_title(header) or f"司法解释 {document_index}"
        # 文号是头部第一行匹配“法释〔”的内容。
        document_number = next((line for line in header if re.search(r"法释\s*[〔\[]", line)), "")
        # 第一条之前可能已经出现分部标题，取离第一条最近的一项。
        initial_part = next((line for line in reversed(header) if PART_RE.match(line)), "")
        # articles 保存当前司法解释中的条文临时结构。
        articles: list[dict] = []
        current_part = initial_part
        current_number = 0
        current_lines: list[str] = []

        def finish_article() -> None:
            """提交当前司法解释条文并清空累积状态。"""

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

        # 只遍历当前司法解释的行区间。
        for line in lines[first_index:end]:
            number = _article_number(line)
            # 新条号表示上一条结束。
            if number:
                finish_article()
                current_number = number
                current_lines = [line]
            # 分部标题也结束当前条，并更新后续条文所属部分。
            elif PART_RE.match(line):
                finish_article()
                current_part = line
            # 普通行追加到当前条文。
            elif current_number:
                current_lines.append(line)
        # 提交当前文件最后一条。
        finish_article()

        # 把临时条文转成最终公共集合记录。
        for sequence, article in enumerate(articles, start=1):
            content = article["content"]
            # 提取这条司法解释明确引用的民法典条号。
            related_articles = related_civil_code_articles(content)
            all_rows.append({
                "id": f"civil_interpretations_{document_index}_{article['article_number']}_{sequence}",
                "law_name": law_name,
                "part_name": article["part_name"],
                "document_number": document_number,
                "article_count": len(articles),
                "content": content,
                "related_article_numbers": related_articles,
                "related_article_numbers_text": [str(number) for number in related_articles],
                "embedding_text": "\n".join(value for value in (law_name, document_number, article["part_name"], content) if value),
            })
    return all_rows


def _case_fields(lines: list[str]) -> dict[str, str]:
    """解析案例头部“字段名：字段值”，并支持值延续到下一行。"""

    # fields 保存全部元数据字段。
    fields: dict[str, str] = {}
    # current_key 记录上一行字段名，普通续行会追加到它。
    current_key = ""
    for line in lines:
        # 字段名最长 20 字且不能包含冒号。
        match = re.match(r"^([^：]{1,20})：\s*(.*)$", line)
        if match:
            current_key = match.group(1).strip()
            fields[current_key] = match.group(2).strip()
        elif current_key:
            # 没出现新字段名就视为上一字段的换行续写。
            fields[current_key] = (fields[current_key] + line).strip()
    return fields


def extract_cases(text: str, diagnostics: dict[str, int] | None = None) -> list[dict]:
    """按“【案例 n】”解析民事案例，过滤非民事和无正文记录。"""

    # split 后第 0 项是首个案例标记之前的文字，因此后面从 parts[1:] 开始。
    parts = re.split(r"(?m)^[ \t]*【案例\s*\d+】[ \t]*$", text)
    # rows 保存有效案例。
    rows: list[dict] = []
    # seen_ids 防止相同标题、案号和正文生成重复记录。
    seen_ids: set[str] = set()
    # 两种标题都表示后面开始进入案情正文。
    markers = {"【案情内容】", "【基本案情】"}

    # 逐个案例块解析。
    for block in parts[1:]:
        lines = [_line(item) for item in block.splitlines()]
        lines = [item for item in lines if item]
        # 空案例块直接跳过。
        if not lines:
            continue
        # 查找案情正文标记位置，找不到时为 -1。
        marker_index = next((index for index, line in enumerate(lines) if line in markers), -1)
        # 标记前是元数据；没有标记时暂时把全部行视为元数据。
        metadata_lines = lines[:marker_index] if marker_index >= 0 else lines
        # 标记后是案情正文。
        body_lines = lines[marker_index + 1:] if marker_index >= 0 else []
        # 解析“标题、案号、案由”等字段。
        fields = _case_fields(metadata_lines)
        title = fields.get("标题", "")
        subtitle = ""
        # 没有明确“标题：”时，兼容把前两行当主副标题的旧格式。
        if not title:
            title = metadata_lines[0] if metadata_lines else ""
            subtitle = metadata_lines[1] if len(metadata_lines) > 1 else ""
        # 优先使用案情正文；没有标记时使用元数据全文作为摘要后备。
        summary = "\n".join(body_lines).strip() or "\n".join(metadata_lines).strip()
        # 读取文书和材料类型，用来排除混入案例集的非民事资料。
        document_type = fields.get("文书种类", "")
        material_type = fields.get("材料类型", "")
        # 司法解释、刑事和行政材料不属于当前民事案例库。
        if material_type == "司法解释/意见稿" or any(word in document_type for word in ("司法解释", "刑事", "行政")):
            continue
        # 标题和摘要都是案例检索的最低要求。
        if not title or not summary:
            continue
        # “详见原文”等占位案例不能作为事实依据。
        if is_placeholder_case(title, summary):
            continue
        # 案号可能为空，因此 ID 还会包含标题和摘要。
        case_number = fields.get("案号", "")
        # 内容哈希使相同案例重复处理仍得到稳定主键。
        digest = hashlib.sha1(f"{title}|{case_number}|{summary}".encode("utf-8")).hexdigest()[:24]
        case_id = f"civil_case_{digest}"
        # 本批中已出现相同 ID 时跳过重复记录。
        if case_id in seen_ids:
            continue
        seen_ids.add(case_id)
        # 提取案例中明确引用的民法典条号。
        references = related_civil_code_articles(f"{title}\n{summary}")
        # 核心争点优先读取明确字段，其次副标题，最后取摘要首段前 500 字。
        core_issue = fields.get("核心法律问题") or subtitle or summary.split("\n", 1)[0][:500]
        # 构造案例公共集合记录；embedding_text 汇总最有检索价值的字段并限长。
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
    """取得“字段：值”中与指定字段同一行的短值。"""

    # (?m) 让 ^ 匹配每一行开头；re.escape 防止字段名中的符号影响正则。
    match = re.search(rf"(?m)^{re.escape(field)}：\s*(.*)$", text)
    return match.group(1).strip() if match else ""


def _section(text: str, field: str, next_fields: list[str]) -> str:
    """提取从某字段开始到下一个指定字段或文本结尾的多行内容。"""

    # 将后续字段转成正则“字段A|字段B”。
    endings = "|".join(re.escape(name) for name in next_fields)
    # (?ms) 同时启用多行和点号跨行；前瞻保证不吞掉下一字段标题。
    pattern = rf"(?ms)^{re.escape(field)}：\s*\n?(.*?)(?=^(?:{endings})：|\Z)"
    # 执行搜索并返回内容部分。
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def extract_evidence(text: str) -> list[dict]:
    """按“案例编号：”拆分证据规则记录。"""

    # 正向前瞻保留每个块开头的“案例编号”字段。
    blocks = [block for block in re.split(r"(?m)(?=^案例编号：)", text) if block.strip()]
    # rows 保存有效证据规则，seen 用于检测重复 ID。
    rows: list[dict] = []
    seen: set[str] = set()
    # 逐个证据块处理。
    for block in blocks:
        # number 是当前记录编号，detail 是“详细内容”多行部分。
        number = _value(block, "案例编号")
        detail = _section(block, "详细内容", ["案例编号"])
        # 详细内容第一条非空行作为展示标题。
        title = next((line.strip() for line in detail.splitlines() if line.strip()), "")
        # 编号或详细内容缺失时无法建立有效证据记录。
        if not number or not detail:
            continue
        # 默认主键由案例编号构成。
        evidence_id = f"civil_evidence_{number}"
        # 同一编号出现多条时，用正文哈希后缀区分，避免主键覆盖。
        if evidence_id in seen:
            digest = hashlib.sha1(detail.encode("utf-8")).hexdigest()[:10]
            evidence_id = f"{evidence_id}_{digest}"
        seen.add(evidence_id)
        # 提取规则名称、摘要和完整证据规则部分。
        rule_name = _value(detail, "规则名称")
        rule_summary = _value(detail, "规则摘要")
        evidence_rule = _section(detail, "证据规则", ["源案例序号", "源子案例序号", "源标题", "源案号"])
        # 提取明确引用的民法典条号。
        references = related_civil_code_articles(detail)
        # 生成证据集合记录和用于向量检索的限长文本。
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
    """提取“1. 内容”形式的有序项目。"""

    return [match.group(1).strip() for match in re.finditer(r"(?m)^\d+\.\s*(.+)$", text)]


def _bullet_items(text: str) -> list[str]:
    """提取“- 内容”形式的无序项目。"""

    return [match.group(1).strip() for match in re.finditer(r"(?m)^-\s*(.+)$", text)]


def extract_processes(text: str) -> list[dict]:
    """解析 CPC 编号的民事处理流程资料。"""

    # 每个流程以“# CPC-数字”标题开始，正向前瞻保留标题。
    blocks = [block for block in re.split(r"(?m)(?=^# CPC-\d+)", text) if block.strip()]
    rows: list[dict] = []
    # 固定字段顺序用于确定每个多行段落在哪里结束。
    fields = ["适用场景", "处理路径", "所需材料", "交付成果", "来源文件", "依据数量", "法律依据"]
    # 逐流程块解析。
    for block in blocks:
        # 标题同时捕获流程 ID 和名称。
        header = re.search(r"(?m)^#\s+(CPC-\d+)\s+(.+)$", block)
        if not header:
            continue
        process_id, title = header.group(1), header.group(2).strip()
        # 适用阶段是单行值，其余部分可能跨多行。
        stages_text = _value(block, "适用阶段")
        scenario = _section(block, "适用场景", fields[1:])
        steps_text = _section(block, "处理路径", fields[2:])
        materials_text = _section(block, "所需材料", fields[3:])
        deliverables_text = _section(block, "交付成果", fields[4:])
        legal_basis = _section(block, "法律依据", [])
        # 从法律依据中提取民法典引用。
        references = related_civil_code_articles(legal_basis)
        # 把文本段落进一步解析成数组和结构化字段。
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
    """解析带编号、问题、回答和质量分数的法律问答资料。"""

    # 每条问答以仅包含“# 数字”的标题开始。
    blocks = [block for block in re.split(r"(?m)(?=^#\s+\d+\s*$)", text) if block.strip()]
    rows: list[dict] = []
    # 逐问答块处理。
    for block in blocks:
        # header 捕获问答编号。
        header = re.search(r"(?m)^#\s+(\d+)\s*$", block)
        # 问题和回答都可能是多行，直到下一字段开始。
        question = _section(block, "问题", ["回答", "来源", "complexity", "clarity", "informativeness"])
        answer = _section(block, "回答", ["来源", "complexity", "clarity", "informativeness"])
        # 三者任一缺失都无法形成有效问答记录。
        if not header or not question or not answer:
            continue
        question_id = f"civil_question_{header.group(1)}"
        source = _value(block, "来源")
        # 同时检查问题和答案中的民法典引用。
        references = related_civil_code_articles(f"{question}\n{answer}")
        # 构造问答记录；content 和 embedding_text 都保留完整问答关系。
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
    """从已加载 JSON 对象中提取和规范民事法律要素记录。"""

    # JSON 顶层可能是 {civil_elements: [...]}，也可能直接就是列表。
    source_rows = payload.get("civil_elements", []) if isinstance(payload, dict) else payload
    rows: list[dict] = []
    # payload 为空时使用空列表，避免遍历 None。
    for source in source_rows or []:
        # 非字典项无法形成结构化记录。
        if not isinstance(source, dict):
            continue
        # 复制源记录，避免修改 JSON 原对象。
        row = dict(source)
        # 主键、标题和摘要统一转成去除首尾空白的字符串。
        row["serial_number"] = str(row.get("serial_number", "")).strip()
        row["title"] = str(row.get("title", "")).strip()
        row["case_summary"] = str(row.get("case_summary", "")).strip()
        # 条号和摘要是要素记录最低要求。
        if not row["serial_number"] or not row["case_summary"]:
            continue
        # 检索文本组合标题和摘要，并限制模型输入长度。
        row["embedding_text"] = _embedding_text(row["title"], row["case_summary"], max_tokens=1500)
        rows.append(row)
    return rows


def load_elements(path: Path) -> list[dict]:
    """从 UTF-8 JSON 文件读取民事要素。"""

    # utf-8-sig 同时兼容带 BOM 和不带 BOM 的 JSON。
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return _load_elements_payload(payload)


def extract_elements(payload: object) -> list[dict]:
    """解析内存中已经加载的民事要素 JSON 对象。"""

    return _load_elements_payload(payload)


def _article_ids_by_number(rows: list[dict]) -> dict[int, str]:
    """建立“民法典条号 -> 法条主键”映射。"""

    result: dict[int, str] = {}
    for row in rows:
        value = row.get("article_number")
        # 条号可能来自文本或数字，统一转为 int。
        try:
            number = int(value)
        except (TypeError, ValueError):
            # 无法转换的异常数据不加入索引。
            continue
        # 保存该条号对应的法条 ID。
        result[number] = str(row.get("id", "")).strip()
    return result


def build_citations(collections: dict[str, list[dict]]) -> list[dict]:
    """根据各资料中提取的民法典条号，构建可追溯的跨集合引用记录。"""

    # 先建立条号到真实法条 ID 的索引。
    article_ids = _article_ids_by_number(collections.get("civil_code_articles", []))
    # 每个来源集合分别声明：主键字段、展示标题字段、引用条号字段。
    source_fields = {
        "civil_interpretations": ("id", "law_name", "related_article_numbers_text"),
        "civil_cases": ("case_id", "title", "related_article_numbers"),
        "civil_evidence": ("evidence_id", "title", "related_article_numbers"),
        "civil_processes": ("process_id", "title", "related_article_numbers"),
        "civil_questions": ("question_id", "question", "related_article_numbers"),
    }
    # citations 保存引用记录，seen 防止同一引用重复生成。
    citations: list[dict] = []
    seen: set[str] = set()

    def add(source_collection: str, source_id: str, title: str, number: int, relation_type: str) -> None:
        """校验并添加一条“来源记录 -> 民法典条文”的引用。"""

        # 用条号找到真实目标法条主键。
        target_id = article_ids.get(number)
        # 来源或目标不存在时不生成悬空引用。
        if not source_id or not target_id:
            return
        # 引用 ID 由源、目标和关系类型共同决定，重复处理结果稳定。
        citation_id = hashlib.sha1(f"{source_collection}|{source_id}|{target_id}|{relation_type}".encode("utf-8")).hexdigest()[:32]
        if citation_id in seen:
            return
        seen.add(citation_id)
        # content 是可直接检索和展示的人类可读关系。
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

    # 处理司法解释、案例、证据、流程和问答中的条号引用。
    for collection, (id_field, title_field, reference_field) in source_fields.items():
        for row in collections.get(collection, []):
            for value in row.get(reference_field, []):
                # 这些字段已由各解析器规范成数字或数字字符串。
                add(collection, str(row.get(id_field, "")), str(row.get(title_field, "")), int(value), "references_article")

    # 民事要素的 serial_number 本身就是其依据法条号。
    for row in collections.get("civil_elements", []):
        value = str(row.get("serial_number", ""))
        if value.isdigit():
            add("civil_elements", value, str(row.get("title", "")), int(value), "based_on_article")
    return citations


def build_elements(articles: list[dict], interpretations: list[dict], cases: list[dict]) -> list[dict]:
    """以每条民法典为中心，聚合最多三条司法解释和三个相关案例。"""

    # 两个索引都使用法条号作为键。
    interpretation_index: dict[int, list[dict]] = {}
    case_index: dict[int, list[dict]] = {}
    # 一份司法解释可能引用多条法条，因此加入多个索引位置。
    for row in interpretations:
        for value in row.get("related_article_numbers_text", []):
            interpretation_index.setdefault(int(value), []).append(row)
    # 一个案例同样可能关联多条民法典。
    for row in cases:
        for value in row.get("related_article_numbers", []):
            case_index.setdefault(int(value), []).append(row)

    # elements 每条对应一条民法典及其关联材料。
    elements: list[dict] = []
    for article in articles:
        # 当前法条号是聚合键。
        number = int(article["article_number"])
        # 各取前三条，控制记录长度和嵌入成本。
        matched_interpretations = interpretation_index.get(number, [])[:3]
        matched_cases = case_index.get(number, [])[:3]
        # 要素记录中的案例元数据取第一个相关案例作为代表。
        first_case = matched_cases[0] if matched_cases else {}
        # 每条司法解释先组合名称、文号和正文，并限制约 500 token。
        interpretation_texts = [
            _embedding_text(row["law_name"], row.get("document_number", ""), row["content"], max_tokens=500)
            for row in matched_interpretations
        ]
        # 相关案例同样组合标题、案号和摘要。
        case_texts = [
            _embedding_text(row["title"], row.get("case_number", ""), row["summary"], max_tokens=500)
            for row in matched_cases
        ]
        # summary 按“法条 -> 司法解释 -> 案例”组合成一个可直接供回答引用的要素摘要。
        summary = _embedding_text(
            f"【法条】\n{article['law_name']} {article['chapter']}\n{article['article_content']}",
            "【司法解释】" if interpretation_texts else "",
            *[f"【司法解释 {index + 1}】\n{text}" for index, text in enumerate(interpretation_texts) if text],
            "【相关案例】" if case_texts else "",
            *[f"【相关案例 {index + 1}】\n{text}" for index, text in enumerate(case_texts) if text],
            max_tokens=1800,
        )
        # 构造最终要素记录，并保留法条与代表案例的主键关联。
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
    """用少量可解释关键词推断用户上传普通文本的文档类型。"""

    # 同时有“第×条”和法律/法规名称特征时判断为法律文件。
    if re.search(r"第[一二三四五六七八九十百千万零〇两\d]+条", text) and any(word in text for word in ("中华人民共和国", "条例", "办法")):
        return "law"
    # 裁判语言和案号特征优先识别为案例。
    if any(word in text for word in ("法院认为", "案号", "判决如下", "裁定如下")):
        return "case"
    # 合同主体、责任和争议解决等特征识别为合同。
    if any(word in text for word in ("甲方", "乙方", "违约责任", "争议解决", "合同")):
        return "contract"
    # 常见证据载体词识别为证据资料。
    if any(word in text for word in ("证据", "转账记录", "聊天记录", "录音", "照片")):
        return "evidence"
    # 没有明显特征时使用通用类型，不强行猜测。
    return "general"


def detect_document_type(text: str) -> str:
    """兼容旧调用方的文档类型检测名称。"""

    return detect_type(text)


def extract_sections(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    """从“标签：内容”单行结构中提取合同或案件关键字段。"""

    # sections 的键是标签，值是同行内容。
    sections: dict[str, str] = {}
    # 逐行、逐标签匹配，标签数量较少且规则容易理解。
    for line in text.splitlines():
        for label in labels:
            # 同时兼容英文冒号和中文冒号。
            match = re.match(rf"^{re.escape(label)}\s*[:：]\s*(.+)$", line)
            if match:
                sections[label] = match.group(1).strip()
    return sections


def build_metadata(text: str, document_type: str, collection: str | None = None) -> dict:
    """根据文档类型选择关注字段，并生成通用解析元数据。"""

    # 合同重点关注主体、期限、付款、解除、违约和争议解决。
    if document_type == "contract":
        labels = ("合同主体", "合同期限", "标的", "付款", "工资待遇", "解除条件", "违约责任", "争议解决")
    # 案件重点关注法院、当事人、请求、事实、说理、法条和结果。
    elif document_type == "case":
        labels = ("法院", "案号", "原告", "被告", "诉讼请求", "案件事实", "法院认为", "引用法条", "判决结果")
    else:
        # 其他类型暂不做通用标签猜测。
        labels = ()
    # 元数据供分块和上传流程展示。
    return {
        "document_type": document_type,
        "collection": collection or "",
        "char_count": len(text),
        "sections": extract_sections(text, labels),
    }


def _safe_truncate_utf8(text: str, max_bytes: int) -> str:
    """按 UTF-8 字节安全截短，不留下半个中文字符。"""

    value = str(text or "")
    encoded = value.encode("utf-8")
    # 未超限直接返回原文。
    if len(encoded) <= max_bytes:
        return value
    # 截取字节后用 errors=ignore 丢弃末尾可能不完整的一个字符。
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _estimate_tokens(value: str) -> int:
    """用约四字符一个 token 的规则做快速长度估算。"""

    return max(1, len(value) // 4) if value else 0


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """按估算 token 数截短文本，仅用于控制模型输入。"""

    value = str(text or "")
    # 非正上限或空文本都返回空字符串。
    if max_tokens <= 0 or not value:
        return ""
    # 已在预算内时保持原文。
    if _estimate_tokens(value) <= max_tokens:
        return value
    # 先按 4 倍字符数快速截取。
    truncated = value[: max_tokens * 4]
    # 估算仍超限时逐字符缩短到预算内。
    while truncated and _estimate_tokens(truncated) > max_tokens:
        truncated = truncated[:-1]
    return truncated


def _embedding_text(*parts: object, max_tokens: int = 1500) -> str:
    """按顺序组合多个检索字段，并在总 token 预算内保留前面的重要内容。"""

    # segments 保存最终采用的非空文本段。
    segments: list[str] = []
    # remaining_tokens 随每个段落加入而减少。
    remaining_tokens = max_tokens
    # parts 的顺序就是信息优先级，例如标题通常排在正文前。
    for part in parts:
        value = str(part or "").strip()
        # 空段或预算耗尽时跳过。
        if not value or remaining_tokens <= 0:
            continue
        # 估算当前段所需 token。
        tokens = _estimate_tokens(value)
        # 整段能放下就完整保留。
        if tokens <= remaining_tokens:
            segments.append(value)
            remaining_tokens -= tokens
            continue
        # 放不下时只取剩余预算内部分，然后停止处理更低优先级字段。
        truncated = _truncate_to_tokens(value, remaining_tokens)
        if truncated:
            segments.append(truncated)
        break
    # 不同字段用换行分隔，保留可读结构。
    return "\n".join(segments).strip()


def _hash_id(prefix: str, text: str) -> str:
    """由内容生成带业务前缀的稳定主键。"""

    # SHA-1 此处用于去重标识而非密码安全，截取 24 位足够区分项目记录。
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _single_record(document_type: str, text: str, *, collection: str | None = None) -> list[dict]:
    """把一份用户上传的法律、案例或证据文档包装成一条基础记录。"""

    value = str(text or "").strip()
    # 空文本没有记录。
    if not value:
        return []
    # 第一条非空行用作标题。
    first_line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    # 有集合时用集合作 ID 前缀，否则用文档类型。
    prefix = collection or document_type or "record"
    # 内容哈希作主键，正文同时生成限长嵌入文本。
    return [{"id": _hash_id(prefix, f"{prefix}|{value}"), "title": first_line or document_type, "content": value, "embedding_text": _embedding_text(value, max_tokens=1500)}]


def _public_records(collection: str, raw_text: str, source_path: Path | None = None) -> list[dict]:
    """根据公共集合名称，把原始文本分发给对应专用解析器。"""

    # 每个分支对应一种资料格式，不能用同一通用正则混合解析。
    if collection == "civil_code_articles":
        return extract_articles(raw_text)
    if collection == "civil_interpretations":
        return extract_interpretations(raw_text)
    if collection == "civil_cases":
        return extract_cases(raw_text)
    if collection == "civil_elements":
        # 有文件路径时直接读 JSON；否则解析已经传入的 JSON 字符串。
        return load_elements(source_path) if source_path is not None else _load_elements_payload(json.loads(raw_text))
    if collection == "civil_evidence":
        return extract_evidence(raw_text)
    if collection == "civil_processes":
        return extract_processes(raw_text)
    if collection == "civil_questions":
        return extract_questions(raw_text)
    if collection == "civil_citations":
        # 引用集合不是原始文件，而是 build_citations 后续派生。
        return []
    # 未知集合返回空列表，外层质量校验会发现没有有效记录。
    return []


def parse(text: str, document_type: str | None = None, *, collection: str | None = None, source_path: Path | None = None) -> dict:
    """统一解析入口，返回清洗文本、类型、结构化记录和元数据。"""

    # raw_text 保留原始换行和格式，专用公共解析器需要依赖这些边界。
    raw_text = str(text or "")
    # cleaned 用于通用类型检测、用户上传和最终返回。
    cleaned = clean_text(raw_text)
    # 指定 collection 表示公共知识库，必须使用该集合专用解析规则。
    if collection:
        # 集合预设类型优先；未知集合才使用显式类型或自动检测。
        resolved_type = COLLECTION_DOCUMENT_TYPE.get(collection, document_type or detect_type(cleaned))
        records = _public_records(collection, raw_text, source_path)
    else:
        # 用户上传没有集合时，优先尊重调用者明确指定的类型。
        resolved_type = document_type or detect_type(cleaned)
        # 法律文本尝试按法条拆分；没有匹配法条时保留整份单记录。
        if resolved_type == "law":
            records = extract_articles(cleaned) or _single_record(resolved_type, cleaned)
        # 案例和证据作为私有单记录，后续 chunk.py 再按长度处理。
        elif resolved_type in PRIVATE_SINGLE_RECORD_DOCUMENT_TYPES:
            records = _single_record(resolved_type, cleaned)
        else:
            # 合同和通用长文不强行结构化，由语义/父子分块处理全文。
            records = []

    # 构建文档级元数据，并补充本次解析出的记录数量。
    metadata = build_metadata(cleaned, resolved_type, collection)
    metadata["record_count"] = len(records)
    # 返回结构稳定的解析结果，供分块器统一处理。
    return {
        "text": cleaned,
        "document_type": resolved_type,
        "collection": collection or "",
        "records": records,
        "metadata": metadata,
    }


def parse_document(text: str, document_type: str | None = None, *, collection: str | None = None, source_path: Path | None = None) -> dict:
    """更易理解的公共别名，参数原样交给 parse。"""

    return parse(text, document_type=document_type, collection=collection, source_path=source_path)


# 明确本模块对外提供的解析入口和辅助函数。
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
