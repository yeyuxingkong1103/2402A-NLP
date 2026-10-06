# -*- coding: utf-8 -*-
"""t22 施工脚本（一次性、可审计）：英文作答的语言适配层。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

设计（captain 已授权解冻 `generator.py`，改动全部限制在 `lang == "en"` 分支内）：
    * `language.py`：新增提示词语言块（zh 分支文本与现状**逐字符相同**）、英文字段标签表、拉丁占比度量、英文框架句；
    * `prompts/qa_prompt.txt`：规则 6 改为 `{language_rule}` 占位符（zh 渲染结果不变）；
    * `prompts/qa_body_prompt.txt`：在「要求：」后插入 `{language_rule}`（zh 传空串 → 渲染结果不变）；
    * `generator.py`：
        ① 两处提示词渲染传入 `language_rule`；
        ② 字段型成句化（阶段⑩）在 `en` 下输出「英文框架句 + 逐字中文原文片段」；
        ③ 数值锚定摘录（阶段⑥）在 `en` 下同样成句化（否则会倾倒中文原句）；
        ④ 募投清单 / 关联方表行兜底在 `en` 下改用英文引导句（项目名/金额保持原文）；
        ⑤ 末尾新增**英文语言闸门**：ratio < 0.5 → 强约束重试一次 → 再退确定性英文框架 → 都不可得则保留中文并记
           `generation.language_degraded`（WARNING），绝不因语言不合规而变成「不清楚」。

本脚本对每个锚点做**精确匹配断言**，任一锚点不匹配即中止且不写盘（避免半成品）。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(r"E:\gao6gongdan\工单3")
LANG = REPO / "研发/app/core/language.py"
PROMPT = REPO / "研发/app/prompts/qa_prompt.txt"
BODY_PROMPT = REPO / "研发/app/prompts/qa_body_prompt.txt"
GEN = REPO / "研发/app/core/generator.py"

LANG_ADD = '''

# ---------------------------------------------------------------------------
# t22（§25）：英文作答支撑（提示词语言块 / 英文字段标签 / 拉丁占比 / 英文框架句）
# ---------------------------------------------------------------------------
# 提示词语言块：**zh 分支的文本必须与改造前 `qa_prompt.txt` 规则 6 的渲染结果逐字符相同**
# （该文件是中英共用文件；只有 en 分支才换成强约束英文指令）。
LANGUAGE_RULE_ZH = "`zh` 为 `en` 时用英文作答（引用写 `[Page: N]`），否则中文作答"
LANGUAGE_RULE_EN = ("**Must answer in English.** Write the answer as English sentences; keep ONLY proper nouns "
                    "(company/person names), numbers, units and any verbatim Chinese source fragment in Chinese; "
                    "never copy a whole Chinese sentence as the answer; citations use `[Page: N]`.")

# 英文字段标签（招股书常见字段；用于把「字段+取值」渲染成英文框架句）。键为中文字段词（与
# `query_understanding.FIELD_KEYWORDS` 同源词），值是通用英文标签——不针对任何具体问题写死答案。
EN_FIELD_LABELS: dict[str, str] = {
    "注册资本": "registered capital", "股本总额": "total share capital", "法定代表人": "legal representative",
    "注册地址": "registered address", "成立日期": "date of establishment", "发行股数": "number of shares issued",
    "发行数量": "number of shares issued", "持股比例": "shareholding percentage", "营业收入": "operating revenue",
    "收入": "revenue", "募集资金": "raised funds", "补充流动资金": "supplementary working capital",
    "关联方": "related parties", "控股股东": "controlling shareholder", "实际控制人": "actual controller",
    "员工人数": "number of employees", "主营业务": "main business", "重要供应商": "important supplier",
    "供应商": "supplier", "客户": "customer", "上游": "upstream industry", "下游": "downstream industry",
    "技术标准": "technical standard", "科技进步奖": "National Science and Technology Progress Award",
    "证券代码": "stock code", "股票代码": "stock code",
}


def english_ratio(text: str) -> float:
    """文本里拉丁字母占比（t22 英文语言闸门的度量；分母为去首尾空白后的总字符数）。"""
    body = str(text or "").strip()
    if not body:
        return 0.0
    letters = sum(1 for char in body if ("a" <= char.lower() <= "z"))
    return round(letters / len(body), 4)


def english_frame(field_zh: str, value_zh: str) -> str:
    """把「字段 + 取值」渲染成英文框架句，取值逐字保留（专名/金额/单位按原文）。

    找不到字段标签时退化为 ``The value is …``（仍是英文句，且不新增事实）。
    """
    label = EN_FIELD_LABELS.get(str(field_zh).strip())
    value = str(value_zh).strip()
    return f"The {label} is {value}." if label else f"The value is {value}."
'''

GEN_EDITS: list[tuple[str, str, int]] = [
    # ① 主提示词渲染：传入 language_rule（zh 分支文本不变）
    (
        '            prompt = template.format(question=question, context="\\n\\n".join(lines),\n'
        '                                     language=language, history=history_text)',
        '            prompt = template.format(question=question, context="\\n\\n".join(lines),\n'
        '                                     language=language,\n'
        '                                     language_rule=(language_mod.LANGUAGE_RULE_EN\n'
        '                                                    if str(language).lower().startswith("en")\n'
        '                                                    else language_mod.LANGUAGE_RULE_ZH),\n'
        '                                     history=history_text)',
        1,
    ),
    # ② 正文/强模型提示词渲染：同样传 language_rule（两处相同文本）
    (
        '        prompt = template.format(question=question, context="\\n\\n".join(lines))',
        '        prompt = template.format(question=question, context="\\n\\n".join(lines),\n'
        '                                 language_rule=(language_mod.LANGUAGE_RULE_EN\n'
        '                                                if str(lang).lower().startswith("en")\n'
        '                                                else ""))',
        2,
    ),
    # ③ 字段型成句化（阶段⑩）：en → 英文框架句 + 逐字中文原文片段
    (
        '                    copula = "是" if _WHO_QUESTION_PATTERN.search(question) else "为"',
        '                    copula = "是" if _WHO_QUESTION_PATTERN.search(question) else "为"\n'
        '                    # t22（§25）：英文提问改用英文框架句；逐字中文片段必须保留 —— 引用校验要求\n'
        '                    # 答案里存在与证据逐字一致、≥6 字（含数值时 ≥4 字）的连续片段（人名类仅 3 字，\n'
        '                    # 只靠英文句会判「引用不在引用处」）。\n'
        '                    if str(lang).lower().startswith("en"):\n'
        '                        log.log_event("generation.field_statement_english", level="WARNING",\n'
        '                                      trace_id=trace, keyword=keyword, value=value,\n'
        '                                      frame=language_mod.english_frame(keyword, value))\n'
        '                        return (f"{language_mod.english_frame(keyword, value)} "\n'
        '                                f"(verbatim source: 「{keyword}：{value}」)\\n引用：{cite_text}")',
        1,
    ),
    # ④ 数值锚定摘录（阶段⑥）：en → 成句化，不倾倒中文原句
    (
        '            if anchors and not (answer_numbers_in_answer() & anchors):\n'
        '                snippet, chunk = find_anchor_sentence(support_pool + list(chunks))\n'
        '                if snippet is not None and chunk is not None:\n'
        '                    cite_text = citation_mod.format_citation(str(getattr(chunk, "file_name", "")),\n'
        '                                                            int(getattr(chunk, "page", 0)))\n'
        '                    candidate_text, candidate_cites, candidate_report, candidate_gate, candidate_actions = \\\n'
        '                        evaluate(f"{snippet}\\n引用：{cite_text}")',
        '            if anchors and not (answer_numbers_in_answer() & anchors):\n'
        '                snippet, chunk = find_anchor_sentence(support_pool + list(chunks))\n'
        '                if snippet is not None and chunk is not None:\n'
        '                    cite_text = citation_mod.format_citation(str(getattr(chunk, "file_name", "")),\n'
        '                                                            int(getattr(chunk, "page", 0)))\n'
        '                    # t22（§25）：英文提问下把摘录句渲染成英文框架句（字段+取值逐字保留），\n'
        '                    # 否则会直接把中文原句当答案正文（实测 EN5「发行股数」曾输出 `| 发行股数 | 1,670万股 |`）。\n'
        '                    anchor_text = f"{snippet}\\n引用：{cite_text}"\n'
        '                    if str(lang).lower().startswith("en"):\n'
        '                        from .query_understanding import FIELD_KEYWORDS as _FK\n'
        '                        _kws = list(_FK.get(str(info.get("field_type") or ""), ()))\n'
        '                        _field, _value = "", ""\n'
        '                        for _kw in _kws:\n'
        '                            _v, _style = _extract_field_value(snippet, _kw, expects_numeric=True)\n'
        '                            if _v:\n'
        '                                _field, _value = _kw, _v\n'
        '                                break\n'
        '                        if _field and _value:\n'
        '                            anchor_text = (f"{language_mod.english_frame(_field, _value)} "\n'
        '                                           f"(verbatim source: 「{_field}：{_value}」)\\n引用：{cite_text}")\n'
        '                        else:\n'
        '                            anchor_text = (f"According to the prospectus: 「{snippet}」\\n引用：{cite_text}")\n'
        '                    candidate_text, candidate_cites, candidate_report, candidate_gate, candidate_actions = \\\n'
        '                        evaluate(anchor_text)',
        1,
    ),
    # ⑤ 募投清单兜底：en → 英文引导句
    (
        '                return "本次募集资金拟投资项目包括：" + "、".join(items) + f"\\n引用：{cite_text}"',
        '                lead = ("The projects to be funded by the raised funds include: "\n'
        '                        if str(lang).lower().startswith("en") else "本次募集资金拟投资项目包括：")\n'
        '                return lead + "、".join(items) + f"\\n引用：{cite_text}"',
        1,
    ),
    # ⑥ 关联方表行兜底：en → 英文引导句
    (
        '                body = "不存在控制关系的关联方企业包括：" + "、".join(',
        '                lead = ("The related-party enterprises with which the issuer has no controlling relationship "\n'
        '                        "include: " if str(lang).lower().startswith("en") else "不存在控制关系的关联方企业包括：")\n'
        '                body = lead + "、".join(',
        1,
    ),
    # ⑦ 英文语言闸门（放在最终引用重建之前）
    (
        '        # 只保留通过校验的引用（引用可回溯 100%），并据此重建答案文本',
        '        # 阶段⑬：英文语言闸门（t22，§25）—— 英文提问的正文必须以英文为主。\n'
        '        # 顺序（captain 已确认）：强约束重试一次 → 确定性英文框架答案 → 都不可得则**保留中文**并记\n'
        '        # `generation.language_degraded`（WARNING）。语言不合规**绝不**降级成「不清楚」（宁可语言不合格，\n'
        '        # 也不能丢答案或编造）。中文分支不进入该阶段。\n'
        '        if str(lang).lower().startswith("en"):\n'
        '            ratio_before = language_mod.english_ratio(citation_mod.answer_body(text))\n'
        '            if ratio_before < ENGLISH_RATIO_MIN:\n'
        '                retry_text = self._generate_text(\n'
        '                    prompt + "\\n\\n【修正要求】The user asked in English: rewrite the answer in ENGLISH "\n'
        '                             "sentences only. Keep company/person names, numbers, units and any verbatim "\n'
        '                             "Chinese source fragment as-is; do not copy whole Chinese sentences. "\n'
        '                             "Output the English answer, then the reference line.",\n'
        '                    question, chunks, lang, trace, log)[0]\n'
        '                c_text, c_cites, c_report, c_gate, c_actions = evaluate(retry_text)\n'
        '                ratio_retry = language_mod.english_ratio(citation_mod.answer_body(c_text))\n'
        '                adopted = (not citation_mod.is_empty_answer(c_text) and c_report.valid > 0 and c_gate.ok\n'
        '                           and ratio_retry >= ENGLISH_RATIO_MIN)\n'
        '                if adopted:\n'
        '                    text, citations, report, gate, actions = (c_text, c_cites, c_report, c_gate, c_actions)\n'
        '                else:\n'
        '                    fallback = field_statement_answer()\n'
        '                    if fallback:\n'
        '                        f_text, f_cites, f_report, f_gate, f_actions = evaluate(fallback)\n'
        '                        f_ratio = language_mod.english_ratio(citation_mod.answer_body(f_text))\n'
        '                        if (not citation_mod.is_empty_answer(f_text) and f_report.valid > 0 and f_gate.ok\n'
        '                                and f_ratio >= ENGLISH_RATIO_MIN):\n'
        '                            text, citations, report, gate, actions = (f_text, f_cites, f_report, f_gate,\n'
        '                                                                       f_actions)\n'
        '                            adopted, ratio_retry = True, f_ratio\n'
        '                ratio_after = language_mod.english_ratio(citation_mod.answer_body(text))\n'
        '                log.log_event("generation.language_gate", level="WARNING", trace_id=trace,\n'
        '                              path=("retry" if adopted and ratio_after != ratio_before else "keep_chinese"),\n'
        '                              adopted=adopted, ratio_before=ratio_before, ratio_after=ratio_after,\n'
        '                              threshold=ENGLISH_RATIO_MIN)\n'
        '                if not adopted:\n'
        '                    log.log_event("generation.language_degraded", level="WARNING", trace_id=trace,\n'
        '                                  ratio=ratio_after, threshold=ENGLISH_RATIO_MIN,\n'
        '                                  body_digest=text_digest(citation_mod.answer_body(text), limit=60),\n'
        '                                  note="英文提问未能生成英文正文：保留中文答案（不编造、不降级为不清楚）")\n'
        '        # 只保留通过校验的引用（引用可回溯 100%），并据此重建答案文本',
        1,
    ),
]


def normalize(text: str) -> str:
    """统一成 LF 便于精确匹配。"""
    return text.replace("\r\n", "\n")


def restore_eol(text: str, had_crlf: bool) -> str:
    """按原文件的换行风格写回（冻结文件避免整文件 EOL 变化）。"""
    return text.replace("\n", "\r\n") if had_crlf else text


def patch(path: Path, edits: list[tuple[str, str, int]], *, allow_zero: bool = False) -> None:
    """按 (old, new, 期望出现次数) 逐个替换；任一断言失败即抛异常（不写盘）。"""
    raw = path.read_text(encoding="utf-8")
    had_crlf = "\r\n" in raw
    body = normalize(raw)
    for index, (old, new, expect) in enumerate(edits, start=1):
        found = normalize(old)
        count = body.count(found)
        if count != expect and not (allow_zero and count == 0):
            raise SystemExit(f"[{path.name}] 锚点 #{index} 命中 {count} 次（期望 {expect}），已中止，未写盘")
        body = body.replace(found, normalize(new))
    path.write_text(restore_eol(body, had_crlf), encoding="utf-8")
    print(f"[{path.name}] 已应用 {len(edits)} 处；sha256={hashlib.sha256(path.read_bytes()).hexdigest()[:16]}")


def main() -> int:
    """入口：语言块 + 提示词占位符 + generator 语言适配。"""
    raw_lang = LANG.read_text(encoding="utf-8")
    if "LANGUAGE_RULE_EN" in raw_lang:
        raise SystemExit("language.py 已包含 t22 语言块，疑似重复执行 → 中止")
    LANG.write_text(raw_lang + LANG_ADD, encoding="utf-8")
    print(f"[language.py] 已追加语言块；sha256={hashlib.sha256(LANG.read_bytes()).hexdigest()[:16]}")

    patch(PROMPT, [(
        "6. `{language}` 为 `en` 时用英文作答（引用写 `[Page: N]`），否则中文作答",
        "6. {language_rule}",
        1,
    )])
    patch(BODY_PROMPT, [(
        "要求：只输出答案本身",
        "要求：{language_rule}只输出答案本身",
        1,
    )])
    # ENGLISH_RATIO_MIN 常量（模块级；阈值依据见 §25：英文框架句实测 ≥0.55，中文摘录 ≤0.42）
    if "ENGLISH_RATIO_MIN" not in GEN.read_text(encoding="utf-8"):
        patch(GEN, [(
            '_UNIT_NUMBER_PATTERN = re.compile(r"\\d[\\d,，.]*\\s*(?:万元|亿元|元|万股|%|％|次|倍)")',
            '_UNIT_NUMBER_PATTERN = re.compile(r"\\d[\\d,，.]*\\s*(?:万元|亿元|元|万股|%|％|次|倍)")\n'
            '# t22（§25）：英文提问的「正文以英文为主」判定阈值。依据（实测，非调参）：\n'
            '#   修前中文摘录 0.00~0.42；英文框架句（含中文专名/单位/逐字片段）实测 0.55~0.78。\n'
            'ENGLISH_RATIO_MIN = 0.5',
            1,
        )])
    patch(GEN, GEN_EDITS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
