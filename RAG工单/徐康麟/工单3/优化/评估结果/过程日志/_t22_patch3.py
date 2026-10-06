# -*- coding: utf-8 -*-
"""t22 补丁 3：① 英文问题的字段关键词来自「跨语言主题映射」（否则 field_type=other 无词可用）；
② 采纳前置拒收「问题回声」（初始答案把问句原样抄回，ratio 高但无信息量）；
③ 主题映射表补齐「发行股数 / 注册地址 / 实际控制人」等字段词。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

实测依据（补丁 2 的英文复测）：
    * EN1 的**初始**答案就是英文问题回声（ratio 0.735 ≥ 0.5 → 语言闸门根本没触发）→ 需在级联早期就拒收；
    * EN4/EN5 的 `classify_question()` 把英文问句判成 ``other``（分类器是中文词表）→ 确定性英文框架拿不到字段词
      → 改从 `language.match_corpus_topic_terms()`（英文短语→中文语料词）取词。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(r"E:\gao6gongdan\工单3")
GEN = REPO / "研发/app/core/generator.py"
LANG = REPO / "研发/app/core/language.py"

GEN_EDITS: list[tuple[str, str, int]] = [
    # ① 拒收「问题回声」：squash 后是问句的子串且长度 ≥ 问句的 60% → 视为无信息量的回声，继续走级联
    (
        '        def need_more() -> bool:\n'
        '            """还缺「可校验的答案」：正文空 或 没有任何合法引用。"""\n'
        '            return citation_mod.is_empty_answer(text) or report.valid == 0',
        '        def is_question_echo(candidate: str) -> bool:\n'
        '            """答案是否是「问题回声」（t22，§25）：把问句原样抄回，ratio 高但毫无信息量。\n'
        '\n'
        '            判据（严格，避免误伤正常短答）：``squash_text(答案)`` 是 ``squash_text(问题)`` 的子串\n'
        '            且长度 ≥ 问题去标点后的 60%。实测动机：英文提问下小模型先回了\n'
        '            `What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?`（ratio 0.735）\n'
        '            并因问句里的公司名与证据逐字重合而通过了引用校验 → 必须在此处拒收，交给后续级联。\n'
        '            """\n'
        '            body = text_utils.squash_text(citation_mod.answer_body(candidate))\n'
        '            question_squashed = text_utils.squash_text(question)\n'
        '            return bool(body) and body in question_squashed and len(body) >= 0.6 * len(question_squashed)\n'
        '\n'
        '        def need_more() -> bool:\n'
        '            """还缺「可校验的答案」：正文空 或 没有任何合法引用 或 只是问题回声。"""\n'
        '            return (citation_mod.is_empty_answer(text) or report.valid == 0\n'
        '                    or is_question_echo(text))',
        1,
    ),
    # ② 确定性英文框架：字段词来源 = FIELD_KEYWORDS ∪ 跨语言主题映射（英文问句 field_type 常为 other）
    (
        '                    pool_en += list(support_pool) + list(chunks)\n'
        '                    en_candidate = ""\n'
        '                    for keyword in _FK_EN.get(str(info.get("field_type") or ""), ()):\n'
        '                        if keyword not in question:\n'
        '                            continue',
        '                    pool_en += list(support_pool) + list(chunks)\n'
        '                    en_candidate = ""\n'
        '                    # 英文字段词来源：① 中文分类命中的字段词；② **跨语言主题映射**（英文问句常被判成\n'
        '                    # ``other``，因为分类器是中文词表）—— 例如 registered capital → 注册资本 / 股本。\n'
        '                    en_keywords: list[str] = []\n'
        '                    for keyword in _FK_EN.get(str(info.get("field_type") or ""), ()):\n'
        '                        if keyword in question and keyword not in en_keywords:\n'
        '                            en_keywords.append(keyword)\n'
        '                    for _phrase, _terms in language_mod.match_corpus_topic_terms(question):\n'
        '                        for _term in _terms:\n'
        '                            if _term not in en_keywords:\n'
        '                                en_keywords.append(_term)\n'
        '                    for keyword in en_keywords:',
        1,
    ),
]


def main() -> int:
    """入口：应用语言表补齐 + generator 两处加固。"""
    # ③ 主题映射补齐字段型词（通用领域词，非单题特判）
    raw = LANG.read_text(encoding="utf-8")
    additions = {
        '"shares": ("股份", "发行"),': '"shares": ("股份", "发行", "发行股数", "发行数量"),',
        '"registered address": ("注册地址", "注册地"),': '"registered address": ("注册地址", "注册地"),\n'
        '    "registered office": ("注册地址", "注册地"),',
    }
    for old, new in additions.items():
        if old not in raw:
            raise SystemExit(f"language.py 锚点缺失：{old}")
        raw = raw.replace(old, new, 1)
    LANG.write_text(raw, encoding="utf-8")
    print(f"[language.py] 主题映射已补齐；sha256={hashlib.sha256(LANG.read_bytes()).hexdigest()[:16]}")

    text = GEN.read_text(encoding="utf-8")
    had_crlf = "\r\n" in text
    body = text.replace("\r\n", "\n")
    for index, (old, new, expect) in enumerate(GEN_EDITS, start=1):
        anchor, replacement = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
        count = body.count(anchor)
        if count != expect:
            raise SystemExit(f"generator.py 锚点 #{index} 命中 {count} 次（期望 {expect}），已中止，未写盘")
        body = body.replace(anchor, replacement)
    GEN.write_text(body.replace("\n", "\r\n") if had_crlf else body, encoding="utf-8")
    print(f"[generator.py] 已应用 {len(GEN_EDITS)} 处；sha256={hashlib.sha256(GEN.read_bytes()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
