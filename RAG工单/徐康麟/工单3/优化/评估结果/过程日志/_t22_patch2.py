# -*- coding: utf-8 -*-
"""t22 补丁 2：英文语言闸门的升级阶梯 + 回声答案拒收 + 确定性英文字段框架兜底。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

动机（补丁 1 实测暴露的两个问题）：
    * 重试可能返回**英文问题回声**（EN1 实测得到 `What is the registered capital of …?`，ratio 0.8 却被采纳）→
      加「回声拒收」与「必须有信息量（≥1 个数字或汉字）」两条采纳前置条件；
    * 只靠一次主提示词重试，非字段型/复杂问题仍可能保持中文（EN3/EN4/EN5 实测 keep_chinese）→
      升级阶梯补两级：②「正文提示词（已带英文语言块）」再生成一次；③ 用「问题字段 + 证据取值」确定性渲染
      英文框架句（取值逐字来自证据，含逐字中文源片段）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

GEN = Path(r"E:\gao6gongdan\工单3\研发\app\core\generator.py")

OLD = '''                c_text, c_cites, c_report, c_gate, c_actions = evaluate(retry_text)
                ratio_retry = language_mod.english_ratio(citation_mod.answer_body(c_text))
                adopted = (not citation_mod.is_empty_answer(c_text) and c_report.valid > 0 and c_gate.ok
                           and ratio_retry >= ENGLISH_RATIO_MIN)
                if adopted:
                    text, citations, report, gate, actions = (c_text, c_cites, c_report, c_gate, c_actions)
                else:
                    fallback = field_statement_answer()
                    if fallback:
                        f_text, f_cites, f_report, f_gate, f_actions = evaluate(fallback)
                        f_ratio = language_mod.english_ratio(citation_mod.answer_body(f_text))
                        if (not citation_mod.is_empty_answer(f_text) and f_report.valid > 0 and f_gate.ok
                                and f_ratio >= ENGLISH_RATIO_MIN):
                            text, citations, report, gate, actions = (f_text, f_cites, f_report, f_gate,
                                                                       f_actions)
                            adopted, ratio_retry = True, f_ratio
                ratio_after = language_mod.english_ratio(citation_mod.answer_body(text))
                log.log_event("generation.language_gate", level="WARNING", trace_id=trace,
                              path=("retry" if adopted and ratio_after != ratio_before else "keep_chinese"),
                              adopted=adopted, ratio_before=ratio_before, ratio_after=ratio_after,
                              threshold=ENGLISH_RATIO_MIN)
'''

NEW = '''                def _english_ok(candidate: str) -> bool:
                    """英文候选是否可采纳：非空、非「问题回声」、且含信息量（≥1 个数字或汉字）。

                    实测动机（§25）：主提示词重试曾返回**英文问题回声**（把问句原样抄回），ratio 0.8 却
                    毫无信息量；而只含英文实体名的句子同样不算答案。中文专名/数字是招股书答案的必要成分，
                    因此要求候选至少含一个数字或一个汉字。
                    """
                    body = citation_mod.answer_body(candidate)
                    if not body.strip():
                        return False
                    if text_utils.squash_text(body) in text_utils.squash_text(question):
                        return False
                    return bool(re.search(r"[0-9]|[\\u4e00-\\u9fff]", body))

                adopted, ratio_retry, path_used = False, 0.0, "keep_chinese"
                c_text, c_cites, c_report, c_gate, c_actions = evaluate(retry_text)
                ratio_retry = language_mod.english_ratio(citation_mod.answer_body(c_text))
                if (not citation_mod.is_empty_answer(c_text) and c_report.valid > 0 and c_gate.ok
                        and ratio_retry >= ENGLISH_RATIO_MIN and _english_ok(c_text)):
                    text, citations, report, gate, actions = (c_text, c_cites, c_report, c_gate, c_actions)
                    adopted, path_used = True, "retry"
                if not adopted:
                    # 第二级：正文提示词（已带英文语言块）再生成一次 —— 不带引用格式负担，模型更容易照做
                    body_only = self._generate_body(question, chunks, lang, trace, log)
                    if body_only:
                        b_text, b_cites, b_report, b_gate, b_actions = evaluate(body_only)
                        b_ratio = language_mod.english_ratio(citation_mod.answer_body(b_text))
                        if (not citation_mod.is_empty_answer(b_text) and b_report.valid > 0 and b_gate.ok
                                and b_ratio >= ENGLISH_RATIO_MIN and _english_ok(b_text)):
                            text, citations, report, gate, actions = (b_text, b_cites, b_report, b_gate,
                                                                      b_actions)
                            adopted, ratio_retry, path_used = True, b_ratio, "body_stage"
                if not adopted:
                    # 第三级：确定性英文字段框架（取值逐字来自证据 + 逐字中文源片段，绝不新增事实）
                    from .query_understanding import FIELD_KEYWORDS as _FK_EN

                    pool_en: list[Any] = []
                    for cite in citations:
                        chunk = self.chunk_lookup(str(getattr(cite, "chunk_id", "") or "")) \\
                            if getattr(cite, "chunk_id", None) else None
                        if chunk is not None:
                            pool_en.append(chunk)
                    pool_en += list(support_pool) + list(chunks)
                    en_candidate = ""
                    for keyword in _FK_EN.get(str(info.get("field_type") or ""), ()):
                        if keyword not in question:
                            continue
                        for chunk in pool_en:
                            value, _style = _extract_field_value(
                                str(getattr(chunk, "content", "") or ""), keyword,
                                expects_numeric=bool(info.get("expects_numeric")))
                            if not value:
                                continue
                            en_cite = citation_mod.format_citation(str(getattr(chunk, "file_name", "")),
                                                                  int(getattr(chunk, "page", 0)))
                            en_candidate = (f"{language_mod.english_frame(keyword, value)} "
                                            f"(verbatim source: 「{keyword}：{value}」)\\n引用：{en_cite}")
                            break
                        if en_candidate:
                            break
                    if not en_candidate:
                        fallback = field_statement_answer()
                        en_candidate = fallback
                    if en_candidate:
                        f_text, f_cites, f_report, f_gate, f_actions = evaluate(en_candidate)
                        f_ratio = language_mod.english_ratio(citation_mod.answer_body(f_text))
                        if (not citation_mod.is_empty_answer(f_text) and f_report.valid > 0 and f_gate.ok
                                and f_ratio >= ENGLISH_RATIO_MIN and _english_ok(f_text)):
                            text, citations, report, gate, actions = (f_text, f_cites, f_report, f_gate,
                                                                      f_actions)
                            adopted, ratio_retry, path_used = True, f_ratio, "deterministic_frame"
                ratio_after = language_mod.english_ratio(citation_mod.answer_body(text))
                log.log_event("generation.language_gate", level="WARNING", trace_id=trace,
                              path=path_used, adopted=adopted, ratio_before=ratio_before,
                              ratio_after=ratio_after, threshold=ENGLISH_RATIO_MIN)
'''


def main() -> int:
    """入口：替换语言闸门内部实现（精确匹配断言，失败不写盘）。"""
    raw = GEN.read_text(encoding="utf-8")
    had_crlf = "\r\n" in raw
    body = raw.replace("\r\n", "\n")
    old, new = OLD.replace("\r\n", "\n").rstrip("\n"), NEW.replace("\r\n", "\n").rstrip("\n")
    if body.count(old) != 1:
        raise SystemExit(f"锚点命中 {body.count(old)} 次（期望 1），已中止，未写盘")
    body = body.replace(old, new)
    GEN.write_text(body.replace("\n", "\r\n") if had_crlf else body, encoding="utf-8")
    print(f"[generator.py] 语言闸门已升级；sha256={hashlib.sha256(GEN.read_bytes()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
