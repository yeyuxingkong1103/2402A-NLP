# -*- coding: utf-8 -*-
"""t22 补丁 4（限缩范围，captain 授权）：① 跨行取值修复；② 问题回声在**最终采纳层**拦截。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

① 跨行取值（**通用缺陷，中文侧同样受益**）
   实测：`招股说明书2.pdf` 把数值在行内折断（`注册资本：人民币5,\n000万元`），
   而 `_extract_field_value()` 按行匹配且「遇空白即停」→ 取到截断值 `人民币5,`。
   修法（两步，均不改判据强度）：
       a. 把「数字 + 逗号 + 空白 + 三位数字」重新拼回（只影响被折断的千分位数值）；
       b. 除「按行」视图外，再加一遍「整段折空白」视图（表格单元格分支仍按行处理）。
② 问题回声最终拦截
   实测：英文问句的首个 LLM 答案会把问句抄回（`What is the registered capital of …?`），
   并在后续级别的证据片段追加后**又被拼回最终答案** → 只在 `need_more()` 层拒收不够，
   必须在**最终写入口**（`render_answer` 之前）再拦一次；拦截后先试确定性字段句，
   仍为回声则按「证据不足」回「不清楚」（回声不是答案，宁可拒答也不给无信息量输出）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

GEN = Path(r"E:\gao6gongdan\工单3\研发\app\core\generator.py")

# --- ① 跨行取值：在 _extract_field_value 里加「修复后的整段视图」 -------------------------
OLD_1 = '''    raw = str(text or "")
    if not raw or not keyword:
        return "", ""
'''
NEW_1 = '''    raw = str(text or "")
    if not raw or not keyword:
        return "", ""
    # t22（§25）：**跨行数值修复** —— PDF 常把千分位数值在行内折断（实测 招股说明书2.pdf：
    # 「注册资本：人民币5,\\n000万元」）；先把「数字, + 空白 + 三位数字」拼回，再对「整段折空白」
    # 视图补抽一遍（表格单元格分支仍按行处理）。只修复字面量，不改变任何判据强度。
    raw_repaired = re.sub(r"(\\d),\\s+(\\d{3})", r"\\1,\\2", raw)
    views: list[str] = [raw_repaired]
    if "\\n" in raw_repaired or "\\r" in raw_repaired:
        views.append(re.sub(r"\\s+", " ", raw_repaired))
'''
OLD_2 = '''    copula = re.compile(_FIELD_COPULA_TEMPLATE.format(kw=re.escape(keyword)))
    separator = re.compile(_FIELD_SEPARATOR_TEMPLATE.format(kw=re.escape(keyword)))
    for line in raw.splitlines():'''
NEW_2 = '''    copula = re.compile(_FIELD_COPULA_TEMPLATE.format(kw=re.escape(keyword)))
    separator = re.compile(_FIELD_SEPARATOR_TEMPLATE.format(kw=re.escape(keyword)))
    for line in [part for view in views for part in (view.splitlines() or [view])]:'''
# 表格单元格分支：取值本身也可能被折断 → 同样做一次数值拼接再校验
OLD_3 = '''                for value_cell in cells[index + 1:]:
                    if not value_cell or value_cell in _FIELD_EMPTY_CELLS:
                        continue
                    if _field_value_ok(value_cell, keyword, expects_numeric=expects_numeric):
                        return value_cell, "separator"
                    break'''
NEW_3 = '''                for value_cell in cells[index + 1:]:
                    if not value_cell or value_cell in _FIELD_EMPTY_CELLS:
                        continue
                    # t22：单元格内的数值同样可能被折断（「人民币5, 000万元」）→ 拼接后再判
                    value_cell = re.sub(r"(\\d),\\s+(\\d{3})", r"\\1,\\2", value_cell)
                    if _field_value_ok(value_cell, keyword, expects_numeric=expects_numeric):
                        return value_cell, "separator"
                    break'''

# --- ② 回声最终拦截（放在 render_answer 之前） ----------------------------------------
OLD_4 = '''        # 只保留通过校验的引用（引用可回溯 100%），并据此重建答案文本'''
NEW_4 = '''        # 阶段⑭：**问题回声最终拦截**（t22，§25）—— 判断层（need_more）拒收不足以防回退：
        # 实测英文问句的首个答案会把问句原样抄回，并在后续级别的证据片段追加后又被拼进最终文本。
        # 这里在**最终写入口**再拦一次：回声答案先换确定性字段句；仍为回声则按「证据不足」回「不清楚」
        # —— 回声不是答案，宁可拒答也不输出无信息量文本（不编造原则）。
        echo_blocked = is_question_echo(text)
        if echo_blocked:
            replacement_text = field_statement_answer()
            if replacement_text:
                r_text, r_cites, r_report, r_gate, r_actions = evaluate(replacement_text)
                if (not citation_mod.is_empty_answer(r_text) and r_report.valid > 0 and r_gate.ok
                        and not is_question_echo(r_text)):
                    text, citations, report, gate, actions = (r_text, r_cites, r_report, r_gate, r_actions)
                    log.log_event("generation.echo_answer_replaced", level="WARNING", trace_id=trace,
                                  chars=len(citation_mod.answer_body(r_text)),
                                  citations=[c.render(language=lang) for c in r_cites])
        if is_question_echo(text):
            log.log_event("generation.echo_answer_blocked", level="ERROR", trace_id=trace,
                          body_digest=text_digest(citation_mod.answer_body(text), limit=60),
                          degrade="回声答案不采纳 → 按证据不足回「不清楚」")
            return self._unknown(question, retrieval, reason="insufficient_evidence", backend=backend,
                                 model=model, language=lang, trace_id=trace, first_token_ms=first_ms,
                                 total_ms=round((time.perf_counter() - started) * 1000, 2))
        # 只保留通过校验的引用（引用可回溯 100%），并据此重建答案文本'''

EDITS = [(OLD_1, NEW_1, 1), (OLD_2, NEW_2, 1), (OLD_3, NEW_3, 1), (OLD_4, NEW_4, 1)]


def main() -> int:
    """入口：应用两处修复（精确匹配断言，失败不写盘）。"""
    text = GEN.read_text(encoding="utf-8")
    had_crlf = "\r\n" in text
    body = text.replace("\r\n", "\n")
    for index, (old, new, expect) in enumerate(EDITS, start=1):
        anchor, replacement = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
        count = body.count(anchor)
        if count != expect:
            raise SystemExit(f"锚点 #{index} 命中 {count} 次（期望 {expect}），已中止，未写盘")
        body = body.replace(anchor, replacement)
    GEN.write_text(body.replace("\n", "\r\n") if had_crlf else body, encoding="utf-8")
    print(f"[generator.py] 补丁 4 已应用 {len(EDITS)} 处；sha256={hashlib.sha256(GEN.read_bytes()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
