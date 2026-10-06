# -*- coding: utf-8 -*-
"""t22 补丁 8（定点修复，一处）：英文候选的「回声」判据与最终拦截层统一。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

现象（实测 trace `q14670912`，`部署/日志/rag_trace.jsonl`）：
    EN1（`What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?`）在语言闸门里
    `path=retry, adopted=true, ratio_before=0.2119, ratio_after=0.7351` —— 重试确实把英文占比抬到阈值之上，
    但该重试文本是**「问句回声 + 追加证据片段」**：
    `What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?  Wuhan Xingtu Xinke…`
    （问句是答案的**前缀**，整句不再是问句子串）→ `_english_ok()` 放行；到阶段⑭才被
    `is_question_echo()` 认出 → `generation.echo_answer_blocked` → 回「不清楚」。
    **后果：一条本可答的英文正例被误拒答**（英文侧硬要求「正例不得误拒」）。

修法（一处，仅 en 分支；不改判据阈值）：
    `_english_ok()` 内把「答案是否为问句子串」的判断换成与阶段⑭**同一个函数** `is_question_echo(candidate)`，
    使「问句作为前缀的回声」在采纳点即被拒收，级联继续走正文提示词/确定性英文框架等后续路径。

实测（`_t22_probe_en1.py EN1 3`）：修后 3/3 稳定 `path=body_stage`、ratio=0.6829、`is_unknown=false`、引用 `[招股说明书1.pdf: 22]`。

本脚本幂等：若目标文本已在位，则只做校验并打印「已应用」，不重复写盘。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

GEN = Path(r"E:\gao6gongdan\工单3\研发\app\core\generator.py")

OLD = '''                    body = citation_mod.answer_body(candidate)
                    if not body.strip():
                        return False
                    if text_utils.squash_text(body) in text_utils.squash_text(question):
                        return False
                    return bool(re.search(r"[0-9]|[\\u4e00-\\u9fff]", body))
'''

NEW = '''                    body = citation_mod.answer_body(candidate)
                    if not body.strip():
                        return False
                    # t22 补丁 8：判据必须与最终拦截层（阶段⑭）**同一函数**。实测该层此前只判「答案是否
                    # 问句子串」，漏掉了「问句本身作为答案前缀、后面再追加证据片段」的回声形态
                    # （`What is the registered capital…? Wuhan Xingtu Xinke…`：ratio 0.7351、引用校验通过
                    # → 被英文闸门采纳 → 到阶段⑭才被拦成「不清楚」，把**本可答的英文正例**变成拒答）。
                    # 改用 is_question_echo() 后，回声在采纳点即被拒收，级联继续走确定性英文框架。
                    if is_question_echo(candidate):
                        return False
                    return bool(re.search(r"[0-9]|[\\u4e00-\\u9fff]", body))
'''


def main() -> int:
    """入口：写入或校验「采纳点与拦截层同判据」这一处改动。"""
    text = GEN.read_text(encoding="utf-8")
    had_crlf = "\r\n" in text
    body = text.replace("\r\n", "\n")
    anchor_old = OLD.replace("\r\n", "\n")
    anchor_new = NEW.replace("\r\n", "\n")
    if anchor_new in body and body.count(anchor_old) == 0:
        print("  [已应用·幂等跳过] _english_ok() 使用 is_question_echo()")
    else:
        if body.count(anchor_old) != 1:
            raise SystemExit(f"锚点命中 {body.count(anchor_old)} 次（期望 1），已中止，未写盘")
        body = body.replace(anchor_old, anchor_new)
        GEN.write_text(body.replace("\n", "\r\n") if had_crlf else body, encoding="utf-8")
        print("  [已写入] _english_ok() 使用 is_question_echo()")
    if "if is_question_echo(candidate):" not in GEN.read_text(encoding="utf-8"):
        raise SystemExit("校验失败：改动未生效")
    print(f"[generator.py] 补丁 8 校验通过；sha256={hashlib.sha256(GEN.read_bytes()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
