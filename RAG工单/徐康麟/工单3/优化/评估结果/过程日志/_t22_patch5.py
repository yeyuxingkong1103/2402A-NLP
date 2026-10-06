# -*- coding: utf-8 -*-
"""t22 补丁 5（第 2 轮、也是最后一轮）：① 英文占比判据修正；② 回声判定加「整句包含」分支。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

依据（captain 对验收判据的表述）：
    「正文以英文为主（**中文仅限专名、金额单位、引用的原文片段等必要之处**）」
→ 英文占比必须**排除 `（verbatim source: 「…」）` 片段与引用行**再计算（判据定义，不是放宽阈值），
   否则「取值本身是长中文」（如注册地址）会被误判为不达标。

回声判定：实测最终文本可能是「问题回声 + 追加的证据片段」→ 整句不再是问句子串，
但**问句本身**是其前缀 → 增加「问句（≥20 字）被答案整段包含」分支。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

GEN = Path(r"E:\gao6gongdan\工单3\研发\app\core\generator.py")
LANG = Path(r"E:\gao6gongdan\工单3\研发\app\core\language.py")

LANG_OLD = (
    'def english_ratio(text: str) -> float:\n'
    '    """文本里拉丁字母占比（t22 英文语言闸门的度量；分母为去首尾空白后的总字符数）。"""\n'
    '    body = str(text or "").strip()\n'
    '    if not body:\n'
    '        return 0.0\n'
    '    letters = sum(1 for char in body if ("a" <= char.lower() <= "z"))\n'
    '    return round(letters / len(body), 4)\n'
)
LANG_NEW = (
    '# 答案里**必要的**中文成分：逐字原文片段 `（verbatim source: 「…」）` / `(verbatim source: "…")`\n'
    '_VERBATIM_SOURCE_PATTERN = re.compile(r"[（(]\\s*verbatim source\\s*[:：]\\s*[「\\"“][^」\\"”]*[」\\"”]\\s*[）)]")\n'
    '\n'
    '\n'
    'def english_ratio(text: str, *, keep_verbatim: bool = False) -> float:\n'
    '    """文本里拉丁字母占比（t22 英文语言闸门的度量）。\n'
    '\n'
    '    验收判据为「正文以英文为主（**中文仅限专名、金额单位、引用的原文片段等必要之处**）」，\n'
    '    因此默认**排除**为避免编造而保留的逐字原文片段 `（verbatim source: 「…」）`；\n'
    '    `keep_verbatim=True` 时按包含片段与引用行的全文本计算（更严口径，供对照）。\n'
    '    """\n'
    '    body = str(text or "").strip()\n'
    '    if not keep_verbatim:\n'
    '        body = _VERBATIM_SOURCE_PATTERN.sub("", body).strip()\n'
    '    if not body:\n'
    '        return 0.0\n'
    '    letters = sum(1 for char in body if ("a" <= char.lower() <= "z"))\n'
    '    return round(letters / len(body), 4)\n'
)

GEN_OLD = (
    '            body = text_utils.squash_text(citation_mod.answer_body(candidate))\n'
    '            question_squashed = text_utils.squash_text(question)\n'
    '            return bool(body) and body in question_squashed and len(body) >= 0.6 * len(question_squashed)'
)
GEN_NEW = (
    '            body = text_utils.squash_text(citation_mod.answer_body(candidate))\n'
    '            question_squashed = text_utils.squash_text(question)\n'
    '            if not body or not question_squashed:\n'
    '                return False\n'
    '            if body in question_squashed and len(body) >= 0.6 * len(question_squashed):\n'
    '                return True\n'
    '            # t22 补丁 5：实测最终文本可能是「问题回声 + 追加的证据片段」→ 整句不再是问句子串，\n'
    '            # 但**问句本身**是答案的前缀 → 同样判为回声（问句 ≥20 字才启用，避免短问句误判）。\n'
    '            return len(question_squashed) >= 20 and question_squashed in body'
)


def patch(path: Path, old: str, new: str) -> None:
    """精确匹配替换（LF 归一化比较；按原换行风格写回）。"""
    raw = path.read_text(encoding="utf-8")
    had_crlf = "\r\n" in raw
    body = raw.replace("\r\n", "\n")
    anchor, replacement = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
    count = body.count(anchor)
    if count != 1:
        raise SystemExit(f"{path.name} 锚点命中 {count} 次（期望 1），已中止，未写盘")
    path.write_text((body.replace(anchor, replacement)).replace("\n", "\r\n") if had_crlf
                    else body.replace(anchor, replacement), encoding="utf-8")
    print(f"[{path.name}] 已更新；sha256={hashlib.sha256(path.read_bytes()).hexdigest()[:16]}")


def main() -> int:
    """入口：先改 language.py 判据，再改 generator.py 回声判定。"""
    patch(LANG, LANG_OLD, LANG_NEW)
    patch(GEN, GEN_OLD, GEN_NEW)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
