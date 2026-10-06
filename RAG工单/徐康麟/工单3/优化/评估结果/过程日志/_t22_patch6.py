# -*- coding: utf-8 -*-
"""t22 补丁 6（定点修复，仅此一处）：`_extract_field_value()` 的取值边界。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

captain 的实测反证（以它为准）：
    `招股说明书2_p0015_x26` 原文是 `3、注册资本：人民币5,000 万元 4、法定代表人：赵马克 …`（**完整、无折断**），
    而 `_extract_field_value(..., '注册资本')` 返回 `('人民币5,000', 'separator')` —— **在 `5,000` 后的空格处截断，丢了「万元」**。
    → 根因在**抽取侧的取值边界**，不在解析层（无需动 `pdf_parser.py`、无需重建索引）。

修法：
    ① 取值捕获允许**内部空格**（到强分隔符 `| ｜ ， 。 ； ; 换行` 为止）；
    ② 新增 `_bound_field_value()` 做边界收敛：先按「下一个编号项（` 4、`）」或「下一个 `字段：`」切掉尾巴，
       再折空白、并把「数字 + 空格 + 单位」粘回（`人民币5,000 万元` → `人民币5,000万元`，
       `5,520.00 万元` → `5,520.00万元`，`1,670 万股` → `1,670万股`）。
    ③ 既有行为保持不变：`法定代表人 → 赵马克`（对）；`expects_numeric=True` 时对纯中文名返回空（对）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

GEN = Path(r"E:\gao6gongdan\工单3\研发\app\core\generator.py")

OLD_HELPER_ANCHOR = 'def _split_clauses(text: str) -> list[str]:'
NEW_HELPER = '''def _bound_field_value(raw_value: str) -> str:
    """把「取值」的边界收敛成**可直接逐字核验的原话**（t22 补丁 6，§25）。

    实测动机（captain 反证）：`3、注册资本：人民币5,000 万元 4、法定代表人：赵马克 …` 里，
    取值在 `5,000` 后的空格处被截断成 `人民币5,000`（丢了 `万元`）→ 英文框架句取不到完整取值 → 降级。
    处理（通用，不做任何题目特判）：
        ① 切掉「下一个编号项」（` 4、` / ` 12.`）之后的尾巴；
        ② 切掉「下一个 `字段：` 形态」之后的尾巴（避免把下一字段吞进取值）；
        ③ 折空白，并把「数字 + 空格 + 单位/量词」粘回（千分位与单位之间的空格是排版产物，不是事实差异）。
    """
    value = re.sub(r"\\s+", " ", str(raw_value or "")).strip(" \\t:：|｜")
    value = re.split(r"\\s+\\d{1,2}\\s*[、.．]\\s*", value, maxsplit=1)[0]
    value = re.split(r"\\s+[\\u4e00-\\u9fffA-Za-z]{2,10}\\s*[:：]", value, maxsplit=1)[0]
    value = re.sub(r"(\\d)\\s+(万元|亿元|元|万股|%|％|股|次|倍|年|月|日)", r"\\1\\2", value)
    return value.strip(" \\t:：|｜")


def _split_clauses(text: str) -> list[str]:'''

EDITS: list[tuple[str, str, int]] = [
    (OLD_HELPER_ANCHOR, NEW_HELPER, 1),
    # 取值组：允许内部空格（到强分隔符为止），随后交给 _bound_field_value 收敛
    (
        '_FIELD_COPULA_TEMPLATE = r"{kw}\\s*(?:是|为)\\s*[「“\\"\']?\\s*([^|｜\\s，。；;、]{{1,40}})"',
        '_FIELD_COPULA_TEMPLATE = r"{kw}\\s*(?:是|为)\\s*[「“\\"\']?\\s*([^|｜，。；;、\\n]{{1,60}})"',
        1,
    ),
    (
        '_FIELD_SEPARATOR_TEMPLATE = r"{kw}\\s*[:：]\\s*[|｜]?\\s*([^|｜\\s，。；;、]{{1,40}})"',
        '_FIELD_SEPARATOR_TEMPLATE = r"{kw}\\s*[:：]\\s*[|｜]?\\s*([^|｜，。；;、\\n]{{1,60}})"',
        1,
    ),
    # 公文式分支取值同样收敛
    (
        '            for match in pattern.finditer(stripped):\n'
        '                value = match.group(1).strip(" \\t:：|｜「」“”\\"\'《》")\n'
        '                if _field_value_ok(value, keyword, expects_numeric=expects_numeric):\n'
        '                    return value, style',
        '            for match in pattern.finditer(stripped):\n'
        '                value = _bound_field_value(match.group(1)).strip(" 「」“”\\"\'《》")\n'
        '                if _field_value_ok(value, keyword, expects_numeric=expects_numeric):\n'
        '                    return value, style',
        1,
    ),
]


def main() -> int:
    """入口：加边界收敛助手 + 放宽取值组（精确匹配断言，失败不写盘）。"""
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
    print(f"[generator.py] 补丁 6 已应用 {len(EDITS)} 处；sha256={hashlib.sha256(GEN.read_bytes()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
