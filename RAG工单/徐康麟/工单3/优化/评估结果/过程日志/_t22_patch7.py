# -*- coding: utf-8 -*-
"""t22 补丁 7（定点修复，两处）：取值边界收敛的收尾（补丁 6 的同一处缺陷）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

补丁 6 落地后**仍残留**两种不干净取值（补丁 6 自带单测的实测输出）：
    * `3、注册资本：人民币5,000 万元 4、法定代表人：赵马克` → `('人民币5,000万元 4', 'separator')`：
      编号项的 `、` 被取值字符类挡在外面，所以「空格 + 编号数字」留在了尾部，
      而补丁 6 的切尾正则 `\\s+\\d{1,2}\\s*[、.．]\\s*` 要求 `、` 在场 → 没命中；
    * 同一条证据的 `法定代表人` → `('赵马克 5', …)`：同一残留，且因为含数字而**通过了 `expects_numeric=True`**
      （该问法本应返回空），等于把「非数值取值」误判成合法数值。

修法（两处，均为通用缺陷、不做任何题目特判）：
    ① `_bound_field_value()` 增加一条「切掉尾部孤单编号」：`\\s+\\d{1,2}$`（取值的强分隔符已保证
       尾部不可能是有意义的末位数字；带单位的取值末位是单位，纯数字取值内部无空格）；
    ② **表格单元格分支**也走 `_bound_field_value()`：此前只有 `字段：取值` 分支做了边界收敛，
       于是表格证据 `| 注册资本 | 5,520.00 万元 |` 保留了排版空格（`5,520.00 万元`），
       与逐字核验用的原话形态不一致。

本脚本幂等：若目标文本已在位，则只做校验并打印「已应用」，不重复写盘。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

GEN = Path(r"E:\gao6gongdan\工单3\研发\app\core\generator.py")

EDITS = [
    (
        "① 取值尾部孤单编号（`万元 4` → `万元`）",
        '    value = re.sub(r"(\\d)\\s+(万元|亿元|元|万股|%|％|股|次|倍|年|月|日)", r"\\1\\2", value)\n',
        '    value = re.sub(r"\\s+\\d{1,2}$", "", value)          '
        '# 切掉「空格 + 1~2 位编号」（编号项标记被字符类挡在取值外）\n'
        '    value = re.sub(r"(\\d)\\s+(万元|亿元|元|万股|%|％|股|次|倍|年|月|日)", r"\\1\\2", value)\n',
    ),
    (
        "② 表格单元格分支同口径收敛",
        '                    value_cell = re.sub(r"(\\d),\\s+(\\d{3})", r"\\1,\\2", value_cell)\n',
        '                    value_cell = _bound_field_value(re.sub(r"(\\d),\\s+(\\d{3})", r"\\1,\\2", '
        'value_cell))\n',
    ),
]


def main() -> int:
    """入口：逐处「已应用则校验 / 未应用则替换」，任一处锚点异常即中止且不写盘。"""
    text = GEN.read_text(encoding="utf-8")
    had_crlf = "\r\n" in text
    body = text.replace("\r\n", "\n")
    changed = 0
    for label, old, new in EDITS:
        if new in body:
            print(f"  [已应用·幂等跳过] {label}")
            continue
        count = body.count(old)
        if count != 1:
            raise SystemExit(f"锚点异常：{label} 命中 {count} 次（期望 1），已中止，未写盘")
        body = body.replace(old, new)
        changed += 1
        print(f"  [已写入] {label}")
    if changed:
        GEN.write_text(body.replace("\n", "\r\n") if had_crlf else body, encoding="utf-8")
    print(f"[generator.py] 补丁 7 处理 {len(EDITS)} 处（本次写入 {changed} 处）；"
          f"sha256={hashlib.sha256(GEN.read_bytes()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
