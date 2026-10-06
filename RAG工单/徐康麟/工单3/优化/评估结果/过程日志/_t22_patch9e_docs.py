# -*- coding: utf-8 -*-
"""t22 补丁 9e（文档补遗）：登记测试文件 `[降级留痕]` 打印措辞的小瑕疵（不改 tester 文件）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

实测（`pytest 测试/在线/test_english_language_path.py -q -rA`，本页 11 passed）：
`DEGRADED_ALLOWED` 的**所有非必达题**都会打印 `[降级留痕] … ratio=…`，即使该轮实际是达标态
（例如 EN7 ratio=0.607、EN6 ratio=0.559 也打印）。断言本身按 `ENGLISH_BODY_REQUIRED` 正确区分 ①/②，
因此**不是失败**，只是打印措辞容易误读；属 tester 的文件边界，本轮**不改测试文件**，仅在 §25 登记。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

FACTS = Path(r"E:\gao6gongdan\工单3\部署\配置\环境事实.md")

ANCHOR = """* 取值边界单测：`_t22_extract_unit.py` → **6/6**（含 captain 反证用例与表格行口径）。
"""

APPEND = """* 复跑留痕（收口后）：`_t22_after7.py` 再跑一轮 → **达标 4 / 降级 3 / 正例拒答 0**（EN6 该轮为达标态 0.5588），
  `pytest 测试/在线/test_english_language_path.py -q -rA` → **11 passed**；
* 登记一个**打印措辞**小瑕疵（不改 tester 文件）：`DEGRADED_ALLOWED` 里的非必达题都会打印 `[降级留痕] … ratio=…`，
  即使该轮实际达标（如 EN7 0.607、EN6 0.559 也打印）。断言按 `ENGLISH_BODY_REQUIRED` 正确区分 ①/②，**不是失败**，
  但容易被误读为「该题降级」，建议 tester 后续把该行改成「非必达题当前态」。
"""


def main() -> int:
    """入口：在 §25.8 验证清单末尾补两句（幂等）。"""
    body = FACTS.read_text(encoding="utf-8").replace("\r\n", "\n")
    if "_t22_after7.py` 再跑一轮" in body:
        print("  [已应用·幂等跳过] §25.8 复跑留痕与打印措辞登记")
    else:
        if body.count(ANCHOR) != 1:
            raise SystemExit(f"锚点命中 {body.count(ANCHOR)} 次（期望 1），已中止，未写盘")
        body = body.replace(ANCHOR, ANCHOR + APPEND)
        FACTS.write_bytes(body.replace("\n", "\r\n").encode("utf-8"))
        print("  [已写入] §25.8 复跑留痕与打印措辞登记")
    data = FACTS.read_bytes()
    print(f"  环境事实.md: {len(data)} B, crcrlf={data.count(bytes([13, 13, 10]))}, "
          f"sha256={hashlib.sha256(data).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
