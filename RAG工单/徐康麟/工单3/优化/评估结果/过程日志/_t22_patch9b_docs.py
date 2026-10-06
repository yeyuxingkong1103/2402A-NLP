# -*- coding: utf-8 -*-
"""t22 补丁 9b（文档补遗）：§25.7 补回被改写掉的第 4 条已知限制。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

原因：补丁 9 重写 §25.7 时只保留了 3 条，而 §25.6 的表内仍引用「**仍作答**（**已知限制**，见 25.7）」——
即自造英文负例「股票期权归属安排」被判为可答这条残余风险。此处补回，避免悬空引用与漏登记。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

FACTS = Path(r"E:\gao6gongdan\工单3\部署\配置\环境事实.md")

ANCHOR = """   **不改判分器 / golden / 测试断言**（冻结边界），仅登记；12 轮全部 ≥ 验收线 90%。
"""

APPEND = """4. **自造英文负例「股票期权归属安排」仍被作答**（§25.6 表内已如实登记）：该问句没有命中任何主题映射词 →
   走英文向量底线（实测向量相似度 **0.531 ≥** `RAG_ANSWER__MIN_VECTOR_SIMILARITY_EN=0.45`）→ 判为可答。
   属**跨语言可答性判据的残余风险**：收紧这条底线必须重测 5 条英文正例（可能误拒正例），本轮**不调参**、仅登记，
   并已由 `测试/在线/test_english_language_path.py` 的回声/负例用例持续覆盖（Tesla 与 2099 两条已拒答）。
"""


def main() -> int:
    """入口：在 §25.7 第 3 条之后追加第 4 条（幂等，命中异常即中止）。"""
    text = FACTS.read_text(encoding="utf-8")
    body = text.replace("\r\n", "\n")
    if "4. **自造英文负例「股票期权归属安排」仍被作答**" in body:
        print("  [已应用·幂等跳过] §25.7 第 4 条")
    else:
        if body.count(ANCHOR) != 1:
            raise SystemExit(f"锚点命中 {body.count(ANCHOR)} 次（期望 1），已中止，未写盘")
        body = body.replace(ANCHOR, ANCHOR + APPEND)
        # 二进制写入：避免 `Path.write_text` 在 Windows 上把 `\n` 再翻译一次（会写出 `\r\r\n`）。
        FACTS.write_bytes(body.replace("\n", "\r\n").encode("utf-8"))
        print("  [已写入] §25.7 第 4 条")
    print(f"  环境事实.md: {FACTS.stat().st_size} B, sha256={hashlib.sha256(FACTS.read_bytes()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
