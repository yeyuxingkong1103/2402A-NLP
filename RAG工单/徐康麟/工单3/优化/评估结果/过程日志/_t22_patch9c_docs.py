# -*- coding: utf-8 -*-
"""t22 补丁 9c（文档补遗）：§25.2 过程产物清单补全 + 登记本轮踩到的写盘坑。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

补两件事（都是实测事实，供后续任务复用）：
    ① 过程产物清单补上文档定稿脚本 `_t22_patch9_docs.py` / `_t22_patch9b_docs.py` 与本节文档的字节体检数字；
    ② 登记 `Path.write_text()` 在 Windows 上把 `\\r\\n` 再翻译成 `\\r\\r\\n` 的坑（本轮把两份 md 写坏过一次，
       已回滚重放并改为二进制写入）——避免后续任务重复踩。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

FACTS = Path(r"E:\gao6gongdan\工单3\部署\配置\环境事实.md")

ANCHOR = """  `_t22_q95_diag.py`（题 95 判分口径诊断）、`_t22_final_verify.txt`（定稿批次原始输出，含 pytest 与 evaluate 全部行）。
"""

APPEND = """* 文档定稿脚本（同样可审计）：`_t22_patch9_docs.py`（收窄 §25 与设计附录 A 的表述，按行区间替换、定位失败不写盘）、
  `_t22_patch9b_docs.py`（补回 §25.7 第 4 条已知限制）、`_t22_patch9c_docs.py`（本节）；
* **实测坑（本轮踩到并已修）**：`Path.write_text()` 在 Windows 上会把字符串里**已有的** `\\r\\n` 再翻译一次 → 写出 `\\r\\r\\n`（行数翻倍）。
  本文件与 `设计\\验收标准.md` 因此被写坏过一次，已**回滚备份后重放**，并把补丁脚本改为二进制写入（`write_bytes`）。
  修复后字节体检：`部署\\配置\\环境事实.md` = 2091 行 / 169937 B（`\\r\\r\\n` 计数 0）、`设计\\验收标准.md` = 404 行 / 59324 B（`\\r\\r\\n` 计数 0）。
"""


def main() -> int:
    """入口：追加过程产物清单尾部（幂等，锚点异常即中止且不写盘）。"""
    text = FACTS.read_text(encoding="utf-8")
    body = text.replace("\r\n", "\n")
    if "_t22_patch9c_docs.py`（本节）" in body:
        print("  [已应用·幂等跳过] §25.2 过程产物清单补遗")
    else:
        if body.count(ANCHOR) != 1:
            raise SystemExit(f"锚点命中 {body.count(ANCHOR)} 次（期望 1），已中止，未写盘")
        body = body.replace(ANCHOR, ANCHOR + APPEND)
        FACTS.write_bytes(body.replace("\n", "\r\n").encode("utf-8"))
        print("  [已写入] §25.2 过程产物清单补遗")
    data = FACTS.read_bytes()
    print(f"  环境事实.md: {len(data)} B, crcrlf={data.count(bytes([13, 13, 10]))}, "
          f"sha256={hashlib.sha256(data).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
