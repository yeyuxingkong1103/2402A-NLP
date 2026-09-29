"""切块指纹与版本（自 chunker.py 拆出）。

与"怎么切块"无关，管的是"切块结果是否变了"的增量判定：
- CHUNKER_VERSION：切块逻辑（正则、归一化、款/项识别）每次行为变更时递增；
- compute_chunk_fingerprint：版本 + 分块结构（条/款/项）的稳定哈希。
打包侧与库存回填（迁移脚本）共用本函数保证公式一致。
"""
from __future__ import annotations

import hashlib

# 切块器版本号：切块逻辑（正则、归一化、款/项识别）每次行为变更时递增。
# 该版本参与切块指纹（compute_chunk_fingerprint），使"切块方式变化"可被
# 导入侧感知并触发分块重建，而不是被 content_hash（正文语义哈希）掩盖。
CHUNKER_VERSION = "3"


def compute_chunk_fingerprint(records: list[tuple[str, str | None, str | None, str | None, str]]) -> str:
    """计算切块指纹：切块器版本 + 分块结果（条/款/项结构）的哈希。

    指纹只用于增量判定的"切块是否变化"维度，不参与 content_hash 的
    防篡改校验（那是数据包与原始文件一致性的职责）。

    参数：
    - records: 每个分块一条记录，五元组依次为
      (chunk_type, article_no, paragraph_no, item_no, content)。
      顺序必须稳定（打包侧按 sequence 排序，回填侧按 sequence/id 排序），
      库存回填（迁移脚本）与打包侧共用本函数以保证公式一致。

    返回：
    - str: 64 位十六进制 SHA256
    """
    digest = hashlib.sha256()
    digest.update(f"chunker:{CHUNKER_VERSION}\n".encode("utf-8"))
    for chunk_type, article_no, paragraph_no, item_no, content in records:
        line = "|".join((
            chunk_type,
            article_no or "",
            paragraph_no or "",
            item_no or "",
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
        ))
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
