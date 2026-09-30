# -*- coding: utf-8 -*-
"""把交付物按 sha 存副本到 handoff/artifacts/ —— 根治"审的版本 ≠ 仓库版本"。

为什么需要
----------
本项目曾两次出现「审查时的版本」与「仓库当前版本」不一致；最近一次（2026-09-17）
t23 交付的 ``web/ui.html`` = ``FECCC523014E``，因后续修复改了同一文件而变成
``060C8AB3BB07``，**原交付版永久不可复现**（当时既无 VCS 也无副本）。

用法
----
    .venv\\Scripts\\python.exe scripts\\snapshot_artifacts.py                # 快照默认清单
    .venv\\Scripts\\python.exe scripts\\snapshot_artifacts.py web/ui.html    # 只快照指定文件

产物
----
* ``handoff/artifacts/<basename>.<sha12><ext>`` —— 副本（与源逐字节相同）
* ``handoff/artifacts/MANIFEST.md`` —— 追加一行：时间 / 源路径 / sha256[:12] / 字节 / 副本名

约定（同时写进 ``handoff/AGENT-WORKING-RULES.md`` 附录第 8 条）
------------------------------------------------------------
* 交付者负责落副本；**审查/验证任务先从 MANIFEST 取副本，并先核 sha 再看内容**；
* 副本**只增不改不删**（历史可追溯）。
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "handoff" / "artifacts"

#: 默认快照清单：审查/验证任务最常判定的那批产物
DEFAULT_TARGETS = (
    "web/ui.html",
    "docs/WEB-UX.md",
    "legal_rag/ingest/chunker.py",
    "legal_rag/generate/prompt.py",
    "legal_rag/config.py",
    "legal_rag/api/app.py",
)

MANIFEST_HEADER = (
    "# 交付副本清单（由 `scripts/snapshot_artifacts.py` 追加，**只增不改**）\n\n"
    "用途：审查/验证任务按 `sha256[:12]` 取用**被审当时的那一份**，避免读到后来被改过的文件。\n\n"
    "| 时间 | 源路径 | sha256[:12] | 字节 | 副本 |\n|---|---|---|---|---|\n"
)


def sha256_12(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def main(argv: list[str]) -> int:
    targets = tuple(argv) if argv else DEFAULT_TARGETS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for rel in targets:
        source = ROOT / rel
        if not source.is_file():
            print("MISSING ", rel)
            continue
        digest = sha256_12(source)
        size = source.stat().st_size
        copy_name = f"{source.stem}.{digest}{source.suffix}"
        target = OUT_DIR / copy_name
        if target.exists():
            print("EXISTS  %s  %d bytes" % (copy_name, size))
        else:
            shutil.copy2(source, target)
            print("COPIED  %s  %d bytes" % (copy_name, size))
        rows.append("| %s | `%s` | `%s` | %d | `%s` |" % (
            time.strftime("%Y-%m-%d %H:%M"), rel, digest, size, copy_name))

    manifest = OUT_DIR / "MANIFEST.md"
    if not manifest.exists():
        manifest.write_text(MANIFEST_HEADER, encoding="utf-8")
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")

    print("manifest ->", manifest.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
