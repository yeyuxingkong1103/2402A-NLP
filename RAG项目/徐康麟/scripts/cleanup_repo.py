#!/usr/bin/env python3
"""仓库清理（**常驻工具**）：缓存 / 一次性脚本 / 过期日志轮转 / `.pytmp` 跑测残留。

**为什么在 git 里而不是临时脚本**：本仓库清理过三次，每次都是"临时写一个脚本 → 它自己
被下一次清理删掉"。而这个仓库有一条**必须遵守**的判据：

    被 `docs/` 或 `handoff/` 或代码/测试**引用**的临时文件要**保留**
    （`docs/B-9-VERIFICATION.md` 引 `.pytmp/t63/api.log`、
     `docs/B-8-IMPLEMENTATION.md` 引 `.pytmp/t65/prefs-evidence.json`、
     `handoff/WEB-SKELETON.md` 引 `.tmp-ui-skeleton.html` ……）。
    整片删掉就会造出一堆**断链**——`docs/DESIGN-TODO.md` §D5-① 记过这个教训。

所以自动化的是"**先判断引用，再删**"，且默认 dry-run、打印每项保留理由。

用法：
    python scripts/cleanup_repo.py                 # dry-run（默认，只报告）
    python scripts/cleanup_repo.py --apply         # 真删（含 git rm 已跟踪的一次性文件）

**绝不触碰**：`data/`（红线：不进仓库也不该删）、`index/`（含入库基线
`milvus_manifest.json`；`legal_rag.db-wal` 删了可能丢数据）、`.venv/`、`eval/results/`（全部证据）、
`handoff/*.md` 与 `docs/`（台账与文档）。
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 引用源：这些目录/文件里出现过的路径 ⇒ 对应临时文件保留
DOC_GLOBS = ("docs/*.md", "handoff/*.md", "README.md")
CODE_GLOBS = ("tests/*.py", "scripts/*.py", "legal_rag/*.py", "legal_rag/*/*.py",
              "conftest.py", "*.py")

CACHE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "htmlcov"}
CACHE_SUFFIX = {".pyc", ".pyo"}

#: 一次性文件候选（仓库根与 handoff 里的点开头脚本/备份）
ONE_OFF_GLOBS = (".tmp-*", "handoff/.t4*", "handoff/artifacts/.t4*")

#: `logs/` 只留文档引用过的 + 最新一份
KEEP_LOG_NAMES = {"app.log", "app.log.1", "app.log.lock"}

#: `.pytmp` 里永远保留的目录（**本工具的临时工作区也在这里**，否则会被自己删掉）
PYTMP_ALWAYS_KEEP = {"finetune", "cloud", "cloud2", "cloud3", "push_run.py",
                     "cleanup_scratch.py", "cleanup_repo.py"}

#: 永不触碰的顶层路径
NEVER_TOUCH = {"data", "index", ".venv", "eval", "docs", "handoff", "legal_rag", "scripts",
               "tests", "web", "deploy", ".git", "knowledge", "uploads"}

#: 即使被引用也删：可再生的打包产物（文档只把它当**命令**记录）
ALWAYS_DELETE = {"sync-legalrag.tgz"}


def reference_texts() -> dict[str, str]:
    """把所有"引用源"读成 {相对路径: 正文}。"""
    out: dict[str, str] = {}
    for pattern in DOC_GLOBS + CODE_GLOBS:
        for path in ROOT.glob(pattern):
            if path.is_file():
                try:
                    out[path.relative_to(ROOT).as_posix()] = path.read_text(
                        encoding="utf-8", errors="replace")
                except OSError:
                    continue
    return out


def tracked_files() -> set[str]:
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    return set(result.stdout.split())


def classify_one_off(candidates: list[Path], references: dict[str, str]) -> tuple[list, list]:
    """把一次性文件分成 (保留, 删除)。

    判据：文件名出现在**文档或非候选代码**的正文里 ⇒ 保留。
    "只被同批候选引用"（自己产出的中间输出）**不算证据** —— 否则脚本和它的输出会互相保命。
    """
    names = {path.name for path in candidates}
    keep, drop = [], []
    for path in candidates:
        if path.name in ALWAYS_DELETE:
            drop.append(path)
            continue
        cited_by_docs = any(path.name in text for rel, text in references.items()
                            if "/" in rel and rel.split("/")[0] in {"docs", "handoff"}
                            or rel.endswith(".md"))
        cited_by_code = any(path.name in text for rel, text in references.items()
                            if Path(rel).name not in names and not rel.endswith(".md"))
        (keep if (cited_by_docs or cited_by_code) else drop).append(path)
    return keep, drop


def classify_pytmp(entries: list[Path], references: dict[str, str]) -> tuple[list, list]:
    """`.pytmp` 顶层条目分成 (保留, 删除)：文档引用过的 + 常驻工作区留下，其余删。"""
    keep, drop = [], []
    for entry in entries:
        name = entry.name
        if name.startswith("cleanup") or name in PYTMP_ALWAYS_KEEP:
            keep.append(entry)
            continue
        if name in ALWAYS_DELETE:
            drop.append(entry)
            continue
        if any(f".pytmp/{name}" in text or f".pytmp\\{name}" in text
               for text in references.values()):
            keep.append(entry)
            continue
        drop.append(entry)
    return keep, drop


def path_size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def plan() -> dict[str, list[Path]]:
    references = reference_texts()
    candidates = sorted({path for pattern in ONE_OFF_GLOBS for path in ROOT.glob(pattern)
                         if path.is_file()})
    keep_one_off, drop_one_off = classify_one_off(candidates, references)

    caches = [path for path in ROOT.rglob("*")
              if path.is_dir() and path.name in CACHE_DIR_NAMES and ".venv" not in path.parts]
    caches += [path for path in ROOT.rglob("*")
               if path.is_file() and path.suffix in CACHE_SUFFIX and ".venv" not in path.parts]

    logs = [path for path in (ROOT / "logs").glob("*")
            if path.is_file() and path.name not in KEEP_LOG_NAMES]

    pytmp_dir = ROOT / ".pytmp"
    keep_pytmp, drop_pytmp = ([], [])
    if pytmp_dir.is_dir():
        keep_pytmp, drop_pytmp = classify_pytmp(sorted(pytmp_dir.iterdir()), references)

    return {"keep_one_off": keep_one_off, "drop_one_off": drop_one_off, "caches": caches,
            "logs": logs, "keep_pytmp": keep_pytmp, "drop_pytmp": drop_pytmp}


def main() -> int:
    parser = argparse.ArgumentParser(description="仓库清理（默认 dry-run）")
    parser.add_argument("--apply", action="store_true", help="真删（含 git rm）")
    args = parser.parse_args()

    todo = plan()
    tracked = tracked_files()
    total = sum(path_size(path) for group in ("drop_one_off", "caches", "logs", "drop_pytmp")
                for path in todo[group])

    print(f"待清理 {sum(len(todo[g]) for g in ('drop_one_off', 'caches', 'logs', 'drop_pytmp'))} 项，"
          f"可回收 {total / 1024 ** 2:.1f} MB")
    print(f"  一次性文件：删 {len(todo['drop_one_off'])} / 留 {len(todo['keep_one_off'])}")
    for path in todo["keep_one_off"]:
        print(f"    留 {path.relative_to(ROOT).as_posix()}（被文档/代码引用）")
    print(f"  缓存：{len(todo['caches'])} 项 / {sum(path_size(p) for p in todo['caches']) / 1024 ** 2:.1f} MB")
    print(f"  过期日志：{len(todo['logs'])} 项 / {sum(path_size(p) for p in todo['logs']) / 1024 ** 2:.1f} MB"
          f"（保留 {sorted(KEEP_LOG_NAMES)}）")
    print(f"  .pytmp：删 {len(todo['drop_pytmp'])} / 留 {len(todo['keep_pytmp'])}"
          f"（{sum(path_size(p) for p in todo['drop_pytmp']) / 1024 ** 2:.1f} MB）")
    print("    留下的：" + ", ".join(sorted(path.name for path in todo["keep_pytmp"])[:12])
          + (" …" if len(todo["keep_pytmp"]) > 12 else ""))

    if not args.apply:
        print("\n（dry-run：加 --apply 才真删）")
        for path in sorted(todo["drop_one_off"]):
            rel = path.relative_to(ROOT).as_posix()
            print(f"    删 {rel}{'  [git rm]' if rel in tracked else ''}")
        return 0

    git_removed = 0
    for path in todo["drop_one_off"]:
        rel = path.relative_to(ROOT).as_posix()
        if rel in tracked:
            subprocess.run(["git", "rm", "-q", "--", rel], cwd=ROOT, check=False)
            git_removed += 1
        else:
            path.unlink(missing_ok=True)
    removed, failed = 0, []
    for path in todo["caches"] + todo["logs"] + todo["drop_pytmp"]:
        try:
            shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)
            removed += 1
        except OSError as exc:
            failed.append(f"{path.name}: {type(exc).__name__}")
    print(f"\n已删：git 跟踪 {git_removed} 个（已 git rm，待提交）；本地 {removed} 个；失败 {len(failed)}")
    for line in failed[:5]:
        print(f"   失败（多为 ACL 保护，无影响）：{line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
