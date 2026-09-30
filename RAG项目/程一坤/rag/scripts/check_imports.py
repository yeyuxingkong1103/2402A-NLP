# -*- coding: utf-8 -*-
"""全仓「调用方 import 完整性」校验（只读，随时可跑）。

用法：
    python scripts/check_imports.py

背景（为什么必须有它）：
    批次 18/19 做过两次大规模模块拆分，当时只 grep 了 backend/ 目录，
    结果 evaluation/ 与 scripts/ 下的调用方留下了旧导入 —— 静态看没问题，
    一到运行时（跑评测脚本）才 ImportError。这是一类"拆分后必然复发"的盲区。

做法：
    对 backend/ 之外的每个 .py 做 AST 解析，把 `from app.x.y import A, B`
    逐名解析到真实模块，校验 A 与 B 是否存在；不存在即报出 文件:行号:名字。
    backend/app/ 下的模块由 pytest 全量覆盖，这里跳过以免噪音。

退出码：0 = 全部可解析；1 = 存在失效导入（便于 CI/脚本串联）。
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

# 本文件位于 <项目根>/scripts/ 下，向上两层即项目根
ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".workbuddy",
    ".pytest_cache", "data", "docs",
    # 临时/备份目录不参与（它们不是交付物）；
    # 备份目录统一用 .dev_bakNN 前缀，靠下面的 startswith 前缀判定一并跳过 ——
    # 否则每开一批（.dev_bak22/23/…）都要回来加一条，实测批次 25 就漏加过一次
    ".pdf_probe",
}

# 备份目录前缀：.dev_bak21 / .dev_bak25 / .dev_bak25b … 全部跳过
BACKUP_DIR_PREFIX = ".dev_bak"


def target_files() -> list[Path]:
    """收集待查文件：backend/ 之外的全部 .py（含 backend/scripts）。"""
    files: list[Path] = []
    for path in ROOT.rglob("*.py"):
        if any(
            part in SKIP_DIRS or part.startswith(BACKUP_DIR_PREFIX)
            for part in path.parts
        ):
            continue
        # backend/app 已由 pytest 全量覆盖
        if path.relative_to(ROOT).as_posix().startswith("backend/app/"):
            continue
        files.append(path)
    return sorted(files)


def check_file(path: Path) -> list[str]:
    """检查单个文件里的 app.* 绝对导入；返回问题描述列表。"""
    problems: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as error:
        return [f"{path.relative_to(ROOT)}:{error.lineno}: 语法错误 {error.msg}"]

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module_name = node.module or ""
        if not module_name.startswith("app."):
            continue
        # 相对导入只在 backend 包内部使用，不在此处解析
        if node.level:
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception as error:  # noqa: BLE001
            problems.append(
                f"{path.relative_to(ROOT)}:{node.lineno}: "
                f"模块导入失败 {module_name}（{type(error).__name__}）"
            )
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            if not hasattr(module, alias.name):
                problems.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: "
                    f"{module_name} 没有 {alias.name}"
                )
    return problems


def main() -> int:
    sys.path.insert(0, str(ROOT / "backend"))
    sys.path.insert(0, str(ROOT))

    files = target_files()
    problems: list[str] = []
    for path in files:
        problems.extend(check_file(path))

    print(f"扫描文件数: {len(files)}（backend/app 由 pytest 覆盖，此处跳过）")
    if problems:
        print("❌ 发现失效导入：")
        for problem in problems:
            print("   -", problem)
        return 1
    print("✅ 全部 app.* 导入均可解析（模块存在 + 符号存在）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
