"""顺序安装缺失依赖，每个包限时；失败即跳过，不阻塞主流程。

用法（在 .venv 中用 python 运行）:
    python _tools/install_missing.py
"""
from __future__ import annotations

import subprocess
import sys
import time

# 按“重要性 -> 体积”排序：先装小而关键的纯 Python 包
PACKAGES: list[tuple[str, int]] = [
    ("loguru", 120),
    ("rank-bm25", 120),
    ("jieba", 180),
    ("pdfplumber", 240),
    ("pymupdf", 240),
    ("pandas", 300),
    ("chromadb", 600),
    ("streamlit", 600),
]

# 已有 / 可继承的包，不需要重装
ALREADY = ["numpy", "pydantic", "sqlalchemy", "torch", "sentence_transformers", "pytest", "requests"]

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"


def installed(module: str) -> bool:
    return (
        subprocess.run(
            [sys.executable, "-c", f"import importlib.util as u,sys; sys.exit(0 if u.find_spec('{module}') else 1)"],
            capture_output=True,
        ).returncode
        == 0
    )


def main() -> int:
    print(f"python: {sys.version.split()[0]} at {sys.executable}")
    for name, timeout in PACKAGES:
        module = name.replace("-", "_")
        if installed(module):
            print(f"[skip] {name}: already importable")
            continue
        started = time.time()
        print(f"[start] {name} (timeout {timeout}s)", flush=True)
        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "pip", "install", "--no-input",
                    "--disable-pip-version-check", "--index-url", MIRROR, name,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"[timeout] {name} after {timeout}s -> skipped")
            continue
        elapsed = time.time() - started
        if proc.returncode == 0:
            print(f"[ok] {name} in {elapsed:.0f}s")
        else:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            print(f"[fail] {name} rc={proc.returncode} in {elapsed:.0f}s :: {' | '.join(tail)}")
    print("\n=== final availability ===")
    for name in [p[0] for p in PACKAGES] + ALREADY:
        print(f"  {'OK ' if installed(name.replace('-', '_')) else 'NO '} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
