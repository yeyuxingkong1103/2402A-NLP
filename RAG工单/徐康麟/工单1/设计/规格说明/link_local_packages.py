"""把本机其它 conda 环境中已存在的**纯 Python 包**链接进当前环境。

背景：算力云部署时依赖由 requirements.txt 正常安装；但在本机开发环境下外网
极慢，而多个既有 conda 环境里已经有这些包。本脚本只处理纯 Python 包（不含
C 扩展），通过创建目录联接（junction）的方式让当前解释器可直接 import，
不复制文件、不污染源环境。

用法（在目标环境的 python 下运行）:
    python _tools/link_local_packages.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# 纯 Python 包（无 C 扩展）候选 -> 提供该包的环境名候选
PURE_PACKAGES: dict[str, list[str]] = {
    "loguru": ["fastapi_ds"],
    "jieba": ["__base__"],
    "rank_bm25": [],
    "pdfplumber": [],
    "streamlit": ["__base__"],
    "pandas": ["__base__"],
}

# 含 C 扩展、必须版本 ABI 匹配的包——只允许同 Python 小版本的环境
BINARY_PACKAGES: dict[str, list[str]] = {
    "pymupdf": ["sglang"],
    "fitz": ["sglang"],
    "chromadb": ["sglang", "langchain_env", "ragas"],
    "sentence_transformers": ["langchain_env", "sglang"],
}

ENVS_ROOT = Path(r"E:\Anaconda\envs")
BASE_SITE = Path(r"E:\Anaconda\Lib\site-packages")


def target_site() -> Path:
    for candidate in Path(sys.prefix).glob("Lib/site-packages"):
        return candidate
    raise SystemExit(f"找不到 site-packages: {sys.prefix}")


def env_site(env: str) -> Path | None:
    if env == "__base__":
        return BASE_SITE if BASE_SITE.exists() else None
    path = ENVS_ROOT / env / "Lib" / "site-packages"
    return path if path.exists() else None


def env_python_version(env: str) -> str | None:
    exe = Path(r"E:\Anaconda\python.exe") if env == "__base__" else ENVS_ROOT / env / "python.exe"
    if not exe.exists():
        return None
    try:
        out = subprocess.run(
            [str(exe), "-c", "import sys;print('%d.%d'%sys.version_info[:2])"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def link(src: Path, dst: Path) -> str:
    """目录联接（Windows junction），跨盘也可用；已存在则跳过。"""
    if dst.exists() or dst.is_symlink():
        return "exists"
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
            capture_output=True,
            text=True,
        )
        return "linked" if result.returncode == 0 else f"failed: {result.stderr.strip()[:80]}"
    try:
        dst.symlink_to(src, target_is_directory=True)
        return "linked"
    except OSError as exc:
        return f"failed: {exc}"


def main() -> int:
    site = target_site()
    this_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"目标环境: {sys.prefix}")
    print(f"site-packages: {site}")
    print(f"Python: {this_version}\n")

    plan: list[tuple[str, str]] = []
    for package, envs in PURE_PACKAGES.items():
        for env in envs:
            source = env_site(env)
            if not source:
                continue
            for candidate in sorted(source.glob(f"{package}*")):
                if any(skip in candidate.name for skip in (".dist-info", "..")):
                    continue
                plan.append((env, str(candidate)))
                break
            break

    for package, envs in BINARY_PACKAGES.items():
        for env in envs:
            version = env_python_version(env)
            if version != this_version:
                print(f"[skip] {package}: {env} 是 Python {version}，与本环境 {this_version} 不兼容")
                continue
            source = env_site(env)
            if not source:
                continue
            for candidate in sorted(source.glob(f"{package}*")):
                if ".dist-info" in candidate.name:
                    continue
                plan.append((env, str(candidate)))
                break
            break

    if not plan:
        print("没有可链接的包。")
        return 0

    for env, src in plan:
        name = Path(src).name
        dst = site / name
        outcome = link(Path(src), dst)
        print(f"[{outcome:>8}] {name:<28} <- {env}")

    print("\n=== 链接后可用性自检 ===")
    modules = ["loguru", "jieba", "pymupdf", "chromadb", "sentence_transformers", "streamlit", "pandas", "pdfplumber", "rank_bm25"]
    for module in modules:
        probe = subprocess.run(
            [sys.executable, "-c", f"import importlib.util as u,sys;sys.exit(0 if u.find_spec('{module}') else 1)"],
            capture_output=True,
        )
        print(f"  {'OK ' if probe.returncode == 0 else 'NO '} {module}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
