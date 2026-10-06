"""从 pip 缓存离线安装 CUDA 版 torch。

## 背景

本机 PyPI 镜像全部不可达（下载超时），无法 `pip install torch --index-url .../cu124`。
但 pip 缓存里存有 **torch 2.6.0+cu124** 的完整轮子
（`Tag: cp310-cp310-win_amd64`，正好匹配本环境），因此可以直接从缓存解包安装。

Windows 版 torch 轮子**自带 CUDA 运行库**
（`cudart64_12.dll` / `cublas64_12.dll` / `cudnn64_9.dll` / `torch_cuda.dll`），
不需要额外的 `nvidia-*` 依赖包。

## 用法

    python 设计/规格说明/install_cuda_torch.py --dry-run   # 只看会做什么
    python 设计/规格说明/install_cuda_torch.py             # 实际安装
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SITE_PACKAGES = PROJECT_ROOT / ".gao6gongdan-src" / "lib" / "site-packages"
CACHE = Path(os.environ["LOCALAPPDATA"]) / "pip" / "cache"
# torch 2.6.0+cu124 轮子在 pip 缓存中的内容哈希（由 check 脚本定位得到）
WHEEL_HASH = "711e848bec5dd2fc3bff1ab60a9acfe528025d34ae111b9a5e1e51e3"


def find_wheel() -> Path:
    matches = list(CACHE.rglob(f"{WHEEL_HASH}.body"))
    if not matches:
        raise SystemExit(
            "未在 pip 缓存中找到 torch 2.6.0+cu124 轮子。\n"
            "请先联网执行：pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124"
        )
    return matches[0]


def classify(entries: list[str]) -> tuple[list[str], list[str]]:
    """区分「torch 包本体」与「torch 的 dist-info」，其余条目忽略。"""
    package: list[str] = []
    dist_info: list[str] = []
    for name in entries:
        if name.startswith("torch/"):
            package.append(name)
        elif name.startswith("torch-") and ".dist-info/" in name:
            dist_info.append(name)
    return package, dist_info


def main() -> int:
    parser = argparse.ArgumentParser(description="从 pip 缓存离线安装 CUDA 版 torch")
    parser.add_argument("--dry-run", action="store_true", help="只检查与报告，不修改文件")
    args = parser.parse_args()

    wheel = find_wheel()
    print(f"轮子: {wheel}")
    print(f"大小: {wheel.stat().st_size / 1024**3:.2f} GB")

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        package, dist_info = classify(names)
        print(f"条目: 共 {len(names)}，其中 torch 包 {len(package)}、dist-info {len(dist_info)}")
        if not package:
            raise SystemExit("轮子结构异常：未找到 torch/ 目录")

        with archive.open([n for n in dist_info if n.endswith("METADATA")][0]) as handle:
            meta = handle.read().decode("utf-8", "ignore")
        version = next(
            (l.split(":", 1)[1].strip() for l in meta.splitlines() if l.startswith("Version:")),
            "?",
        )
        print(f"版本: {version}")

        if args.dry_run:
            print("\n[dry-run] 将会执行：")
            print(f"  1. 删除 {SITE_PACKAGES / 'torch'}")
            print(f"  2. 解包 {len(package)} 个文件到 {SITE_PACKAGES}")
            print(f"  3. 解包 {len(dist_info)} 个元数据文件")
            return 0

        # ---- 1. 备份与清理旧 torch ----
        target = SITE_PACKAGES / "torch"
        backup = SITE_PACKAGES / "torch_cpu_backup"
        if target.exists():
            if not backup.exists():
                print("备份原 torch（首次安装 CUDA 版时建议保留以便回滚）…")
                shutil.copytree(target, backup)
            print("删除原 torch 目录…")
            shutil.rmtree(target)
        for stale in SITE_PACKAGES.glob("torch-*.dist-info"):
            shutil.rmtree(stale, ignore_errors=True)

        # ---- 2. 解包 ----
        print("解包中（约 2.4 GB，请稍候）…")
        started = time.perf_counter()
        done = 0
        for name in package + dist_info:
            archive.extract(name, SITE_PACKAGES)
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(package) + len(dist_info)} …", flush=True)
        elapsed = time.perf_counter() - started
        print(f"解包完成，用时 {elapsed:.1f}s")

    # ---- 3. 验证 ----
    print("\n验证 CUDA 可用性…")
    import subprocess

    code = (
        "import torch;"
        "print('torch', torch.__version__);"
        "print('cuda build', torch.version.cuda);"
        "print('available', torch.cuda.is_available());"
        "print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    print(result.stdout.strip() or result.stderr.strip()[-800:])
    return 0 if "available True" in result.stdout else 2


if __name__ == "__main__":
    raise SystemExit(main())
