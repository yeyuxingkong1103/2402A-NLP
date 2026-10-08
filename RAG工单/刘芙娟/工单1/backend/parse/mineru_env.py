# -*- coding: utf-8 -*-
"""MinerU 外部依赖：配置写入、环境构造、可执行文件定位、子进程调用。

不读写产物目录——本模块只关心"怎么把 MinerU 跑起来"。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import EXIT_ARG, EXIT_DEPENDENCY, REPO_ROOT, ScriptError


def write_mineru_config(config_path: Path, pipeline_models: Path, vlm_models: Path) -> None:
    """写项目内 MinerU 配置。只写模型路径与来源，不写任何凭据（constitution 原则 III）。"""
    payload = {"models-dir": {"pipeline": str(pipeline_models), "vlm": str(vlm_models)},
               "model-source": "local", "config_version": "1.3.2"}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")


def build_mineru_env(config_path: Path) -> dict[str, str]:
    """构造子进程环境。只在副本上叠加，不修改当前进程与系统环境变量。"""
    env = dict(os.environ)
    env["MINERU_TOOLS_CONFIG_JSON"] = str(config_path.resolve())
    env["MINERU_MODEL_SOURCE"] = "local"  # 强制离线，禁止联网兜底
    env["MINERU_DEVICE_MODE"] = "cpu"     # 本机无 NVIDIA 显卡；-d 参数自 3.0.0 已删除
    return env


def check_weights(backend: str, pipeline_models: Path, vlm_models: Path) -> Path:
    """校验所选后端的权重目录存在。不存在即失败——拒绝联网下载（docs/04 §11）。"""
    chosen = pipeline_models if backend == "pipeline" else vlm_models
    if not chosen.is_dir():
        raise ScriptError(
            EXIT_ARG,
            f"{backend} 后端权重目录不存在：{chosen}\n"
            f"       拒绝联网下载。请先手动下载权重，或用选项指向已有目录。")
    return chosen


def resolve_mineru_exe() -> Path:
    """按受支持运行时派生 mineru 可执行文件路径（constitution 原则 I）。"""
    bindir = Path(sys.executable).parent
    for candidate in (bindir / "Scripts" / "mineru.exe", bindir / "mineru.exe",
                      bindir / "Scripts" / "mineru", bindir / "mineru"):
        if candidate.exists():
            return candidate
    found = shutil.which("mineru")
    if found:
        return Path(found)
    raise ScriptError(
        EXIT_DEPENDENCY,
        f"未找到 mineru 可执行文件（已查找 {bindir} 及其 Scripts 子目录）\n"
        f"       请先安装：{sys.executable} -m pip install -r {REPO_ROOT / 'requirements.txt'}")


def probe_mineru_version(mineru_exe: Path, env: dict[str, str]) -> str:
    """取 MinerU 版本号，用于产物溯源与 docs/04 §12 的参数回填（FR-014）。"""
    try:
        proc = subprocess.run([str(mineru_exe), "-v"], env=env,
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<探测失败: {exc.__class__.__name__}>"
    output = (proc.stdout or proc.stderr or "").strip()
    return output.splitlines()[0] if output else "<未知>"


def run_mineru(mineru_exe: Path, pdf_path: Path, out_dir: Path, backend: str,
               env: dict[str, str]) -> None:
    """子进程调用 MinerU。崩溃不影响主进程，退出码被精确映射。"""
    command = [str(mineru_exe), "-p", str(pdf_path), "-o", str(out_dir), "-b", backend]
    print(f"      执行: {' '.join(command)}")
    try:
        proc = subprocess.run(command, env=env)
    except OSError as exc:
        raise ScriptError(EXIT_DEPENDENCY, f"无法启动 MinerU 子进程：{exc}") from exc
    if proc.returncode != 0:
        raise ScriptError(EXIT_DEPENDENCY, f"MinerU 子进程退出码 {proc.returncode}")
