"""调用 MinerU 解析 PDF/DOCX，并定位其产物。

MinerU 装在独立 venv（D:\\Mineru\\venv，实测 3.4.5），与本项目 Python 环境隔离，
因此走 subprocess 而不是 import。附带好处：解析进程用完即退，不与在线推理抢显存
（技术方案 10.1 对 ingest-worker 的要求）。
"""
from __future__ import annotations

import pathlib
import subprocess

MINERU_EXE = r"D:\Mineru\venv\Scripts\mineru.exe"

# 单册解析耗时实测约 20 秒（docx）~ 70 秒（91 页 pdf），留足余量
TIMEOUT_SECONDS = 1800

# 只有这几类输入走 MinerU 的 pdf 分支，接受 -b/-m；docx/pptx/xlsx 走 office 分支，传了会报错
PDF_LIKE_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg")


def run_mineru(src: pathlib.Path, outdir: pathlib.Path,
               backend: str = "pipeline", method: str = "txt") -> None:
    """解析单个文件。

    backend 默认 pipeline 而非 MinerU 自己的默认值 hybrid-engine——后者会加载
    2.2G 的 VLM 模型，本机只有 8GB 显存，要和别的进程抢。
    """
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [MINERU_EXE, "-p", str(src), "-o", str(outdir)]
    if src.suffix.lower() in PDF_LIKE_SUFFIXES:
        cmd += ["-b", backend, "-m", method]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=TIMEOUT_SECONDS, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        # 抛出而非返回错误码，由编排层决定重试或中断整册——失败不能静默
        raise RuntimeError(f"MinerU 解析失败（{src.name}）：{result.stderr[-2000:]}")


def locate_product_md(outdir: pathlib.Path) -> pathlib.Path:
    """递归找产物 markdown。

    目录名随输入类型变化（pdf → txt/、docx → office/），所以必须 rglob。
    找不到直接抛错——整册缺失属于硬失败，返回 None 会让上层静默跳过。
    """
    candidates = [p for p in outdir.rglob("*.md") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"未在 {outdir} 下找到 MinerU 产出的 markdown")
    # 派生产物带后缀、文件名更长；主产物是裸文件名，取最短的那个
    return min(candidates, key=lambda p: len(p.name))
