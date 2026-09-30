"""PaddleOCR-VL 兜底：只对 MinerU 判定低质的页触发。

存在的理由（技术方案 3.2 ④）：MinerU 强在版面与结构，扫描件或图版页的文字层它救不回来；
PaddleOCR-VL 强在"看图出字"，但整篇结构不行。两者是"结构 + 兜底"关系，不是重复投入。
触发式调用能显著压住显存与时间——全量跑 VLM 在 8GB 卡上会和在线推理抢资源。

⚠️ 本路径在现有三册语料上**不会触发**：三册文本层都干净，实测 1260/1260 条零缺号。
因此 run_paddleocr 是未经真实数据检验的代码，首次遇到扫描件时必须先小样验证再入库。
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess

# 实测位置。技术方案写 paddlepaddle-gpu 2.6.2，实机是 3.3.1，以实机为准
PADDLEOCR_EXE = r"D:\Model\Paddle-OCR\venv\Scripts\paddleocr.exe"
VL_MODEL_DIR = r"D:\Model\Paddle-OCR\official_models\PaddleOCR-VL"
MODEL_CACHE_HOME = r"D:\Model\Paddle-OCR"

# 正常页约 400 汉字；低于此值判为"几乎无文本层"
MIN_CHARS_PER_PAGE = 80
# 替换字符占比超过此值判为乱码
REPLACEMENT_RATIO = 0.1

PAGE_MARKER = re.compile(r"<!--\s*page\s+(\d+)\s*-->", re.IGNORECASE)


def low_quality_pages(md_text: str) -> list[int]:
    """返回需要 OCR 兜底的页码列表。

    无 page 标记时返回空——不做猜测性触发，避免在没有页信息时全量跑 VLM。
    """
    boundaries = list(PAGE_MARKER.finditer(md_text))
    if not boundaries:
        return []
    flagged: list[int] = []
    for idx, marker in enumerate(boundaries):
        start = marker.end()
        end = boundaries[idx + 1].start() if idx + 1 < len(boundaries) else len(md_text)
        body = "".join(md_text[start:end].split())   # 去空白后按字符数判定
        if len(body) < MIN_CHARS_PER_PAGE:
            flagged.append(int(marker.group(1)))
            continue
        if body.count("�") / len(body) > REPLACEMENT_RATIO:
            flagged.append(int(marker.group(1)))
    return flagged


def _build_env() -> dict[str, str]:
    """构造子进程环境。

    必须显式指定模型来源与缓存目录：本机 HuggingFace 不可达（实测连不通），
    默认源会导致下载超时；模型已在本地随 PaddleOCR-VL 一并落盘。
    """
    env = dict(os.environ)
    env["MODEL_SOURCE"] = "modelscope"
    env["CACHE_HOME"] = MODEL_CACHE_HOME
    return env


def run_paddleocr(image_paths: list[pathlib.Path], outdir: pathlib.Path) -> dict[str, str]:
    """对给定页面图片跑 PaddleOCR-VL，返回 {图片名: 识别文本}。

    PaddleOCR 装在独立 venv，走 subprocess。单页失败不抛出——技术方案 3.2 ④ 定的是
    "以 MinerU 为准，OCR 只补缺"，兜底失败不应拖垮整册入库，但会记进报告。
    """
    outdir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    for img in image_paths:
        cmd = [PADDLEOCR_EXE, "doc_parser", "-i", str(img),
               "--vl_rec_model_dir", VL_MODEL_DIR,
               "--save_path", str(outdir)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=_build_env(),
                              encoding="utf-8", errors="replace", timeout=600)
        if proc.returncode == 0:
            results[img.name] = proc.stdout
    return results
