# app/core/mineru_service.py
"""用 MinerU 解析 PDF —— 独立 venv + 子进程调用。

## 为什么是子进程，而不是当库 import

1. **依赖硬冲突**：MinerU 要求 `openai<3`，本项目主环境是 `openai==3.13.0`
   （`llm_service` / `chain_service` 都在用）。装进主环境就要降级，风险外溢。
2. **内存**：这台机器只有 7.6G。子进程解析完即退出，内存立刻归还；
   常驻服务会一直占着几个 GB。
3. 沿用项目已有的 `evals/.venv` 模式 —— 依赖冲突就用独立环境，先例一致。

## 为什么调用前要先去水印

MinerU 只做解析，**不做水印擦除**。所以流程是：
先用 PyMuPDF 把水印擦掉、另存一份干净 PDF，再交给 MinerU。
这一步在 `pdf_service` 里编排，本模块只管「拿到干净 PDF → 还你 Markdown」。

## 档位选择：basic

实测同一份 28 页 WHO 报告：

    改造前（PyMuPDF + PaddleOCR + pdfplumber）  14830 字 / 254 秒
    MinerU basic（小模型跑 GPU）                 15003 字 /  26 秒

小模型（版面 / OCR / 公式 / 表格）走 **torch 后端跑在 GPU 上**，
28 页峰值显存约 0.9GB。

`standard` 档要多跑一个 **VLM**，而 VLM 引擎目前只有 CPU 版的 llama.cpp
（没装 vLLM，它在 Linux 上会拉一整套重型依赖）—— 单次加载就要约 361 秒，
而子进程模式**每次调用都要重付**这笔开销，因此不实用。

> **CUDA 版本要对齐（踩过的坑）**：本机驱动 566.36 只支持到 CUDA 12.7，
> 而 `pip install mineru` 默认拉到的 torch 是 cu130 构建，
> `torch.cuda.is_available()` 会是 `False`，MinerU 随即静默退回 onnx/CPU
> （表现为解析慢一倍多）。需按 README 换成 cu126 的 torch，
> 并用 `mineru-kit models show` 确认 `Effective small backend` 是 `torch`。
"""
import os
import re
import subprocess
import tempfile
from typing import Dict, Optional, Tuple

from app.config import settings
import logging

logger = logging.getLogger(__name__)

# MinerU 会把图内嵌成 base64 塞进 Markdown。实测 28 页文档嵌了 25 张、约 7MB，
# 这些数据对检索毫无价值，必须剥掉再入库，否则每次上传都灌进几 MB 垃圾。
_IMG_MD_RE = re.compile(r"!\[[^\]]*\]\(\s*data:image/[^)]*\)")
_IMG_HTML_RE = re.compile(r'<img[^>]*src="data:image/[^"]*"[^>]*/?>', re.I)
# 剥完图后可能留下的大段空白
_BLANK_RE = re.compile(r"\n{3,}")


def binary() -> Optional[str]:
    """MinerU 可执行文件路径；未安装时返回 None（调用方据此降级）。"""
    path = settings.MINERU_BIN
    if path and os.path.exists(path):
        return path
    return None


def available() -> bool:
    return binary() is not None


def strip_images(markdown: str) -> Tuple[str, int]:
    """剥掉 base64 内嵌图，返回 (干净文本, 剥掉的图片数)。"""
    n = len(_IMG_MD_RE.findall(markdown)) + len(_IMG_HTML_RE.findall(markdown))
    out = _IMG_MD_RE.sub("", markdown)
    out = _IMG_HTML_RE.sub("", out)
    out = _BLANK_RE.sub("\n\n", out)
    return out.strip(), n


def _env_for_subprocess() -> Dict[str, str]:
    """子进程环境。

    必须显式指定 modelscope：这台机器 huggingface.co 直连不通
    （实测返回 000），MinerU 默认的 auto 策略会先去探测 HF 再回退，
    平白多等一轮超时。
    """
    env = dict(os.environ)
    env["MINERU_MODEL_SOURCE"] = settings.MINERU_MODEL_SOURCE
    return env


def parse(pdf_path: str, tier: Optional[str] = None,
          timeout: Optional[int] = None) -> Tuple[str, Dict]:
    """把 PDF 交给 MinerU，返回 (Markdown, 统计字典)。

    失败一律抛异常，由调用方决定是否降级 —— 本模块不吞错。
    """
    exe = binary()
    if exe is None:
        raise RuntimeError("MinerU 未安装：找不到 %s" % settings.MINERU_BIN)

    tier = tier or settings.MINERU_TIER
    timeout = timeout or settings.MINERU_TIMEOUT

    with tempfile.TemporaryDirectory(prefix="mineru_") as tmp:
        out_md = os.path.join(tmp, "out.md")
        cmd = [exe, "parse", pdf_path, "--pages", "all",
               "--tier", tier, "-o", out_md]
        logger.info("调用 MinerU（档位 %s）: %s", tier, os.path.basename(pdf_path))

        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=_env_for_subprocess())
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-500:]
            raise RuntimeError("MinerU 退出码 %d：%s" % (proc.returncode, tail))

        if not os.path.exists(out_md):
            raise RuntimeError("MinerU 未产出 Markdown：%s" % out_md)

        with open(out_md, encoding="utf-8") as f:
            raw = f.read()

    text, dropped = strip_images(raw)
    stat = {
        "tier": tier,
        "raw_chars": len(raw),
        "text_chars": len(text),
        "images_dropped": dropped,
    }
    logger.info("MinerU 解析完成：%d 字（剥掉 %d 张内嵌图）",
                stat["text_chars"], dropped)
    return text, stat
