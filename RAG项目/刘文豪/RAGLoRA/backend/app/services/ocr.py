# -*- coding: utf-8 -*-
"""文档解析分流：按 PDF 类型选择解析器。

清单要求「文档解析（PDF，OCR，pdfplumber，多模态大模型，MinerU，PDFPlumber（表格），
minerU，Paddle-OCR-VL）」。本模块把它们组织成一条**按需分流**的链路，
而不是对每份文档都上最重的解析器。

    PDF ─→ PyMuPDF 抽文本
            │
            ├─ 页均字符数 ≥ 阈值 → **文本层 PDF** → 复用现有分块链路（最快）
            │
            └─ 页均字符数 < 阈值 → **扫描件**（无文本层）
                  ├─ MinerU  (独立 conda env, vlm-engine, GPU)  ← 默认，高精度
                  └─ PaddleOCR (rag_env)                        ← 可选，见下

    表格 → pdfplumber（文本层 PDF 的表格提取）

⚠️ 为什么 MinerU 是默认而不是 PaddleOCR
---------------------------------------
**PaddleOCR 的模型在本机不存在**（`~/.paddlex` 目录为空），
启用它会触发模型下载。而 MinerU 的模型**已在本地**
（`D:\\MinerU\\models`，含 VLM 主模型 2.2GB + OCR + 表格模型，共约 5.6GB），
且 `mineru.json` 已配置 `models-dir` 指向本地路径。

因此：
    * `engine="mineru"`   —— **默认**，零下载
    * `engine="paddleocr"` —— 可选，**首次调用会下载模型**，
      需显式开启：`RAGLORA_OCR_ALLOW_DOWNLOAD=1`

这符合项目的「本地优先，下载需授权」原则。

跨环境调用说明
--------------
MinerU 装在**独立的 conda 环境** `mineru` 里（与后端运行的 `rag_env` 不同），
因为它的依赖会与 rag_env 冲突。所以这里用 `subprocess` 调
`D:\\anaconda3\\envs\\mineru\\Scripts\\mineru.exe`，而不是 import。
父项目 `D:\\桌面\\RAG\\mineru_parse.py` 用的是同一套方式。
"""
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from ..core import config
from ..core.logging import get_logger

log = get_logger("ocr")

# MinerU 可执行文件（独立环境）。可用环境变量覆盖。
MINERU_EXE = os.environ.get(
    "RAGLORA_MINERU_EXE",
    r"D:\anaconda3\envs\mineru\Scripts\mineru.exe",
)
MINERU_BACKEND = os.environ.get("RAGLORA_MINERU_BACKEND", "vlm-engine")

# 是否允许为 OCR 下载模型（PaddleOCR 需要）。默认关。
ALLOW_DOWNLOAD = os.environ.get("RAGLORA_OCR_ALLOW_DOWNLOAD", "0") == "1"

# 判定「扫描件」的页均字符数阈值（见 config.OCR_TEXT_THRESHOLD）
TEXT_THRESHOLD = config.OCR_TEXT_THRESHOLD


def _fitz():
    """延迟导入 PyMuPDF —— 保持模块导入轻量。"""
    import fitz
    return fitz


# ================================================================ 类型判定
def probe_pdf(path: Path, sample_pages: int = 5) -> dict:
    """判断 PDF 是「文本层」还是「扫描件」。

    **抽样前 N 页而不是全读** —— 几百页的 PDF 全读一遍只为算个平均值不划算。
    实测本项目的医疗指南（98 页）抽样 5 页已足以判定。

    返回 {kind, avg_chars_per_page, sampled, total_pages}
    """
    fitz = _fitz()
    doc = fitz.open(str(path))
    try:
        total = len(doc)
        n = min(sample_pages, total)
        chars = 0
        for i in range(n):
            chars += len(doc[i].get_text("text").strip())
        avg = chars / n if n else 0.0
        return {
            "kind": "text" if avg >= TEXT_THRESHOLD else "scanned",
            "avg_chars_per_page": round(avg, 1),
            "sampled": n,
            "total_pages": total,
        }
    finally:
        doc.close()


# ================================================================ 各引擎
def extract_text_layer(path: Path) -> list[tuple[int, str]]:
    """文本层 PDF：交回现有实现（含页眉页脚清洗）。"""
    from .ingest import extract_pdf
    return extract_pdf(path)


def extract_with_mineru(path: Path, timeout: int = 900) -> str:
    """用 MinerU 解析扫描件，返回纯文本。

    MinerU 输出 markdown；这里剥掉常见的 markdown 标记，
    交给下游按段落/条文切分（下游的分块策略是针对纯文本写的）。

    失败时抛异常，由调用方决定降级 —— **不静默返回空串**，
    否则会产出「解析成功但内容是空的」这种最难查的结果。
    """
    if not Path(MINERU_EXE).exists():
        raise FileNotFoundError(f"MinerU 可执行文件不存在: {MINERU_EXE}")

    outdir = Path(tempfile.mkdtemp(prefix="raglora_mineru_"))
    t0 = time.time()
    try:
        cmd = [MINERU_EXE, "-p", str(path), "-o", str(outdir), "-b", MINERU_BACKEND]
        log.info("MinerU 解析中（backend=%s）: %s", MINERU_BACKEND, path.name)
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-800:]
            raise RuntimeError(f"MinerU 退出码 {proc.returncode}: {tail}")

        # MinerU 的输出目录结构：<outdir>/<stem>/auto/<stem>.md
        mds = list(outdir.rglob("*.md"))
        if not mds:
            raise RuntimeError(f"MinerU 未产出 markdown，输出目录内容: "
                               f"{[str(p.relative_to(outdir)) for p in outdir.rglob('*')][:20]}")
        text = mds[0].read_text(encoding="utf-8", errors="replace")
        text = _strip_markdown(text)
        log.info("MinerU 完成 %s | %.0f 字 | %.1fs", path.name, len(text), time.time() - t0)
        return text
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def extract_with_paddleocr(path: Path) -> str:
    """用 PaddleOCR 解析扫描件（可选，需下载模型）。

    ⚠️ 本机 `~/.paddlex` 为空，**首次调用会联网下载模型**。
    因此默认禁用，需 `RAGLORA_OCR_ALLOW_DOWNLOAD=1` 显式开启。
    """
    if not ALLOW_DOWNLOAD:
        raise RuntimeError(
            "PaddleOCR 模型未下载（~/.paddlex 为空）。"
            "如确实要使用，请设 RAGLORA_OCR_ALLOW_DOWNLOAD=1 后重试 —— "
            "首次调用会联网下载模型。"
        )

    # 关掉 PaddleOCR 启动时的联网检查（模型已下时没必要每次都探）
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False)
    results = ocr.predict(str(path))
    lines: list[str] = []
    for res in results:
        # PaddleOCR 3.x 返回的结构里，识别文本在 rec_texts
        for t in (res.get("rec_texts") or []):
            if t:
                lines.append(t)
    return "\n".join(lines)


def extract_tables(path: Path, max_pages: int | None = None) -> list[dict]:
    """用 pdfplumber 抽取表格（可选能力，供后续把表格并入知识库）。

    返回 [{"page": n, "rows": [[cell, ...], ...]}]。
    只对**文本层** PDF 有意义 —— 扫描件要用 MinerU 的表格模型。
    """
    import pdfplumber

    out: list[dict] = []
    with pdfplumber.open(str(path)) as pdf:
        pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
        for i, page in enumerate(pages, 1):
            for tbl in (page.extract_tables() or []):
                rows = [[(c or "").strip() for c in row] for row in tbl if row]
                if len(rows) >= 2:                     # 少于 2 行不算表
                    out.append({"page": i, "rows": rows})
    return out


# ================================================================ 分流入口
def parse_pdf(path: Path, ocr: str = "auto",
              engine: str | None = None) -> tuple[list[tuple[int, str]], dict]:
    """解析 PDF，返回 (pages, meta)。

    `ocr`：
        "auto"  —— 按页均字符数自动判定（默认）
        "force" —— 强制走 OCR
        "off"   —— 强制走文本层（不对的 PDF 会得到空内容）

    `engine`：OCR 引擎，"mineru"（默认）或 "paddleocr"

    返回的 `pages` 与 `ingest.extract_pdf` 同构（[(页码, 文本)]），
    这样下游分块链路无需感知本模块的存在。
    """
    engine = engine or os.environ.get("RAGLORA_OCR_ENGINE", "mineru")
    info = probe_pdf(path)
    use_ocr = (ocr == "force") or (ocr == "auto" and info["kind"] == "scanned")

    meta = {
        "file": path.name,
        "pdf_kind": info["kind"],
        "avg_chars_per_page": info["avg_chars_per_page"],
        "total_pages": info["total_pages"],
        "ocr_requested": ocr,
        "used_ocr": use_ocr,
        "engine": None,
        "tables": 0,
        "elapsed_ms": 0,
    }
    t0 = time.time()

    if not use_ocr:
        meta["elapsed_ms"] = round((time.time() - t0) * 1000)
        return extract_text_layer(path), meta

    # ---- 走 OCR ----
    log.info("%s 判定为扫描件（页均 %.1f 字 < 阈值 %d），启用 %s",
             path.name, info["avg_chars_per_page"], TEXT_THRESHOLD, engine)
    try:
        if engine == "paddleocr":
            text = extract_with_paddleocr(path)
        else:
            text = extract_with_mineru(path)
    except Exception as e:
        # MinerU 失败时，若允许下载则降级到 PaddleOCR（设计文档 §3.5 的降级策略）
        if engine == "mineru" and ALLOW_DOWNLOAD:
            log.warning("MinerU 失败，降级 PaddleOCR: %s", e)
            try:
                text = extract_with_paddleocr(path)
                engine = "paddleocr(降级)"
            except Exception as e2:
                raise RuntimeError(
                    f"OCR 全部失败。MinerU: {e}；PaddleOCR: {e2}"
                ) from e2
        else:
            raise RuntimeError(
                f"OCR 失败（引擎 {engine}）：{e}。"
                f"PaddleOCR 可作为备选，但其模型需下载 —— "
                f"设 RAGLORA_OCR_ALLOW_DOWNLOAD=1 可启用降级。"
            ) from e

    # OCR 结果按「页」切分不了（MinerU 输出的是整篇 markdown），
    # 统一塞进页码 0，表示「来自 OCR、无原始页码」，
    # 下游 format_source 会退化成「来源文件名」而不会显示错误的页码。
    pages = [(0, text)]
    meta["engine"] = engine
    meta["elapsed_ms"] = round((time.time() - t0) * 1000)
    log.info("%s OCR 完成 | %d 字 | %.1fs", path.name, len(text), time.time() - t0)
    return pages, meta


# ================================================================ 工具
_MD_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")           # 图片
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")         # 链接 -> 保留文字
_MD_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]{5,}\|?\s*$")  # 表格分隔行
_MD_MARKS = re.compile(r"^\s{0,3}#{1,6}\s*|^\s*[-*+]\s+|^\s*\d+\.\s+|`{1,3}|\*{1,2}|_{1,2}")


def _strip_markdown(md: str) -> str:
    """把 MinerU 输出的 markdown 转成纯文本，供现有分块链路使用。

    保留表格内容（按行拼平），丢弃图片链接 —— 图片对纯文本 RAG 无价值，
    留着只会变成噪声 token。
    """
    out: list[str] = []
    for line in md.splitlines():
        if _MD_IMG.search(line) and not _MD_IMG.sub("", line).strip():
            continue                                  # 整行只有图片
        if _MD_TABLE_SEP.match(line):
            continue                                  # 表格分隔行
        line = _MD_IMG.sub("", line)
        line = _MD_LINK.sub(r"\1", line)
        line = _MD_MARKS.sub("", line)
        line = line.rstrip()
        if line.strip():
            out.append(line)
    return "\n".join(out)


def available() -> dict:
    """各引擎的可用性（供健康检查与文档）。"""
    return {
        "pymupdf": True,
        "mineru": {
            "ok": Path(MINERU_EXE).exists(),
            "exe": MINERU_EXE,
            "backend": MINERU_BACKEND,
            "models_dir": r"D:\MinerU\models",
        },
        "paddleocr": {
            "ok": ALLOW_DOWNLOAD,
            "note": "模型需下载；设 RAGLORA_OCR_ALLOW_DOWNLOAD=1 启用"
                    if not ALLOW_DOWNLOAD else "已允许（首次调用会下载模型）",
        },
        "pdfplumber": True,
        "text_threshold": TEXT_THRESHOLD,
        "default_engine": os.environ.get("RAGLORA_OCR_ENGINE", "mineru"),
    }
