# -*- coding: utf-8 -*-
"""pipeline/pdf_parse.py —— PDF 解析（MinerU 优先，pypdf 兜底）。

在链路中的位置：
    backend/pipeline 构建管线的第一步：把 PDF 变成"按页的纯文本"。

输出统一为 [{"page": 1, "text": "第 1 页文字"}, ...]，
这样后面的清洗与分块就不必关心文本来自哪种解析器。

两个解析器的分工：
    MinerU  复杂版面（表格、公式）提得好，但是外部 CLI、可能失败
    pypdf   只提文本层，胜在稳定不会失败，作为兜底
表格会被转成"竖线分隔"的文本（见 table_to_text），使表 1 里的数值也能参与关键词检索。
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .config import BASE_DIR

# ------------------------------ PDF 解析


def mineru_available() -> bool:
    """探测 MinerU 命令行是否可用。

    返回：
        PATH 中存在 mineru 可执行文件则 True。
    说明：
        用 shutil.which 而不是 try-import，因为 MinerU 是以 CLI 子进程方式调用的。
    """
    return shutil.which("mineru") is not None

def table_to_text(table_html: str) -> str:
    """把 MinerU 输出的 HTML 表格转成"竖线分隔"的纯文本。

    参数：
        table_html: MinerU content_list.json 里的 table_body 字段（HTML 片段）
    返回：
        每行形如 "单元格A | 单元格B" 的多行文本；空表格返回空串。

    为什么需要它：
        表格是标准文档里信息密度最高的部分（如"表 1 净化装置性能参数"），
        纯文本提取会把它压成一堆无法理解的数字。转成"行 | 列"文本后，
        表格数值也能参与 BM25 关键词检索，这正是 V2 迭代里
        "复杂元素补齐（表 1 数值也能检索到）"的实现方式。
    """
    rows = []
    # 按 </tr> 切行；用正则而不是 HTML 解析库，是为了不给这个轻量管线引入额外依赖
    for row in re.split(r"</\s*tr\s*>", table_html or "", flags=re.I):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=re.I | re.S)
        if cells:
            # 去标签 -> 反转义实体（&amp; 之类）-> 去空白，得到单元格纯文本
            values = [html.unescape(re.sub(r"<[^>]+>", "", cell)).strip() for cell in cells]
            rows.append(" | ".join(value for value in values if value))
    return "\n".join(row for row in rows if row)

def mineru_pages(content: list[dict]) -> list[dict]:
    """把 MinerU 的 content_list.json 归一化成统一的"按页分组的纯文本"结构。

    参数：
        content: MinerU 输出的内容块列表，每项含 type / page_idx / text / table_body 等
    返回：
        [{"page": 1, "text": "第 1 页文字"}, ...]，页码从 1 开始（MinerU 内部是 0 起）。
        若解析不出任何内容，返回 [{"page": 1, "text": ""}] 占位，让后续流程不因空输入崩掉。

    为什么只取 text 和 table：
        标题、公式等类型对本检索场景贡献有限，而表格必须保留（见 table_to_text 的说明）。
        归一化之后，后面的清洗/分块就不用再关心 PDF 到底来自 MinerU 还是 pypdf。
    """
    by_page: dict[int, list[str]] = {}
    for item in content:
        if not isinstance(item, dict):
            continue
        page = item.get("page_idx", 0)  # MinerU 的页码是 0 起，后面统一 +1
        if item.get("type") == "text" and item.get("text"):
            by_page.setdefault(page, []).append(item["text"])
        elif item.get("type") == "table":
            # 表题可能是字符串也可能是列表，统一成字符串
            caption = item.get("table_caption") or []
            caption = " ".join(map(str, caption)) if isinstance(caption, list) else str(caption)
            body = table_to_text(item.get("table_body", ""))
            text = "\n".join(part for part in (caption, body) if part)
            if text:
                by_page.setdefault(page, []).append(text)
    return [
        {"page": page + 1, "text": "\n".join(by_page[page]).strip()}
        for page in sorted(by_page)
        if "".join(by_page[page]).strip()  # 丢掉只有空白字符的页，减少后续无意义处理
    ] or [{"page": 1, "text": ""}]

def parse_with_mineru(pdf_path: Path) -> list[dict]:
    """调用 MinerU 命令行解析 PDF。

    参数：
        pdf_path: 待解析 PDF 的完整路径
    返回：
        与 mineru_pages 相同的按页结构。
    异常：
        MinerU 进程返回非 0，或输出里找不到 *_content_list.json，抛 RuntimeError；
        调用方 parse_pdf 会捕获并回退到 pypdf。

    注意：
        每次解析前先清空同名输出目录，因为 MinerU 会往同一目录追加产物，
        残留的旧文件会让 rglob 拿到上一次的结果，出现"改了 PDF 但结果没变"的假象。
    """
    output = BASE_DIR / "data" / "mineru_output" / pdf_path.stem
    if output.exists():
        shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)
    # 超时默认 1800 秒：MinerU 首次运行要下载模型，标准文档页数多时解析也慢
    process = subprocess.run(
        [shutil.which("mineru"), "-p", str(pdf_path), "-o", str(output.parent), "-b", "pipeline", "-m", os.getenv("MINERU_METHOD", "auto")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",  # MinerU 日志可能混入非 UTF-8 字节，用 replace 避免解码直接崩
        timeout=int(os.getenv("MINERU_TIMEOUT", "1800")),
    )
    if process.returncode:
        # 只截取末尾 800 字符：真正的报错信息在最后，前面全是进度刷屏
        detail = (process.stderr or process.stdout or "")[-800:]
        raise RuntimeError(f"MinerU 解析失败：{detail}")
    files = sorted(output.rglob("*_content_list.json"))
    if not files:
        raise RuntimeError("MinerU 输出缺少 content_list.json")
    return mineru_pages(json.loads(files[0].read_text(encoding="utf-8")))

def parse_with_pypdf(pdf_path: Path) -> list[dict]:
    """用 pypdf 做兜底解析（纯文本层提取）。

    参数：
        pdf_path: 待解析 PDF 的完整路径
    返回：
        与 mineru_pages 相同的按页结构。

    定位：
        这是"保证流程能跑通"的底线方案：它提不到表格结构、提不到版面信息，
        但胜在不依赖外部模型、不会失败，所以留作 MinerU 不可用时的退路。
        延迟 import pypdf，是为了没装 pypdf 时也不影响 MinerU 路径。
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return [{"page": number, "text": page.extract_text() or ""} for number, page in enumerate(reader.pages, 1)]

def parse_pdf(pdf_path: Path) -> list[dict]:
    """选择解析器：MinerU 适合复杂版面；没有 MinerU 或解析失败时使用 pypdf。

    参数：
        pdf_path: 待解析 PDF 的完整路径
    返回：
        [{"page": N, "text": "..."}] 的按页结构（两种解析器输出格式已统一）。

    降级策略：
        显式设置 RAG_PARSER=pypdf 可强制走 pypdf；否则只要 MinerU 可用就优先用它。
        MinerU 抛任何异常都被吞掉并回退 —— 单份 PDF 解析失败不应该让整个构建任务中断。
    """
    if os.getenv("RAG_PARSER", "mineru").lower() != "pypdf" and mineru_available():
        try:
            return parse_with_mineru(pdf_path)
        except Exception as exc:
            print(f"[pipeline] MinerU 失败，回退 pypdf：{exc}", file=sys.stderr)  # 打到 stderr，不污染 API 的返回值
    return parse_with_pypdf(pdf_path)
