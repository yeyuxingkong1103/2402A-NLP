"""MinerU 结果解析：把「结果压缩包 + 接口返回」解析成正文与页数。

从 `app.models.mineru` 按职责拆出的解析层（批次 25-2）。职责边界：
- 只做解析，**不发任何网络请求、不读客户端状态**；
- 输入是 zip 字节流（已下载好的 `zipfile.ZipFile`）与接口返回的 dict，
  输出是正文 `str` 与页数 `int`；
- 因此可以只用内存构造的 zip 做单测，不需要 mock 任何传输。

为什么单独成文件（拆分动机，不是"为拆而拆"）：
`app/models/mineru.py` 原本 446 行，超过 `docs/目录与命名约定.md` §3.4 的
单文件 300 行上限。本模块的四个函数彼此内聚，且与"提交 / 上传 / 轮询 / 下载"
的流程代码零耦合 —— 是最自然的、风险最低的一刀。

调用关系：`MineruClient.parse()` 在拿到结果包后调用 `extract_markdown()` 与
`resolve_page_count()`；后者内部又用 `find_entry()` / `count_pdf_pages()`。
"""

from __future__ import annotations

import json
import zipfile
from typing import Any


def find_entry(names: list[str], target: str) -> str | None:
    """在压缩包条目里找一个文件名，容忍带目录前缀（如 sub/layout.json）。

    参数：
        names:  `ZipFile.namelist()` 的全部条目名
        target: 要找的文件名（不带目录）
    返回：
        命中的条目名；没有时返回 None。同名多份时取路径最短的那个
        （最靠近包根，通常是 MinerU 的主产物而非备份副本）。
    """
    exact = [name for name in names if name == target]
    if exact:
        return exact[0]
    nested = [name for name in names if name.endswith(f"/{target}")]
    return sorted(nested, key=len)[0] if nested else None


def count_pdf_pages(payload: bytes) -> int:
    """用 PyMuPDF 数 PDF 页数；依赖缺失或文件损坏时返回 0（不阻断主流程）。

    参数：
        payload: PDF 字节流（取自结果包内的 *_origin.pdf）
    返回：
        页数；拿不到时返回 0（调用方据此继续退化到别的页数来源）。
    """
    try:
        import fitz
    except ImportError:  # pragma: no cover - 依赖缺失时的降级
        return 0
    try:
        with fitz.open(stream=payload, filetype="pdf") as document:
            return int(document.page_count)
    except Exception:  # noqa: BLE001 - 页数只用于统计，任何异常都降级为 0
        return 0


def extract_markdown(archive: zipfile.ZipFile) -> str:
    """从结果包里取出 full.md 作为正文。

    参数：
        archive: 已打开的 MinerU 结果包
    返回：
        Markdown 正文（UTF-8 解码，坏字节用 replace 而不是抛错 ——
        正文里偶发的乱码不该让整篇文档失败）
    异常：
        ValueError: 包内没有 full.md（调用方会收敛成 complete=False）
    """
    names = archive.namelist()
    candidates = [
        name
        for name in names
        if name == "full.md" or name.endswith("/full.md")
    ]
    if not candidates:
        raise ValueError(f"结果包内未找到 full.md（含 {len(names)} 个文件）")
    # 同名多份时取路径最短的那个（最靠近包根）
    target = sorted(candidates, key=len)[0]
    return archive.read(target).decode("utf-8", errors="replace")


def resolve_page_count(result: dict, archive: zipfile.ZipFile, trace: dict[str, Any]) -> int:
    """解析页数，按可靠性从高到低取第一个可用的来源。

    批次 22 实测：MinerU v4 的 extract_result **不一定带 extract_progress**
    （本次返回里就没有），而 vlm 后端的结果包也**不含页图**，所以不能只靠
    这两个来源。实际可用的顺序：

    1. `extract_progress.total_pages`（接口给出时最直接）
    2. `layout.json` 的 `pdf_info` 长度（每个元素一页，含 page_idx）
    3. 结果包内 `*_origin.pdf` 的页数（用 PyMuPDF 数，最准但要读文件）
    4. 结果包内页图数量（老后端会把每页导成 png/jpg）

    参数：
        result:  轮询拿到的 extract_result 条目
        archive: 结果包
        trace:   诊断轨迹（记录实际命中的来源，便于事后核对）
    返回：
        页数；四个来源都不可用时返回 0（页数只用于统计，不阻断正文）
    """
    progress = result.get("extract_progress")
    if isinstance(progress, dict):
        total = progress.get("total_pages")
        if isinstance(total, int) and total > 0:
            trace["events"].append({"step": "page_count", "source": "extract_progress"})
            return total

    names = archive.namelist()

    layout_name = find_entry(names, "layout.json")
    if layout_name:
        try:
            layout = json.loads(archive.read(layout_name).decode("utf-8"))
            pdf_info = layout.get("pdf_info") if isinstance(layout, dict) else None
            if isinstance(pdf_info, list) and pdf_info:
                trace["events"].append(
                    {"step": "page_count", "source": "layout.json", "pages": len(pdf_info)}
                )
                return len(pdf_info)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            pass  # layout.json 坏了不影响正文，继续退化

    origin_name = next(
        (name for name in names if name.endswith("_origin.pdf")), None
    )
    if origin_name:
        pages = count_pdf_pages(archive.read(origin_name))
        if pages:
            trace["events"].append(
                {"step": "page_count", "source": "origin.pdf", "pages": pages}
            )
            return pages

    images = [
        name
        for name in names
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if images:
        trace["events"].append(
            {"step": "page_count", "source": "page_images", "pages": len(images)}
        )
        return len(images)

    trace["events"].append({"step": "page_count", "source": "(无可用来源)"})
    return 0
