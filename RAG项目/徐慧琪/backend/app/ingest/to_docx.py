"""把 markdown 逐行搬成 docx，供 MinerU 解析。

存在的理由：MinerU 的 -p 实测只认 pdf/image/docx/pptx/xlsx，喂 .md 直接报
`No supported documents found`，而中册语料只有 md。
产出的 docx 是**合成中间件**，内容零改写——空行也保留成空段落，因为空行是
「款」的分隔信号，丢了会让整条的款合并成一个块。
"""
from __future__ import annotations

import pathlib

import docx


def md_to_docx(md_path: pathlib.Path, out_path: pathlib.Path) -> pathlib.Path:
    """逐行转成 docx 段落，保留空行。返回产出路径。"""
    lines = md_path.read_text(encoding="utf-8").splitlines()
    document = docx.Document()
    for line in lines:
        document.add_paragraph(line)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(out_path)
    return out_path
