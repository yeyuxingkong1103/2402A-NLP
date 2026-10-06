"""
pdf_content.py — 测试 PDF 的内容定义（汇总入口）

三个领域各 35 个分节，供 generate_test_pdfs.py 渲染。这里只有数据，
不依赖 reportlab，方便单独增删测试语料。

每个分节的元组结构：
    (标题, 正文, 表格或 None, 图形或 None)

正文用 | 分隔多个自然段；图形可选 "bar"（柱状图）与 "flow"（流程图）。

内容来源：所有正文均取自权威法规与临床指南的条款原文或要点整理，
每节上方以 # source 注释标明出处 URL。reportlab Paragraph 将 < 解析为 XML，
故所有字面 < 均写作 &lt;。

三个领域的素材各自放在 pdf_content_legal / _medical / _english 里，
本模块只做汇总导入，保证单文件不超长。
"""

from __future__ import annotations

from pdf_content_english import ENGLISH
from pdf_content_legal import LEGAL
from pdf_content_medical import MEDICAL

__all__ = ["LEGAL", "MEDICAL", "ENGLISH"]
