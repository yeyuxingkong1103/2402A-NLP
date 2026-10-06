# -*- coding: utf-8 -*-
"""T8 三级测试公共库（离线 / 在线 / 用户 共用）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

模块划分（**口径只在这里实现一次**，测试文件只负责编排与断言调用）：
    * ``paths``      —— 路径解析与语料自动发现（禁止硬编码 PDF 文件名）；
    * ``pdf_probe``  —— 只读 PDF 取证（页文本 / 页数 / 表格数），用于引用可回溯核验；
    * ``assertions`` —— 判定口径唯一实现（命中 / 引用 / 首字 / 表格缺陷 / RAGAS 标注）；
    * ``golden``     —— 14 题固定输入与期望（含 evidence_verbatim）加载与完整性校验；
    * ``negatives``  —— 不可答负例集与同页互污染（N-5）用例集。

用法（工作目录 = E:\\gao6gongdan\\工单3）：
    pwsh -NoProfile -File run_py.ps1 -m pytest 测试/离线 -v
"""

from __future__ import annotations

__all__ = ["paths", "pdf_probe", "assertions", "golden", "negatives"]
