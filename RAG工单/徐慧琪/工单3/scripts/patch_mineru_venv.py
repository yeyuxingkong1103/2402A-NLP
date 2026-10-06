# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：MinerU venv 补丁（跳过印章 OCR）

背景（本机实测 2026-10-03）：
  MinerU 3.4.5 的 pipeline 在遇到 `label == "seal"` 的版面块时会按需加载印章
  检测权重 `seal_PP-OCRv4_det_server_infer.pth`；本机 PDF-Extract-Kit 快照的
  `models/OCR/paddleocr_torch/` 只有 ch 的 det/rec 两个模型，**没有印章模型**，
  且工单硬约束禁止下载 → 全量解析中途 FileNotFoundError 失败。

补丁内容（仅改 venv 内的副本 `D:\\model\\mineru-venv\\...\\batch_analyze.py`）：
  给印章收集逻辑加环境变量开关 `MINERU_SKIP_SEAL`（默认 "1" = 跳过）。
  跳过时印章块保持"无文本"，不会把印章文字混入正文；如需还原原生行为，
  设 `MINERU_SKIP_SEAL=0`。

用法：python scripts/patch_mineru_venv.py
"""
from __future__ import annotations

import os
import sys

VENV_MINERU = r"D:\model\mineru-venv\Lib\site-packages\mineru"
TARGET = os.path.join(VENV_MINERU, "backend", "pipeline", "batch_analyze.py")

ORIGINAL = """        seal_ocr_items = []
        for ocr_res_list_dict in ocr_res_list_all_page:
            for layout_res_item in ocr_res_list_dict['layout_res']:
                if layout_res_item.get("label") == "seal":
                    seal_ocr_items.append((ocr_res_list_dict, layout_res_item))"""

PATCHED = """        seal_ocr_items = []
        # [工单02 本机补丁 2026-10-03] 印章检测权重 seal_PP-OCRv4_det_server_infer.pth
        # 本机缺失且硬约束禁止下载 → 默认跳过印章 OCR（印章块保持无文本，
        # 不会把印章文字混入正文）。还原原生行为请设 MINERU_SKIP_SEAL=0。
        import os as _os_seal
        if _os_seal.getenv("MINERU_SKIP_SEAL", "1") != "1":
            for ocr_res_list_dict in ocr_res_list_all_page:
                for layout_res_item in ocr_res_list_dict['layout_res']:
                    if layout_res_item.get("label") == "seal":
                        seal_ocr_items.append((ocr_res_list_dict, layout_res_item))"""


def main() -> int:
    if not os.path.isfile(TARGET):
        print(f"× 未找到 venv 内的 MinerU 副本：{TARGET}")
        print("  请先执行：D:\\model\\mineru-venv\\Scripts\\python.exe -m pip install "
              "--no-deps --ignore-installed mineru==3.4.5")
        return 1

    with open(TARGET, encoding="utf-8") as fh:
        content = fh.read()

    if "MINERU_SKIP_SEAL" in content:
        print("√ 补丁已存在，无需重复应用")
        return 0
    if ORIGINAL not in content:
        print("× 未匹配到印章收集代码（MinerU 版本可能不同），未修改")
        return 1

    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(content.replace(ORIGINAL, PATCHED, 1))
    print(f"√ 已打补丁：{TARGET}")
    print("  默认跳过印章 OCR；MINERU_SKIP_SEAL=0 可还原")
    return 0


if __name__ == "__main__":
    sys.exit(main())
