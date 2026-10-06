# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：MinerU CLI 入口（在独立 venv `D:\\model\\mineru-venv` 中运行）

为什么需要独立 venv：
  MinerU 3.4.5（模型清单与本机 PDF-Extract-Kit 快照完全匹配：PP-DocLayoutV2、
  unimernet、paddleocr_torch、TabRec/TabCls）依赖 transformers 4.x 的旧 API
  （如 `pytorch_utils.find_pruneable_heads_and_indices`），而主环境是
  transformers 5.15（该 API 已被移除）。
  venv 用 `--system-site-packages` 复用主环境的 CUDA torch，仅隔离
  transformers 及其直接依赖，主环境（工单1/工单2 其余模块）不受影响。

用法（参数与 mineru CLI 完全一致，仅解释器换成 venv）：
  D:\\model\\mineru-venv\\Scripts\\python.exe scripts/mineru_run.py \
      -p 招股说明书1.pdf -o out -b pipeline
"""
from __future__ import annotations

import sys


def main() -> None:
    from mineru.cli.client import main as mineru_main  # click 命令（读 sys.argv）
    mineru_main()


if __name__ == "__main__":
    main()
