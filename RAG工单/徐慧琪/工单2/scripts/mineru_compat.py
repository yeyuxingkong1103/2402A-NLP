# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：MinerU VLM 后端兼容层（CLI 包装）

背景（本机实测，2026-10-03）：
  1. MinerU 2.7.6 的 pipeline 后端需要 `models/Layout/YOLO/*.pt` 与 `models/MFD/YOLO/*.pt`
     两个权重，本机 PDF-Extract-Kit 快照缺失，且工单硬约束禁止下载 → 只能用 VLM 后端。
  2. VLM 后端在本机 transformers 5.15 下有两处 Windows 兼容问题：
       a. 导入 `transformers.models.qwen2_vl.*` 时会触发深层导入链，在小栈主线程上
          静默崩溃（access violation，无 traceback）→ 与工单1 bootstrap.py 记录的问题同源；
       b. transformers 5.x 的 Qwen2VLConfig 移除了 `max_position_embeddings` 属性，
          而 mineru_vl_utils 0.2.8 的 TransformersVlmClient 需要读取它。
  对策：本脚本 = bootstrap（大栈线程 + 导入链预热）+ 属性补丁 + 原样调用 mineru CLI。
  不修改 site-packages；模型路径来自本地 mineru.json，不触发任何下载。

用法（参数与 `mineru` CLI 完全一致）：
  python scripts/mineru_compat.py -p 招股说明书1.pdf -o out -s 128 -e 128 -b vlm-auto-engine
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src import bootstrap  # noqa: E402  —— 必须最先导入（离线环境变量 + 大栈 + 预热）


def _patched_main() -> None:
    """在大栈线程中执行：先补 transformers 5.x 兼容属性，再原样调用 mineru CLI。"""
    try:
        from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig

        if not hasattr(Qwen2VLConfig, "max_position_embeddings"):
            # Qwen2.5-VL 位置编码上限 32768+；mineru_vl_utils 仅用它作 max_length 上界
            Qwen2VLConfig.max_position_embeddings = 32768
    except Exception as exc:  # noqa: BLE001
        print(f"[mineru_compat] 兼容补丁未生效（VLM 后端可能失败）: {exc}",
              file=sys.stderr, flush=True)

    # 关键补丁：mineru_vl_utils 在未显式给出 max_new_tokens 时用
    # `max_length = model_max_length`（=32768）作为生成长度上限。实测该设置下
    # 单页推理极慢（一小时量级未收敛）。这里强制改为 max_new_tokens=2048
    # （单页 markdown 输出的合理上限），保证生成在有限步数内结束。
    try:
        from mineru_vl_utils.vlm_client.transformers_client import TransformersVlmClient

        _orig_build = TransformersVlmClient.build_generate_kwargs

        def _build_generate_kwargs(self, sampling_params):  # noqa: ANN001
            kwargs = _orig_build(self, sampling_params)
            if "max_length" in kwargs:
                kwargs.pop("max_length")
                kwargs["max_new_tokens"] = 2048
            return kwargs

        TransformersVlmClient.build_generate_kwargs = _build_generate_kwargs
    except Exception as exc:  # noqa: BLE001
        print(f"[mineru_compat] 生成长度补丁未生效: {exc}", file=sys.stderr, flush=True)

    from mineru.cli.client import main  # console_scripts: mineru = mineru.cli.client:main

    main()


if __name__ == "__main__":
    bootstrap.run_with_large_stack(_patched_main)
