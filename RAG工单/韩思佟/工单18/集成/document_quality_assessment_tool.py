# -*- coding: utf-8 -*-
"""
工单18 —— 把质检 Skill 注册成 RAGFlow Agent 的工具（技能）

落地位置：`ragflow/agent/tools/document_quality_assessment.py`
注册机制：`agent/tools/__init__.py` 会自动扫描本目录下所有 .py 并收集类
          （跳过 `__*` 与 `base*`），无需手动 registry。
约定式结构（与 agent/tools/arxiv.py / google.py 完全一致）：
  - `XxxParam(ToolParamBase)`：定义 meta（name/description/parameters）
  - `Xxx(ToolBase)`：定义 `component_name` 与 `_invoke()`
  - `ToolParamBase.get_meta()` 会把 meta 转成 OpenAI function-calling schema
"""
from __future__ import annotations

import abc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# ---- 在 RAGFlow 容器内按下列位置导入
try:
    from agent.tools.base import ToolBase, ToolMeta, ToolParamBase
    from api.utils.api_utils import token_required  # noqa: F401（仅示意）
except Exception:                                                # noqa: BLE001
    # 脱离 RAGFlow 时的最小桩，保证文件可 import、可单测
    class ToolMeta(Dict):
        pass

    class ToolParamBase:                                         # noqa: D101
        def __init__(self):
            self.meta = {
                "name": "document_quality_assessment",
                "description": "文档质量评估",
                "parameters": {},
            }

        def check(self):
            pass

        def get_input_form(self) -> dict:
            return {}

        def get_meta(self):
            return {"type": "function", "function": self.meta}

    class ToolBase:                                              # noqa: D101
        component_name = "DocumentQualityAssessment"

        def _invoke(self, **kwargs):
            raise NotImplementedError

        def get_output_form(self) -> dict:
            return {}

        def description(self) -> str:
            return self._param.meta.get("description", "")


class DocumentQualityAssessmentParam(ToolParamBase):
    """工具参数定义（会被转成 function-calling schema 供 LLM 调用）"""

    def __init__(self):
        self.meta: ToolMeta = {
            "name": "document_quality_assessment",
            "description": (
                "对一个文件夹里的知识库文档做质量评估。"
                "输出：格式分布统计、PDF 文字型/扫描型分流、文档长度分布、"
                "MD5+SimHash 重复检测、敏感信息检测、文档分类标签与解析路由建议。"
                "适合在把文档导入知识库之前做体检。"
            ),
            "parameters": {
                "path": {
                    "type": "string",
                    "description": "待评估的文件夹绝对路径，例如 /data/documents",
                    "default": "{sys.query}",
                    "required": True,
                },
                "limit": {
                    "type": "integer",
                    "description": "最多处理多少个文件（留空表示不限）",
                    "default": 200,
                    "required": False,
                },
            },
        }
        super().__init__()
        self.limit = 200

    def check(self):
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit 必须为正整数或留空")

    def get_input_form(self) -> dict:
        return {
            "path": {"name": "文件夹路径", "type": "line"},
            "limit": {"name": "最多处理文件数", "type": "line"},
        }


class DocumentQualityAssessment(ToolBase, metaclass=abc.ABCMeta):
    """工具主体：调用 Skill 资源层脚本，产出结构化报告"""

    component_name = "DocumentQualityAssessment"

    def _invoke(self, **kwargs) -> Any:
        path = kwargs.get("path") or kwargs.get("sys.query")
        if not path:
            return "缺少参数：path（待评估文件夹路径）"
        limit = kwargs.get("limit") or getattr(self._param, "limit", None)

        # 定位 Skill 脚本目录
        here = Path(__file__).resolve().parent
        candidates = [
            os.getenv("DQA_SCRIPTS", ""),
            str(here.parent.parent / "document_quality_assessment" / "scripts"),
            "/ragflow/document_quality_assessment/scripts",
        ]
        for c in candidates:
            if c and Path(c).exists() and c not in sys.path:
                sys.path.insert(0, c)
                break

        from dqa import (analyze_format_distribution, analyze_length_distribution,
                         detect_duplicates, detect_sensitive_info,
                         classify_documents, build_report)
        from dqa.pdf_type import scan_pdf_types

        root = Path(path)
        if not root.exists():
            return f"目录不存在：{path}"

        fmt = analyze_format_distribution(root)
        pdf_t = scan_pdf_types(root, max_files=limit)
        ln = analyze_length_distribution(root, max_files=limit)
        dd = detect_duplicates(root, max_files=limit)
        se = detect_sensitive_info(root, max_files=limit)
        lb = classify_documents(fmt.get("_files", []), pdf_t, ln, dd, se)
        rep = build_report(root, fmt, pdf_t, ln, dd, se, lb, 0.0)

        ov = rep["总览"]
        summary = (
            f"文件 {ov['文件总数']} 份（{ov['总大小MB']}MB）；"
            f"扫描型PDF {ov['PDF扫描型']}、混合型 {ov['PDF混合型']}、文字型 {ov['PDF文字型']}；"
            f"MD5重复可清理 {ov['MD5精确重复可清理']}；"
            f"版本冲突待确认 {ov['版本冲突待确认']}；"
            f"敏感信息待审核 {ov['敏感信息待审核']}；"
            f"需OCR路由 {ov['需要OCR路由']}"
        )
        pending = {
            "PDF类型待确认": rep["二_PDF页面类型识别"].get("待确认列表", [])[:5],
            "版本冲突待确认": rep["四_重复检测"].get("待确认的版本冲突列表", [])[:5],
            "敏感信息待审核": rep["五_敏感信息检测"].get("待审核敏感信息列表", [])[:5],
        }
        # 供画布下游节点引用
        self.set_output("summary", summary)
        self.set_output("pending", json.dumps(pending, ensure_ascii=False))
        self.set_output("tags", json.dumps(
            lb.get("标签清单", [])[:20], ensure_ascii=False))
        self.set_output("report", json.dumps(rep, ensure_ascii=False)[:8000])
        return summary + "\n\n待确认/待审核：" + json.dumps(pending, ensure_ascii=False)

    def set_output(self, key, value):
        """兼容 RAGFlow 与本地自测的 outputs 写入"""
        if hasattr(self, "_param") and hasattr(self._param, "outputs"):
            self._param.outputs[key] = value
        self.outputs = getattr(self, "outputs", {})
        self.outputs[key] = value

    def get_output_form(self) -> dict:
        return {
            "summary": {"name": "质检结论", "type": "line"},
            "pending": {"name": "待确认/待审核列表", "type": "line"},
            "tags": {"name": "文档标签", "type": "line"},
            "report": {"name": "完整报告 JSON", "type": "paragraph"},
        }
