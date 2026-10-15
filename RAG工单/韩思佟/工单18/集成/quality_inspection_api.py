# -*- coding: utf-8 -*-
"""
工单18 —— RAGFlow 新增质检 API 端点
POST /v1/document/quality-inspection

落地位置：新建 `ragflow/api/apps/quality_app.py`
注册机制：`api/apps/__init__.py::register_page()` 自动扫描 `*_app.py`，
          路径前缀 `/v1/<name>`，即 `/v1/document/quality-inspection`。
（源码事实：RAGFlow 0.23.1 的 API 层是 Quart，`*_app.py` → `/v1/<name>`，
  `*sdk/*.py` → `/api/v1`；鉴权用 `@login_required`，响应统一 `get_json_result`。）

请求体：
  {"path": "/data/documents", "limit": 200, "format": "json"}
响应：
  {"code":0,"data":{...quality_report...},"message":"success"}
  或 {"code":0,"data":{"html":"<base64>"}} 当 format=html
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

# ---- 在 RAGFlow 容器里，把本 Skill 的 scripts 目录加入 sys.path
SKILL_SCRIPTS = os.getenv(
    "DQA_SCRIPTS",
    "/ragflow/document_quality_assessment/scripts")
if SKILL_SCRIPTS not in sys.path:
    sys.path.insert(0, SKILL_SCRIPTS)

# Quart 风格（RAGFlow 0.23.1 用的是 Quart，不是 Flask）
try:
    from quart import Blueprint, request
    from api.utils.api_utils import get_json_result, server_error_response
    manager = Blueprint("quality", __name__)
except Exception:                                                # noqa: BLE001
    # 便于脱离 RAGFlow 单独做单元测试
    manager, request = None, None

    def get_json_result(data=True, message="success"):
        return {"code": 0, "data": data, "message": message}

    def server_error_response(e):
        return {"code": 500, "message": str(e)}


def _run_quality_inspection(path: str, limit: int | None = None,
                            config: str | None = None) -> dict:
    """调用 Skill 资源层脚本，返回结构化报告"""
    from dqa import (analyze_format_distribution, analyze_length_distribution,
                     detect_duplicates, detect_sensitive_info,
                     classify_documents, build_report)
    from dqa.pdf_type import scan_pdf_types

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在：{path}")

    cfg = _load_cfg(config)
    exts = set(cfg.get("extensions", [".pdf", ".docx", ".doc", ".md", ".txt"]))
    pdf_cfg = cfg.get("pdf", {})

    fmt = analyze_format_distribution(root)
    pdf_t = scan_pdf_types(
        root,
        min_chars_per_page=pdf_cfg.get("min_chars_per_page", 50),
        scan_ratio_threshold=pdf_cfg.get("scan_ratio_threshold", 0.70),
        max_files=limit)
    ln = analyze_length_distribution(root, exts, limit)
    dd = detect_duplicates(root, exts,
                           simhash_threshold=cfg.get("dedup", {}).get(
                               "simhash_threshold", 0.90),
                           max_files=limit)
    se = detect_sensitive_info(root, exts,
                               enable=cfg.get("sensitive", {}).get("enable"),
                               max_files=limit)
    lb = classify_documents(fmt.get("_files", []), pdf_t, ln, dd, se,
                            long_threshold=cfg.get("length", {}).get(
                                "long_doc_threshold", 50000))
    return build_report(root, fmt, pdf_t, ln, dd, se, lb, 0.0, config=cfg)


def _load_cfg(path: str | None) -> dict:
    default = {
        "extensions": [".pdf", ".docx", ".doc", ".md", ".markdown", ".txt",
                       ".xlsx", ".xls", ".csv"],
        "pdf": {"min_chars_per_page": 50, "scan_ratio_threshold": 0.70},
        "length": {"long_doc_threshold": 50000},
        "dedup": {"simhash_threshold": 0.90},
        "sensitive": {"enable": {"手机号": True, "邮箱": True,
                                 "身份证": True, "银行卡": False}},
    }
    if path and Path(path).exists():
        try:
            import yaml
            return {**default, **(yaml.safe_load(Path(path).read_text(
                encoding="utf-8")) or {})}
        except Exception:                                        # noqa: BLE001
            return default
    return default


if manager is not None:
    @manager.route("/quality-inspection", methods=["POST"])
    async def quality_inspection():
        """触发文档质检 Skill，返回结构化报告（JSON 或 HTML 简报）"""
        try:
            req = await request.get_json() or {}
            path = req.get("path") or req.get("folder")
            if not path:
                return get_json_result(data=False, message="缺少参数 path")
            limit = req.get("limit")
            fmt = req.get("format", "json")

            from concurrent.futures import ThreadPoolExecutor
            import asyncio
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as ex:
                rep = await loop.run_in_executor(
                    ex, _run_quality_inspection, path, limit,
                    req.get("config"))

            if fmt == "html":
                from dqa.report import to_html
                html_bytes = to_html(rep).encode("utf-8")
                return get_json_result(
                    data={"html": base64.b64encode(html_bytes).decode()})
            return get_json_result(data=rep)
        except Exception as e:                                   # noqa: BLE001
            return server_error_response(e)

    @manager.route("/quality-inspection/report", methods=["GET"])
    async def quality_report_html():
        """直接返回最近一次报告的 HTML 简报（便于浏览器查看）"""
        p = Path(os.getenv("DQA_LAST_REPORT", "/tmp/quality_report.html"))
        if not p.exists():
            return get_json_result(data=False, message="暂无报告")
        return p.read_text(encoding="utf-8"), 200, {"Content-Type":
                                                    "text/html; charset=utf-8"}


# ------------------------------------------------------------------ 本地自测
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    print(json.dumps(_run_quality_inspection(args.path, args.limit),
                     ensure_ascii=False, indent=2)[:3000])
