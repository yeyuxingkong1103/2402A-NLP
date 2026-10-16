# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill 并集成至智能体工作流工单

模块：document_quality_assessment.cli
职责：Skill 的命令行入口（智能体/工作流/人工都可直接调用）。

用法::

    python -m document_quality_assessment --folder <目录> [--limit 200] [--resume/--no-resume]
                                          [--workers 4] [--json out.json] [--html out.html] [--quiet]

退出码：0 成功；2 参数/配置错误；3 运行期异常。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from .config import ConfigError, load_config
from .skill import DocumentQualityAssessmentSkill

EXIT_OK, EXIT_ARGS, EXIT_RUNTIME = 0, 2, 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="document_quality_assessment",
        description="知识库文档质量评估 Skill（格式统计 / PDF 分流 / 长度分布 / MD5+SimHash 重复 / 敏感信息）",
    )
    p.add_argument("--folder", "-f", default=None, help="待质检目录（递归遍历）")
    p.add_argument("--files", "-F", nargs="*", default=None, help="显式文件列表（与 --folder 可同时给）")
    p.add_argument("--config", "-c", default=None, help="配置文件路径（默认 研发/assessment_config.yaml）")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 个文件（抽样口径；0=全量）")
    p.add_argument("--workers", type=int, default=None, help="并行进程数（覆盖配置 performance.max_workers）")
    p.add_argument("--resume", dest="resume", action="store_true", default=True, help="断点续跑（默认开启）")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="忽略 state.json，全量重跑")
    p.add_argument("--json", dest="json_path", default=None, help="JSON 报告输出路径")
    p.add_argument("--html", dest="html_path", default=None, help="HTML 简报输出路径")
    p.add_argument("--no-report", action="store_true", help="只跑分析，不写报告文件")
    p.add_argument("--summary", action="store_true", help="仅打印摘要 JSON")
    p.add_argument("--quiet", action="store_true", help="静默（只落日志到 部署/日志/inspection.log）")
    return p


def _setup_logging(cfg, quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    handlers: list[logging.Handler] = []
    try:
        log_path = cfg.path_of("output.run_log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 日志文件不可写不影响主流程
        pass
    if not quiet:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers or None)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return EXIT_ARGS

    if not args.folder and not args.files:
        print("[参数错误] 至少需要 --folder 或 --files 之一", file=sys.stderr)
        return EXIT_ARGS
    _setup_logging(cfg, args.quiet)

    started = time.time()
    try:
        skill = DocumentQualityAssessmentSkill(cfg)
        result = skill.run(
            folder=args.folder,
            files=args.files,
            resume=args.resume,
            limit=args.limit,
            workers=args.workers,
            write_reports=not args.no_report,
            json_path=args.json_path,
            html_path=args.html_path,
            extra_meta={"cli_argv": argv if argv is not None else sys.argv[1:], "invoked_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        )
    except Exception as exc:  # noqa: BLE001 - CLI 兜底
        print(f"[运行失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME

    summary = result.to_summary()
    if args.quiet:
        pass
    elif args.summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("=" * 74)
        print(f"质检完成，用时 {time.time() - started:.1f}s")
        for k, v in summary.items():
            print(f"  {k:24s} = {v}")
        if result.json_path:
            print(f"  JSON 报告：{result.json_path}")
        if result.html_path:
            print(f"  HTML 简报：{result.html_path}")
        print("=" * 74)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
