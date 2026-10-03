"""命令行入口：索引构建 / 问答 / 健康检查 / 统计。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 入口（对应 设计/接口设计.md §2.17、§5）

用法（工作目录 = E:\\gao6gongdan\\工单2）::

    pwsh -NoProfile -File run_py.ps1 研发/app/main.py ask "注册资本是多少？"
    pwsh -NoProfile -File run_py.ps1 研发/app/main.py health
    pwsh -NoProfile -File run_py.ps1 研发/app/main.py stats
    pwsh -NoProfile -File run_py.ps1 研发/app/main.py build --rebuild
    pwsh -NoProfile -File run_py.ps1 研发/app/main.py serve --port 8600   # 备用界面（标准库 http.server）

退出码：0 成功；2 业务失败；3 参数错误。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 允许以 `python 研发/app/main.py` 方式直接运行
SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.errors import RAGError  # noqa: E402
from app.core.logging_conf import flush_logs, logger, setup_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(description="工单2 RAG 问答系统命令行入口")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="向文档提问（流式输出）")
    ask.add_argument("question", help="问题（中文或英文）")
    ask.add_argument("--conversation", default="", help="会话 ID（多轮追问时传入）")
    ask.add_argument("--extractive", action="store_true", help="强制抽取式回答（不调用 LLM）")
    ask.add_argument("--no-stream", action="store_true", help="一次性输出（不逐块打印）")
    ask.add_argument("--json", action="store_true", help="输出 Answer 的 JSON")

    build = sub.add_parser("build", help="构建索引")
    build.add_argument("--pdf", default="", help="PDF 路径")
    build.add_argument("--rebuild", action="store_true", help="强制重建")
    build.add_argument("--max-pages", type=int, default=0, help="仅解析前 N 页")

    sub.add_parser("health", help="打印健康信息（JSON）")
    sub.add_parser("stats", help="打印库表统计（JSON）")
    sub.add_parser("conversations", help="列出会话")

    serve = sub.add_parser("serve", help="启动标准库备用界面（本机演示）")
    serve.add_argument("--port", type=int, default=8600, help="监听端口（0=随机）")
    serve.add_argument("--host", default="127.0.0.1", help="监听地址")
    return parser


def cmd_ask(args: argparse.Namespace) -> int:
    """提问命令。"""
    from app.core.qa_engine import QAEngine

    engine = QAEngine(force_extractive=args.extractive)
    if not engine.load_index():
        print("业务失败：索引未就绪，请先执行 pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py", file=sys.stderr)
        return 2
    conversation_id = args.conversation or None
    if args.json:
        answer = engine.ask(args.question, conversation_id=conversation_id)
        print(json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    print(f"问题：{args.question}")
    for event, payload in engine.stream(args.question, conversation_id=conversation_id):
        if event == "first_token":
            print(f"[首字响应 {payload['first_token_ms']:.0f} ms]", flush=True)
        elif event == "delta" and not args.no_stream:
            print(payload.get("text", ""), end="", flush=True)
        elif event == "citations":
            citations = payload.get("citations", [])
            if citations:
                print("\n引用：", " ".join(f"[页码: {c['page']}] {c['chunk_id']}" for c in citations))
        elif event == "error":
            print(f"\n[错误 {payload['code']}] {payload['message']}", file=sys.stderr)
        elif event == "done":
            answer = payload["answer"]
            print(
                f"\n模式={answer.mode} 首字={answer.first_token_ms:.0f}ms "
                f"端到端={answer.total_ms:.0f}ms 引用数={len(answer.citations)} trace={payload['trace_id']}"
            )
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """构建索引命令（委托给脚本，保持单一实现）。"""
    sys.argv = ["build_index.py"]
    if args.pdf:
        sys.argv += ["--pdf", args.pdf]
    if args.rebuild:
        sys.argv += ["--rebuild"]
    if args.max_pages:
        sys.argv += ["--max-pages", str(args.max_pages)]
    script = SOURCE_ROOT / "scripts" / "build_index.py"
    import runpy

    try:
        runpy.run_path(str(script), run_name="__main__")
        return 0
    except SystemExit as exc:  # 脚本自身退出码透传
        return int(exc.code or 0)


def cmd_health(_args: argparse.Namespace) -> int:
    """健康检查命令。"""
    from app.core.qa_engine import QAEngine

    engine = QAEngine()
    engine.load_index()
    print(json.dumps(engine.health(), ensure_ascii=False, indent=2))
    return 0


def cmd_stats(_args: argparse.Namespace) -> int:
    """统计命令。"""
    from app.core.qa_engine import QAEngine

    engine = QAEngine()
    print(json.dumps(engine.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_conversations(_args: argparse.Namespace) -> int:
    """列出会话。"""
    from app.core.conversation import get_conversation_manager

    manager = get_conversation_manager()
    for conversation in manager.list_conversations():
        print(f"{conversation.conversation_id} | {conversation.title} | 消息 {conversation.message_count} | {conversation.updated_at}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """启动标准库备用界面。"""
    from app.ui.serve_fallback import main as serve_main

    return serve_main(["--host", args.host, "--port", str(args.port)])


def main(argv: list[str] | None = None) -> int:
    """入口分发。"""
    args = build_parser().parse_args(argv)
    setup_logging()
    settings = get_settings()
    try:
        if args.command == "ask":
            return cmd_ask(args)
        if args.command == "build":
            return cmd_build(args)
        if args.command == "health":
            return cmd_health(args)
        if args.command == "stats":
            return cmd_stats(args)
        if args.command == "conversations":
            return cmd_conversations(args)
        if args.command == "serve":
            return cmd_serve(args)
        print(f"未知命令: {args.command}", file=sys.stderr)
        return 3
    except RAGError as exc:
        logger.exception("app.main", "业务失败", code=exc.code)
        print(f"业务失败[{exc.code}]：{exc.message}", file=sys.stderr)
        return 2
    except Exception as exc:
        logger.exception("app.main", "未预期异常")
        print(f"异常：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        print(f"（日志目录：{settings.paths.logs}）", file=sys.stderr)
        flush_logs()


if __name__ == "__main__":
    raise SystemExit(main())
