"""应用启动入口。

提供三种运行方式：

1. 启动网页界面（默认，等价于 ``scripts/run_app.sh``）::

       python -m app.main
       python -m app.main web --port 8501

2. 命令行问答（便于快速验证与脚本化测试）::

       python -m app.main ask "武汉兴图新科电子股份有限公司注册资本是多少？"
       python -m app.main ask --interactive

3. 环境与索引自检::

       python -m app.main doctor

设计原则：入口只做参数解析与组件编排，业务逻辑全部在 ``app/core`` 中，
因此同一套逻辑既被网页界面复用，也被测试脚本复用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许 `python -m app.main` 与 `python 研发/app/main.py` 两种方式。
#
# 本文件位于 <root>/研发/app/main.py，因此：
#   parents[0] = app    parents[1] = 研发（需要加入 sys.path，才能 import app.*）
#   parents[2] = 项目根（定位 app/ui、data、logs 等）
SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging_conf import log_stage, logger, setup_logging  # noqa: E402


def cmd_web(args: argparse.Namespace) -> int:
    """启动 Streamlit 界面。"""
    import subprocess

    settings = get_settings()
    app_path = SOURCE_ROOT / "app" / "ui" / "streamlit_app.py"
    if not app_path.exists():
        logger.error("app.main", "界面文件不存在", path=str(app_path))
        print(f"[错误] 界面文件不存在: {app_path}", file=sys.stderr)
        return 2

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    log_stage("阶段3", "启动网页界面", port=args.port, address=args.address)
    print(f"界面地址: http://{args.address}:{args.port}")
    print(f"知识库  : {settings.paths.sqlite_path}")
    try:
        return subprocess.call(command, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\n已停止。")
        return 0


def build_engine(force_extractive: bool = False):
    """构造问答引擎（延迟导入，避免 doctor 以外的命令加载重依赖）。"""
    from app.core.qa_engine import QAEngine

    return QAEngine(force_extractive=force_extractive)


def cmd_ask(args: argparse.Namespace) -> int:
    """命令行问答。"""
    engine = build_engine(force_extractive=args.extractive)
    stats = engine.stats()
    if not stats["index_ready"]:
        print("[错误] 索引未就绪，请先执行: python scripts/build_index.py", file=sys.stderr)
        return 3

    def answer_once(question: str, conversation_id: str | None) -> None:
        result = engine.ask(question, conversation_id=conversation_id, allow_llm=not args.extractive, save=True)
        print("\n" + "=" * 60)
        print(f"问题: {question}")
        print(f"回答: {result.answer}")
        if result.citations:
            print("引用:")
            for citation in result.citations:
                print(f"  {citation.label()} {citation.snippet[:80]}")
        print(
            f"[模式={result.mode} 首字={result.first_token_ms:.0f}ms "
            f"总耗时={result.total_ms:.0f}ms 片段={result.retrieved_count} 页码={result.pages}]"
        )
        print("=" * 60)

    if args.interactive:
        conversation_id = engine.new_conversation(title="命令行会话")
        print("进入交互模式，输入问题后回车；输入 :q 退出，输入 :clear 清空会话。")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                break
            if not question:
                continue
            if question in {":q", ":quit", "exit", "quit"}:
                print("已退出。")
                break
            if question == ":clear":
                engine.clear_conversation(conversation_id)
                print("会话已清空。")
                continue
            answer_once(question, conversation_id)
        return 0

    if not args.question:
        print("[错误] 请提供问题，或使用 --interactive", file=sys.stderr)
        return 2
    answer_once(" ".join(args.question), None)
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    """环境自检：依赖、索引、组件降级原因。"""
    setup_logging()
    settings = get_settings()
    report: dict[str, object] = {"python": sys.version.split()[0], "project_root": str(PROJECT_ROOT)}

    # 依赖检查
    dependencies = [
        "numpy",
        "pymupdf",
        "loguru",
        "jieba",
        "sqlalchemy",
        "pydantic",
        "chromadb",
        "sentence_transformers",
        "streamlit",
        "openai",
        "ragas",
    ]
    availability: dict[str, str] = {}
    for module in dependencies:
        try:
            __import__(module)
            availability[module] = "OK"
        except Exception as exc:
            availability[module] = f"MISSING ({type(exc).__name__})"
    report["dependencies"] = availability

    # 文档与索引
    from app.storage.sqlite_manager import get_sqlite_manager

    store = get_sqlite_manager()
    documents = store.list_documents()
    report["documents"] = [
        {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "pages": doc.page_count,
            "chunks": doc.chunk_count,
            "tables": doc.table_count,
            "status": doc.status,
        }
        for doc in documents
    ]
    report["sqlite"] = str(settings.paths.sqlite_path)

    engine = build_engine()
    report["stats"] = engine.stats()
    report["retriever"] = engine.retriever.health()

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not engine.stats()["index_ready"]:
        print("\n[下一步] 索引未就绪，请运行: python scripts/build_index.py")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.main", description="基于 PDF 文档的 RAG 问答系统")
    sub = parser.add_subparsers(dest="command")

    web = sub.add_parser("web", help="启动 Streamlit 网页界面")
    web.add_argument("--port", type=int, default=8501)
    web.add_argument("--address", type=str, default="0.0.0.0")
    web.set_defaults(func=cmd_web)

    ask = sub.add_parser("ask", help="命令行问答")
    ask.add_argument("question", nargs="*", help="要提问的问题")
    ask.add_argument("--interactive", action="store_true", help="进入交互模式")
    ask.add_argument("--extractive", action="store_true", help="强制使用抽取式回答（不调用 LLM）")
    ask.set_defaults(func=cmd_ask)

    doctor = sub.add_parser("doctor", help="环境与索引自检")
    doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        # 默认行为：启动界面
        args = parser.parse_args(["web"])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
