# -*- coding: utf-8 -*-
"""工单4 统一问答入口（CLI）：``研发/app/main.py``。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

契约（设计/接口设计.md §2.2）：

    pwsh -NoProfile -File run_py.ps1 研发/app/main.py --ask "武汉力源信息技术股份有限公司注册资本是多少？"
    pwsh -NoProfile -File run_py.ps1 研发/app/main.py --ask "What is ...?" --lang en
    pwsh -NoProfile -File run_py.ps1 研发/app/main.py --ask-file 研发/data/eval/golden_qa.jsonl --out out.jsonl
    pwsh -NoProfile -File run_py.ps1 研发/app/main.py --interactive
    pwsh -NoProfile -File run_py.ps1 研发/app/main.py --serve --port 8123

退出码：``0`` 正常（**含「不清楚」拒答，那是正确行为**）/ ``2`` 参数错误 / ``3`` 索引缺失 /
``4`` 后端不可用且无可降级路径 / ``127`` 解释器缺失（由 run_py.ps1 给出）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):   # 允许 `python 研发/app/main.py` 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import describe_runtime, get_config                    # noqa: E402
from app.core.errors import IndexMissingError, LLMProbeError, RagError      # noqa: E402
from app.core.logging_conf import get_logger, setup_logging                 # noqa: E402
from app.core.qa_engine import QAEngine                                     # noqa: E402
from app.storage.sqlite_manager import SqliteManager                        # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-基于Graph RAG 实现金融问答"

EXIT_OK = 0
EXIT_PARAM = 2
EXIT_INDEX = 3
EXIT_BACKEND = 4


def build_parser() -> argparse.ArgumentParser:
    """CLI 参数表（与 设计/接口设计.md §2.2 一一对应）。"""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=f"工单4：{WORK_ORDER} —— 招股说明书图像内容 RAG 问答（CLI）")
    parser.add_argument("--ask", help="单次提问（中文或英文）")
    parser.add_argument("--ask-file", help="批量提问文件（jsonl，每行含 question，可选 id）")
    parser.add_argument("--out", help="批量结果落盘路径（jsonl）")
    parser.add_argument("--lang", default="auto", choices=["zh", "en", "auto"], help="语言路径（默认 auto）")
    parser.add_argument("--top-k", type=int, default=None, help="检索条数（默认取配置 5；生成提示词固定用前 3 条）")
    parser.add_argument("--interactive", action="store_true", help="交互式多轮会话")
    # 工单5（人工智能NLP-RAG-Query 理解优化）：多轮 Query 理解交互入口。
    # 与 --interactive 的区别：--chat 走 MultiTurnEngine（状态化焦点实体 + 查询改写），
    # 支持「他 / 这个公司 / 那另一家公司呢？」这类依赖上下文的追问。
    parser.add_argument("--chat", action="store_true",
                        help="多轮对话（工单5）：交互式多轮，自动指代消解与查询改写")
    parser.add_argument("--session-id", default=None, help="会话标识（多轮指代消解用）")
    parser.add_argument("--json", action="store_true", help="机器可读输出（单行 JSON）")
    parser.add_argument("--evidence", action="store_true", help="打印命中的证据块")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热（**仅调试**；会引入冷启动延迟）")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存（延迟测量口径用）")
    parser.add_argument("--health", action="store_true", help="打印健康检查后退出")
    parser.add_argument("--retrieve", help="只检索不生成（诊断用）")
    parser.add_argument("--serve", action="store_true", help="启动纯标准库中文 Web 界面（serve_fallback）")
    parser.add_argument("--port", type=int, default=None, help="Web 端口（默认 8123，见 RAG_SERVER__FALLBACK_PORT）")
    parser.add_argument("--host", default=None, help="Web 监听地址（默认 127.0.0.1）")
    parser.add_argument("--run-id", default=None, help="运行号（写入日志）")
    # 工单6（人工智能NLP-RAG-混合检索任务）：检索策略三个入口之一（另外两个是 Web 界面与 HTTP API）。
    # 不传 = 工单4/5 基线行为（原生 hybrid_search）；传了 = 走检索策略层。
    parser.add_argument("--strategy", default=None,
                        choices=["vector", "fulltext", "hybrid", "hybrid_rrf", "hybrid_voting"],
                        help="检索策略预设：vector / fulltext / hybrid(加权平均) / "
                             "hybrid_rrf / hybrid_voting（默认不传 = 工单4/5 基线链路）")
    parser.add_argument("--vector-weight", type=float, default=None,
                        help="混合检索：向量权重（配合 --strategy，默认取预设 0.5）")
    parser.add_argument("--fulltext-weight", type=float, default=None,
                        help="混合检索：全文权重（配合 --strategy，默认取预设 0.5）")
    parser.add_argument("--rerank", default=None, choices=["none", "rule", "tfidf", "cross_encoder"],
                        help="重排模型（配合 --strategy，默认取预设 rule）")
    parser.add_argument("--embedding-model", default=None,
                        help="嵌入模型 id（配合 --strategy，例如 bge-m3 / m3e-base）")
    parser.add_argument("--fulltext-fields", default=None,
                        help="全文检索字段，逗号分隔（配合 --strategy，默认 content；可选 "
                             "content,title,keywords,summary）")
    return parser


def _build_strategy(args: argparse.Namespace):
    """工单6：把 CLI 的 ``--strategy`` 等参数组装成 ``StrategyConfig``；未指定则返回 ``None``。"""
    name = getattr(args, "strategy", None)
    if not name:
        return None
    # 绝对导入：main.py 既可能以包模块运行，也可能被 run_py.ps1 当脚本直接执行
    # （`python 研发/app/main.py`），此时相对导入会抛 "attempted relative import with
    # no known parent package"。文件内其它延迟导入（如 serve_fallback）同样用绝对导入。
    from app.core import retrieval_strategy as rs   # noqa: PLC0415 —— 延迟导入，避免拖慢 --help

    cfg = rs.preset(name)
    fields = None
    raw_fields = getattr(args, "fulltext_fields", None)
    if raw_fields:
        fields = tuple(p.strip() for p in str(raw_fields).replace("，", ",").split(",") if p.strip())
    strategy = cfg.replace(
        vector_weight=getattr(args, "vector_weight", None),
        fulltext_weight=getattr(args, "fulltext_weight", None),
        rerank_model=getattr(args, "rerank", None),
        embedding_model=getattr(args, "embedding_model", None),
        fulltext_fields=fields,
        top_k=(int(args.top_k) if getattr(args, "top_k", None) else None),
    )
    problems = strategy.validate()
    if problems:
        raise SystemExit("检索策略参数不合法：" + "；".join(problems))
    return strategy


def _engine(args: argparse.Namespace, log) -> QAEngine:
    """构造问答引擎（含运行时持久化与可选预热）。

    ⚠️ 工单8 修改（实测踩坑，务必保留）：运行期 SQLite **必须落在本工单自己的目录**。
    原实现用 ``Path(cfg.paths.index_dir) / "qa_runtime.sqlite3"``，而本工单的
    ``RAG_DATA__INDEX_DIR`` 指向**只读的 工单7 索引目录** ⇒ 一旦有人跑 ``app/main.py``，
    就会在 工单7 下创建/写入 ``qa_runtime.sqlite3``（实测确实发生过，见 设计/环境事实.md §3.4）。
    这里改为写到 ``研发/data/runtime/``（本工单内），彻底避免越界写入。
    """
    cfg = get_config()
    runtime_dir = Path(__file__).resolve().parents[1] / "data" / "runtime"
    try:
        runtime_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # noqa: BLE001 —— 目录建不出来时退化为不落盘
        log.log_event("main.runtime_dir_failed", level="WARNING",
                      error=f"{type(exc).__name__}: {exc}", action="问答继续，不落盘历史")
    db_path = runtime_dir / "qa_runtime.sqlite3"
    try:
        storage = SqliteManager(db_path, logger=log)
    except Exception as exc:  # noqa: BLE001 —— 持久化失败不阻断问答（显式降级）
        log.log_event("main.storage_failed", level="WARNING", error=f"{type(exc).__name__}: {exc}",
                      action="问答继续，不落盘历史")
        storage = None
    engine = QAEngine(cfg=cfg, logger=log, storage=storage)
    if not args.no_warmup:
        # 预热必须包含生成模型：qwen2.5:3b 冷启动首字 3430.8 ms（必违约），预热后 23.5~25.2 ms
        # （设计/环境事实.md §8）。--no-warmup 仅用于「冷启动测量」实验，报告中必须标注。
        warm = engine.warmup(with_generate=True)
        log.log_event("main.warmup", **warm)
    return engine


def _print_answer(result, *, as_json: bool, with_evidence: bool) -> None:
    """人类可读 / JSON 两种输出。"""
    if as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False))
        return
    print("─" * 78)
    print(f"问题：{result.question}")
    print(f"答案：{result.answer}")
    if result.citations:
        print("引用：")
        for cite in result.citations:
            print(f"  - [{cite.get('corpus')}: 第{cite.get('page')}页] "
                  f"{cite.get('modality')}"
                  + (f"/{cite.get('modality_sub')}" if cite.get("modality_sub") else "")
                  + f" chunk={cite.get('chunk_id')}")
    if result.refused:
        print(f"拒答：{result.refusal_reason}（{result.refusal_detail}）")
    lat = result.latency_ms
    print(f"耗时：检索 {lat.get('retrieve')} ms / 生成 {lat.get('generate')} ms / "
          f"首字 {lat.get('first_token')} ms / 总计 {lat.get('total')} ms")
    print(f"后端：{result.backend}:{result.model}｜生成方式：{result.generation_method}"
          f"｜缓存：{'命中' if result.cache.get('hit') else '未命中'}｜trace_id={result.trace_id}")
    if with_evidence:
        for hit in result.evidence:
            print(f"  ✦ {hit.get('chunk_id')} [{hit.get('type')}] p{hit.get('page')} "
                  f"score={hit.get('final_score')}")
            print(f"    {str(hit.get('text'))[:200]}…")


def _read_questions(path: Path) -> list[dict]:
    """读批量问题文件（jsonl；每行必须有 ``question``）。"""
    rows: list[dict] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RagError(f"{path} 第 {index} 行不是合法 JSON：{exc}", code="RAG-5100", stage="cli") from exc
        if not str(row.get("question") or "").strip():
            raise RagError(f"{path} 第 {index} 行缺少 question 字段", code="RAG-5100", stage="cli")
        rows.append(row)
    return rows


def _run_batch(engine: QAEngine, args: argparse.Namespace, log) -> int:
    """批量问答（``--ask-file`` + ``--out``）。"""
    path = Path(args.ask_file)
    if not path.is_file():
        print(f"[参数错误] 批量问题文件不存在：{path}", file=sys.stderr)
        return EXIT_PARAM
    rows = _read_questions(path)
    out_path = Path(args.out) if args.out else None
    results: list[dict] = []
    started = time.perf_counter()
    for row in rows:
        question = str(row["question"])
        result = engine.answer(question, lang=args.lang, top_k=args.top_k,
                               session_id=args.session_id, return_evidence=args.evidence,
                               use_cache=not args.no_cache, strategy=_build_strategy(args))
        payload = {"id": row.get("id"), "question": question, "answer": result.answer,
                   "refused": result.refused, "refusal_reason": result.refusal_reason,
                   "citations": result.citations, "cited_pages": result.cited_pages,
                   "latency_ms": result.latency_ms, "backend": result.backend, "model": result.model,
                   "generation_method": result.generation_method, "cache": result.cache,
                   "trace_id": result.trace_id, "language": result.language,
                   "chunks": result.chunks}
        results.append(payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(f"[{row.get('id')}] {question}")
            print(f"    → {result.answer}")
            print(f"    引用={result.cited_pages} 耗时={result.latency_ms.get('total')} ms "
                  f"首字={result.latency_ms.get('first_token')} ms 方式={result.generation_method}")
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for payload in results:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(f"[OK] 已写出 {len(results)} 条结果 → {out_path}")
    total_ms = round((time.perf_counter() - started) * 1000, 2)
    log.log_event("cli.batch_done", questions=len(results), total_ms=total_ms,
                  avg_ms=round(total_ms / max(len(results), 1), 2),
                  refused=sum(1 for r in results if r["refused"]))
    return EXIT_OK


def _run_interactive(engine: QAEngine, args: argparse.Namespace) -> int:
    """交互式多轮问答（``--interactive``；``:q`` 退出，``:new`` 新会话）。"""
    session_id = args.session_id or f"cli-{int(time.time())}"
    print(f"工单4 RAG 问答（交互模式）｜会话 {session_id}｜输入 :q 退出，:new 新建会话")
    while True:
        try:
            question = input("问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return EXIT_OK
        if not question:
            continue
        if question in {":q", ":quit", "exit", "quit"}:
            return EXIT_OK
        if question == ":new":
            session_id = f"cli-{int(time.time())}"
            print(f"已新建会话：{session_id}")
            continue
        result = engine.answer(question, lang=args.lang, top_k=args.top_k, session_id=session_id,
                               return_evidence=args.evidence, use_cache=not args.no_cache,
                               strategy=_build_strategy(args))
        _print_answer(result, as_json=args.json, with_evidence=args.evidence)


def _run_serve(engine: QAEngine, args: argparse.Namespace) -> int:
    """启动纯标准库 Web 界面（共享同一 QAEngine，不复制业务逻辑）。"""
    from app.ui.serve_fallback import serve

    cfg = get_config()
    host = args.host or cfg.server_host
    port = int(args.port or getattr(cfg, "fallback_port", 8123))
    try:
        serve(engine, host=host, port=port)
    except KeyboardInterrupt:
        print("\n[OK] 已停止 Web 服务")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口：参数解析 → 引擎构造 → 分派执行 → 退出码。"""
    args = build_parser().parse_args(argv)
    if args.run_id:
        import os

        os.environ["RAG_RUN_ID"] = str(args.run_id)
    cfg = get_config(refresh=bool(args.run_id))
    log = setup_logging(cfg, run_id=args.run_id)
    log.log_event("cli.start", argv=sys.argv[1:], runtime=describe_runtime())

    actions = [bool(args.ask), bool(args.ask_file), bool(args.interactive), bool(args.serve),
               bool(args.health), bool(args.retrieve), bool(args.chat)]
    if sum(actions) == 0:
        build_parser().print_help()
        return EXIT_PARAM
    if sum(actions) > 1:
        print("[参数错误] --ask / --ask-file / --interactive / --chat / --serve / --health / "
              "--retrieve 只能选一个", file=sys.stderr)
        return EXIT_PARAM
    if args.top_k is not None and not (1 <= int(args.top_k) <= 20):
        print("[参数错误] --top-k 必须在 1~20 之间", file=sys.stderr)
        return EXIT_PARAM

    # 工单5（人工智能NLP-RAG-Query 理解优化）：--chat 转交多轮对话 CLI。
    # 它在自己的进程分支里构造 MultiTurnEngine，避免这里先加载一遍索引再重复加载。
    if args.chat:
        try:
            from app.chat_cli import main as _chat_main
        except ImportError as exc:  # noqa: BLE001
            print(f"[依赖缺失] 无法导入多轮对话 CLI：{exc}", file=sys.stderr)
            return EXIT_DEPS
        chat_argv = ["--chat", "--session-id", str(args.session_id or "cli"), "--lang", args.lang]
        if args.no_warmup:
            chat_argv.append("--no-warmup")
        if args.no_cache:
            chat_argv.append("--no-cache")
        return _chat_main(chat_argv)

    try:
        engine = _engine(args, log)
    except IndexMissingError as exc:
        print(f"[索引缺失] {exc}\n请先运行：pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py --include-images",
              file=sys.stderr)
        return EXIT_INDEX

    try:
        if args.health:
            print(json.dumps(engine.health(), ensure_ascii=False, indent=2))
            return EXIT_OK
        if args.retrieve:
            payload = engine.retrieve_only(args.retrieve, top_k=int(args.top_k or 10))
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return EXIT_OK
        if args.serve:
            return _run_serve(engine, args)
        if args.interactive:
            return _run_interactive(engine, args)
        if args.ask_file:
            return _run_batch(engine, args, log)
        result = engine.answer(args.ask, lang=args.lang, top_k=args.top_k,
                               session_id=args.session_id, return_evidence=args.evidence,
                               use_cache=not args.no_cache, strategy=_build_strategy(args))
        _print_answer(result, as_json=args.json, with_evidence=args.evidence)
        return EXIT_OK
    except IndexMissingError as exc:
        print(f"[索引缺失] {exc}", file=sys.stderr)
        return EXIT_INDEX
    except LLMProbeError as exc:
        print(f"[后端不可用] {exc}", file=sys.stderr)
        return EXIT_BACKEND
    except RagError as exc:
        print(f"[处理失败] {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            log.log_event("cli.exit", run_id=cfg.run_id)
        except Exception:  # noqa: BLE001 —— 退出日志失败不得影响退出码
            pass


if __name__ == "__main__":
    sys.exit(main())
