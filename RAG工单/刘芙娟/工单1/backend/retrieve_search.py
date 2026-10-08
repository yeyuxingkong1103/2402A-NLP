"""S9 混合检索的命令行入口。**本特性唯一的进程外接口。**

运行方式（MUST 用模块形式，以仓库根为工作目录）：

    D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search <子命令> [选项]

不要用 `rag/python.exe backend/retrieve_search.py` —— 那样 `sys.path[0]` 会变成
`backend/` 而不是仓库根，`import backend.retrieve` 会失败。既有管线的
`-m backend.serve` / `-m backend.query_embed` 约定同理（docs/05 §5）。

契约见 specs/008-hybrid-retrieval/contracts/cli.md。

---

**本文件只做 CLI 骨架**（参数解析、分派、退出码映射）。三个子命令的实现分别在：

    backend/retrieve/report.py     输出渲染
    backend/retrieve/checkup.py    自检（纯函数断言 + 语料一致性）
    backend/retrieve/service.py    检索编排（**与服务端共用同一份**）
    backend/retrieve/cli_support.py 装配（参数 + .env → 索引 / 查询向量）

---

## 为什么必须有这个 CLI

constitution 的 `TODO(SIMILARITY_THRESHOLD)` 至今未回填 —— 阈值从未标定过。
标定要做的是"对若干条已知答案的问题反复试跑、看两路分数分布"，这个过程
MUST 能脱离 Web 服务与浏览器完成。CLI 不是服务端点的复制品，它是**标定工具**。

`--no-semantic` / `--no-lexical` 因此也**只是标定开关**，不是降级开关：
它们让"混合比单路好多少"这件事可测量。**服务端 MUST NOT 暴露它们** ——
单向降级正是 spec 的 Edge Case 禁止的静默失效。
"""

import argparse
import sys
import traceback

from backend.retrieve import (
    EXIT_ARGS,
    EXIT_DATA,
    EXIT_DEP,
    EXIT_DESCRIPTION,
    EXIT_OK,
    RetrievalError,
    assert_dim_matches_index_package,
)
from backend.retrieve import checkup, cli_support, report, selfcheck
from backend.retrieve import service as retrieve_service

__all__ = ["build_parser", "main"]


class _Parser(argparse.ArgumentParser):
    """参数错误的退出码改为 1。

    ⚠️ 与 `backend/query_embed.py` 的同名类同一理由，这里同样成立：
    `argparse` 默认在参数错误时 `sys.exit(2)`，而本契约里 **2 = 数据/校验问题**。
    两者撞在一起，"选项写错了"与"语料与清单不一致"会返回同一个退出码 ——
    自动化脚本会去查一个不存在的问题，而真正的问题（参数写错）被掩盖。
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        print("%s：%s" % (self.prog, message), file=sys.stderr)
        raise SystemExit(EXIT_ARGS)


def _add_tuning_options(parser: argparse.ArgumentParser) -> None:
    """`search` 与 `corpus` 共用的检索参数覆盖项（默认取 `.env`）。"""

    parser.add_argument(
        "--candidates", type=int, default=None,
        help="每路候选数（覆盖 RETRIEVAL_CANDIDATES）",
    )
    parser.add_argument(
        "--rrf-k", type=int, default=None, help="RRF 平滑常数（覆盖 RRF_K）"
    )
    parser.add_argument(
        "--admit-rank", type=int, default=None,
        help="关键词名次准入线（覆盖 LEXICAL_ADMIT_RANK）",
    )
    parser.add_argument(
        "--min-coverage", type=float, default=None,
        help="关键词准入所需的查询词覆盖率（覆盖 LEXICAL_MIN_COVERAGE）",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="backend.retrieve_search",
        description="医知源 · 混合检索（语义 + BM25 关键词）",
    )
    parser.add_argument("--traceback", action="store_true",
                        help="出现未预期异常时打印完整堆栈")
    sub = parser.add_subparsers(dest="command", metavar="<子命令>")

    p_search = sub.add_parser("search", help="对单条问题执行混合检索")
    p_search.add_argument("question", help="要检索的问题原文")
    p_search.add_argument("--top-k", type=int, default=None,
                          help="返回条数（覆盖 TOP_K）")
    p_search.add_argument("--threshold", type=float, default=None,
                          help="语义路余弦阈值（覆盖 SIMILARITY_THRESHOLD）")
    p_search.add_argument("--json", action="store_true",
                          help="以 JSON 输出 RetrievalResult")
    p_search.add_argument("--no-semantic", action="store_true",
                          help="仅标定用：跳过语义路（同时不加载 BGE-M3）")
    p_search.add_argument("--no-lexical", action="store_true",
                          help="仅标定用：跳过关键词路（复现纯语义基线）")
    _add_tuning_options(p_search)
    p_search.set_defaults(func=cmd_search)

    p_corpus = sub.add_parser("corpus", help="语料自检：条数、字段完整性、与清单一致性")
    _add_tuning_options(p_corpus)
    p_corpus.set_defaults(func=cmd_corpus)

    p_self = sub.add_parser("selfcheck", help="纯函数自检（不连库、不加载权重）")
    p_self.set_defaults(func=cmd_selfcheck)

    return parser


def cmd_selfcheck(args: argparse.Namespace) -> int:
    """纯函数自检。**不连 Milvus、不加载模型权重、不读 `.env`。**"""

    failures, lines = selfcheck.run_selfcheck()
    for line in lines:
        print(line)
    return EXIT_OK if failures == 0 else EXIT_DATA


def cmd_corpus(args: argparse.Namespace) -> int:
    """语料自检：连库拉语料并报告一致性。"""

    index = cli_support.build_index(args)
    failures, lines = checkup.check_corpus(index)
    print("uri: %s" % cli_support.uri())
    report.render_corpus(index, failures, lines)
    return EXIT_OK if failures == 0 else EXIT_DATA


def cmd_search(args: argparse.Namespace) -> int:
    """对单条问题执行混合检索。"""

    if args.no_semantic and args.no_lexical:
        print("错误：--no-semantic 与 --no-lexical 互斥，只能给一个。", file=sys.stderr)
        return EXIT_ARGS

    index = cli_support.build_index(args)

    top_k = _read(args.top_k, cli_support.config_value("top_k"))
    threshold = _read(args.threshold, cli_support.config_value("similarity_threshold"))
    if top_k is None or threshold is None:
        print(
            "错误：缺少 top_k / similarity_threshold 配置，且命令行未给出。\n"
            "  请检查 .env（可参考 .env.example），或加 --top-k / --threshold。",
            file=sys.stderr,
        )
        return EXIT_ARGS

    # 语义路被跳过时**不加载权重** —— 标定关键词路不该先等 10 秒。
    query_vector = [] if args.no_semantic else cli_support.encode(args.question)

    trace = retrieve_service.search_traced(
        args.question,
        query_vector,
        top_k,
        threshold,
        index=index,
        skip_semantic=args.no_semantic,
        skip_lexical=args.no_lexical,
    )

    if args.json:
        report.render_search_json(trace)
    else:
        report.render_search(
            trace,
            question=args.question,
            threshold=threshold,
            source_note=cli_support.source_note(index, args),
        )
    return EXIT_OK


def _read(override, configured):
    return override if override is not None else configured


def _force_utf8_output() -> None:
    """把标准输出/错误切到 UTF-8。

    Windows 控制台默认 cp936，中文报告会输出成乱码。**读不了的报告等于没跑** ——
    而这份报告正是标定阈值时唯一要看的东西。与 `backend/serve.py`、
    `backend/query_embed.py` 的同名函数是同一件事，刻意不共用（调用时机不同，
    抽出去只多一条依赖，而收益是三行代码）。
    """

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()

    # 维度声明自检。它很便宜（不需要 torch、不连库），而维度漂移的后果
    # （维度对不上却不报错）正是本特性要防的那一类。
    try:
        assert_dim_matches_index_package()
    except RetrievalError as exc:
        print("错误：%s" % exc.message, file=sys.stderr)
        return exc.code

    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        # 不带子命令直接运行：打印帮助并以参数错误退出。
        # 不用 `required` 子命令，是为了让退出码可控（1 而非 argparse 默认的 2）。
        parser.print_help()
        return EXIT_ARGS

    try:
        return args.func(args)
    except RetrievalError as exc:
        print("错误：%s" % exc.message, file=sys.stderr)
        print("（%s）" % EXIT_DESCRIPTION.get(exc.code, "未知"), file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        print("已中断。", file=sys.stderr)
        return EXIT_ARGS
    except Exception as exc:  # noqa: BLE001 —— 兜底，避免把堆栈直接抛给使用者
        print("未预期的错误：%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        if args.traceback:
            traceback.print_exc()
        return EXIT_DEP if isinstance(exc, ImportError) else EXIT_DATA


if __name__ == "__main__":
    raise SystemExit(main())
