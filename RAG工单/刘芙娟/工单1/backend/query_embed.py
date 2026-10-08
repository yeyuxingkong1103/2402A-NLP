"""S8 提问留存的命令行入口。**本特性唯一的进程外接口。**

运行方式（MUST 用模块形式，以仓库根为工作目录）：

    D:/zg6_Project/9/med_rag/rag/python.exe -m backend.query_embed <子命令> [选项]

不要用 `rag/python.exe backend/query_embed.py` —— 那样 `sys.path[0]` 会变成
`backend/` 而不是仓库根，`import backend.query` 会失败。既有管线的
`-m backend.pipeline` 约定同理（docs/05 §5）。

契约见 specs/007-query-embedding/contracts/cli.md。

---

**本文件只做 CLI 骨架**（参数解析、分派、退出码映射）。四个子命令的实现分别在：

    backend/query/batch.py      embed   批量重算（幂等）
    backend/query/report.py     verify  只读自检
    backend/query/maintain.py   purge / show

这样切分是为了守住 FR-023 的 300 行上限 —— 四个子命令全塞在一个文件里
会到 438 行。**入口仍然唯一**（`-m backend.query_embed`），拆的是实现，
不是接口。
"""

import argparse
import sys
import traceback
from datetime import date, datetime

from backend.query import DATE_FORMAT, EXIT_ARGS, EXIT_DATA, QueryError
from backend.query.batch import cmd_embed
from backend.query.maintain import cmd_purge, cmd_show
from backend.query.report import cmd_verify

# 值取自 backend/query/__init__.py 的 EXIT_*；这里只做「常量名 → 人话」的映射，
# 不重新定义数值。
EXIT_DESCRIPTION = {
    0: "成功",
    1: "参数错误",
    2: "门禁失败（编码参数不一致）",
    3: "模型不可用",
    4: "数据异常",
}


# ---- 参数解析 ----

def _parse_date(text: str) -> date:
    try:
        return datetime.strptime(text, DATE_FORMAT).date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "日期应为 YYYYMMDD 形式（如 20260927），收到 %r" % text
        ) from exc


class _Parser(argparse.ArgumentParser):
    """参数错误的退出码改为 1。

    ⚠️ 这一条不是洁癖，是修一个真实的契约冲突：

    `argparse` 默认在参数错误时 **`sys.exit(2)`**，而本契约里
    **2 = 门禁失败**（与 `docs/05` §5 的「2 校验失败」对齐）。两者撞在一起，
    于是"日期格式写错了"与"查询侧指纹和索引对不上"会返回同一个退出码 ——
    自动化脚本会去查一个根本不存在的问题，而真正的问题（参数写错）被掩盖。

    参数错误在本契约里是 **1**（`EXIT_ARGS`），因此把 `error()` 的出口改掉。
    这需要继承而不是改常量：`sys.exit(2)` 写死在 CPython 的 `argparse` 里。
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        print("%s：%s" % (self.prog, message), file=sys.stderr)
        raise SystemExit(EXIT_ARGS)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="backend.query_embed",
        description="医知源 · 提问留存的批量处理与维护",
    )
    # 未预期异常默认只打印类型与消息（可读），堆栈按需索取 ——
    # 排障时想要，日常使用时不想被几十行刷屏。
    parser.add_argument("--traceback", action="store_true",
                        help="出现未预期异常时打印完整堆栈")
    sub = parser.add_subparsers(dest="command", metavar="<子命令>")

    p_embed = sub.add_parser("embed", help="批量向量化尚未完成的记录")
    p_embed.add_argument("--date", type=_parse_date, default=None,
                         help="只处理某一天（YYYYMMDD）")
    p_embed.add_argument("--dry-run", action="store_true",
                         help="只报告将处理多少条，不写")
    p_embed.add_argument("--limit", type=int, default=None,
                         help="最多处理 N 条（调试用）")
    p_embed.set_defaults(func=cmd_embed)

    p_verify = sub.add_parser("verify", help="只读自检：门禁 + 格式 + 向量")
    p_verify.set_defaults(func=cmd_verify)

    p_purge = sub.add_parser("purge", help="按日期清理留存（破坏性，默认预演）")
    p_purge.add_argument("--date", type=_parse_date, default=None,
                         help="删除指定日期")
    p_purge.add_argument("--before", type=_parse_date, default=None,
                         help="删除该日期之前的所有留存")
    p_purge.add_argument("--dry-run", action="store_true",
                         help="只打印将删除什么（不给 --yes 时本来就是预演）")
    p_purge.add_argument("--yes", action="store_true",
                         help="确认真删。MUST 显式给出")
    p_purge.set_defaults(func=cmd_purge)

    p_show = sub.add_parser("show", help="打印某条记录的可读摘要")
    p_show.add_argument("--answer-id", required=True, help="回答标识（UUID）")
    p_show.set_defaults(func=cmd_show)

    return parser


def _force_utf8_output() -> None:
    """把标准输出/错误切到 UTF-8。

    Windows 控制台默认 cp936，中文报告与错误信息会输出成乱码。**读不了的
    报告等于没跑** —— 排障时第一步就是看 `verify` 说了什么。

    与 `backend/serve.py` 的 `_force_utf8_output` 是同一件事，刻意不共用：
    两处的调用时机不同（这里是进程启动，那里必须在 uvicorn 配置日志之前），
    抽成一个共享函数只会让依赖关系多一条，而收益是三行代码。
    """

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        # 不带子命令直接运行：打印帮助并以参数错误退出。
        # 不用 `parse_args` 的 required 子命令，是为了让退出码可控（1 而非 2）。
        parser.print_help()
        return EXIT_ARGS

    try:
        return args.func(args)
    except QueryError as exc:
        print("错误：%s" % exc.message, file=sys.stderr)
        print("（%s）" % EXIT_DESCRIPTION.get(exc.code, "未知"), file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        print("已中断。", file=sys.stderr)
        return EXIT_ARGS
    except Exception as exc:  # noqa: BLE001 —— 兜底，避免把堆栈直接抛给使用者
        # 未预期异常：打印类型与消息（可读），堆栈只在需要时通过 --traceback 给出。
        print("未预期的错误：%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        if args.traceback:
            traceback.print_exc()
        return EXIT_DATA


if __name__ == "__main__":
    raise SystemExit(main())
