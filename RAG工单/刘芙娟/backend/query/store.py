"""留存文件的读写与清理。**本包唯一碰文件系统的地方。**

格式契约见 `specs/007-query-embedding/contracts/store.md`。

---

⚠️ 并发前提（**是个前提，不是保证**）：

当前 `backend/serve.py` 以单进程、单事件循环启动，且本模块的写入路径中
**没有 `await`**，因此两个请求的写入不可能交错 —— 一次 `write()` 写入整行
是安全的。

**这个保证依赖两个条件，任一被打破就失效：**

    1. 不用 `uvicorn --workers N`；
    2. 写入路径中不引入 `await`（例如"异步落盘"）。

若将来要打破，MUST 改用文件锁（Windows 上 `msvcrt.locking`）。

这段话写在这里而不是文档里，是因为**这里才是加 worker 时会被翻到的地方**。
不写的话，后来者会默认它本来就是并发安全的，然后在加 worker 的那天得到一个
只在并发下出现、且破坏的是数据文件本身的问题。
"""

import json
import os
from collections.abc import Callable, Iterator
from datetime import date, datetime
from pathlib import Path

from . import (
    DATE_FORMAT,
    EXIT_DATA,
    F_ASKED_AT,
    FIELD_ORDER,
    FILE_SUFFIX,
    QUESTIONS_DIR,
    QueryError,
)

__all__ = [
    "serialize",
    "append_record",
    "iter_records",
    "rewrite_records",
    "list_files",
    "purge_files",
    "file_for",
    "count_lines",
]


def file_for(day: date, root: Path | None = None) -> Path:
    """某一天的文件路径。文件名用**本地日期**。

    为什么不是 UTC：运维按"今天的问题"找文件时用的是本地时间。按 UTC 切分的话，
    晚上 8 点之后的问题会落到"明天"的文件里，每次找都要换算一次。
    """

    return (root or QUESTIONS_DIR) / (day.strftime(DATE_FORMAT) + FILE_SUFFIX)


def serialize(record: dict) -> str:
    """把一条记录序列化成单行 JSON（**不含换行**）。

    三条约定：
      · `ensure_ascii=False` —— grep 得到的是中文而不是 \\uXXXX
                           （与 backend/api/events.py 同一约定）
      · `separators=(",", ":")` —— 去空格，单行体积小约 5%
      · 字段按 FIELD_ORDER 重排 —— 调用方传进来的顺序不作数，
        写入顺序由契约决定，不由调用方的字典构造顺序决定

    缺失的字段**补 `None` 而不是报错**：写入路径在请求链路上，为一个可选字段
    让用户的提问失败是不划算的。读取方（verify）会严格校验，那才是该报错的地方。
    """

    ordered = {key: record.get(key) for key in FIELD_ORDER}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def append_record(record: dict, root: Path | None = None) -> Path:
    """追加一条记录，返回写入的文件路径。

    文件名日期 **从记录自身的 `asked_at` 推导**，不由调用方另传一个时间。

    为什么不让调用方传：那样会有两个时间来源（记录里的、参数里的），
    而它们必然在某次修改中漂开 —— 漂开的表现是"某条记录出现在它自己
    时间戳之外的那一天文件里"，排查时几乎不会有人想到去核对这两个值。

    ⚠️ **一次 `write()` 写入整行**（含末尾换行），MUST NOT 分多次写。

    分多次写会引入字节交错的窗口 —— 即使当前是单进程（见模块头的并发前提），
    这个窗口也没有任何收益去承担：整行拼好后一次写出，成本一样。
    """

    asked_at = _asked_at_of(record, "append_record")
    path = file_for(asked_at.date(), root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = serialize(record) + "\n"

    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
        fh.flush()

    return path


def iter_records(
    paths: list[Path],
) -> Iterator[tuple[Path, int, dict]]:
    """逐行读取，产出 `(文件, 行号, 记录)`。**行号从 1 起。**

    行号是必需的：`verify` 报错时要指出**哪一行**坏了，否则一个损坏的记录
    在一份上万行的文件里等于找不到。

    空行 MUST 报错而非跳过（contracts/store.md §1）：跳过会让"文件被写坏"
    变成一个永远不会被发现的静默事实。
    """

    for path in paths:
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                text = raw.strip()
                if not text:
                    raise QueryError(
                        EXIT_DATA, "空行：%s 第 %d 行" % (path.name, lineno)
                    )
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise QueryError(
                        EXIT_DATA,
                        "非法 JSON：%s 第 %d 行 —— %s" % (path.name, lineno, exc.msg),
                    ) from exc
                if not isinstance(record, dict):
                    raise QueryError(
                        EXIT_DATA,
                        "不是 JSON 对象：%s 第 %d 行" % (path.name, lineno),
                    )
                yield path, lineno, record


def rewrite_records(
    path: Path, transform: Callable[[int, dict], dict]
) -> int:
    """整文件原子回写。返回被替换的行数。

    ⚠️ 更新已有行 MUST NOT 用追加写（会变成重复记录）。

    流程：读全部 → 逐行 transform → 写 `.tmp` → `os.replace` 原子替换。
    中途失败时**原文件保持完好** —— 这是 `.tmp` + `os.replace` 的全部意义，
    也是与 `backend/embed/store.py` 一致的既有模式。

    transform 返回改写后的记录。**是否算作"已修改"由序列化结果比对决定**，
    不由调用方返回了什么对象决定 —— 见下方说明。

    ⚠️ 变化检测用**序列化比对**，不用对象同一性：

    最初的实现是 `if updated is not record`，看着更省事。但它有一个后果很隐蔽的
    缺陷：调用方只要**构造一个内容相同的新 dict** 返回，就会被记成"已修改"。
    而"重复运行第二次处理 0 条"（SC-009 的幂等验收）正是靠这个计数断言的 ——
    于是幂等性会在实现完全正确的情况下验收失败，或者反过来，在实现有 bug 时
    被一个凑巧相等的计数掩盖。

    **把判断交给"这一行的字节有没有变"，调用方就不必关心返回新对象还是原对象。**
    代价是每行多一次序列化，相对于写文件可以忽略。
    """

    if not path.is_file():
        return 0

    original = path.read_text(encoding="utf-8")
    out_lines: list[str] = []
    changed = 0

    for lineno, raw in enumerate(original.splitlines(), start=1):
        text = raw.strip()
        if not text:
            raise QueryError(EXIT_DATA, "空行：%s 第 %d 行" % (path.name, lineno))
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise QueryError(
                EXIT_DATA,
                "非法 JSON：%s 第 %d 行 —— %s" % (path.name, lineno, exc.msg),
            ) from exc
        before = serialize(record)
        after = serialize(transform(lineno, record))
        if after != before:
            changed += 1
        out_lines.append(after)

    # 没有任何行变化就不碰原文件。否则 `embed` 每跑一次都会重写全部 jsonl
    # （内容字节不变，但 mtime 被刷新）—— 「只处理未完成的记录」在文件层面
    # 也就没做到。跳过写盘比写完再比对更省事，也不会留下 `.tmp`。
    if changed == 0:
        return 0

    tmp = str(path) + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(out_lines) + ("\n" if out_lines else ""))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise

    return changed


def list_files(day: date | None = None, root: Path | None = None) -> list[Path]:
    """列出留存文件，按文件名（即日期）升序。

    读取方 MUST NOT 假设文件名连续 —— 周末、停机日会缺文件。
    """

    base = root or QUESTIONS_DIR
    if not base.is_dir():
        return []
    if day is not None:
        candidate = file_for(day, base)
        return [candidate] if candidate.is_file() else []
    return sorted(p for p in base.glob("*" + FILE_SUFFIX) if p.is_file())


def count_lines(path: Path) -> int:
    """行数（不含末尾空行）。用于 purge 的预演报告。"""

    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def purge_files(
    day: date | None = None,
    before: date | None = None,
    dry_run: bool = True,
    root: Path | None = None,
) -> list[tuple[Path, int, int]]:
    """删除留存文件。返回 `[(文件, 条数, 字节数)]`。

    **默认 `dry_run=True`** —— 破坏性操作必须由人显式发起，与既有管线的
    `--confirm-rebuild` 同一取向（`docs/05` §5）。

    `day` 与 `before` 互斥，由调用方（CLI）保证；两者都为空时**不删任何东西**
    并返回空列表 —— 宁可什么都不做，也不要因为参数漏传把整个目录清空。
    """

    if day is not None:
        targets = list_files(day, root)
    elif before is not None:
        targets = [
            p for p in list_files(None, root)
            if _date_of(p) is not None and _date_of(p) < before
        ]
    else:
        return []

    report: list[tuple[Path, int, int]] = []
    for path in targets:
        size = path.stat().st_size
        lines = count_lines(path)
        report.append((path, lines, size))
        if not dry_run:
            path.unlink()

    return report


def _date_of(path: Path) -> date | None:
    stem = path.stem
    try:
        return datetime.strptime(stem, DATE_FORMAT).date()
    except ValueError:
        return None


def _asked_at_of(record: dict, where: str) -> datetime:
    """从记录里取 asked_at。缺失或无法解析时报 QueryError 而非静默用当天。

    静默用当天会让"这条记录本该落在别的文件里"变成一个永远查不出来的事实。
    """

    raw = record.get(F_ASKED_AT)
    if not isinstance(raw, str):
        raise QueryError(
            EXIT_DATA, "%s：记录的 %s 缺失或不是字符串：%r" % (where, F_ASKED_AT, raw)
        )
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise QueryError(
            EXIT_DATA, "%s：%s 不是合法 ISO 8601：%r" % (where, F_ASKED_AT, raw)
        ) from exc
