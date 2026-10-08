"""生成结果的落盘。**本包唯一碰文件系统的地方。**

格式与 S8 的 `data/questions/` 对称：按本地日期分文件、单行 JSON、一次写整行。

---

⚠️ **并发前提（是个前提，不是保证）** —— 与 `backend/query/store.py` 同一取向：

当前 `backend/serve.py` 以单进程、单事件循环启动。本模块的写入路径里**没有
`await`**（`save_result` 是同步函数），因此两个请求的写入不可能交错，一次
`write()` 写入整行是安全的。

**这个保证依赖两个条件，任一被打破就失效：**

    1. 不用 `uvicorn --workers N`；
    2. 不在写入路径中引入 `await`（例如"异步落盘"）。

若将来要打破，MUST 改用文件锁（Windows 上 `msvcrt.locking`）。这段话写在这里
而不是文档里，因为**这里才是加 worker 时会被翻到的地方**。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from . import ANSWERS_DIR
from .result import GenerationResult

logger = logging.getLogger(__name__)

__all__ = ["save_result", "file_for", "list_files", "list_records"]

# 文件名日期用**本地日期**，与 `backend/query/store.py` 的 `DATE_FORMAT` 同值。
#
# ⚠️ 两边必须一致 —— 不一致的话，同一天的提问与回答会落在不同日期命名的文件里，
#    而"查一下今天问了什么、答了什么"这件事就没有一个统一的切分。
DATE_FORMAT = "%Y%m%d"
FILE_SUFFIX = ".jsonl"


def file_for(day: str, root: Path | None = None) -> Path:
    """某一天的结果文件路径。`day` 为 `YYYYMMDD`。"""

    return (root or ANSWERS_DIR) / (day + FILE_SUFFIX)


def save_result(result: GenerationResult, root: Path | None = None) -> Path:
    """追加一条生成结果，返回写入的文件路径。

    ⚠️ **本函数 MUST NOT 向调用方抛出异常。**

    与 `backend/api/capture.py` 的 `capture_question` 同一理由：用户来是为了问答，
    不是为了帮系统写日志。为一次落盘失败把一个已经生成好的回答打成错误，是把
    系统的内部维护成本转嫁给用户。失败时留一条 ERROR 日志 —— 判定标准是
    "事后能不能查出来"，能查出来就不是"吞掉"。
    """

    try:
        directory = root or ANSWERS_DIR
        directory.mkdir(parents=True, exist_ok=True)

        day = datetime.now().astimezone().strftime(DATE_FORMAT)
        path = file_for(day, directory)
        line = json.dumps(result.to_record(), ensure_ascii=False, separators=(",", ":")) + "\n"

        # 一次 write() 写入整行（含末尾换行），MUST NOT 分多次写。
        # 分多次会引入字节交错的窗口 —— 即使当前是单进程（见模块头的并发前提），
        # 这个窗口也没有任何收益去承担。
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)

        logger.debug("generation_saved answer_id=%s file=%s", result.answer_id, path.name)
        return path

    except Exception as exc:  # noqa: BLE001 —— 见函数文档字符串
        logger.error(
            "generation_save_failed answer_id=%s error=%s: %s",
            result.answer_id,
            type(exc).__name__,
            exc,
        )
        return root or ANSWERS_DIR


def list_files(root: Path | None = None) -> list[Path]:
    """列出结果文件，按文件名（即日期）升序。

    读取方 MUST NOT 假设文件名连续 —— 周末、停机日会缺文件。
    """

    base = root or ANSWERS_DIR
    if not base.is_dir():
        return []
    return sorted(p for p in base.glob("*" + FILE_SUFFIX) if p.is_file())


def list_records(root: Path | None = None) -> list[dict]:
    """读回全部结果记录。坏行**跳过并告警**，不像 S8 那样整份报错。

    为什么口径与 `backend/query/store.py` 不同：那份留存是**可重跑的工作队列**，
    一行坏了就必须让人知道，否则会漏处理；这份是**只读的审计副本**，一行坏了
    不该让"查看历史回答"整个功能不可用。两者的用途不同，处置因此不同。
    """

    records: list[dict] = []
    for path in list_files(root):
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            text = raw.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                logger.warning("结果文件有坏行，已跳过：%s 第 %d 行", path.name, lineno)
    return records
