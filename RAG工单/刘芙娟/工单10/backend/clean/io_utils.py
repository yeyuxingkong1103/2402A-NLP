"""产物写出。

一律先写临时文件、成功后 os.replace 改名——中途失败不会留下可被误认为
完整的 blocks.jsonl（User Story 4 / Edge Cases）。
"""

from __future__ import annotations

import json
import os

from .errors import CleanError, EXIT_FAIL


def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, records) -> int:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    count = 0
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False))
                fh.write("\n")
                count += 1
        os.replace(tmp, path)
    except OSError as exc:
        _cleanup(tmp)
        raise CleanError(EXIT_FAIL, "写出失败：%s\n  %s" % (path, exc)) from exc
    return count


def write_text(path: str, content: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except OSError as exc:
        _cleanup(tmp)
        raise CleanError(EXIT_FAIL, "写出失败：%s\n  %s" % (path, exc)) from exc


def _cleanup(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
