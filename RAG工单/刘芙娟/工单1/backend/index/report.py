"""终端报告渲染。纯输出，不碰库、不碰文件。

按显示宽度对齐而不是 `len()` —— 汉字占 2 列，用 `len()` 算会让中文标签的
冒号对不齐（`backend/embed_chunks.py` 里踩过同一个坑）。
"""

from __future__ import annotations

import os
import sys
import unicodedata
from typing import Any

STEP_TOTAL = 6
LABEL_WIDTH = 20


def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def say(text: str = "") -> None:
    print(text, flush=True)


def step(number: int, text: str) -> None:
    say("[%d/%d] %s" % (number, STEP_TOTAL, text))


def field(label: str, value: Any) -> str:
    return "  %s%s %s" % (label, " " * max(1, LABEL_WIDTH - display_width(label)), value)


def hint(text: str) -> None:
    say("      · %s" % text)


def diff_block(lines: list[str]) -> None:
    say("      差异：")
    for line in lines:
        say("    " + line)


def env_block(uri: str, collection: str, python_version: str, pymilvus_version: str,
              numpy_version: str, token_env: str, token_set: bool) -> None:
    say("===== 运行环境 =====")
    say(field("解释器", python_version))
    say(field("pymilvus", pymilvus_version))
    say(field("numpy", numpy_version))
    say(field("Milvus 地址", uri))
    say(field("collection", collection))
    # 只报变量名与是否设置，**不回显取值**（宪法原则 III）
    say(field(token_env, "已设置" if token_set else "未设置（按无凭据连接）"))


def env(uri: str, collection: str, token_env: str) -> None:
    """打印运行环境。只读依赖的版本号，**不连库** —— 所以依赖未装时也能跑。"""
    import numpy

    try:
        import pymilvus

        pymilvus_version = str(getattr(pymilvus, "__version__", "未知"))
    except ModuleNotFoundError:
        pymilvus_version = "未安装 —— 安装：rag/python.exe -m pip install pymilvus"
    env_block(
        uri=uri,
        collection=collection,
        python_version=sys.version.split()[0],
        pymilvus_version=pymilvus_version,
        numpy_version=numpy.__version__,
        token_env=token_env,
        token_set=bool(os.environ.get(token_env)),
    )


def config_block(config: dict[str, Any], config_hash: str) -> None:
    """打印本次的参数清单本身（不只哈希）—— 门禁失败时要靠它逐项定位差异。"""
    say("===== 本次 pipeline_config =====")
    for key in sorted(config):
        if key != "embed":
            say(field(key, config[key]))
    say(field("embed", ""))
    for key in sorted(config["embed"]):
        say("    %-16s %s" % (key, config["embed"][key]))
    say("")
    say(field("pipeline_config_hash", config_hash))


def summary(collection: str, uri: str, created: bool, index_type: str, metric_type: str,
            dim: int, written: int, total: int, config_hash: str,
            doc_lines: list[str], manifest_path: str | None) -> None:
    say("")
    say("===== 汇总 =====")
    say(field("collection", "%s（%s）" % (collection, "本次新建" if created else "已存在")))
    say(field("Milvus", uri))
    say(field("索引 / 度量", "%s / %s" % (index_type, metric_type)))
    say(field("维度", dim))
    say(field("本次写入", "%d 行" % written))
    say(field("库内总行数", total))
    say(field("pipeline_config_hash", config_hash))
    say(field("索引清单", manifest_path or "（未写入）"))
    say("  结果：")
    for line in doc_lines:
        say("    %s" % line)


def failure(doc_id: str, message: str) -> None:
    say("      ✗ %s：%s" % (doc_id, message))
