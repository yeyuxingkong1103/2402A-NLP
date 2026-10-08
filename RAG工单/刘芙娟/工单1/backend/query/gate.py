"""编码指纹门禁。

**本特性唯一一件"事后无法补救"的事。**

`docs/04_数据管线设计.md` §8 把"向量空间漂移"列为头号风险。它的失效方式是
**静默的**：用同一模型但不同口径编码查询，向量仍然合法、维度仍然对、余弦相似度
仍然算得出来 —— 只是排序结果悄悄变差，没有任何异常。

本期交付的向量**没有任何消费者**（检索模块未实现），所以漂移的代价不会立刻
显现，只会在检索接入时集中爆发。等那时才发现，磁盘上已积压一批口径不明的向量，
且无从判断历史上哪些回答是错的。

因此本模块**只做一件事**：在算任何向量之前，确认查询侧与索引侧的编码参数
逐字段一致；不一致就拒绝，而不是继续算。

⚠️ **纯只读。** 本模块 MUST NOT 写任何文件（FR-017）——
"校验先于写入"只有在校验本身不写的前提下才成立。
"""

import json
import os
from pathlib import Path
from typing import Any

from backend.embed import MAX_LENGTH, MODEL_DIR
from backend.embed.model import fingerprint

from . import EXIT_GATE, INDEX_MANIFEST, QueryError

__all__ = [
    "GATE_FIELDS",
    "load_index_fingerprint",
    "compare_fingerprints",
    "check_gate",
]

# 参与比对的 10 个字段。顺序即报告里的呈现顺序。
#
# 每一项都在挡一类具体的漂移：
#   model_dir / weight_file / weight_bytes / config_sha256  → 换了权重
#   max_length                                              → 改了截断长度
#   pooling / normalization / dtype                         → 改了编码口径
#   dim                                                     → 改了向量维度
#   rule_version                                            → 口径实现本身改了版本
GATE_FIELDS: tuple[str, ...] = (
    "model_dir",
    "config_sha256",
    "weight_file",
    "weight_bytes",
    "max_length",
    "pooling",
    "normalization",
    "dtype",
    "dim",
    "rule_version",
)


def load_index_fingerprint(manifest_path: Path | str = INDEX_MANIFEST) -> dict:
    """读索引清单里的编码指纹（`pipeline_config.embed` 块）。

    清单缺失 / 非法 JSON / 结构不符 —— 三种情况都 MUST 明确失败，
    MUST NOT 跳过校验继续（FR-018）。

    跳过校验的后果比失败严重得多：没有基准的"校验"等于没有校验，
    而它看起来是"通过"的。
    """

    path = Path(manifest_path)
    if not path.is_file():
        raise QueryError(
            EXIT_GATE,
            "索引清单不存在：%s\n"
            "  指纹门禁需要一个比对基准。请先运行 S6 入库"
            "（backend/index_milvus.py）生成清单。" % path,
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise QueryError(
            EXIT_GATE, "索引清单不是合法 JSON：%s —— %s" % (path, exc.msg)
        ) from exc

    embed = (
        data.get("pipeline_config", {}).get("embed")
        if isinstance(data, dict) else None
    )
    if not isinstance(embed, dict):
        raise QueryError(
            EXIT_GATE,
            "索引清单缺少 pipeline_config.embed 块：%s\n"
            "  说明这份清单不是本管线产出的，或结构已变更。" % path,
        )

    return embed


def local_fingerprint(
    model_dir: str = MODEL_DIR, max_length: int = MAX_LENGTH
) -> dict:
    """本进程会用于编码的指纹。

    直接转发 `backend/embed/model.py` 的 `fingerprint()` —— 本模块不重新实现
    任何一项。"查询侧指纹"与"索引侧指纹"必须出自同一个函数，否则比对的是
    两套定义，而它们可能同时自洽却互不兼容。
    """

    return fingerprint(model_dir, max_length)


def compare_fingerprints(index_fp: dict, local_fp: dict) -> list[tuple[str, Any, Any]]:
    """逐字段比对，返回差异项 `[(字段名, 索引侧值, 查询侧值)]`。空列表 = 一致。

    只比对 `GATE_FIELDS` 里的字段，且**只报差异不报缺失**之外的判断 ——
    多余字段（比如索引侧将来新增了一项）不视为不一致：那是索引侧向前兼容，
    不是漂移。
    """

    diffs: list[tuple[str, Any, Any]] = []
    for name in GATE_FIELDS:
        expected, actual = index_fp.get(name, "<缺失>"), local_fp.get(name, "<缺失>")
        if not _same(name, expected, actual):
            diffs.append((name, expected, actual))
    return diffs


def check_gate(
    manifest_path: Path | str = INDEX_MANIFEST,
    model_dir: str = MODEL_DIR,
    max_length: int = MAX_LENGTH,
) -> dict:
    """校验通过则返回汇总信息；不一致则抛 `QueryError(EXIT_GATE, ...)`。

    返回的字典给启动输出与 `verify` 用：`collection` / `total_chunks` /
    `diffs`（必为空）/ `index` / `local`。
    """

    index_fp = load_index_fingerprint(manifest_path)
    local_fp = local_fingerprint(model_dir, max_length)
    diffs = compare_fingerprints(index_fp, local_fp)

    if diffs:
        lines = ["编码指纹不一致，拒绝计算查询向量："]
        for name, expected, actual in diffs:
            lines.append("  · %s：索引侧 %r，查询侧 %r" % (name, expected, actual))
        lines.append(
            "  索引侧来自 %s；查询侧来自当前进程的编码参数。"
            % Path(manifest_path)
        )
        lines.append(
            "  处置：要么用与索引一致的参数重启，要么重新入库"
            "（backend/index_milvus.py）后再启动。"
        )
        raise QueryError(EXIT_GATE, "\n".join(lines))

    summary = _manifest_summary(manifest_path)
    return {
        "index": index_fp,
        "local": local_fp,
        "diffs": [],
        **summary,
    }


def _manifest_summary(manifest_path: Path | str) -> dict:
    """清单里的非指纹信息，仅用于报告（拿不到就留空，不影响门禁结论）。"""

    path = Path(manifest_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 —— 上面已经校验过，这里只是取展示信息
        return {"collection": "?", "total_chunks": "?"}
    if not isinstance(data, dict):
        return {"collection": "?", "total_chunks": "?"}
    return {
        "collection": data.get("collection", "?"),
        "total_chunks": data.get("total_chunks", "?"),
    }


def _same(name: str, expected: Any, actual: Any) -> bool:
    """比较某个字段的两个取值。

    **只有 `model_dir` 做路径归一化**：同一目录可以写成 `E:\\x`、`E:/x`、
    带或不带结尾分隔符 —— 这些是同一个目录，判成不一致只会制造假告警，
    而**假告警会让人开始忽略这个门禁**，那比不设门禁更糟。

    其余字段**严格相等**。特别是 `weight_bytes`：差一个字节都要报 ——
    权重被替换过（哪怕只是重新下载了同一模型）正是要抓的东西。
    对它们做路径归一化是无意义的（它们不是路径），而"顺手对所有字符串
    都归一化"会让 `rule_version` 这类字段的比对语义变得不可名状。
    """

    if name == "model_dir" and isinstance(expected, str) and isinstance(actual, str):
        return (
            os.path.normcase(os.path.normpath(expected.strip()))
            == os.path.normcase(os.path.normpath(actual.strip()))
        )
    return expected == actual
