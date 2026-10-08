"""读四份产物并做 V1–V12 校验。

**本模块不写任何东西**（不写库、不写文件）。这是 FR-005「一切校验先于任何写操作」
的结构保证 —— 写库的能力只存在于 store.py。

校验清单见 specs/005 的 data-model.md §4。其中两条最容易写错：

* **V2 行对齐必须逐条比**，不能退化为集合比较。集合相等而顺序错位时，向量会配上
  错误的文本 —— 检索命中后引用会指向别的段落。这是本步骤最危险的输入错误。
* **V10 按字节判长**，不是按字符。Milvus 的 VARCHAR max_length 以 UTF-8 字节计，
  而本语料的中文占 3 字节/字、表格里的 ASCII 占 1 字节，实测平均 1.9。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import (
    DIM,
    EXIT_VALIDATION,
    MAX_LENGTHS,
    META_KEY_PATTERN,
    META_MAX_BYTES,
    META_REQUIRED_KEYS,
    SCALAR_FIELDS,
    IngestError,
)

# S5 承诺的 L2 归一化容差（specs/004 的 NORM_TOLERANCE）
NORM_TOLERANCE = 1e-5

# 已经是独立列的 chunk 字段，不再重复进 `chunk_meta`。
# 由 SCALAR_FIELDS 推出而非手写 —— 加列时排除集自动跟上，不会两边不一致。
_COLUMN_KEYS = frozenset(SCALAR_FIELDS)

_META_KEY_RE = re.compile(META_KEY_PATTERN)


def build_chunk_meta(chunk: dict[str, Any]) -> dict[str, Any]:
    """把 S4 记录里**没有单独成列**的那部分收进一个 dict，供 Milvus 的 JSON 字段。

    是「整条记录减去已有列」而不是手写白名单：S4 将来新增字段会自动进来，S6 不用改。
    键按字典序排列，保证同一份产物每次生成的 JSON 完全一致（便于比对）。
    """
    return {key: chunk[key] for key in sorted(chunk) if key not in _COLUMN_KEYS}


@dataclass
class DocInputs:
    """一份文档的全部输入，已通过校验。"""

    doc_id: str
    file_name: str
    source_hash: str
    chunk_rule_version: str
    matrix: np.ndarray  # float32[N, 1024]
    rows: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    fingerprint: dict[str, Any]

    @property
    def count(self) -> int:
        return len(self.chunks)

    @property
    def page_count(self) -> int:
        return max(int(c["page_end"]) for c in self.chunks)


def _fail(check: str, message: str) -> None:
    raise IngestError(EXIT_VALIDATION, "[%s] %s" % (check, message))


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        _fail("输入", "产物不存在：%s\n（请先运行上游步骤：S4 分块 / S5 向量化）" % path)
    try:
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        _fail("输入", "读取失败：%s\n  %s" % (path, exc))
    if not rows:
        _fail("输入", "产物为空：%s" % path)
    return rows


def resolve_docs(doc_ids: list[str], chunks_dir: str) -> list[str]:
    """位置参数留空 = 处理 data/chunks/ 下的全部文档（D3，沿用 embed_chunks.py 的约定）。"""
    if doc_ids:
        return list(doc_ids)
    if not os.path.isdir(chunks_dir):
        _fail("输入", "分块产物目录不存在：%s" % chunks_dir)
    suffix = ".chunks.jsonl"
    found = sorted(
        name[: -len(suffix)]
        for name in os.listdir(chunks_dir)
        if name.endswith(suffix)
    )
    if not found:
        _fail("输入", "分块产物目录下没有 *.chunks.jsonl：%s" % chunks_dir)
    return found


def load_doc(doc_id: str, chunks_dir: str, emb_dir: str) -> DocInputs:
    """按 **精确文件名** 读取 —— 所以 S5 遗留的 {doc_id}.npy.tmp.npy 天然被忽略（E9）。"""
    chunk_path = os.path.join(chunks_dir, "%s.chunks.jsonl" % doc_id)
    npy_path = os.path.join(emb_dir, "%s.npy" % doc_id)
    rows_path = os.path.join(emb_dir, "%s.rows.jsonl" % doc_id)
    fp_path = os.path.join(emb_dir, "%s.fingerprint.json" % doc_id)

    chunks = _read_jsonl(chunk_path)
    rows = _read_jsonl(rows_path)

    if not os.path.isfile(npy_path):
        _fail("V11", "向量矩阵不存在：%s" % npy_path)
    try:
        matrix = np.load(npy_path)
    except (OSError, ValueError) as exc:
        _fail("V11", "读取向量矩阵失败：%s\n  %s" % (npy_path, exc))

    if not os.path.isfile(fp_path):
        _fail("输入", "模型指纹不存在：%s\n（S5 的产物不完整，请重跑 S5）" % fp_path)
    try:
        with open(fp_path, encoding="utf-8") as fh:
            fingerprint = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _fail("输入", "读取模型指纹失败：%s\n  %s" % (fp_path, exc))

    validate(doc_id, matrix, rows, chunks)
    return DocInputs(
        doc_id=doc_id,
        file_name=str(chunks[0]["file_name"]),
        source_hash=str(chunks[0]["source_hash"]),
        chunk_rule_version=str(chunks[0]["chunk_rule_version"]),
        matrix=matrix,
        rows=rows,
        chunks=chunks,
        fingerprint=fingerprint,
    )


def validate(
    doc_id: str,
    matrix: np.ndarray,
    rows: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> None:
    """V1–V12。任一不过即抛 IngestError（退出码 2），调用方此时尚未连接 Milvus。"""

    # ---- V11：矩阵契约 ----
    if matrix.dtype != np.float32:
        _fail("V11", "向量矩阵 dtype 应为 float32，实际 %s" % matrix.dtype)
    if matrix.ndim != 2 or matrix.shape[1] != DIM:
        _fail("V11", "向量矩阵形状应为 (N, %d)，实际 %s" % (DIM, tuple(matrix.shape)))

    # ---- V1：行数一致 ----
    if matrix.shape[0] != len(rows):
        _fail(
            "V1",
            "向量行数与对齐表行数不等：npy %d 行 vs rows.jsonl %d 行" % (matrix.shape[0], len(rows)),
        )
    if len(rows) != len(chunks):
        _fail(
            "V1",
            "对齐表行数与分块数不等：rows.jsonl %d 行 vs chunks.jsonl %d 行" % (len(rows), len(chunks)),
        )

    # ---- V5：L2 归一化（COSINE 度量下非归一化向量会让打分失真）----
    norms = np.linalg.norm(matrix, axis=1)
    bad_norm = [
        (i, float(norms[i]))
        for i in range(len(norms))
        if abs(float(norms[i]) - 1.0) > NORM_TOLERANCE
    ]
    if bad_norm:
        sample = "、".join("第 %d 行 ‖v‖=%.6f" % (i, n) for i, n in bad_norm[:5])
        _fail(
            "V5",
            "%d 行的模长不是 1（容差 %g）：%s\n（S5 的 L2 归一化被破坏，请重跑 S5）"
            % (len(bad_norm), NORM_TOLERANCE, sample),
        )

    # ---- 逐条：V4 / V2 / V3 / V6 / V7 / V8 / V9 / V10 ----
    seen: set[str] = set()
    hashes: set[str] = set()
    rule_versions: set[str] = set()

    for i, (row, chunk) in enumerate(zip(rows, chunks)):
        # V4：row_index 必须与行号一致（.npy 取值的依据）
        if int(row.get("row_index", -1)) != i:
            _fail("V4", "rows.jsonl 第 %d 行的 row_index 是 %s，应为 %d" % (i, row.get("row_index"), i))

        row_cid, chunk_cid = str(row["chunk_id"]), str(chunk["chunk_id"])

        # V2：逐条对齐 —— 绝不退化为集合比较
        if row_cid != chunk_cid:
            _fail(
                "V2",
                "第 %d 行对齐错位：rows 里的 chunk_id=%s，chunks 里的是 %s\n"
                "（向量会配上错误的文本，检索引用会指向别的段落。请重跑 S5）"
                % (i, row_cid, chunk_cid),
            )

        # V3：chunk_id 唯一（Milvus 主键）
        if chunk_cid in seen:
            _fail("V3", "chunk_id 重复：%s" % chunk_cid)
        seen.add(chunk_cid)

        # V6：页码完整有序 —— 引用卡片的全部依据
        page_start, page_end = chunk.get("page_start"), chunk.get("page_end")
        for name, value in (("page_start", page_start), ("page_end", page_end)):
            if value is None or int(value) < 1:
                _fail("V6", "%s 的 %s 非法：%r（页码缺失的 chunk 数必须为 0）" % (chunk_cid, name, value))
        if int(page_end) < int(page_start):
            _fail("V6", "%s 的 page_end(%s) < page_start(%s)" % (chunk_cid, page_end, page_start))

        # V7 / V8：一份文档的 chunk 必须同属一份源文件、一条规则版本
        hashes.add(str(chunk.get("source_hash", "")))
        rule_versions.add(str(chunk.get("chunk_rule_version", "")))

        # V9：文本非空
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            _fail("V9", "%s 的 text 为空或全空白" % chunk_cid)

        # V10：按 **字节** 判长，并指名是哪条 chunk 的哪个字段
        for field in ("text", "file_name", "section", "block_type"):
            value = chunk.get(field)
            if value is None:
                continue
            limit = MAX_LENGTHS[field]
            size = len(str(value).encode("utf-8"))
            if size > limit:
                _fail(
                    "V10",
                    "%s 的 %s 占 %d 字节，超过 max_length=%d\n"
                    "（Milvus 的 VARCHAR 以 UTF-8 字节计。若语料确实更长，"
                    "需 release → alter_collection_field → load 抬高上限）"
                    % (chunk_cid, field, size, limit),
                )

    for cid_field, limit in (("chunk_id", MAX_LENGTHS["chunk_id"]), ("doc_id", MAX_LENGTHS["doc_id"])):
        size = len(str(chunks[0][cid_field]).encode("utf-8"))
        if size > limit:
            _fail("V10", "%s 占 %d 字节，超过 max_length=%d" % (cid_field, size, limit))

    # ---- V12：chunk_meta（JSON 字段）----
    # 校验放在写库之前，是为了让"超限"变成一句指名的报错，而不是 Milvus 插入时
    # 一个不知道哪条出错的异常。
    largest: tuple[int, str] = (0, "")
    for chunk in chunks:
        meta = build_chunk_meta(chunk)
        missing = [key for key in META_REQUIRED_KEYS if key not in meta]
        if missing:
            _fail(
                "V12",
                "%s 的 chunk_meta 缺少必需键 %s\n（S4 产物结构变了，S6 的存储契约要跟着更新）"
                % (chunk["chunk_id"], missing),
            )
        for key in meta:
            if not _META_KEY_RE.match(key):
                _fail(
                    "V12",
                    "%s 的 chunk_meta 键名非法：%r\n（Milvus 只允许字母、数字、下划线）"
                    % (chunk["chunk_id"], key),
                )
        size = len(json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if size > largest[0]:
            largest = (size, str(chunk["chunk_id"]))

    if largest[0] > META_MAX_BYTES:
        _fail(
            "V12",
            "最大的一条 chunk_meta 是 %s，占 %d 字节，超过 Milvus 单个 JSON 字段上限 %d 字节\n"
            "（`chunk_meta` 收的是「S4 记录减去已有列」，所以这通常意味着某个 chunk 的 "
            "text_for_embedding 异常大）" % (largest[1], largest[0], META_MAX_BYTES),
        )

    if len(hashes) != 1:
        _fail("V7", "一份文档的 chunk 出现 %d 个不同的 source_hash：%s" % (len(hashes), sorted(hashes)))
    if len(rule_versions) != 1:
        _fail(
            "V8",
            "一份文档的 chunk 出现 %d 个不同的 chunk_rule_version：%s\n"
            "（产物是拼凑的，无法确定它由哪套参数产出）" % (len(rule_versions), sorted(rule_versions)),
        )
    if str(chunks[0]["doc_id"]) != doc_id:
        _fail("输入", "产物里的 doc_id(%s) 与请求的 doc_id(%s) 不一致" % (chunks[0]["doc_id"], doc_id))


__all__ = ["DocInputs", "build_chunk_meta", "load_doc", "resolve_docs", "validate"]
