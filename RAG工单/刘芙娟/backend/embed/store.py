"""读写 S5 的输入与产物。

产物（docs/04 §4 / §8 契约，都在 data/embeddings/ 下）：
    {doc_id}.npy            float32[N, 1024]，逐行 L2 归一化
    {doc_id}.rows.jsonl     第 i 行描述 .npy 的第 i 行
    {doc_id}.fingerprint.json  模型指纹，供 S6 与后端启动校验
"""

from __future__ import annotations

import json
import os

from . import EXIT_BAD_INPUT, NORM_TOLERANCE, EmbedError


def project_root() -> str:
    """backend/embed/store.py -> backend/embed -> backend -> 仓库根。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def list_doc_ids(chunks_dir: str) -> list[str]:
    if not os.path.isdir(chunks_dir):
        raise EmbedError(EXIT_BAD_INPUT, "%s 不存在，请先运行 S4 分块" % chunks_dir)
    suffix = ".chunks.jsonl"
    found = sorted(f[: -len(suffix)] for f in os.listdir(chunks_dir)
                   if f.endswith(suffix))
    if not found:
        raise EmbedError(EXIT_BAD_INPUT, "%s 下没有 *.chunks.jsonl，请先运行 S4 分块" % chunks_dir)
    return found


def load_chunks(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise EmbedError(EXIT_BAD_INPUT, "分块产物不存在：%s\n（请先运行 S4：chunk_clean.py）" % path)
    try:
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbedError(EXIT_BAD_INPUT, "读取失败：%s\n  %s" % (path, exc)) from exc
    if not rows:
        raise EmbedError(EXIT_BAD_INPUT, "分块产物为空：%s" % path)
    return rows


def embed_texts(rows: list[dict]) -> list[str]:
    """取编码输入。**必须是 text_for_embedding**，不得退化为 text（FR-008）。"""
    texts = []
    for i, row in enumerate(rows):
        if "text_for_embedding" not in row:
            raise EmbedError(
                EXIT_BAD_INPUT,
                "第 %d 行缺 text_for_embedding 字段（旧版本分块产物？）\n"
                "不得退化为用 text —— 两者口径不同，混用会让检索静默变差。" % i)
        texts.append(row["text_for_embedding"])
    return texts


def _atomic_write(path: str, writer) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    try:
        writer(tmp)
        os.replace(tmp, path)
    except OSError as exc:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise EmbedError(1, "写出失败：%s\n  %s" % (path, exc)) from exc


def save_matrix(path: str, matrix) -> None:
    import numpy as np

    def write(tmp: str) -> None:
        # 必须传**文件对象**：np.save 给不以 .npy 结尾的**路径**会自动补 .npy，
        # 于是写到了 "xxx.npy.tmp.npy"，随后的 os.replace 就找不到源文件。
        # （实测踩过：WinError 2，几分钟的编码白算。）
        with open(tmp, "wb") as fh:
            np.save(fh, matrix, allow_pickle=False)

    _atomic_write(path, write)


def save_jsonl(path: str, records: list[dict]) -> None:
    def write(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _atomic_write(path, write)


def save_json(path: str, payload: dict) -> None:
    def write(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    _atomic_write(path, write)


def save_text(path: str, content: str) -> None:
    def write(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
    _atomic_write(path, write)


def build_rows(matrix, chunks: list[dict]) -> tuple[list[dict], list[float], list[str]]:
    """行对齐表。vector_norm 是**实测模长**，不是硬写的 1.0（FR-015）。

    错位是这里最危险的失败模式：chunk_id 与向量行号一旦对不上，引用会指向
    错误的原文，而且**不会报错**。所以这里逐行断言。
    """
    import numpy as np
    norms = np.linalg.norm(matrix, axis=1)
    rows, bad = [], []
    for i, (chunk, norm) in enumerate(zip(chunks, norms)):
        value = float(norm)
        rows.append({
            "row_index": i,
            "chunk_id": chunk["chunk_id"],
            "char_len": chunk.get("char_len", 0),
            "vector_norm": value,
        })
        if abs(value - 1.0) > NORM_TOLERANCE:
            bad.append("%s: ‖v‖₂ = %.8f" % (chunk["chunk_id"], value))
    return rows, [float(n) for n in norms], bad


def find_empty_texts(rows: list[dict], texts: list[str]) -> list[str]:
    """空文本会生成零向量，而零向量与任何文本的余弦都是 0——静默污染检索。"""
    return [rows[i]["chunk_id"] for i, t in enumerate(texts) if not t.strip()]
