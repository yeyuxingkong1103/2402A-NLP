# -*- coding: utf-8 -*-
"""工单6 多嵌入模型向量索引构建 —— 为 bge-m3 之外的模型（如 m3e-base）建 FAISS 索引。

工单编号：人工智能NLP-RAG-混合检索任务

用法
----
.. code-block:: powershell

    # 列出可建索引的模型（含本机安装/索引状态）
    pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index_multi.py --list

    # 为 m3e-base 建 768 维索引（**实测 45.5 秒**，2676 块；早先估算 4~5 分钟，以实测为准）
    pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index_multi.py --model m3e-base

    # 只校验已建索引的自洽性（不嵌入、不写盘）
    pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index_multi.py --model m3e-base --verify-only

设计纪律
--------
* **块集合与行序完全复用** ``bge-m3_1024/chunks_indexed.jsonl``
  （该文件行序 == ``ids.json`` 行序 == ``vectors.npy`` 行序，工单4 已实测一致）⇒
  新索引只是「同一批块的另一套向量」，**不重新解析 PDF、不重新分块、不改动既有索引**。
* BM25 与嵌入模型无关（jieba 分词 + 词频），因此**不重复落盘 bm25.pkl**；
  检索时由 ``retrieval_strategy.bundle_for_model`` 复用主 bundle 的 BM25 与块表。
* ``faiss`` 在非 ASCII 路径下 ``write_index`` 不可用（工单4 实测），
  统一走 ``faiss.serialize_index`` 字节序列化。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 允许直接以脚本方式运行（把 研发/ 加入 sys.path）
_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent.parent
if str(_REPO / "研发") not in sys.path:
    sys.path.insert(0, str(_REPO / "研发"))

import numpy as np  # noqa: E402

from app.core import embedding_models as em  # noqa: E402
from app.core.config import get_config  # noqa: E402
from app.core.vector_store import l2_normalize, try_build_faiss  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"
SOURCE_SLUG = "bge-m3_1024"          # 块集合与行序的权威来源


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_chunks(index_root: Path) -> tuple[list[str], list[str]]:
    """读源索引的块 id 与内容（行序即向量行序）。"""
    src = index_root / SOURCE_SLUG
    ids_path = src / "ids.json"
    chunks_path = src / "chunks_indexed.jsonl"
    if not ids_path.is_file() or not chunks_path.is_file():
        raise SystemExit(f"[FATAL] 源索引不完整：需要 {ids_path} 与 {chunks_path}")
    ids = list(json.loads(ids_path.read_text(encoding="utf-8")).get("ids") or [])
    texts: list[str] = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                texts.append(str(json.loads(line).get("content") or ""))
    if len(ids) != len(texts):
        raise SystemExit(f"[FATAL] 源索引自洽性失败：ids={len(ids)} chunks={len(texts)}")
    return ids, texts


def cmd_list(index_root: Path, args: argparse.Namespace) -> int:
    print(f"[{WORK_ORDER}] 嵌入模型清单（index_root={index_root}）")
    print(f"{'id':<14}{'ollama 模型':<40}{'维度':<6}{'已安装':<8}{'索引':<8}索引目录")
    for spec in em.list_models(index_root=index_root, probe=True):
        print(f"{spec.id:<14}{spec.ollama_model:<40}{spec.dim:<6}"
              f"{'是' if spec.available else '否':<8}"
              f"{'已建' if spec.index_built else '未建':<8}{spec.slug}")
    return 0


def cmd_build(index_root: Path, args: argparse.Namespace) -> int:
    spec = em.resolve_model(args.model, index_root=index_root, probe=True)
    if not spec.available:
        print(f"[FATAL] 模型未在 Ollama 中安装：{spec.ollama_model}")
        return 2
    target = index_root / spec.slug
    ids, texts = _read_chunks(index_root)
    print(f"[{WORK_ORDER}] 目标索引 = {target}")
    print(f"  模型 = {spec.ollama_model}（注册维度 {spec.dim}）")
    print(f"  块数 = {len(texts)}（行序复用 {SOURCE_SLUG}）")

    if args.verify_only:
        return _verify(target, spec, ids, texts)

    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    vectors = np.zeros((len(texts), spec.dim), dtype=np.float32)
    batch = max(1, int(args.batch_size))
    done = 0
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        result = em.embed_texts_with_model(chunk, spec, base_url=args.base_url,
                                           batch_size=batch, timeout=float(args.timeout))
        vectors[start:start + len(chunk)] = result.vectors
        done += len(chunk)
        if done % (batch * 8) == 0 or done == len(texts):
            rate = done / max(time.perf_counter() - started, 1e-6)
            left = (len(texts) - done) / max(rate, 1e-6)
            print(f"  进度 {done}/{len(texts)}  速度 {rate:.1f} 块/s  预计剩余 {left/60:.1f} min",
                  flush=True)
    vectors = l2_normalize(vectors)
    np.save(target / "vectors.npy", vectors)
    (target / "ids.json").write_text(
        json.dumps({"ids": ids, "dim": spec.dim, "model": spec.ollama_model,
                    "count": len(ids), "normalized": True, "dtype": "float32",
                    "updated_at": _now_iso(),
                    "row_order_source": SOURCE_SLUG,
                    "faiss_note": "faiss 1.7.4 在非 ASCII 路径 write_index 不可用；"
                                  "已改用 faiss.serialize_index 字节序列化"},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    faiss_index = try_build_faiss(vectors, enabled=True)
    faiss_note = None
    if faiss_index is not None:
        import faiss  # noqa: PLC0415

        (target / "faiss.index").write_bytes(faiss.serialize_index(faiss_index))
    else:
        faiss_note = "faiss 不可用，检索回退 numpy 精确内积（结果等价）"
    elapsed = round(time.perf_counter() - started, 2)
    (target / "index_manifest.json").write_text(
        json.dumps({
            "work_order": WORK_ORDER, "built_at": _now_iso(), "model": spec.ollama_model,
            "model_id": spec.id, "dim": spec.dim, "count": len(ids),
            "vectors_sha1": hashlib.sha1(vectors.tobytes()).hexdigest(),
            "ids_sha1": hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest(),
            "row_order_source": SOURCE_SLUG, "faiss": faiss_index is not None,
            "faiss_note": faiss_note, "elapsed_s": elapsed,
            "note": "本索引只含向量与 id；块表 / BM25 / 页索引复用主 bundle",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] 已写出 {target}（{elapsed:.1f}s，vectors {vectors.shape}，"
          f"faiss={'是' if faiss_index is not None else '否(回退numpy)'}）")
    return _verify(target, spec, ids, texts)


def _verify(target: Path, spec: em.EmbeddingModelSpec, ids: list[str], texts: list[str]) -> int:
    """只读校验：文件齐全、维度一致、行数一致、向量已归一化。"""
    print(f"[校验] {target}")
    problems: list[str] = []
    vectors_path = target / "vectors.npy"
    ids_path = target / "ids.json"
    if not vectors_path.is_file():
        problems.append("缺少 vectors.npy")
    if not ids_path.is_file():
        problems.append("缺少 ids.json")
    if problems:
        for item in problems:
            print(f"  [FAIL] {item}")
        return 1
    vectors = np.load(vectors_path)
    meta = json.loads(ids_path.read_text(encoding="utf-8"))
    saved_ids = list(meta.get("ids") or [])
    checks = [
        (f"行数一致：vectors={vectors.shape[0]} ids={len(saved_ids)} 源={len(ids)}",
         vectors.shape[0] == len(saved_ids) == len(ids)),
        (f"维度一致：vectors={vectors.shape[1]} 注册={spec.dim}",
         int(vectors.shape[1]) == int(spec.dim)),
        (f"id 内容与源索引一致（前 8 位 sha1={hashlib.sha1(chr(10).join(saved_ids).encode()).hexdigest()[:8]}）",
         saved_ids == ids),
        (f"向量已 L2 归一化（首行范数={float(np.linalg.norm(vectors[0])):.6f}）",
         abs(float(np.linalg.norm(vectors[0])) - 1.0) < 1e-3),
        ("faiss.index 存在", (target / "faiss.index").is_file()),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'OK' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print(f"[{'PASS' if ok else 'FAIL'}] {target.name}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"[{WORK_ORDER}] 多嵌入模型向量索引构建（bge / m3e 可切换）")
    parser.add_argument("--model", default="m3e-base", help="模型 id 或 Ollama 标签")
    parser.add_argument("--list", action="store_true", help="只列出模型与索引状态")
    parser.add_argument("--verify-only", action="store_true", help="只校验，不嵌入不写盘")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--index-root", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = get_config()
    index_root = Path(args.index_root) if args.index_root else Path(cfg.paths.index_dir)
    if args.list:
        return cmd_list(index_root, args)
    return cmd_build(index_root, args)


if __name__ == "__main__":
    raise SystemExit(main())
