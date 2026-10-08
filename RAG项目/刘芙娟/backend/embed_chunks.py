#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S5 向量化：把分块产物编码成向量矩阵。

============================ 这个脚本做什么 ============================

输入：  data/chunks/{doc_id}.chunks.jsonl
输出：  data/embeddings/{doc_id}.npy             float32[N, 1024]，逐行 L2 归一化
        data/embeddings/{doc_id}.rows.jsonl      第 i 行描述 .npy 的第 i 行
        data/embeddings/{doc_id}.fingerprint.json 模型指纹，供 S6 与后端启动校验

编码输入是 chunk 的 **text_for_embedding**（章节路标 + 正文），不是 text。
text 供人核对、text_for_embedding 供检索，二者不可混（specs/003 的 output.py）。

实现拆在 backend/embed/ 下（model 编码口径 / store 读写），本文件是唯一入口。
**编码口径只有 model.py 一处定义**，S5 与将来的后端查询共用它。

============================ 运行方法 ============================

  D:/zg6_Project/9/med_rag/rag/python.exe backend/embed_chunks.py
  D:/zg6_Project/9/med_rag/rag/python.exe backend/embed_chunks.py d6da41b5d356
  D:/zg6_Project/9/med_rag/rag/python.exe backend/embed_chunks.py --batch-size 32
  D:/zg6_Project/9/med_rag/rag/python.exe backend/embed_chunks.py --print-fingerprint

退出码：0 成功 / 2 输入问题 / 3 模型权重问题 / 4 有空文本 / 5 向量不合法

============================ 两条必须知道的约定 ============================

1. **截断长度 max_length = 4096**（specs/004 的 D1 裁决）。实测最长 chunk 是
   3260 token，此值之下零截断。若换语料后出现更长的 chunk，报告里会列出
   「超长清单」并给出丢失比例 —— **绝不静默截断**。

2. **模型指纹变了就必须重建库**。指纹包含权重哈希、文件大小，**以及
   max_length / 归一化方式 / dtype / 池化方式**。任何一项变了，向量的数值
   就会变，而检索只会悄悄变差、不会报错（docs/04 §8）。

依赖：标准库 + numpy + torch + transformers（均已安装）。不依赖 FlagEmbedding。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from embed import (  # noqa: E402
    BATCH_SIZE, DIM, EXIT_BAD_VECTOR, EXIT_EMPTY_TEXT, EXIT_OK, MAX_LENGTH,
    MODEL_DIR, WEIGHT_FILE, EmbedError,
)
from embed.model import Encoder, fingerprint  # noqa: E402
from embed.store import (  # noqa: E402
    build_rows, embed_texts, find_empty_texts, list_doc_ids, load_chunks,
    project_root, save_json, save_jsonl, save_matrix,
)

RESET = "\r"


def _columns(text: str) -> int:
    """终端显示宽度。汉字占 2 列，用 len() 算会少算一半，清除时留下残影。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def say(message: str) -> None:
    sys.stdout.write(RESET + message + "\n")
    sys.stdout.flush()


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return "%.0f 秒" % seconds
    return "%d 分 %02d 秒" % (int(seconds // 60), int(seconds % 60))


class Progress:
    """批进度条。限流到每 0.3 秒最多刷一次，避免终端被刷屏。"""

    def __init__(self, total: int) -> None:
        self.total = total
        self.started = self.last = time.perf_counter()
        self.written = 0                 # 上一次实际写了多少列，用于精确清除

    def __call__(self, done: int, total: int) -> None:
        now = time.perf_counter()
        if done < total and now - self.last < 0.3:
            return                       # 限流
        self.last = now
        elapsed = now - self.started
        ratio = done / total if total else 1.0
        eta = elapsed / ratio - elapsed if ratio else 0.0
        width = 24
        filled = int(width * ratio)
        line = ("      [%s] %3d%%  %d/%d  已用 %s  剩余约 %s"
                % ("=" * filled + "-" * (width - filled), ratio * 100, done, total,
                   fmt_time(elapsed), fmt_time(eta)))
        # 按**显示列数**补空格再覆盖：ETA 从"1 分 05 秒"缩到"0 秒"时行会变短，
        # 只按上次长度清除会在尾部留下残影（实测踩过）。
        sys.stdout.write(RESET + line + " " * max(0, self.written - _columns(line)))
        sys.stdout.flush()
        self.written = _columns(line)
        if done >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def render_report(doc_id, shape, norms, token_lens, oversize, elapsed, fp) -> str:
    ordered = sorted(token_lens)
    n = len(ordered) or 1
    lines = ["", "=" * 68, "向量化报告（S5）", "=" * 68, ""]
    lines.append("  文档                      %s" % doc_id)
    lines.append("  chunk 数 / 向量形状         %d  →  %s" % (n, tuple(shape)))
    lines.append("  dtype                     float32（Milvus FLOAT_VECTOR 要求）")
    lines.append("  模长区间                   [%.8f, %.8f]   容差 ±1e-5" % (min(norms), max(norms)))
    lines.append("  耗时                      %.1f 秒（CPU）" % elapsed)
    lines.append("")
    lines.append("  token 长度（text_for_embedding，含特殊符）：")
    lines.append("    中位 %d / 最大 %d / 最小 %d" % (ordered[n // 2], ordered[-1], ordered[0]))
    if oversize:
        lines.append("")
        lines.append("  ⚠ 超长清单（token 数 > max_length=%d，会被截断）：%d 条"
                     % (MAX_LENGTH, len(oversize)))
        for cid, tok, chars, lost in oversize:
            lines.append("    %s  %d token / %d 字  丢失 %.0f%%" % (cid, tok, chars, lost * 100))
    else:
        lines.append("  超长清单                   0 条（max_length=%d 覆盖全部）" % MAX_LENGTH)
    lines.append("")
    lines.append("  模型指纹（变了就必须重建库）：")
    for key in ("model_dir", "config_sha256", "weight_bytes", "max_length",
                "pooling", "normalization", "dtype", "dim", "rule_version"):
        value = fp[key]
        if key == "config_sha256":
            value = str(value)[:16] + "…"
        lines.append("    %-16s %s" % (key, value))
    lines.append("")
    return "\n".join(lines)


def process(doc_id: str, chunks_dir: str, out_dir: str, opts) -> int:
    chunks_path = os.path.join(chunks_dir, "%s.chunks.jsonl" % doc_id)
    chunks = load_chunks(chunks_path)
    texts = embed_texts(chunks)
    say("[1/4] 读取分块产物 … %d 个 chunk" % len(chunks))

    empty = find_empty_texts(chunks, texts)
    if empty:
        raise EmbedError(
            EXIT_EMPTY_TEXT,
            "有 %d 个 chunk 的文本为空：%s\n"
            "空文本会生成零向量，而零向量与任何文本的余弦相似度都是 0 —— "
            "会静默污染检索（docs/04 §8）。请先修分块产物。"
            % (len(empty), "、".join(empty[:5])))

    say("[2/4] 加载模型 … %.2f GB，请稍候" % (os.path.getsize(
        os.path.join(opts.model_dir, WEIGHT_FILE)) / 1e9))
    started = time.perf_counter()
    encoder = Encoder(opts.model_dir, opts.max_length, opts.batch_size)
    say("      模型就绪（%.1f 秒）" % (time.perf_counter() - started))

    token_lens = encoder.token_lengths(texts)
    ordered = sorted(token_lens)
    say("[3/4] 计算 token 长度 … 中位 %d / 最大 %d"
        % (ordered[len(ordered) // 2], ordered[-1]))

    say("[4/4] 编码 %d 条 …" % len(texts))
    matrix = encoder.encode_documents(texts, on_batch=Progress(len(texts)))
    elapsed = time.perf_counter() - started

    if tuple(matrix.shape) != (len(chunks), DIM):
        raise EmbedError(EXIT_BAD_VECTOR, "向量形状异常：%s，期望 (%d, %d)"
                         % (tuple(matrix.shape), len(chunks), DIM))

    rows, norms, bad = build_rows(matrix, chunks)
    if len(rows) != matrix.shape[0]:
        raise EmbedError(EXIT_BAD_VECTOR,
                         "行对齐失败：rows=%d，npy=%d。行号与 chunk_id 错位会让引用指向错误原文。"
                         % (len(rows), matrix.shape[0]))
    if bad:
        raise EmbedError(EXIT_BAD_VECTOR,
                         "有 %d 行未通过 L2 归一化校验（容差 1e-5）：\n  %s"
                         % (len(bad), "\n  ".join(bad[:5])))

    oversize = [(c["chunk_id"], tok, c.get("char_len", 0), 1 - opts.max_length / tok)
                for c, tok in zip(chunks, token_lens) if tok > opts.max_length]
    fp = encoder.fingerprint

    os.makedirs(out_dir, exist_ok=True)
    save_matrix(os.path.join(out_dir, "%s.npy" % doc_id), matrix)
    save_jsonl(os.path.join(out_dir, "%s.rows.jsonl" % doc_id), rows)
    save_json(os.path.join(out_dir, "%s.fingerprint.json" % doc_id), fp)
    # 报告不往终端刷（太长），写文件备查：--report 才会打印
    report_path = os.path.join(out_dir, "%s.report.txt" % doc_id)
    save_text(report_path, render_report(doc_id, matrix.shape, norms, token_lens,
                                         oversize, elapsed, fp))
    if opts.report:
        print(render_report(doc_id, matrix.shape, norms, token_lens, oversize, elapsed, fp))

    if oversize:
        say("      ⚠ 有 %d 条超过截断上限，详见 %s"
            % (len(oversize), os.path.basename(report_path)))
    say("已完成 %d 个向量 → %s" % (len(rows), out_dir))
    say("      %s float32，行对齐 %d/%d，模长 %.6f ± %.1e"
        % (tuple(matrix.shape), len(rows), matrix.shape[0],
           sum(norms) / len(norms), max(abs(n - 1.0) for n in norms)))
    return EXIT_OK


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="embed_chunks.py",
                                     description="S5 向量化：分块产物 -> 向量矩阵")
    parser.add_argument("doc_ids", nargs="*", help="文档标识；留空 = 全部")
    parser.add_argument("--chunks-dir", default=None, help="默认 data/chunks")
    parser.add_argument("--out-dir", default=None, help="默认 data/embeddings")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="权重目录，默认 %s" % MODEL_DIR)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH,
                        help="截断长度，默认 %d（实测最长 chunk 3260 token）" % MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="批大小，默认 %d" % BATCH_SIZE)
    parser.add_argument("--report", action="store_true",
                        help="额外把完整报告打印到终端（默认只写 report.txt）")
    parser.add_argument("--print-fingerprint", action="store_true",
                        help="只打印模型指纹后退出（不加载权重、不产出）")
    args = parser.parse_args(argv)

    if args.print_fingerprint:
        for key, value in fingerprint(args.model_dir, args.max_length).items():
            print("%-16s %s" % (key, value))
        return EXIT_OK

    root = project_root()
    chunks_dir = args.chunks_dir or os.path.join(root, "data", "chunks")
    out_dir = args.out_dir or os.path.join(root, "data", "embeddings")
    doc_ids = args.doc_ids or list_doc_ids(chunks_dir)

    for doc_id in doc_ids:
        code = process(doc_id, chunks_dir, out_dir, args)
        if code != EXIT_OK:
            return code
    return EXIT_OK


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EmbedError as exc:
        print("\n[失败] %s" % exc.message, file=sys.stderr)
        sys.exit(exc.code)
    except KeyboardInterrupt:
        sys.exit(130)
