"""S4 分块：把清洗产物切成可检索、可引用的语义块。

============================ 这个脚本做什么 ============================

输入：  data/clean/{doc_id}.blocks.jsonl
输出：  data/chunks/{doc_id}.chunks.jsonl     分块产物，S5 向量化的输入
        data/chunks/{doc_id}.report.txt       自检指标
        data/chunks/{doc_id}.decisions.jsonl  待裁决项（仅在需要裁决时产出）

分块依据（specs/003-semantic-chunking 的 D1/D3）：
  Q2=A 规则式分块：切分边界 = 段落对齐 + 章节边界，**不加载任何模型**
  D1   目标 300-700 字；仅当两个相邻 section 同属一个父章节时才允许跨节合并
  D3   遇到判不准的边界就停下来交人裁决，绝不自己拍板

实现拆在 backend/chunk/ 下（core/loader/decide/output），本文件是唯一入口。

============================ 运行方法 ============================

  D:/zg6_Project/9/med_rag/rag/python.exe backend/chunk_clean.py
  D:/zg6_Project/9/med_rag/rag/python.exe backend/chunk_clean.py d6da41b5d356
  D:/zg6_Project/9/med_rag/rag/python.exe backend/chunk_clean.py --no-cross-section

退出码：0 成功 / 2 输入问题 / 3 有待裁决项 / 4 页码缺失

============================ 待裁决项 ============================

脚本碰到判不准的边界时**不会自己决定**：把问题写进 decisions.jsonl，以退出码 3
结束，并且**不产出 chunks.jsonl**——避免交出一份边界其实由脚本瞎猜的产物。

答复写进 data/chunks/{doc_id}.answers.json。支持四档，**优先级从高到低**：

  {
    "d6da41b5d356:D007": "merge_up",        // 1. 精确到单条
    "parent:6 高血压与中医药 > 6.3 中医特色适宜技术": "merge_up",
                                            // 2. 按父章节批量——同父之下都是兄弟小节，
                                            //    section 元数据损失最小，推荐用这档
    "kind:section_too_small": "merge_up",   // 3. 按类别批量（最宽，可能横跨大章，慎用）
    "*": "keep"                             // 4. 兜底：其余全部 keep
  }

每条的 kind / parent / 可选值见 decisions.jsonl。回填后再重跑同一条命令即可。

依赖：**纯 Python 标准库**，不加载任何模型。BGE-M3 只在 S5 向量化时才需要
（本机已有权重：E:/资料/BAAI--bge-m3）。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from chunk.core import (  # noqa: E402
    EXIT_BAD_INPUT, EXIT_NO_PAGE, EXIT_OK, EXIT_PENDING,
    HIGH, ChunkError, project_root,
)
from chunk.decide import (  # noqa: E402
    Decisions, apply_oversized, audit_unfinished, merge, note_small_sections,
)
from chunk.loader import build_units, load_blocks, resolve_source_hash  # noqa: E402
from chunk.output import build_chunk, load_answers, render_report, write_jsonl, write_text  # noqa: E402


def process(doc_id: str, clean_path: str, out_dir: str, opts) -> int:
    blocks = load_blocks(clean_path)
    decisions = Decisions(doc_id, load_answers(out_dir, doc_id))

    units = build_units(blocks)
    if not units:
        raise ChunkError(EXIT_BAD_INPUT, "%s 没有任何正文块" % clean_path)

    note_small_sections(units, decisions)   # 必须在合并之前，答复才查得到
    chunks, crossed = merge(units, opts, decisions)
    final: list[list[dict]] = []
    apply_oversized(chunks, final, decisions, opts)
    audit_unfinished(final, decisions)

    source_hash = resolve_source_hash(doc_id)
    records = [build_chunk(doc_id, blocks[0].get("file_name", ""), source_hash, i, seg)
               for i, seg in enumerate(final)]
    report = render_report(records, decisions, crossed, source_hash)

    os.makedirs(out_dir, exist_ok=True)
    if decisions.items:
        write_jsonl(os.path.join(out_dir, "%s.decisions.jsonl" % doc_id), decisions.items)
    write_text(os.path.join(out_dir, "%s.report.txt" % doc_id), report)
    print(report)

    pending = decisions.pending()
    if pending:
        print("=" * 68)
        print("有 %d 项待你裁决，**未产出 chunks.jsonl**（脚本不私自决定分块边界）" % len(pending))
        for item in pending[:12]:
            print("\n  [%s] %s" % (item["decision_id"], item["why"]))
            print("       位置：p%s  %s" % (item["page"], " ".join(item["block_ids"])))
            print("       内容：%s" % item["excerpt"])
            print("       可选：%s" % " / ".join(item["choices"]))
        if len(pending) > 12:
            print("\n  … 另有 %d 项，全部见 decisions.jsonl" % (len(pending) - 12))
        print('\n把答复写进 %s 后重跑本命令（格式：{"<decision_id>": "<选项>"}）。'
              % os.path.join(out_dir, "%s.answers.json" % doc_id))
        return EXIT_PENDING

    nopage = [r for r in records if not r.get("page_start")]
    if nopage:
        print("[失败] %d 个 chunk 缺页码，引用链路已断" % len(nopage), file=sys.stderr)
        return EXIT_NO_PAGE
    write_jsonl(os.path.join(out_dir, "%s.chunks.jsonl" % doc_id), records)
    print("已写出 %d 个 chunk → %s" % (len(records), out_dir))
    return EXIT_OK


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="chunk_clean.py",
                                     description="S4 分块：清洗产物 -> 语义块")
    parser.add_argument("doc_ids", nargs="*", help="文档标识；留空 = 全部")
    parser.add_argument("--clean-dir", default=None, help="默认 data/clean")
    parser.add_argument("--out-dir", default=None, help="默认 data/chunks")
    parser.add_argument("--no-cross-section", action="store_true",
                        help="关闭 D1 的有限跨节合并")
    args = parser.parse_args(argv)
    opts = argparse.Namespace(high=HIGH, cross_section=not args.no_cross_section)

    root = project_root()
    clean_dir = args.clean_dir or os.path.join(root, "data", "clean")
    out_dir = args.out_dir or os.path.join(root, "data", "chunks")
    if args.doc_ids:
        doc_ids = args.doc_ids
    elif os.path.isdir(clean_dir):
        doc_ids = sorted(f[: -len(".blocks.jsonl")] for f in os.listdir(clean_dir)
                         if f.endswith(".blocks.jsonl"))
    else:
        doc_ids = []
    if not doc_ids:
        raise ChunkError(EXIT_BAD_INPUT, "%s 下没有 *.blocks.jsonl，请先运行 S3 清洗" % clean_dir)

    for doc_id in doc_ids:
        code = process(doc_id, os.path.join(clean_dir, "%s.blocks.jsonl" % doc_id), out_dir, opts)
        if code != EXIT_OK:
            return code
    return EXIT_OK


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ChunkError as exc:
        print("\n[失败] %s" % exc.message, file=sys.stderr)
        sys.exit(exc.code)
    except KeyboardInterrupt:
        sys.exit(130)
