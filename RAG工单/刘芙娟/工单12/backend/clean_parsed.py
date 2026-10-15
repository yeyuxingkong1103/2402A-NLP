#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S3 清洗：MinerU 解析产物 -> 可下游消费的正文块。

============================ 这个脚本做什么 ============================

输入：  data/parsed/{doc_id}/<文档名>/auto/<文档名>_content_list.json
        （MinerU 的扁平结构化产物，每项自带 page_idx）

输出：  data/clean/{doc_id}.blocks.jsonl    清洗后的正文块，下游分块的输入
        data/clean/{doc_id}.dropped.jsonl   被剔除的每一块 + 命中的具体规则
        data/clean/{doc_id}.report.txt      按四类分组的处置报告

清洗按四类组织（用户指定的分类）：
  剔除类  版式噪声：页眉/页脚/印刷页码/侧边文字/参考文献条目/空块
  保护类  绝不可误杀：引文标注、紧急转诊表述、剂量数值、表格
  转换类  无损转换：反斜杠反转义、HTML 实体解码、表格转 Markdown、
          词内空格合并(D1)、page_idx -> 1-based 页码
  修复类  只合并跨页断句；标题粘连与标题层级失真只报告、不修复

============================ 运行方法 ============================

    D:/zg6_Project/9/med_rag/rag/python.exe backend/clean_parsed.py

不带参数 = 清洗 data/parsed/ 下的全部文档。常用写法：

    # 只洗一份
    D:/zg6_Project/9/med_rag/rag/python.exe backend/clean_parsed.py d6da41b5d356

    # 只出报告，不写产物
    D:/zg6_Project/9/med_rag/rag/python.exe backend/clean_parsed.py --dry-run

    # 自检（D1 空格合并的 5 组基线）
    D:/zg6_Project/9/med_rag/rag/python.exe backend/clean_parsed.py --self-test

退出码：0 成功 / 2 输入问题 / 3 未知块类型 / 4 保护类被误杀 / 5 清洗后为空

依赖：Python 标准库 + jieba（已装）。jieba 缺失时自动降级为不合并空格并告警，
      不会静默跳过。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:  # Windows 控制台默认 GBK，中文报告会乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from clean import space_merge  # noqa: E402
from clean.errors import CleanError, EXIT_BAD_INPUT, EXIT_OK  # noqa: E402
from clean.paths import (  # noqa: E402
    default_input_root,
    default_output_dir,
    discover_doc_ids,
    find_content_list,
    project_root,
)
from clean.pipeline import process_one  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clean_parsed.py",
        description="S3 清洗：把 MinerU 的 content_list.json 变成正文块",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="退出码：0 成功 / 2 输入问题 / 3 未知块类型 / 4 保护类被误杀 / 5 清洗后为空",
    )
    parser.add_argument(
        "doc_ids", nargs="*",
        help="文档标识（data/parsed/ 下的目录名）。留空 = 全部文档",
    )
    parser.add_argument("--input-root", default=None, help="解析产物根目录，默认 data/parsed")
    parser.add_argument("--output-dir", default=None, help="清洗产物目录，默认 data/clean")
    parser.add_argument(
        "--content-list", default=None,
        help="直接指定某一份 content_list.json（与单个 doc_id 搭配使用）",
    )

    group = parser.add_argument_group("四类规则开关")
    group.add_argument("--no-convert", action="store_true",
                       help="关闭转换类（反斜杠/实体/表格 Markdown）")
    group.add_argument("--no-space-merge", action="store_true",
                       help="关闭 D1 的 CJK 词内空格合并")
    group.add_argument("--no-repair", action="store_true",
                       help="关闭修复类（跨页断句合并）")
    group.add_argument("--drop-page-footnote", action="store_true",
                       help="连页脚注释一起剔除（默认保留：常含剂量注解）")

    parser.add_argument("--dry-run", action="store_true", help="只出报告，不写产物")
    parser.add_argument("--self-test", action="store_true", help="跑 D1 的 5 组基线自检后退出")
    parser.add_argument("--quiet", action="store_true", help="不打印报告正文")
    return parser


def make_opts(args) -> argparse.Namespace:
    return argparse.Namespace(
        convert=not args.no_convert,
        space_merge=not args.no_space_merge,
        repair=not args.no_repair,
        drop_page_footnote=args.drop_page_footnote,
        dry_run=args.dry_run,
    )


def run_self_test() -> int:
    failures = space_merge.self_test()
    if failures:
        print("D1 自检未通过：")
        for line in failures:
            print("  ✗ " + line)
        return EXIT_BAD_INPUT
    print("D1 自检通过：%d 组基线全部符合预期" % len(space_merge.BASELINE))
    return EXIT_OK


def resolve_targets(args) -> list[tuple[str, str]]:
    """返回 [(doc_id, content_list_path), ...]"""
    root = project_root()
    input_root = args.input_root or default_input_root(root)

    if args.content_list:
        if not args.doc_ids:
            raise CleanError(
                EXIT_BAD_INPUT, "--content-list 需与单个 doc_id 搭配使用（用于确定 doc_id）"
            )
        if len(args.doc_ids) > 1:
            raise CleanError(EXIT_BAD_INPUT, "--content-list 只能搭配一个 doc_id")
        return [(args.doc_ids[0], os.path.abspath(args.content_list))]

    doc_ids = args.doc_ids
    if not doc_ids:
        doc_ids = discover_doc_ids(input_root)
        if not doc_ids:
            raise CleanError(
                EXIT_BAD_INPUT,
                "%s 下没有任何文档目录，无待清洗内容。\n"
                "（区别于静默成功：请先运行 S2 解析）" % input_root,
            )

    targets = []
    for doc_id in doc_ids:
        doc_dir = os.path.join(input_root, doc_id)
        if not os.path.isdir(doc_dir):
            raise CleanError(
                EXIT_BAD_INPUT, "doc_id 不存在：%s（期望目录 %s）" % (doc_id, doc_dir)
            )
        targets.append((doc_id, find_content_list(doc_dir)))
    return targets


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        return run_self_test()

    root = project_root()
    output_dir = args.output_dir or default_output_dir(root)
    opts = make_opts(args)

    targets = resolve_targets(args)
    print("清洗 %d 份文档" % len(targets))
    print("输出目录：%s" % output_dir)
    print("规则开关：convert=%s space_merge=%s repair=%s drop_page_footnote=%s"
          % (opts.convert, opts.space_merge, opts.repair, opts.drop_page_footnote))

    for doc_id, content_list_path in targets:
        report = process_one(content_list_path, doc_id, output_dir, opts)
        if not args.quiet:
            print(report.render())

    print("完成：%d 份文档" % len(targets))
    return EXIT_OK


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CleanError as exc:
        print("\n[失败] %s" % exc.message, file=sys.stderr)
        sys.exit(exc.code)
    except KeyboardInterrupt:
        print("\n[中断]", file=sys.stderr)
        sys.exit(130)
