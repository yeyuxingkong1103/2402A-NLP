# -*- coding: utf-8 -*-
"""爬虫唯一入口：抓取 / 汇总，全部命令走这里。

用法（在项目根目录执行）：
    python -m src.crawler.pipeline run --source nmpa        --list data/raw/drugs.txt
    python -m src.crawler.pipeline run --source nmpa        --names 布洛芬 阿莫西林 --merge
    python -m src.crawler.pipeline run --source chictr      --terms-file terms.txt --max 30
    python -m src.crawler.pipeline run --source wanfang     --terms-file terms.txt --max 30
    python -m src.crawler.pipeline run --source guideline   --urls-file guidelines.txt
    python -m src.crawler.pipeline run --source otc_ann [--pages 3] [--max 20]   # 自动爬公告列表+下附件
    python -m src.crawler.pipeline all [--merge]          # 一把梭：吃 drugs_standard.json 药名跑 nmpa+otc_ann
    python -m src.crawler.pipeline report

本文件只做三件事：解析命令 -> 调度数据源 -> 报告结果。
抓取细节全在 spiders.py，路径全在 schema.PATHS，这里不出现任何硬编码路径。
调度实现已按职责拆分（本文件保留入口 + 报告）：
  - ``crawl_sources.py``：各数据源 run_* 与 RUNNERS
  - ``crawl_storage.py``：落盘 / 去重 / 断点
"""
import argparse
import json
import os

from .base import get_logger, setup_logging
from .crawl_sources import (RUNNERS, extract_seed_names, merge_nmpa, run_all,
                            run_chictr, run_guideline, run_nmpa, run_otc_ann,
                            run_wanfang)
from .crawl_storage import (_count_docs, otc_done, record_nmpa_failure,
                            sink_doc)
from .schema import PATHS, load_lines
from .spiders import load_name_list


# ======================= 站点注册表 =======================
# 新增数据源 = 在 spiders.py 写抓取逻辑，然后这里加一行。
# mode 决定 CLI 要求哪个输入参数：names（名单）/ terms（检索词）/ urls（直链）
SOURCES = {
    "nmpa":        {"mode": "names", "desc": "NMPA 药品批准文号（含医保/基药/OTC）"},
    "chictr":      {"mode": "terms", "desc": "中国临床试验注册中心"},
    "wanfang":     {"mode": "terms", "desc": "万方中文文献"},
    "guideline":   {"mode": "urls",  "desc": "指南 PDF 直链"},
    "otc_ann":     {"mode": "crawl", "desc": "NMPA 处方药转换为非处方药公告 + 附件原件(PDF/docx)"},
}


# report 的验收目标（数量达标打勾，可按需改）
TARGETS = {"guideline": 5, "chictr": 30, "wanfang": 30, "otc_ann": 10}


# ======================= 汇总报告 =======================
def report():
    """按注册表统计真实目录，不再手写目录清单（修掉了统计失真的问题）"""
    print("=" * 56)
    print("爬虫产出汇总")
    print("=" * 56)

    counts = {
        "guideline":   _count_docs(PATHS["guideline"]),
        "chictr":      _count_docs(PATHS["chictr"]),
        "wanfang":     _count_docs(PATHS["wanfang"]),
        "otc_ann":     len(otc_done(PATHS["otc_ann"])),
    }
    # nmpa 特殊：不按 doc 落盘，统计进度行数 + 合并后的记录数
    nmpa_done = len(load_lines(PATHS["nmpa_progress"]))
    nmpa_items = 0
    if os.path.exists(PATHS["nmpa_out"]):
        try:
            with open(PATHS["nmpa_out"], encoding="utf-8") as f:
                nmpa_items = len(json.load(f))
        except Exception:
            nmpa_items = 0

    print("\n【REST 系：按文档计】")
    print("%-14s %8s %8s %6s" % ("数据源", "已有", "目标", "状态"))
    print("-" * 56)
    for src in ("guideline", "chictr", "wanfang", "otc_ann"):
        n, target = counts[src], TARGETS.get(src, 0)
        flag = "OK" if n >= target else "不足"
        print("%-14s %8d %8d %6s" % (src, n, target, flag))

    print("\n【NMPA 药品：按药名 / 记录计】")
    print("-" * 56)
    print("%-14s %8d %8s %6s" % ("药名进度", nmpa_done, "-", "-"))
    print("%-14s %8d %8s %6s" % ("合并记录", nmpa_items, "-", "-"))

    print("\n【失败记录】")
    print("-" * 56)
    total_fail = 0
    for src in ("guideline", "chictr", "wanfang"):
        p = os.path.join(PATHS[src], "_failed.jsonl")
        n = len(load_lines(p))
        total_fail += n
        print("%-14s %8d" % (src, n))
    for name, p in (("nmpa", PATHS["nmpa_failed"]),):
        n = len(load_lines(p))
        total_fail += n
        print("%-14s %8d" % (name, n))
    if total_fail == 0:
        print("无失败记录")

    print("\n【产出目录】")
    print("-" * 56)
    for src, d in sorted(PATHS.items()):
        mark = "存在" if os.path.exists(d) else "无"
        print("%-14s %-34s %s" % (src, d, mark))
    print()


# ======================= 命令行 =======================
def _chdir_project_root():
    """路径都是相对项目根的 data/raw，这里上溯定位根目录，避免 cwd 影响"""
    d = os.path.abspath(os.getcwd())
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "src")) and os.path.isdir(os.path.join(d, "data")):
            os.chdir(d)
            return
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent


def _read_items(args, mode):
    """按 mode 取输入：crawl 模式无需输入；其余见各分支。"""
    if mode == "crawl":
        return []          # otc_ann 自己爬列表，不需要外部输入文件
    if mode == "names":
        if args.list_file:
            return load_name_list(args.list_file)
        return args.names or []
    path = args.terms_file if mode == "terms" else args.urls_file
    if not path:
        raise SystemExit("错误：%s 需要 --%s" % (
            args.source, "terms-file" if mode == "terms" else "urls-file"))
    return load_name_list(path)


def main():
    _chdir_project_root()
    # 日志要落在项目根的 data/raw 下，所以必须先切到根目录再初始化
    setup_logging(PATHS["crawler_log"])
    ap = argparse.ArgumentParser(
        prog="python -m src.crawler.pipeline",
        description="医疗 RAG 爬虫（数据源：%s）" % ", ".join(sorted(SOURCES)))
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="抓取数据")
    run.add_argument("--source", required=True, choices=sorted(SOURCES),
                     help="; ".join("%s=%s" % (k, v["desc"]) for k, v in sorted(SOURCES.items())))
    run.add_argument("--names", nargs="+", help="药名/名称，空格分隔")
    run.add_argument("--list", dest="list_file", help="名单文件，每行一个（names 模式）")
    run.add_argument("--terms-file", help="检索词文件，每行一个（terms 模式）")
    run.add_argument("--urls-file", help="直链文件：doc_id<TAB>url（urls 模式）")
    run.add_argument("--max", type=int, help="每个检索词/药名最多抓多少条；otc_ann 为最多抓多少篇公告")
    run.add_argument("--interval", type=float, default=3, help="请求间隔秒，默认 3")
    run.add_argument("--pages", type=int, default=3, help="otc_ann：最多翻多少页列表，默认 3")
    run.add_argument("--merge", action="store_true", help="nmpa：抓完把进度合并为 drugs.json")
    run.add_argument("--yibao", help="医保目录文件（xlsx/xls/pdf）；不给则用 PATHS 默认位置，"
                                     "默认位置也没有就自动从医保局官网下载")

    # 一把梭：基于本地种子药名一次性跑完所有「可自动」源
    allc = sub.add_parser("all", help="一把梭：吃 drugs_standard.json 药名跑 nmpa+otc_ann")
    allc.add_argument("--max", type=int, help="每药最多抓多少条；otc_ann 为最多抓多少篇公告")
    allc.add_argument("--interval", type=float, default=3, help="请求间隔秒，默认 3")
    allc.add_argument("--pages", type=int, default=3, help="otc_ann：最多翻多少页列表，默认 3")
    allc.add_argument("--merge", action="store_true", help="nmpa：跑完把进度合并为 drugs.json")
    allc.add_argument("--yibao", help="医保目录文件（xlsx/xls/pdf）；不给则用 PATHS 默认位置")

    sub.add_parser("report", help="产出汇总")

    args = ap.parse_args()

    if args.cmd == "report":
        report()
        return

    if args.cmd == "all":
        run_all(args)
        report()
        return

    mode = SOURCES[args.source]["mode"]
    args.items = _read_items(args, mode)
    if mode != "crawl" and not args.items:
        raise SystemExit("错误：没有可抓的输入项（检查 --names / --list / --terms-file / --urls-file）")
    if mode == "crawl":
        get_logger().info("数据源=%s（自动爬列表，无需输入文件）", args.source)
    else:
        get_logger().info("共 %d 个输入项，数据源=%s", len(args.items), args.source)
    RUNNERS[args.source](args)
    report()
# ======================= 兼容再导出 =======================
# 拆分前这些名字都在本模块，保留导出以免旧引用失效。
__all__ = [
    "SOURCES", "TARGETS", "RUNNERS", "main", "report", "run_all",
    "run_nmpa", "run_chictr", "run_wanfang", "run_guideline", "run_otc_ann",
    "merge_nmpa", "extract_seed_names", "sink_doc", "record_nmpa_failure",
]


if __name__ == "__main__":
    main()

