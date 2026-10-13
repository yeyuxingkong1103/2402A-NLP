# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
"""
工单07 · 解压附件 + 语料规范化 + 工程自检（幂等，可重复运行）

只使用**工程目录内**的文件，不访问工程外的任何路径。

输入（工程内）
--------------
    ccf_competition.zip    工单附件（46.5 MB，20 个条目）
    sample_questions.pdf   工单要求的示例问题来源

输出
----
    ccf_competition/{pdf,txt}/     解压后的原始语料（文件名已还原为中文）
    data/corpus/{pdf,txt}/         规范化副本（本工程各模块实际读取的位置）
    data/corpus/sample_questions.pdf

关于中文文件名（一处**重要的更正**）
------------------------------------
`ccf_competition.zip` 由中文 Windows 的打包工具生成，条目名按 ZIP 规范以
**GBK 字节**存储且未置 UTF-8 标志位（flag bit 11）。Python `zipfile` 此时按
规范默认用 **CP437** 解码，于是一串 GBK 汉字就变成了乱码：

    默认解码: 2020-02-14__╞╜░▓╥°╨╨╣╔╖▌╙╨╧▐╣½╦╛__000001__...
    还原:     2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf

该映射是**无损、可逆**的：`name.encode('cp437').decode('gbk')` 即可还原。
（此前一版脚本曾误判为"有损不可逆"——那是因为当时拿到的是**已被二次破坏**的
解压副本：某人用 UTF-8+replace 解码过一次，U+FFFD 替换符落盘，信息才真正丢失。
直接对 zip 条目做 cp437→gbk 还原则完好无损。此结论已用 zip 内 20 个条目逐一验证。）

规范化命名
----------
还原出的原名较长（`{日期}__{公司全称}__{代码}__{公司简称}__{年}年__{年度报告}`）。
本工程统一压成短名，便于引用来源展示与 PDF/TXT 严格配对：

    {披露日期}__{公司简称}__{股票代码}__{报告年度}年度报告.{pdf,txt}

短名由原名的第 1、4、2、3 段直接取出，**不再依赖任何人工映射表**。

用法
----
    python scripts/setup_project.py            # 解压 + 规范化 + 自检
    python scripts/setup_project.py --check    # 只自检（不重新解压/复制）
    python scripts/setup_project.py --stale-to _stale   # 把旧乱码名副本挪到 _stale/
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ZIP_PATH = os.path.join(ROOT, "ccf_competition.zip")
SRC_CORPUS = os.path.join(ROOT, "ccf_competition")
DST_CORPUS = os.path.join(ROOT, "data", "corpus")

# 原名：{日期}__{全称}__{代码}__{简称}__{年}年__{报告名}.{ext}
_ORIG_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})__(?P<full>[^_]+)__(?P<code>\d{6})__"
    r"(?P<short>[^_]+)__(?P<year>\d{4})年__(?P<kind>[^_]+)$"
)

REQUIRED_SRC = [
    "bootstrap.py", "config.py", "pdf_parser.py", "pymupdf_parser.py",
    "table_parser.py", "table_chunker.py", "chunker.py", "image_blocks.py",
    "embedder.py", "vector_store.py", "fulltext.py", "fusion.py",
    "rerankers.py", "strategies.py", "query_rewrite.py", "coref.py",
    "dialogue.py", "doc_router.py", "rag.py", "runtime.py", "tokenizer.py",
]

DATA_DIRS = [
    "data/chunks", "data/index", "data/qdrant", "data/eval",
    "data/figures", "data/tables", "data/feedback",
    "data/parsed_mineru", "docs", "logs",
]

EXPECTED = 9


def _log(msg: str) -> None:
    print(msg, flush=True)


def _fix_name(name: str) -> str:
    """ZIP 条目名：CP437 误解码 → GBK 还原。失败则原样返回。"""
    try:
        return name.encode("cp437").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def _canonical(orig: str) -> str | None:
    """原始名 → 规范化短名。"""
    stem, ext = os.path.splitext(orig)
    m = _ORIG_RE.match(stem)
    if not m:
        return None
    return (f"{m['date']}__{m['short']}__{m['code']}__{m['year']}年度报告{ext}")


def extract_zip(stale_to: str | None) -> tuple[list[str], list[str]]:
    """解压附件到 ccf_competition/。返回 (已解压, 问题)。"""
    problems: list[str] = []
    if not os.path.isfile(ZIP_PATH):
        problems.append(f"缺少附件 {os.path.basename(ZIP_PATH)}")
        return [], problems

    extracted: list[str] = []
    with zipfile.ZipFile(ZIP_PATH) as z:
        for info in z.infolist():
            raw = _fix_name(info.filename)
            if raw.endswith("/"):
                continue
            dst = os.path.join(ROOT, raw.replace("/", os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(info) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
            extracted.append(raw)
    _log(f"  解压 {len(extracted)} 个文件 → ccf_competition/")

    # 旧版脚本遗留的乱码名副本：挪走而非删除，保证可回溯
    if stale_to:
        stale_dir = os.path.join(ROOT, stale_to)
        moved = 0
        for sub in ("pdf", "txt"):
            d = os.path.join(SRC_CORPUS, sub)
            if not os.path.isdir(d):
                continue
            os.makedirs(os.path.join(stale_dir, sub), exist_ok=True)
            for f in os.listdir(d):
                if _canonical(_fix_name(f)):
                    continue          # 名字正常，保留
                shutil.move(os.path.join(d, f),
                            os.path.join(stale_dir, sub, f))
                moved += 1
        if moved:
            _log(f"  旧乱码名副本 {moved} 个 → {stale_to}/（未删除，可人工核对）")
    return extracted, problems


def normalize_corpus() -> list[str]:
    """ccf_competition/ → data/corpus/（短名，PDF/TXT 严格同名）。"""
    problems: list[str] = []
    for sub in ("pdf", "txt"):
        src_dir = os.path.join(SRC_CORPUS, sub)
        dst_dir = os.path.join(DST_CORPUS, sub)
        if not os.path.isdir(src_dir):
            problems.append(f"缺少原始目录 {sub}/")
            continue
        os.makedirs(dst_dir, exist_ok=True)
        n = skip = 0
        for f in sorted(os.listdir(src_dir)):
            canon = _canonical(_fix_name(f))
            if not canon:
                skip += 1
                continue
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, canon))
            n += 1
        _log(f"  {sub}/ 规范化 {n} 份" + (f"（跳过 {skip} 个非语料文件）" if skip else ""))

    src_q = os.path.join(ROOT, "sample_questions.pdf")
    if os.path.isfile(src_q):
        os.makedirs(DST_CORPUS, exist_ok=True)
        shutil.copy2(src_q, os.path.join(DST_CORPUS, "sample_questions.pdf"))
        _log("  sample_questions.pdf 已复制")
    else:
        problems.append("缺少 sample_questions.pdf")
    return problems


def check_src() -> list[str]:
    missing = [m for m in REQUIRED_SRC
               if not os.path.isfile(os.path.join(ROOT, "src", m))]
    _log(f"[3/4] RAG 系统模块：{len(REQUIRED_SRC) - len(missing)}/{len(REQUIRED_SRC)} 在位")
    for m in missing:
        _log(f"      [缺失] src/{m}")
    return missing


def check_corpus() -> list[str]:
    problems: list[str] = []
    pdf_dir = os.path.join(DST_CORPUS, "pdf")
    txt_dir = os.path.join(DST_CORPUS, "txt")
    pdfs = sorted(f for f in os.listdir(pdf_dir)
                  if f.lower().endswith(".pdf")) if os.path.isdir(pdf_dir) else []
    txts = sorted(f for f in os.listdir(txt_dir)
                  if f.lower().endswith(".txt")) if os.path.isdir(txt_dir) else []

    _log(f"[4/4] 语料自检：PDF {len(pdfs)} 份 / TXT {len(txts)} 份")
    if len(pdfs) != EXPECTED:
        problems.append(f"PDF 份数 {len(pdfs)} != 预期 {EXPECTED}")
    if len(txts) != EXPECTED:
        problems.append(f"TXT 份数 {len(txts)} != 预期 {EXPECTED}")
    ps = {os.path.splitext(f)[0] for f in pdfs}
    ts = {os.path.splitext(f)[0] for f in txts}
    problems += [f"PDF 无对应 TXT：{s}" for s in sorted(ps - ts)]
    problems += [f"TXT 无对应 PDF：{s}" for s in sorted(ts - ps)]
    for f in pdfs:
        _log(f"      {f}")
    if not os.path.isfile(os.path.join(DST_CORPUS, "sample_questions.pdf")):
        problems.append("缺少 data/corpus/sample_questions.pdf")
    for p in problems:
        _log(f"      [问题] {p}")
    return problems


def make_dirs() -> None:
    for rel in DATA_DIRS:
        os.makedirs(os.path.join(ROOT, rel.replace("/", os.sep)), exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="工单07 解压附件、语料规范化与工程自检")
    ap.add_argument("--check", action="store_true", help="只自检，不解压/复制")
    ap.add_argument("--stale-to", default="_stale",
                    help="旧乱码名副本挪到该目录（默认 _stale/；传空串则不挪）")
    args = ap.parse_args()

    _log(f"工程根目录：{ROOT}\n")
    problems: list[str] = []

    if args.check:
        _log("[1/4] 跳过解压（--check）")
        _log("[2/4] 跳过语料规范化（--check）")
    else:
        _log("[1/4] 解压 ccf_competition.zip（文件名 cp437→gbk 还原）")
        _e, p = extract_zip(args.stale_to or None)
        problems += p
        _log("[2/4] 规范化语料 ccf_competition/ → data/corpus/")
        problems += normalize_corpus()

    problems += check_src()
    problems += check_corpus()
    make_dirs()

    if problems:
        _log(f"\n[失败] 共 {len(problems)} 项问题，详见上文。")
        return 1
    _log("\n[通过] 附件解压、语料规范化与工程自检全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
