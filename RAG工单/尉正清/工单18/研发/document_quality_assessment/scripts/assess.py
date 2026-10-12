# 工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流
"""DocumentQualityAssessmentSkill 主入口：跑一轮质检，出 JSON 报告 + HTML 简报

## 怎么用

    python scripts/assess.py <目录或单个文件> [-c 配置] [-o 输出目录] [选项]

## 一条流程串起五大功能

    遍历文件 ─> 逐个解析（全文 / 逐页字符数 / SimHash / 敏感信息命中）─> 缓存落盘
                        │
                        └─> 汇总出报告：格式分布、PDF 分流、长度分布、重复检测、
                                        敏感信息、分类标签

## 为什么「解析」和「判定」要分开

1700 份文档解析一遍要几分钟，而**调阈值是任务四的家常便饭**（扫描页占比调一下、
SimHash 距离调一下、长度区间调一下）。所以：

- 缓存里存的是**事实**：字符数、每页字符数、SimHash、敏感信息命中
- 判定（扫描型还是文字型、打什么标签、算不算重复）**每次出报告现算**

结果：改 `assessment_config.yaml` 里的判定阈值，重跑只要几秒，不用重新解析 1700 份 PDF。
（改的是**抽取类**配置 —— 敏感信息正则、SimHash 取多少字符 —— 才需要 `--refresh`。）

## 中断恢复

缓存键是「路径 + 大小 + 修改时间」，追加写在 `workdir/parsed.jsonl`。
Ctrl+C、断电、跑到一半改代码，重跑都从断点接着来；文件真被改过会自动重解析（键变了）。
Ctrl+C 时会 `finally` 落盘，最多丢最后不到 `checkpoint_every` 份。
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SKILL_DIR = HERE.parent
sys.path.insert(0, str(HERE))

import common           # noqa: E402
import dedup            # noqa: E402
import extract          # noqa: E402
import report           # noqa: E402
import sensitive        # noqa: E402
import summarize        # noqa: E402

common.utf8_console()   # 控制台先切成 UTF-8，否则中文文件名能把质检打崩


# ---------------------------------------------------------------- 单文件解析

def parse_one(path, cfg, ckpt):
    """解析一个文件，结果写进缓存。返回 (记录, 是否命中缓存)。

    记录里全是**事实**，不含任何「这算不算扫描型/重复/敏感」的结论。
    任何异常都吞掉记进 `error` —— 知识库里坏文件是常态，不能让一份坏 PDF 掀掉整轮质检。
    """
    cached = ckpt.get(path)
    if cached is not None:
        return cached, True

    rec = {"path": str(path), "ext": Path(path).suffix.lower(), "size": 0,
           "chars": None, "error": "", "pages": None, "covers": None, "total_pages": 0,
           "simhash": None, "hits": []}

    try:
        rec["size"] = Path(path).stat().st_size
    except OSError as e:
        rec["error"] = f"读不到文件：{e}"
        ckpt.put(path, rec)
        return rec, False

    limit_mb = cfg["scan"]["max_parse_mb"]
    if rec["size"] / 1048576 > limit_mb:
        rec["error"] = f"文件过大（{rec['size'] / 1048576:.0f}MB > {limit_mb}MB），跳过正文抽取"
    else:
        text, err = _read(path, rec, cfg)
        rec["error"] = err
        if not err:
            # 必须先 strip 再数字符：扫描件每页抽出来的是空串，拼起来却是一串换行，
            # 一份 8 页的纯扫描件不 strip 会数出 7 个「字符」—— 那是换行符，不是内容。
            text = text.strip()
            rec["chars"] = len(text)
            if text:
                d = cfg["dedup"]
                rec["simhash"] = dedup.simhash(
                    dedup.sample_text(text, d["simhash_sample_chars"]),
                    d["simhash_bits"], d["simhash_shingle"])
                rec["hits"] = sensitive.scan_text(text, cfg, source=str(path))

    ckpt.put(path, rec)
    return rec, False


def _read(path, rec, cfg):
    """按扩展名读正文。PDF 顺带把逐页字符数和图片覆盖率一起拿回来
    （见 extract.pdf_parse —— 判扫描页要这两个量）。"""
    if rec["ext"] == ".pdf":
        text, counts, covers, total = extract.pdf_parse(
            path, cfg["pdf_type"]["sample_pages"])
        rec["pages"], rec["covers"], rec["total_pages"] = counts, covers, total
        if text is None:
            return "", "PDF 打不开（损坏或加密）"
        return text, ""
    return extract.extract_text(path, cfg["scan"]["max_parse_mb"])


# ---------------------------------------------------------------- 主流程

def assess(root, cfg=None, workdir=None, limit=0, refresh=False, progress=True):
    """跑一轮质检，返回报告 dict。CLI 和 API 端点都调这个函数。"""
    cfg = cfg or common.load_config()
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"路径不存在：{root}")

    files = sorted(common.iter_files(root, cfg))
    if limit:
        files = files[:limit]
    if not files:
        raise ValueError(f"{root} 下没有配置里登记的扩展名文件"
                         f"（scan.extensions = {cfg['scan']['extensions']}）")

    ckpt = common.Checkpoint(workdir or cfg["output"]["workdir"],
                             cfg["output"]["checkpoint_every"])
    if refresh:
        ckpt.reset()

    records, reused = [], 0
    t0 = time.time()
    try:
        for i, p in enumerate(files, 1):
            rec, hit = parse_one(p, cfg, ckpt)
            reused += hit
            records.append(rec)
            if progress and (i % 25 == 0 or i == len(files)):
                _progress(i, len(files), t0, reused)
    finally:
        # 中断也要落盘，否则断点续跑白做
        ckpt.flush()

    rep = summarize.build_report(root, records, cfg)
    rep["summary"]["elapsed_sec"] = round(time.time() - t0, 1)
    rep["summary"]["reused_from_cache"] = reused
    if progress:
        common.log(f"解析完成，用时 {time.time() - t0:.0f}s（缓存命中 {reused}/{len(files)}）")
    return rep


def _progress(done, total, t0, reused):
    el = time.time() - t0
    rate = done / el if el else 0
    eta = (total - done) / rate if rate else 0
    common.log(f"  [{done}/{total}] {done / total:>4.0%}  用时 {el:.0f}s  "
               f"预计还剩 {eta:.0f}s  缓存命中 {reused}")


def write_reports(rep, out_dir, cfg, html=True):
    """落盘 JSON（给程序读）+ HTML 简报（给人看）。返回 (json路径, html路径|None)。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp = report.write_json(rep, out / "quality_report.json")
    hp = report.write_html(rep, out / "quality_report.html", cfg) if html else None
    return jp, hp


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="assess.py",
        description="文档质量评估 Skill：格式分布 / PDF 分流 / 长度分布 / 重复检测 / "
                    "敏感信息 / 分类标签")
    ap.add_argument("root", help="要质检的目录（或单个文件）")
    ap.add_argument("-c", "--config", help="配置文件路径，默认用 Skill 自带的 assessment_config.yaml")
    ap.add_argument("-o", "--out", default="quality_report", help="报告输出目录（默认 quality_report）")
    ap.add_argument("--workdir", help="解析缓存目录，默认取配置里的 output.workdir")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个文件（试跑用）")
    ap.add_argument("--refresh", action="store_true",
                    help="丢掉解析缓存重新解析（改了敏感信息正则/SimHash 长度才需要）")
    ap.add_argument("--json-only", action="store_true", help="只出 JSON，不出 HTML")
    ap.add_argument("--quiet", action="store_true", help="不打印解析进度")
    a = ap.parse_args(argv)

    cfg = common.load_config(a.config)
    try:
        rep = assess(a.root, cfg, a.workdir, a.limit, a.refresh, progress=not a.quiet)
    except KeyboardInterrupt:
        common.log("\n已中断。解析缓存已落盘，重跑会接着来（不用加 --refresh）。")
        return 130
    except (FileNotFoundError, ValueError) as e:
        common.log(f"错误：{e}")
        return 2

    jp, hp = write_reports(rep, a.out, cfg, html=not a.json_only)
    common.log(f"JSON 报告：{jp}")
    if hp:
        common.log(f"HTML 简报：{hp}")
    report.print_brief(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
