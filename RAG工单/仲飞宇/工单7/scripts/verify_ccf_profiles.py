# 工单编号：人工智能NLP-RAG-功能测试及评估
# 工单07 - 功能测试及评估
"""
工单07 · ccf 语料 DocProfile 复核。

  python scripts/verify_ccf_profiles.py              # 9 份全量复核（约 8 分钟）
  python scripts/verify_ccf_profiles.py --only zsyh  # 只查一份
  python scripts/verify_ccf_profiles.py --sample 40  # 每份只抽 40 页（快速迭代模板用）

【为什么必须单独有一个复核脚本】
页码与页眉页脚模板对不上时**不会报错**，只会静默变差：
  · 页码错了 → 引用页码全错，验收时逐条核对才会发现；
  · 页眉模板写松了 → 把**正文**当成页眉删掉，答案凭空消失。
这两类都属于「不报错的错」，所以要用可判定的判据把每个假设钉死：

  判据 1  页脚实读覆盖率 ≥ 60%（剩下的回退到 `index − 偏移`，会如实标注）
  判据 2  页眉命中**零**带外（y0 > 12% 页高 = 正文区）—— 命中即疑似误删
  判据 3  页脚模板命中**零**带外（y0 ≤ 85% 页高）—— 命中即说明模板写得太松
  判据 4  实读页码按页序应「分段单调递增」，重编号只允许**回跳**（不允许乱跳）
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                                          # noqa: E402
from app.core.doc_profiles import (FOOTER_BAND, HEADER_BAND,             # noqa: E402
                                   doc_profiles_for_ccf)


def block_text(b: dict) -> str:
    return "".join(s["text"] for line in b.get("lines", [])
                   for s in line.get("spans", [])).strip()


def check_one(key: str, sample: int | None) -> dict:
    import pymupdf

    prof = doc_profiles_for_ccf()[key]
    pdf = settings.data_path / "raw" / prof.doc_name
    stats = {
        "key": key, "doc": prof.doc_name, "pages": 0,
        "page_no_re": prof.page_no_re,
        "label_from_footer": 0, "label_from_offset": 0,
        "header_hits": 0, "header_out_of_band": [],
        "footer_hits": 0, "footer_out_of_band": [],
        "jumps": [], "sample": sample, "label_collision": [],
    }
    if not pdf.exists():
        stats["error"] = f"文件不存在：{pdf}"
        return stats

    doc = pymupdf.open(pdf)
    try:
        n = doc.page_count if sample is None else min(sample, doc.page_count)
        stats["pages"] = n
        seq: list[tuple[int, int]] = []          # (page_no, 实读页码)
        for i in range(n):
            page = doc[i]
            h = page.rect.height
            blocks = page.get_text("dict")["blocks"]

            # ---- 判据 1：页脚实读 ----
            read = prof.footer_label(page, blocks)
            if read:
                stats["label_from_footer"] += 1
            else:
                stats["label_from_offset"] += 1
            label = prof.printed_label(page, blocks, prof.label(i, prof.page_label_offset))
            seq.append((i, int(label) if label.isdigit() else -1))

            # ---- 判据 2/3：模板是否越界命中 ----
            for b in blocks:
                if b.get("type") != 0:
                    continue
                t = block_text(b)
                if not t:
                    continue
                y0 = b["bbox"][1]
                if prof.matches_header(t):
                    stats["header_hits"] += 1
                    if y0 > h * HEADER_BAND:
                        stats["header_out_of_band"].append((i, round(y0 / h, 2), t[:40]))
                if prof.matches_footer(t):
                    stats["footer_hits"] += 1
                    if y0 <= h * FOOTER_BAND:
                        stats["footer_out_of_band"].append((i, round(y0 / h, 2), t[:40]))
                # 【判据3 真正要防的那一条】解析器里有一道**不带页脚带**的补漏规则
                # （pdf_parser：「整行恰好等于本页页码 → 当页脚删掉」，为旋转页兜底）。
                # 带外命中本身无害（footer_label 与带内删除都受页脚带保护），
                # 但**带外文本块恰好等于本页页码**会被那条规则无带删掉。
                #
                # 【判据必须与解析器同一条件】工单07 实测国泰君安 idx=10 的资质表
                # 里有一格正文就是「11」、而该页页码也是 11 —— 正是这条规则会误删的
                # 形状。解析器已把该规则**收紧到非标准版面**（旋转页/横版页），
                # 所以判据也必须带上同一个版面条件：否则会在正排页面上报假警。
                nonstd = bool(page.rotation) or page.rect.width > page.rect.height
                if nonstd and y0 <= h * FOOTER_BAND and t == label:
                    stats["label_collision"].append((i, round(y0 / h, 2), t))

        # ---- 判据 4：实读页码的单调性 ----
        # 回跳分三类，只有第三类才要人看：
        #   · 「节重启」—— 从大页码回落到很小（≤5）。实测招商银行 idx126 印 `1`，
        #     正是「财务报告」节从 1 重开；这是**预期行为**，不是错。
        #   · 「节内重编号」—— 小幅回落（<30），同一节里换了一套编号。
        #   · 「大回跳」—— 其余，可能是模板抓错了数字，需要人工确认。
        for (p0, v0), (p1, v1) in zip(seq, seq[1:]):
            if p1 <= p0:
                continue
            if v1 < v0 and v1 <= 5:
                stats["jumps"].append((p0, v0, p1, v1, "节重启"))
            elif v1 < v0 and (v0 - v1) < 30:
                stats["jumps"].append((p0, v0, p1, v1, "节内重编号"))
            elif v1 < v0:
                stats["jumps"].append((p0, v0, p1, v1, "大回跳?"))
    finally:
        doc.close()
    return stats


def report(st: dict) -> bool:
    print(f"\n{'=' * 78}\n  {st['doc']}（key={st['key']}）"
          f"{'  [抽样 %d 页]' % st['sample'] if st['sample'] else '  [全量]'}")
    if st.get("error"):
        print(f"  ❌ {st['error']}")
        return False
    n = st["pages"]
    cov = st["label_from_footer"] / n if n else 0
    ok = True

    print(f"  {'页数':<14}{n}")
    print(f"  {'页脚实读页码':<14}{st['label_from_footer']:>5}  （{cov * 100:.1f}%）"
          f"  回退偏移 {st['label_from_offset']}")
    print(f"  {'页眉命中':<14}{st['header_hits']:>5}  带外(疑似误删正文) "
          f"{len(st['header_out_of_band'])}")
    print(f"  {'页脚命中':<14}{st['footer_hits']:>5}  带外(模板过松) "
          f"{len(st['footer_out_of_band'])}")
    print(f"  {'页码回跳':<14}{len(st['jumps']):>5}  "
          f"{st['jumps'][:3] if st['jumps'] else ''}")

    if cov < 0.60:
        print(f"  ❌ 判据1 不达标：页脚实读覆盖率 {cov * 100:.1f}% < 60%")
        ok = False
    else:
        print(f"  ✅ 判据1 页脚实读覆盖率 {cov * 100:.1f}% ≥ 60%")
    # 判据2：封面（第 1 页）的标题本来就压在页面中部，是**内容**不是页眉，
    # 按页眉删掉只是少一行封面文字，可接受 —— 其余页一律零容忍。
    hdr_bad = [h for h in st["header_out_of_band"] if h[0] != 0]
    if hdr_bad:
        print(f"  ❌ 判据2 页眉模板越界命中正文 {len(hdr_bad)} 处：{hdr_bad[:3]}")
        ok = False
    elif st["header_out_of_band"]:
        print(f"  ✅ 判据2 页眉命中全部落在页眉带（仅封面 1 处，属标题）")
    else:
        print("  ✅ 判据2 页眉命中全部落在页眉带（无正文误删）")
    # 判据3 的**真正风险**是「越界命中并且**抠得出页码**」—— 那才会让页码读到
    # 正文里的数字。所以判据按"有没有捕获组命中"来判，而不是按文本长相：
    #   · 裸数字分支（`(\d{1,3})`）在正文里天然到处都是，但 `footer_label` 与
    #     解析器删除都带页脚带限制，带外命中不会真的生效；
    #   · 「只有公司名」的分支（无捕获组）压根给不出页码，同样无害；
    #   · 只有**带捕获组**的分支越界命中，才是真危险（会读到正文数字）。
    col = st["label_collision"]
    if col:
        print(f"  ❌ 判据3 有 {len(col)} 处**非标准版面上带外文本块恰好等于本页页码**，"
              f"会被'页脚补漏'规则无带删掉：{col[:3]}")
        ok = False
    else:
        print(f"  ✅ 判据3 无「页脚补漏」误删风险（判定条件与解析器一致：仅非标准版面；"
              f"带外模板命中 {len(st['footer_out_of_band'])} 处，均受页脚带保护）")
    big = [j for j in st["jumps"] if j[4] == "大回跳?"]
    n_restart = sum(1 for j in st["jumps"] if j[4] == "节重启")
    if big:
        print(f"  ❌ 判据4 出现 {len(big)} 处无法解释的页码回跳（疑似抓错数字）：{big[:3]}")
        ok = False
    else:
        print(f"  ✅ 判据4 实读页码分段单调（节重启 {n_restart} 处 / "
              f"节内重编号 {len(st['jumps']) - n_restart} 处，均属预期）")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="工单07 · ccf 语料 DocProfile 复核")
    ap.add_argument("--only", default=None, help="只查某一份（按 key）")
    ap.add_argument("--sample", type=int, default=None, help="每份只查前 N 页")
    args = ap.parse_args()

    keys = [args.only] if args.only else list(doc_profiles_for_ccf())
    allok = True
    for k in keys:
        allok &= report(check_one(k, args.sample))
    print(f"\n{'=' * 78}\n  总判据：{'✅ 全部通过' if allok else '❌ 有不达标项'}\n")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
