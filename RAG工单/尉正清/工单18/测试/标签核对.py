# 工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流
"""分类标签准确率核对 —— 工单验收标准「准确率需达到 85% 以上」

## 为什么要用**另一个**指标来核对

标签是按「每页可提取字符数」判的。如果核对时还去数字符数，那是拿同一个指标
自己证明自己，测出来必然 100%，没有意义。

所以这里换一个**物理上独立**的信号：**页面里有没有图片**。

    扫描件   = 纸扫进来的，每页是一张大图，没有字体
    文字型   = 数字排版的原生 PDF，用字体渲染，没有整页大图

字符数是从**文字层**读的，图片是从**页面资源**读的，两条路互不相干。
两者结论一致，才能说明「字符数少 = 扫描件」这条启发式在这批语料上站得住。

## 判据

对每一页取两个布尔量：`has_font`（这页有字体在用）、`has_image`（这页引用了图片）。
整份文档按页多数投票：

    图片页占比 ≥ 70%  →  独立判定为「扫描型」
    图片页占比 ≤ 30%  →  独立判定为「文字型」
    中间               →  独立判定为「混合型」

阈值（70%/30%）和主流程配置里的取一致，这样两个判定可以直接比。
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "研发"
                       / "document_quality_assessment" / "scripts"))
import common   # noqa: E402

common.utf8_console()

IMG_HI, IMG_LO = 0.7, 0.3


def page_signals(pdf, max_pages=30):
    """独立信号：逐页看有没有图片 / 有没有字体。

    只看前 max_pages 页，和主流程的 sample_pages 对齐。
    """
    import fitz
    with fitz.open(pdf) as doc:
        n = doc.page_count
        limit = n if not max_pages else min(n, max_pages)
        img_pages = 0
        for i in range(limit):
            try:
                p = doc[i]
                has_img = bool(p.get_images(full=True))
                # 字体：页面里有没有可选中的文字对象。用 rawdict 避开
                # 「字符数阈值」那条老路，这里只关心「有没有文字对象」
                has_txt = bool(p.get_text("rawdict").get("blocks"))
            except Exception:
                continue
            if has_img and not has_txt:
                # 有图没字 —— 最典型的扫描页
                img_pages += 1
            elif has_img and has_txt:
                # 图文混排：算半页（占位图、印章、页眉 logo 都长这样）
                img_pages += 0.5
        return img_pages / limit if limit else 0.0


def independent_kind(ratio):
    if ratio >= IMG_HI:
        return "scanned"
    if ratio <= IMG_LO:
        return "text"
    return "mixed"


def main(report_path, per_tag=25, seed=18):
    rep = json.loads(Path(report_path).read_text(encoding="utf-8"))
    docs = rep["details"]["tags"]["docs"]
    by_tag = {}
    for path, tags, primary in docs:
        by_tag.setdefault(tags, []).append(path)

    random.seed(seed)
    print(f"标签共 {len(by_tag)} 种，每种随机抽 {per_tag} 份做独立核对\n")
    tally, detail = Counter(), []
    for tag, paths in sorted(by_tag.items(), key=lambda x: -len(x[1])):
        sample = random.sample(paths, min(per_tag, len(paths)))
        agree = 0
        for p in sample:
            ratio = page_signals(p)
            ind = independent_kind(ratio)
            # 标签名 → 期望的 PDF 类型
            expect = {"Scan_PDF": "scanned", "Text_PDF": "text",
                      "Mixed_PDF": "mixed"}.get(tag.split("、")[0])
            ok = (ind == expect)
            agree += ok
            detail.append((tag, Path(p).name, ind, round(ratio, 2), ok))
        tally[tag] = (agree, len(sample))
        print(f"  {tag:<28} {agree}/{len(sample)} 一致  ({agree / len(sample):.0%})")

    tot_a = sum(a for a, _ in tally.values())
    tot_n = sum(n for _, n in tally.values())
    print(f"\n合计 {tot_a}/{tot_n} = {tot_a / tot_n:.1%}"
          f"（工单验收线 85%）→ {'通过' if tot_a / tot_n >= 0.85 else '不通过'}")

    print("\n不一致的样本（人工看这些）：")
    bad = [d for d in detail if not d[4]]
    if not bad:
        print("  无")
    for tag, name, ind, ratio, _ in bad:
        print(f"  {name}  标签={tag}  独立判定={ind}  图片页占比={ratio}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "results/imdr/quality_report.json",
         int(sys.argv[2]) if len(sys.argv) > 2 else 25)
