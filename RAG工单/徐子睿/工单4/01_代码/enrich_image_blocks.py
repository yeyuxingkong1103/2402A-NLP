# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 模块：tools/enrich_image_blocks —— 图像块裁剪与图注补充
# 说明：CLIP 生成的整页图像块数量多、文本泛，会淹没正文/表格块。这里：
#   1) 只保留“真正有图”的整页块（页面含图注，或抽到过图表，或矢量绘图面积大）；
#   2) 把页面的**图注文字**（含“图”的短行，如“2008 年中国IC市场应用结构与增长(亿元)”）
#      拼进图像块文本 → 让“XX图”这类问题能直接命中该页。
# 输出：覆盖写 data/vision/image_blocks.jsonl
# ⚠️ 流水线顺序：render_pages -> vision_build(CLIP) -> **enrich_image_blocks(本步)** -> build_kb。
#    若在 enrich 之后重跑 vision_build，会把本步补的图注/页面文字覆盖掉（实测踩坑），必须再跑一次本步。
import os
import re
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")
import pymupdf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "data", "figures")
VIS = os.path.join(ROOT, "data", "vision")


def page_captions(doc):
    caps = {}
    for i, page in enumerate(doc, 1):
        lines = []
        for ln in page.get_text().splitlines():
            s = ln.strip()
            if not s or len(s) > 50:
                continue
            if "图" in s and re.search(r"(图\s*\d|图：|图:|\d{4}\s*年.*图|结构图|示意图|流程图)", s):
                lines.append(s)
        # 矢量绘图面积占比（组织结构图这类矢量图）
        try:
            area = sum(abs(pymupdf.Rect(d["rect"]).get_area()) for d in page.get_drawings() if d.get("rect"))
        except Exception:  # noqa: BLE001
            area = 0
        ratio = area / (page.rect.width * page.rect.height or 1)
        full = re.sub(r"\s+", " ", page.get_text() or "").strip()
        caps[i] = {"captions": lines[:3], "draw_ratio": round(ratio, 3),
                   "head": full[:150], "tail": full[-350:], "full": full[:1200]}
    return caps


def main():
    doc_id = sys.argv[1] if len(sys.argv) > 1 else "招股说明书2"
    pdf = os.path.join(ROOT, "data", "pdfs", doc_id + ".pdf")
    doc = pymupdf.open(pdf)
    caps = page_captions(doc)

    fig_pages = set()
    fj = os.path.join(FIG, "figures.json")
    if os.path.exists(fj):
        for x in json.load(open(fj, encoding="utf-8")):
            if x["doc"] == doc_id:
                fig_pages.add(x["page"])

    def _norm(p):
        return (p or "").replace("/", "\\")

    # 垃圾裁图过滤：PDF2 第 343-349 页（第十四节 有关声明）每页被切成 8 条
    # 595x94 的横条（签名/声明页），既非图表也无有效图注，必须剔除。
    fig_text = {}
    if os.path.exists(fj):
        for x in json.load(open(fj, encoding="utf-8")):
            if x["doc"] == doc_id:
                fig_text[_norm(x["file"])] = (x.get("text_layer") or "").strip()

    strip_paths = set()
    if os.path.exists(fj):
        for x in json.load(open(fj, encoding="utf-8")):
            if x["doc"] == doc_id and (x["w"] >= 500 or x["h"] < 150):
                strip_paths.add(_norm(x["file"]))

    src = os.path.join(VIS, "image_blocks.jsonl")
    out, kept, dropped = [], 0, 0
    for line in open(src, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        b = json.loads(line)
        if b["doc"] != doc_id:
            out.append(b)
            continue
        if b["kind"] == "figure" and _norm(b.get("img_path")) in strip_paths:
            dropped += 1
            continue
        info = caps.get(b["page"], {})
        caps_join = " / ".join(info.get("captions") or [])
        if b["kind"] == "page":
            # 页面级块：标签 + 图注 + 页面文字（截断 400 字）。
            # ⚠️ 只给“整页块”补页面文字，**绝不**给同页的多张裁图都补整页文字，
            #    否则同一页正文会在索引里被复制 N 份（实测：销售流程页每页 8 张裁图
            #    导致该页文字权重 ×8，把真正的正文/表格块挤下去）。
            snippet = (info.get("full") or "")[:400]
            cap_txt = ("；页面文字：" + snippet) if snippet else ""
            if caps_join:
                cap_txt += "；图注：" + caps_join
        else:
            # 图表裁图：只用它自己的图内文字层 + 图注，不重复整页正文
            tl = fig_text.get(_norm(b.get("img_path")), "")
            cap_txt = ("；图内文字：" + tl[:200]) if tl else ""
            if caps_join:
                cap_txt += "；图注：" + caps_join
        if b["kind"] == "page":
            keep = bool(info.get("captions")) or (b["page"] in fig_pages) or info.get("draw_ratio", 0) >= 0.05
            if not keep:
                dropped += 1
                continue
        # 可重复执行：文本从 clip_tags 重建，不再叠加
        if b.get("clip_tags"):
            tags = "、".join("%s(%s)" % (t["zh"], t["en"]) for t in b["clip_tags"])
            head = "【图片·第%d页整页】" % b["page"] if b["kind"] == "page" else "【图片·第%d页图表】" % b["page"]
            b["text"] = head + "图像语义标签：" + tags + cap_txt
        else:
            b["text"] = b["text"] + cap_txt
        b["captions"] = info.get("captions", [])
        out.append(b)
        kept += 1
    with open(src, "w", encoding="utf-8") as f:
        for b in out:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print("kept=%d dropped=%d total=%d" % (kept, dropped, len(out)))
    for b in out:
        if b["page"] in (38, 39, 72, 310):
            print("  p%-4d %s | %s" % (b["page"], b["kind"], b["text"][:140]))


if __name__ == "__main__":
    main()
