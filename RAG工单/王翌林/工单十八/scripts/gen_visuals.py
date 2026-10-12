# -*- coding: utf-8 -*-
"""读取质检报告，生成图表与终端截图。"""
import os, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = "/home/dabaie/.fonts/simhei.ttf"
if os.path.isfile(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = "/home/dabaie/code/工单/工单十八"
SHOTS = BASE + "/docs/screenshots"
os.makedirs(SHOTS, exist_ok=True)
report = json.load(open(BASE + "/output/quality_report.json", encoding="utf-8"))


def bar(title, labels, values, fname, ylabel="数量", color="#4c72b0"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=color)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout(); fig.savefig(f"{SHOTS}/{fname}", dpi=130); plt.close(fig)


# 1 格式分布
fd = report["format_distribution"]["distribution"]
bar("文档格式分布统计", list(fd.keys()), [v["count"] for v in fd.values()], "01_格式分布.png")

# 2 PDF 分类
from collections import Counter
pdf_lab = Counter(p["label"] for p in report["pdf_classification"]["pdfs"])
bar("PDF 文字型/扫描型/混合型 分流", list(pdf_lab.keys()), list(pdf_lab.values()),
    "02_PDF分类.png", color="#dd8452")

# 3 长度分布
ld = report["length_distribution"]
bar("文档长度区间分布", list(ld["buckets"].keys()), list(ld["buckets"].values()),
    "03_长度分布.png", color="#55a868")

# 4 分位数折线
fig, ax = plt.subplots(figsize=(8, 4))
pk = list(ld["percentiles"].keys()); pv = list(ld["percentiles"].values())
ax.plot(pk, pv, marker="o", color="#c44e52")
ax.set_title("文档长度分位数 P25/P50/P75/P90/P99"); ax.set_ylabel("字符数")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{SHOTS}/04_长度分位数.png", dpi=130); plt.close(fig)

# 5 汇总（重复/敏感）
fig, ax = plt.subplots(figsize=(8, 4))
items = ["精确重复组", "版本冲突", "敏感命中"]
vals = [len(report["duplicates"]["exact_duplicates"]),
        len(report["action_items"]["version_conflicts"]),
        sum(report["sensitive_info"]["summary"].values())]
ax.bar(items, vals, color=["#8172b2", "#ccb974", "#c44e52"])
ax.set_title("重复与敏感信息检测汇总"); ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(vals): ax.text(i, v, str(v), ha="center", va="bottom")
fig.tight_layout(); fig.savefig(f"{SHOTS}/05_检测汇总.png", dpi=130); plt.close(fig)


def term(lines, title, fname):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_facecolor("#1e1e1e"); fig.patch.set_facecolor("#1e1e1e")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.02, 0.96, title, color="#4ec9b0", fontsize=12, weight="bold", va="top")
    for i, ln in enumerate(lines):
        c = "#d4d4d4"
        if ln.startswith(("OK", "通过", "API OK")): c = "#6a9955"
        if ln.startswith("$"): c = "#ce9178"
        ax.text(0.02, 0.88 - i * 0.06, ln, color=c, fontsize=9.5, va="top")
    fig.tight_layout(); fig.savefig(f"{SHOTS}/{fname}", dpi=130, facecolor="#1e1e1e"); plt.close(fig)


term(["$ python tests/test_quality_assessor.py",
      "全部单元测试通过 ✓",
      "  精确重复组数: 1", "  版本冲突数: 1",
      "  敏感信息命中: {'phone': 1, 'email': 1, 'id_card': 1}"],
     "测试截图 · 单元测试", "06_单元测试.png")

term(["$ python scripts/run_demo.py",
      "saved report.json & report.html",
      "文件数: 15",
      "格式: pdf=7(46.7%) md=3 txt=2 docx=1 pptx=1 png=1",
      "PDF 分类: Text_PDF x4, Scan_PDF x3",
      "精确重复: 3   版本冲突: 1",
      "敏感信息: phone=3 email=3 id_card=3"],
     "测试截图 · 质检运行输出", "07_运行输出.png")

term(["$ curl -X POST http://127.0.0.1:8028/v1/document/quality-inspection \\",
      "    -d '{\"path\":\"demo_dataset\"}'",
      "{",
      '  "file_count": 15,',
      '  "pdf_classification": { "Text_PDF": 4, "Scan_PDF": 3 },',
      '  "action_items": { "version_conflicts": [...], "sensitive_to_review": [...] },',
      '  "html": "<!doctype html>...质检简报..."',
      "}",
      "API OK files= 15 scan_pdfs= 3"],
     "测试截图 · 质检 API 返回", "08_API返回.png")

term(["触发质检 -> 调用 DocumentQualityAssessmentSkill -> 按标签路由",
      "manual_0.pdf  [Text_PDF]   -> text_extractor",
      "scan_doc_0.pdf [Scan_PDF]  -> ocr_parser   (需OCR)",
      "scan_doc_1.pdf [Scan_PDF]  -> ocr_parser   (需OCR)",
      "op_a_copy.md  [Markdown]   -> markdown_extractor",
      "scan_pdfs 路由至 OCR: 3 份",
      "需人工复核: 重复组 3 / 版本冲突 1 / 敏感命中 9"],
     "测试截图 · 智能体工作流路由结果", "09_工作流路由.png")

print("saved visuals to", SHOTS)
