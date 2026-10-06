# -*- coding: utf-8 -*-
"""
图像抽取模块（含矢量图整页渲染回退）
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

职责：
  1. 从《招股说明书2.pdf》（可选《招股说明书1.pdf》）抽取图像并落盘到 results/images/；
  2. 对「矢量绘制」的图表（组织结构图、市场结构图等）做整页渲染回退；
  3. 用 CLIP 对每张图做零样本分类，标注图表类型；
  4. 输出图片清单 results/image_inventory.json（页码/尺寸/来源/CLIP 分类）。

为什么不能直接用 rag_core.pdf_parse.extract_images：
  本 PDF 每页都嵌有 15 个 143x127 的「八维教育」水印切片（全文档共 5250 个），
  rag_core 的 min_size=120 过滤拦不住它们，导致：
    · 每页都被判定为「有内嵌图」，矢量图整页渲染回退永远不会触发；
    · 图片清单被水印淹没，向量库被灌入数千条无检索价值的图片块。
  本模块在 rag_core 的基础上补三件事：
    · 水印过滤：按 xref 在全文档的重复次数识别水印/背景切片；
    · 图页判定增强：rag_core._page_looks_like_figure 要求正文 <200 字，
      而第 39 页组织结构图有 323 字的竖排节点文字，会被漏判；本模块补充
      「标题关键词 + 绘图对象密度」判据；
    · 高清渲染：图页按 dpi=200 渲染（rag_core 固定 150），保证图中数字可读。

用法：
    python src/image_extractor.py                     # 抽取《招股说明书2》全部图页
    python src/image_extractor.py --docs 2 --pages 39,72,310   # 只抽指定页
    python src/image_extractor.py --no-clip           # 跳过 CLIP 分类（省算力）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

# --- 导入共享核心库（工单01~13 共用，禁止修改）---
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fitz  # PyMuPDF

from rag_core import config  # noqa: E402
from rag_core.pdf_parse import _page_looks_like_figure  # noqa: E402

# ---------------------------------------------------------------------------
# 路径常量（工单04 的全部产物都落在本工单目录下，不污染 data/）
# ---------------------------------------------------------------------------
WORKDIR = Path(__file__).resolve().parents[1]          # 工单04-图像内容解析及检索优化/
RESULTS_DIR = WORKDIR / "results"
IMAGES_DIR = RESULTS_DIR / "images"
INVENTORY_JSON = RESULTS_DIR / "image_inventory.json"
INVENTORY_MD = RESULTS_DIR / "image_inventory.md"

# 可选文档：键为 --docs 传入的编号
DOCS = {
    "2": {"name": "招股说明书2", "path": config.PDF_PROSPECTUS_2},
    "1": {
        "name": "招股说明书1",
        # 优先用无水印版：水印会干扰多模态模型对图的识别
        "path": (config.PDF_PROSPECTUS_1_NW
                 if Path(config.PDF_PROSPECTUS_1_NW).exists()
                 else config.PDF_PROSPECTUS_1),
    },
}

# 水印/背景切片判定：单边小于该像素且在全文档重复出现
WATERMARK_MAX_SIDE = 200
WATERMARK_MIN_REPEAT = 5
# 有效图像的最小边长（小于此值视为装饰性元素）
MIN_SIDE = 200
# 图页整页渲染分辨率（150 会让图表中的小字号数字糊掉）
RENDER_DPI = 200

# 图页标题关键词：招股书图表的题注/正文常带这些词
FIGURE_CAPTION_PAT = re.compile(
    r"(组织结构图|结构图|应用结构与增长|增长率|示意图|流程图|趋势图|"
    r"分布图|对比图|占比图|构成图|图\s*\d+[-－—]?\d*)"
)


# ---------------------------------------------------------------------------
# 水印识别
# ---------------------------------------------------------------------------
def count_xref_occurrences(doc: "fitz.Document") -> Counter:
    """统计每个图像 xref 在全文档的出现次数（水印会在每页重复出现）。"""
    cnt: Counter = Counter()
    for page in doc:
        for img in page.get_images(full=True):
            cnt[img[0]] += 1
    return cnt


def is_watermark(xref: int, counts: Counter, n_pages: int) -> bool:
    """
    判定某 xref 是否为水印/背景切片。

    依据：小尺寸 + 高重复。招股书里的「八维教育」水印是 143x127 的切片，
    全文档重复 5250 次；正常插图通常只出现 1~2 次。
    """
    return counts.get(xref, 0) >= max(WATERMARK_MIN_REPEAT, int(0.3 * n_pages))


# ---------------------------------------------------------------------------
# 图页判定（rag_core._page_looks_like_figure 的增强版）
# ---------------------------------------------------------------------------
def detect_figure_page(page) -> tuple[bool, str]:
    """
    判断某页是否为「图表页」，返回 (是否图页, 判定依据)。

    三条判据（满足其一即可）：
      a) 题注关键词命中 + 有矢量绘图对象   —— 招股书图表的标准特征
      b) 正文稀疏 + 绘图对象多             —— 等价 rag_core 的原始启发式
      c) 绘图对象密集 + 汉字量不大          —— 兜住「整页都是图、但节点文字多」的情况
         （第 39 页组织结构图 323 字，用 rag_core 的 <200 字阈值会漏判）

    注：本函数只做「页面级」判定，真正的矢量图回退还要求该页没有可用的内嵌图。
    """
    text = page.get_text("text")
    n_draw = len(page.get_drawings())
    han = len(re.findall(r"[一-鿿]", text))

    if FIGURE_CAPTION_PAT.search(text) and n_draw >= 10:
        return True, "题注关键词+矢量绘图"
    if _page_looks_like_figure(page):
        return True, "正文稀疏+矢量绘图"
    if n_draw >= 25 and han < 600:
        return True, "绘图对象密集"
    return False, ""


def extract_caption(page, max_len: int = 80) -> str:
    """抽取该页最可能的图题注（取关键词命中的最短一行，避免截到正文长句）。"""
    cands: list[str] = []
    for line in page.get_text("text").split("\n"):
        s = line.strip()
        if not s or len(s) > max_len:
            continue
        if FIGURE_CAPTION_PAT.search(s):
            cands.append(s)
    if not cands:
        return ""
    return min(cands, key=len)


# ---------------------------------------------------------------------------
# 单页图像抽取
# ---------------------------------------------------------------------------
def _save_pixmap(doc, xref: int, out_file: Path) -> tuple[int, int]:
    """从 xref 导出位图（处理 CMYK / 透明通道），返回 (宽, 高)。"""
    pix = fitz.Pixmap(doc, xref)
    if pix.n - pix.alpha >= 4:                     # CMYK -> RGB
        pix = fitz.Pixmap(fitz.csRGB, pix)
    pix.save(out_file)
    return pix.width, pix.height


def _img_size(doc, xref: int) -> tuple[int, int]:
    pix = fitz.Pixmap(doc, xref)
    return pix.width, pix.height


def extract_doc_images(
    pdf_path: Path,
    doc_name: str,
    out_dir: Path,
    pages: list[int] | None = None,
    min_side: int = MIN_SIDE,
    dpi: int = RENDER_DPI,
    force_render_pages: bool = False,
) -> list[dict]:
    """
    抽取一份 PDF 中的图像，产出图片清单记录。

    Args:
        pages: 只处理这些页码（1 起）；None 表示全文档
        min_side: 有效图像最小边长
        dpi: 整页渲染回退时的分辨率
        force_render_pages: True 时对图页即使有内嵌图也追加整页渲染

    Returns:
        [{image_id, doc, page, source, file, width, height, caption, ...}]
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"源 PDF 不存在：{pdf_path}")

    out_dir = Path(out_dir) / doc_name
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    counts = count_xref_occurrences(doc)
    n_pages = doc.page_count
    records: list[dict] = []

    for i, page in enumerate(doc):
        page_no = i + 1
        if pages and page_no not in pages:
            continue

        is_fig, why = detect_figure_page(page)
        caption = extract_caption(page) if is_fig else ""

        # --- L1：内嵌位图（已过滤水印）---
        usable = []
        for j, img in enumerate(page.get_images(full=True)):
            xref, w, h = img[0], img[2], img[3]
            if is_watermark(xref, counts, n_pages):
                continue
            if w < min_side or h < min_side:
                continue
            usable.append((j, xref, w, h))

        got = False
        for j, xref, w, h in usable:
            fp = out_dir / f"p{page_no}_img{j}.png"
            try:
                ww, hh = _save_pixmap(doc, xref, fp)
            except Exception as e:                       # 单张失败不影响整体
                print(f"  [warn] 第{page_no}页图像{j}(xref={xref})导出失败：{e}")
                continue
            records.append({
                "image_id": f"{doc_name}#p{page_no}#img{j}",
                "doc": doc_name,
                "page": page_no,
                "source": "embedded",
                "file": str(fp.relative_to(WORKDIR)).replace("\\", "/"),
                "abs_file": str(fp),
                "width": ww,
                "height": hh,
                "xref": xref,
                "bytes": fp.stat().st_size,
                "caption": caption,
                "figure_page": is_fig,
                "figure_reason": why,
                "dup_xref": counts.get(xref, 0) > 2,
            })
            got = True

        # --- L2：矢量图回退 —— 整页渲染 ---
        # 触发条件：该页被判定为图页 且（没有可用内嵌图 或 强制渲染）
        if is_fig and (not got or force_render_pages):
            fp = out_dir / f"p{page_no}_fullpage.png"
            pix = page.get_pixmap(dpi=dpi)
            pix.save(fp)
            ww, hh = pix.width, pix.height               # 渲染后的真实像素尺寸
            records.append({
                "image_id": f"{doc_name}#p{page_no}#fullpage",
                "doc": doc_name,
                "page": page_no,
                "source": "page_render",                 # 矢量图回退产物
                "file": str(fp.relative_to(WORKDIR)).replace("\\", "/"),
                "abs_file": str(fp),
                "width": int(ww),
                "height": int(hh),
                "xref": None,
                "bytes": fp.stat().st_size,
                "caption": caption,
                "figure_page": True,
                "figure_reason": why + ("|有内嵌图仍强制渲染" if got else "|无可用内嵌图"),
                "dup_xref": False,
            })
            got = True

        if not got:
            print(f"  [info] 第{page_no}页无有效图像"
                  + (f"（图页判据：{why}）" if is_fig else ""))

    doc.close()
    return records


# ---------------------------------------------------------------------------
# CLIP 零样本分类（工单备注要求「CLIP 或多模态大模型」，两者级联使用）
# ---------------------------------------------------------------------------
_CLIP_WARNED = False


def classify_with_clip(image_path: str | Path, top_n: int = 3) -> list[dict]:
    """
    用 CLIP 给图像打类型标签。

    CLIP 在这里的角色是「粗筛」：判定图属于组织结构图/柱状图/…，据此选择
    VLM 的专用 prompt（组织结构图走 _ORG_PROMPT，图表走 _CHART_PROMPT）。
    模型缺失或推理失败时返回空列表，不阻断主流程（下游按通用 prompt 处理）。
    """
    global _CLIP_WARNED
    try:
        from rag_core.image_parse import CLIPImageParser
        return CLIPImageParser().classify(image_path, top_n=top_n)
    except Exception as e:
        if not _CLIP_WARNED:             # 只提示一次，避免刷屏
            _CLIP_WARNED = True
            print(f"  [warn] CLIP 分类不可用（{type(e).__name__}: {e}），"
                  f"后续图像按通用 prompt 解析；"
                  f"如需 CLIP 请安装 torch/transformers 或指定本地模型")
        return []


# ---------------------------------------------------------------------------
# 清单落盘
# ---------------------------------------------------------------------------
def save_inventory(records: list[dict], doc_stats: dict) -> Path:
    """写入 results/image_inventory.json 与可读版 image_inventory.md。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "workorder": "人工智能NLP-RAG-图像内容解析及检索优化",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "min_side": MIN_SIDE,
        "render_dpi": RENDER_DPI,
        "n_images": len(records),
        "n_embedded": sum(1 for r in records if r["source"] == "embedded"),
        "n_page_render": sum(1 for r in records if r["source"] == "page_render"),
        "doc_stats": doc_stats,
        "images": records,
    }
    INVENTORY_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 图像清单（image_inventory.json 可读版）",
        "",
        "工单编号：人工智能NLP-RAG-图像内容解析及检索优化",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 图片总数：{payload['n_images']}"
        f"（内嵌位图 {payload['n_embedded']} / 整页渲染回退 {payload['n_page_render']}）",
        f"- 有效图像最小边长：{MIN_SIDE}px；整页渲染分辨率：{RENDER_DPI}dpi",
        "",
        "| # | 图像ID | 页码 | 来源 | 尺寸 | 类型(CLIP top1) | 题注 |",
        "|---|--------|------|------|------|------------------|------|",
    ]
    for k, r in enumerate(records, 1):
        clip = r.get("clip_labels") or []
        top = f"{clip[0]['label']}({clip[0]['score']})" if clip else "—"
        lines.append(
            f"| {k} | {r['image_id']} | {r['page']} | {r['source']} | "
            f"{r['width']}x{r['height']} | {top} | {r.get('caption') or '—'} |"
        )
    INVENTORY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return INVENTORY_JSON


def load_inventory() -> dict:
    """供其它脚本读取图片清单。"""
    if not INVENTORY_JSON.exists():
        raise FileNotFoundError(
            f"未找到图片清单 {INVENTORY_JSON}，请先运行：python src/image_extractor.py")
    return json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))


def inventory_by_page(doc: str | None = None) -> dict[tuple[str, int], list[dict]]:
    """按 (文档名, 页码) 归组，便于把语义描述回填到 PDF 图像块。"""
    inv = load_inventory()
    out: dict[tuple[str, int], list[dict]] = {}
    for r in inv["images"]:
        if doc and r["doc"] != doc:
            continue
        out.setdefault((r["doc"], r["page"]), []).append(r)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="招股书图像抽取（含矢量图整页渲染回退）")
    ap.add_argument("--docs", default="2",
                    help="要处理的文档编号，逗号分隔：2=招股说明书2（默认），1=招股说明书1")
    ap.add_argument("--pages", default="",
                    help="只处理指定页码，逗号分隔（如 39,72,310）；留空表示全文档")
    ap.add_argument("--min-side", type=int, default=MIN_SIDE, help="有效图像最小边长")
    ap.add_argument("--dpi", type=int, default=RENDER_DPI, help="整页渲染分辨率")
    ap.add_argument("--force-render-pages", action="store_true",
                    help="图页即使有内嵌图也追加一张整页渲染图")
    ap.add_argument("--no-clip", action="store_true", help="跳过 CLIP 分类")
    args = ap.parse_args()

    pages = [int(x) for x in args.pages.split(",") if x.strip()] or None
    doc_keys = [d.strip() for d in args.docs.split(",") if d.strip()]

    all_records: list[dict] = []
    doc_stats: dict[str, dict] = {}
    t0 = time.perf_counter()

    for key in doc_keys:
        if key not in DOCS:
            raise SystemExit(f"未知文档编号 {key}，可选 {list(DOCS)}")
        spec = DOCS[key]
        print(f"\n[抽取] 《{spec['name']}》 {spec['path']}")
        recs = extract_doc_images(
            spec["path"], spec["name"], IMAGES_DIR,
            pages=pages, min_side=args.min_side, dpi=args.dpi,
            force_render_pages=args.force_render_pages,
        )

        if not args.no_clip:
            print(f"[CLIP] 对 {len(recs)} 张图做零样本分类…")
            for r in recs:
                r["clip_labels"] = classify_with_clip(r["abs_file"], top_n=3)
                if r["clip_labels"]:
                    print(f"   p{r['page']:<4} {r['source']:<11} "
                          f"→ {r['clip_labels'][0]['label']} "
                          f"({r['clip_labels'][0]['score']})")
        else:
            for r in recs:
                r["clip_labels"] = []

        all_records.extend(recs)
        doc_stats[spec["name"]] = {
            "pages": len(pages) if pages else fitz.open(spec["path"]).page_count,
            "n_embedded": sum(1 for r in recs if r["source"] == "embedded"),
            "n_page_render": sum(1 for r in recs if r["source"] == "page_render"),
            "n_figure_page": sum(1 for r in recs if r.get("figure_page")),
        }

    # 去重：同一页同一来源重复时只保留一条
    uniq, seen = [], set()
    for r in all_records:
        k = (r["doc"], r["page"], r["source"], r["xref"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    uniq.sort(key=lambda r: (r["doc"], r["page"], r["source"]))
    for r in uniq:                       # abs_file 仅内部使用，不写入清单
        r.pop("abs_file", None)
    # 回填 abs_file 供 CLIP 之后的下游脚本（写盘前加回绝对路径字段）
    for r in uniq:
        r["abs_file"] = str((WORKDIR / r["file"]).resolve())

    path = save_inventory(uniq, doc_stats)
    print(f"\n[完成] 共 {len(uniq)} 张图"
          f"（内嵌 {sum(1 for r in uniq if r['source'] == 'embedded')} / "
          f"整页渲染 {sum(1 for r in uniq if r['source'] == 'page_render')}），"
          f"耗时 {time.perf_counter() - t0:.1f}s")
    print(f"        清单：{path}")
    print(f"        图片目录：{IMAGES_DIR}")


if __name__ == "__main__":
    main()
