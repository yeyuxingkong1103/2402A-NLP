"""
图像区域构建脚本（工单4 步骤 1/2）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

做三件事：
    1. 复用**工单3 的表格解析**，拿到每页的表格 bbox（缓存到 data/processed/tables_*.json）；
       这些 bbox 用来把表格线框从「图元」里排除 —— 否则带边框的表格会被检成图片。
    2. 逐页检测图区域（src/image_detector.py），裁剪成 PNG 落盘；
    3. 产出图清单 data/images/figures.json，并顺手算出**每页要从正文里剔除的行**
       （图内散字 + 图题），缓存给解析阶段用。

用法：
    python scripts/build_images.py                  # 两份文档
    python scripts/build_images.py --only liyuan    # 只做一份
    python scripts/build_images.py --dry-run        # 只统计不落盘
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pypdf import PdfReader  # noqa: E402

from src import config  # noqa: E402
from src.diagram_topology import extract_topology, topology_usable  # noqa: E402
from src.image_detector import (  # noqa: E402
    FigureRegion,
    detect_page_figures,
    crop_figure,
    drop_lines_for_page,
)

PARSED_TABLES_DIR = config.PROCESSED_DIR


def load_or_build_tables(doc: dict, use_cache: bool = True) -> dict[int, list[tuple]]:
    """取某文档每页的表格 bbox。优先读缓存，避免每次重跑 15~50s 的 find_tables。"""
    cache = PARSED_TABLES_DIR / f"tables_{doc['key']}.json"
    if use_cache and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return {int(k): [tuple(v) for v in vs] for k, vs in data.items()}

    from src.table_parser import parse_pdf_tables

    pdf_path = config.RAW_DIR / doc["file"]
    t0 = time.time()
    # 注意：want_text_outside 默认 False 时**只返回表格列表**，
    # 传 True 才会多返回一份"剔掉表格区域后的正文"——别按两个值去解包。
    tables = parse_pdf_tables(pdf_path, doc=doc["name"])
    boxes: dict[int, list[tuple]] = {}
    for t in tables:
        if t.bbox and any(t.bbox):
            boxes.setdefault(t.page, []).append(tuple(t.bbox))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({str(k): v for k, v in boxes.items()}, ensure_ascii=False),
                     encoding="utf-8")
    print(f"    [表格] {doc['key']}: {len(tables)} 张表 / {len(boxes)} 页有表，"
          f"耗时 {time.time() - t0:.1f}s（已缓存）")
    return boxes


def build_doc(doc: dict, dry_run: bool = False) -> tuple[list[dict], dict[str, list[str]]]:
    """处理一份文档，返回（图清单, {页码: 待剔除正文行}）。"""
    import pymupdf

    pdf_path = config.RAW_DIR / doc["file"]
    table_boxes = load_or_build_tables(doc)
    doc_img_dir = config.IMAGE_DIR / doc["key"]
    doc_img_dir.mkdir(parents=True, exist_ok=True)

    pym_doc = pymupdf.open(str(pdf_path))
    pypdf_pages = PdfReader(str(pdf_path)).pages

    figures: list[dict] = []
    drops: dict[str, list[str]] = {}
    t0 = time.time()

    for i, page in enumerate(pym_doc, 1):
        if not page.get_images(full=True):
            continue
        pf = detect_page_figures(page, i, gap=config.IMAGE_GAP_PT,
                                 min_w=config.IMAGE_MIN_W, min_h=config.IMAGE_MIN_H,
                                 table_boxes=table_boxes.get(i))
        if not pf.figures:
            continue

        # 正文剔除：图内散字 + 图题
        try:
            raw = pypdf_pages[i - 1].extract_text() or ""
        except Exception:  # noqa: BLE001 单页失败不中断
            raw = ""
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        to_drop = sorted(drop_lines_for_page(pf, lines))
        if to_drop:
            drops[str(i)] = to_drop

        for fig in pf.figures:
            if not dry_run:
                out = doc_img_dir / f"p{i}_f{fig.index}.png"
                crop_figure(page, fig, out, dpi=config.IMAGE_CROP_DPI)

            # 几何拓扑：只对矢量/混合图划算（纯位图整张流程图没有可用的矢量连接线，
            # 实测 p110/p113 的流程图是**一整张位图**，拓扑抽不出来，
            # 这时 topology_usable=False，描述里会回退到多模态模型给的层级）。
            topo = None
            if fig.kind in ("vector", "mixed"):
                try:
                    tp = extract_topology(page, fig.bbox, table_boxes.get(i))
                    topo = {
                        "usable": topology_usable(tp),
                        "render": tp.render(),
                        "n_nodes": len(tp.nodes),
                        "n_edges": len(tp.edges),
                        "nodes": [{"label": n.label, "bbox": [round(v, 1) for v in n.bbox]}
                                  for n in tp.nodes],
                    }
                except Exception as exc:  # noqa: BLE001 拓扑只是增强，失败不能中断构建
                    topo = {"usable": False, "render": "", "error": f"{type(exc).__name__}: {exc}"}

            figures.append({
                "doc_key": doc["key"],
                "doc_name": doc["name"],
                "page": i,
                "index": fig.index,
                "bbox": list(fig.bbox),
                "kind": fig.kind,
                "n_elems": fig.n_elems,
                "caption": fig.caption,
                "image": str(Path(fig.raster_path).relative_to(config.ROOT_DIR))
                if fig.raster_path else "",
                "px_w": fig.px_w,
                "px_h": fig.px_h,
                "topology": topo,
            })

    pym_doc.close()
    print(f"    [图像] {doc['key']}: {len(figures)} 个图区域，"
          f"{len(drops)} 页有正文剔除，耗时 {time.time() - t0:.1f}s")
    return figures, drops


def main() -> int:
    ap = argparse.ArgumentParser(description="检测并裁剪 PDF 图像区域（工单4）")
    ap.add_argument("--only", default="", help="只处理某个 doc_key（liyuan/xingtu）")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不落盘")
    ap.add_argument("--no-cache", action="store_true", help="忽略表格 bbox 缓存")
    args = ap.parse_args()

    config.ensure_dirs()
    if args.no_cache:
        for doc in config.DOCS:
            (PARSED_TABLES_DIR / f"tables_{doc['key']}.json").unlink(missing_ok=True)

    all_figs: list[dict] = []
    all_drops: dict[str, dict] = {}
    t0 = time.time()

    for doc in config.DOCS:
        if args.only and doc["key"] != args.only:
            continue
        print(f"  → {doc['key']}  {doc['name']}")
        figs, drops = build_doc(doc, dry_run=args.dry_run)
        all_figs.extend(figs)
        all_drops[doc["key"]] = drops

    if not args.dry_run:
        # **合并**上一次的解析结果，而不是覆盖。
        # 多模态解析 68 张图要 8 分钟且要花钱，重跑几何检测（几秒）时不该把它冲掉。
        # 键用 (doc_key, page, index)：图的位置和序号稳定，换了就是真的换了图。
        prev = {}
        if config.FIGURES_JSON.exists():
            try:
                for old in json.loads(config.FIGURES_JSON.read_text(encoding="utf-8")):
                    prev[(old.get("doc_key"), old.get("page"), old.get("index"))] = old
            except Exception:  # noqa: BLE001 旧清单坏了不该拦住重建
                prev = {}
        carried = 0
        for f in all_figs:
            old = prev.get((f["doc_key"], f["page"], f["index"]))
            if not old:
                continue
            for k in ("desc", "semantics", "elapsed_ms", "error", "clip_index"):
                if k in old:
                    f[k] = old[k]
                    if k == "desc" and old[k]:
                        carried += 1
        if carried:
            print(f"  [合并] 沿用上一次的 {carried} 张图语义描述（未重调多模态模型）")

        config.FIGURES_JSON.write_text(
            json.dumps(all_figs, ensure_ascii=False, indent=1), encoding="utf-8")
        for key, drops in all_drops.items():
            (config.PROCESSED_DIR / f"figure_drop_{key}.json").write_text(
                json.dumps(drops, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n合计 {len(all_figs)} 个图区域，耗时 {time.time() - t0:.1f}s")
    if not args.dry_run:
        print(f"清单 → {config.FIGURES_JSON.relative_to(config.ROOT_DIR)}")

    # 概览：按图题有无、按页面分布
    no_cap = [f for f in all_figs if not f["caption"]]
    print(f"  有图题 {len(all_figs) - len(no_cap)} / 无图题 {len(no_cap)}")
    by_doc: dict[str, int] = {}
    for f in all_figs:
        by_doc[f["doc_key"]] = by_doc.get(f["doc_key"], 0) + 1
    print("  按文档:", by_doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
