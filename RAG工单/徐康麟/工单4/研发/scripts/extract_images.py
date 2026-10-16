# -*- coding: utf-8 -*-
"""工单4 图像解析总入口（extract_images）。

任务：**人工智能NLP-RAG-图像内容解析及检索优化**。

一条命令跑通「图像内容解析全链路」并落盘全部产物：

    ① 图像资产提取 + 水印/装饰去重（image_parser）
    ② 矢量框图 → 层级结构树（diagram_parser）        ← 工单第 5 题
    ③ 位图图表 → 行业—数值结构化证据（chart_extractor）← 工单第 6 题
    ④ 中文 CLIP 图像语义嵌入 + 相似度检索通道（multimodal_encoder）
    ⑤ 汇总成可检索、可引用的 image_blocks.jsonl（文件名 + 1-based 物理页码 + xref/bbox）

用法（工作目录 = ``E:\\gao6gongdan\\工单4``）::

    # 全量（两份 PDF 共 898 页）
    pwsh -NoProfile -File run_py.ps1 研发/scripts/extract_images.py --all

    # 只跑两张图像题的载体页，并做断言（失败即非零退出）
    pwsh -NoProfile -File run_py.ps1 研发/scripts/extract_images.py \\
        --pdf 招股说明书2.pdf --pages 39,40,72,113 --assert-q5-q6

**证据纪律**：所有数值都由几何 + 正文锚点算出，代码里没有任何"答案常量"；
识别不确定的地方一律写 ``warnings`` 与 ``extraction_method``，不伪造。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---- 让脚本在 run_py.ps1 之外也能直接跑 -------------------------------
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[1]                     # 研发/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import fitz  # PyMuPDF

from app.core import image_parser as ip
from app.core import diagram_parser as dp
from app.core import chart_extractor as ce
from app.core import multimodal_encoder as me

PROCESSED = _REPO / "data" / "processed"
INDEX = _REPO / "data" / "index"
RAW = _REPO / "data" / "raw"
EVIDENCE = PROCESSED / "evidence"

QA5 = {"sale_dept_parent": "销售部", "sub_dept_parent": "大客户销售部"}
Q5_EXPECT = {"销售部": 4, "大客户销售部": 6}
Q6_MAX = "汽车"
Q6_NEG = "IC卡"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def _norm_name(name: str) -> str:
    return "".join(str(name or "").split())


def render_region_png(page: "fitz.Page", rect: Sequence[float], out: Path, zoom: float = 2.0) -> str:
    """把 PDF 上某块区域渲染成 PNG（证据图 / 供 CLIP 编码）。"""

    out.parent.mkdir(parents=True, exist_ok=True)
    clip = fitz.Rect(*[float(v) for v in rect])
    pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(zoom, zoom))
    pix.save(str(out))
    return str(out)


def png_bytes_of_region(page: "fitz.Page", rect: Sequence[float], zoom: float = 2.0) -> bytes:
    clip = fitz.Rect(*[float(v) for v in rect])
    pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(zoom, zoom))
    return pix.tobytes("png")


# ---------------------------------------------------------------------------
# 框图块
# ---------------------------------------------------------------------------
def build_diagram_block(doc: "fitz.Document", corpus: str, page_no: int, res: dp.DiagramResult) -> dict[str, Any]:
    """把 DiagramResult 变成可检索、可引用的图像块。"""

    page = doc[page_no - 1]
    cross = dp.collect_cross_references(doc, res, exclude_page=page_no)
    summ = dp.summarize_hierarchy(res)

    xs = [b.x0 for b in res.boxes] or [0.0]
    ys = [b.y0 for b in res.boxes] or [0.0]
    xe = [b.x1 for b in res.boxes] or [0.0]
    ye = [b.y1 for b in res.boxes] or [0.0]
    bbox = [min(xs), min(ys), max(xe), max(ye)]

    # ---- 「列聚类复原串」：把按 (行, 列) 阅读序排好的节点名**无缝拼接** ----
    # 为什么必须给这一串：页 39 下半部分是竖排文字、每汉字一个 span，
    # 朴素 get_text() 得到的是「销\n售\n部」这种带换行的碎片，
    # 检索端做子串匹配时 '销售部'/'大客户销售部'/'销售处' **全部为 False**（T3 已实测）。
    # 这一串就是「朴素文本提取救不回来」的那部分，必须由本模块落进 chunk。
    ordered = sorted([b for b in res.boxes if b.name], key=lambda b: (round(b.y0, 1), b.cx))
    names = [b.name for b in ordered]
    recovered = "".join(names)
    recovered_joined = "、".join(names)

    lines = ["%s 组织结构图（%s 第%d页）" % (corpus.replace(".pdf", ""), corpus, page_no)]
    lines.append("列聚类复原节点（按 x 列聚类、字间无换行）：%s" % recovered)
    lines.append("节点清单：%s" % recovered_joined)
    lines.extend(summ)
    lines.append("组织层级：")
    lines.append(res.tree_text)
    for c in cross:
        lines.append("交叉印证（第%d页）：%s" % (c["page"], c["quote"]))
    text = "\n".join(lines)

    return {
        "chunk_id": "img:%s:p%d:diagram" % (corpus, page_no),
        "type": "image",
        "file_name": corpus,
        "page": page_no,
        "page_basis": ip.PAGE_BASIS,
        "corpus": corpus,
        "xref": None,
        "bbox": [round(v, 2) for v in bbox],
        "image_sha1": None,
        "modality_sub": "diagram",
        "text": text,
        "content": text,
        "text_len": len(text),
        "extraction_method": "geometry+" + res.evidence,
        "source": "diagram_parser/%s#p%d" % (corpus, page_no),
        "structured": {
            "title": "%s 组织结构图" % corpus.replace(".pdf", ""),
            "nodes": names,
            "recovered_text": recovered,
            "recovered_text_joined": recovered_joined,
            "children_count": res.children_count,
            "edges": res.edges,
            "evidence": res.evidence,
            "n_boxes": res.n_boxes,
            "n_segments": res.n_segments,
            "bbox_by_node": res.node_bbox,
            "summary_sentences": summ,
            "cross_refs": cross,
            "warnings": res.warnings,
        },
    }


# ---------------------------------------------------------------------------
# 图表块
# ---------------------------------------------------------------------------
def build_chart_block(
    corpus: str,
    page_no: int,
    asset: "ip.ImageAsset | None",
    xref: int,
    bbox: Sequence[float],
    res: ce.ChartResult,
    title: str,
    source_note: str,
    unit: str,
) -> dict[str, Any]:
    if res.chart_type == "bar":
        text = ce.format_bar_chart_text(title, source_note, res, unit=unit or "%")
        series = [b.to_dict() for b in res.bars]
        method = res.extraction_method
    else:
        text = "\n".join([
            "%s（%s 3D 饼图，未做角度定量反算）" % (title or "位图图表", corpus),
            source_note or "",
            "图内识别到的行业标签：%s。" % "、".join(res.pie_labels),
        ]).strip()
        series = []
        method = "geometry_only+template_ocr"
    return {
        "chunk_id": "img:%s:p%d:chart:x%d" % (corpus, page_no, xref),
        "type": "image",
        "file_name": corpus,
        "page": page_no,
        "page_basis": ip.PAGE_BASIS,
        "corpus": corpus,
        "xref": xref,
        "bbox": [round(float(v), 2) for v in bbox],
        "image_sha1": getattr(asset, "sha1", None),
        "image_bytes": getattr(asset, "size_bytes", None),
        "modality_sub": "chart",
        "text": text,
        "content": text,
        "text_len": len(text),
        "extraction_method": method,
        "source": "chart_extractor/%s#p%d#x%d" % (corpus, page_no, xref),
        "structured": {
            "chart_type": res.chart_type,
            "title": title,
            "unit": unit or ("%" if res.chart_type == "bar" else ""),
            "source_note": source_note,
            "series": series,
            "extremes": res.extremes,
            "anchor": res.anchor,
            "x_axis": res.x_axis,
            "scale_percent_per_px": res.scale_percent_per_px,
            "label_method": res.label_method,
            "value_method": res.value_method,
            "quantitative": res.quantitative,
            "pie_labels": res.pie_labels,
            "image_size": [res.width, res.height],
            "warnings": res.warnings,
        },
    }


def build_picture_block(
    corpus: str, page_no: int, asset: "ip.ImageAsset", clip_tags: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    parts = ["%s 第%d页 插图" % (corpus, page_no)]
    if asset.caption:
        parts.append("图题：%s" % asset.caption)
    if asset.source_note:
        parts.append(asset.source_note)
    if clip_tags:
        parts.append("CLIP 语义标签（Top-%d）：%s"
                     % (len(clip_tags), "、".join("%s(%.3f)" % (t["text"], t["score"]) for t in clip_tags)))
    else:
        parts.append("（未做 CLIP 语义标注：编码器不可用或该图未编码）")
    text = "\n".join(parts)
    return {
        "chunk_id": "img:%s:p%d:picture:x%d" % (corpus, page_no, asset.xref),
        "type": "image",
        "file_name": corpus,
        "page": page_no,
        "page_basis": ip.PAGE_BASIS,
        "corpus": corpus,
        "xref": asset.xref,
        "bbox": [round(float(v), 2) for v in asset.bbox],
        "image_sha1": asset.sha1,
        "image_bytes": asset.size_bytes,
        "modality_sub": "picture",
        "text": text,
        "content": text,
        "text_len": len(text),
        "extraction_method": "clip_zero_shot" if clip_tags else "geometry_only",
        "source": "image_parser/%s#p%d#x%d" % (corpus, page_no, asset.xref),
        "structured": {"caption": asset.caption, "source_note": asset.source_note,
                       "unit": asset.unit, "clip_tags": clip_tags or [],
                       "width": asset.width, "height": asset.height},
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def process_pdf(
    pdf_path: Path,
    pages: Sequence[int] | None,
    *,
    watermark: ip.WatermarkInfo | None = None,
    do_clip: bool = True,
    clip_limit: int = 400,
    batch_size: int = 8,
) -> dict[str, Any]:
    corpus = pdf_path.name
    log("=" * 78)
    log("[%s] 开始解析（pages=%s）" % (corpus, "全量" if not pages else list(pages)))
    t0 = time.time()

    assets, wm = ip.extract_assets_for_document(pdf_path, pages=pages, watermark=watermark)
    kept = [a for a in assets if a.keep]
    log("  资产提取：%d 条记录（去重过滤后 %d 个内容候选）  水印 xref=%s  耗时 %.1fs"
        % (len(assets), len(kept), wm.blacklist, time.time() - t0))

    doc = fitz.open(str(pdf_path))
    blocks: list[dict[str, Any]] = []
    diagram_pages: list[int] = []
    try:
        sel_pages = sorted({a.page for a in assets}) if pages else list(range(1, doc.page_count + 1))
        if pages:
            sel_pages = sorted(set(list(pages)) | set(sel_pages))

        # ---- ② 框图 ----
        t1 = time.time()
        for pno in sel_pages:
            if pno < 1 or pno > doc.page_count:
                continue
            page = doc[pno - 1]
            ok, nb, ns = dp.looks_like_diagram(page, page.rect)
            if not ok:
                continue
            res = dp.parse_diagram_page(page, corpus, pno)
            if res is None:
                continue
            # 质量门（保护索引）：真正的层级图至少有 2 条父子边 + 4 个有名节点；
            # 表格单元格网格等误检在这一步被挡掉（实测 PDF2 有 36 页过 looks_like_diagram，
            # 其中一部分是复杂表格版式）。
            n_named = len([b for b in res.boxes if b.name])
            if len(res.edges) < 2 or n_named < 4:
                log("  [skip] 第%d页 形似框图但层级证据不足（edges=%d nodes=%d evidence=%s）"
                    % (pno, len(res.edges), n_named, res.evidence))
                continue
            diagram_pages.append(pno)
            blk = build_diagram_block(doc, corpus, pno, res)
            blocks.append(blk)
            render_region_png(page, blk["bbox"], EVIDENCE / ("%s_p%d_diagram.png" % (corpus, pno)))
            log("  框图：第%d页 boxes=%d segs=%d evidence=%s children=%s"
                % (pno, res.n_boxes, res.n_segments, res.evidence,
                   json.dumps(res.children_count, ensure_ascii=False)))
        log("  框图扫描完成：命中 %s（%.1fs）" % (diagram_pages or "无", time.time() - t1))

        # ---- ③ 图表 ----
        t2 = time.time()
        chart_candidates: list[tuple[ip.ImageAsset, Any, ce.ChartResult]] = []
        for a in kept:
            if a.width < 60 or a.height < 60:
                continue
            try:
                img = ce.decode_image(doc, a.xref)
            except Exception as exc:
                log("  [warn] 第%d页 xref=%d 解码失败：%s" % (a.page, a.xref, exc))
                continue
            r = ce.analyze_bar_chart(img, doc[a.page - 1].get_text())
            if r is None:
                r = ce.analyze_pie_chart(img)
            if r is None:
                continue
            chart_candidates.append((a, img, r))
            title, src, unit = a.caption, a.source_note, a.unit
            if r.chart_type == "bar":
                unit = "%"
            blk = build_chart_block(corpus, a.page, a, a.xref, a.bbox, r, title, src, unit or "")
            blocks.append(blk)
            render_region_png(doc[a.page - 1], a.bbox, EVIDENCE / ("%s_p%d_chart_x%d.png" % (corpus, a.page, a.xref)),
                              zoom=2.0)
            log("  图表：第%d页 xref=%d → %s 柱数=%d 极值=%s"
                % (a.page, a.xref, r.chart_type, len(r.bars),
                   json.dumps(r.extremes, ensure_ascii=False)[:140]))
        log("  图表抽取完成：%d 张（%.1fs）" % (len(chart_candidates), time.time() - t2))

        # ---- ④ 其它内容图块（picture）——必须在 CLIP 之前建好，
        #         否则 CLIP 标签找不到可挂载的 picture 块（实测踩过这个顺序坑）
        chart_xrefs = {(c[0].page, c[0].xref) for c in chart_candidates}
        picture_blocks: list[dict[str, Any]] = []
        for a in kept:
            if (a.page, a.xref) in chart_xrefs:
                continue
            if any(b["modality_sub"] == "diagram" and b["page"] == a.page for b in blocks):
                continue
            # picture 是**弱证据**（页眉 banner、签章页大图等）：只收「确实占版面」的图，
            # 免得几百条低信息块挤掉正文证据（T4 侧同权检索会受影响）。
            page_rect = doc[a.page - 1].rect
            if a.bbox_area < 0.01 * float(page_rect.width * page_rect.height):
                continue
            blk = build_picture_block(corpus, a.page, a, None)
            picture_blocks.append(blk)
            blocks.append(blk)

        # ---- ⑤ 中文 CLIP 编码 ----
        t3 = time.time()
        clip_info: dict[str, Any] = {"available": False, "reason": "未启用（--no-clip）"}
        embs = None
        clip_ids: list[dict[str, Any]] = []
        if do_clip:
            enc = me.ClipImageEncoder(batch_size=batch_size)
            clip_info = enc.status.to_dict()
            clip_info["text_encoder_workaround"] = enc.text_encoder_workaround
            if enc.status.available:
                items: list[tuple[dict[str, Any], Any]] = []
                # 位图候选
                for a in kept[:clip_limit]:
                    try:
                        items.append((
                            {"chunk_id": "img:%s:p%d:img:x%d" % (corpus, a.page, a.xref),
                             "corpus": corpus, "page": a.page, "xref": a.xref,
                             "bbox": [round(float(v), 2) for v in a.bbox], "kind": "bitmap"},
                            ce.decode_image(doc, a.xref)))
                    except Exception as exc:
                        clip_info.setdefault("failures", []).append(
                            {"page": a.page, "xref": a.xref, "error": "%s: %s" % (type(exc).__name__, exc)})
                # 框图（无位图）→ 渲染该区域再编码
                for blk in blocks:
                    if blk["modality_sub"] != "diagram":
                        continue
                    try:
                        from PIL import Image as _Image
                        raw = png_bytes_of_region(doc[blk["page"] - 1], blk["bbox"])
                        items.append((
                            {"chunk_id": blk["chunk_id"], "corpus": corpus, "page": blk["page"],
                             "xref": None, "bbox": blk["bbox"], "kind": "diagram-render"},
                            _Image.open(io.BytesIO(raw)).convert("RGB")))
                    except Exception as exc:
                        clip_info.setdefault("failures", []).append(
                            {"page": blk["page"], "error": "%s: %s" % (type(exc).__name__, exc)})
                if items:
                    t4 = time.time()
                    embs = enc.encode_images([it[1] for it in items], batch_size=batch_size)
                    clip_ids = [it[0] for it in items]
                    clip_info["encoded"] = int(embs.shape[0])
                    clip_info["encode_seconds"] = round(time.time() - t4, 2)
                    clip_info["n_failures"] = len(clip_info.get("failures", []))
                    log("  CLIP：编码 %d 张（device=%s, %.2fs）"
                        % (embs.shape[0], enc.status.device, clip_info["encode_seconds"]))
                    # CLIP 语义标签（零样本）用于 picture 块
                    tag_texts = ["组织架构图", "柱状图/条形图", "饼图", "数据表格", "公司印章或签字",
                                 "产品照片", "证书或营业执照", "装饰性图案"]
                    top_tags: list[list[dict[str, Any]]] = []
                    if clip_ids:
                        sims = enc.encode_texts(tag_texts) @ embs.T   # (T, N)
                        for j in range(embs.shape[0]):
                            col = sims[:, j]
                            order = col.argsort()[::-1][:3]
                            top_tags.append([{"text": tag_texts[int(i)], "score": float(col[int(i)])}
                                             for i in order])
                    # picture 块的 clip_tags 需要同步重写 text（标签要进可检索正文）
                    for j, blk in enumerate(picture_blocks):
                        if j >= len(top_tags):
                            continue
                        blk["structured"]["clip_tags"] = top_tags[j]
                        blk["extraction_method"] = "clip_zero_shot"
                        parts = ["%s 第%d页 插图" % (corpus, blk["page"])]
                        if blk["structured"].get("caption"):
                            parts.append("图题：%s" % blk["structured"]["caption"])
                        if blk["structured"].get("source_note"):
                            parts.append(blk["structured"]["source_note"])
                        parts.append("CLIP 语义标签（Top-%d）：%s" % (
                            len(top_tags[j]),
                            "、".join("%s(%.3f)" % (t["text"], t["score"]) for t in top_tags[j])))
                        blk["text"] = "\n".join(parts)
                        blk["content"] = blk["text"]
                        blk["text_len"] = len(blk["text"])
            else:
                log("  CLIP 不可用：%s" % enc.status.reason)
                clip_info["reason"] = enc.status.reason
            # 文本塔是否走了「CLS 代替 pooler」的上游 bug 规避（编码后才确定）
            clip_info["text_encoder_workaround"] = bool(
                clip_info.get("text_encoder_workaround") or enc.text_encoder_workaround)

        # ---- 落盘（CLIP 嵌入不在这里写：多份 PDF 的嵌入要在 main 里**合并后一次落盘**，
        #          否则后一份语料会覆盖前一份，索引里就只剩最后一份 PDF 的图）----
        blocks.sort(key=lambda b: (b["page"], b["modality_sub"], str(b.get("xref"))))
        return {
            "corpus": corpus,
            "assets": assets,
            "watermark": wm,
            "kept": kept,
            "blocks": blocks,
            "clip": clip_info,
            "clip_embs": embs,
            "clip_ids": clip_ids,
            "diagram_pages": diagram_pages,
            "chart_count": len(chart_candidates),
            "seconds": round(time.time() - t0, 2),
        }
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# 断言（工单第 5、6 题）
# ---------------------------------------------------------------------------
def assert_q5_q6(blocks: Sequence[dict[str, Any]], corpus: str) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True

    # T3 golden 冻结的「必须连续出现」复原串（页39 竖排文字列聚类结果）
    Q5_REQUIRED_CONTIGUOUS = (
        "电话及网络销售部渠道销售部大客户销售部国际贸易部"
        "珠海销售处深圳销售处北京销售处武汉销售处广州销售处成都销售处"
    )
    Q5_FORBIDDEN = ["上海销售处", "7个销售处", "七个销售处", "7家销售处", "七家销售处",
                    "5个部门", "五个部门", "3个部门", "三个部门"]
    Q5_KEYWORDS = ["电话及网络销售部", "渠道销售部", "大客户销售部", "国际贸易部",
                   "北京", "广州", "成都", "深圳", "武汉", "珠海"]

    dg = [b for b in blocks if b["modality_sub"] == "diagram" and b["page"] == 39 and b["corpus"] == corpus]
    if not dg:
        ok = False
        lines.append("[FAIL] 第5题：未在 %s 第39页 解析出框图块" % corpus)
    else:
        blk = dg[0]
        st = blk["structured"]
        content = blk.get("content") or blk.get("text") or ""

        # ---- (a) 列聚类复原串必须落进 chunk（这是 T3 golden 的 retrieval_evidence）----
        if Q5_REQUIRED_CONTIGUOUS in content:
            lines.append("[OK  ] 第5题：列聚类复原串已落进 chunk（连续、字间无换行）✓")
        else:
            ok = False
            lines.append("[FAIL] 第5题：chunk 里找不到连续复原串：%s" % Q5_REQUIRED_CONTIGUOUS)
        miss = [k for k in Q5_KEYWORDS if k not in content]
        if miss:
            ok = False
            lines.append("[FAIL] 第5题：answer_keywords 缺失：%s" % miss)
        else:
            lines.append("[OK  ] 第5题：answer_keywords 全部命中（%d/%d）" % (len(Q5_KEYWORDS), len(Q5_KEYWORDS)))
        bad = [w for w in Q5_FORBIDDEN if w in content]
        if bad:
            ok = False
            lines.append("[FAIL] 第5题：chunk 命中 forbidden_substrings：%s" % bad)
        else:
            lines.append("[OK  ] 第5题：未命中任何 forbidden_substring ✓")
        # ---- 结论句必须在同一 chunk，供 T5 直接引用 ----
        for need in ("销售部 由 4 个部门构成", "大客户销售部 由 6 个销售处构成"):
            if need in content:
                lines.append("[OK  ] 第5题：结论句已在同 chunk：%s" % need)
            else:
                ok = False
                lines.append("[FAIL] 第5题：缺少结论句：%s" % need)

        cc = {_norm_name(k): v for k, v in (st.get("children_count") or {}).items()}
        for name, want in Q5_EXPECT.items():
            got = cc.get(name)
            mark = "OK  " if got == want else "FAIL"
            if got != want:
                ok = False
            lines.append("[%s] 第5题：%s 的下级数 = %s（期望 %d）" % (mark, name, got, want))
        kids = []
        for e in st.get("edges") or []:
            if _norm_name(e[0]) == QA5["sale_dept_parent"]:
                kids.append(_norm_name(e[1]))
        if "上海分公司" in kids:
            ok = False
            lines.append("[FAIL] 第5题：上海分公司 被错当成 销售部 的下级")
        else:
            lines.append("[OK  ] 第5题：上海分公司 未被计入销售部下级 ✓")
        cr = st.get("cross_refs") or []
        if not cr:
            ok = False
            lines.append("[FAIL] 第5题：未找到正文交叉印证（页40/页113）")
        else:
            pages = sorted({c["page"] for c in cr})
            lines.append("[OK  ] 第5题：正文交叉印证 %d 条，覆盖物理页 %s" % (len(cr), pages))
            for c in cr[:4]:
                lines.append("        第%d页：%s" % (c["page"], c["quote"][:70]))

    ch = [b for b in blocks if b["modality_sub"] == "chart" and b["page"] == 72 and b["corpus"] == corpus]
    if not ch:
        ok = False
        lines.append("[FAIL] 第6题：未在 %s 第72页 解析出图表块" % corpus)
    else:
        bar = None
        for b in ch:
            if (b["structured"] or {}).get("chart_type") == "bar":
                bar = b
        if bar is None:
            ok = False
            lines.append("[FAIL] 第6题：第72页未识别出条形图")
        else:
            st = bar["structured"]
            content = bar.get("content") or bar.get("text") or ""
            series = st.get("series") or []
            mark = "OK  " if len(series) == 7 else "FAIL"
            if len(series) != 7:
                ok = False
            lines.append("[%s] 第6题：抽出 %d 组「行业—增长率」" % (mark, len(series)))
            for s in series:
                lines.append("        %-6s %6s%%  dir=%s  printed=%-8s 独立复核=%s  (%s)"
                             % (s.get("label"), s.get("value"), s.get("direction"),
                                s.get("printed_value"), s.get("ocr_verified"), s.get("value_source")))
            # ---- T3 golden 要求：汽车/14.0 与 IC卡/-2.0 必须落在**同一 chunk 的文本**里 ----
            for lbl, val in (("汽车", "14.0"), ("IC卡", "-2.0")):
                if lbl in content and val in content:
                    lines.append("[OK  ] 第6题：chunk 文本同时含「%s」与「%s」✓" % (lbl, val))
                else:
                    ok = False
                    lines.append("[FAIL] 第6题：chunk 文本缺「%s」或「%s」（页72 文本层没有 14.0/-2.0，"
                                 "必须由位图抽取写进同一 chunk）" % (lbl, val))
            ex = st.get("extremes") or {}
            gmax, gneg = ex.get("max_label"), list(ex.get("negatives") or [])
            m1 = "OK  " if _norm_name(gmax) == Q6_MAX else "FAIL"
            if _norm_name(gmax) != Q6_MAX:
                ok = False
            lines.append("[%s] 第6题：增长率最快 = %s（期望 %s，值 %s）"
                         % (m1, gmax, Q6_MAX, ex.get("max_value")))
            m2 = "OK  " if (_norm_name(gneg[0]) if gneg else "") == Q6_NEG else "FAIL"
            if (_norm_name(gneg[0]) if gneg else "") != Q6_NEG:
                ok = False
            lines.append("[%s] 第6题：负增长 = %s（期望 %s，值 %s）"
                         % (m2, gneg, Q6_NEG, ex.get("negative_values")))
            if ex.get("top_hint_matches_max") is False:
                ok = False
                lines.append("[FAIL] 第6题：正文提示榜首「%s」与几何最大柱「%s」不一致" % (ex.get("top_hint"), gmax))
            else:
                lines.append("[OK  ] 第6题：正文「仅次于%s」与几何最大柱一致 ✓" % ex.get("top_hint"))
    return ok, lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="工单4 图像解析全链路（资产/框图/图表/CLIP/索引块）")
    ap.add_argument("--pdf", default=None,
                    help="只处理指定文件名（默认处理 研发/data/raw 下全部 PDF）；多个用逗号分隔")
    ap.add_argument("--pages", default=None,
                    help="只处理这些 1-based 物理页码（逗号分隔）；水印判据仍按全量页频次计算")
    ap.add_argument("--all", action="store_true", help="全量（默认行为，显式写出便于文档引用）")
    ap.add_argument("--assert-q5-q6", action="store_true", dest="assert_q5_q6",
                    help="对工单第5、6题做断言；失败返回非零退出码")
    ap.add_argument("--assert-only", action="store_true", dest="assert_only",
                    help="不重新解析，直接读 研发/data/processed/image_blocks.jsonl 做断言"
                         "（用于校验 --all 全量产物，避免为了断言把全量产物覆盖成 4 页版本）")
    ap.add_argument("--diag-metrics", default=None,
                    help="**标定用**：打印这些 1-based 物理页的「表格网格 vs 框图」判别指标"
                         "（boxes/segs/adjRatio/medNameLen/fill），用于复核 diagram_parser 的 0.025 阈值")
    ap.add_argument("--no-clip", action="store_true", help="跳过中文 CLIP 编码")
    ap.add_argument("--clip-limit", type=int, default=400, help="单份 PDF 最多编码多少张位图（默认 400）")
    ap.add_argument("--batch-size", type=int, default=8, help="CLIP 批大小（CPU 建议 8，GPU 可 16~32）")
    args = ap.parse_args(argv)

    # ---- --assert-only：直接对已落盘的全量产物做断言（不重新解析、不覆盖产物）----
    if args.assert_only:
        corpus = "招股说明书2.pdf"
        blocks = ip.read_jsonl(PROCESSED / "image_blocks.jsonl")
        log("[断言] 读取已落盘产物：%s（%d 块）" % (PROCESSED / "image_blocks.jsonl", len(blocks)))
        if not any(b.get("corpus") == corpus for b in blocks):
            log("[FAIL] 产物里没有 %s 的图像块" % corpus)
            return 3
        ok, lines = assert_q5_q6(blocks, corpus)
        for line in lines:
            log(line)
        log("[断言] 总体：%s" % ("全部通过" if ok else "存在失败项"))
        return 0 if ok else 2

    if args.diag_metrics:
        # 「表格网格 vs 框图」判别指标标定（diagram_parser._looks_like_table_grid 的阈值依据）
        want = sorted({int(x) for x in str(args.diag_metrics).replace("，", ",").split(",") if x.strip()})
        _names = ([x.strip() for x in str(args.pdf).replace("，", ",").split(",") if x.strip()]
                  if args.pdf else None)
        _pdfs = sorted(p for p in RAW.glob("*.pdf") if (not _names or p.name in _names))
        targets = _pdfs
        if not targets:
            log("[diag-metrics] 未找到 PDF")
            return 127
        for pdf in targets:
            doc = fitz.open(str(pdf))
            try:
                log("[diag-metrics] %s" % pdf.name)
                log("  %-6s %-6s %-6s %-9s %-11s %-8s %-8s %s"
                    % ("page", "boxes", "segs", "adjRatio", "medNameLen", "fill", "minGap", "evidence"))
                for pno in want:
                    if pno < 1 or pno > doc.page_count:
                        continue
                    page = doc[pno - 1]
                    boxes, segs = dp.extract_boxes_and_segments(page, page.rect)
                    if not boxes:
                        log("  %-6d （无矩形）" % pno)
                        continue
                    bs = boxes[:200]
                    n = len(bs)
                    adj = tot = 0
                    mingap = 1e9
                    for i in range(n):
                        a = bs[i]
                        for j in range(i + 1, n):
                            b = bs[j]
                            tot += 1
                            ovl_x = min(a.x1, b.x1) - max(a.x0, b.x0)
                            ovl_y = min(a.y1, b.y1) - max(a.y0, b.y0)
                            gx = max(a.x0, b.x0) - min(a.x1, b.x1)
                            gy = max(a.y0, b.y0) - min(a.y1, b.y1)
                            if (ovl_y > 1.0 and -2.5 <= gx <= 2.5) or (ovl_x > 1.0 and -2.5 <= gy <= 2.5):
                                adj += 1
                            d = max(gx, gy)
                            if 0 <= d < mingap:
                                mingap = d
                    pa = float(page.rect.width * page.rect.height)
                    fill = sum((b.x1 - b.x0) * (b.y1 - b.y0) for b in bs) / pa
                    res = dp.parse_diagram_page(page, pdf.name, pno)
                    nms = [b.name for b in res.boxes if b.name] if res else []
                    ln = [len(x) for x in nms] or [0]
                    log("  %-6d %-6d %-6d %-9.3f %-11.1f %-8.3f %-8.1f %s"
                        % (pno, n, len(segs), adj / max(1, tot), sum(ln) / len(ln), fill,
                           (mingap if mingap < 1e8 else -1), (res.evidence if res else "-")))
            finally:
                doc.close()
        return 0

    pages: list[int] | None = None
    if args.pages:
        pages = sorted({int(x) for x in str(args.pages).replace("，", ",").split(",") if x.strip()})

    names: list[str] | None = None
    if args.pdf:
        names = [x.strip() for x in str(args.pdf).replace("，", ",").split(",") if x.strip()]

    pdfs = sorted(p for p in RAW.glob("*.pdf"))
    if names:
        pdfs = [p for p in pdfs if p.name in names]
    if not pdfs:
        log("[extract_images] 未找到 PDF：%s（--pdf=%s）" % (RAW, names))
        return 127

    PROCESSED.mkdir(parents=True, exist_ok=True)
    INDEX.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    all_assets: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    all_blocks: list[dict[str, Any]] = []
    all_diagrams: list[dict[str, Any]] = []
    all_charts: list[dict[str, Any]] = []
    wm_records: list[dict[str, Any]] = []
    per_pdf: list[dict[str, Any]] = []
    wm: ip.WatermarkInfo | None = None
    clip_embs_all: list[Any] = []
    clip_ids_all: list[dict[str, Any]] = []
    clip_manifests: list[dict[str, Any]] = []

    for pdf in pdfs:
        res = process_pdf(pdf, pages, watermark=None, do_clip=not args.no_clip,
                          clip_limit=args.clip_limit, batch_size=args.batch_size)
        wm_records.append(res["watermark"].to_dict())
        if wm is None:
            wm = res["watermark"]
        all_assets.extend(ip.asset_to_record(a) for a in res["assets"])
        all_candidates.extend(ip.asset_to_record(a) for a in res["kept"])
        all_blocks.extend(res["blocks"])
        all_diagrams.extend(b for b in res["blocks"] if b["modality_sub"] == "diagram")
        all_charts.extend(b for b in res["blocks"] if b["modality_sub"] == "chart")
        if res.get("clip_embs") is not None and getattr(res["clip_embs"], "size", 0):
            clip_embs_all.append(res["clip_embs"])
            clip_ids_all.extend(res["clip_ids"])
        if (res["clip"] or {}).get("available"):
            clip_manifests.append({"corpus": res["corpus"], **res["clip"]})
        per_pdf.append({"corpus": res["corpus"], "assets": len(res["assets"]),
                        "candidates": len(res["kept"]), "blocks": len(res["blocks"]),
                        "diagrams": len(res["diagram_pages"]), "charts": res["chart_count"],
                        "clip": res["clip"], "seconds": res["seconds"]})

    # ---- CLIP 嵌入：**多份语料合并后一次落盘**（否则后一份会覆盖前一份）----
    clip_saved: dict[str, str] = {}
    if clip_embs_all:
        import numpy as _np
        merged = _np.concatenate(clip_embs_all, axis=0)
        manifest = {
            "available": True,
            "model_name": me.MODEL_NAME,
            "model_dir": str(me.resolve_model_dir()),
            "device": (clip_manifests[0].get("device") if clip_manifests else ""),
            "n_total": int(merged.shape[0]),
            "corpora": clip_manifests,
            "text_encoder_workaround": any(m.get("text_encoder_workaround") for m in clip_manifests),
            "note": "每个 id 的 bbox/xref/page 见 images_clip_ids.json；相似度只用于排名融合（RRF），禁止与 bge-m3 分数相加",
        }
        clip_saved = me.save_embeddings(INDEX, merged, clip_ids_all, manifest)
        log("[extract_images] CLIP 嵌入落盘：%s（n=%d, dim=%d, device=%s）"
            % (clip_saved["npy"], merged.shape[0], merged.shape[1],
               manifest.get("device")))
    elif not args.no_clip:
        log("[extract_images] 未产出 CLIP 嵌入（编码器不可用或无候选图）")

    n1 = ip.write_jsonl(PROCESSED / "images.jsonl", all_assets)
    n2 = ip.write_jsonl(PROCESSED / "image_candidates.jsonl", all_candidates)
    n3 = ip.write_jsonl(PROCESSED / "image_blocks.jsonl", all_blocks)
    n4 = ip.write_jsonl(PROCESSED / "diagrams.jsonl", all_diagrams)
    n5 = ip.write_jsonl(PROCESSED / "charts.jsonl", all_charts)
    (PROCESSED / "watermark_blacklist.json").write_text(
        json.dumps(wm_records, ensure_ascii=False, indent=2), encoding="utf-8")

    log("=" * 78)
    log("[extract_images] 落盘：images.jsonl=%d  image_candidates.jsonl=%d  image_blocks.jsonl=%d  "
        "diagrams.jsonl=%d  charts.jsonl=%d" % (n1, n2, n3, n4, n5))
    log("[extract_images] 目录：%s" % PROCESSED)
    for p in per_pdf:
        log("  %s：资产 %d / 候选 %d / 块 %d / 框图 %d / 图表 %d，CLIP=%s"
            % (p["corpus"], p["assets"], p["candidates"], p["blocks"], p["diagrams"], p["charts"],
               p["clip"].get("encoded", "不可用")))

    rc = 0
    if args.assert_q5_q6:
        log("-" * 78)
        log("[断言] 工单第 5 / 6 题（载体：招股说明书2.pdf 物理页 39 / 72）")
        corpus = "招股说明书2.pdf"
        if not any(b["corpus"] == corpus for b in all_blocks):
            log("[FAIL] 本次运行没有解析 招股说明书2.pdf 的相关页（--pages 是否包含 39,72？）")
            return 3
        ok, lines = assert_q5_q6(all_blocks, corpus)
        for line in lines:
            log(line)
        log("[断言] 总体：%s" % ("全部通过" if ok else "存在失败项"))
        rc = 0 if ok else 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
