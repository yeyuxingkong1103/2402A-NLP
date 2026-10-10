# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
import json
from pathlib import Path

import fitz
import pytest

from rag04.config import get_settings, PROJECT_ROOT
from rag04.schema import FigureBlock
from rag04.ingest.loader import open_pdf
from rag04.ingest.figures import (
    cluster_rects, find_captions, detect_figures, render_figure, expand,
    collect_image_rects, looks_like_caption, has_figure_evidence,
)

GY1 = PROJECT_ROOT / "招股说明书1.pdf"
GY2 = PROJECT_ROOT / "招股说明书2.pdf"
GT_PATH = Path(__file__).parent / "fixtures" / "figure_gt.json"
needs_corpus = pytest.mark.skipif(not GY2.exists(), reason="语料缺失")
needs_gy1 = pytest.mark.skipif(not GY1.exists(), reason="语料缺失")
GT = json.loads(GT_PATH.read_text(encoding="utf-8"))


def _gt_for(doc_id, page):
    for item in GT.get(doc_id, []):
        if item["page"] == page:
            return item
    raise KeyError(f"真值缺失：{doc_id} p{page}")


# ---------- 纯函数单测（不需语料）----------

def test_cluster_rects_merges_adjacent():
    a = fitz.Rect(0, 0, 10, 10)
    b = fitz.Rect(15, 0, 25, 10)      # 间距 5 < gap=12 → 合并
    c = fitz.Rect(500, 500, 510, 510)  # 远离 → 独立
    groups = cluster_rects([a, b, c], gap=12.0)
    assert len(groups) == 2
    merged = max(groups, key=lambda r: r.get_area())
    assert merged.x0 == 0 and merged.x1 == 25


def test_cluster_rects_keeps_distant_separate():
    a = fitz.Rect(0, 0, 10, 10)
    b = fitz.Rect(100, 0, 110, 10)
    assert len(cluster_rects([a, b], gap=5.0)) == 2


def test_expand_grows_rect_and_clamps_to_page():
    page_rect = fitz.Rect(0, 0, 100, 100)
    r = fitz.Rect(10, 10, 20, 20)
    out = expand(r, 0.15, page_rect)
    assert out.x0 < 10 and out.y0 < 10
    assert out.x1 > 20 and out.y1 > 20

    edge = fitz.Rect(0, 0, 10, 10)
    e = expand(edge, 0.15, page_rect)
    assert e.x0 >= 0 and e.y0 >= 0, "不得越出页面边界"


def test_expand_is_at_least_15_percent():
    page_rect = fitz.Rect(0, 0, 1000, 1000)
    r = fitz.Rect(100, 100, 200, 200); w = r.width
    out = expand(r, 0.15, page_rect)
    assert out.width >= w * 1.30  # 两侧各 15%


def test_expand_returns_empty_for_off_page_rect():
    """整段落在页面外的候选不得被夹成反向矩形（get_pixmap 会崩溃）。"""
    page_rect = fitz.Rect(0, 0, 600, 800)
    out = expand(fitz.Rect(700, 10, 800, 20), 0.15, page_rect)
    assert out.is_empty, f"整段页外的候选应返回空矩形，实际={tuple(out)}"
    assert out.width >= 0 and out.height >= 0, "不得返回反向（x0>x1）矩形"


# ---------- 证据闸门严格题注判定（不需语料）----------

def test_caption_predicate_rejects_prose_and_table_header():
    """证据闸门不得把表头/正文/公司名+日期当题注：这五类曾伪造整页图。"""
    assert not looks_like_caption("金额  增长率  金额"), "p11 表头不得成为题注证据"
    assert not looks_like_caption(
        "2016 年至2018 年，公司营业收入的复合增长率为57.82%，净利润复合增长"
    ), "p41 正文句不得成为题注证据"
    assert not looks_like_caption(
        "备产业链较长及大型作战信息化项目规模较大的影响，当军方经总体单位向各装"
    ), "p9/p47 风险提示套话不得成为题注证据"
    assert not looks_like_caption("华创兴图  2012-05-09"), \
        "公司名+日期不得冒充「图N」（空白不得跨过，且「图」前不得是汉字）"
    assert not looks_like_caption("兴图19件"), "「兴图19件」同样不得冒充「图19」"
    # 句末标点与超长必须否决：题注是一行标题，不是句子
    assert not looks_like_caption("网络化视频指挥系统结构示意图。")
    assert not looks_like_caption("示意图" + "长" * 90)


def test_caption_predicate_accepts_true_captions():
    """真题注形状必须继续被承认，含 p72 的 IC 市场图题注。"""
    assert looks_like_caption("2008 年中国IC 市场应用结构与增长(亿元)")
    assert looks_like_caption("图1-1 主要产品收入")
    assert looks_like_caption("图 2：")
    assert looks_like_caption("表1-1 主要财务数据")
    assert looks_like_caption("网络化视频指挥系统结构示意图")
    assert looks_like_caption("兴图新科采购流程图")


# ---------- 真实语料回归 ----------

@needs_corpus
def test_org_chart_short_crop_regression():
    """头号风险：裁切框必须完整覆盖 6 个销售处所在行（否则静默丢答案）。

    组织结构图在 **1-based 第 39 页**（idx 38，印刷页 38）。
    真值下边界 y=645 覆盖最后一行 6 个销售处的渐变条（y≈566~645）。
    """
    doc = open_pdf(GY2)
    s = get_settings()
    figs = detect_figures(doc[38], "招股说明书2", 39, s)
    assert figs, "第 39 页必须检出图区"
    gt = fitz.Rect(*_gt_for("招股说明书2", 39)["must_include"])
    covering = [f for f in figs if fitz.Rect(*f.bbox).contains(gt)]
    assert covering, (
        f"第 39 页检出图区未完整覆盖真值 {tuple(gt)}（6 个销售处会被裁掉）；"
        f"实际检出={[f.bbox for f in figs]}"
    )


@needs_corpus
def test_ic_chart_detected():
    """IC 市场图在 **1-based 第 72 页**（idx 71，印刷页 71）。

    该图是**两张嵌入位图**（638x479 + 525x473），不是矢量绘制——
    只做矢量聚类会完全漏掉它。
    """
    doc = open_pdf(GY2)
    s = get_settings()
    figs = detect_figures(doc[71], "招股说明书2", 72, s)
    gt = fitz.Rect(*_gt_for("招股说明书2", 72)["must_include"])
    assert any(fitz.Rect(*f.bbox).contains(gt) for f in figs), \
        f"第 72 页 IC 市场图未被完整覆盖；实际={[f.bbox for f in figs]}"


@needs_corpus
def test_ic_page_passes_strict_evidence_gate():
    """p72 真题注须通过严格闸门：短 + 无句末标点 + 「应用结构」标题关键词。"""
    doc = open_pdf(GY2)
    assert has_figure_evidence(doc[71]) is True, \
        "IC 市场图页（p72）被严格证据闸门误杀，题注识别过紧"


@needs_corpus
def test_image_info_path_finds_raster_chart():
    """直接验证位图来源这条路：不依赖矢量聚类也能定位 IC 市场图。"""
    doc = open_pdf(GY2)
    s = get_settings()
    rects = collect_image_rects(doc[71])
    merged = fitz.Rect()
    for r in rects:
        merged |= r
    gt = fitz.Rect(*_gt_for("招股说明书2", 72)["must_include"])
    assert rects, "第 72 页应检出真实图表位图（应滤除 143x127 水印）"
    assert merged.contains(gt), f"位图并集 {tuple(merged)} 未覆盖真值 {tuple(gt)}"
    for r in rects:
        assert r.get_area() > 10_000, "水印 logo 未被滤除"


@needs_corpus
def test_watermark_is_filtered_out():
    """143x127 的平铺水印（约 1.8 万像素）不得被当成图表。"""
    doc = open_pdf(GY2)
    rects = collect_image_rects(doc[38])
    assert rects == [], f"组织结构图页只有水印与细条，应全部滤除，实际={rects}"


@needs_corpus
def test_detection_is_not_degenerate_full_page():
    """必须是真正的图区检测，不能一律返回整页。"""
    doc = open_pdf(GY2)
    s = get_settings()
    figs = detect_figures(doc[38], "招股说明书2", 39, s)
    for f in figs:
        r = fitz.Rect(*f.bbox)
        assert r.get_area() < 0.85 * doc[38].rect.get_area(), \
            "检出区域接近整页，说明检测退化"


@needs_corpus
def test_org_chart_uses_caption_anchor_or_cluster():
    doc = open_pdf(GY2)
    s = get_settings()
    figs = detect_figures(doc[38], "招股说明书2", 39, s)
    assert any(f.detect_method in ("caption_anchor", "vector_cluster") for f in figs)
    assert all(0.0 <= f.confidence <= 1.0 for f in figs)


@needs_corpus
def test_captions_found_on_ic_page():
    doc = open_pdf(GY2)
    caps = find_captions(doc[71])
    joined = " ".join(t for _, t in caps)
    assert "IC" in joined or "应用结构" in joined, f"未找到图题注，实际={joined[:200]}"


@needs_corpus
def test_render_figure_writes_png(tmp_path):
    doc = open_pdf(GY2)
    s = get_settings()
    object.__setattr__(s, "fig_cache_dir", tmp_path)
    figs = detect_figures(doc[38], "招股说明书2", 39, s)
    path = render_figure(doc[38], figs[0], s)
    p = Path(path)
    assert p.exists() and p.stat().st_size > 5000
    assert p.suffix == ".png"
    # 硬约束：DPI 200..220、外扩 >= 15%；并据裁剪宽度推算期望像素宽度，
    # 使 DPI 回退（如 210→150）必然失败，而不是靠 >300 的宽松下限。
    assert 200 <= s.render_dpi <= 220, f"渲染 DPI 越界：{s.render_dpi}"
    assert s.crop_margin >= 0.15, f"裁切外扩不足 15%：{s.crop_margin}"
    crop_w = fitz.Rect(*figs[0].bbox).width
    expected_w = crop_w * s.render_dpi / 72.0
    with fitz.open(p) as im:
        assert im[0].rect.width > 300, "渲染分辨率过低，竖排小字将不可读"
        # im[0].rect 是「点」，会随嵌入分辨率归一化；必须取真实像素宽度，
        # 否则 DPI 回退（210→150）时 rect.width 不变、断言依旧通过。
        pix_w = fitz.Pixmap(str(p)).width
        assert abs(pix_w - expected_w) <= 3, (
            f"渲染像素宽度 {pix_w}px 与 {s.render_dpi} DPI 预期 "
            f"{expected_w:.1f}px 不符，DPI 约束可能失效"
        )


@needs_corpus
@pytest.mark.parametrize("bad_bbox", [(0.0, 0.0, 0.0, 0.0),   # 空
                                      (10.0, 10.0, 0.0, 0.0),  # 反向
                                      (5.0, 5.0, 5.0, 5.0),    # 零面积
                                      (0.0, 0.0)],             # 长度不合法
                         ids=["empty", "reversed", "zero-area", "malformed"])
def test_render_figure_rejects_invalid_bbox(tmp_path, bad_bbox):
    """公开 API 收到空/非法 bbox 时必须抛 ValueError，而不是在 get_pixmap 崩溃。"""
    doc = open_pdf(GY2)
    s = get_settings()
    object.__setattr__(s, "fig_cache_dir", tmp_path)
    fig = FigureBlock(
        doc_id="t", page=39, bbox=bad_bbox, image_path="",
        caption="", detect_method="full_page", confidence=0.3,
    )
    with pytest.raises(ValueError):
        render_figure(doc[38], fig, s)


@needs_gy1
@pytest.mark.parametrize("idx", [8, 10, 40, 46, 65],
                         ids=["p9-risk-prose", "p11-table-header",
                              "p41-prose", "p47-risk-prose", "p66-company-date"])
def test_prose_pages_produce_no_full_page_figure(idx):
    """具体回归：这 5 页曾被关键词误判为图题注，伪造出与文本块重复的整页图。

    0-based idx 8/10/40/46/65 = 1-based 第 9/11/41/47/66 页。
    页内均无真实图形（只有水印与装饰细线），两条证据闸门必须同时否决：
    (a) 图形证据为空；(b) 文本只是表头/正文/风险套话/公司名+日期。
    """
    doc = open_pdf(GY1)
    s = get_settings()
    figs = detect_figures(doc[idx], "招股说明书1", idx + 1, s)
    assert not any(f.detect_method == "full_page" for f in figs), (
        f"第 {idx + 1} 页误判出整页图：{[(f.detect_method, f.bbox) for f in figs]}"
    )


@needs_gy1
def test_full_page_fallback_is_rare_across_corpus():
    """整页兜底必须稀有：只允许用于「有图区正向证据但定位失败」的页。

    缺陷背景：表格边框是矢量绘制，但会被 _line_heavy/fig_min_area 滤空，
    旧逻辑随即整页兜底——修复前 招股说明书1 前 120 页约 90% 返回 full_page。
    这等于把整页文本再当图交给 VLM，描述与文本块重复、污染检索并放大成本。
    表格已由 tables.py 单独抽取，这类无证据页必须不产出图区。

    严格证据闸门（图形 + 真题注形状）修复后实测 0/120 = 0.0%。
    1.5x 实测值仍是 0，而 `frac < 0` 不可能成立，故取最小可判定正阈值 1%
    （120 页里 >=2 页回归即失败；1 页 = 0.83% 作为安全余量），远低于 0.10 上限。
    """
    doc = open_pdf(GY1)
    s = get_settings()
    n = min(120, doc.page_count)
    full_pages = sum(
        1 for i in range(n)
        if any(f.detect_method == "full_page"
               for f in detect_figures(doc[i], "招股说明书1", i + 1, s))
    )
    frac = full_pages / n
    assert frac < 0.01, (
        f"整页兜底比例 {frac:.1%}（{full_pages}/{n} 页）过高，检测已退化"
        "（修复后实测 0.0%，阈值 1%）"
    )
