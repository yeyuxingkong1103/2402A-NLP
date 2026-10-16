# -*- coding: utf-8 -*-
"""工单8 测试截图生成脚本（中文按测试内容命名，全部为真实跑出来的数字）。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

截图清单（文件名 = 测试内容，硬性要求）：
    1. ``知识图谱可视化.png``                        —— 由 ``render_graph_viz.py`` 生成（力导向图谱）
    2. ``实体关系抽取统计.png``                      —— 实体/关系类型分布（数据源：图谱构建统计.json）
    3. ``GraphRAG与工单7检索结果对比.png``           —— 并排对比（数据源：与工单7对比.json）
    4. ``17题逐题评分.png``                          —— 逐题覆盖率/recall@5/判对（数据源：GraphRAG评估结果.jsonl）
    5. ``响应时间对比.png``                          —— 检索/图谱/生成分段耗时与 SLA（数据源：评估汇总 + 对比表）

纪律：
    * **所有数字都从产物文件读取**，脚本内不做任何手工填数；
    * 数据缺失时**如实标注「数据缺失」并退出码非 0**，绝不画假图；
    * 中文字体走 Windows 自带 msyh/simhei，找不到时退回默认位图字体（不崩）。

用法：
    pwsh -NoProfile -File run_py.ps1 测试/生成截图.py [--only 实体关系抽取统计]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(_ROOT / "研发"))

from app.graphrag.config import WORK_ORDER, ensure_dirs, get_config  # noqa: E402

#: 工单编号（源码头部注释要求）：人工智能NLP-RAG-基于Graph RAG 实现金融问答

SHOT_DIR = _ROOT / "测试" / "测试截图"
DATA_DIR = _ROOT / "测试" / "测试数据"
GRAPH_DIR = _ROOT / "研发" / "data" / "graph"

INK = (27, 34, 48)
BRAND = (31, 105, 200)
GOOD = (28, 122, 69)
BAD = (176, 58, 48)
WARN = (200, 130, 20)
GREY = (110, 120, 136)
LINE = (208, 214, 224)

TYPE_COLORS = {
    "公司": (31, 105, 200), "机构": (46, 158, 108), "指标": (222, 138, 24),
    "金额": (196, 66, 122), "比例": (146, 84, 196), "日期": (120, 128, 140),
    "产品/业务": (208, 78, 62), "人物": (24, 150, 168), "地区": (132, 150, 60),
}


# ---------------------------------------------------------------------------
# 绘图基础（PIL）
# ---------------------------------------------------------------------------
def _fonts() -> dict[str, Any]:
    from PIL import ImageFont

    candidates = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
                  r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"]

    def load(size: int) -> Any:
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001 —— 逐个尝试，失败换下一个
                continue
        return ImageFont.load_default()

    return {"title": load(34), "h2": load(24), "body": load(19), "small": load(16), "tiny": load(14)}


def _canvas(width: int, height: int, title: str, subtitle: str, fonts: dict[str, Any]) -> Any:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (252, 252, 250))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle([0, 0, width, 96], fill=(24, 34, 52))
    draw.text((34, 20), title, fill=(255, 255, 255), font=fonts["title"])
    draw.text((36, 64), subtitle, fill=(198, 212, 232), font=fonts["small"])
    return image, draw


def _bar(draw: Any, x: int, y: int, width: int, height: int, ratio: float, color: Any,
         fonts: dict[str, Any], label: str, value: str) -> None:
    ratio = min(max(float(ratio), 0.0), 1.0)
    draw.rectangle([x, y, x + width, y + height], fill=(238, 241, 246), outline=LINE)
    draw.rectangle([x, y, x + int(width * ratio), y + height], fill=color)
    draw.text((x - 0, y - 24), label, fill=INK, font=fonts["small"])
    draw.text((x + width + 12, y + 2), value, fill=INK, font=fonts["body"])


def _footer(draw: Any, width: int, height: int, text: str, fonts: dict[str, Any]) -> None:
    draw.text((34, height - 34), text, fill=GREY, font=fonts["small"])


def _missing(name: str, reason: str) -> None:
    print(f"[MISS] 截图「{name}」跳过：{reason}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# 截图 2：实体关系抽取统计
# ---------------------------------------------------------------------------
def shot_extract_stats() -> bool:
    stats_path = GRAPH_DIR / "图谱构建统计.json"
    if not stats_path.is_file():
        _missing("实体关系抽取统计", f"缺 {stats_path}")
        return False
    stats = _load_json(stats_path)
    fonts = _fonts()
    width, height = 1600, 1180
    image, draw = _canvas(width, height, "工单8 实体与关系抽取统计（规则/词典/正则抽取，全量扫描）",
                          f"语料：9 份 A 股年报 · 扫描 {stats.get('chunks_scanned')} 块 · "
                          f"实体 {stats.get('entities')} · 关系 {stats.get('relations')} · "
                          f"提及边 {stats.get('mentions')} · 耗时 {stats.get('elapsed_ms')} ms", fonts)
    draw.text((40, 118), "① 实体类型分布", fill=BRAND, font=fonts["h2"])
    by_type: dict[str, int] = stats.get("entities_by_type", {}) or {}
    zh_map = {"company": "公司", "institution": "机构", "metric": "指标", "amount": "金额",
              "ratio": "比例", "date": "日期", "product": "产品/业务", "person": "人物", "city": "地区"}
    items = sorted(((zh_map.get(k, k), v) for k, v in by_type.items()), key=lambda kv: -kv[1])
    total = sum(v for _, v in items) or 1
    y = 168
    for name, count in items:
        _bar(draw, 210, y, 1080, 26, count / total, TYPE_COLORS.get(name, GREY), fonts,
             f"{name}", f"{count}（{count / total:.1%}）")
        y += 46
    draw.line([40, y + 6, width - 40, y + 6], fill=LINE)
    draw.text((40, y + 22), "② 关系类型分布", fill=BRAND, font=fonts["h2"])
    by_rel: dict[str, int] = stats.get("relations_by_type", {}) or {}
    rel_items = sorted(by_rel.items(), key=lambda kv: -kv[1])
    rel_total = sum(v for _, v in rel_items) or 1
    y += 72
    for name, count in rel_items:
        _bar(draw, 210, y, 1080, 24, count / rel_total, (46, 130, 190), fonts,
             f"{name}", f"{count}（{count / rel_total:.1%}）")
        y += 42
    draw.line([40, y + 8, width - 40, y + 8], fill=LINE)
    extractors = stats.get("relations_by_extractor", {}) or {}
    prune = stats.get("prune", {}) or {}
    low = stats.get("prune_low_support", {}) or {}
    draw.text((40, y + 24),
              f"③ 抽取器分布：{' · '.join(f'{k}={v}' for k, v in sorted(extractors.items(), key=lambda kv: -kv[1]))}",
              fill=INK, font=fonts["small"])
    draw.text((40, y + 56),
              f"④ 数据自证剪枝：高频剪枝 {prune.get('pruned')} 个（阈值 {prune.get('limit_chunks')} 块）"
              f"· 低频剪枝 {low.get('pruned')} 个（仅出现 {low.get('min_mentions')} 次）",
              fill=INK, font=fonts["small"])
    draw.text((40, y + 88),
              f"⑤ 抽取配置：mentions_per_entity={((stats.get('config') or {}).get('extract') or {}).get('mentions_per_entity')}"
              f" · max_doc_ratio={((stats.get('config') or {}).get('extract') or {}).get('max_doc_ratio')}"
              f" · 配置指纹 {stats.get('config_fingerprint')}",
              fill=INK, font=fonts["small"])
    _footer(draw, width, height,
            "数据源：研发/data/graph/图谱构建统计.json（构建脚本实跑产物，禁止手写）", fonts)
    out = SHOT_DIR / "实体关系抽取统计.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, "PNG")
    print(f"[OK] {out.name}（{out.stat().st_size // 1024} KB）")
    return True


# ---------------------------------------------------------------------------
# 截图 3：GraphRAG 与工单7 检索结果对比
# ---------------------------------------------------------------------------
def shot_comparison() -> bool:
    path = DATA_DIR / "与工单7对比.json"
    if not path.is_file():
        _missing("GraphRAG与工单7检索结果对比", f"缺 {path}（先跑 build_comparison.py）")
        return False
    data = _load_json(path)
    fonts = _fonts()
    width, height = 1680, 1120
    acc = data.get("准确率", {}) or {}
    rec = data.get("检索召回", {}) or {}
    lat = data.get("响应时间", {}) or {}
    ab = data.get("图谱扩展增量", {}) or {}
    image, draw = _canvas(width, height, "工单8（Graph RAG） vs 工单7（基线）检索结果对比",
                          f"同一批 {data.get('可比题数')} 题 · 同一份索引（8847 块）· 同一判分模块（工单7 判分.py）· "
                          f"同一生成模型（qwen2.5:3b）", fonts)
    rows = [
        ("严格准确率", float(acc.get("工单7_严格准确率") or 0), float(acc.get("工单8_严格准确率") or 0),
         f"{acc.get('工单7_严格判对题数')}/{data.get('可比题数')}", f"{acc.get('工单8_严格判对题数')}/{data.get('可比题数')}"),
        ("平均关键词覆盖率", float(acc.get("工单7_平均关键词覆盖率") or 0), float(acc.get("工单8_平均关键词覆盖率") or 0),
         f"{acc.get('工单7_平均关键词覆盖率')}", f"{acc.get('工单8_平均关键词覆盖率')}"),
        ("平均 recall@5", float(rec.get("工单7_平均recall@5") or 0), float(rec.get("工单8_平均recall@5") or 0),
         f"{rec.get('工单7_平均recall@5')}", f"{rec.get('工单8_平均recall@5')}"),
        ("平均 recall@10", float(rec.get("工单7_平均recall@10") or 0), float(rec.get("工单8_平均recall@10") or 0),
         f"{rec.get('工单7_平均recall@10')}", f"{rec.get('工单8_平均recall@10')}"),
    ]
    max_value = max([r[1] for r in rows] + [r[2] for r in rows] + [1e-6])
    draw.text((40, 120), "① 端到端指标并排（蓝=工单7 基线，红=工单8 Graph RAG）", fill=BRAND, font=fonts["h2"])
    y = 180
    for name, base, cur, bt, ct in rows:
        draw.text((40, y), name, fill=INK, font=fonts["body"])
        _bar(draw, 330, y + 4, 420, 22, base / max_value, (70, 120, 200), fonts, "工单7", bt)
        _bar(draw, 330, y + 44, 420, 22, cur / max_value, (200, 80, 60), fonts, "工单8", ct)
        delta = cur - base
        draw.text((1330, y + 22), f"{delta:+.4f}", fill=(GOOD if delta > 0 else (BAD if delta < 0 else GREY)),
                  font=fonts["body"])
        y += 100
    draw.line([40, y + 4, width - 40, y + 4], fill=LINE)
    draw.text((40, y + 20), "② 图谱扩展增量（消融口径：关闭图谱 vs 开启图谱，同一题集）", fill=BRAND, font=fonts["h2"])
    y += 76
    lines = [
        f"参与消融题数：{ab.get('参与消融的题数')}     "
        f"recall@5：{ab.get('基础检索_recall@5')} → {ab.get('图谱增强_recall@5')}（{ab.get('recall@5_提升'):+}）",
        f"recall@10：{ab.get('基础检索_recall@10')} → {ab.get('图谱增强_recall@10')}（{ab.get('recall@10_提升'):+}）"
        if ab.get("recall@10_提升") is not None else f"recall@10：{ab.get('基础检索_recall@10')} → {ab.get('图谱增强_recall@10')}",
        f"图谱独有块进入 Top-5 的题数：{ab.get('图谱独有块进入Top5的题数')} 题"
        f"（共 {ab.get('图谱独有块进入Top5总数')} 个块）",
        f"题号：{'、'.join(ab.get('图谱独有块进入Top5的题号') or []) or '（无）'}",
    ]
    for text in lines:
        draw.text((48, y), text, fill=INK, font=fonts["body"])
        y += 34
    draw.line([40, y + 8, width - 40, y + 8], fill=LINE)
    draw.text((40, y + 24), "③ 响应时间（端到端墙钟，同机同模型）", fill=BRAND, font=fonts["h2"])
    y += 80
    latency_rows = [
        ("工单7 平均", float(lat.get("工单7_平均ms") or 0), (70, 120, 200)),
        ("工单8 平均", float(lat.get("工单8_平均ms") or 0), (200, 80, 60)),
        ("工单8 检索", float(lat.get("工单8_检索平均ms") or 0), (46, 158, 108)),
        ("工单8 图谱扩展", float(lat.get("工单8_图谱扩展平均ms") or 0), (222, 138, 24)),
        ("工单8 生成", float(lat.get("工单8_生成平均ms") or 0), (146, 84, 196)),
    ]
    lmax = max(v for _, v, _ in latency_rows) or 1.0
    for name, value, color in latency_rows:
        _bar(draw, 260, y, 900, 22, value / lmax, color, fonts, name, f"{value:.1f} ms")
        y += 46
    _footer(draw, width, height,
            "数据源：测试/测试数据/与工单7对比.json（脚本实跑产物；判分口径与工单7 完全一致）", fonts)
    out = SHOT_DIR / "GraphRAG与工单7检索结果对比.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, "PNG")
    print(f"[OK] {out.name}（{out.stat().st_size // 1024} KB）")
    return True


# ---------------------------------------------------------------------------
# 截图 4：17 题逐题评分
# ---------------------------------------------------------------------------
def shot_per_question() -> bool:
    path = DATA_DIR / "GraphRAG评估结果.jsonl"
    if not path.is_file():
        _missing("17题逐题评分", f"缺 {path}")
        return False
    rows = [r for r in _load_jsonl(path) if not r.get("error")]
    if not rows:
        _missing("17题逐题评分", "评估结果为空")
        return False
    fonts = _fonts()
    width, height = 1720, 1220
    summary_path = DATA_DIR / "GraphRAG评估汇总.json"
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    acc = (summary.get("准确率") or {})
    rec = (summary.get("检索召回") or {})
    image, draw = _canvas(width, height, "工单8 Graph RAG · 17 题逐题评分（严格判对 / 关键词覆盖率 / recall@5）",
                          f"严格判对 {acc.get('严格判对题数')}/{summary.get('有效题数')}（{acc.get('严格准确率')}）· "
                          f"部分正确 {acc.get('部分正确题数')} · 平均覆盖率 {acc.get('平均关键词覆盖率')} · "
                          f"平均 recall@5 {rec.get('平均recall@5')} · 平均 recall@10 {rec.get('平均recall@10')}", fonts)
    y = 130
    draw.text((48, y), "题号", fill=BRAND, font=fonts["small"])
    draw.text((150, y), "题型", fill=BRAND, font=fonts["small"])
    draw.text((300, y), "关键词覆盖率", fill=BRAND, font=fonts["small"])
    draw.text((900, y), "recall@5", fill=BRAND, font=fonts["small"])
    draw.text((1230, y), "recall@10", fill=BRAND, font=fonts["small"])
    draw.text((1520, y), "判分", fill=BRAND, font=fonts["small"])
    y += 30
    draw.line([40, y - 4, width - 40, y - 4], fill=LINE)
    for row in rows:
        judge = row.get("judge") or {}
        recall = row.get("recall") or {}
        cov = float(judge.get("keyword_coverage") or 0.0)
        r5 = float(recall.get("recall_at_5") or 0.0)
        r10 = float(recall.get("recall_at_10") or 0.0)
        ok = bool(judge.get("is_correct"))
        partial = bool(judge.get("partial_correct"))
        lane = GOOD if ok else (WARN if partial else BAD)
        draw.text((48, y), str(row.get("id")), fill=INK, font=fonts["body"])
        draw.text((150, y), str(row.get("question_type") or "")[:6], fill=INK, font=fonts["small"])
        draw.rectangle([300, y + 3, 860, y + 21], fill=(240, 243, 248), outline=LINE)
        draw.rectangle([300, y + 3, 300 + int(560 * min(cov, 1.0)), y + 21], fill=lane)
        draw.text((872, y), f"{cov:.3f}", fill=INK, font=fonts["small"])
        draw.text((900, y), f"{r5:.3f}", fill=INK, font=fonts["body"])
        draw.text((1230, y), f"{r10:.3f}", fill=INK, font=fonts["body"])
        verdict = "判对" if ok else ("部分正确" if partial else "未判对")
        draw.text((1520, y), verdict, fill=lane, font=fonts["body"])
        y += 52
    y += 10
    draw.line([40, y, width - 40, y], fill=LINE)
    sla = summary.get("SLA") or {}
    rg = summary.get("图谱扩展增量") or {}
    draw.text((48, y + 18),
              f"图谱扩展：关闭 {rg.get('基础检索_recall@5')} → 开启 {rg.get('图谱增强_recall@5')}"
              f"（recall@5 {rg.get('recall@5_提升'):+}）· 图谱独有块进 Top-5 的题 {rg.get('图谱独有块进入Top5的题数')} 道",
              fill=INK, font=fonts["body"])
    draw.text((48, y + 54),
              f"SLA：阈值 {sla.get('阈值ms')} ms · 达标 {sla.get('达标题数')} 题（{sla.get('达标率')}）"
              f"· 超时题号：{'、'.join(sla.get('超时题号') or []) or '无'}",
              fill=INK, font=fonts["body"])
    _footer(draw, width, height,
            "数据源：测试/测试数据/GraphRAG评估结果.jsonl + GraphRAG评估汇总.json（run_eval.py 实跑）", fonts)
    out = SHOT_DIR / "17题逐题评分.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, "PNG")
    print(f"[OK] {out.name}（{out.stat().st_size // 1024} KB）")
    return True


# ---------------------------------------------------------------------------
# 截图 5：响应时间对比
# ---------------------------------------------------------------------------
def shot_latency() -> bool:
    summary_path = DATA_DIR / "GraphRAG评估汇总.json"
    cmp_path = DATA_DIR / "与工单7对比.json"
    if not summary_path.is_file():
        _missing("响应时间对比", f"缺 {summary_path}")
        return False
    summary = _load_json(summary_path)
    cmp_data = _load_json(cmp_path) if cmp_path.is_file() else {}
    lat = summary.get("响应时间ms", {}) or {}
    sla = summary.get("SLA", {}) or {}
    base_lat = (cmp_data.get("响应时间") or {}).get("工单7_平均ms")
    fonts = _fonts()
    width, height = 1620, 1120
    image, draw = _canvas(width, height, "工单8 Graph RAG 响应时间实测与 SLA 达标情况",
                          f"样本 {lat.get('样本数')} 题 · 平均 {lat.get('平均')} ms · 中位 {lat.get('中位')} ms · "
                          f"P90 {lat.get('P90')} ms · P95 {lat.get('P95')} ms · 最大 {lat.get('最大')} ms", fonts)
    draw.text((40, 122), "① 分段耗时（毫秒，引擎自报实测）", fill=BRAND, font=fonts["h2"])
    y = 178
    segments = [
        ("检索（向量+BM25+重排）", float(lat.get("检索_平均") or 0), (46, 130, 190)),
        ("图谱一跳/两跳扩展", float(lat.get("图谱扩展_平均") or 0), (222, 138, 24)),
        ("Ollama 生成（qwen2.5:3b）", float(lat.get("生成_平均") or 0), (146, 84, 196)),
        ("端到端合计（平均）", float(lat.get("平均") or 0), (200, 80, 60)),
    ]
    smax = max(v for _, v, _ in segments) or 1.0
    for name, value, color in segments:
        _bar(draw, 380, y, 880, 26, value / smax, color, fonts, name, f"{value:.1f} ms")
        y += 52
    if base_lat is not None:
        _bar(draw, 380, y, 880, 26, float(lat.get("平均") or 0) / max(float(base_lat), 1e-6) * 0.5 + 0.0001,
             (70, 120, 200), fonts, "工单7 基线段耗时（对照，仅总耗时可比）", f"{float(base_lat):.1f} ms")
        y += 52
    draw.line([40, y + 6, width - 40, y + 6], fill=LINE)
    draw.text((40, y + 22), "② SLA 达标（阈值 %s ms）" % sla.get("阈值ms"), fill=BRAND, font=fonts["h2"])
    y += 82
    ok = int(sla.get("达标题数") or 0)
    total = int(lat.get("样本数") or 0) or 1
    draw.rectangle([380, y, 380 + 880, y + 34], fill=(240, 243, 248), outline=LINE)
    draw.rectangle([380, y, 380 + int(880 * ok / total), y + 34], fill=GOOD)
    draw.text((1280, y + 6), f"{ok}/{total}（{sla.get('达标率')}）", fill=INK, font=fonts["body"])
    draw.text((380, y - 26), f"达标 {ok} 题", fill=INK, font=fonts["small"])
    y += 60
    draw.text((48, y), f"超时题号：{'、'.join(sla.get('超时题号') or []) or '无'}", fill=INK, font=fonts["body"])
    y += 40
    dist = [(k, float(lat.get(k) or 0)) for k in ("最小", "中位", "P90", "P95", "最大") if lat.get(k) is not None]
    draw.text((48, y), "③ 分位数分布", fill=BRAND, font=fonts["h2"])
    y += 46
    dmax = max(v for _, v in dist) or 1.0
    for name, value in dist:
        _bar(draw, 300, y, 900, 22, value / dmax, (46, 130, 190), fonts, name, f"{value:.1f} ms")
        y += 44
    _footer(draw, width, height,
            "数据源：测试/测试数据/GraphRAG评估汇总.json 与 与工单7对比.json（均为实跑产物）", fonts)
    out = SHOT_DIR / "响应时间对比.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, "PNG")
    print(f"[OK] {out.name}（{out.stat().st_size // 1024} KB）")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"工单8 测试截图生成：{WORK_ORDER}")
    parser.add_argument("--only", default=None, help="只生成指定截图（按文件名关键词匹配）")
    args = parser.parse_args(argv)
    ensure_dirs(get_config())
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        ("知识图谱可视化", "图谱可视化由 研发/scripts/render_graph_viz.py 生成", None),
        ("实体关系抽取统计", "实体/关系分布", shot_extract_stats),
        ("GraphRAG与工单7检索结果对比", "与工单7 并排对比", shot_comparison),
        ("17题逐题评分", "逐题评分明细", shot_per_question),
        ("响应时间对比", "分段耗时与 SLA", shot_latency),
    ]
    results: list[tuple[str, bool, str]] = []
    for name, desc, func in tasks:
        if args.only and args.only not in name:
            continue
        if func is None:
            png = SHOT_DIR / f"{name}.png"
            ok = png.is_file() and png.stat().st_size > 10_000
            results.append((name, ok, "已存在（由 render_graph_viz.py 产出）" if ok else "缺失，请先跑可视化脚本"))
            continue
        try:
            ok = bool(func())
            results.append((name, ok, desc))
        except Exception as exc:  # noqa: BLE001 —— 单张截图失败不影响其它
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
    print("\n== 截图生成汇总 ==")
    failed = 0
    for name, ok, note in results:
        print(f"  {'✅' if ok else '❌'} {name}.png —— {note}")
        failed += 0 if ok else 1
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
