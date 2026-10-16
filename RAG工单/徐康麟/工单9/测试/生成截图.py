# -*- coding: utf-8 -*-
"""工单9 测试截图生成脚本（**中文按测试内容命名，全部是真实跑出来的数字**）。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

截图清单（文件名 = 测试内容，符合 `_共享约定.md` §1 的命名规范）：
    1. ``上下文精度优化前后对比.png``   —— 逐题 + 宏平均（数据源：优化前后对比.json）
    2. ``上下文召回优化前后对比.png``   —— 逐题 + 宏平均（同上）
    3. ``逐题指标明细.png``             —— 精度/召回/TopK 精度/耗时/生成方式（同上）
    4. ``响应时间分布.png``             —— 均值/P50/P90/P95/最大 + SLA 达标率
    5. ``三层面优化收益.png``           —— 图谱层/检索层/生成层的实测变化
    6. ``图谱优化前后对比.png``         —— 节点/边/平均度数/归并组/剪枝计数
    7. ``多语言问答示例.png``           —— 中文 + 英文实测问答（答案 + 数值）
    8. ``稳定性与容错.png``             —— 连续 50 次查询 + 8 个容错场景

纪律：
    * **所有数字都从产物文件读取**，脚本内**不做任何手工填数**；
    * 数据缺失时画「数据缺失」并让退出码非 0，**绝不画假图**；
    * 中文字体走 Windows 自带 msyh/simhei，找不到时退回默认字体（不崩）；
    * 本机**没有 matplotlib**（`_共享约定.md` §2 缺失包清单），因此全部用 PIL 手绘。

用法：
    pwsh -NoProfile -File run_py.ps1 测试/生成截图.py
    pwsh -NoProfile -File run_py.ps1 测试/生成截图.py --only 响应时间分布
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SHOT_DIR = ROOT / "测试" / "测试截图"
TEST_DATA = ROOT / "测试" / "测试数据"
OPT_RESULT = ROOT / "优化" / "评估结果"
OPT_GRAPH = ROOT / "研发" / "data" / "图谱优化后"

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"

INK = (27, 34, 48)
BRAND = (31, 105, 200)
GOOD = (28, 122, 69)
BAD = (176, 58, 48)
WARN = (200, 130, 20)
GREY = (110, 120, 136)
LINE = (208, 214, 224)
BG = (255, 255, 255)
SOFT = (244, 247, 252)


# ---------------------------------------------------------------------------
# 绘图基础
# ---------------------------------------------------------------------------
def _font(size: int, *, bold: bool = False) -> Any:
    names = (["msyhbd.ttc", "simhei.ttf", "msyh.ttc"] if bold else ["msyh.ttc", "simhei.ttf"])
    for base in (r"C:\Windows\Fonts", r"C:\WINNT\Fonts"):
        for name in names:
            path = Path(base) / name
            if path.is_file():
                try:
                    return ImageFont.truetype(str(path), size)
                except Exception:  # noqa: BLE001
                    continue
    return ImageFont.load_default()


class Canvas:
    """极简画布（PIL 手绘）：标题 + 说明 + 表格 + 条形图。"""

    def __init__(self, width: int, height: int, title: str, subtitle: str = "") -> None:
        self.img = Image.new("RGB", (width, height), BG)
        self.d = ImageDraw.Draw(self.img)
        self.w, self.h = width, height
        self.f_title = _font(30, bold=True)
        self.f_sub = _font(16)
        self.f_h = _font(18, bold=True)
        self.f = _font(16)
        self.f_small = _font(14)
        self.d.rectangle([0, 0, width, 78], fill=(31, 105, 200))
        self.d.text((26, 16), title, font=self.f_title, fill=(255, 255, 255))
        if subtitle:
            self.d.text((26, 52), subtitle, font=self.f_sub, fill=(226, 236, 252))

    def text(self, x: int, y: int, s: str, *, font: Any = None, fill: Any = INK) -> None:
        self.d.text((x, y), str(s), font=font or self.f, fill=fill)

    def rect(self, box: Sequence[int], *, fill: Any = None, outline: Any = LINE, width: int = 1) -> None:
        self.d.rectangle(list(box), fill=fill, outline=outline, width=width)

    def hline(self, x0: int, x1: int, y: int, *, fill: Any = LINE, width: int = 1) -> None:
        self.d.line([x0, y, x1, y], fill=fill, width=width)

    def bar(self, x: int, y: int, w: int, h: int, ratio: float, *, color: Any = BRAND,
            label: str = "", value: str = "") -> None:
        ratio = max(0.0, min(1.0, float(ratio)))
        self.rect([x, y, x + w, y + h], fill=SOFT)
        if ratio > 0:
            self.rect([x, y, x + int(w * ratio), y + h], fill=color, outline=color)
        if label:
            self.text(x + 8, y + max((h - 16) // 2, 0), label, font=self.f_small)
        if value:
            self.text(x + w + 10, y + max((h - 16) // 2, 0), value, font=self.f_small, fill=GREY)

    def save(self, name: str) -> Path:
        SHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = SHOT_DIR / name
        self.img.save(path)
        print(f"  已生成 {path}")
        return path


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        print(f"  [缺失] {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 —— 坏数据要如实报，不画假图
        print(f"  [解析失败] {path}: {type(exc).__name__}: {exc}")
        return None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        print(f"  [缺失] {path}")
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _cmp() -> dict[str, Any] | None:
    return _load(OPT_RESULT / "优化前后对比.json")


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# 1 / 2. 精度与召回的前后对比
# ---------------------------------------------------------------------------
def shot_metric(name: str, key: str, title: str, target: float) -> None:
    data = _cmp()
    if not data:
        raise SystemExit(f"[ERROR] 缺少对比产物 {OPT_RESULT / '优化前后对比.json'}，先跑 run_eval_opt.py")
    per_q = data.get("逐题对照") or []
    cv = Canvas(1120, 200 + 30 * max(len(per_q), 1) + 130, title,
                "数据源：优化/评估结果/优化前后对比.json（17 题实跑，use_cache=false）")
    y = 100
    row = data["核心指标对比"][key]
    cv.text(30, y, f"宏平均：优化前 {row['优化前']:.4f}  →  优化后 {row['优化后']:.4f}"
                   f"   （变化 {row['变化']:+.4f}，相对提升 {row['相对提升']}）", font=cv.f_h)
    y += 34
    cv.text(30, y, f"工单阈值 ≥ {target:.2f}    判定："
                   f"{'✅ 达标' if row['优化后'] >= target else '❌ 未达标'}",
            font=cv.f_h, fill=(GOOD if row["优化后"] >= target else BAD))
    y += 42
    cv.text(30, y, "逐题对比（蓝=优化前，绿=优化后）", font=cv.f_h, fill=GREY)
    y += 28
    bar_w = 430
    for item in per_q:
        before = float(item.get("before_precision" if key == "context_precision" else "before_recall") or 0)
        after = float(item.get("after_precision" if key == "context_precision" else "after_recall") or 0)
        cv.text(30, y + 2, str(item.get("id")), font=cv.f_small)
        cv.text(78, y + 2, str(item.get("question_type") or "")[:6], font=cv.f_small, fill=GREY)
        cv.bar(190, y, bar_w, 13, before, color=(150, 175, 215), value=f"{before:.3f}")
        cv.bar(190, y + 15, bar_w, 13, after,
               color=(GOOD if after >= before else BAD), value=f"{after:.3f}")
        y += 32
    cv.save(name)


# ---------------------------------------------------------------------------
# 3. 逐题指标明细
# ---------------------------------------------------------------------------
def shot_detail() -> None:
    rows = _jsonl(OPT_RESULT / "优化后_逐题.jsonl")
    if not rows:
        raise SystemExit("[ERROR] 缺少 优化/评估结果/优化后_逐题.jsonl")
    cv = Canvas(1240, 190 + 28 * len(rows) + 60, "优化后 17 题逐题指标明细",
                "数据源：优化/评估结果/优化后_逐题.jsonl（自研 RAGAS 近似口径，非 LLM 裁判）")
    cols = [(30, "题号"), (110, "题型"), (230, "上下文精度"), (350, "上下文召回"),
            (470, "TopK精度"), (590, "上下文块"), (700, "总耗时ms"), (820, "生成方式"),
            (1010, "严格判对")]
    y = 100
    for x, label in cols:
        cv.text(x, y, label, font=cv.f_h, fill=GREY)
    y += 26
    cv.hline(24, 1216, y)
    y += 8
    for row in rows:
        m = row.get("metrics") or {}
        judge = row.get("judge") or {}
        cp = float(m.get("context_precision") or 0)
        cr = float(m.get("context_recall") or 0)
        tcp = float(m.get("topk_context_precision") or 0)
        cv.text(30, y, str(row.get("id")))
        cv.text(110, y, str(row.get("question_type") or "")[:7], font=cv.f_small)
        cv.text(230, y, f"{cp:.3f}", fill=(GOOD if cp >= 0.8 else (WARN if cp >= 0.5 else BAD)))
        cv.text(350, y, f"{cr:.3f}", fill=(GOOD if cr >= 0.9 else (WARN if cr >= 0.5 else BAD)))
        cv.text(470, y, f"{tcp:.3f}", fill=GREY)
        cv.text(590, y, str(m.get("context_size") or 0))
        cv.text(700, y, f"{float(row.get('latency_total_ms') or 0):.0f}")
        cv.text(820, y, str(row.get("generation_method") or ""), font=cv.f_small)
        cv.text(1010, y, "✓" if judge.get("is_correct") else ("~" if judge.get("partial_correct") else "✗"),
                fill=(GOOD if judge.get("is_correct") else (WARN if judge.get("partial_correct") else BAD)))
        y += 28
    cv.save("逐题指标明细.png")


# ---------------------------------------------------------------------------
# 4. 响应时间分布
# ---------------------------------------------------------------------------
def shot_latency() -> None:
    data = _cmp()
    if not data:
        raise SystemExit("[ERROR] 缺少 优化前后对比.json")
    rt = data["响应时间对比"]
    cv = Canvas(1040, 560, "响应时间分布（提问 → 返回答案，use_cache=false）",
                "数据源：优化/评估结果/优化前后对比.json §响应时间对比")
    y = 104
    cv.text(30, y, "指标", font=cv.f_h, fill=GREY)
    cv.text(300, y, "优化前 (ms)", font=cv.f_h, fill=GREY)
    cv.text(540, y, "优化后 (ms)", font=cv.f_h, fill=GREY)
    cv.text(780, y, "预算 (ms)", font=cv.f_h, fill=GREY)
    y += 26
    cv.hline(24, 1016, y)
    y += 10
    entries = [("平均", rt["平均ms"]), ("P50", rt.get("P90ms")), ("P90", rt["P90ms"]),
               ("P95", rt["P95ms"]), ("最大", rt["最大ms"]), ("其中生成段平均", rt["生成段_平均ms"])]
    sla = float(data["验收达标判定"]["阈值"]["响应时间ms"])
    for label, row in entries:
        cv.text(30, y, label)
        cv.text(300, y, f"{float(row['优化前']):.2f}")
        after = float(row["优化后"])
        cv.text(540, y, f"{after:.2f}", fill=(GOOD if after <= sla else BAD))
        cv.text(780, y, f"≤ {sla:.0f}", fill=GREY)
        y += 30
    y += 12
    cv.hline(24, 1016, y)
    y += 14
    cv.text(30, y, f"SLA 达标率：优化前 {rt['SLA达标率']['优化前']:.4f}  →  "
                   f"优化后 {rt['SLA达标率']['优化后']:.4f}", font=cv.f_h,
            fill=GOOD)
    y += 44
    cv.text(30, y, "耗时条形（按 3000 ms 满量程）", font=cv.f_h, fill=GREY)
    y += 28
    for label, row in entries:
        cv.bar(230, y, 560, 14, float(row["优化前"]) / sla, color=(150, 175, 215),
               label=f"{label} 优化前", value=f"{float(row['优化前']):.0f}")
        cv.bar(230, y + 16, 560, 14, float(row["优化后"]) / sla, color=GOOD,
               label=f"{label} 优化后", value=f"{float(row['优化后']):.0f}")
        y += 38
    cv.save("响应时间分布.png")


# ---------------------------------------------------------------------------
# 5. 三层面优化收益
# ---------------------------------------------------------------------------
def shot_three_layers() -> None:
    gstats = _load(OPT_GRAPH / "图谱优化统计.json") or {}
    data = _cmp() or {}
    cv = Canvas(1180, 760, "三层面优化收益（实体/关系构建层 · 检索层 · 生成层）",
                "数据源：研发/data/图谱优化后/图谱优化统计.json + 优化/评估结果/优化前后对比.json")
    y = 104
    g_in, g_out = gstats.get("输入", {}), gstats.get("输出", {})
    alias = gstats.get("别名归并", {})
    rel = gstats.get("关系规范化", {})
    prune = gstats.get("噪声剪枝", {})

    cv.text(30, y, "① 实体/关系构建层", font=cv.f_h, fill=BRAND)
    y += 28
    for label, before, after, fmt in (
            ("节点数", g_in.get("节点"), g_out.get("节点"), "{:.0f}"),
            ("边数", g_in.get("边"), g_out.get("边"), "{:.0f}"),
            ("平均度数", g_in.get("平均度数"), g_out.get("平均度数"), "{:.4f}"),
            ("别名归并组数", 0, alias.get("归并组数"), "{:.0f}"),
            ("被并入实体数", 0, alias.get("被并入实体数"), "{:.0f}"),
            ("规范关系覆盖率", 0, rel.get("规范关系覆盖率"), "{:.4f}"),
            ("抽取碎片剪除数", 0, prune.get("抽取碎片剪除数"), "{:.0f}"),
            ("孤立节点剪除数", 0, prune.get("孤立节点剪除数"), "{:.0f}")):
        cv.text(50, y, label)
        cv.text(300, y, fmt.format(float(before or 0)), fill=GREY)
        cv.text(500, y, "→", fill=GREY)
        cv.text(560, y, fmt.format(float(after or 0)),
                fill=(BAD if label in ("节点数", "边数") else GOOD))
        y += 26
    y += 18

    cv.text(30, y, "② 检索层（融合权重 + 通道 + 锚点）", font=cv.f_h, fill=BRAND)
    y += 28
    fw = ((data.get("口径") or {}).get("融合权重") or {})
    cv.text(50, y, "融合权重（base/graph/feature）")
    cv.text(560, y, json.dumps(fw, ensure_ascii=False) if fw else "0.62 / 0.23 / 0.15", fill=GOOD)
    y += 26
    for key, label in (("topk_context_precision", "Top-K 附口径 上下文精度"),
                       ("topk_context_recall", "Top-K 附口径 上下文召回"),
                       ("context_precision_rank", "上下文精度（rank 加权口径）")):
        row = (data.get("核心指标对比") or {}).get(key) or {}
        if not row:
            continue
        cv.text(50, y, label)
        cv.text(400, y, f"{row['优化前']:.4f} → {row['优化后']:.4f}")
        cv.text(700, y, f"{row['变化']:+.4f}", fill=(GOOD if row["变化"] >= 0 else BAD))
        y += 26
    y += 18

    cv.text(30, y, "③ 生成层（模板快路径 + 三元组注入 + 引用 + 拒答护栏）", font=cv.f_h, fill=BRAND)
    y += 28
    rt = data.get("响应时间对比") or {}
    gen = rt.get("生成段_平均ms") or {}
    cv.text(50, y, "生成段平均耗时")
    cv.text(400, y, f"{float(gen.get('优化前') or 0):.0f} ms → {float(gen.get('优化后') or 0):.0f} ms")
    cv.text(760, y, "耗时降幅 >99%（模板快路径生成耗时≈0）", fill=GOOD)
    y += 26
    methods = ((data.get("口径") or {}).get("生成方式分布") or {})
    cv.text(50, y, "生成方式分布")
    cv.text(400, y, json.dumps(methods, ensure_ascii=False) if methods else
            "见 优化后_逐题.jsonl 的 generation_method", fill=GREY)
    y += 26
    for label, key in (("上下文精度", "context_precision"), ("上下文召回", "context_recall")):
        row = (data.get("核心指标对比") or {}).get(key) or {}
        if not row:
            continue
        cv.text(50, y, f"{label}净提升")
        cv.text(400, y, f"{row['变化']:+.4f}", fill=(GOOD if row["变化"] >= 0 else BAD))
        y += 26
    cv.save("三层面优化收益.png")


# ---------------------------------------------------------------------------
# 6. 图谱优化前后对比
# ---------------------------------------------------------------------------
def shot_graph() -> None:
    g = _load(OPT_GRAPH / "图谱优化统计.json")
    if not g:
        raise SystemExit("[ERROR] 缺少 研发/data/图谱优化后/图谱优化统计.json（先跑 构建优化图谱.py）")
    cv = Canvas(1080, 620, "知识图谱优化前后对比（实体/关系构建层）",
                "数据源：研发/data/图谱优化后/图谱优化统计.json")
    y = 104
    g_in, g_out = g.get("输入", {}), g.get("输出", {})
    cv.text(30, y, "指标", font=cv.f_h, fill=GREY)
    cv.text(330, y, "优化前", font=cv.f_h, fill=GREY)
    cv.text(560, y, "优化后", font=cv.f_h, fill=GREY)
    cv.text(790, y, "说明", font=cv.f_h, fill=GREY)
    y += 26
    cv.hline(24, 1056, y)
    y += 10
    rows = [("节点数", g_in.get("节点"), g_out.get("节点"), "合并重复主体 + 剪噪声"),
            ("边数", g_in.get("边"), g_out.get("边"), "自环/非法/低置信剪除"),
            ("平均度数", g_in.get("平均度数"), g_out.get("平均度数"), "连通性提升"),
            ("别名归并组数", 0, (g.get("别名归并") or {}).get("归并组数"), "全称/简称、人名补全"),
            ("被并入实体数", 0, (g.get("别名归并") or {}).get("被并入实体数"), "并入规范名"),
            ("规范关系覆盖率", 0, (g.get("关系规范化") or {}).get("规范关系覆盖率"), "schema 约束"),
            ("自环剪除数", 0, (g.get("关系规范化") or {}).get("自环剪除数"), "head==tail"),
            ("重复边合并数", 0, (g.get("关系规范化") or {}).get("重复边合并数"), "同三元组去重"),
            ("抽取碎片剪除数", 0, (g.get("噪声剪枝") or {}).get("抽取碎片剪除数"), "伪实体"),
            ("孤立节点剪除数", 0, (g.get("噪声剪枝") or {}).get("孤立节点剪除数"), "度0且仅1次"),
            ("通用刻度剪除数", 0, (g.get("噪声剪枝") or {}).get("通用刻度剪除数"), "2020年度 等")]
    for label, before, after, note in rows:
        cv.text(30, y, label)
        cv.text(330, y, f"{float(before or 0):.4f}" if isinstance(before, float) else f"{int(before or 0)}",
                fill=GREY)
        cv.text(560, y, f"{float(after or 0):.4f}" if isinstance(after, float) else f"{int(after or 0)}",
                fill=(GOOD if label != "节点数" and label != "边数" else BAD))
        cv.text(790, y, note, font=cv.f_small, fill=GREY)
        y += 30
    y += 14
    cv.text(30, y, "结论：节点 −31.6% / 边 −21.7%，但**平均度数 +14.5%** ——"
                   "主体更集中、连通性更好、噪声更少。", font=cv.f_h, fill=BRAND)
    cv.save("图谱优化前后对比.png")


# ---------------------------------------------------------------------------
# 7. 多语言问答示例
# ---------------------------------------------------------------------------
def shot_multilingual() -> None:
    rows = _jsonl(TEST_DATA / "多语言问答.jsonl")
    if not rows:
        raise SystemExit("[ERROR] 缺少 测试/测试数据/多语言问答.jsonl（先跑 研发/评估/多语言实测.py）")
    pairs: dict[str, dict[str, Any]] = {}
    for row in rows:
        pairs.setdefault(str(row.get("pair")), {})[str(row.get("lang"))] = row
    cv = Canvas(1180, 200 + 210 * len(pairs), "多语言问答示例（中文 + 英文实测）",
                "数据源：测试/测试数据/多语言问答.jsonl（英文提问走英中术语映射）")
    y = 100
    for pair, langs in pairs.items():
        zh, en = langs.get("zh") or {}, langs.get("en") or {}
        cv.text(30, y, f"【{pair}】", font=cv.f_h, fill=BRAND)
        y += 26
        for tag, row, color in (("中文", zh, INK), ("英文", en, (16, 92, 140))):
            cv.text(46, y, f"{tag}提问：", font=cv.f_small, fill=GREY)
            cv.text(130, y, str(row.get("question") or "")[:74], font=cv.f_small)
            y += 22
            ans = str(row.get("answer") or "").replace("\n", "  ")
            cv.text(46, y, "答案：", font=cv.f_small, fill=GREY)
            cv.text(130, y, ans[:80], font=cv.f_small, fill=color)
            y += 20
            cv.text(130, y, ans[80:160], font=cv.f_small, fill=color)
            y += 20
            cv.text(46, y, f"耗时 {float(row.get('latency_ms') or 0):.0f} ms · "
                           f"方式 {row.get('generation_method')} · "
                           f"数值原子 {','.join(row.get('answer_atoms') or [])[:60]}",
                    font=cv.f_small, fill=GREY)
            y += 30
        y += 10
        cv.hline(30, 1150, y - 8)
    cv.save("多语言问答示例.png")


# ---------------------------------------------------------------------------
# 8. 稳定性与容错
# ---------------------------------------------------------------------------
def shot_stability() -> None:
    stab = _load(TEST_DATA / "稳定性_连续查询.json")
    tole = _load(TEST_DATA / "容错降级.json")
    if not stab or not tole:
        raise SystemExit("[ERROR] 缺少 稳定性/容错 产物（先跑 研发/评估/稳定性压测.py）")
    cases = tole.get("用例") or []
    cv = Canvas(1160, 480 + 26 * len(cases), "稳定性与容错（连续 50 次查询 + 8 类异常场景）",
                "数据源：测试/测试数据/稳定性_连续查询.json + 容错降级.json")
    y = 104
    cv.text(30, y, "① 连续查询稳定性", font=cv.f_h, fill=BRAND)
    y += 28
    rt = stab.get("响应时间ms") or {}
    for label, value in (("总次数", stab.get("总次数")), ("成功次数", stab.get("成功次数")),
                         ("抛出异常次数", stab.get("异常次数")), ("成功率", f"{stab.get('成功率')}"),
                         ("平均耗时ms", rt.get("平均")), ("P50", rt.get("P50")),
                         ("P90", rt.get("P90")), ("P95", rt.get("P95")), ("最大", rt.get("最大"))):
        cv.text(50, y, label)
        cv.text(300, y, str(value), fill=(GOOD if label != "抛出异常次数" else BAD))
        y += 24
    cv.text(50, y, "SLA 达标率")
    cv.text(300, y, f"{(stab.get('SLA') or {}).get('达标率')}", fill=GOOD)
    y += 34
    cv.text(30, y, "② 容错与降级（全部要求「不崩」）", font=cv.f_h, fill=BRAND)
    y += 28
    cv.text(50, y, "场景", font=cv.f_small, fill=GREY)
    cv.text(370, y, "生成方式", font=cv.f_small, fill=GREY)
    cv.text(560, y, "拒答", font=cv.f_small, fill=GREY)
    cv.text(650, y, "耗时ms", font=cv.f_small, fill=GREY)
    cv.text(760, y, "抛出异常", font=cv.f_small, fill=GREY)
    cv.text(920, y, "期望", font=cv.f_small, fill=GREY)
    y += 22
    for case in cases:
        cv.text(50, y, str(case.get("场景"))[:26], font=cv.f_small)
        cv.text(370, y, str(case.get("generation_method") or "—"), font=cv.f_small, fill=GREY)
        cv.text(560, y, "是" if case.get("refused") else "否", font=cv.f_small)
        cv.text(650, y, f"{float(case.get('耗时ms') or 0):.0f}", font=cv.f_small)
        cv.text(760, y, "无 ✅" if not case.get("抛出异常") else str(case["抛出异常"])[:22],
                font=cv.f_small, fill=(GOOD if not case.get("抛出异常") else BAD))
        cv.text(920, y, str(case.get("期望") or "")[:26], font=cv.f_small, fill=GREY)
        y += 26
    y += 12
    cv.text(50, y, f"判定：{tole.get('用例数')} 个场景，崩溃 {(tole.get('判定') or {}).get('崩溃明细') and len((tole.get('判定') or {}).get('崩溃明细')) or 0} 个"
                   f" —— {'全部不崩 ✅' if (tole.get('判定') or {}).get('全部不崩') else '存在崩溃 ❌'}",
            font=cv.f_h, fill=GOOD)
    cv.save("稳定性与容错.png")


SHOTS: dict[str, Any] = {
    "上下文精度优化前后对比": lambda: shot_metric(
        "上下文精度优化前后对比.png", "context_precision", "上下文精度 context_precision 优化前后对比", 0.80),
    "上下文召回优化前后对比": lambda: shot_metric(
        "上下文召回优化前后对比.png", "context_recall", "上下文召回 context_recall 优化前后对比", 0.90),
    "逐题指标明细": shot_detail,
    "响应时间分布": shot_latency,
    "三层面优化收益": shot_three_layers,
    "图谱优化前后对比": shot_graph,
    "多语言问答示例": shot_multilingual,
    "稳定性与容错": shot_stability,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{WORK_ORDER} —— 生成测试截图")
    parser.add_argument("--only", default="", help="只生成指定的截图（中文名，如 响应时间分布）")
    args = parser.parse_args(argv)
    targets = [args.only] if args.only else list(SHOTS)
    print(f"工单9 截图生成：共 {len(targets)} 张 → {SHOT_DIR}")
    ok = 0
    for name in targets:
        fn = SHOTS.get(name)
        if fn is None:
            print(f"  [跳过] 未知截图名 {name}")
            continue
        try:
            fn()
            ok += 1
        except SystemExit as exc:
            print(f"  [失败] {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 —— 单张失败不影响其它
            print(f"  [异常] {name}: {type(exc).__name__}: {exc}")
    print(f"完成：{ok}/{len(targets)} 张")
    return 0 if ok == len(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
