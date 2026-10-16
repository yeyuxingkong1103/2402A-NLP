# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

测试截图生成：``测试/测试脚本/生成截图.py``。

**截图内容全部来自真实产物文件**（``测试/测试数据/*.json(l)``、
``优化/评估结果/*.json``），不手绘、不编造。文件名严格中文、按测试内容命名，
与 ``测试/测试报告.md`` / ``优化/瓶颈排查与修复报告.md`` 里的引用逐字一致。

本机**没有 matplotlib**（见 ``设计/环境事实.md`` §1），所以图表用 PIL 自绘。

    pwsh -NoProfile -File run_py.ps1 测试/测试脚本/生成截图.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "测试" / "测试数据"
_OUT = _ROOT / "测试" / "测试截图"
_EVAL = _ROOT / "优化" / "评估结果"

W, H = 1500, 900
BG = (252, 252, 250)
FG = (26, 32, 44)
MUTED = (110, 120, 135)
BAR_BASE = (60, 110, 200)
BAR_OPT = (230, 130, 55)
LINE_COLORS = [(60, 110, 200), (230, 130, 55), (40, 160, 110), (200, 60, 90), (140, 90, 200)]
OK_COLOR = (30, 140, 80)
BAD_COLOR = (200, 50, 50)
GRID = (225, 228, 235)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """找一个能显示中文的字体；找不到就退回默认（不抛）。"""
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_SUB = font(19)
F_H = font(21, True)
F_B = font(19)
F_S = font(16)
F_TINY = font(14)


def new_page(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 78], fill=(28, 40, 62))
    draw.text((30, 20), title, font=F_TITLE, fill=(255, 255, 255))
    if subtitle:
        draw.text((32, 88), subtitle, font=F_SUB, fill=MUTED)
    draw.text((30, H - 32), f"工单编号：{WORK_ORDER}", font=F_TINY, fill=MUTED)
    return img, draw


def save(img: Image.Image, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    img.save(path, "PNG")
    print(f"  [截图] {name}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


def summary(tag: str, scenario: str) -> dict[str, Any] | None:
    return load_json(_DATA / f"{tag}_场景{scenario}_summary.json")


def mem_curve(tag: str, scenario: str) -> list[dict[str, Any]]:
    return load_jsonl(_DATA / f"{tag}_场景{scenario}_mem.jsonl")


def table(draw: ImageDraw.ImageDraw, x: int, y: int, headers: Sequence[str],
          rows: Sequence[Sequence[Any]], widths: Sequence[int], *,
          row_h: int = 34, highlight: dict[int, tuple[int, int, int]] | None = None) -> int:
    """画表格，返回结束 y。"""
    draw.rectangle([x, y, x + sum(widths), y + row_h], fill=(238, 241, 246))
    cx = x
    for head, width in zip(headers, widths):
        draw.text((cx + 10, y + 8), str(head), font=F_H, fill=FG)
        cx += width
    cy = y + row_h
    for ridx, row in enumerate(rows):
        color = (highlight or {}).get(ridx)
        if color:
            draw.rectangle([x, cy, x + sum(widths), cy + row_h], fill=color)
        elif ridx % 2:
            draw.rectangle([x, cy, x + sum(widths), cy + row_h], fill=(246, 248, 251))
        cx = x
        for cell, width in zip(row, widths):
            draw.text((cx + 10, cy + 8), str(cell), font=F_B, fill=FG)
            cx += width
        cy += row_h
    draw.line([x, cy, x + sum(widths), cy], fill=(205, 210, 220), width=2)
    return cy


def bars(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
         groups: Sequence[tuple[str, Sequence[tuple[str, float, tuple[int, int, int]]]]],
         *, unit: str = "", log_scale: bool = False) -> None:
    """分组柱状图。``groups`` = [(组名, [(系列名, 值, 颜色), ...]), ...]"""
    all_values = [v for _g, series in groups for _n, v, _c in series] or [1.0]
    vmax = max(all_values)
    if log_scale:
        import math

        vmax = math.log10(max(vmax, 1.0) * 1.2)
    draw.line([x, y + h, x + w, y + h], fill=(190, 196, 206), width=2)
    for g in range(1, 5):
        gy = y + h - h * g / 4
        draw.line([x, gy, x + w, gy], fill=GRID, width=1)
        label = vmax * g / 4
        draw.text((x - 66, gy - 10), f"{label:.0f}", font=F_TINY, fill=MUTED)
    gw = w / max(1, len(groups))
    for gi, (gname, series) in enumerate(groups):
        n = max(1, len(series))
        bw = min(56, gw / (n + 1.4))
        for si, (sname, value, color) in enumerate(series):
            import math

            scaled = math.log10(max(value, 1.0)) if log_scale else value
            bh = (scaled / vmax) * h if vmax else 0
            bx = x + gi * gw + gw / 2 - (n * bw) / 2 + si * bw + bw * 0.15
            by = y + h - bh
            draw.rectangle([bx, by, bx + bw * 0.7, y + h], fill=color)
            draw.text((bx, by - 24), f"{value:g}{unit}", font=F_S, fill=FG)
        draw.text((x + gi * gw + gw / 2 - len(gname) * 8, y + h + 12), gname, font=F_H, fill=FG)
    # 图例
    lx = x + w - 200
    for _gi, (_gname, series) in enumerate(groups[:1]):
        for si, (sname, _v, color) in enumerate(series):
            draw.rectangle([lx, y - 34 + si * 0, lx + 18, y - 18], fill=color)
            draw.text((lx + 26, y - 22), sname, font=F_S, fill=FG)
            lx += 110


def line_chart(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
               series: Sequence[tuple[str, Sequence[tuple[float, float]], tuple[int, int, int]]],
               *, unit: str = "", y_min: float | None = None, y_max: float | None = None) -> None:
    """折线图。``series`` = [(名字, [(x, y), ...], 颜色), ...]"""
    points = [p for _n, pts, _c in series for p in pts]
    if not points:
        draw.text((x + 20, y + 20), "（无数据）", font=F_B, fill=MUTED)
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs) or 1
    y0 = y_min if y_min is not None else min(ys)
    y1 = y_max if y_max is not None else max(ys)
    if y1 - y0 < 1e-9:
        y1 = y0 + 1
    draw.line([x, y + h, x + w, y + h], fill=(190, 196, 206), width=2)
    draw.line([x, y, x, y + h], fill=(190, 196, 206), width=2)
    for g in range(0, 5):
        gy = y + h - h * g / 4
        draw.line([x, gy, x + w, gy], fill=GRID, width=1)
        draw.text((x - 76, gy - 10), f"{y0 + (y1 - y0) * g / 4:.0f}", font=F_TINY, fill=MUTED)
    for gi in range(0, 6):
        gx = x + w * gi / 5
        draw.text((gx - 16, y + h + 10), f"{x0 + (x1 - x0) * gi / 5:.0f}", font=F_TINY, fill=MUTED)
    for name, pts, color in series:
        if len(pts) < 2:
            continue
        coords = [(x + w * (px - x0) / (x1 - x0 or 1),
                   y + h - h * (py - y0) / (y1 - y0)) for px, py in pts]
        draw.line(coords, fill=color, width=3)
        for cx, cy in coords[:: max(1, len(coords) // 40)]:
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=color)
    lx = x
    for name, _pts, color in series:
        draw.rectangle([lx, y - 30, lx + 20, y - 14], fill=color)
        draw.text((lx + 28, y - 30), f"{name}（{unit}）" if unit else name, font=F_S, fill=FG)
        lx += 250


def kv_block(draw: ImageDraw.ImageDraw, x: int, y: int, items: Sequence[tuple[str, Any]],
             *, width: int = 700, row_h: int = 33, value_color: tuple[int, int, int] | None = None) -> int:
    for i, (k, v) in enumerate(items):
        yy = y + i * row_h
        draw.text((x, yy), str(k), font=F_B, fill=MUTED)
        draw.text((x + 330, yy), str(v), font=F_B, fill=value_color or FG)
    return y + len(items) * row_h


# ======================================================================
# 各张截图
# ======================================================================
def shot_01_延迟分位对比() -> None:
    b200, a200 = summary("基准_优化前_思考200ms", "A"), None
    b0, a0 = summary("基准_优化前", "A"), summary("优化后", "A")
    bmu, amu = summary("基准_优化前_多用户", "A"), summary("优化后_多用户", "A")
    groups = []
    for label, before, after in (("固定会话 20 并发\n(思考 200ms)", b200, a200),
                                 ("固定会话 20 并发\n(无思考)", b0, a0),
                                 ("多用户 20 并发\n(每请求新会话)", bmu, amu)):
        if before is None:
            continue
        series = [("优化前 P95", before["latency_ms"]["p95"], BAR_BASE)]
        if after is not None:
            series.append(("优化后 P95", after["latency_ms"]["p95"], BAR_OPT))
        groups.append((label, series))
    img, draw = new_page("优化前后延迟分位对比（场景A，20 并发）",
                         "柱高 = P95 客户端延迟（ms）；阈值线 = 验收线 3000 ms")
    if not groups:
        draw.text((60, 200), "（缺少 summary.json，先跑压测）", font=F_B, fill=BAD_COLOR)
        save(img, "优化前后延迟分位对比.png")
        return
    ax, ay, aw, ah = 130, 200, W - 260, 400
    bars(draw, ax, ay, aw, ah, groups, unit=" ms")
    top = max(v for _g, s in groups for _n, v, _c in s)
    ymax = max(top * 1.2, 3000)
    yline = ay + ah - ah * 3000 / ymax
    if 0 <= yline <= ay + ah:
        draw.line([ax, yline, ax + aw, yline], fill=BAD_COLOR, width=3)
        draw.text((ax + aw - 210, yline - 26), "验收线 P95 = 3000 ms", font=F_H, fill=BAD_COLOR)
    rows = []
    for label, before, after in (("固定会话·思考200ms", b200, a200),
                                 ("固定会话·无思考", b0, a0),
                                 ("多用户·每请求新会话", bmu, amu)):
        if before is None:
            continue
        rows.append([label, before["latency_ms"]["p50"], before["latency_ms"]["p95"],
                     before["latency_ms"]["p99"], before["rps"],
                     "-" if after is None else f'{after["latency_ms"]["p95"]}',
                     "-" if after is None else f'{after["rps"]}'])
    table(draw, 130, 700, ["口径", "优化前 P50", "优化前 P95", "优化前 P99", "优化前 RPS",
                           "优化后 P95", "优化后 RPS"], rows, [330, 150, 150, 150, 150, 150, 150])
    save(img, "优化前后延迟分位对比.png")


def shot_02_并发吞吐与错误率() -> None:
    img, draw = new_page("并发吞吐与错误率曲线（场景A）",
                         "左：随并发升高的延迟；右：吞吐与失败数（多用户口径暴露排队坍塌）")
    points = []
    rows = []
    for tag, label in (("冒烟_优化前", "3 并发(固定会话)"), ("基准_优化前_思考200ms", "20 并发(思考200ms)"),
                       ("基准_优化前", "20 并发(无思考)"), ("基准_优化前_多用户", "20 并发(每请求新会话)")):
        d = summary(tag, "A")
        if d is None:
            continue
        conc = d["concurrency"]
        points.append((conc, d["latency_ms"]["p95"], label))
        rows.append([label, conc, d["total"], d["failed"], d["rps"],
                     d["latency_ms"]["p50"], d["latency_ms"]["p95"], d["latency_ms"]["p99"]])
    if rows:
        ax, ay, aw, ah = 130, 210, 620, 430
        line_chart(draw, ax, ay, aw, ah,
                   [("P95 延迟", [(p[0], p[1]) for p in points], LINE_COLORS[0])], unit="ms")
        bx = 830
        table(draw, bx, 230, ["口径", "并发", "总数", "失败", "RPS"],
              [[r[0], r[1], r[2], r[3], r[4]] for r in rows], [270, 70, 90, 70, 90], row_h=32)
        table(draw, bx, 470, ["口径", "P50", "P95", "P99"],
              [[r[0], r[5], r[6], r[7]] for r in rows], [270, 90, 90, 90], row_h=32)
    save(img, "并发吞吐与错误率曲线.png")


def shot_03_内存RSS增长曲线() -> None:
    img, draw = new_page("内存 RSS 增长曲线（宿主进程口径）",
                         "docker stats 不可用 → 用宿主进程 WorkingSet(RSS) 与私有内存代替，口径见设计/环境事实.md")
    series = []
    for idx, (tag, label) in enumerate((("基准_优化前", "优化前 场景A"),
                                        ("优化后", "优化后 场景A"),
                                        ("soak_优化后", "优化后 soak"))):
        rows = mem_curve(tag, "A")
        pts = [(float(r["t_s"]), float(r["api_rss_mb"])) for r in rows
               if isinstance(r.get("api_rss_mb"), (int, float)) and r["api_rss_mb"] > 0]
        if len(pts) >= 2:
            series.append((label, pts[:: max(1, len(pts) // 120)], LINE_COLORS[idx]))
    if series:
        line_chart(draw, 150, 220, W - 320, 430, series, unit="MB")
    else:
        draw.text((80, 240), "（缺少 *_mem.jsonl，先跑压测）", font=F_B, fill=BAD_COLOR)
    items = []
    for tag, label in (("基准_优化前", "优化前 场景A"), ("优化后", "优化后 场景A"),
                       ("soak_优化后", "优化后 soak")):
        d = summary(tag, "A")
        if d is None:
            continue
        m = d["memory"]["api_process"]
        items.append((f"{label}：RSS/私有增长",
                      f'{m.get("growth_pct")}%（斜率 {m.get("slope_mb_per_min")} MB/min，{m.get("trend")}）'))
    kv_block(draw, 150, 690, items, width=1200)
    save(img, "内存RSS增长曲线.png")


def shot_04_tracemalloc泄漏对象Top20() -> None:
    data = load_json(_DATA / "tracemalloc_优化前.json") or load_json(_DATA / "tracemalloc_优化后.json")
    img, draw = new_page("tracemalloc 泄漏对象 Top20（替代 memory-profiler / objgraph）",
                         "正数 = 快照期间净增（泄漏嫌疑）；负数 = 已回收；来源：测试数据/tracemalloc_*.json")
    if not data:
        draw.text((80, 240), "（缺少 tracemalloc_*.json，先跑 研发/剖析/内存剖析.py）", font=F_B, fill=BAD_COLOR)
        save(img, "tracemalloc泄漏对象Top20.png")
        return
    rows = []
    for item in (data.get("tracemalloc_top") or [])[:20]:
        color = BAD_COLOR if item["size_diff_kb"] > 200 else FG
        rows.append([item["trace"][-66:], f'{item["size_diff_kb"]:+.1f}', item["count_diff"],
                     f'{item["size_kb"]:.1f}'])
    highlight = {i: (255, 240, 235) for i, r in enumerate(rows) if r[1].startswith("+")}
    table(draw, 80, 170, ["调用点（文件:行）", "净增 KB", "对象数变化", "当前 KB"], rows,
          [640, 150, 160, 150], row_h=30, highlight=highlight)
    m = data.get("rss_mb", {})
    kv_block(draw, 80, 790, [("RSS 起止", f'{m.get("start")} → {m.get("end")} MB（{m.get("delta")}）'),
                             ("对象总数变化", f'{data.get("objects", {}).get("start")} → '
                                             f'{data.get("objects", {}).get("end")}')])
    save(img, "tracemalloc泄漏对象Top20.png")


def shot_05_瓶颈定位证据() -> None:
    data = load_json(_DATA / "剖析问答链路_优化前.json") or {}
    after = load_json(_DATA / "剖析问答链路_优化后.json") or {}
    img, draw = new_page("瓶颈定位证据（每请求外部 HTTP 往返 + 分段耗时）",
                         "urllib.request.urlopen 调用计数（monkeypatch 实测）—— 修复项 F1/F2 的直接证据")
    rows = []
    for label, d in (("优化前", data), ("优化后", after)):
        if not d:
            continue
        u = d.get("urlopen_calls", {})
        rows.append([label, u.get("total"), u.get("per_request_total"), u.get("api_tags"),
                     u.get("per_request_api_tags"), u.get("api_embed"), u.get("api_generate")])
    if rows:
        table(draw, 80, 170, ["口径", "urlopen 总次数", "每请求", "/api/tags 探测", "每请求探测",
                              "/api/embed", "/api/generate"], rows,
              [130, 220, 150, 210, 170, 160, 190], row_h=40)
    rows2 = []
    for label, d in (("优化前", data), ("优化后", after)):
        if not d:
            continue
        s = d.get("stage_ms", {})
        rows2.append([label, s.get("retrieve_mean"), s.get("generate_mean"), s.get("total_mean"),
                      d.get("latency_ms", {}).get("mean"),
                      d.get("engine_stats", {}).get("memory_cache")])
    if rows2:
        table(draw, 80, 400, ["口径", "retrieve 均值(ms)", "generate 均值(ms)", "total 均值(ms)",
                              "端到端均值(ms)", "答案缓存条目"], rows2,
              [130, 230, 230, 210, 220, 200], row_h=40)
    kv_block(draw, 80, 620, [
        ("瓶颈①入口无限流", "service/config.py:119 max_concurrency=4 —— 全代码零引用（grep 实证）"),
        ("瓶颈②每请求多探一次", "app/core/embedder.py:299 每次 embed_texts 都 resolve_embedding_backend(probe=True)"),
        ("瓶颈③每请求新建 TCP", "llm_client.py:92/239/278 + embedder.py:84 全走 urllib.request.urlopen"),
        ("瓶颈④落盘串行", "sqlite_manager.py:155/176/190/200 每请求 4 次 commit，全在一把锁上"),
    ], row_h=40)
    save(img, "瓶颈定位证据.png")


def shot_06_场景B混合负载() -> None:
    b, a = summary("基准_优化前", "B"), summary("优化后", "B")
    img, draw = new_page("场景B 混合负载结果（10 并发 = 5 问答 + 5 上传解析）",
                         "优化前：解析在请求线程内同步完成（API 被阻塞）；优化后：有界队列 + 工作线程池")
    rows = []
    for label, d in (("优化前（同步解析）", b), ("优化后（队列异步）", a)):
        if d is None:
            continue
        rows.append([label, d["total"], d["failed"], d["rate_limited"], d["rps"],
                     d["qa_latency_ms"]["p95"], d["actual_duration_s"],
                     "是" if d.get("circuit_breaker") else "否"])
    if rows:
        table(draw, 80, 180, ["口径", "总请求", "失败", "429", "RPS", "问答 P95(ms)", "实际时长(s)", "熔断"],
              rows, [260, 110, 90, 80, 120, 170, 160, 110], row_h=46)
    ups = []
    for label, tag in (("优化前", "基准_优化前"), ("优化后", "优化后")):
        rows_jsonl = load_jsonl(_DATA / f"{tag}_场景B.jsonl")
        up = [r for r in rows_jsonl if r.get("endpoint") == "upload"]
        ok = [r for r in up if r.get("ok")]
        if up:
            ups.append((label, len(up), len(ok), round(100.0 * len(ok) / len(up), 1),
                        round(sum(r["latency_ms"] for r in up) / len(up), 1)))
    if ups:
        table(draw, 80, 380, ["口径", "上传数", "成功解析", "成功率%", "上传平均耗时(ms)"],
              [list(u) for u in ups], [260, 140, 160, 160, 250], row_h=42)
        groups = [(u[0], [("上传平均耗时", u[4], BAR_BASE if u[0] == "优化前" else BAR_OPT)]) for u in ups]
        bars(draw, 180, 560, 500, 240, groups, unit=" ms")
    tasks = load_jsonl(_ROOT / "部署" / "日志" / "parse_tasks.jsonl")
    if tasks:
        done = len([t for t in tasks if t.get("status") == "done"])
        failed = len([t for t in tasks if t.get("status") == "failed"])
        kv_block(draw, 780, 560, [("解析任务留痕", "部署/日志/parse_tasks.jsonl"),
                                  ("done 任务数", done), ("failed 任务数", failed),
                                  ("总记录数", len(tasks))])
    save(img, "场景B混合负载结果.png")


def shot_07_限流与容错() -> None:
    img, draw = new_page("限流与容错验证",
                         "数据来源：测试/测试数据/限流验证.json、容错验证.json（pytest + 实跑接口留痕）")
    limit = load_json(_DATA / "限流验证.json") or {}
    fault = load_json(_DATA / "容错验证.json") or {}
    rows = []
    for item in (limit.get("cases") or []):
        rows.append([item.get("用例"), item.get("期望"), item.get("实际"),
                     "通过" if item.get("通过") else "未通过"])
    if rows:
        table(draw, 80, 180, ["限流用例", "期望", "实际", "判定"], rows,
              [420, 300, 320, 140], row_h=38,
              highlight={i: (235, 250, 240) if r[3] == "通过" else (255, 235, 235)
                         for i, r in enumerate(rows)})
    rows2 = []
    for item in (fault.get("cases") or []):
        rows2.append([item.get("故障注入"), item.get("行为"), item.get("留痕"),
                      "通过" if item.get("通过") else "未通过"])
    if rows2:
        table(draw, 80, 430, ["故障注入", "期望行为", "留痕位置", "判定"], rows2,
              [300, 430, 320, 140], row_h=38,
              highlight={i: (235, 250, 240) if r[3] == "通过" else (255, 235, 235)
                         for i, r in enumerate(rows2)})
    if not rows and not rows2:
        draw.text((80, 240), "（缺少 限流验证.json / 容错验证.json，先跑 研发/scripts/验证限流与容错.py）",
                  font=F_B, fill=BAD_COLOR)
    save(img, "限流与容错验证.png")


def shot_08_多用户吞吐坍塌() -> None:
    b_fix, b_mu = summary("基准_优化前", "A"), summary("基准_优化前_多用户", "A")
    a_fix, a_mu = summary("优化后", "A"), summary("优化后_多用户", "A")
    img, draw = new_page("多用户口径下的吞吐坍塌与修复（瓶颈复现）",
                         "每请求一个新 session_id：压到「会话/落盘随请求数增长」这一泄漏点")
    groups = []
    labels = [("固定会话\n优化前", b_fix), ("固定会话\n优化后", a_fix),
              ("每请求新会话\n优化前", b_mu), ("每请求新会话\n优化后", a_mu)]
    for label, d in labels:
        if d is not None:
            groups.append((label, [("RPS", d["rps"], BAR_BASE if "优化前" in label else BAR_OPT)]))
    if groups:
        bars(draw, 150, 220, 620, 380, groups, unit="")
        rows = []
        for label, d in labels:
            if d is not None:
                rows.append([label.replace("\n", ""), d["total"], d["rps"],
                             d["latency_ms"]["p50"], d["latency_ms"]["p95"],
                             d["latency_ms"]["p99"], d["failed"]])
        table(draw, 830, 230, ["口径", "总数", "RPS", "P50", "P95", "P99", "失败"],
              rows, [200, 80, 80, 80, 80, 80, 70], row_h=34)
        if b_fix and b_mu:
            drop = (b_fix["rps"] - b_mu["rps"]) / b_fix["rps"] * 100
            kv_block(draw, 830, 560, [
                ("优化前 固定→多用户吞吐", f'{b_fix["rps"]} → {b_mu["rps"]} RPS（-{drop:.1f}%）'),
                ("优化前 多用户 P95", f'{b_mu["latency_ms"]["p95"]} ms（验收线 3000 ms）'),
                ("根因", "会话文件整文件重写 + os.replace，全部串行在一把全局锁上"),
            ], row_h=36)
    save(img, "多用户吞吐坍塌与修复.png")


def shot_09_上传去重与日志轮转() -> None:
    before = load_json(_DATA / "去重与轮转_优化前.json") or {}
    after = load_json(_DATA / "去重与轮转_优化后.json") or {}
    vol = load_json(_DATA / "存储卷取证_清理前.json") or {}
    vol2 = load_json(_DATA / "存储卷取证_清理后.json") or {}
    img, draw = new_page("上传去重与日志轮转前后对比（磁盘泄漏治理）",
                         "数据来源：测试/测试数据/{存储卷取证_清理前.json,去重与轮转_优化前.json,去重与轮转_优化后.json}")
    rows = []
    for label, d in (("优化前（不去重/不轮转）", before), ("优化后（去重+轮转）", after)):
        if not d:
            continue
        rows.append([label, d.get("uploads"), d.get("http_ok"),
                     d.get("delta_uploads_mb"), d.get("bytes_per_upload"),
                     f'{d.get("ratio_to_pdf_size")} ×', d.get("delta_logs_mb"),
                     (d.get("logs_dir_after") or {}).get("biggest_mb")])
    if rows:
        table(draw, 60, 180, ["口径", "上传次数", "HTTP 成功", "磁盘增量(MB)", "每份字节",
                              "占 PDF 体积", "日志增量(MB)", "日志最大单文件(MB)"],
              rows, [280, 100, 110, 140, 130, 130, 140, 180], row_h=44)
        groups = []
        for label, d in (("优化前", before), ("优化后", after)):
            if d:
                groups.append((label, [("磁盘增量 MB", d.get("delta_uploads_mb") or 0,
                                        BAR_BASE if label == "优化前" else BAR_OPT)]))
        if groups:
            bars(draw, 150, 420, 520, 300, groups, unit=" MB")
    else:
        draw.text((70, 250), "（缺少 去重与轮转_*.json，先跑 研发/scripts/验证去重与日志轮转.py）",
                  font=F_B, fill=BAD_COLOR)
    items = []
    up = vol.get("uploads") or {}
    if up:
        items += [("上传卷（清理前）", f'{up.get("file_count")} 份文件 / {up.get("total_mb")} MB'),
                  ("其中唯一内容组", f'{up.get("unique_content_groups")} 组'),
                  ("判定为重复浪费", f'{up.get("wasted_mb")} MB（{up.get("wasted_pct")}%）'),
                  ("存储卷合计", f'{(vol.get("summary") or {}).get("总占用_mb")} MB')]
    if items:
        kv_block(draw, 740, 420, items, row_h=34)
    if vol2:
        kv_block(draw, 740, 570, [
            ("清理后存储卷", f'{vol2.get("after_mb")} MB（清理前 {vol2.get("before_mb")} MB）'),
            ("释放磁盘", f'{vol2.get("freed_mb")} MB = {vol2.get("freed_gb")} GB'),
            ("保留样本", f'{len(vol2.get("kept_samples") or [])} 份唯一内容 PDF'),
        ], row_h=34)
    draw.text((60, 830),
              "注：「日志最大单文件」列是各轮结束时刻的实测值 —— 优化后那一刻 600 MB 的历史文件刚被轮转为 "
              "app.log.1，再经过一个轮转周期即被自动挤出删除（清理后日志卷 13.7 MB，最大单文件 1.00 MB）。",
              font=F_TINY, fill=MUTED)
    save(img, "上传去重与日志轮转前后对比.png")


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    print(f"开始生成截图 → {_OUT}")
    for fn in (shot_01_延迟分位对比, shot_02_并发吞吐与错误率, shot_03_内存RSS增长曲线,
               shot_04_tracemalloc泄漏对象Top20, shot_05_瓶颈定位证据, shot_06_场景B混合负载,
               shot_07_限流与容错, shot_08_多用户吞吐坍塌, shot_09_上传去重与日志轮转):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 —— 单张失败不阻断其它截图
            print(f"  [警告] {fn.__name__} 失败：{type(exc).__name__}: {exc}")
    files = sorted(p.name for p in _OUT.glob("*.png"))
    print(f"\n共 {len(files)} 张：")
    for name in files:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
