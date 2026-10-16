# -*- coding: utf-8 -*-
"""人工智能NLP-RAG-功能测试及评估 —— 测试截图渲染器。

把**真实产物文件**里的数字渲染成 PNG（绝不手绘假数据）：

数据来源优先级
--------------
1. **评测官产物**（优先）：``测试/测试数据/评估汇总.json``、``评估结果.jsonl``、``检索问题分析.md``；
2. **本工单回归产物**：``测试/测试数据/回归结果.json``（由 ``测试/回归测试/run_all.py`` 真实跑出）。

输出（``测试/测试截图/``，中文文件名）：

============================  ==================================================
文件名                         数据来源
============================  ==================================================
``整体评估汇总.png``           ``评估汇总.json``（配置/准确率/召回/SLA/拒答/页命中/题型分组）
``响应时间测试.png``           ``评估汇总.json`` 的 ``响应时间ms`` + ``逐题速览``（含 3000ms SLA 线）
``准确率汇总.png``             ``评估汇总.json`` 的 ``准确率`` + 三张分组表
``检索问题分析.png``           ``检索问题分析.md`` 的 TOP5 表 + ``评估汇总.json`` 的索引可达性
``单轮问答回归.png``           ``回归结果.json`` → 单轮问答组
``多轮对话回归.png``           ``回归结果.json`` → 多轮对话组（含 TC-M01/M02 设计判据）
``异常输入测试.png``           ``回归结果.json`` → 异常输入组
``问题1_检索结果.png``         ``评估结果.jsonl`` → Q01 的 top-10 命中明细
``问题2_评估结果.png``         ``评估结果.jsonl`` → Q02 的判分明细 + 检索侧证据
``回归测试总览.png``           ``回归结果.json`` 汇总（三组通过/失败）
============================  ==================================================

每个数字都能在来源文件里定位；截图页脚写出「来源文件 + 生成时间」，便于逐图复核。

运行：
    python 测试\\生成截图.py            # 渲染全部（缺回归产物时只跳过依赖它的图）
    python 测试\\生成截图.py --only 整体评估汇总
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent / "回归测试"))

import _公共 as K  # noqa: E402

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

# --------------------------------------------------------------------------- 资源
ANNOTATION = "工单7 人工智能NLP-RAG-功能测试及评估"     # 右上角标注（每张图都有）
FONT_CANDIDATES = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                   r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simsun.ttc"]

DATA_DIR = K.TEST_DATA_DIR
SHOT_DIR = K.SHOT_DIR
SUMMARY_JSON = DATA_DIR / "评估汇总.json"
RESULTS_JSONL = DATA_DIR / "评估结果.jsonl"
ANALYSIS_MD = DATA_DIR / "检索问题分析.md"
REGRESSION_JSON = DATA_DIR / "回归结果.json"

# 配色
BG = (255, 255, 255)
INK = (28, 32, 38)
MUTED = (98, 106, 118)
BAR_BG = (26, 60, 110)
ACCENT = (0, 108, 190)
LINE = (206, 214, 224)
ZEBRA = (246, 249, 253)
OK_C = (0, 128, 64)
BAD_C = (190, 30, 30)
WARN_C = (196, 120, 0)
HEAD_FILL = (232, 240, 250)

WIDTH = 1400
MAX_H = 4200


def pick_font_path() -> str:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return path
    raise SystemExit("!! 未找到中文字体（msyh.ttc / simhei.ttf）")


FONT_PATH = pick_font_path()
_FONT_CACHE: dict[tuple[int, bool], Any] = {}


def font(size: int, bold: bool = False) -> Any:
    key = (size, bold)
    if key not in _FONT_CACHE:
        path = FONT_PATH
        if bold and Path(r"C:\Windows\Fonts\msyhbd.ttc").exists():
            path = r"C:\Windows\Fonts\msyhbd.ttc"
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


# --------------------------------------------------------------------------- 数据
def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"!! 读取失败 {path}: {type(exc).__name__}: {exc}")
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return rows


def pct(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def num(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "-" if value is None else str(value)


def yn(value: Any) -> str:
    return "是" if value else "否"


def flag(value: Any) -> bool:
    """把判分字段统一成布尔值。

    ``判分.py`` 的 ``page_hit`` 返回 ``{"hit": bool, ...}``、``cross_corpus_leak`` /
    ``citation_leak`` 返回 ``{"leak": bool, ...}``；直接取真值会把「命中=False 的字典」
    当成 True（非空字典恒为真）。这里显式解包，避免截图出现反向结论。
    """
    if isinstance(value, dict):
        for key in ("hit", "leak", "ok", "value", "result"):
            if key in value:
                return bool(value[key])
        return False
    return bool(value)


def short(text: Any, limit: int = 40) -> str:
    s = str(text or "").replace("\n", " ")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def md_section(md: str, heading_re: str, stop_re: str = r"\n#{2,3} ") -> str:
    """取出 Markdown 中匹配 ``heading_re`` 的段落正文。"""
    match = re.search(heading_re, md)
    if not match:
        return ""
    rest = md[match.end():]
    stop = re.search(stop_re, rest)
    return rest[: stop.start()] if stop else rest


def md_table_rows(section: str) -> list[list[str]]:
    """解析 Markdown 管道表（跳过表头分隔行）。"""
    rows: list[list[str]] = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    return rows


# --------------------------------------------------------------------------- 画布
class Page:
    """一页截图：统一页眉（含右上角工单标注）、表格、键值区、柱状图与页脚。"""

    def __init__(self, title: str, subtitle: str = "", *, width: int = WIDTH) -> None:
        self.W = width
        self.M = 46
        self.img = Image.new("RGB", (width, MAX_H), BG)
        self.d = ImageDraw.Draw(self.img)
        self.y = 0
        self._header(title, subtitle)

    # ---------------------------------------------------------------- 基础绘制
    def text(self, x: int, y: int, s: str, size: int = 18, *, fill: Any = INK,
             bold: bool = False) -> float:
        self.d.text((x, y), s, font=font(size, bold), fill=fill)
        return self.d.textlength(s, font=font(size, bold))

    def _header(self, title: str, subtitle: str) -> None:
        self.d.rectangle([0, 0, self.W, 96], fill=BAR_BG)
        self.text(self.M, 26, title, 30, fill=(255, 255, 255), bold=True)
        w = self.d.textlength(ANNOTATION, font=font(15))
        self.text(int(self.W - self.M - w), 38, ANNOTATION, 15, fill=(206, 222, 245))
        if subtitle:
            self.text(self.M, 64, subtitle, 15, fill=(178, 198, 226))
        self.y = 118

    def section(self, title: str) -> None:
        self.y += 16
        self.d.rectangle([self.M, self.y, self.M + 5, self.y + 24], fill=ACCENT)
        self.text(self.M + 14, self.y - 2, title, 21, bold=True)
        self.y += 36
        self.d.line([self.M, self.y - 8, self.W - self.M, self.y - 8], fill=LINE, width=1)

    def note(self, text: str, *, size: int = 15, fill: Any = MUTED) -> None:
        for line in self.wrap(text, self.W - 2 * self.M, size):
            self.text(self.M, self.y, line, size, fill=fill)
            self.y += size + 7
        self.y += 4

    def wrap(self, text: str, max_w: float, size: int) -> list[str]:
        f = font(size)
        lines, cur = [], ""
        for ch in str(text):
            if ch == "\n":
                lines.append(cur)
                cur = ""
                continue
            if self.d.textlength(cur + ch, font=f) <= max_w:
                cur += ch
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
        return lines or [""]

    def ellipsize(self, text: str, max_w: float, size: int) -> str:
        f = font(size)
        s = str(text or "")
        if self.d.textlength(s, font=f) <= max_w:
            return s
        while s and self.d.textlength(s + "…", font=f) > max_w:
            s = s[:-1]
        return s + "…"

    # ---------------------------------------------------------------- 键值区
    def kv(self, rows: Sequence[tuple[str, str]], *, per_row: int = 2, size: int = 17,
           colors: Sequence[Any] | None = None) -> None:
        col_w = (self.W - 2 * self.M) // per_row
        label_w = 168
        row_h = size + 16
        for idx, (label, value) in enumerate(rows):
            col = idx % per_row
            if col == 0 and idx:
                self.y += row_h
            x = self.M + col * col_w
            fill = (colors[idx] if colors and idx < len(colors) and colors[idx] else INK)
            self.d.rectangle([x, self.y + 2, x + col_w - 14, self.y + row_h - 4], fill=ZEBRA)
            self.text(x + 10, self.y + 8, label, size, fill=MUTED)
            self.text(x + label_w, self.y + 8, self.ellipsize(value, col_w - label_w - 26, size),
                      size, fill=fill, bold=True)
        self.y += row_h + 8

    # ---------------------------------------------------------------- 表格
    def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]], *,
              aligns: Sequence[str] | None = None, weights: Sequence[float] | None = None,
              size: int = 16, row_colors: Sequence[Any] | None = None) -> None:
        avail = self.W - 2 * self.M
        aligns = list(aligns or ["left"] * len(headers))
        n = len(headers)
        # 逐级缩字号，直到自然宽度塞得下
        fsize = size
        while True:
            fb = font(fsize, True)
            natural = []
            for i, head in enumerate(headers):
                w = self.d.textlength(str(head), font=fb)
                for row in rows:
                    cell = row[i] if i < len(row) else ""
                    w = max(w, self.d.textlength(str(cell), font=font(fsize)))
                natural.append(w + 22)
            if sum(natural) <= avail or fsize <= 12:
                break
            fsize -= 1
        if weights:
            total_w = sum(weights)
            natural = [avail * w / total_w for w in weights]
        scale = min(1.0, avail / max(1.0, sum(natural)))
        widths = [max(56.0, w * scale) for w in natural]
        over = sum(widths) - avail
        if over > 0:                       # 还差一点就从最宽的列里扣
            widest = widths.index(max(widths))
            widths[widest] -= over
        row_h = fsize + 16
        x0 = self.M
        # 表头
        self.d.rectangle([x0, self.y, x0 + sum(widths), self.y + row_h], fill=BAR_BG)
        x = x0
        for i, head in enumerate(headers):
            self.text(int(x + 10), self.y + 8, self.ellipsize(str(head), widths[i] - 18, fsize),
                      fsize, fill=(255, 255, 255), bold=True)
            x += widths[i]
        self.y += row_h
        # 数据行
        for r, row in enumerate(rows):
            fill = ZEBRA if r % 2 == 0 else BG
            if row_colors and r < len(row_colors) and row_colors[r]:
                fill = row_colors[r]
            self.d.rectangle([x0, self.y, x0 + sum(widths), self.y + row_h], fill=fill)
            self.d.line([x0, self.y, x0 + sum(widths), self.y], fill=LINE, width=1)
            x = x0
            for i in range(n):
                cell = str(row[i] if i < len(row) else "")
                text = self.ellipsize(cell, widths[i] - 18, fsize)
                tx = int(x + 10)
                if aligns[i] == "right":
                    tx = int(x + widths[i] - 10 - self.d.textlength(text, font=font(fsize)))
                elif aligns[i] == "center":
                    tx = int(x + (widths[i] - self.d.textlength(text, font=font(fsize))) / 2)
                color = INK
                if cell in ("是",) and i > 0:
                    color = OK_C
                self.text(tx, self.y + 8, text, fsize, fill=color)
                x += widths[i]
            self.y += row_h
        self.d.rectangle([x0, self.y, x0 + sum(widths), self.y], fill=LINE)
        self.y += 10

    # ---------------------------------------------------------------- 柱状图
    def bars(self, labels: Sequence[str], values: Sequence[float], *,
             threshold: float | None = None, height: int = 240, unit: str = "ms") -> None:
        avail = self.W - 2 * self.M
        top = self.y + 10
        bottom = top + height
        vmax = max(list(values) + ([threshold] if threshold else [])) * 1.12 or 1.0
        self.d.line([self.M, bottom, self.W - self.M, bottom], fill=LINE, width=2)
        n = max(1, len(values))
        slot = avail / n
        bar_w = max(12, int(slot * 0.62))
        for i, value in enumerate(values):
            bx = int(self.M + i * slot + (slot - bar_w) / 2)
            bh = int(height * float(value) / vmax)
            color = ACCENT if (threshold is None or float(value) <= threshold) else BAD_C
            self.d.rectangle([bx, bottom - bh, bx + bar_w, bottom], fill=color)
            self.text(bx - 4, bottom - bh - 18, f"{float(value):.0f}", 12, fill=MUTED)
            lb = str(labels[i])
            self.text(int(bx + bar_w / 2 - self.d.textlength(lb, font=font(12)) / 2),
                      bottom + 6, lb, 12, fill=MUTED)
        if threshold:
            ty = bottom - int(height * threshold / vmax)
            self.d.line([self.M, ty, self.W - self.M, ty], fill=BAD_C, width=2)
            self.text(self.W - self.M - 210, ty - 18, f"SLA 阈值 {threshold:.0f}{unit}", 13,
                      fill=BAD_C, bold=True)
        avg = sum(values) / len(values) if values else 0
        ay = bottom - int(height * avg / vmax)
        self.d.line([self.M, ay, self.W - self.M, ay], fill=WARN_C, width=2)
        self.text(self.M + 4, ay - 18, f"平均 {avg:.0f}{unit}", 13, fill=WARN_C, bold=True)
        self.y = bottom + 30

    # ---------------------------------------------------------------- 收尾
    def finish(self, sources: Sequence[str], *, extra_note: str = "") -> Image.Image:
        self.y += 14
        self.d.line([self.M, self.y, self.W - self.M, self.y], fill=LINE, width=1)
        self.y += 10
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"数据来源：{s}" for s in sources]
        lines.append(f"渲染时间：{stamp}｜渲染器：测试/生成截图.py（所有数字取自真实产物文件）")
        if extra_note:
            lines.append(extra_note)
        for line in lines:
            for piece in self.wrap(line, self.W - 2 * self.M, 13):
                self.text(self.M, self.y, piece, 13, fill=MUTED)
                self.y += 18
        h = max(900, self.y + 12)
        return self.img.crop((0, 0, self.W, min(h, MAX_H)))


# --------------------------------------------------------------------------- 各图
def _config_rows(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        ("模型", str(cfg.get("模型"))),
        ("后端", str(cfg.get("backend"))),
        ("嵌入模型", str(cfg.get("embed_model"))),
        ("top_k", str(cfg.get("top_k"))),
        ("use_cache", str(cfg.get("use_cache"))),
        ("索引块 total", str(cfg.get("索引chunk_total"))),
        ("关键词阈值", str(cfg.get("关键词阈值"))),
        ("部分正确阈值", str(cfg.get("部分正确阈值"))),
        ("索引 run_id", str(cfg.get("索引run_id"))),
        ("SLA_ms", str(cfg.get("SLA_ms"))),
    ]


def render_summary(out: list[Path]) -> None:
    data = load_json(SUMMARY_JSON)
    if not data:
        print(f"!! 跳过 整体评估汇总.png（缺 {SUMMARY_JSON}）")
        return
    acc = data.get("准确率") or {}
    rec = data.get("检索召回") or {}
    rt = data.get("响应时间ms") or {}
    sla = data.get("SLA") or {}
    ref = data.get("拒答") or {}
    page = data.get("页命中") or {}
    p = Page("整体评估汇总（17 题功能评估）",
             f"运行时间 {data.get('生成时间')}｜有效题 {data.get('有效题数')}/{data.get('总数')}｜"
             f"异常题 {data.get('异常题数')}")
    p.section("一、运行配置（评估汇总.json → 配置 / 运行元信息）")
    p.kv(_config_rows(data.get("配置") or {}), per_row=2)
    p.section("二、核心指标")
    p.kv([
        ("题目总数", str(data.get("总数"))),
        ("有效题数", str(data.get("有效题数"))),
        ("异常题数", str(data.get("异常题数"))),
        ("严格准确率", f"{pct(acc.get('严格准确率'))}（{acc.get('严格判对题数')}/{data.get('有效题数')}）"),
        ("宽松准确率", f"{pct(acc.get('宽松准确率(严格+部分)'))}（严格+部分）"),
        ("平均关键词覆盖率", pct(acc.get("平均关键词覆盖率"))),
        ("平均 recall@5", pct(rec.get("平均recall@5"))),
        ("平均 recall@10", pct(rec.get("平均recall@10"))),
        ("SLA 达标率", f"{pct(sla.get('达标率'))}（{sla.get('达标题数')}/{data.get('有效题数')}，≤{sla.get('阈值ms')}ms）"),
        ("首字达标率", pct(sla.get("首字达标率"))),
        ("平均响应时间", f"{num(rt.get('平均'))} ms"),
        ("P90 响应时间", f"{num(rt.get('P90'))} ms"),
        ("最大响应时间", f"{num(rt.get('最大'))} ms"),
        ("拒答率", f"{pct(ref.get('拒答率'))}（{ref.get('拒答题数')} 题：{','.join(ref.get('拒答题号') or []) or '无'}）"),
        ("页命中率", f"{pct(page.get('命中率'))}（未命中：{','.join(page.get('未命中题号') or []) or '无'}）"),
        ("跨语料泄漏", f"top10 {data.get('跨语料泄漏', {}).get('top10题数')} 题／引用级 "
                    f"{data.get('跨语料泄漏', {}).get('引用级泄漏题数')} 题"),
    ], per_row=2)
    p.section("三、按题型分组（评估汇总.json → 按题型分组）")
    rows = []
    for name, item in (data.get("按题型分组") or {}).items():
        rows.append([name, item.get("题数"), item.get("判对题数"), pct(item.get("判对率")),
                     pct(item.get("平均关键词覆盖率")), pct(item.get("平均recall@5")),
                     pct(item.get("平均recall@10")), num(item.get("平均总耗时ms"), 0),
                     item.get("拒答题数")])
    p.table(["题型", "题数", "判对", "判对率", "平均覆盖率", "recall@5", "recall@10",
             "平均耗时ms", "拒答"], rows,
            aligns=["left", "right", "right", "right", "right", "right", "right", "right", "right"])
    p.note(f"判对题号：{'、'.join(acc.get('判对题号') or [])}｜部分正确："
           f"{'、'.join(acc.get('部分正确题号') or []) or '无'}｜"
           f"未判对 {len(acc.get('未判对题号') or [])} 题：{'、'.join(acc.get('未判对题号') or [])}")
    img = p.finish([str(SUMMARY_JSON.relative_to(K.REPO_ROOT)),
                    f"检索问题分析.md（逐题标签）"])
    path = SHOT_DIR / "整体评估汇总.png"
    img.save(path)
    out.append(path)


def render_latency(out: list[Path]) -> None:
    data = load_json(SUMMARY_JSON)
    if not data:
        print(f"!! 跳过 响应时间测试.png（缺 {SUMMARY_JSON}）")
        return
    rt = data.get("响应时间ms") or {}
    sla = data.get("SLA") or {}
    brief = data.get("逐题速览") or []
    p = Page("响应时间测试（端到端 ≤3000ms 验收）",
             f"样本 {rt.get('样本数')} 题｜配置 use_cache={ (data.get('配置') or {}).get('use_cache') }｜"
             f"模型 {(data.get('配置') or {}).get('模型')}")
    p.section("一、分段耗时统计（评估汇总.json → 响应时间ms）")
    p.table(["指标", "平均ms", "P90ms", "最大ms", "最小ms"],
            [["端到端 total", num(rt.get("平均")), num(rt.get("P90")), num(rt.get("最大")), num(rt.get("最小"))],
             ["检索 retrieve", num(rt.get("检索_平均")), num(rt.get("检索_P90")), "-", "-"],
             ["生成 generate", num(rt.get("生成_平均")), num(rt.get("生成_P90")), "-", "-"],
             ["首字 first_token", num(rt.get("首字_平均")), num(rt.get("首字_P90")), num(rt.get("首字_最大")), "-"]],
            aligns=["left", "right", "right", "right", "right"])
    p.section("二、SLA 判定（阈值取自 配置.SLA_ms）")
    p.kv([("SLA 阈值", f"{sla.get('阈值ms')} ms"), ("达标题数", f"{sla.get('达标题数')} 题"),
          ("达标率", pct(sla.get("达标率"))),
          ("超时题号", "、".join(sla.get("超时题号") or []) or "无"),
          ("首字达标率", pct(sla.get("首字达标率"))),
          ("中位/P50", f"{num(rt.get('中位'))} ms")], per_row=3)
    p.section("三、逐题端到端耗时（评估汇总.json → 逐题速览，红线为 3000ms SLA）")
    labels = [str(b.get("题号")) for b in brief]
    values = [float(b.get("总耗时ms") or 0) for b in brief]
    p.bars(labels, values, threshold=float(sla.get("阈值ms") or 3000), height=250)
    p.section("四、逐题明细")
    rows = [[b.get("题号"), b.get("题型"), b.get("公司"), num(b.get("总耗时ms"), 2),
             yn(b.get("页命中")), "是" if b.get("拒答") else "否",
             str(b.get("生成方式"))] for b in brief]
    p.table(["题号", "题型", "公司", "总耗时ms", "页命中", "拒答", "生成方式"], rows,
            aligns=["left", "left", "left", "right", "center", "center", "left"])
    p.note("说明：单机 Ollama 仅 1 个 CPU 推理实例，与其他任务争抢算力时耗时会整体抬升；"
           "因此本页所有数字必须连同运行时间与并发状态一起解读（详见 检索问题分析.md §3.5）。")
    img = p.finish([str(SUMMARY_JSON.relative_to(K.REPO_ROOT))])
    path = SHOT_DIR / "响应时间测试.png"
    img.save(path)
    out.append(path)


def render_accuracy(out: list[Path]) -> None:
    data = load_json(SUMMARY_JSON)
    if not data:
        print(f"!! 跳过 准确率汇总.png（缺 {SUMMARY_JSON}）")
        return
    acc = data.get("准确率") or {}
    p = Page("准确率汇总（严格判对：关键词覆盖率 ≥0.6 且未拒答）",
             f"有效题 {data.get('有效题数')}｜严格判对 {acc.get('严格判对题数')} 题｜"
             f"宽松（严格+部分）{acc.get('部分正确题数', 0)} 题部分正确")
    p.section("一、总览")
    p.kv([
        ("严格准确率", f"{pct(acc.get('严格准确率'))}（{acc.get('严格判对题数')}/{data.get('有效题数')}）"),
        ("宽松准确率", pct(acc.get("宽松准确率(严格+部分)"))),
        ("平均关键词覆盖率", pct(acc.get("平均关键词覆盖率"))),
        ("判对题号", "、".join(acc.get("判对题号") or []) or "无"),
        ("部分正确题号", "、".join(acc.get("部分正确题号") or []) or "无"),
        ("未判对题数", str(len(acc.get("未判对题号") or []))),
    ], per_row=2)
    p.section("二、按题型分组")
    rows = [[name, it.get("题数"), it.get("判对题数"), pct(it.get("判对率")),
             pct(it.get("平均关键词覆盖率")), pct(it.get("平均recall@5"))]
            for name, it in (data.get("按题型分组") or {}).items()]
    p.table(["题型", "题数", "判对", "判对率", "平均覆盖率", "平均recall@5"], rows,
            aligns=["left", "right", "right", "right", "right", "right"])
    p.section("三、按公司分组")
    rows = [[name, it.get("题数"), it.get("判对题数"), pct(it.get("判对率")),
             pct(it.get("平均关键词覆盖率")), pct(it.get("平均recall@10"))]
            for name, it in (data.get("按公司分组") or {}).items()]
    p.table(["公司", "题数", "判对", "判对率", "平均覆盖率", "平均recall@10"], rows,
            aligns=["left", "right", "right", "right", "right", "right"])
    p.section("四、按生成方式分组（区分「答错」与「答对但被护栏回退丢掉」）")
    rows = [[name, it.get("题数"), it.get("判对题数"), pct(it.get("判对率")),
             pct(it.get("平均关键词覆盖率")), num(it.get("平均总耗时ms"), 0)]
            for name, it in (data.get("按生成方式分组") or {}).items()]
    p.table(["生成方式", "题数", "判对", "判对率", "平均覆盖率", "平均耗时ms"], rows,
            aligns=["left", "right", "right", "right", "right", "right"])
    counter = data.get("LLM候选对照口径(反事实诊断)") or {}
    p.note("反事实诊断口径（" + str(counter.get("说明", ""))[:110] + "…）：候选判对 "
           f"{counter.get('候选判对题数')} 题（{pct(counter.get('候选口径准确率'))}），"
           f"被护栏丢弃的判对题 {counter.get('被护栏丢弃的判对题数')} 题"
           f"（{'、'.join(counter.get('被护栏丢弃的判对题号') or []) or '无'}）。")
    img = p.finish([str(SUMMARY_JSON.relative_to(K.REPO_ROOT))])
    path = SHOT_DIR / "准确率汇总.png"
    img.save(path)
    out.append(path)


def render_retrieval_issues(out: list[Path]) -> None:
    data = load_json(SUMMARY_JSON) or {}
    md = ANALYSIS_MD.read_text(encoding="utf-8") if ANALYSIS_MD.exists() else ""
    if not md and not data:
        print("!! 跳过 检索问题分析.png（缺 检索问题分析.md 与 评估汇总.json）")
        return
    reach = data.get("索引可达性") or {}
    p = Page("检索问题分析（索引可达性 / TOP5 问题 / 跨语料泄漏）",
             f"分析文件 {ANALYSIS_MD.name}｜索引 run_id={(data.get('配置') or {}).get('索引run_id')}｜"
             f"chunk_total={(data.get('配置') or {}).get('索引chunk_total')}")
    p.section("一、索引可达性（评估汇总.json → 索引可达性）")
    p.kv([
        ("关键词总数", str(reach.get("关键词总数"))),
        ("索引内可达", f"{reach.get('索引内可达')}（单块可达率 {pct(reach.get('单块可达率'))}）"),
        ("单块内不可达", f"{reach.get('单块内不可达')} 个"),
        ("其中切块边界切断", str(reach.get("其中属切块边界切断"))),
        ("其中疑似解析丢失", str(reach.get("其中疑似解析阶段即丢失"))),
        ("索引可达且落在期望页", str(reach.get("索引内可达且落在期望页"))),
        ("可达但 top10 未召回", f"{reach.get('索引可达但top10未召回关键词数')} 个关键词"),
        ("泄漏题（top10）", "、".join(((data.get('跨语料泄漏') or {}).get('top10题号') or [])) or "无"),
    ], per_row=2)
    if reach.get("索引可达但top10未召回明细"):
        detail = reach["索引可达但top10未召回明细"]
        rows = [[qid, "、".join(kws[:4])] for qid, kws in detail.items()]
        p.note("索引里明明存在、却未进 top-10 的关键词（按题号）：")
        p.table(["题号", "未召回关键词（最多列 4 个）"], rows, aligns=["left", "left"],
                weights=[0.08, 0.92])
    p.section("二、检索问题 TOP5（检索问题分析.md → 四、检索问题 TOP5）")
    top5 = md_table_rows(md_section(md, r"##\s*四、检索问题\s*TOP5", r"\n##\s")) if md else []
    if top5:
        body = top5[1:] if len(top5) > 1 else top5
        # 逐条**整段换行**渲染（不截断证据文本），保证每个字都看得全
        for row in body:
            rank = row[0] if row else ""
            problem = row[1] if len(row) > 1 else ""
            impact = row[2] if len(row) > 2 else ""
            evidence = row[3] if len(row) > 3 else ""
            self_y = p.y
            p.d.rectangle([p.M, self_y, p.M + 34, self_y + 26], fill=ACCENT)
            p.text(p.M + 8, self_y + 3, str(rank), 16, fill=(255, 255, 255), bold=True)
            p.text(p.M + 46, self_y + 2, f"{problem}（影响：{impact}）", 17, bold=True)
            p.y += 32
            for line in p.wrap(f"证据：{evidence}", p.W - 2 * p.M - 46, 15):
                p.text(p.M + 46, p.y, line, 15, fill=MUTED)
                p.y += 21
            p.y += 8
    else:
        p.note("（未在 检索问题分析.md 中解析到 TOP5 表）")
    if md:
        p.section("三、关键结论摘录（原文逐字，仅截断）")
        for key in ("### 3.4 表格依赖与数字表述污染", "### 3.5 稳定性与响应时间"):
            body = md_section(md, re.escape(key), r"\n#{2,4} ")
            if not body:
                continue
            p.note(key.replace("### ", ""), size=18, fill=INK)
            for line in [ln.strip() for ln in body.splitlines() if ln.strip().startswith("-")][:4]:
                p.note("  " + short(line.lstrip("- ").strip(), 200), size=15)
    img = p.finish([str(ANALYSIS_MD.relative_to(K.REPO_ROOT)),
                    str(SUMMARY_JSON.relative_to(K.REPO_ROOT))])
    path = SHOT_DIR / "检索问题分析.png"
    img.save(path)
    out.append(path)


def _regression_group(name: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    data = load_json(REGRESSION_JSON)
    if not data:
        return None, None
    for group in data.get("分组") or []:
        if group.get("分组") == name:
            return group, data
    return None, data


def render_regression_group(out: list[Path], group_name: str, filename: str,
                            title: str, columns: Sequence[str]) -> None:
    group, data = _regression_group(group_name)
    if not group:
        print(f"!! 跳过 {filename}（{REGRESSION_JSON.name} 中无「{group_name}」分组或文件缺失）")
        return
    assert data is not None
    p = Page(title, f"回归时间 {data.get('生成时间')}｜用例 {group.get('通过')}/{group.get('用例数')} 通过｜"
                    f"本组耗时 {num(float(group.get('耗时ms', 0)) / 1000, 1)}s")
    p.section("一、汇总")
    p.kv([("用例数", str(group.get("用例数"))), ("通过", str(group.get("通过"))),
          ("失败", str(group.get("失败"))), ("通过率", pct(group.get("通过率"))),
          ("耗时", f"{num(float(group.get('耗时ms', 0)) / 1000, 1)} s"),
          ("引擎", "QAEngine" + (" + MultiTurnEngine 对照" if group_name == "多轮对话" else ""))],
         per_row=3)
    p.section("二、逐用例结果（回归结果.json → 分组.用例）")
    rows = []
    colors = []
    for case in group.get("用例") or []:
        row = [case.get("用例编号"), short(case.get("用例名称"), 22)]
        for col in columns:
            row.append(_cell_for(case, col))
        row.append("PASS" if case.get("通过") else "FAIL")
        rows.append(row)
        colors.append(None if case.get("通过") else (255, 238, 238))
    p.table(["用例编号", "用例名称", *columns, "结论"], rows,
            aligns=["left", "left", *(["right"] * len(columns)), "center"],
            row_colors=colors)
    fails = [c for c in group.get("用例") or [] if not c.get("通过")]
    if fails:
        p.section("三、失败断言点名（不隐藏、不降级）")
        rows = []
        for case in fails:
            for item in case.get("断言") or []:
                if not item["通过"]:
                    rows.append([case.get("用例编号"), item["名称"], short(item["实测"], 48)])
        p.table(["用例编号", "失败断言", "实测"], rows, aligns=["left", "left", "left"],
                weights=[0.12, 0.3, 0.58])
    img = p.finish([str(REGRESSION_JSON.relative_to(K.REPO_ROOT))])
    path = SHOT_DIR / filename
    img.save(path)
    out.append(path)


def _company_from_corpus(name: str) -> str:
    """从语料文件名取公司简称：``...__601318__中国平安__2019年__...`` → ``中国平安``。"""
    parts = str(name or "").split("__")
    return parts[3] if len(parts) >= 4 and parts[3] else ""


def _cell_for(case: dict[str, Any], column: str) -> str:
    obs = case.get("关键实测") or {}
    corpora = case.get("引用语料") or []
    # 公司列：优先用例自身记录的「公司」，缺失时由**实际引用语料文件名**派生（不臆造）
    company = case.get("公司") or "、".join(
        dict.fromkeys(c for c in (_company_from_corpus(x) for x in corpora) if c))
    mapping: dict[str, Any] = {
        "题号": case.get("题库题号") or "",
        "公司": company,
        "输入长度": case.get("输入长度", ""),
        "答案字数": case.get("答案字数", ""),
        "引用页": ",".join(str(x) for x in (case.get("引用页") or [])) or "无",
        "引用语料": short((case.get("引用语料") or ["无"])[0], 24),
        "耗时ms": num(case.get("耗时ms"), 0),
        "拒答": "是" if case.get("拒答") else "否",
        "覆盖率": pct(obs.get("关键词覆盖率")) if obs.get("关键词覆盖率") is not None else "-",
        "判对": yn(obs.get("是否判对(覆盖率>=0.6且未拒答)")) if obs.get("是否判对(覆盖率>=0.6且未拒答)") is not None else "-",
        "生成方式": str(case.get("生成方式") or "-"),
        "改写策略": str(case.get("改写策略") or "-"),
        "焦点实体": str(case.get("焦点实体") or "-"),
        "指代已消解": yn(case.get("指代已消解")) if case.get("指代已消解") is not None else "-",
        "输入": short(case.get("输入"), 34),
        "拒答原因": str(case.get("拒答原因") or "-"),
    }
    return str(mapping.get(column, ""))


def render_overview(out: list[Path]) -> None:
    data = load_json(REGRESSION_JSON)
    if not data:
        print(f"!! 跳过 回归测试总览.png（缺 {REGRESSION_JSON}）")
        return
    summary = data.get("汇总") or {}
    p = Page("回归测试总览（三组串行、单引擎共享、真实执行）",
             f"运行时间 {data.get('生成时间')}｜结论 {summary.get('结论')}｜"
             f"用例 {summary.get('通过')}/{summary.get('用例总数')} 通过")
    p.section("一、环境事实")
    env = data.get("环境") or {}
    p.kv([("解释器", str(env.get("解释器"))), ("版本", str(env.get("解释器版本"))),
          ("工作目录", str(env.get("工作目录"))),
          ("索引", f"块数 {(data.get('索引') or {}).get('chunk_total')}／run_id "
                 f"{(data.get('索引') or {}).get('run_id')}"),
          ("嵌入模型", str((data.get('索引') or {}).get('embed_model'))),
          ("日志", str(data.get("日志")))], per_row=2)
    p.section("二、分组汇总（回归结果.json → 汇总）")
    rows = [[g.get("分组"), g.get("用例数"), g.get("通过"), g.get("失败"), pct(g.get("通过率")),
             num(float(g.get("耗时ms", 0)) / 1000, 1)] for g in data.get("分组") or []]
    rows.append(["合计", summary.get("用例总数"), summary.get("通过"), summary.get("失败"),
                 pct(summary.get("通过率")), num(float(summary.get("总耗时ms", 0)) / 1000, 1)])
    p.table(["用例集", "用例数", "通过", "失败", "通过率", "耗时(s)"], rows,
            aligns=["left", "right", "right", "right", "right", "right"])
    p.section("三、预热与串行纪律")
    warm = data.get("预热") or {}
    # 兼容两种形状：单轮/异常模块写的是引擎 warmup 结果本身；多轮模块写成 {"base": …, "multiturn": …}
    base = warm.get("base") or warm
    ft = base.get("fulltext_index")
    if isinstance(ft, dict) and (ft.get("fulltext_index") or ft.get("ok") is not None):
        inner = ft.get("fulltext_index") if isinstance(ft.get("fulltext_index"), dict) else ft
        ft_txt = f"ok（chunks={inner.get('chunks')}）" if inner.get("chunks") is not None else str(ft)[:60]
    else:
        ft_txt = str(ft)[:60]
    gen = base.get("generate")
    gen_txt = "ok" if isinstance(gen, dict) and gen.get("ok") else str(gen)[:70]
    p.kv([("索引块数", str(base.get("index_chunks"))),
          ("预热总耗时", f"{num(base.get('elapsed_ms'), 0)} ms"),
          ("全文倒排预热", ft_txt),
          ("生成后端预热", gen_txt),
          ("多轮引擎", str(warm.get("multiturn") or base.get("multiturn"))),
          ("致命错误", str(data.get("致命错误") or "无"))], per_row=2)
    img = p.finish([str(REGRESSION_JSON.relative_to(K.REPO_ROOT))])
    path = SHOT_DIR / "回归测试总览.png"
    img.save(path)
    out.append(path)


def render_question1(out: list[Path]) -> None:
    rows = load_jsonl(RESULTS_JSONL)
    row = next((r for r in rows if r.get("id") == "Q01"), None)
    if not row:
        print(f"!! 跳过 问题1_检索结果.png（缺 {RESULTS_JSONL} 或 Q01）")
        return
    hits = row.get("hits") or []
    judge = row.get("judge") or {}
    p = Page("问题1 检索结果（Q01 · 平安银行董事长致辞 · top-10 命中）",
             f"题型 {row.get('question_type')}｜公司 {row.get('company')}｜"
             f"top_k={row.get('top_k')}｜trace_id={row.get('trace_id')}")
    p.section("一、题面")
    p.note(str(row.get("question")), size=17, fill=INK)
    p.section("二、检索命中 top-10（评估结果.jsonl → hits）")
    table_rows = []
    for hit in hits[:10]:
        table_rows.append([
            hit.get("rank"), hit.get("chunk_id"), f"p{hit.get('page')}",
            hit.get("modality") or hit.get("type"), short(hit.get("company"), 10),
            num(hit.get("final_score"), 4), hit.get("bm25_rank"), hit.get("vector_rank"),
            short(hit.get("text_head"), 30),
        ])
    p.table(["位次", "chunk_id", "页", "模态", "公司", "final_score", "BM25秩", "向量秩", "片段开头"],
            table_rows, aligns=["right", "left", "right", "center", "left", "right", "right",
                                "right", "left"],
            weights=[0.05, 0.16, 0.05, 0.07, 0.09, 0.11, 0.07, 0.07, 0.33])
    p.section("三、判分与耗时（评估结果.jsonl → judge / latency_ms / gate）")
    p.kv([
        ("关键词覆盖率", f"{pct(judge.get('keyword_coverage'))}（{judge.get('keyword_hit_count')}/{judge.get('keyword_total')}）"),
        ("是否判对", yn(judge.get("is_correct"))),
        ("部分正确", yn(judge.get("partial_correct"))),
        ("recall@5", pct(judge.get("recall_at_5"))),
        ("recall@10", pct(judge.get("recall_at_10"))),
        ("页命中", yn(flag(judge.get("page_hit")))),
        ("引用页", str(row.get("cited_pages"))),
        ("命中块数", str(row.get("hit_count"))),
        ("检索耗时", f"{num((row.get('latency_ms') or {}).get('retrieve'))} ms"),
        ("生成耗时", f"{num((row.get('latency_ms') or {}).get('generate'))} ms"),
        ("首字耗时", f"{num((row.get('latency_ms') or {}).get('first_token'))} ms"),
        ("端到端", f"{num(row.get('total_ms'))} ms（SLA {'达标' if row.get('sla_ok') else '超时'}）"),
        ("生成方式", str(row.get("generation_method"))),
        ("可答性闸门", f"{yn((row.get('gate') or {}).get('answerable'))}（{(row.get('gate') or {}).get('reason')}）"),
        ("引用语料", short((row.get("citations") or [{}])[0].get("corpus"), 44)),
        ("跨语料泄漏", yn(flag(judge.get("cross_corpus_leak")))),
    ], per_row=3)
    p.section("四、未命中关键词（judge.keyword_miss）")
    p.note("、".join(judge.get("keyword_miss") or []) or "无", size=16, fill=BAD_C)
    p.section("五、答案开头（评估结果.jsonl → answer，截断 260 字）")
    p.note(short(row.get("answer"), 260), size=16)
    img = p.finish([str(RESULTS_JSONL.relative_to(K.REPO_ROOT)),
                    str(SUMMARY_JSON.relative_to(K.REPO_ROOT))])
    path = SHOT_DIR / "问题1_检索结果.png"
    img.save(path)
    out.append(path)


def render_question2(out: list[Path]) -> None:
    rows = load_jsonl(RESULTS_JSONL)
    row = next((r for r in rows if r.get("id") == "Q02"), None)
    if not row:
        print(f"!! 跳过 问题2_评估结果.png（缺 {RESULTS_JSONL} 或 Q02）")
        return
    judge = row.get("judge") or {}
    gen = row.get("generation") or {}
    reach = row.get("index_reach_summary") or {}
    md = ANALYSIS_MD.read_text(encoding="utf-8") if ANALYSIS_MD.exists() else ""
    block = md_section(md, r"###\s*Q02", r"\n###\s") if md else ""
    tags = ""
    for line in block.splitlines():
        if "问题标签" in line:
            tags = line.split("**问题标签**：")[-1].strip()
            break
    p = Page("问题2 评估结果（Q02 · 招商银行创新商业模式 · 判分明细）",
             f"题型 {row.get('question_type')}｜公司 {row.get('company')}｜"
             f"判对 {yn(judge.get('is_correct'))}｜覆盖率 {pct(judge.get('keyword_coverage'))}")
    p.section("一、题面")
    p.note(str(row.get("question")), size=17, fill=INK)
    p.section("二、判分结果（评估结果.jsonl → judge）")
    p.kv([
        ("覆盖率", f"{pct(judge.get('keyword_coverage'))}（{judge.get('keyword_hit_count')}/{judge.get('keyword_total')}）"),
        ("阈值", str(judge.get("threshold"))),
        ("是否判对", yn(judge.get("is_correct"))),
        ("部分正确", yn(judge.get("partial_correct"))),
        ("recall@5", pct(judge.get("recall_at_5"))),
        ("recall@10", pct(judge.get("recall_at_10"))),
        ("页命中", f"{yn(flag(judge.get('page_hit')))}（期望页 {row.get('expected_pages_hint')}）"),
        ("跨语料泄漏", yn(flag(judge.get("cross_corpus_leak")))),
        ("引用级泄漏", yn(flag(judge.get("citation_leak")))),
        ("拒答", f"{yn(row.get('refused'))}（{row.get('refusal_reason')}）"),
    ], per_row=3)
    p.section("三、期望关键词核对")
    p.note("命中：" + ("、".join(judge.get("keyword_hit") or []) or "无"), size=16, fill=OK_C)
    p.note("缺失：" + ("、".join(judge.get("keyword_miss") or []) or "无"), size=16, fill=BAD_C)
    p.section("四、检索侧证据（评估结果.jsonl → index_reach_summary / generation）")
    p.kv([
        ("关键词总数", str(reach.get("关键词总数"))),
        ("索引内可达", str(reach.get("索引内可达"))),
        ("单块内不可达", str(reach.get("单块内不可达"))),
        ("切块边界切断", "、".join(reach.get("其中属切块边界切断") or []) or "无"),
        ("可达且落期望页", str(reach.get("索引内可达且落在期望页"))),
        ("生成方式", str(gen.get("method") or row.get("generation_method"))),
        ("返回字数", str(gen.get("returned_chars"))),
        ("生成耗时", f"{num(gen.get('total_ms'))} ms"),
        ("LLM 候选", short(row.get("llm_candidate"), 56) or "无"),
        ("候选被拒原因", str(row.get("llm_rejected_reason") or "无")),
    ], per_row=2)
    if row.get("missing_atoms"):
        p.note("护栏判定缺失原子（导致 LLM 候选被丢弃）："
               + "、".join(short(a, 18) for a in (row.get("missing_atoms") or [])[:6]), size=15)
    p.section("五、检索问题分析.md 的 Q02 结论（逐字摘录，截断）")
    if tags:
        p.note(f"问题标签：{tags}", size=16, fill=WARN_C)
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("- top10 合并正文仍缺") or line.startswith("- 单块都没有"):
            p.note(short(line.lstrip("- "), 180), size=15)
    p.section("六、答案（评估结果.jsonl → answer，截断 300 字）")
    p.note(short(row.get("answer"), 300), size=16)
    img = p.finish([str(RESULTS_JSONL.relative_to(K.REPO_ROOT)),
                    str(ANALYSIS_MD.relative_to(K.REPO_ROOT))])
    path = SHOT_DIR / "问题2_评估结果.png"
    img.save(path)
    out.append(path)


# --------------------------------------------------------------------------- 主流程
JOBS = {
    "整体评估汇总": (render_summary, "整体评估汇总.png"),
    "响应时间测试": (render_latency, "响应时间测试.png"),
    "准确率汇总": (render_accuracy, "准确率汇总.png"),
    "检索问题分析": (render_retrieval_issues, "检索问题分析.png"),
    "回归测试总览": (render_overview, "回归测试总览.png"),
    "单轮问答回归": (lambda out: render_regression_group(
        out, "单轮问答", "单轮问答回归.png", "单轮问答回归（真实执行 · QAEngine.answer）",
        ["题号", "公司", "答案字数", "引用页", "耗时ms", "覆盖率", "判对", "生成方式"]), None),
    "多轮对话回归": (lambda out: render_regression_group(
        out, "多轮对话", "多轮对话回归.png", "多轮对话回归（指代消解 · 三组对照 · 真实执行）",
        ["引用语料", "引用页", "耗时ms", "改写策略", "焦点实体", "指代已消解", "拒答"]), None),
    "异常输入测试": (lambda out: render_regression_group(
        out, "异常输入", "异常输入测试.png", "异常输入测试（空/空格/符号/乱码/超长/注入/英文）",
        ["输入", "输入长度", "拒答", "拒答原因", "引用页", "耗时ms"]), None),
    "问题1_检索结果": (render_question1, "问题1_检索结果.png"),
    "问题2_评估结果": (render_question2, "问题2_评估结果.png"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="工单7 测试截图渲染器（真实数据）")
    parser.add_argument("--only", action="append", default=None,
                        help="只渲染指定图（可重复；取值见 JOBS 的键）")
    parser.add_argument("--list", action="store_true", help="列出可渲染的图名")
    args = parser.parse_args(argv)

    if args.list:
        for name, (_fn, filename) in JOBS.items():
            print(f"{name}  ->  {filename or '(分组渲染)'}")
        return 0

    K.bootstrap_env()
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== {K.WORK_ORDER} / 测试截图渲染 ==")
    print(f"字体：{FONT_PATH}")
    print(f"输出目录：{SHOT_DIR}")
    print(f"数据源：{SUMMARY_JSON}｜{RESULTS_JSONL}｜{ANALYSIS_MD}｜{REGRESSION_JSON}")
    for path in (SUMMARY_JSON, RESULTS_JSONL, ANALYSIS_MD, REGRESSION_JSON):
        print(f"  [{'OK ' if path.exists() else 'MISS'}] {path.name} "
              f"({path.stat().st_size if path.exists() else 0} 字节)")

    names = args.only or list(JOBS)
    produced: list[Path] = []
    for name in names:
        job = JOBS.get(name)
        if not job:
            print(f"!! 未知图名：{name}")
            continue
        fn, filename = job
        try:
            before = len(produced)
            fn(produced)
            if filename and len(produced) > before:
                print(f"  已渲染 {filename}")
        except Exception as exc:  # noqa: BLE001 —— 单图失败不影响其他图
            import traceback

            print(f"!! 渲染 {name} 失败：{type(exc).__name__}: {exc}")
            print(traceback.format_exc()[-1200:])

    print(f"\n共渲染 {len(produced)} 张：")
    for path in produced:
        from PIL import Image as _I

        with _I.open(path) as im:
            print(f"  {path.name}｜{im.width}×{im.height}｜{path.stat().st_size / 1024:.0f} KB")
    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
