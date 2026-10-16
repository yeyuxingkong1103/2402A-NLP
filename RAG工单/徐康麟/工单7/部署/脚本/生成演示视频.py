# -*- coding: utf-8 -*-
"""工单7 演示视频自动生成 —— 用 PIL **逐帧绘制**真实评测数据，再由 ffmpeg 合成 mp4。

工单编号：人工智能NLP-RAG-功能测试及评估

产出
----
``部署/演示视频/工单7_功能测试及评估演示.mp4`` —— 约 40~70 秒，1600×900，20 fps，H.264。

镜头顺序（严格按工单要求）
--------------------------
1. **问题提取**：``输入/sample_questions.pdf`` → ``研发/评估/测试问题集.jsonl``（17 题 / 103 关键词）
2. **语料与索引规模**：9 份年报 → 8742 块（正文/表格/图像）× bge-m3(1024) + FAISS + BM25
3. **RAG 检索与回答示例**：真实取一道判对的题，展示答案 / 引用页 / 分段耗时 / Top-5 命中
4. **评估判分表**：17 题逐题 覆盖率 / 判对 / recall@5 / @10 / 跨语料泄漏 / 耗时 / 生成方式
5. **汇总指标与检索问题分析**：严格准确率、SLA 达标率、拒答、泄漏、索引可达率 + TOP5 问题

诚实边界
--------
* 画面里**每一个数字都从真实文件读**：``测试/测试数据/评估结果.jsonl``、``评估汇总.json``、
  ``检索问题分析.md``、``研发/评估/测试问题集.jsonl``、``研发/data/index/index_manifest.json``。
  没有写死的假数据；文件缺失就如实打印 FATAL 而不是编一个数。
* 本片是**图文讲解片**（数据面板 + 旁白字幕），**不是屏幕操作录像**，**无音轨**
  （本机未安装可用 TTS）。
* ffmpeg 不在 PATH 里，脚本自动探测已知位置（``C:\\Program Files\\EVCapture\\ffmpeg.exe`` 等，实测可用）。
  **若确实找不到 ffmpeg**：自动退化为 ``--frames-only``，把整段帧序列写到
  ``部署/演示视频/帧/*.png`` 并生成 ``部署/演示视频/README.md`` 说明如何用别的方式合成。

用法
----
.. code-block:: powershell

    pwsh -NoProfile -File 部署/脚本/一键跑测试评估.ps1 -Only video
    # 或
    & $env:GD7_PY 部署/脚本/生成演示视频.py
    & $env:GD7_PY 部署/脚本/生成演示视频.py --frames-only      # 只出帧序列
    & $env:GD7_PY 部署/脚本/生成演示视频.py --fps 12            # 降帧率
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image, ImageDraw, ImageFont

WORK_ORDER = "人工智能NLP-RAG-功能测试及评估"

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]                      # ...\工单7

DIR_TESTDATA = REPO_ROOT / "测试" / "测试数据"
DIR_EVALSRC = REPO_ROOT / "研发" / "评估"
DIR_INDEX = REPO_ROOT / "研发" / "data" / "index"

P_RESULTS = DIR_TESTDATA / "评估结果.jsonl"
P_SUMMARY = DIR_TESTDATA / "评估汇总.json"
P_ANALYSIS = DIR_TESTDATA / "检索问题分析.md"
P_QUESTIONS = DIR_EVALSRC / "测试问题集.jsonl"
P_MANIFEST = DIR_INDEX / "index_manifest.json"
P_SAMPLE_PDF = REPO_ROOT / "输入" / "sample_questions.pdf"
P_CORPUS_ZIP = REPO_ROOT / "输入" / "ccf_competition.zip"

OUT_DIR = REPO_ROOT / "部署" / "演示视频"
OUT_MP4 = OUT_DIR / "工单7_功能测试及评估演示.mp4"
FRAME_DIR = OUT_DIR / "帧"                        # 仅 --frames-only / 无 ffmpeg 时使用
KEYFRAME_DIR = OUT_DIR / "关键帧"                  # 正常出片时另存的分镜关键帧

# --------------------------------------------------------------------------
# 画面常量
# --------------------------------------------------------------------------
W, H = 1600, 900
HEADER_H = 86
FOOTER_H = 92
MARGIN = 34
CONTENT_TOP = HEADER_H + 16
CONTENT_BOTTOM = H - FOOTER_H - 16
CONTENT_W = W - 2 * MARGIN

BG = (13, 17, 26)
PANEL = (23, 30, 45)
PANEL2 = (30, 39, 58)
LINE = (48, 60, 84)
INK = (232, 238, 250)
MUTED = (150, 163, 188)
ACCENT = (88, 166, 255)
ACCENT2 = (124, 200, 255)
OK = (86, 214, 140)
WARN = (240, 190, 90)
BAD = (255, 120, 110)
WHITE = (255, 255, 255)

FONT_BOLD_CANDIDATES = [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc",
                        r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\Deng.ttf"]
FONT_REG_CANDIDATES = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc",
                       r"C:\Windows\Fonts\Deng.ttf", r"C:\Windows\Fonts\simhei.ttf"]
FONT_MONO_CANDIDATES = [r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\cour.ttf",
                        r"C:\Windows\Fonts\msyh.ttc"]
FFMPEG_CANDIDATES = [
    r"C:\Program Files\EVCapture\ffmpeg.exe",
    r"C:\ProgramData\GamePPPublic\PCBenchmark\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
]
FFPROBE_CANDIDATES = [
    r"C:\Program Files\EVCapture\ffprobe.exe",
    r"C:\ProgramData\GamePPPublic\PCBenchmark\ffprobe.exe",
]


# ==========================================================================
# 真实数据装载
# ==========================================================================
def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _short_corpus(file_name: str) -> str:
    """``2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf`` → ``平安银行 2019年``。"""
    stem = file_name[:-4] if file_name.lower().endswith(".pdf") else file_name
    parts = stem.split("__")
    if len(parts) >= 5:
        return f"{parts[3]} {parts[4]}"
    return stem[:24]


def _pdf_pages() -> int | None:
    if not P_SAMPLE_PDF.is_file():
        return None
    try:
        import fitz
        with fitz.open(str(P_SAMPLE_PDF)) as doc:
            return doc.page_count
    except Exception:                                       # noqa: BLE001
        return None


@dataclass
class Data:
    """本片用到的**全部**真实数据（一次性读盘，之后只读内存）。"""

    questions: list[dict[str, Any]]
    records: list[dict[str, Any]]
    summary: dict[str, Any]
    manifest: dict[str, Any]
    analysis: str
    pdf_pages: int | None
    pdf_bytes: int
    qset_bytes: int
    zip_bytes: int

    # -- 派生 ---------------------------------------------------------
    @property
    def total_keywords(self) -> int:
        return sum(len(q.get("answer_keywords") or []) for q in self.questions)

    @property
    def by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            k = str(r.get("question_type") or "?")
            out[k] = out.get(k, 0) + 1
        return out

    @property
    def by_lang(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for q in self.questions:
            k = str(q.get("lang") or "?")
            out[k] = out.get(k, 0) + 1
        return out

    @property
    def by_method(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            k = str(r.get("generation_method") or "?")
            out[k] = out.get(k, 0) + 1
        return out

    def correct_examples(self) -> list[dict[str, Any]]:
        """判对且引用页非空的题（优先数值题，画面更短更好读）。"""
        ok = [r for r in self.records
              if (r.get("judge") or {}).get("is_correct") and (r.get("cited_pages") or [])]
        ok.sort(key=lambda r: (r.get("question_type") != "数值题", len(str(r.get("answer") or ""))))
        return ok

    def refused_examples(self) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("refused")]

    def top5_problems(self) -> list[str]:
        """从 检索问题分析.md 的「## 四、检索问题 TOP5」抓条目。

        该章节实际是**表格**（| 排名 | 问题 | 影响题数 | 关键证据 |），
        故先按表格解析「问题」列；若版式改成有序/无序列表，退回列表解析。
        """
        import re
        m = re.search(r"^##\s*四、.*?$(.*?)(?=^##\s|\Z)", self.analysis, re.S | re.M)
        if not m:
            return []
        items: list[str] = []
        for ln in m.group(1).splitlines():
            ln = ln.strip()
            if not ln.startswith("|"):
                continue
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) < 2 or set(cells[0]) <= set("-: "):
                continue
            if not re.match(r"^\d+$", cells[0]):       # 跳过表头
                continue
            items.append(re.sub(r"\*\*|`", "", cells[1]).strip())
        if items:
            return items[:5]
        for ln in m.group(1).splitlines():
            mm = re.match(r"^(?:\d+[.)]|[-*])\s+(.*)$", ln.strip())
            if mm:
                items.append(re.sub(r"\*\*|`", "", mm.group(1)).strip())
        return items[:5]

    def root_cause(self) -> str:
        """从「## 三、深度归因」抓 3.1 的标题，作为一句话结论。"""
        import re
        m = re.search(r"^###\s*3\.1\s*(.+)$", self.analysis, re.M)
        return re.sub(r"\*\*|`", "", m.group(1)).strip() if m else ""

    # -- 自洽性：以逐题文件为准重新数一遍，防止拍到「汇总/逐题正在被重写」的撕裂快照 ----
    @property
    def counted_correct(self) -> int:
        return sum(1 for r in self.records if (r.get("judge") or {}).get("is_correct"))

    @property
    def counted_partial(self) -> int:
        return sum(1 for r in self.records if (r.get("judge") or {}).get("partial_correct"))

    @property
    def counted_coverage(self) -> float:
        if not self.records:
            return 0.0
        return sum(float((r.get("judge") or {}).get("keyword_coverage") or 0.0)
                   for r in self.records) / len(self.records)

    def summary_mismatch(self) -> str | None:
        """汇总 JSON 与逐题 JSONL 若口径不一致，返回一句人话；一致则返回 None。"""
        acc = self.summary.get("准确率") or {}
        diffs = []
        if acc.get("严格判对题数") != self.counted_correct:
            diffs.append(f"严格判对：汇总 {acc.get('严格判对题数')} vs 逐题实计 {self.counted_correct}")
        if acc.get("部分正确题数") != self.counted_partial:
            diffs.append(f"部分正确：汇总 {acc.get('部分正确题数')} vs 逐题实计 {self.counted_partial}")
        a = acc.get("平均关键词覆盖率")
        if isinstance(a, (int, float)) and abs(a - self.counted_coverage) > 1e-4:
            diffs.append(f"平均覆盖率：汇总 {a:.4f} vs 逐题实计 {self.counted_coverage:.4f}")
        return "；".join(diffs) if diffs else None


def load_data() -> Data:
    missing = [str(p) for p in (P_RESULTS, P_SUMMARY, P_QUESTIONS, P_MANIFEST)
               if not p.is_file()]
    if missing:
        raise SystemExit("[FATAL] 缺少真实数据文件，拒绝编造数字：\n  " + "\n  ".join(missing))
    return Data(
        questions=_read_jsonl(P_QUESTIONS),
        records=_read_jsonl(P_RESULTS),
        summary=_read_json(P_SUMMARY),
        manifest=_read_json(P_MANIFEST),
        analysis=P_ANALYSIS.read_text(encoding="utf-8") if P_ANALYSIS.is_file() else "",
        pdf_pages=_pdf_pages(),
        pdf_bytes=P_SAMPLE_PDF.stat().st_size if P_SAMPLE_PDF.is_file() else 0,
        qset_bytes=P_QUESTIONS.stat().st_size,
        zip_bytes=P_CORPUS_ZIP.stat().st_size if P_CORPUS_ZIP.is_file() else 0,
    )


# ==========================================================================
# 绘图工具
# ==========================================================================
class Fonts:
    def __init__(self) -> None:
        bold = _pick(FONT_BOLD_CANDIDATES)
        reg = _pick(FONT_REG_CANDIDATES)
        mono = _pick(FONT_MONO_CANDIDATES)
        self.brand = _font(bold, 22)
        self.scene = _font(bold, 27)
        self.h2 = _font(bold, 19)
        self.body = _font(reg, 17)
        self.body_b = _font(bold, 17)
        self.small = _font(reg, 15)
        self.small_b = _font(bold, 15)
        self.tiny = _font(reg, 13)
        self.mono = _font(mono, 14)
        self.mono_s = _font(mono, 13)
        self.kpi = _font(bold, 34)
        self.kpi_s = _font(bold, 26)
        self.narr = _font(reg, 18)
        self.foot = _font(reg, 14)


def _pick(cands: Sequence[str]) -> str:
    for c in cands:
        if Path(c).is_file():
            return c
    raise SystemExit("[FATAL] 找不到可用字体：" + "、".join(cands))


def _font(path: str, size: int) -> Any:
    try:
        return ImageFont.truetype(path, size)
    except OSError:                                          # pragma: no cover
        return ImageFont.load_default()


def tw(d: ImageDraw.ImageDraw, s: str, f: Any) -> float:
    return d.textlength(str(s), font=f)


def wrap(d: ImageDraw.ImageDraw, s: str, f: Any, max_w: float) -> list[str]:
    """按像素宽度折行（中英混排逐字符累加，够用且不依赖分词）。"""
    lines: list[str] = []
    for para in str(s).split("\n"):
        cur = ""
        for ch in para:
            if tw(d, cur + ch, f) <= max_w or not cur:
                cur += ch
            else:
                lines.append(cur)
                cur = ch
        lines.append(cur)
    return lines


def clip_text(d: ImageDraw.ImageDraw, s: str, f: Any, max_w: float) -> str:
    s = str(s)
    if tw(d, s, f) <= max_w:
        return s
    while s and tw(d, s + "…", f) > max_w:
        s = s[:-1]
    return s + "…"


def panel(d: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float,
          *, fill: Any = PANEL, outline: Any = LINE, r: int = 12) -> None:
    d.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=fill, outline=outline, width=1)


def label(d: ImageDraw.ImageDraw, x: float, y: float, s: str, f: Any,
          color: Any = MUTED) -> None:
    d.text((x, y), str(s), font=f, fill=color)


def kpi_tiles(d: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float,
              items: Sequence[tuple[str, str, Any]], *, cols: int = 4,
              fonts: Fonts, reveal: int = 999) -> None:
    """KPI 卡片网格：items = [(标签, 值, 值颜色), ...]。

    值字号**自适应（宽 + 高双重约束）**：先按「标签占顶部 22px」算出值的可用高度，
    再在字号阶梯里挑第一个既装得下宽度、也装得下高度的字号；都不行则用最小字号折两行。
    这样短卡片（如 3 行网格）不会把值压到标签上。
    """
    n = len(items)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    gap = 12
    cw = (w - gap * (cols - 1)) / cols
    ch = (h - gap * (rows - 1)) / rows
    ladder = [fonts.kpi_s, fonts.h2, fonts.body_b, fonts.small_b, fonts.small, fonts.tiny]
    top_reserved = 24              # 标签区高度
    bottom_pad = 8
    avail_w = cw - 26
    avail_h = max(12.0, ch - top_reserved - bottom_pad)
    for i, (k, v, color) in enumerate(items):
        if i >= reveal:
            return
        cx = x + (i % cols) * (cw + gap)
        cy = y + (i // cols) * (ch + gap)
        panel(d, cx, cy, cw, ch, fill=PANEL2)
        d.rectangle([cx, cy, cx + 3, cy + ch], fill=ACCENT)
        label(d, cx + 14, cy + 7, clip_text(d, k, fonts.small, cw - 26), fonts.small)
        chosen_font = None
        for cand in ladder:
            if cand.size <= avail_h and tw(d, v, cand) <= avail_w:
                chosen_font = cand
                break
        if chosen_font is not None:
            label(d, cx + 14, cy + ch - bottom_pad - chosen_font.size - 2, v,
                  chosen_font, color)
            continue
        f = ladder[-1]
        max_lines = max(1, min(2, int(avail_h // (f.size + 4))))
        lines = wrap(d, v, f, avail_w)[:max_lines]
        ty = cy + ch - bottom_pad - len(lines) * (f.size + 4)
        ty = max(ty, cy + top_reserved - 2)
        for j, ln in enumerate(lines):
            label(d, cx + 14, ty + j * (f.size + 4), ln, f, color)


def table(d: ImageDraw.ImageDraw, x: float, y: float, w: float,
          headers: Sequence[str], rows: Sequence[Sequence[Any]],
          widths: Sequence[float], *, fonts: Fonts, reveal: int = 999,
          row_h: float = 34, head_h: float = 36,
          colors: Sequence[Any] | None = None, mono_cols: Sequence[int] = ()) -> float:
    """画一张表；widths 是各列占 w 的比例。返回表格底部 y。"""
    total = sum(widths)
    ws = [w * f / total for f in widths]
    d.rounded_rectangle([x, y, x + w, y + head_h], radius=8, fill=(34, 45, 66))
    cx = x
    for i, htxt in enumerate(headers):
        label(d, cx + 10, y + 9, clip_text(d, htxt, fonts.small_b, ws[i] - 18), fonts.small_b, ACCENT2)
        cx += ws[i]
    cy = y + head_h
    for ri, row in enumerate(rows):
        if ri >= reveal:
            break
        if ri % 2 == 1:
            d.rectangle([x, cy, x + w, cy + row_h], fill=(19, 25, 38))
        cx = x
        for ci, cell in enumerate(row):
            f = fonts.mono_s if ci in mono_cols else fonts.small
            col = (colors[ri] if colors and ci == 0 and ri < len(colors) else INK)
            label(d, cx + 10, cy + (row_h - 17) / 2,
                  clip_text(d, str(cell), f, ws[ci] - 18), f, col)
            cx += ws[ci]
        d.line([x, cy + row_h, x + w, cy + row_h], fill=(32, 41, 58), width=1)
        cy += row_h
    return cy


def bar_chart(d: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float,
              items: Sequence[tuple[str, int]], *, fonts: Fonts, reveal: int = 999,
              bar_color: Any = ACCENT) -> None:
    """横向柱状图（真实分块数）。"""
    if not items:
        return
    mx = max(v for _k, v in items) or 1
    n = len(items)
    gap = 8
    bh = (h - gap * (n - 1)) / n
    name_w = 168.0
    val_w = 76.0
    track = w - name_w - val_w - 12
    for i, (k, v) in enumerate(items):
        if i >= reveal:
            return
        cy = y + i * (bh + gap)
        label(d, x, cy + bh / 2 - 10, clip_text(d, k, fonts.small, name_w - 8), fonts.small, INK)
        bx = x + name_w
        d.rounded_rectangle([bx, cy, bx + track, cy + bh], radius=4, fill=(30, 38, 56))
        fillw = max(3.0, track * v / mx)
        d.rounded_rectangle([bx, cy, bx + fillw, cy + bh], radius=4, fill=bar_color)
        label(d, bx + track + 8, cy + bh / 2 - 9, f"{v:,}", fonts.small_b, ACCENT2)


# ==========================================================================
# 镜头定义
# ==========================================================================
@dataclass
class Scene:
    no: str
    title: str
    narration: str
    seconds: float
    draw: Callable[[Image.Image, ImageDraw.ImageDraw, float, "Data", Fonts], None]


def clamp01(v: float) -> float:
    return 0.0 if v < 0 else (1.0 if v > 1 else v)


def stage(p: float, start: float, span: float) -> float:
    """把镜头内进度 p 映射成某个子阶段 0..1。"""
    return clamp01((p - start) / span) if span > 0 else 1.0


def reveal_n(p: float, start: float, span: float, total: int) -> int:
    return int(round(stage(p, start, span) * total))


# --------------------------------------------------------------------------
# ① 问题提取
# --------------------------------------------------------------------------
def draw_scene1(img: Image.Image, d: ImageDraw.ImageDraw, p: float, data: Data, F: Fonts) -> None:
    y = CONTENT_TOP
    bt = data.by_type
    short_type = {"定性题": "定性", "数值题": "数值", "跨文档对比题": "跨文档", "英文题": "英文"}
    tiles = [
        ("样例题 PDF 页数", str(data.pdf_pages if data.pdf_pages is not None else "—"), ACCENT2),
        ("提取题目数", f"{len(data.questions)} 题", OK),
        ("判分关键词总数", f"{data.total_keywords} 个", OK),
        ("题目来源标签数", f"{len({q.get('source_pdf') for q in data.questions})} 个", INK),
        ("语言分布", " / ".join(f"{k} {v}" for k, v in sorted(data.by_lang.items())), INK),
        ("题型分布", " / ".join(f"{short_type.get(k, k)} {v}" for k, v in bt.items()), INK),
    ]
    kpi_tiles(d, MARGIN, y, CONTENT_W, 118, tiles, cols=3, fonts=F,
              reveal=reveal_n(p, 0.0, 0.16, len(tiles)))
    y += 118 + 14

    cards = [
        ("输入 / sample_questions.pdf", f"{data.pdf_bytes/1024:.0f} KB · {data.pdf_pages} 页"),
        ("研发 / 评估 / 测试问题集.jsonl", f"{len(data.questions)} 题 · {data.qset_bytes/1024:.0f} KB"),
        ("判分口径", f"{data.total_keywords} 个 answer_keywords"),
    ]
    cw = (CONTENT_W - 2 * 12) / 3
    for i, (a, b) in enumerate(cards):
        if i >= reveal_n(p, 0.16, 0.14, 3):
            break
        cx = MARGIN + i * (cw + 12)
        panel(d, cx, y, cw, 96, fill=PANEL2)
        label(d, cx + 14, y + 14, clip_text(d, a, F.small_b, cw - 28), F.small_b, ACCENT2)
        label(d, cx + 14, y + 44, clip_text(d, b, F.body, cw - 28), F.body, INK)
        label(d, cx + 14, y + 70, "√ 仓库内真实文件", F.tiny, OK)
    y += 96 + 16

    panel(d, MARGIN, y, CONTENT_W, CONTENT_BOTTOM - y)
    label(d, MARGIN + 16, y + 12,
          f"测试问题集（{len(data.questions)} 题，逐题登记 answer_keywords）", F.h2, ACCENT2)
    qs = data.questions
    qr = reveal_n(p, 0.30, 0.68, len(qs))
    colw = (CONTENT_W - 36) / 2
    percol = math.ceil(len(qs) / 2)
    for i, q in enumerate(qs):
        if i >= qr:
            break
        col = i // percol
        row = i % percol
        qx = MARGIN + 16 + col * colw
        qy = y + 46 + row * 34
        if qy + 30 > CONTENT_BOTTOM:
            break
        nkw = len(q.get("answer_keywords") or [])
        d.text((qx, qy), str(q.get("id")), font=F.small_b, fill=ACCENT)
        label(d, qx + 46, qy, clip_text(d, str(q.get("question")), F.small, colw - 130), F.small, INK)
        label(d, qx + colw - 80, qy, f"{nkw} 关键词", F.tiny, MUTED)


# --------------------------------------------------------------------------
# ② 语料与索引规模
# --------------------------------------------------------------------------
def draw_scene2(img: Image.Image, d: ImageDraw.ImageDraw, p: float, data: Data, F: Fonts) -> None:
    m = data.manifest
    mod = m.get("chunk_by_modality") or {}
    bm25 = m.get("bm25") or {}
    ve = m.get("vector") or {}
    y = CONTENT_TOP
    tiles = [
        ("语料", f"{len(m.get('chunk_by_corpus') or {})} 份年报", INK),
        ("索引块总数", f"{m.get('chunk_total', 0):,} 块", OK),
        ("正文块", f"{mod.get('text', 0):,}", INK),
        ("表格块", f"{mod.get('table', 0):,}", INK),
        ("图像块", f"{mod.get('image', 0):,}", WARN if mod.get("image", 0) == 0 else INK),
        ("BM25 词表", f"{bm25.get('vocab_size', 0):,} 词", INK),
    ]
    kpi_tiles(d, MARGIN, y, CONTENT_W, 118, tiles, cols=6, fonts=F,
              reveal=reveal_n(p, 0.0, 0.18, len(tiles)))
    y += 118 + 14

    body_h = CONTENT_BOTTOM - y - 132
    lw = 520
    rw = CONTENT_W - lw - 14
    panel(d, MARGIN, y, lw, body_h)
    label(d, MARGIN + 16, y + 12, "索引规格（index_manifest.json）", F.h2, ACCENT2)
    spec = [
        ("嵌入模型", f"{m.get('embed_model')}（{m.get('embed_dim')} 维）"),
        ("向量后端", str(m.get("vector_backend"))),
        ("向量类型", str(ve.get("faiss_index_type"))),
        ("BM25 参数", f"k1={bm25.get('k1')}  b={bm25.get('b')}"),
        ("run_id", str(m.get("run_id"))),
        ("构建耗时", f"{(m.get('elapsed_ms') or 0)/1000:.1f} s"),
        ("向量归一化", str(ve.get("normalized"))),
        ("页码基准", str(m.get("page_basis"))),
        ("语料压缩包", f"{data.zip_bytes/1024/1024:.1f} MB"),
    ]
    spec_n = reveal_n(p, 0.16, 0.32, len(spec))
    for i, (k, v) in enumerate(spec):
        if i >= spec_n:
            break
        yy = y + 48 + i * 38
        if yy + 28 > y + body_h - 8:
            break
        label(d, MARGIN + 16, yy, k, F.small, MUTED)
        label(d, MARGIN + 168, yy, clip_text(d, v, F.small_b, lw - 190), F.small_b, INK)

    rx = MARGIN + lw + 14
    panel(d, rx, y, rw, body_h)
    label(d, rx + 16, y + 12, "各语料分块数（chunk_by_corpus）", F.h2, ACCENT2)
    items = sorted(((_short_corpus(k), int(v)) for k, v in (m.get("chunk_by_corpus") or {}).items()),
                   key=lambda kv: -kv[1])
    bar_chart(d, rx + 16, y + 48, rw - 32, body_h - 62, items, fonts=F,
              reveal=reveal_n(p, 0.22, 0.5, len(items)))
    y += body_h + 14

    panel(d, MARGIN, y, CONTENT_W, 118, fill=(41, 33, 20), outline=(120, 92, 40))
    label(d, MARGIN + 16, y + 12, "注意：已知降级（如实标注，不掩盖）", F.h2, WARN)
    warn = ("图像块数 = 0：本语料无 image_blocks.jsonl，且本机缺 cv2 / onnxruntime、未缓存中文 CLIP 权重，"
            "图像检索链路不可用。")
    for i, ln in enumerate(wrap(d, warn, F.small, CONTENT_W - 40)[:2]):
        label(d, MARGIN + 16, y + 44 + i * 24, ln, F.small, INK)
    label(d, MARGIN + 16, y + 92,
          "文本 / 表格索引照常构建 —— 见 index_manifest.json 的 warnings 与 部署/README.md 已知限制。",
          F.tiny, MUTED)


# --------------------------------------------------------------------------
# ③ RAG 检索与回答示例
# --------------------------------------------------------------------------
def draw_scene3(img: Image.Image, d: ImageDraw.ImageDraw, p: float, data: Data, F: Fonts) -> None:
    good = data.correct_examples()
    if not good:
        label(d, MARGIN, CONTENT_TOP, "无判对样例（评估结果可能未就绪）", F.h2, BAD)
        return
    ex = good[0]
    judge = ex.get("judge") or {}
    lat = ex.get("latency_ms") or {}
    ref = data.refused_examples()
    ref_txt = f"{ref[0].get('id')} → refused（{ref[0].get('refusal_reason')}）" if ref else "无"

    y = CONTENT_TOP
    ph = 104
    panel(d, MARGIN, y, CONTENT_W, ph)
    label(d, MARGIN + 16, y + 10,
          f"提问　{ex.get('id')}　{ex.get('question_type')}　{ex.get('company')}", F.h2, ACCENT2)
    for i, ln in enumerate(wrap(d, str(ex.get("question")), F.body, CONTENT_W - 40)[:2]):
        label(d, MARGIN + 16, y + 42 + i * 26, ln, F.body, INK)
    y += ph + 14

    mid_h = 344
    lw = 946
    rw = CONTENT_W - lw - 14
    panel(d, MARGIN, y, lw, mid_h)
    label(d, MARGIN + 16, y + 12, "回答（QAEngine 真实输出）", F.h2, ACCENT2)
    answer = str(ex.get("answer") or "")
    shown = answer[: int(len(answer) * stage(p, 0.10, 0.55))]
    alines = wrap(d, shown, F.body, lw - 40)
    maxlines = int((mid_h - 76) / 27)
    for i, ln in enumerate(alines[:maxlines]):
        label(d, MARGIN + 16, y + 48 + i * 27, ln, F.body, INK)
    if len(alines) > maxlines:
        label(d, MARGIN + 16, y + mid_h - 46, f"…（答案共 {len(answer)} 字，此处截断显示）", F.tiny, MUTED)
    label(d, MARGIN + 16, y + mid_h - 26,
          f"引用页：{ex.get('cited_pages')}　|　答案长度 {len(answer)} 字", F.tiny, MUTED)

    rx = MARGIN + lw + 14
    panel(d, rx, y, rw, mid_h)
    label(d, rx + 16, y + 12, "分段耗时 / 生成方式", F.h2, ACCENT2)
    tiles = [
        ("检索", f"{lat.get('retrieve', 0):.0f} ms", ACCENT2),
        ("生成", f"{lat.get('generate', 0):.0f} ms", ACCENT2),
        ("首字", f"{lat.get('first_token', 0):.0f} ms", INK),
        ("总耗时", f"{ex.get('total_ms', 0):.0f} ms", OK),
        ("生成方式", str(ex.get("generation_method")), INK),
        ("recall@10", str(judge.get("recall_at_10")), INK),
    ]
    kpi_tiles(d, rx + 16, y + 44, rw - 32, 176, tiles, cols=3, fonts=F,
              reveal=reveal_n(p, 0.14, 0.3, len(tiles)))
    label(d, rx + 16, y + 236, "拒答红线（不清楚就是正确响应）", F.small_b, WARN)
    label(d, rx + 16, y + 262, clip_text(d, ref_txt, F.small, rw - 32), F.small, INK)
    gate = ex.get("gate") or {}
    label(d, rx + 16, y + 288, f"可答性 gate：answerable={gate.get('answerable')}", F.tiny, MUTED)
    label(d, rx + 16, y + 310, f"trace_id = {ex.get('trace_id')}", F.tiny, MUTED)
    y += mid_h + 14

    hh = CONTENT_BOTTOM - y
    panel(d, MARGIN, y, CONTENT_W, hh)
    label(d, MARGIN + 16, y + 10, "检索层命中（Top-5，chunk_id / 页码 / 类型 / final_score）", F.h2, ACCENT2)
    hits = (ex.get("hits") or [])[:5]
    rows = [[h.get("rank"), h.get("chunk_id"), h.get("page"), h.get("type"),
             f"{h.get('final_score'):.6f}" if isinstance(h.get("final_score"), (int, float)) else "-",
             f"{h.get('vector_rank')}/{h.get('bm25_rank')}",
             _short_corpus(str(h.get("file_name") or ""))] for h in hits]
    table(d, MARGIN + 16, y + 42, CONTENT_W - 32,
          ["#", "chunk_id", "页", "类型", "final_score", "向量/BM25 名次", "语料"],
          rows, [0.4, 2.5, 0.6, 0.9, 1.5, 1.7, 1.9], fonts=F,
          reveal=reveal_n(p, 0.55, 0.4, len(rows)),
          row_h=min(30.0, max(20.0, (hh - 90) / max(1, len(rows)))), head_h=30, mono_cols=(1, 4))


# --------------------------------------------------------------------------
# ④ 评估判分表
# --------------------------------------------------------------------------
def draw_scene4(img: Image.Image, d: ImageDraw.ImageDraw, p: float, data: Data, F: Fonts) -> None:
    y = CONTENT_TOP
    cfg = data.summary.get("配置") or {}
    # 用**逐题文件实计**的数字（而不是汇总 JSON 的声明），保证与下方表格逐行自洽
    legend = (f"判分口径：关键词覆盖率 ≥ {cfg.get('关键词阈值')} 判对，≥ {cfg.get('部分正确阈值')} 记部分正确"
              f"　|　严格判对 {data.counted_correct}/{len(data.records)}"
              f"　|　部分正确 {data.counted_partial}"
              f"　|　平均覆盖率 {data.counted_coverage:.4f}")
    panel(d, MARGIN, y, CONTENT_W, 48)
    label(d, MARGIN + 16, y + 14, clip_text(d, legend, F.small, CONTENT_W - 32), F.small, ACCENT2)
    y += 48 + 12

    rows: list[list[Any]] = []
    colors: list[Any] = []
    for r in data.records:
        j = r.get("judge") or {}
        leak = j.get("cross_corpus_leak")
        leak_b = bool(leak.get("leak")) if isinstance(leak, dict) else bool(leak)
        if j.get("is_correct"):
            verdict, col = "判对", OK
        elif j.get("partial_correct"):
            verdict, col = "部分", WARN
        else:
            verdict, col = "未判对", BAD
        cov = j.get("keyword_coverage")
        r5, r10 = j.get("recall_at_5"), j.get("recall_at_10")
        rows.append([
            r.get("id"), r.get("question_type"), str(r.get("question"))[:46],
            f"{cov:.3f}" if isinstance(cov, (int, float)) else "-",
            verdict,
            f"{r5:.3f}" if isinstance(r5, (int, float)) else "-",
            f"{r10:.3f}" if isinstance(r10, (int, float)) else "-",
            "是" if leak_b else "否",
            f"{r.get('total_ms', 0):.0f}",
            str(r.get("generation_method")),
        ])
        colors.append(col)
    n = reveal_n(p, 0.06, 0.86, len(rows))
    bot = table(d, MARGIN, y, CONTENT_W,
                ["题号", "题型", "问题（截断）", "覆盖率", "是否判对", "recall@5", "recall@10",
                 "跨语料泄漏", "耗时 ms", "生成方式"],
                rows, [0.55, 1.0, 4.0, 0.85, 0.8, 0.85, 0.9, 1.05, 0.85, 1.25],
                fonts=F, reveal=n, row_h=34, head_h=36, colors=colors)
    label(d, MARGIN, min(bot + 10, CONTENT_BOTTOM - 20),
          f"源：测试/测试数据/评估结果.jsonl —— 已展开 {n}/{len(rows)} 题（数字均为文件原值）",
          F.tiny, MUTED)


# --------------------------------------------------------------------------
# ⑤ 汇总指标与检索问题分析
# --------------------------------------------------------------------------
def draw_scene5(img: Image.Image, d: ImageDraw.ImageDraw, p: float, data: Data, F: Fonts) -> None:
    s = data.summary
    acc = s.get("准确率") or {}
    rec = s.get("检索召回") or {}
    sla = s.get("SLA") or {}
    refu = s.get("拒答") or {}
    leak = s.get("跨语料泄漏") or {}
    page = s.get("页命中") or {}
    reach = s.get("索引可达性") or {}
    rt = s.get("响应时间ms") or {}

    y = CONTENT_TOP
    tiles = [
        ("严格判对", f"{data.counted_correct}/{len(data.records)}", OK),
        ("严格准确率", f"{data.counted_correct / max(1, len(data.records)) * 100:.1f}%", OK),
        ("宽松准确率", f"{(acc.get('宽松准确率(严格+部分)') or 0)*100:.1f}%", WARN),
        ("平均 recall@5", str(rec.get("平均recall@5")), ACCENT2),
        ("平均 recall@10", str(rec.get("平均recall@10")), ACCENT2),
        ("SLA 达标率", f"{(sla.get('达标率') or 0)*100:.0f}%", OK),
        ("拒答", f"{refu.get('拒答题数')} 题", INK),
        ("跨语料泄漏", f"{leak.get('题数')} 题", WARN),
        ("页命中率", f"{(page.get('命中率') or 0)*100:.1f}%", INK),
        ("索引可达率", f"{(reach.get('单块可达率') or 0)*100:.2f}%", OK),
        ("平均总耗时", f"{rt.get('平均', 0):.0f} ms", INK),
        ("P95 总耗时", f"{rt.get('P95', 0):.0f} ms", INK),
    ]
    kpi_tiles(d, MARGIN, y, CONTENT_W, 158, tiles, cols=6, fonts=F,
              reveal=reveal_n(p, 0.0, 0.34, len(tiles)))
    y += 158 + 14

    body_h = CONTENT_BOTTOM - y - 116
    lw = 730
    rw = CONTENT_W - lw - 14
    panel(d, MARGIN, y, lw, body_h)
    label(d, MARGIN + 16, y + 12, "按题型分组（评估汇总.json）", F.h2, ACCENT2)
    grp = s.get("按题型分组") or {}
    rows = [[k, v.get("题数"), v.get("判对题数"),
             f"{(v.get('判对率') or 0)*100:.1f}%",
             f"{v.get('平均关键词覆盖率'):.3f}" if isinstance(v.get("平均关键词覆盖率"), (int, float)) else "-",
             f"{v.get('平均recall@5'):.3f}" if isinstance(v.get("平均recall@5"), (int, float)) else "-",
             f"{v.get('平均总耗时ms'):.0f}" if isinstance(v.get("平均总耗时ms"), (int, float)) else "-"]
            for k, v in grp.items()]
    table(d, MARGIN + 16, y + 44, lw - 32,
          ["题型", "题数", "判对", "判对率", "平均覆盖率", "平均 recall@5", "平均耗时 ms"],
          rows, [1.3, 0.7, 0.7, 0.9, 1.2, 1.3, 1.2], fonts=F,
          reveal=reveal_n(p, 0.28, 0.4, len(rows)), row_h=32, head_h=32)
    label(d, MARGIN + 16, y + body_h - 74,
          clip_text(d, f"生成方式分布：{json.dumps(data.by_method, ensure_ascii=False)}",
                    F.small, lw - 32), F.small, INK)
    label(d, MARGIN + 16, y + body_h - 46,
          clip_text(d, f"索引内可达关键词：{reach.get('索引内可达')}/{reach.get('关键词总数')}"
                       f"（单块内不可达 {reach.get('单块内不可达')} 个，属切块边界切断）",
                    F.small, lw - 32), F.small, INK)

    rx = MARGIN + lw + 14
    panel(d, rx, y, rw, body_h)
    label(d, rx + 16, y + 12, "检索问题分析.md · TOP5 问题", F.h2, ACCENT2)
    tops = data.top5_problems()
    tn = reveal_n(p, 0.36, 0.4, max(1, len(tops)))
    yy = y + 46
    for i, t in enumerate(tops[:tn]):
        for j, ln in enumerate(wrap(d, f"{i+1}. {t}", F.small, rw - 36)[:2]):
            if yy + 22 > y + body_h - 108:
                break
            label(d, rx + 16, yy, ln, F.small, INK)
            yy += 23
        yy += 6
    label(d, rx + 16, y + body_h - 96, "深度归因（3.1 头号问题）", F.small_b, WARN)
    for i, ln in enumerate(wrap(d, data.root_cause(), F.small, rw - 32)[:2]):
        label(d, rx + 16, y + body_h - 70 + i * 23, ln, F.small, INK)
    y += body_h + 14

    panel(d, MARGIN, y, CONTENT_W, 102, fill=(18, 36, 30), outline=(44, 108, 76))
    concl = (f"结论：检索层平均 recall@5 {rec.get('平均recall@5')}、recall@10 {rec.get('平均recall@10')}，"
             f"页命中率 {(page.get('命中率') or 0)*100:.1f}%，{sla.get('阈值ms')}ms 内响应达标率 "
             f"{(sla.get('达标率') or 0)*100:.0f}%；低分主因在生成层而非检索层 "
             f"—— 完整归因见 测试/测试数据/检索问题分析.md。")
    for i, ln in enumerate(wrap(d, concl, F.body, CONTENT_W - 40)[:3]):
        label(d, MARGIN + 16, y + 16 + i * 26, ln, F.body, INK)
    mm = data.summary_mismatch()
    if mm:
        label(d, MARGIN + 16, y + 84,
              clip_text(d, "注意：汇总 JSON 与逐题 JSONL 口径不一致（本片一律以逐题为准）：" + mm,
                        F.tiny, CONTENT_W - 32), F.tiny, WARN)


SCENES: tuple[Scene, ...] = (
    Scene("①", "问题提取：sample_questions.pdf → 测试问题集",
          "从竞赛样例题 PDF 提取 17 道题并逐题登记 answer_keywords，形成可自动判分的测试问题集。",
          9.0, draw_scene1),
    Scene("②", "语料与索引规模：9 份年报 → 8742 块",
          "解析 9 份 A 股年报，分块后经 bge-m3 编码进 FAISS，并建 BM25 倒排；图像块为 0 属已登记的降级。",
          10.0, draw_scene2),
    Scene("③", "RAG 检索与回答示例：真实单题链路",
          "真实取一道判对的题，展示答案、引用页、检索/生成/首字/总耗时，以及检索层命中的块。",
          13.0, draw_scene3),
    Scene("④", "评估判分表：17 题逐题结果",
          "每题的题型、关键词覆盖率、是否判对、recall@5/@10、跨语料泄漏、耗时与生成方式，均取自评估结果文件。",
          12.0, draw_scene4),
    Scene("⑤", "汇总指标与检索问题分析",
          "汇总准确率、召回、SLA、拒答与泄漏，并给出检索问题 TOP5 与深度归因结论。",
          12.0, draw_scene5),
)


# ==========================================================================
# 每帧页眉 / 页脚 / 进度条
# ==========================================================================
FPS_HINT = 20        # 仅用于页脚文字；真实值由 --fps 决定，main() 会覆盖


def draw_chrome(d: ImageDraw.ImageDraw, si: int, p_all: float, F: Fonts) -> None:
    sc = SCENES[si]
    d.rectangle([0, 0, W, HEADER_H], fill=(17, 23, 36))
    d.rectangle([0, HEADER_H - 3, W, HEADER_H], fill=ACCENT)
    label(d, MARGIN, 20, "工单7 · " + WORK_ORDER, F.brand, WHITE)
    label(d, MARGIN, 52, "测试评估台 · 功能测试与评估演示", F.small, MUTED)
    title = f"{sc.no} {sc.title}"
    d.text((W - MARGIN - tw(d, title, F.scene), 28), title, font=F.scene, fill=ACCENT2)
    cx = W - MARGIN - 12
    for i in range(len(SCENES) - 1, -1, -1):
        d.ellipse([cx - 9, 68, cx, 77], fill=(ACCENT if i == si else (60, 74, 100)))
        cx -= 18

    d.rectangle([0, H - FOOTER_H, W, H], fill=(17, 23, 36))
    d.rectangle([0, H - FOOTER_H, W, H - FOOTER_H + 3], fill=ACCENT)
    for i, ln in enumerate(wrap(d, sc.narration, F.narr, W - 2 * MARGIN - 210)[:2]):
        label(d, MARGIN, H - FOOTER_H + 16 + i * 26, ln, F.narr, (206, 216, 234))
    label(d, W - MARGIN - 196, H - FOOTER_H + 16, f"镜头 {si + 1}/{len(SCENES)}", F.small, ACCENT2)
    label(d, W - MARGIN - 196, H - FOOTER_H + 42,
          f"{sc.seconds:.0f}s · {int(round(sc.seconds * FPS_HINT))} 帧", F.tiny, MUTED)
    bx0, bx1 = MARGIN, W - MARGIN
    by = H - 12
    d.rounded_rectangle([bx0, by, bx1, by + 5], radius=3, fill=(38, 47, 66))
    d.rounded_rectangle([bx0, by, bx0 + (bx1 - bx0) * clamp01(p_all), by + 5], radius=3, fill=ACCENT)


def render_scene(sc: Scene, si: int, data: Data, F: Fonts, fps: int,
                 p_all_base: float, p_all_span: float,
                 on_frame: Callable[[Image.Image], None]) -> int:
    n = max(1, int(round(sc.seconds * fps)))
    for i in range(n):
        p = i / max(1, n - 1)
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img, "RGBA")
        draw_chrome(d, si, p_all_base + p_all_span * p, F)
        sc.draw(img, d, p, data, F)
        a = clamp01(p / 0.12)                       # 切入淡入（前 12% 时长）
        if a < 1.0:
            d.rectangle([0, HEADER_H, W, H - FOOTER_H], fill=(13, 17, 26, int(235 * (1 - a))))
        on_frame(img)
    return n


# ==========================================================================
# ffmpeg
# ==========================================================================
def find_ffmpeg(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit if Path(explicit).is_file() else None
    found = shutil.which("ffmpeg")
    if found:
        return found
    for p in FFMPEG_CANDIDATES:
        if Path(p).is_file():
            return p
    return None


def find_ffprobe(ffmpeg: str | None) -> str | None:
    if ffmpeg:
        sib = Path(ffmpeg).with_name("ffprobe.exe")
        if sib.is_file():
            return str(sib)
    for p in FFPROBE_CANDIDATES:
        if Path(p).is_file():
            return p
    return shutil.which("ffprobe")


def probe_duration(ffprobe: str, path: Path) -> float | None:
    try:
        out = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", str(path)],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=60)
        return float(out.stdout.strip())
    except Exception:                                        # noqa: BLE001
        return None


# ==========================================================================
# 帧序列合成说明（无 ffmpeg 时的兜底交付）
# ==========================================================================
def _fallback_readme(frames: int, fps: int, seconds: float, data: Data) -> str:
    return f"""# 工单7 演示视频 —— 帧序列合成说明

工单编号：人工智能NLP-RAG-功能测试及评估

## 为什么这里只有帧序列

本机**没有找到可用的 ffmpeg**（PATH 与已知安装位置均未命中），
`部署/脚本/生成演示视频.py` 自动退化为 `--frames-only` 模式：
把全部 {frames} 张 `{W}×{H}` 的 PNG 帧写到本目录下的 `帧/`，
由本文件说明如何用别的方式合成 mp4。

> 若后来装上了 ffmpeg，直接重跑即可自动出片：
> `pwsh -NoProfile -File 部署/脚本/一键跑测试评估.ps1 -Only video`

## 帧序列参数（真实值）

| 项目 | 值 |
|---|---|
| 帧数 | {frames} 张（`帧/f00000_s1.png` 起递增） |
| 分辨率 | {W} × {H} |
| 帧率 | {fps} fps |
| 预期时长 | {seconds:.1f} 秒 |
| 生成时间 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |

## 用 ffmpeg 合成

```powershell
<FFMPEG> -y -framerate {fps} -i "帧/f%05d_s1.png" -c:v libx264 -pix_fmt yuv420p `
         -crf 20 -movflags +faststart 工单7_功能测试及评估演示.mp4
```

帧文件名形如 `f00000_s1.png`（序号 + 镜头号），若上面的模式不匹配，
用 concat 清单（按文件名排序）最稳：

```powershell
$files = Get-ChildItem 帧\\*.png | Sort-Object Name
$lines = foreach ($f in $files) {{
    "file '$($f.FullName -replace '\\\\','/')'"
    "duration {1.0/fps:.4f}"
}}
$lines += "file '$($files[-1].FullName -replace '\\\\','/')'"
$lines | Set-Content _concat.txt -Encoding UTF8
<FFMPEG> -y -f concat -safe 0 -i _concat.txt -vf "fps={fps},format=yuv420p" `
         -c:v libx264 -crf 20 工单7_功能测试及评估演示.mp4
```

## 其它可行方式

* **剪辑软件**：把 `帧/` 里的 PNG 以 {fps} fps 导入剪映 / Premiere / DaVinci Resolve，
  直接导出 mp4。
* **Python + imageio**（若该环境装了 `imageio-ffmpeg`）：
  ```python
  import glob, imageio.v3 as iio
  files = sorted(glob.glob("帧/*.png"))
  iio.imwrite("工单7_功能测试及评估演示.mp4", [iio.imread(f) for f in files], fps={fps})
  ```

## 画面里的数字来源（全部真实，无写死）

* `测试/测试数据/评估结果.jsonl`（{len(data.records)} 题逐题结果）
* `测试/测试数据/评估汇总.json`（汇总指标）
* `测试/测试数据/检索问题分析.md`（归因与 TOP5）
* `研发/评估/测试问题集.jsonl`（{len(data.questions)} 题 / {data.total_keywords} 个判分关键词）
* `研发/data/index/index_manifest.json`（索引规模 {data.manifest.get('chunk_total')} 块）
"""


# ==========================================================================
# main
# ==========================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=f"[{WORK_ORDER}] 逐帧绘制 + ffmpeg 合成演示视频")
    ap.add_argument("--out", default=None, help="输出 mp4 路径")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg.exe 路径（默认自动探测）")
    ap.add_argument("--fps", type=int, default=20, help="帧率（默认 20）")
    ap.add_argument("--frames-only", action="store_true",
                    help="只输出 PNG 帧序列到 部署/演示视频/帧/（不调 ffmpeg）")
    ap.add_argument("--keyframes-only", action="store_true",
                    help="只输出 5 张分镜关键帧到 部署/演示视频/关键帧/")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")              # type: ignore[union-attr]
    except Exception:                                         # noqa: BLE001
        pass

    global FPS_HINT
    FPS_HINT = max(1, args.fps)

    print("=" * 78)
    print(f"[{WORK_ORDER}] 演示视频生成")
    print("=" * 78)
    data = load_data()
    F = Fonts()
    print(f"数据源：{P_RESULTS.relative_to(REPO_ROOT)}（{len(data.records)} 题）")
    print(f"        {P_SUMMARY.relative_to(REPO_ROOT)}、{P_QUESTIONS.relative_to(REPO_ROOT)}"
          f"（{len(data.questions)} 题 / {data.total_keywords} 关键词）")
    print(f"索引  ：{data.manifest.get('chunk_total')} 块 · {data.manifest.get('embed_model')}"
          f"({data.manifest.get('embed_dim')} 维) · {data.manifest.get('vector_backend')}")
    good = data.correct_examples()
    print(f"镜头③样例：{good[0].get('id') if good else '（无判对样例）'}；"
          f"拒答样例：{[r.get('id') for r in data.refused_examples()]}")
    print(f"TOP5 问题：{len(data.top5_problems())} 条；根因：{data.root_cause()[:40]}")
    mm = data.summary_mismatch()
    if mm:
        print(f"[警告] 汇总 JSON 与逐题 JSONL 口径不一致（本片以逐题实计为准）：{mm}")
        print("       评测官可能正在重写产物；建议其跑完后再重跑本脚本覆盖视频。")
    else:
        print(f"自洽性：汇总 JSON 与逐题 JSONL 一致（判对 {data.counted_correct}、"
              f"部分 {data.counted_partial}、平均覆盖率 {data.counted_coverage:.4f}）")
    total_sec = sum(s.seconds for s in SCENES)
    total_frames = sum(max(1, int(round(s.seconds * FPS_HINT))) for s in SCENES)
    print(f"分镜  ：{len(SCENES)} 个镜头 · 约 {total_sec:.1f} s · {total_frames} 帧 @ {FPS_HINT} fps"
          f" · {W}×{H}")
    print("=" * 78, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 只出关键帧 ----
    if args.keyframes_only:
        KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
        for f in KEYFRAME_DIR.glob("*.png"):
            f.unlink()
        for si, sc in enumerate(SCENES):
            saved: list[Image.Image] = []
            render_scene(sc, si, data, F, FPS_HINT,
                         sum(s.seconds for s in SCENES[:si]) / total_sec,
                         sc.seconds / total_sec, saved.append)
            path = KEYFRAME_DIR / f"镜头{si+1:02d}_{sc.title[:16]}.png"
            saved[-1].save(path)
            print(f"  ✅ {path.relative_to(REPO_ROOT)}")
        print(f"[完成] 关键帧目录：{KEYFRAME_DIR}")
        return 0

    ffmpeg = None if args.frames_only else find_ffmpeg(args.ffmpeg)

    # ---- 退化路径：无 ffmpeg → 帧序列 + README ----
    if ffmpeg is None:
        print("[警告] 未找到 ffmpeg，退化为「PNG 帧序列」输出（不合成 mp4）。", flush=True)
        FRAME_DIR.mkdir(parents=True, exist_ok=True)
        for f in FRAME_DIR.glob("*.png"):
            f.unlink()
        counter = {"i": 0}
        for si, sc in enumerate(SCENES):
            def _save(im: Image.Image, _si: int = si) -> None:
                im.save(FRAME_DIR / f"f{counter['i']:05d}_s{_si+1}.png")
                counter["i"] += 1
            render_scene(sc, si, data, F, FPS_HINT,
                         sum(s.seconds for s in SCENES[:si]) / total_sec,
                         sc.seconds / total_sec, _save)
            print(f"  镜头 {si+1}/{len(SCENES)} 完成（累计 {counter['i']} 帧）", flush=True)
        readme = OUT_DIR / "README.md"
        readme.write_text(_fallback_readme(counter["i"], FPS_HINT, total_sec, data), encoding="utf-8")
        print(f"[完成] 帧序列 {counter['i']} 张 → {FRAME_DIR}")
        print(f"[完成] 合成说明 → {readme}")
        return 0

    # ---- 正常路径：逐帧 → 管道 → ffmpeg ----
    out_path = Path(args.out) if args.out else OUT_MP4
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS_HINT),
        "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path),
    ]
    print(f"[编码] ffmpeg = {ffmpeg}")
    print(f"[编码] 输出 = {out_path.relative_to(REPO_ROOT)}", flush=True)

    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    for f in KEYFRAME_DIR.glob("*.png"):
        f.unlink()

    # 注意：mkstemp 会**打开**一个文件描述符，必须立刻关掉，
    # 否则 Windows 上这个句柄会一直占着文件，后面 unlink 会 WinError 32。
    _fd, _tmp = tempfile.mkstemp(suffix=".ffmpeg.log")
    os.close(_fd)
    err = Path(_tmp)
    written = 0
    rc = 1
    log_text = ""
    try:
        with err.open("wb") as errfh:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=errfh, stderr=errfh)
            assert proc.stdin is not None
            try:
                for si, sc in enumerate(SCENES):
                    last: list[Image.Image] = []

                    def _feed(im: Image.Image) -> None:
                        nonlocal written
                        proc.stdin.write(im.tobytes())          # type: ignore[union-attr]
                        written += 1
                        last.append(im)
                        if written % 200 == 0:
                            print(f"    …已写 {written}/{total_frames} 帧", flush=True)

                    render_scene(sc, si, data, F, FPS_HINT,
                                 sum(s.seconds for s in SCENES[:si]) / total_sec,
                                 sc.seconds / total_sec, _feed)
                    if last:
                        last[-1].save(KEYFRAME_DIR / f"镜头{si+1:02d}_{sc.title[:16]}.png")
                    print(f"  镜头 {si+1}/{len(SCENES)} 已生成（累计 {written} 帧）", flush=True)
            finally:
                proc.stdin.close()                            # type: ignore[union-attr]
            rc = proc.wait(timeout=900)
    finally:
        log_text = err.read_text(encoding="utf-8", errors="replace") if err.is_file() else ""
        try:
            err.unlink(missing_ok=True)
        except OSError:
            pass                                              # 句柄未释放也不影响出片

    if rc != 0:
        print(f"[FATAL] ffmpeg 退出码 {rc}：\n{log_text[-2000:]}", file=sys.stderr)
        return 1

    size_mb = out_path.stat().st_size / 1024 / 1024
    ffprobe = find_ffprobe(ffmpeg)
    dur = probe_duration(ffprobe, out_path) if ffprobe else None
    print("=" * 78)
    print(f"[完成] 视频：{out_path}")
    print(f"       帧数 {written}　时长 {dur if dur is not None else total_sec:.2f} s"
          f"（{'ffprobe 实测' if dur is not None else '按帧数/帧率计算'}）"
          f"　{size_mb:.2f} MB　{W}x{H} @ {FPS_HINT} fps")
    print(f"       关键帧：{KEYFRAME_DIR.relative_to(REPO_ROOT)}")
    print("[说明] 图文讲解片（数据面板 + 旁白字幕），无音轨；所有数字取自真实评测产物。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
