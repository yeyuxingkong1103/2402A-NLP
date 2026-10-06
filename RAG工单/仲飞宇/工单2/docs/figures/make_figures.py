#!/usr/bin/env python
"""生成 docs/figures/ 下的四张流程图 SVG。

用法：
    .venv/bin/python docs/figures/make_figures.py

本脚本只产出 **SVG**。同目录的 `fig*.png` 是由 SVG 转出来的**派生物**（给不认 SVG 的
老版 Office 用），转换命令见 docs/流程图.md 文末；本脚本不带转 PNG 的依赖，
sharp 只是个用一次的临时工具。**改了 SVG 记得重新转 PNG**，否则两边会不一致。

为什么是脚本，而不是手写 SVG 或 mermaid-cli：
  - 本机没有 graphviz / pandoc / mermaid-cli，而装 mermaid-cli 要拖一整个 chromium
    （外网仅 6Mbps，代价太高）；
  - 四张图共用同一套配色与排版规则，只有脚本化才能保证它们看起来是「一套图」；
  - 改文案只要改下面 FIGS 里的字符串再重跑，不用碰任何坐标。

版面模型：每张图由若干「行」自下而上堆叠组成。
  - 行内节点默认整体居中排布（parallel row）；
  - 用 main= 指定「主路径节点」后，它固定画在中轴上，其余节点甩到它左右两侧——
    流程图里的分支（错误分支、可选分支）就是这么做出来的，主路径因此始终是一条直线。
画布宽高由内容反推，不需要手填。

与 docs/流程图.md 的关系：那边是同一套图的 Mermaid 源码（可读、可版本管理、GitHub 能渲染），
这边是给 Word / PPT 用的矢量图。两边的结构与文案要同步改。
"""
from __future__ import annotations

import math
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# ---------- 样式 ----------
# 配色按「层」区分：同一层的节点同色，读者扫一眼就能定位它属于哪一段。
STYLES: dict[str, dict[str, str]] = {
    "start":    {"fill": "#0f172a", "stroke": "#0f172a", "text": "#ffffff", "sub": "#cbd5e1"},
    "box":      {"fill": "#ffffff", "stroke": "#94a3b8", "text": "#0f172a", "sub": "#64748b"},
    "api":      {"fill": "#dbeafe", "stroke": "#3b82f6", "text": "#0f172a", "sub": "#1d4ed8"},
    "orch":     {"fill": "#ede9fe", "stroke": "#8b5cf6", "text": "#0f172a", "sub": "#6d28d9"},
    "online":   {"fill": "#dcfce7", "stroke": "#22c55e", "text": "#0f172a", "sub": "#15803d"},
    "offline":  {"fill": "#ffedd5", "stroke": "#f97316", "text": "#0f172a", "sub": "#c2410c"},
    "store":    {"fill": "#ccfbf1", "stroke": "#14b8a6", "text": "#0f172a", "sub": "#0f766e"},
    "ext":      {"fill": "#f1f5f9", "stroke": "#94a3b8", "text": "#0f172a", "sub": "#64748b"},
    "decision": {"fill": "#fef3c7", "stroke": "#f59e0b", "text": "#0f172a", "sub": "#b45309"},
    "error":    {"fill": "#fee2e2", "stroke": "#ef4444", "text": "#0f172a", "sub": "#b91c1c"},
    "ok":       {"fill": "#dcfce7", "stroke": "#22c55e", "text": "#0f172a", "sub": "#15803d"},
}

GROUPS: dict[str, dict[str, str]] = {
    "client":  {"fill": "#f8fafc", "stroke": "#cbd5e1", "label": "#475569"},
    "api":     {"fill": "#eff6ff", "stroke": "#bfdbfe", "label": "#1d4ed8"},
    "orch":    {"fill": "#f5f3ff", "stroke": "#ddd6fe", "label": "#6d28d9"},
    "core":    {"fill": "#f8fafc", "stroke": "#e2e8f0", "label": "#475569"},
    "online":  {"fill": "#f0fdf4", "stroke": "#bbf7d0", "label": "#15803d"},
    "offline": {"fill": "#fff7ed", "stroke": "#fed7aa", "label": "#c2410c"},
    "ext":     {"fill": "#f8fafc", "stroke": "#e2e8f0", "label": "#475569"},
}

FS_T, FS_S = 14.0, 11.5          # 标题 / 副标题字号
LH_T, LH_S = 20.0, 16.0          # 行高
PAD_X, PAD_Y = 15.0, 11.0
SUB_GAP = 5.0
MIN_W = 108.0
CUT = 13.0                       # 判断节点切角
FONT = '"Microsoft YaHei","PingFang SC","Noto Sans CJK SC","Hiragino Sans GB",sans-serif'


def text_w(s: str, fs: float) -> float:
    """估宽：CJK 按一个字等于字号，ASCII 按 0.55 倍。够用且不引外部字体度量。"""
    return sum(fs if ord(c) > 0x2E80 else fs * 0.55 for c in s)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Node:
    def __init__(self, nid: str, title: str, sub: str | None = None, style: str = "box"):
        self.id = nid
        self.style = style
        self.lines = title.split("\n")
        self.sublines = sub.split("\n") if sub else []
        widest = max([text_w(l, FS_T) for l in self.lines]
                     + [text_w(l, FS_S) for l in self.sublines])
        self.w = max(MIN_W, widest + 2 * PAD_X)
        self.h = 2 * PAD_Y + len(self.lines) * LH_T
        if self.sublines:
            self.h += SUB_GAP + len(self.sublines) * LH_S
        self.x = self.y = 0.0

    def draw(self) -> str:
        c = STYLES[self.style]
        st = f'fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="1.6"'
        if self.style == "start":
            shape = (f'<rect x="{self.x:.1f}" y="{self.y:.1f}" width="{self.w:.1f}" '
                     f'height="{self.h:.1f}" rx="{self.h / 2:.1f}" {st}/>')
        elif self.style == "decision":
            x, y, w, h = self.x, self.y, self.w, self.h
            shape = (f'<path d="M {x + CUT:.1f} {y:.1f} H {x + w - CUT:.1f} L {x + w:.1f} {y + CUT:.1f} '
                     f'V {y + h - CUT:.1f} L {x + w - CUT:.1f} {y + h:.1f} H {x + CUT:.1f} '
                     f'L {x:.1f} {y + h - CUT:.1f} V {y + CUT:.1f} Z" {st}/>')
        else:
            shape = (f'<rect x="{self.x:.1f}" y="{self.y:.1f}" width="{self.w:.1f}" '
                     f'height="{self.h:.1f}" rx="8" {st}/>')

        out = [shape]
        cy = self.y + PAD_Y + LH_T / 2
        for i, line in enumerate(self.lines):
            out.append(f'<text x="{self.x + self.w / 2:.1f}" y="{cy + i * LH_T:.1f}" '
                       f'text-anchor="middle" dominant-baseline="middle" font-size="{FS_T}" '
                       f'fill="{c["text"]}">{esc(line)}</text>')
        if self.sublines:
            sy = self.y + PAD_Y + len(self.lines) * LH_T + SUB_GAP + LH_S / 2
            for i, line in enumerate(self.sublines):
                out.append(f'<text x="{self.x + self.w / 2:.1f}" y="{sy + i * LH_S:.1f}" '
                           f'text-anchor="middle" dominant-baseline="middle" font-size="{FS_S}" '
                           f'fill="{c["sub"]}">{esc(line)}</text>')
        return "\n  ".join(out)

    # 连接点
    @property
    def top(self): return (self.x + self.w / 2, self.y)
    @property
    def bottom(self): return (self.x + self.w / 2, self.y + self.h)
    @property
    def left(self): return (self.x, self.y + self.h / 2)
    @property
    def right(self): return (self.x + self.w, self.y + self.h / 2)


class Fig:
    # side_gap 必须给得下「边标签的宽度」：中轴节点与侧边节点之间的连线是水平的，
    # 标签就落在这一段缝隙里，缝太窄标签会压住两侧的框。
    def __init__(self, title: str, row_gap: float = 34.0, side_gap: float = 74.0,
                 col_gap: float = 26.0, margin: float = 34.0):
        self.title = title
        self.row_gap = row_gap
        self.side_gap = side_gap
        self.col_gap = col_gap
        self.margin = margin
        self.rows: list[list[Node]] = []
        self._main: list[int | None] = []
        self.edges: list[tuple] = []
        self.groups: list[tuple] = []
        self.nodes: dict[str, Node] = {}

    def add(self, nodes: list[Node], main: int | None = None) -> None:
        """加一行。main=i 时第 i 个节点钉在中轴上，其余分列两侧。

        main 允许用负数（同 list 下标，惯例是 main=-1 表示「最后一个节点走中轴」）。
        这里必须先归一成非负下标再存：下面的布局用的是 row[:main] / row[main+1:] 这两段切片，
        而 main=-1 时 row[main+1:] 会变成 row[0:]——也就是**整行**，节点会被摆两遍、位置全乱。
        """
        if main is not None and main < 0:
            main += len(nodes)
        for n in nodes:
            self.nodes[n.id] = n
        self.rows.append(nodes)
        self._main.append(main)

    def group(self, label: str, row_from: int, row_to: int, palette: str) -> None:
        """给 [row_from, row_to] 这几行套一个底色分组框。"""
        self.groups.append((label, row_from, row_to, palette))

    def edge(self, src: str, dst: str, label: str | None = None,
             dashed: bool = False, from_side: str | None = None) -> None:
        self.edges.append((src, dst, label, dashed, from_side))

    # ---------- 排版 ----------
    def _layout(self) -> tuple[float, float, float]:
        y = self.margin + 46  # 让开标题
        max_left = max_right = 0.0
        for row, main in zip(self.rows, self._main):
            h = max(n.h for n in row)
            if main is None:
                total = sum(n.w for n in row) + self.col_gap * (len(row) - 1)
                x = -total / 2
                for n in row:
                    n.x = x
                    x += n.w + self.col_gap
            else:
                m = row[main]
                m.x = -m.w / 2
                # main 左侧的节点，从右往左依次外推
                cursor = m.x - self.side_gap
                for n in reversed(row[:main]):
                    n.x = cursor - n.w
                    cursor = n.x - self.side_gap
                cursor = m.x + m.w + self.side_gap
                for n in row[main + 1:]:
                    n.x = cursor
                    cursor = n.x + n.w + self.side_gap
            for n in row:
                n.y = y + (h - n.h) / 2
                max_left = max(max_left, -n.x)
                max_right = max(max_right, n.x + n.w)
            y += h + self.row_gap
        return max_left, max_right, y - self.row_gap + self.margin

    def _bbox(self, r0: int, r1: int) -> tuple[float, float, float, float]:
        xs0 = min(n.x for r in self.rows[r0:r1 + 1] for n in r)
        xs1 = max(n.x + n.w for r in self.rows[r0:r1 + 1] for n in r)
        ys0 = min(n.y for r in self.rows[r0:r1 + 1] for n in r)
        ys1 = max(n.y + n.h for r in self.rows[r0:r1 + 1] for n in r)
        return xs0, ys0, xs1, ys1

    def render(self) -> str:
        max_left, max_right, height = self._layout()
        pad = 26.0
        cx = self.margin + pad + max_left
        width = cx + max_right + pad + self.margin

        # 平移到画布坐标
        for row in self.rows:
            for n in row:
                n.x += cx
        for i, (label, r0, r1, pal) in enumerate(self.groups):
            x0, y0, x1, y1 = self._bbox(r0, r1)
            self.groups[i] = (label, x0 - 16, y0 - 30, x1 + 16, y1 + 14, pal)

        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
               f'viewBox="0 0 {width:.0f} {height:.0f}" font-family={FONT!r}>',
               '<defs>',
               '<marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
               'markerHeight="7" orient="auto-start-reverse">'
               '<path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker>',
               '<marker id="arwd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
               'markerHeight="7" orient="auto-start-reverse">'
               '<path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/></marker>',
               '</defs>',
               f'<rect width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>',
               f'<text x="{self.margin:.0f}" y="{self.margin + 18:.0f}" font-size="17" '
               f'font-weight="600" fill="#0f172a">{esc(self.title)}</text>']

        # 分组底色（只画框）
        for label, x0, y0, x1, y1, pal in self.groups:
            c = GROUPS[pal]
            out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" '
                       f'rx="12" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="1.2"/>')

        # 连线（画在节点下面）
        for src, dst, label, dashed, from_side in self.edges:
            out.append(self._edge_path(self.nodes[src], self.nodes[dst], label, dashed, from_side))

        for row in self.rows:
            for n in row:
                out.append("  " + n.draw())

        # 分组标题**最后画**，并且垫一层与分组同色的不透明底。
        # 连线是从上一行下来、直插节点顶部的，必然横穿分组框的上边缘——正好是标题所在的
        # 那一条带。标题先画就会被线划开；放在最上层再加底色，线就从标题后面「穿过」，
        # 视觉上断开，比调坐标躲开它可靠得多（改文案时也不用重新算位置）。
        for label, x0, y0, x1, y1, pal in self.groups:
            c = GROUPS[pal]
            tw = text_w(label, 12.5) + 16
            out.append(f'<rect x="{x0 + 8:.1f}" y="{y0 + 6:.1f}" width="{tw:.1f}" height="21" '
                       f'rx="6" fill="{c["fill"]}"/>')
            out.append(f'<text x="{x0 + 16:.1f}" y="{y0 + 21:.1f}" font-size="12.5" font-weight="600" '
                       f'fill="{c["label"]}">{esc(label)}</text>')
        out.append("</svg>")
        return "\n".join(out) + "\n"

    def _edge_path(self, a: Node, b: Node, label, dashed, from_side) -> str:
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        col = "#94a3b8" if dashed else "#64748b"
        mk = "arwd" if dashed else "arw"

        if from_side == "right" or (abs(a.y - b.y) < 4 and b.x > a.x):
            x1, y1 = a.right
            x2, y2 = b.left
            mid = (x1 + x2) / 2
            d = f"M {x1:.1f} {y1:.1f} C {mid:.1f} {y1:.1f} {mid:.1f} {y2:.1f} {x2:.1f} {y2:.1f}"
            lx, ly = mid, (y1 + y2) / 2
        elif from_side == "left" or (abs(a.y - b.y) < 4 and b.x < a.x):
            x1, y1 = a.left
            x2, y2 = b.right
            mid = (x1 + x2) / 2
            d = f"M {x1:.1f} {y1:.1f} C {mid:.1f} {y1:.1f} {mid:.1f} {y2:.1f} {x2:.1f} {y2:.1f}"
            lx, ly = mid, (y1 + y2) / 2
        else:
            x1, y1 = a.bottom
            x2, y2 = b.top
            dy = max(16.0, min(46.0, (y2 - y1) / 2))
            d = (f"M {x1:.1f} {y1:.1f} C {x1:.1f} {y1 + dy:.1f} {x2:.1f} {y2 - dy:.1f} "
                 f"{x2:.1f} {y2:.1f}")
            lx, ly = (x1 + x2) / 2, (y1 + y2) / 2

        s = [f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.6"{dash} '
             f'marker-end="url(#{mk})"/>']
        if label:
            w = text_w(label, 11.5) + 10
            s.append(f'<rect x="{lx - w / 2:.1f}" y="{ly - 10:.1f}" width="{w:.1f}" height="20" '
                     f'rx="6" fill="#ffffff" stroke="#e2e8f0" stroke-width="1"/>')
            s.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
                     f'dominant-baseline="middle" font-size="11.5" fill="#475569">{esc(label)}</text>')
        return "\n  ".join(s)


# =====================================================================
# 图 1 · 系统架构
# =====================================================================
def fig1() -> Fig:
    f = Fig("图 1 · 系统架构（分层与可替换边界）", row_gap=30)
    f.add([Node("c1", "浏览器聊天页", "app/static/index.html", "box"),
           Node("c2", "curl / Postman / Jmeter", None, "box")])
    f.group("客户端", 0, 0, "client")

    f.add([Node("a1", "/health", None, "api"),
           Node("a2", "/chat", "/chat/stream", "api"),
           Node("a3", "/knowledge/upload", "/knowledge/list", "api"),
           Node("a4", "/role/list", "/role/{id} · /role", "api")])
    f.group("接入层 · FastAPI（app/main.py + app/api/）", 1, 1, "api")

    f.add([Node("l3", "RAGPipeline · app/core/pipeline.py",
                "检索 → 重排 → 提示词 → LLM → 后处理 → 写回记忆", "orch")])
    f.group("编排层 · 只编排流程，不实现具体能力", 2, 2, "orch")

    f.add([Node("m1", "在线链路", "retrieve/ · reranker · embedding\nllm · prompt/ · postprocess", "online"),
           Node("m2", "离线链路", "document/\nparser·ocr·cleaner\nchunker·enhance", "offline"),
           Node("m3", "存储", "milvus_store 向量+正文\nsql_store 登记\nmemory 短期记忆", "store")])
    f.group("核心模块 · app/core/", 3, 3, "core")

    # 外部依赖的**排列顺序**是按「谁用它」放的：前两个归在线/离线链路（左侧两列），
    # 后三个归存储（右侧）。这样上面的虚线到下面几乎不交叉——纯粹为了可读性，
    # 不代表这几样东西之间有什么分组关系。
    f.add([Node("e1", "Ollama", "LLM qwen3 · Embedding bge-m3", "ext"),
           Node("e2", "bge-reranker", "供在线链路精排（可选）", "ext"),
           Node("e3", "Milvus", "Lite / 独立服务", "ext"),
           Node("e4", "关系库", "SQLite / MySQL", "ext"),
           Node("e5", "短期记忆", "内存 / Redis", "ext")])
    f.group("外部依赖 · 每一种实现都能在 .env 里换掉", 4, 4, "ext")

    f.edge("c1", "a2")
    f.edge("c2", "a2")
    f.edge("a2", "l3")
    f.edge("a3", "l3", dashed=True)
    f.edge("l3", "m1")
    f.edge("l3", "m2")
    f.edge("l3", "m3")
    f.edge("m1", "e1", dashed=True)
    f.edge("m2", "e1", dashed=True)
    f.edge("m3", "e3", dashed=True)
    f.edge("m3", "e4", dashed=True)
    f.edge("m3", "e5", dashed=True)
    return f


# =====================================================================
# 图 2 · 离线入库
# =====================================================================
def fig2() -> Fig:
    f = Fig("图 2 · 离线入库：一份文档怎么变成可检索的知识", row_gap=30)
    f.add([Node("s", "文档：PDF / 图片 / md / txt", None, "start")])
    f.add([Node("p", "parser.py 解析", None, "offline")])
    f.edge("s", "p")

    f.add([Node("q1", "PDF 能直接抽出文本？", None, "decision"),
           Node("empty", "返回空文本 → 接口报 400", "扫描件且 OCR 关闭", "error")], main=0)
    f.edge("p", "q1")
    f.edge("q1", "empty", "OCR 关闭", from_side="right", dashed=True)

    # 三条支路并排、再一起汇入清洗——解析器的三级兜底就是这么个形状。
    # 中间那条（纯文本）是主路，所以它对齐中轴，左右两条是兜底。
    f.add([Node("ocr", "ocr.py RapidOCR 逐页识别", "单页失败只记警告，不连累整篇", "offline"),
           Node("t", "纯文本", None, "offline"),
           Node("pl", "退回 pdfplumber 抽表格", "文本过少时兜底", "offline")])
    f.edge("q1", "ocr", "无文字层")
    f.edge("q1", "t", "能")
    f.edge("q1", "pl", "文本过少")

    f.add([Node("clean", "cleaner.py 清洗", "去水印 / 页眉页脚 / 空白归一", "offline")])
    f.edge("t", "clean")
    f.edge("ocr", "clean")
    f.edge("pl", "clean")

    f.add([Node("chunk", "chunker.py 分块",
                "fixed / sentence / paragraph / title / semantic ＋ 重叠", "offline")])
    f.edge("clean", "chunk")

    # 「丢弃」是旁路：两条判断的「是」都指向它，主路继续往下走。
    # 之前把它画在主路上，看起来像一步正常流程——这是这张图改版的原因之一。
    f.add([Node("q3", "低于 MIN_CHUNK_CHARS？", None, "decision"),
           Node("drop", "丢弃", "低质量块 / 重复内容", "error")], main=0)
    f.edge("chunk", "q3")
    f.edge("q3", "drop", "是", from_side="right")

    f.add([Node("q4", "内容哈希重复？", None, "decision")], main=0)
    f.edge("q3", "q4", "否")
    f.edge("q4", "drop", "是", from_side="right")

    f.add([Node("q5", "开启摘要？", "SUMMARY_ENABLED / --summary", "decision")], main=0)
    f.edge("q4", "q5", "否")

    f.add([Node("sum", "LLM 生成一句摘要", "存进 summary 字段（费 LLM）", "online")], main=0)
    f.edge("q5", "sum", "是")

    f.add([Node("emb", "embedding.py 向量化", "bge-m3 → 1024 维，L2 归一化", "offline")])
    f.edge("q5", "emb", "否")
    f.edge("sum", "emb")

    f.add([Node("store", "ingest.store_chunks",
                "① 写 Milvus：向量 + 正文　② 写关系库：登记这份文档", "store")])
    f.edge("emb", "store")

    f.add([Node("inv", "hybrid_retriever.invalidate(role_id)",
                "失效本进程的 BM25 索引缓存", "store")])
    f.edge("store", "inv")

    f.add([Node("end", "这些 chunk 从此可被检索", None, "ok")])
    f.edge("inv", "end")
    return f


# =====================================================================
# 图 3 · 在线问答
# =====================================================================
def fig3() -> Fig:
    f = Fig("图 3 · 在线问答：一次提问的完整旅程", row_gap=28)
    f.add([Node("u", "用户提问：POST /chat 或 /chat/stream", None, "start")])
    f.add([Node("v", "role_id 在角色表里？", None, "decision"),
           Node("r404", "404 未知角色", "须在开始吐流前判断", "error")], main=0)
    f.edge("u", "v")
    f.edge("v", "r404", "否", from_side="right")

    f.add([Node("rw", "QUERY_REWRITE_ENABLED？", "默认 false", "decision"),
           Node("rew", "query_rewrite.py", "改写/扩写成多条 query\n（多一趟 LLM）", "online")], main=0)
    f.edge("v", "rw", "是")
    f.edge("rw", "rew", "是", from_side="right")
    f.edge("rew", "ret", "多条 query")

    f.add([Node("e503", "503 服务暂时不可用", "依赖掉线（仅吐流之前）", "error"),
           Node("ret", "检索：稠密 + BM25 → RRF → 精排截断", "内部见 图 4", "online")], main=1)
    f.edge("rw", "ret", "否")
    f.edge("ret", "e503", "掉线", from_side="left", dashed=True)

    f.add([Node("hist", "memory.get_history", "按「会话 + 角色」取短期记忆", "store")])
    f.add([Node("build", "prompt/templates.build_messages",
                "人设 + 知识片段 + 多轮历史 + 当前问题\n按字符预算裁剪", "online")])
    f.add([Node("llm", "llm.py 生成", "流式 / 非流式", "online"),
           Node("eerr", "SSE error 事件", "流已开始，HTTP 已是 200\n状态码用不了", "error")], main=0)
    f.add([Node("post", "postprocess.py 清洗", "去推理段 / 去幻觉引用 / 空白归一", "online")])
    f.add([Node("empty", "答案为空？", "模型没产出", "decision"),
           Node("err", "EmptyAnswerError", "非流式 502 / 流式 error 事件\n且不写进记忆", "error")], main=0)
    f.add([Node("mem", "memory.add 写回一问一答", "支撑多轮对话", "store")])
    f.add([Node("out", "返回 answer + sources", None, "ok")])

    f.edge("ret", "hist")
    f.edge("hist", "build")
    f.edge("build", "llm")
    f.edge("llm", "eerr", "失败", from_side="right", dashed=True)
    f.edge("llm", "post")
    f.edge("post", "empty")
    f.edge("empty", "err", "是", from_side="right")
    f.edge("empty", "mem", "否")
    f.edge("mem", "out")
    return f


# =====================================================================
# 图 4 · 检索内部展开
# =====================================================================
def fig4() -> Fig:
    f = Fig("图 4 · 检索内部：混合召回 → 融合 → 精排", row_gap=28)
    f.add([Node("q", "问题", None, "start")])
    f.add([Node("rw", "启用 Query 改写？", "默认关", "decision")], main=0)
    f.add([Node("qs", "改写成多条 query", "原问题恒在首位", "online"),
           Node("q1", "单条原问题", "默认路径", "online")], main=-1)
    f.edge("q", "rw")
    f.edge("rw", "qs", "是", from_side="left")
    f.edge("rw", "q1", "否", from_side="right")

    f.add([Node("each", "每条 query 各跑两路召回", None, "online")])
    f.edge("qs", "each")
    f.edge("q1", "each")

    f.add([Node("d1", "① 稠密召回", "embedding.embed_query → 1024 维", "online"),
           Node("b1", "② BM25 召回", "jieba 分词", "online")])
    f.edge("each", "d1")
    f.edge("each", "b1")
    f.add([Node("d2", "milvus_store.search", "role_id 过滤 + SCORE_THRESHOLD", "store"),
           Node("b2", "BM25Index.search", "每进程缓存 · 故意不设阈值", "store")])
    f.edge("d1", "d2")
    f.edge("b1", "b2")

    f.add([Node("rrf", "③ RRF 融合",
                "只看排名、不看分数，k=60 → 天然吸收两路不同量纲", "online")])
    f.edge("d2", "rrf")
    f.edge("b2", "rrf")

    f.add([Node("pool", "④ 池宽截断到 RERANK_POOL",
                "融合分已排序，取最好的 N 条\n（开改写时条数可达池宽数倍，不截会超时）", "online")])
    f.edge("rrf", "pool")

    f.add([Node("rr", "RERANKER？", None, "decision"),
           Node("bge", "bge-reranker 交叉编码", "逐对打分；不可用/分数残缺则降级", "online")], main=0)
    f.edge("pool", "rr")
    f.edge("rr", "bge", "bge", from_side="right")
    f.add([Node("sf", "按融合分排序", "score_fusion（默认）", "online")], main=0)
    f.edge("rr", "sf", "score_fusion")
    f.edge("bge", "sf", "降级", from_side="right", dashed=True)

    f.add([Node("topk", "⑤ 截断到 TOP_K", None, "online")])
    f.edge("sf", "topk")
    f.add([Node("out", "chunks → 交给提示词模板", None, "ok")])
    f.edge("topk", "out")
    return f


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in (("fig1_architecture", fig1), ("fig2_ingest", fig2),
                          ("fig3_chat", fig3), ("fig4_retrieval", fig4)):
        svg = builder().render()
        path = OUT_DIR / f"{name}.svg"
        path.write_text(svg, encoding="utf-8")
        print(f"写出 {path}（{len(svg.splitlines())} 行）")


if __name__ == "__main__":
    main()
