"""
表格解析模块（工单3 核心）
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

为什么需要单独一个模块
----------------------
工单2 的表格处理是「把 pdfplumber 的 extract_tables() 结果直接转成 Markdown」。
在这份《招股说明书2.pdf》上实测，这个做法有两个硬伤：

1) **表格根本抽不到。** 工单2 用「一行里出现 ≥3 个数字的行数 ≥3」来决定是否抽表格。
   而第 157 页的「存在控制关系的关联方」「不存在控制关系的关联方」两张表里
   只有 ``42.35%`` 这一个百分数，判定为 False —— **整张表被跳过**。
   而工单验收题 id=3、id=4 的答案**全部在这两张表里**。
   → 结论：用「数字密度」猜表格，在这类「名称 + 关系描述」表上会漏干净。

2) **抽到了也是烂的。** 该 PDF 的表格用的是**双线边框**，pdfplumber 会把两条线之间的
   缝隙（实测 5.4pt）也当成一列。于是：

       x 边界: 84.5 | 89.9 | 155.9 | 161.3 | 166.7 | 362.6 | 368.0 | 373.5 | 505.4 | 511.7
       间距:      5.4   66.0     5.4     5.4   195.9     5.4     5.5   131.9     6.3

   三列的表被切成九列，中间夹着 5~6pt 的「缝隙列」。
   更糟的是**窄数字会掉进缝隙列**：``序号`` 表头落在第 2 列，而数据 ``1`` 落在第 1 列 ——
   表头与数据**错位**，行列对应关系实际已经断了。

本模块的做法
------------
不信任 pdfplumber 切好的网格，而是**从词级 x 坐标重建列**：

    ① 取表格 bbox 内的所有词（含精确 x0/x1/top）
    ② 取该表格的所有单元格 x 边界，按 8pt 阈值**合并缝隙边界** → 得到真实列边界
    ③ 每个词按 x 中心落到某一列
    ④ 词按 top 聚成行；只有单列有内容的行视为上一行的**续行**（处理单元格内换行）
    ⑤ 首行（或前几行）作为表头

为什么 8pt 是安全的：实测缝隙是 5.4~6.3pt，而最小真实列宽 66pt，两者差一个数量级。
阈值可在 config 里调（`TABLE_GAP_MERGE_PT`）。

输出形态
--------
每张表输出「**行级语义文本**」，而不是一张 Markdown 表：

    关联方名称：赵马克；持股比例：42.35%；与本公司关系：公司控股股东

原因是检索与生成真正需要的是**列名与值的绑定关系**。把列名写进每一行，
既让 BM25 能命中「持股比例」这类列名，也让生成模型不必自己去数列。
Markdown 表也一并保留（`to_markdown`），供人工核对。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# 工单编号（工单3「备注」要求代码注释包含它）
WORK_ORDER_NO_TABLE = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

# 缝隙边界合并阈值（pt）。实测双线边框缝隙 5.4~6.3pt，最小真实列宽 66pt。
GAP_MERGE_PT = 8.0

# 同一行的 top 容差（pt）。正文行高约 15pt，取 1/3 足够区分行、又不把同一行拆开。
ROW_TOL_PT = 5.0

# 关键词 → 该表可能是什么表（用于给表格打个类型标签，便于检索时区分）
_TABLE_KIND_RULES = [
    (("关联方名称", "与本公司关系"), "关联方表"),
    (("序号", "项目名称", "计划总投资"), "募投项目表"),
    (("发行股数", "每股面值"), "本次发行概况表"),
    (("项目", "金额"), "费用/金额表"),
    (("年度", "资产", "负债"), "财务附表"),
]


@dataclass
class Table:
    """一张结构化后的表格。"""

    page: int
    index: int
    headers: list[str]
    rows: list[list[str]]
    doc: str = ""
    caption: str = ""          # 表格上方最近的引导句，如「1、存在控制关系的关联方」
    n_cols: int = 0
    kind: str = ""             # 表类型标签（启发式）
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)   # 表格区域，用于把这块从正文里剔除
    raw_col_gaps: list[float] = field(default_factory=list)  # 合并前的边界间距，留档便于核对

    # ---------------------------------------------------------------- 输出

    def semantic_rows(self) -> list[str]:
        """
        行级语义文本：把列名绑到值上。

        这是**真正喂给检索与生成**的形态。空单元格直接跳过（不输出「列名：」）。
        """
        out: list[str] = []
        for r in self.rows:
            parts = []
            for h, v in zip(self.headers, r):
                v = (v or "").strip()
                if not v:
                    continue
                h = (h or "").strip()
                parts.append(f"{h}：{v}" if h else v)
            if parts:
                out.append("；".join(parts))
        return out

    def to_text(self, max_rows: int = 40) -> str:
        """
        整表文本（含表头、表名、来源页码），可直接作为一个检索单元。

        max_rows 截断防超长表把上下文预算吃爆；被截断时明确写出还有多少行没展示，
        而不是悄悄丢掉 —— 悄悄丢掉会让模型以为表就这么长。
        """
        head = f"【表】{self.caption or self.kind or '表格'}（第{self.page}页）"
        if self.kind and self.caption and self.kind not in self.caption:
            head += f"[{self.kind}]"
        lines = [head]
        rows = self.semantic_rows()
        lines.extend(rows[:max_rows])
        if len(rows) > max_rows:
            lines.append(f"（本表共 {len(rows)} 行，此处省略后 {len(rows) - max_rows} 行）")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Markdown 形态，仅供人工核对（不进索引）。"""
        w = len(self.headers)
        md = ["| " + " | ".join(self.headers) + " |", "|" + "---|" * w]
        for r in self.rows:
            md.append("| " + " | ".join((r + [""] * w)[:w]) + " |")
        return "\n".join(md)

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "index": self.index,
            "doc": self.doc,
            "caption": self.caption,
            "kind": self.kind,
            "headers": self.headers,
            "n_rows": len(self.rows),
            "n_cols": self.n_cols,
            "semantic_rows": self.semantic_rows(),
        }


# ------------------------------------------------------------------ 列重建


def merge_gap_edges(edges: list[float], gap_pt: float = GAP_MERGE_PT) -> list[float]:
    """
    把「间距小于 gap_pt」的相邻边界并成一条，返回真实列边界。

    输入应是排序后的 x 边界（来自单元格的 x0/x1）。输出仍为升序，
    首尾保留原极值（表格左右外框不能丢）。

    这一步就是修「双线边框被当列」的关键：缝隙 5.4pt 会被并掉，而 66pt 的真实列间距不受影响。
    """
    if not edges:
        return []
    xs = sorted(set(round(e, 1) for e in edges))
    if len(xs) == 1:
        return xs

    groups: list[list[float]] = [[xs[0]]]
    for prev, cur in zip(xs, xs[1:]):
        if cur - prev < gap_pt:
            groups[-1].append(cur)
        else:
            groups.append([cur])

    # 每组取代表值：首组用最小值（表格左外框），末组用最大值（右外框），
    # 中间组用组内均值（线缝的中心）。
    out: list[float] = []
    n = len(groups)
    for i, g in enumerate(groups):
        if i == 0:
            out.append(min(g))
        elif i == n - 1:
            out.append(max(g))
        else:
            out.append(sum(g) / len(g))
    return sorted(set(out))


def _col_index(x: float, bounds: list[float]) -> int:
    """按 x 落列。落在边界上归右侧列；超出范围夹到首/末列。"""
    if x <= bounds[0]:
        return 0
    for i in range(len(bounds) - 1):
        if bounds[i] <= x < bounds[i + 1]:
            return i
    return len(bounds) - 2


def _cluster_1d(values: list[float], tol: float) -> list[float]:
    """把一组标量按 tol 聚成簇，返回每簇均值（升序）。用于求表格的横线位置。"""
    out: list[list[float]] = []
    for v in sorted(values):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [sum(g) / len(g) for g in out]


def build_grid(words: list[dict], bounds: list[float],
               cells: list[tuple[float, float, float, float]] | None = None,
               row_tol: float = ROW_TOL_PT) -> list[list[str]]:
    """
    重建二维网格。**优先按单元格 bbox 分派词，没有单元格信息时才退化到「按 top 聚行」**。

    words 需含 x0/x1/bottom/top/text（pdfplumber 的 extract_words 输出）；
    cells 为 pdfplumber ``Table.cells``（每个单元格的 (x0, top, x1, bottom)）。

    ---------------------------------------------------------------
    为什么必须按单元格分派（实测踩到的坑）
    ---------------------------------------------------------------
    「词按全局 top 聚行」在一种版式上会**张冠李戴**：单元格内的**标签垂直居中**，
    而值从单元格**顶部**开始排。第 24 页「(一)发行基本情况」表实测坐标：

        top=216.99  每股发行价格        ← 标签（居中）
        top=216.99  通过向询价对象询价或中国证监会认可的其他定价方式确定
        top=243.09  2.18元(按截至2010年6月30日经审计净资产除以发行前总
        top=249.87  发行前每股净资产    ← 下一个标签（居中，比值的首行低 33pt）
        top=256.71  股本计算)

    按 top 聚行的结果是：

        每股发行价格 | 通过向询价对象询价或…确定2.18元(按…除以发行前总
        发行前每股净资产 | 股本计算)          ← 值与键完全错位

    而正确答案是「每股发行价格 = 通过向询价对象询价…确定」「发行前每股净资产 = 2.18元(…)」。
    改成按单元格分派后，单元格内的多行文本自然落到同一格并按 top 排序拼接，
    **续行合并（_merge_continuation_rows）在主路径上就再也不需要了**。
    """
    ncol = max(1, len(bounds) - 1)
    if not words:
        return []

    if cells:
        row_tops = _cluster_1d([c[1] for c in cells], row_tol)
        if row_tops:
            buckets: list[list[list[tuple[float, float, str]]]] = [
                [[] for _ in range(ncol)] for _ in row_tops
            ]
            plan: list[tuple[int, int, float, float, float, float]] = []
            for (x0, top, x1, bottom) in cells:
                r = min(range(len(row_tops)), key=lambda i: abs(row_tops[i] - top))
                c = _col_index((x0 + x1) / 2, bounds)
                plan.append((r, c, x0, top, x1, bottom))

            leftovers: list[dict] = []
            for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
                cx = (w["x0"] + w["x1"]) / 2
                cy = (w["top"] + w["bottom"]) / 2
                placed = False
                for (r, c, x0, top, x1, bottom) in plan:
                    if x0 - 1.5 <= cx <= x1 + 1.5 and top - 1.5 <= cy <= bottom + 1.5:
                        buckets[r][c].append((w["top"], w["x0"], w["text"]))
                        placed = True
                        break
                if not placed:      # 兼底：单元格 bbox 没盖住的词按列 + 最近横线归位
                    leftovers.append(w)
            for w in leftovers:
                cx = (w["x0"] + w["x1"]) / 2
                c = _col_index(cx, bounds)
                r = min(range(len(row_tops)), key=lambda i: abs(row_tops[i] - w["top"]))
                buckets[r][c].append((w["top"], w["x0"], w["text"]))

            return [
                [
                    " ".join(t for _, _, t in sorted(b)).strip() if b else ""
                    for b in row
                ]
                for row in buckets
            ]

    # ---- 退化路径：没有单元格信息 → 按 top 聚行 + 续行合并
    groups: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if groups and abs(w["top"] - groups[-1][0]["top"]) <= row_tol:
            groups[-1].append(w)
        else:
            groups.append([w])

    grid: list[list[str]] = []
    for g in groups:
        row_cells: list[list[str]] = [[] for _ in range(ncol)]
        for w in sorted(g, key=lambda w: w["x0"]):
            cx = (w["x0"] + w["x1"]) / 2
            row_cells[_col_index(cx, bounds)].append(w["text"])
        grid.append([" ".join(c).strip() for c in row_cells])

    return _merge_continuation_rows(grid)


def _merge_continuation_rows(grid: list[list[str]]) -> list[list[str]]:
    """
    合并「单元格内换行」产生的续行。

    判据：某行**只有一列**有内容，且上一行有 ≥2 列有内容 → 视为上一行的续行，
    把它追加到上一行对应的那一列。

    实测两例都能修对：
      * 「其他与主营业务相关」/「的营运资金」  → 合成「其他与主营业务相关的营运资金」
      * 「武汉市环保局已对该项…审批意见」四行   → 合成一个完整单元格
    """
    out: list[list[str]] = []
    for row in grid:
        filled = [i for i, c in enumerate(row) if c.strip()]
        if (len(filled) == 1 and out):
            prev_filled = [i for i, c in enumerate(out[-1]) if c.strip()]
            if len(prev_filled) >= 2 and filled[0] == prev_filled[-1]:
                j = filled[0]
                out[-1][j] = (out[-1][j] + row[j]).strip()
                continue
        out.append(list(row))
    return out


# ------------------------------------------------------------------ 表头


def split_header(grid: list[list[str]], max_header_rows: int = 3) -> tuple[list[str], list[list[str]]]:
    """
    分离表头与数据行。

    判据（顺序执行，缺一不可）——纯靠"第几行"是判不出来的，因为表头行与数据行的
    「非空单元格个数」往往完全一样（实测 id=2 的募投表就是 3 对 3）：

      ① **表头行里不能有"纯数字"单元格**。``1`` / ``3,393.40`` / ``(1,234)`` 一律算纯数字；
         ``2010年1-6月`` / ``(万元)`` / ``持有公司股份5%以上的股东`` 不算 —— 它们是列名或描述。
         这一条把「序号 1 / 项目名称 仓储及物流中心」这种首行数据挡在外面。
      ② **一旦下一行的非空列集合与本行完全相同，就停止收表头** —— 说明下面已经在重复数据行了。
         这一条把 p157 那种「全部行都只有 2 个非空单元格」的表头正确截断在第一行。
      ③ 最多收 ``max_header_rows`` 行，用于「列名跨两行」的表（如 p306 的
         「项目总投资(万元)」被排成两行，中间还夹着一行别的列名）。

    三条合起来实测能同时修对：p22 的募投表（1 行表头）、p157 的两张关联方表（1 行）、
    p306 的募投主表（3 行表头）。
    第一行就不像表头（含纯数字）时，退化为「列1/列2/...」，不硬凑。
    """
    if not grid:
        return [], []
    ncol = max(len(r) for r in grid)

    # 特例：两列的「键值表」（本次发行概况、发行费用概算这类）。
    # 它的第一行就是数据（``股票种类 | 人民币普通股(A股)``），没有列名行。
    # 判据：两列且第一行第二格明显偏长（>8 字）—— 列名很少这么长，
    # 而值（「人民币普通股(A股)」「人民币1.00元」）一定很长。
    # 不处理的话表头会变成 ``股票种类 | 人民币普通股(A股)``，
    # 于是数据行被输出成「股票种类：发行股数」—— 键与值张冠李戴。
    if ncol == 2 and grid and len((grid[0][1] or "").strip()) > 8:
        return ["字段", "内容"], [list(r) for r in grid]

    header_rows: list[list[str]] = []
    body_start = 0
    prev_pattern: tuple[int, ...] | None = None

    for i, row in enumerate(grid[:max_header_rows]):
        filled = tuple(j for j, c in enumerate(row) if c.strip())
        if not filled:
            break
        if any(_is_pure_number(row[j]) for j in filled):
            break                                    # ① 出现纯数字 → 已经是数据行
        if prev_pattern is not None and filled == prev_pattern:
            break                                    # ② 非空列集合没变 → 数据行在重复
        header_rows.append(row)
        prev_pattern = filled
        body_start = i + 1

    if not header_rows:
        headers = [f"列{i + 1}" for i in range(ncol)]
    else:
        headers = []
        for j in range(ncol):
            parts = [r[j].strip() for r in header_rows if j < len(r) and r[j].strip()]
            headers.append(" ".join(parts).strip())
        # 表头里的空列名补成「列N」，避免 semantic_rows 输出成「：值」
        headers = [h if h else f"列{j + 1}" for j, h in enumerate(headers)]

    return headers, grid[body_start:]


# 纯数字/金额：「1」「3,393.40」「(1,234)」「25.04%」都算；「2010年1-6月」「(万元)」不算
_PURE_NUM_RE = re.compile(r"^[\(\[]?[\d,]+(\.\d+)?[\)\]]?%?$")


def _is_pure_number(cell: str) -> bool:
    """判断单元格是否为「纯数值」——用于把数据行与表头行区分开。"""
    s = (cell or "").strip()
    if not s or len(s) > 16:
        return False
    return bool(_PURE_NUM_RE.match(s))



def guess_kind(headers: list[str], rows: list[list[str]]) -> str:
    """按表头关键词猜表类型（只用于展示与加权，不参与正确性判断）。"""
    joined = " ".join(headers)
    for keys, kind in _TABLE_KIND_RULES:
        if all(k in joined for k in keys):
            return kind
    return ""


def is_meaningful(headers: list[str], rows: list[list[str]]) -> bool:
    """
    过滤掉「抽出来的假表」（页眉页脚框、目录点线框等）。

    判据：至少 2 列、至少 1 个数据行、且数据行里至少有一个非空单元格。
    """
    if len(headers) < 2 or not rows:
        return False
    return any(any((c or "").strip() for c in r) for r in rows)


# ------------------------------------------------------------------ 入口


def _caption_above(page, bbox, max_len: int = 60, rows_below: int = 0) -> str:
    """
    取表格**正上方最近的短行**作为表名（如「1、存在控制关系的关联方」）。

    必须用 bbox 而不是「整页文本的最后一行」—— 一页里常有多张表，
    取最后一行会把别的表的表名或正文段落误当成本表表名。

    表名是极强的检索信号：问题问「存在控制关系的关联方是谁」，
    表名一模一样 —— 把它并进表格文本，BM25 直接命中。
    """
    try:
        words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    except Exception:  # noqa: BLE001
        return ""
    top_edge = bbox[1]
    above = [w for w in words if w["bottom"] <= top_edge + 1]
    if not above:
        return ""

    # 聚成行，取最靠近表格的那一行
    above.sort(key=lambda w: (w["top"], w["x0"]))
    lines: list[list[dict]] = []
    for w in above:
        if lines and abs(w["top"] - lines[-1][0]["top"]) <= ROW_TOL_PT:
            lines[-1].append(w)
        else:
            lines.append([w])
    if not lines:
        return ""
    tail = sorted(lines[-1], key=lambda w: w["x0"])
    text = " ".join(w["text"] for w in tail).strip()
    if not (2 <= len(text) <= max_len):
        return ""
    # 排除每页重复的页眉（公司名 + 招股说明书 / 招股意向书），它离表格也很近
    if re.search(r"招股(意向)?说明书|招股意向书", text):
        return ""
    return text



def parse_tables_in_page(page, page_no: int, page_text: str = "",
                         doc: str = "") -> list[Table]:
    """解析单页内的所有表格。page 为 pdfplumber 的 page 对象。"""
    try:
        found = page.find_tables()
    except Exception:  # noqa: BLE001  单页失败不能中断整份文档
        return []

    out: list[Table] = []
    for ti, t in enumerate(found):
        try:
            edges: list[float] = []
            for c in t.cells:
                if not c:
                    continue
                edges.append(c[0])
                edges.append(c[2])
            if len(edges) < 4:
                continue

            xs = sorted(set(round(e, 1) for e in edges))
            gaps = [round(b - a, 1) for a, b in zip(xs, xs[1:])]
            bounds = merge_gap_edges(edges)
            if len(bounds) < 3:      # 少于 2 列的表不处理
                continue

            crop = page.crop(t.bbox)
            words = crop.extract_words(use_text_flow=False, keep_blank_chars=False)
            words = [w for w in words if (w.get("text") or "").strip()]
            if not words:
                continue

            # 单元格 bbox（pdfplumber 的网格）——按格分派词，修「标签垂直居中导致键值错配」
            cell_boxes = [
                (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
                for c in (t.cells or [])
                if c
            ]
            grid = build_grid(words, bounds, cells=cell_boxes or None)
            if not grid:
                continue
            ncol = len(bounds) - 1
            grid = [(r + [""] * ncol)[:ncol] for r in grid]

            headers, body = split_header(grid)
            if not is_meaningful(headers, body):
                continue

            out.append(Table(
                page=page_no, index=ti, headers=headers, rows=body, doc=doc,
                caption=_caption_above(page, t.bbox),
                n_cols=ncol, kind=guess_kind(headers, body),
                bbox=tuple(round(float(v), 1) for v in t.bbox),
                raw_col_gaps=gaps,
            ))
        except Exception:  # noqa: BLE001
            continue
    return out


def text_outside_tables(page, boxes: list[tuple[float, float, float, float]],
                        row_tol: float = ROW_TOL_PT,
                        drop_lines: set[str] | None = None) -> str:
    """
    重建「页面正文 = 所有词 - 落在表格框内的词 - 表格的引导句」。

    为什么必须做第一步（摘掉表格框内的词）：pdfplumber / pypdf 的整页文本里，
    表格内容是被**展平成数据行**的（``赵马克 42.35% 公司控股股东``），
    并和上下文的冗长段落混在一起。如果只把表格另外索引一份，同一批事实就会以
    「高质量结构化块」和「低质量稀释段落」两种形态同时存在 ——
    检索时稀释段落可能先被召回，表格解析的收益就体现不出来，
    消融实验也就比不出东西。

    为什么还要做第二步（drop_lines 摘掉引导句）—— 这是实测补的一刀：
    表格上方的引导句（表名）已经被 ``_caption_above`` 抓进表格块头部了，
    却仍然留在正文里。第 22 页实测后果极典型：正文里留下一个
    「四、本次发行情况 / 五、募集资金用途 / 本次募集资金拟投资以下项目：」
    的**碎片块** —— 它和问题的字面重合度极高（因为那些字几乎原样出现在问题里），
    于是排到了第 1 名，而**真正装了数据的表格块连 top-8 都没进**。
    用户看到的是「检索命中了」，模型拿到的却是一个没有数据的标题。
    → 做法：表名既然是表的一部分，正文里就不再保留，同一份信息只有一种形态。
    """
    try:
        words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    except Exception:  # noqa: BLE001
        return ""

    def inside(w: dict) -> bool:
        cx = (w["x0"] + w["x1"]) / 2
        cy = (w["top"] + w["bottom"]) / 2
        for (x0, top, x1, bottom) in boxes:
            if x0 - 2 <= cx <= x1 + 2 and top - 2 <= cy <= bottom + 2:
                return True
        return False

    kept = [w for w in words if not inside(w)]
    if not kept:
        return ""

    kept.sort(key=lambda w: (w["top"], w["x0"]))
    lines: list[list[dict]] = []
    for w in kept:
        if lines and abs(w["top"] - lines[-1][0]["top"]) <= row_tol:
            lines[-1].append(w)
        else:
            lines.append([w])

    drop = {re.sub(r"\s+", "", s) for s in (drop_lines or set()) if s}
    out: list[str] = []
    for ln in lines:
        text = " ".join(w["text"] for w in sorted(ln, key=lambda w: w["x0"])).strip()
        if drop and re.sub(r"\s+", "", text) in drop:
            continue          # 表名 → 只保留在表格块里
        out.append(text)
    return "\n".join(out).strip()


def parse_pdf_tables(pdf_path: Path, doc: str = "",
                     page_texts: dict[int, str] | None = None,
                     verbose: bool = False,
                     want_text_outside: bool = False):
    """
    解析整份 PDF 的表格。

    对每一页都跑 find_tables（**不再用数字密度做前置筛选** —— 正是那个启发式
    漏掉了关联方表）。实测代价可接受：力源 350 页 15.3s、兴图 548 页 51.4s。

    want_text_outside=True 时额外返回 ``{页码: 剔掉表格区域后的正文}``（只含有表格的页）。
    """
    import pdfplumber

    pdf_path = Path(pdf_path)
    doc = doc or pdf_path.stem
    page_texts = page_texts or {}
    all_tables: list[Table] = []
    outside: dict[int, str] = {}

    with pdfplumber.open(str(pdf_path)) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            tabs = parse_tables_in_page(page, i, page_texts.get(i, ""), doc)
            all_tables.extend(tabs)
            if want_text_outside and tabs:
                boxes = [t.bbox for t in tabs]
                # 表名一并从正文剔除（它已经写进表格块的头部，保留会造成「有标题无数据」的碎片块）
                outside[i] = text_outside_tables(page, boxes, drop_lines={t.caption for t in tabs})
            if verbose and i % 50 == 0:
                print(f"  [{doc}] 表格解析进度 {i}/{total}，已得 {len(all_tables)} 张表")

    return (all_tables, outside) if want_text_outside else all_tables
