# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件说明：表格结构化与定位（工单三核心）。HTML 表格 → 单元格矩阵 → 表头识别 → 每行一个块，并输出可读定位串。
import difflib                                                                     # 判定 OCR 变体的水印串
import re                                                                          # 解析表格标签、清洗水印噪声
from functools import lru_cache                                                     # 缓存水印判定结果

_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)                            # 表格行
_CELL = re.compile(r"<t[dh]([^>]*)>(.*?)</t[dh]>", re.S | re.I)                   # 单元格，含属性串用于读 rowspan/colspan
_TAG = re.compile(r"<[^>]+>")                                                     # 任意 HTML 标签
# 水印噪声：两篇招股说明书都是带水印版本，水印为"八维教育 / 人工智能 / rengongzhineng liumin"斜向重复，
# MinerU 的表格识别读的是页面图像，必然把水印读进单元格（实测 项目名称gongzhin、20,658.33ong、合计教育、股票种类敏）。
# 试过 MinerU 的 -m txt 走文本层，水印依旧出现——所以只能在结构化阶段清洗。
# 下面两张词表用实测比对得出：拉丁片段全部是 rengongzhineng/liumin 的子串；八维、数育在干净文本层中出现 0 次。
_NOISE = re.compile(r"(?<=[一-鿿])[a-z]{2,9}|[a-z][a-z0-9]{1,8}(?=[一-鿿])|(?<=[0-9])[a-z]{2,9}")  # 紧贴中文或数字的拉丁串
_KEEP = {"ms", "mbps", "qps", "mp4", "ios", "ps", "mb", "kb", "gb", "hz", "khz", "mhz", "ghz",  # 合法单位与技术词，避免误删
         "dpi", "ppi", "min", "mm", "cm", "km", "kg", "kw", "sdk", "api", "it", "ai", "usb", "led", "lcd"}
_WM_LATIN = "rengongzhinengliumin"                                                # 水印拉丁文基准串
_WM_ANY = re.compile(r"八维教育|维教育|维教|八维|数育")                              # 干净文本层中出现 0 次的词，可在任意位置剔除
_WM_EDGE = re.compile(r"^(?:人工智能)+|(?:人工智能)+$")                            # 人工智能在正文中确有合法出现，故只剔单元格首尾的


@lru_cache(maxsize=4096)                                                          # 同一个词只判定一次，避免每格重复算相似度
def _is_wm_latin(tok: str) -> bool:                                               # 判断拉丁串是否为水印
    """水印串本身、其子串、以及 OCR 变体都算水印；engineering、ionic 这类整词相似度低，不会被误删。"""
    if tok in _KEEP: return False                                                 # 词表内的合法词一律保留
    return tok in _WM_LATIN or difflib.SequenceMatcher(None, tok, _WM_LATIN).ratio() > 0.6  # 子串命中或相似度超阈值


def clean_cell(text: str) -> str:                                                 # 单元格文本清洗
    """去标签、反转义、剔水印、压空白，保证表头与数值可被 BM25 原样命中。"""
    import html as _html                                                          # 局部导入，避免与模块名冲突
    s = _html.unescape(_TAG.sub(" ", text or ""))                                 # 去标签并还原 &amp; 等实体
    s = _NOISE.sub(lambda m: m.group(0) if m.group(0) in _KEEP else "", s)         # 剔紧贴中文/数字的短噪声
    s = re.sub(r"[a-z][a-z0-9]*", lambda m: "" if _is_wm_latin(m.group(0)) else m.group(0), s)  # 剔独立的水印拉丁词
    s = _WM_EDGE.sub("", _WM_ANY.sub("", s))                                       # 先剔任意位置的中文水印，再剔首尾的"人工智能"
    return re.sub(r"\s+", " ", s).strip()                                          # 压空白并去首尾空格


def _attr(attrs: str, name: str, default: int = 1) -> int:                        # 读 rowspan/colspan
    """属性缺失或非法时回落默认值 1，单元格跨度异常不至于让整张表错位。"""
    m = re.search(rf'{name}\s*=\s*"?(\d+)', attrs or "", re.I)                    # 匹配 name="2" 或 name=2 两种写法
    return int(m.group(1)) if m else default                                      # 命中取数值，否则给默认


def to_matrix(html: str) -> list:                                                 # HTML → 规整二维矩阵
    """展开 rowspan/colspan 补齐空缺：不展开时"关联方名称/持股比例/与本公司关系"三列会与数据行错位。"""
    grid, carry = [], {}                                                          # grid 为结果矩阵；carry 记录纵向合并待填的格子
    for row_html in _ROW.findall(html or ""):                                     # 逐个表格行
        r, row, c = len(grid), [], 0                                              # 当前行号、当前行、列游标
        grid.append(row)                                                          # 先把当前行挂进矩阵
        for attrs, body in _CELL.findall(row_html):                               # 逐个单元格
            while (r, c) in carry: row.append(carry.pop((r, c))); c += 1           # 该位被上方 rowspan 占用则先填继承值
            txt = clean_cell(body)                                                # 清洗文本
            rs, cs = _attr(attrs, "rowspan"), _attr(attrs, "colspan")              # 读跨行与跨列
            row += [txt] + [""] * (cs - 1)                                        # 横向铺开：首格写值，其余留空占位
            for dr in range(1, rs):                                               # 纵向铺开
                for dc in range(cs): carry[(r + dr, c + dc)] = txt                 # 交给后续行自动补齐
            c += cs                                                               # 列游标右移
        while (r, c) in carry: row.append(carry.pop((r, c))); c += 1               # 行尾仍可能有纵向合并待填
    return [row for row in grid if any(x.strip() for x in row)]                    # 丢掉全空行


def is_header(row: list) -> bool:                                                 # 表头行判定
    """各格都短且不含数字才算表头；像"中文名称/武汉兴图新科…"那种键值对首行会被判成数据，避免丢内容。"""
    vals = [c.strip() for c in row if c.strip()]                                   # 非空格子
    return len(vals) >= 2 and all(len(v) <= 12 for v in vals) and not any(re.search(r"\d", v) for v in vals)  # 短且纯非数字


def structure_tables(blocks: list) -> list:                                        # 表格块 → 行块 + 整表块
    """每张表产出 N 个行块（精确定位到行）与 1 个整表块（覆盖"分别有哪些"这类跨行问题）。"""
    out, seen = [], {}                                                             # out 收集结果；seen 统计同页表序，用于生成表编号
    for b in blocks:                                                               # 遍历解析出的块
        if b.get("type") != "table": out.append(b); continue                        # 非表格块原样透传
        grid = to_matrix(b.get("html") or "")                                       # 还原单元格矩阵
        if not grid: continue                                                       # 空表直接丢弃
        key = (b["doc"], b["page_no"])                                              # 同页多表，按出现顺序编号
        seen[key] = seen.get(key, 0) + 1                                            # 该页第几张表
        tid = f'{b["doc"]}#p{b["page_no"]}#t{seen[key]}'                            # 表格唯一标识，前端与评测按它判定命中
        head = grid[0] if is_header(grid[0]) else []                                # 识别表头，识别不出就整表当数据
        body = grid[1:] if head else grid                                            # 数据行
        ctx = " | ".join(x for x in (b.get("section"), b.get("caption")) if x)       # 章节名 + 表题构成的检索语境
        lines = [" | ".join(x for x in (ctx, " | ".join(x for x in head if x)) if x)]  # 整表块首行写语境与表头
        for i, row in enumerate(body, 1):                                            # 逐数据行生成行块
            pairs = [f"{h}:{v}" if h else v for h, v in zip(head + [""] * len(row), row) if v.strip()]  # 表头:值 成对，缺表头时只留值
            text = "\n".join(x for x in (ctx, " | ".join(pairs)) if x)               # 行块文本，自带语境
            out.append(dict(doc=b["doc"], type="table_row", page_no=b["page_no"],    # 行块：类型、页码
                            content=text, caption=b.get("caption", ""),              # 内容与表题
                            table_id=tid, row_index=i, headers=" | ".join(x for x in head if x),  # 表格定位三要素
                            locator=f'{b["doc"]} P{b["page_no"]} 表{seen[key]} 第{i}行'))  # 可读定位串，前端直接展示
            lines.append(" | ".join(x for x in row if x.strip()))                    # 同时累积整表文本
        out.append(dict(doc=b["doc"], type="table", page_no=b["page_no"],            # 整表块：承载"有哪些/分别是多少"这类需跨行汇总的问题
                        content="\n".join(lines), caption=b.get("caption", ""),      # 全文内容与表题
                        table_id=tid, row_index=0, headers=" | ".join(x for x in head if x),  # 定位元数据，行号 0 表示整表
                        locator=f'{b["doc"]} P{b["page_no"]} 表{seen[key]}（共{len(body)}行）'))  # 可读定位串
    return out                                                                     # 返回行块与整表块
