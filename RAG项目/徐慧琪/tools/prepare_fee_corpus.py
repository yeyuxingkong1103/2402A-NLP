"""从 _sources/ 的原始下载件重建 data/raw/fee/ 的语料文本。

存在的理由：段落边界**就是**入库后的检索片段边界 —— ingest_fee.split_paragraphs
只按空行切。所以「一段里放什么」不是排版问题而是检索契约问题，值得留成可重跑
的脚本，而不是一份手工编辑、来路不明的文本。

守则：**只做清洗与分段，一个字都不改写**。语料里的数字要能被费用回查
（技术方案 6.4 硬规矩①）逐字命中，改写等于把出处抹掉。

解析口径：对电子件**直接提取文本**（docx 读原生 XML、pdf 读文本层），不走
MinerU + PaddleOCR-VL —— 这是技术方案 §3.2「不设文本层直接抽取旁路」的
**例外**，经用户 2026-09-29 裁决并落档（设计文档 §一 / `_sources/README.md`）。
理由：政务公示件是电子件、文本层无损，而 OCR 对数字有误识风险，本功能恰恰
靠数字精确。**例外仅限电子件**，扫描件仍走 MinerU + PaddleOCR-VL。

用法：cd D:/xinzg6/fl && python tools/prepare_fee_corpus.py
"""
from __future__ import annotations

import html
import pathlib
import re
import zipfile

import pypdf

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "raw" / "fee" / "_sources"
OUT_DIR = ROOT / "data" / "raw" / "fee"

# 分节标记：这些行开启一个新段落，其余行归属上一段。
# 每份文件的层级不同，故逐份声明 —— 用一套通用规则去套四份异构文件，
# 只会调好一份、碰坏三份。
_NUM_CN = r"^[一二三四五六七八九十]+、"
_NUM_ARABIC = r"^\d+、"
_NUM_INSIDE = r"^（[一二三四五六七八九十]+）"

# 「编号后紧跟金额单位」的都不是编号，是价目表里的一行。智深的费率表写着
# 「7.4 万元+标的额 100 万元以上部分的4%」，`^\d+\.\d+` 会把它当成小节号，
# 于是整张表从这一行裂开：上一段守着「100万—1000万部分」，下一段只剩公式，
# 档位与费率分了家。回查只验数字在不在片段里，拦不住「档位用错」
_NOT_MONEY = r"(?!\s*(?:万元|元|%|％))"
_NUM_DECIMAL = rf"^\d+\.\d+{_NOT_MONEY}"

SPECS = [
    {
        "src": "_probe_sf87.docx", "kind": "docx",
        "out": "司法部等三部门关于规范律师服务收费的意见（司发通〔2021〕87号）.txt",
        # 条款体：每个「（N）」条独立成段；顶级的「一、二、」也要切，
        # 否则「一、总体要求」的正文与「二、」的标题会落进同一段
        "sections": [_NUM_CN, _NUM_INSIDE],
    },
    {
        "src": "_probe_bj_zhishen.docx", "kind": "docx",
        "out": "北京智深律师事务所收费标准.txt",
        "sections": [_NUM_CN, _NUM_ARABIC, _NUM_DECIMAL, _NUM_INSIDE],
    },
    {
        "src": "_probe_bj_changmin_pdf.pdf", "kind": "pdf",
        "out": "北京昌民律师事务所收费标准.txt",
        "sections": [_NUM_CN, _NUM_ARABIC, _NUM_DECIMAL, _NUM_INSIDE],
    },
    {
        "src": "_probe_bj_rongli.pdf", "kind": "pdf",
        "out": "北京融理律师事务所收费标准.txt",
        "sections": [_NUM_CN, _NUM_ARABIC, _NUM_DECIMAL, _NUM_INSIDE],
    },
]

# 合并阈值：短于此的单元（小标题、单条列表项）并入相邻单元，避免出现
# 「只有标题、没有内容」的检索片段。**够长的单元原样保留、不再切** ——
# 按累积字符数硬切会把一张费率表拦腰截断（100万—1000万那档留在上一段、
# 对应的 4% 落到下一段），而回查挡不住「档位用错」，只能靠片段本身完整。
MIN_UNIT_CHARS = 200

# 单个片段的上限来自 fee_corpus 的 text 字段（VARCHAR 4000 字节 ≈ 1333 汉字）。
# 超了会被 Milvus **静默截断**，截掉的可能正是费用数字
MAX_UNIT_CHARS = 1200

# 顶级小节：「一、」「二、」这一级。一段里最多允许出现一个 —— 见 build_text 的断言
TOP_SECTION = re.compile(r"^[一二三四五六七八九十]+、")


def docx_text(path: pathlib.Path) -> str:
    """取 .docx 正文。表格单元格会被压成竖排（每格一行），语义仍可读。"""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"(?is)<w:tab[^>]*/>", "\t", xml)
    xml = re.sub(r"(?is)</w:p>", "\n", xml)
    return html.unescape(re.sub(r"(?s)<[^>]+>", "", xml))


def pdf_text(path: pathlib.Path) -> str:
    """取 PDF 文本层。本项目对**法条 PDF** 一律走 MinerU + PaddleOCR-VL；
    这四份收费文件是政务平台公示的电子件、文本层无损，故直接取用 ——
    该例外经**用户 2026-09-29 裁决**并落档（见模块 docstring），
    不是本脚本作者自开的旁路。"""
    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def strip_page_numbers(text: str) -> str:
    """删掉整行只有 1~4 位数字的行 —— PDF 文本层的页码会混进正文，
    留着就会变成「片段里的一个数字」，让费用回查把页码当成金额出处。"""
    return "\n".join(ln for ln in text.split("\n")
                     if not re.fullmatch(r"\d{1,4}", ln.strip()))


def reflow(text: str) -> str:
    """在句末标点后的编号处断行。

    PDF 文本层常把一整页挤成一行（融理那份就是），不恢复换行则整份文件
    只有一个段落 = 一个片段，检索命中的粒度就是「一整篇」。
    """
    text = re.sub(r"(?<=[。；])\s*(?=（[一二三四五六七八九十]+）)", "\n", text)
    text = re.sub(r"(?<=[。；])\s*(?=\d{1,2}[.．])", "\n", text)
    text = re.sub(r"(?<=[。；])\s*(?=[一二三四五六七八九十]+、)", "\n", text)
    return re.sub(r"\n{2,}", "\n", text)


def sectionize(text: str, patterns: list[str]) -> list[str]:
    """按分节标记切成语义单元。标记行开启新单元，随后的行都归它。"""
    regexes = [re.compile(p) for p in patterns]
    units: list[str] = []
    buf: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if buf and any(rx.match(stripped) for rx in regexes):
            units.append("\n".join(buf))
            buf = []
        buf.append(stripped)
    if buf:
        units.append("\n".join(buf))
    return units


def merge_units(units: list[str]) -> list[str]:
    """把**单行的短单元**并入相邻的完整单元；其余原样保留。

    「单行」这个条件是关键：sectionize 按标记切，一个标记行连同它下面所有
    非标记的行构成一个单元 —— 所以**多行单元一定带内容**，是完整的语义单位，
    哪怕它短（「二、计时收费」连它的三行收费标准只有 80 字）；而单行单元只
    可能是「标题后紧跟下一个标记」，那种没内容的标题才需要并进邻居。

    早期版本按长度累积合并，把「二、计时收费」这类短而完整的小节黏进了下一节，
    于是**两个小节挤进同一片段**：source_no 指向段首的「二、」，数字却来自段内的
    「三、风险代理收费」—— 检索命中后「依据」指错小节，而回查只验数字在不在
    片段里，拦不住这个（2026-09-29 真链路冒烟实证）。
    """
    out: list[str] = []
    buf = ""

    def is_top(unit: str) -> bool:
        return bool(TOP_SECTION.match(unit))

    def settle() -> None:
        """结掉攒着的碎片：以顶级小节开头的自成一段，其余并进上一段。

        out 为空时没有可并入的邻居（整份文件全是小标题），也必须自成一段 ——
        直接写 out[-1] 会 IndexError，而报错信息看不出所以然。
        """
        nonlocal buf
        if not buf:
            return
        if is_top(buf) or not out:
            out.append(buf)
        else:
            out[-1] = f"{out[-1]}\n{buf}"
        buf = ""

    for unit in units:
        if is_top(unit):
            # 顶级小节是硬边界：攒下的碎片不能跨过它。不设这条边界，一份
            # 「一、…」的正文与「二、…」的标题就会落进同一段（实测踩到）
            settle()
            buf = unit
        elif "\n" not in unit and len(unit) < MIN_UNIT_CHARS:
            buf = f"{buf}\n{unit}" if buf else unit
        else:
            out.append(f"{buf}\n{unit}" if buf else unit)
            buf = ""
    settle()
    return out


def build_text(spec: dict) -> str:
    """原始件 → 语料文本（清洗 → 断行恢复 → 分节 → 黏合）。"""
    raw = (docx_text if spec["kind"] == "docx" else pdf_text)(SRC_DIR / spec["src"])
    text = reflow(strip_page_numbers(raw))
    units = merge_units(sectionize(text, spec["sections"]))
    oversized = [len(u) for u in units if len(u) > MAX_UNIT_CHARS]
    if oversized:
        # 硬拦而不是截断：超长片段会被 Milvus 静默截掉尾巴，而尾巴上
        # 很可能正是一个费率
        raise ValueError(f"{spec['out']} 有 {len(oversized)} 段超长：{oversized}")
    # 一段里挤进两个顶级小节 = source_no 指向段首那个、数字却可能来自另一个。
    # 费用回查只验数字在不在片段里，认不出这种错位（2026-09-29 真链路冒烟踩过
    # 一次：昌民的「二、计时收费」与「三、风险代理收费」同段，模型拿后者的
    # 回款提成当成了律师费区间），故在这里硬拦
    tops = [(u.split("\n")[0][:24], sum(1 for line in u.split("\n")
                                        if TOP_SECTION.match(line))) for u in units]
    crowded = [head for head, count in tops if count > 1]
    if crowded:
        raise ValueError(f"{spec['out']} 有 {len(crowded)} 段含多个顶级小节：{crowded}")
    return "\n\n".join(units) + "\n"


def main() -> int:
    for spec in SPECS:
        body = build_text(spec)
        (OUT_DIR / spec["out"]).write_text(body, encoding="utf-8", newline="\n")
        units = [u for u in body.split("\n\n") if u.strip()]
        sizes = sorted(len(u) for u in units)
        print(f"{spec['out']}：{len(units)} 段，{sizes[0]}~{sizes[-1]} 字符")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
