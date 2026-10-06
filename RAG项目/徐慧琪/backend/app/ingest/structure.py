"""层级还原：把 MinerU 产物 markdown 还原成编—分编—章—节—条—款—项。

存在的理由（这是本模块最关键的设计约束）：**层级必须按词元判定，不能看 `#` 层级。**
实测撞到四种情况，任何一个处理错都会静默错切：
  1. MinerU 把 10 条误标成 `## 第二百一十条…`（上册 210/261/348/366/373/393/395/400/401/412）
  2. MinerU 把标记转义成 `\\## 第三编 合同`（中册经 md→docx 往返后）
  3. docx 路径的标题带粗体 `# **第五编 婚姻家庭**`
  4. 标题被包进引用块 `> ## 第三编 合同`
统一剥掉这些标记后按「第X编/分编/章/节/条」判定，同一套规则对三个解析路径都成立。
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field

from app.ingest.cn_num import cn2int

# 归一化要剥掉的标记字符：\ # * ` > 空白 全角空格
# 两端都要剥——粗体是成对的，只剥前导会在尾部留下 `**` 污染路径
LEAD_CHARS = "\\#*`> \t　"

CN = r"[零一二三四五六七八九十百千两]+"

# 顺序即优先级：分编必须早于编，否则「第一分编」被「^第X编」提前匹配成「第一编」
LEVEL_PATTERNS: list[tuple[str, str]] = [
    ("条", rf"^第({CN})条"),
    ("分编", rf"^第({CN})分编"),
    ("编", rf"^第({CN})编"),
    ("章", rf"^第({CN})章"),
    ("节", rf"^第({CN})节"),
]


def normalize(line: str) -> str:
    """剥掉 markdown 标记与引用前缀，得到可用于词元判定的纯文本。"""
    return line.strip().strip(LEAD_CHARS).strip()


def classify(line: str) -> tuple[str | None, int | None, str]:
    """判定一行属于哪个层级，返回 (层级名, 序号, 归一化文本)。

    非层级行返回 (None, None, 归一化文本)——这些就是条的正文或款的正文。
    """
    text = normalize(line)
    for kind, pattern in LEVEL_PATTERNS:
        if m := re.match(pattern, text):
            return kind, cn2int(m.group(1)), text
    return None, None, text


# 层级路径按由外到内的顺序拼装；某册没有分编时自然跳过该级
PATH_ORDER = ("编", "分编", "章", "节")

# 进入某一级时要清空的更深层级，避免上一编的分编/章串到新编下
DEEPER_LEVELS = {
    "编": ("分编", "章", "节"),
    "分编": ("章", "节"),
    "章": ("节",),
}


@dataclass
class Article:
    """一条法条。text 是含条号的完整原文，paragraphs 是按空行拆出的款/项。"""
    number: int
    number_cn: str
    text: str
    path: str
    paragraphs: list[str] = field(default_factory=list)


def _split_body(md_path: pathlib.Path,
                header_markers: tuple[str, ...]) -> list[str]:
    """切掉头部目录区，只留正文。

    目录里会出现与正文一模一样的「第X编 第X章」字符串，不切掉会把目录当正文，
    导致第一条的路径错成目录里的编章。取最后一个 header_marker 之后开始。
    """
    lines = md_path.read_text(encoding="utf-8").splitlines()
    cut = -1
    for idx, line in enumerate(lines):
        if line.strip() in header_markers:
            cut = idx
    return lines[cut + 1:]


def restore_articles(md_path: pathlib.Path,
                     header_markers: tuple[str, ...] = ("附则", "目 录", "目录"),
                     ) -> list[Article]:
    """把一册的产物 markdown 还原成条列表，每条带编/分编/章/节路径。"""
    lines = _split_body(md_path, header_markers)
    cursor: dict[str, str] = {}          # 当前所处的层级路径
    articles: list[Article] = []
    # 正在累积的条。款之间以空行为分隔，空行会先落一个占位再被下一行替换
    pending_number: int | None = None
    pending_cn = ""
    pending_paras: list[str] = []

    def flush() -> None:
        """把累积中的条落成 Article。"""
        nonlocal pending_number, pending_cn, pending_paras
        if pending_number is not None:
            path = " > ".join(cursor[k] for k in PATH_ORDER if k in cursor)
            articles.append(Article(number=pending_number, number_cn=pending_cn,
                                    text="\n".join(pending_paras), path=path,
                                    paragraphs=list(pending_paras)))
        pending_number, pending_cn, pending_paras = None, "", []

    for raw in lines:
        kind, num, text = classify(raw)
        if kind in DEEPER_LEVELS or kind == "节":
            flush()
            cursor[kind] = text
            for deeper in DEEPER_LEVELS.get(kind, ()):
                cursor.pop(deeper, None)
        elif kind == "条":
            flush()
            # 首段保留整行（含条号）——逐字命中要拿 text 去 PDF 里比，
            # 而 PDF 原文是含条号的，剥掉条号会让校验全数失配
            m = re.match(rf"^第({CN})条", text)
            pending_cn = m.group(1) if m else ""
            pending_number = num
            pending_paras = [text]
        else:
            if pending_number is None:
                continue                  # 条之前的说明性文字不入任何条
            if not raw.strip():
                pending_paras.append("")  # 空行=款的分隔，占位待下一行替换
            elif pending_paras and pending_paras[-1] == "":
                pending_paras[-1] = text  # 新的一款
            elif pending_paras:
                pending_paras[-1] += text # 同款续行
            else:
                pending_paras.append(text)
    flush()
    # 去掉占位与分隔留下的空段
    for art in articles:
        art.paragraphs = [p for p in art.paragraphs if p.strip()]
        art.text = "\n".join(art.paragraphs)
    return articles
