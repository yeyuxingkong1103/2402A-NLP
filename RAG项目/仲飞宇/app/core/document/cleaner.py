"""数据清洗：空白归一化、去水印/页眉页脚。

位置：在 parser 抽出原始文本之后、chunker 切块之前跑一次（parse_file / parse_bytes
的末尾调用 clean_text），全链路只此一个入口——「入库的文本都洗过」这个前提靠它成立。
只做与语言无关的机械清洗（空白、整行水印），不改写句子，理由见下面 WATERMARK_PATTERNS。
"""
from __future__ import annotations

import re

# 常见水印 / 页眉页脚特征（按行过滤）。
#
# 两条硬规矩，都是被实测逼出来的——**整行匹配**，且宽松特征必须锚定行首/行尾：
#
# 1) 以前 `扫描二维码` / `关注公众号` / `更多.*(访问|查看|下载)` / `仅供(内部|参考|学习)`
#    是裸 `re.search`，正文行里出现这些词就**整行删掉**，入库还 200 OK。实测 6 条正常
#    正文删掉 5 条：「更多关于高血压的饮食建议，可访问我们的健康专栏」「关注公众号后回复
#    关键词即可获取随访表」「扫描二维码可加入患者互助群，群内有医生答疑」全部蒸发。
# 2) 现在改成：正文里"提了一嘴"这些词的行要**活下来**，只有"整行几乎就是这句水印"才删。
#    所以宽松特征统一锚定（`^`/`$`）并限制中间可夹的字符数——页脚水印永远是短的。
#    宁可漏删一条页脚（下游只是多一句噪音），也不能静默吃掉正文（检索答非所问且查不出）。
WATERMARK_PATTERNS = [
    r"^\s*http[s]?://\S+\s*$",   # 纯 URL 行
    r"^\s*www\.\S+\s*$",
    r"^\s*第\s*\d+\s*页\s*(共\s*\d+\s*页)?\s*$",  # 页码
    # 「扫描二维码」类：整行只有扫码/关注我们，中间可夹一个顿号逗号
    r"^\s*(?:微信)?扫描二维码\s*[，,、]?\s*(?:关注(?:我们|公众号))?\s*$",
    # 「关注公众号 XX」类：公众号名最多 12 字，再多就是正文了
    r"^\s*(?:扫码)?关注(?:微信)?公众号\s*[:：]?\s*\w{0,12}\s*$",
    # 「更多…请访问/查看/下载」类：必须收在行尾，可跟一个网址
    r"^\s*更多.{0,15}(?:请)?(?:访问|查看|下载)\s*[:：]?\s*(?:https?://\S+|www\.\S+)?\s*$",
    # 「仅供内部/参考/学习」类：整行就是这句声明
    r"^\s*仅供(?:内部|参考|学习)(?:交流|使用|参考)?\s*$",
]


def remove_watermark_lines(lines: list[str]) -> list[str]:
    """过滤掉匹配水印特征的行。"""
    return [ln for ln in lines if not _is_watermark(ln)]


def _is_watermark(line: str) -> bool:
    return any(re.search(p, line, re.IGNORECASE) for p in WATERMARK_PATTERNS)


def clean_text(text: str) -> str:
    """统一换行/空白，去掉水印行与连续空行。"""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 顺序有意：全角空格必须先转半角，下面那句 `[ \t]+` 的压缩才管得住它（那个字符类
    # 只认半角空格和 tab）。否则中文 PDF / 网页复制来的文本会在行内留下看不见的全角空格，
    # 按字符数算的 chunk 预算被白占，同一句话还会以两种形态参与去重与检索。
    text = text.replace("　", " ")  # 全角空格 -> 半角

    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    lines = remove_watermark_lines(lines)

    # 连续空行压缩成一个，而不是全删：paragraph / title 分块就靠空行断段（见
    # chunker._split_paragraphs），空行删光会让整篇并成一段、只剩硬切；
    # 而 PDF 抽出来的文本常带成串空行，留一个就够，多留只是白占预算。
    out: list[str] = []
    blank = 0
    for ln in lines:
        if not ln:
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(ln)
    return "\n".join(out).strip()
