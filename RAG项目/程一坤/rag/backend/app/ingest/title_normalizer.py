"""文档标题归一化：从页面内容提取可读标题（站点后缀剥离 + 黑名单 + 可疑判定）。

为什么独立：这块是纯文本启发式规则，最近频繁改动（"工资支付暂行规定_
人力资源和社会保障部" 那次的站点后缀剥离），与打包流程无关；
单独成文件后规则改动只动这一处，并有专门单测直击。
"""

import re
from pathlib import Path

# 只有这些扩展名才有 HTML <title>；PDF 是二进制，按 UTF-8 强读既取不到标题、
# 也会白白把整个文件当文本解码一遍（批次 22 接入 PDF 时补的判定）
_HTML_SUFFIXES = {".html", ".htm"}


def _extract_document_title(source_path: Path, cleaned_content: str, explicit_title: str | None = None) -> str:
    """从页面内容提取可读的文档标题。

    优先级：
    1. HTML：页面 <title>，去掉站点后缀；若结果落在站点名黑名单或可疑模式则跳到 ②
       （PDF 无内嵌标题，直接跳到 ②）
    2. 采集目标里的 document_title（人工核对值）
    3. 都不满足 → 报错，禁止用文件名哈希或正文句子兜底

    可疑标题判定（命中任一条视为提取失败）：
    - 长度 > 40 字
    - 含发文字号特征："第…号"、"令"、"公布"、"根据"
    - 以正文开头特征："中华人民共和国境内的"、"在中华人民共和国"
    - 只有"中华人民共和国XX法"但没有"实施条例"，而页面正文出现"实施条例"（防止截断）
    """
    is_html = source_path.suffix.lower() in _HTML_SUFFIXES

    # 优先级 1：从 HTML <title> 提取
    html_content = (
        source_path.read_text(encoding="utf-8", errors="ignore") if is_html else ""
    )
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)

    if title_match:
        title_text = title_match.group(1).strip()
        # 去掉常见站点后缀。后缀候选分两层：
        # 1. 明确站点名（含习惯简称）
        # 2. 通用部委/机构名：分隔符后的尾段以 部/委员会/总局/局/法院/检察院
        #    结尾即视为站点名（真实采集页 <title> 形如
        #    "工资支付暂行规定_人力资源和社会保障部"，部委名枚举不完，
        #    必须靠词尾特征兜住）。正文法规名极少出现在分隔符之后，
        #    误剥风险远小于漏剥。
        title_text = re.sub(
            r"\s*[-_—|]\s*("
            r"中华人民共和国最高人民法院"
            r"|司法部|中国政府网|国家.*?网"
            r"|[\u4e00-\u9fa5]{2,12}(?:部|委员会|总局|局|法院|检察院)"
            r").*$",
            "",
            title_text,
        ).strip()

        # 站点名黑名单：<title> 整个就是站点名 → 提取失败，走人工核对标题
        site_name_blacklist = {
            "国家行政法规库",
            "中华人民共和国最高人民法院",
            "最高人民法院",
            "中国政府网",
            "全国人大",
            "全国人民代表大会",
            # 常见部委站点名（<title> 只含站点名时兜底）
            "人力资源和社会保障部",
            "人力资源社会保障部",
            "财政部",
            "住房和城乡建设部",
            "交通运输部",
            "工业和信息化部",
            "农业农村部",
            "商务部",
            "生态环境部",
            "司法部",
            "公安部",
            "教育部",
            "民政部",
            "水利部",
            "国家税务总局",
            "国家医疗保障局",
        }

        # 检查是否为有效标题
        if title_text and len(title_text) > 3 and title_text not in site_name_blacklist:
            # 可疑标题判定
            is_suspicious = (
                len(title_text) > 40  # 过长
                or re.search(r"第.{1,5}号|令|公布|根据", title_text)  # 发文字号特征
                or title_text.startswith(("中华人民共和国境内的", "在中华人民共和国"))  # 正文开头
                or (  # 截断判定：只有"XX法"但正文有"实施条例"
                    re.match(r"^中华人民共和国.{2,10}法$", title_text)
                    and "实施条例" not in title_text
                    and "实施条例" in cleaned_content[:500]
                )
            )

            if not is_suspicious:
                return title_text

    # 优先级 2：使用人工核对的标题
    if explicit_title:
        return explicit_title

    # 优先级 3：都取不到时报错
    reason = (
        "HTML <title> 是站点名、为空或可疑，且未提供人工核对标题。"
        if is_html
        else "PDF 没有内嵌文档标题，必须在采集目标中提供人工核对标题。"
    )
    raise ValueError(
        f"无法从页面内容提取文档标题：{source_path.name}\n"
        f"{reason}\n"
        f"请检查页面内容或在采集目标中指定 document_title。"
    )
