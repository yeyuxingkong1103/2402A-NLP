import re
from collections.abc import Collection

# 法规结构标题行（章/节/编/部分/篇 + 可选标题）：
# 例：「第一章」「第一章 总则」「第五节」「第二部分」
_CHAPTER_HEADING_PATTERN = re.compile(
    r"^第[一二三四五六七八九十百千万零〇两\d]+"
    r"(?:章|节|编|部分|篇)(?:[\s　].*)?$"
)

# 条文标题行（「第X条」，含「第X条之一」）：条文号单独成行时也是结构标题
_ARTICLE_HEADING_PATTERN = re.compile(
    r"^第[一二三四五六七八九十百千万零〇两\d]+条(?:之[一二三四五六七八九十\d]+)?$"
)

# 章节名称型短行：法规里「总则」「附则」这类独立成行的章节名，
# 以及「目录」（PDF 里常见排成「目 录」，中间带空格）
_SECTION_NAME_PATTERN = re.compile(r"^(?:总则|分则|附则|通则|序言|前言|目\s*录)$")

# 站点页脚样板行（批次 38 建；批次 39 扩到中国政府网/法院网页脚）。
# 判断依据：这些词只出现在"站点运营方署名 / 备案号 / 移动端引导"里，
# 法规正文不会以这些形态出现（判定前已对全库 11 篇做过逐行核对，见批次 39 报告）。
_BOILERPLATE_CONTAINS = (
    # 版权与备案（court.gov.cn / gov.cn 共用）
    "版权所有", "版权声明：", "网站标识码",
    # 站点运营方署名（中国政府网页脚「主办单位：国务院办公厅　运行维护单位…」）
    "主办单位", "运行维护单位", "中文域名",
    # 移动端引导与客户端推广（中国政府网正文工具条/页脚）
    "扫一扫在手机打开", "国务院客户端",
)
_BOILERPLATE_PREFIXES = (
    # 页头面包屑与联系方式
    "所在位置", "字号", "总机", "举报电话", "传真", "邮编",
    # 稿件来源标注（gov.cn 新闻类页面「来源：人力资源社会保障部网站」）
    "来源：",
    # 页脚链接区标题与正文工具条（gov.cn / court.gov.cn 共有形态）
    "相关链接", "链接：", "返回顶部", "打印本页", "我要纠错", "关闭窗口",
)
_ICP_PATTERN = re.compile(r"ICP备\d+号|京公网安备")
# 链接区标题行：「相关链接：」「友情链接」「链接：」等。这类标题出现后紧随的
# 若干行是站点推荐的链接标题（court.gov.cn 的「相关链接」区就在正文容器内），
# 属导航而非法规正文。窗口有界（最多 6 行）且遇条文行/结构标题立即终止，
# 避免误吞正文（批次 39）。
_LINK_AREA_HEADING_PATTERN = re.compile(r"^(?:相关链接|友情链接|相关阅读|链接)\s*[:：]?$")
_LINK_AREA_MAX_LINES = 6
# 条文行（以「第X条」开头，可能带正文）
_ARTICLE_LINE_PATTERN = re.compile(r"^第[一二三四五六七八九十百千万零〇两\d]+条")
_LONG_NUMBER_LINE_PATTERN = re.compile(r"^\d{10,}号?$")  # 如 11040102700145号
_HEX_HASH_LINE_PATTERN = re.compile(r"^[0-9a-fA-F]{16,}$")  # 站点统计校验串
# 整行就是一个方括号控件标签：如「【打印】」「【我要纠错】」「【字体：大 中 小】」。
# 必须是"控件词白名单"才删 —— 批次 39 试跑发现只按"整行被【】包住"判定会误删
# 典型案例的正文小标题（【基本案情】【裁判结果】【典型意义】），故收窄为白名单。
_BRACKET_CONTROL_PATTERN = re.compile(r"^[【\[]([^】\]]{1,12})[】\]]$")
# 正文工具条控件词（命中其一才认定为控件标签）
_CONTROL_LABELS = (
    "打印", "我要纠错", "关闭窗口", "字体", "字号", "收藏", "分享",
    "复制", "返回", "放大", "缩小",
)


def _is_structural_heading(line: str) -> bool:
    """判断是否为法规结构标题行。

    批次 22 新增：PDF 与 HTML 里「第一章」「总则」常常独立成行且长度 < 6、
    不含标点，会被下面的「孤立短行」导航规则误删 —— 标题被删会让整章的条文
    失去归属。这类行按结构特征白名单放行，不参与导航判定。
    """
    stripped = line.strip()
    return bool(
        _CHAPTER_HEADING_PATTERN.match(stripped)
        or _ARTICLE_HEADING_PATTERN.match(stripped)
        or _SECTION_NAME_PATTERN.match(stripped)
    )


def clean_text(
    content: str,
    repeated_headers: Collection[str] | None = None,
    repeated_footers: Collection[str] | None = None,
    watermarks: Collection[str] | None = None,
) -> str:
    """保守清洗解析文本并保留段落边界。"""
    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")
    normalized_content = normalized_content.replace("﻿", "")
    normalized_content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", normalized_content)

    headers = _normalize_rules(repeated_headers)
    footers = _normalize_rules(repeated_footers)
    watermark_rules = tuple(rule for rule in _normalize_rules(watermarks) if rule)

    cleaned_lines: list[str] = []
    # 链接区剩余可跳过的行数（批次 39）：见 _LINK_AREA_HEADING_PATTERN 说明
    link_area_remaining = 0
    for line in normalized_content.split("\n"):
        stripped_line = _normalize_line(line)
        if not stripped_line:
            cleaned_lines.append("")
            link_area_remaining = 0
            continue
        # 链接区标题：自身剔除，并开始跳过随后的链接标题行
        if _LINK_AREA_HEADING_PATTERN.match(stripped_line):
            link_area_remaining = _LINK_AREA_MAX_LINES
            continue
        if link_area_remaining:
            # 遇到条文行/结构标题说明已回到正文，立即停止跳过
            if _ARTICLE_LINE_PATTERN.match(stripped_line) or _is_structural_heading(
                stripped_line
            ):
                link_area_remaining = 0
            else:
                link_area_remaining -= 1
                continue
        if stripped_line in headers or stripped_line in footers:
            continue
        # 剔除页面导航型文本行（防止导航栏污染正文）
        if _is_navigation_line(stripped_line):
            continue
        # 剔除站点页脚样板行（版权/ICP 备案/面包屑/校验串，批次 38）
        if _is_site_boilerplate_line(stripped_line):
            continue
        for watermark in watermark_rules:
            stripped_line = stripped_line.replace(watermark, "")
        cleaned_lines.append(_normalize_line(stripped_line))

    return _trim_empty_lines("\n".join(cleaned_lines))


def _normalize_rules(rules: Collection[str] | None) -> set[str]:
    if not rules:
        return set()
    return {_normalize_line(rule) for rule in rules if rule}


def _normalize_line(line: str) -> str:
    return re.sub(r"[ \t]+", " ", line).strip()


def _trim_empty_lines(content: str) -> str:
    return content.strip("\n")


def _is_navigation_line(line: str) -> bool:
    """判断是否为页面导航型文本行。

    导航型文本特征：
    - 以站点菜单关键词开头（首页、机构设置、法院资讯等）
    - 孤立短行（< 6 字符且不含标点）
    - 同一行内出现 3 个以上栏目词

    例外：法规结构标题（第一章 / 总则 / 第五条）永远不算导航行，
    即使它很短也不含标点 —— 详见 _is_structural_heading。
    """
    # 去掉前置的括号、空格等非文字符号，再检测
    stripped = line.lstrip("[ 【（「『")

    # 特征 0：法规结构标题行一律保留（否则「第一章」「总则」会被下面的短行规则吃掉）
    if _is_structural_heading(line):
        return False

    # 特征 1：以常见导航关键词开头
    navigation_prefixes = (
        "首页", "机构设置", "法院资讯", "权威发布", "公报", "审判",
        "关注：", "中文版", "English", "登录", "搜索", "更多",
        "关于我们", "联系我们", "网站地图", "友情链接", "版权声明"
    )
    for prefix in navigation_prefixes:
        if stripped.startswith(prefix):
            return True

    # 特征 2：孤立短行（< 6 字符且不含标点）
    if len(line) < 6 and not any(c in line for c in "，。；：！？、""''（）【】"):
        return True

    # 特征 3：同一行内出现 3 个以上栏目词（站点菜单连续短行）
    menu_keywords = [
        "首页", "资讯", "公告", "通知", "政策", "法规", "文件",
        "公开", "服务", "互动", "专题", "数据", "统计"
    ]
    keyword_count = sum(1 for keyword in menu_keywords if keyword in line)
    if keyword_count >= 3:
        return True

    return False


def _is_site_boilerplate_line(line: str) -> bool:
    """判断是否为站点页头/页脚样板行（批次 38 建，批次 39 扩充）。

    典型来源：court.gov.cn 页脚「中华人民共和国最高人民法院 版权所有 /
    11040102700145号 / 京ICP备05023036号 / 32 位校验串」、页头面包屑
    「所在位置：…」「字号：」；中国政府网页脚「主办单位：国务院办公厅　
    运行维护单位：中国政府网运行中心 / 版权所有：中国政府网　中文域名：
    中国政府网.政务 / 网站标识码bm01000001　京ICP备05070218号」与正文
    工具条「【打印】【我要纠错】【关闭窗口】」「扫一扫在手机打开当前页」。
    法规正文不含这些形态，整行剔除安全。
    法规结构标题例外在这里无需重复处理——结构标题不含下列任何特征。
    """
    stripped = line.strip()
    if not stripped:
        return False
    for token in _BOILERPLATE_CONTAINS:
        if token in stripped:
            return True
    if _ICP_PATTERN.search(stripped):
        return True
    for prefix in _BOILERPLATE_PREFIXES:
        if stripped.startswith(prefix):
            return True
    bracket = _BRACKET_CONTROL_PATTERN.match(stripped)
    if bracket and any(label in bracket.group(1) for label in _CONTROL_LABELS):
        return True
    if _LONG_NUMBER_LINE_PATTERN.match(stripped):
        return True
    if _HEX_HASH_LINE_PATTERN.match(stripped):
        return True
    return False
