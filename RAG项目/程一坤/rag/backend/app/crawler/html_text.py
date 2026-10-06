"""HTML 正文提取与清洗（原 requester.py 并入部分）。

从 HTML 提取正文文本并忽略脚本、样式等无关节点，
清洗结果用于计算稳定的内容哈希。
"""

# 导入 HTML 解析器，用于提取正文文本
from html.parser import HTMLParser

# 导入正则表达式，用于规范化文本空白
import re


# 从 HTML 提取正文文本并忽略脚本、样式等无关节点
class _HtmlTextExtractor(HTMLParser):
    """收集 HTML 正文文本并忽略非正文节点。"""

    # 定义会引入换行的块级标签集合
    BLOCK_TAGS = {
        "p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6",
        "tr", "table", "section", "article",
    }

    # 初始化文本收集状态
    def __init__(self) -> None:
        # 调用父类初始化方法
        super().__init__()
        # 用于收集提取的文本片段
        self._text_parts: list[str] = []
        # 记录当前是否在被忽略的节点内部
        self._ignored_depth = 0
        # 定义需要完全忽略的标签集合
        self._ignored_tags = {
            "head", "script", "style", "nav", "footer", "header"
        }

    # 遇到开始标签时判断是否进入忽略区域
    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        # 标签名统一转为小写
        normalized_tag = tag.lower()
        # 如果是需要忽略的标签，增加忽略深度计数
        if normalized_tag in self._ignored_tags:
            self._ignored_depth += 1
            return
        # 如果不在忽略区域且是块级标签，插入换行符
        if not self._ignored_depth and normalized_tag in self.BLOCK_TAGS:
            self._text_parts.append("\n")

    # 遇到结束标签时判断是否退出忽略区域
    def handle_endtag(self, tag: str) -> None:
        # 标签名统一转为小写
        normalized_tag = tag.lower()
        # 如果是忽略标签的结束，减少忽略深度计数
        if normalized_tag in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1
            return
        # 如果不在忽略区域且是块级标签，插入换行符
        if not self._ignored_depth and normalized_tag in self.BLOCK_TAGS:
            self._text_parts.append("\n")

    # 只收集不在忽略区域内的文本内容
    def handle_data(self, data: str) -> None:
        # 如果当前不在被忽略的节点内，收集文本
        if not self._ignored_depth:
            self._text_parts.append(data)

    # 返回提取并规范化的正文文本
    def get_text(self) -> str:
        # 拼接所有收集的文本片段
        content = "".join(self._text_parts)
        # 将连续的多个换行符压缩为单个换行符
        return re.sub(r"\n{2,}", "\n", content).strip()


# 从 HTML 提取并清洗正文文本，用于计算稳定的内容哈希
def _extract_cleaned_text(html_content: str) -> str:
    """从 HTML 提取并清洗正文文本（用于稳定哈希计算）。"""
    # 创建 HTML 文本提取器实例
    extractor = _HtmlTextExtractor()
    # 解析 HTML 并提取正文
    extractor.feed(html_content)
    # 获取提取的原始文本
    raw_text = extractor.get_text()
    # 统一换行符格式为 LF
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    # 将连续空格和制表符压缩为单个空格
    normalized = re.sub(r"[ \t]+", " ", normalized)
    # 移除首尾空白后返回
    return normalized.strip()
