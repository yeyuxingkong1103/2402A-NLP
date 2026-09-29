# 导入数据类装饰器，用于定义统一解析结果
from dataclasses import dataclass, field

# 导入 HTML 解析器，用于移除无关节点并提取正文
from html.parser import HTMLParser

# 导入路径类型，用于读取本地文件
from pathlib import Path

# 导入正则表达式，用于清理连续空白
import re

# 导入 PDF 解析编排函数
from app.ingest.pdf_parser import ParsedPdf, parse_pdf

# 导入统一文档清洗函数
from app.ingest.cleaner import clean_text


# 保存所有文件格式统一的解析结果
@dataclass(frozen=True)
class ParsedDocument:
    # 保存输入文件路径
    source_path: Path

    # 保存清洗后的正文
    content: str

    # 保存统一格式名称
    content_format: str

    # 保存页数，非 PDF 文件为空
    page_count: int | None = None

    # 保存实际使用的解析器名称
    parser_name: str = ""

    # 保存后续元数据扩展字段
    metadata: dict[str, str] = field(default_factory=dict)


# 收集 HTML 正文文本并忽略非正文节点
class _HtmlTextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6",
        "tr", "table", "section", "article",
    }

    # HTML 空元素（无结束标签），进入主内容容器后不参与嵌套深度计数
    _VOID_TAGS = frozenset({
        "br", "img", "input", "meta", "link", "hr", "area", "base",
        "col", "embed", "source", "track", "wbr",
    })

    # 主内容容器标记（批次 38 新增 zoom；批次 39 扩到 UCAP-CONTENT）。
    # 命中时只收容器内文本，避开页头面包屑与页脚版权样板；未命中则回退全页提取。
    # 两个标记都来自"页面自身的正文区约定"，不是按站点域名硬编码：
    #   - zoom          ：TRS 型 CMS（court.gov.cn 法规/新闻/案例页）正文区
    #                     <div class="txt_txt" id="zoom">；同站点的案例页与
    #                     司法解释页共用该模板，故一条规则覆盖全站。
    #   - UCAP-CONTENT  ：中国政府网（gov.cn）正文区
    #                     <div class="pages_content" id="UCAP-CONTENT">。
    _MAIN_CONTENT_MARKERS = ("zoom", "UCAP-CONTENT")

    # 初始化 HTML 文本收集状态
    def __init__(self) -> None:
        super().__init__()
        self._text_parts: list[str] = []
        self._ignored_depth = 0
        self._ignored_tags = {
            "head", "script", "style", "nav", "footer", "header", "aside"
        }
        # 主内容容器跟踪状态：_main_depth 在容器内 > 0，未进入/已离开为 0
        self._main_seen = False
        self._main_depth = 0
        self._main_text_parts: list[str] = []

    # 在标签属性里找主内容容器标记
    def _matches_main_marker(self, attrs: list[tuple[str, str | None]]) -> bool:
        """按「完整标记」匹配 class/id，避免子串误命中。

        class 可能多值（如 "txt_txt big"）→ 按空白切词逐个比对；
        id 是单值 → 整串比对。均大小写不敏感。批次 39 修：
        原实现用子串匹配，`class="imagezoom"`（图片放大 JS 挂点）会被
        误判成正文区，只收到组件内的说明文字、丢掉整篇正文。
        """
        markers = {marker.lower() for marker in self._MAIN_CONTENT_MARKERS}
        for name, value in attrs:
            if not value or name not in {"class", "id"}:
                continue
            candidates = value.split() if name == "class" else [value]
            if any(candidate.lower() in markers for candidate in candidates):
                return True
        return False

    # 遇到开始标签时判断是否进入无关节点
    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in self._ignored_tags:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        # 命中主内容容器标记：开始只收容器内文本（div 同名嵌套用深度计数）
        if not self._main_seen and self._matches_main_marker(attrs):
            self._main_seen = True
            self._main_depth = 1
        elif self._main_depth:
            if normalized_tag == "div":
                self._main_depth += 1
        if normalized_tag in self.BLOCK_TAGS:
            self._text_parts.append("\n")
            if self._main_depth:
                self._main_text_parts.append("\n")

    # 遇到结束标签时退出无关节点并保留块级边界
    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if not self._ignored_depth and normalized_tag in self.BLOCK_TAGS:
            self._text_parts.append("\n")
            if self._main_depth:
                self._main_text_parts.append("\n")
        # 主内容容器的同名结束标签：深度归零即离开正文区
        if self._main_depth and normalized_tag == "div":
            self._main_depth -= 1

    # 只收集正文区域中的文本
    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._text_parts.append(data)
            if self._main_depth:
                self._main_text_parts.append(data)

    # 返回收集到的 HTML 文本（命中主内容容器时优先只取容器内文本）
    def get_text(self) -> str:
        content = "".join(self._text_parts)
        full_text = re.sub(r"\n{2,}", "\n", content).strip()
        # 主内容提取成功（有实质文本）才采用，否则回退全页文本
        if self._main_seen and self._main_depth == 0:
            main_text = re.sub(r"\n{2,}", "\n", "".join(self._main_text_parts)).strip()
            if len(main_text) >= 100:
                return main_text
        return full_text


# 通过统一入口解析支持的文件格式
def parse_document(
    source_path: Path,
    mineru_client: object | None = None,
    qwen_vl_client: object | None = None,
) -> ParsedDocument:
    # 文件不存在时立即报告输入错误
    if not source_path.is_file():
        raise FileNotFoundError(f"解析文件不存在：{source_path}")

    # 根据扩展名选择对应解析方式
    suffix = source_path.suffix.lower()

    # TXT 使用 UTF-8 读取并清理空白
    if suffix == ".txt":
        return _parse_text_file(source_path, "txt", "text")

    # Markdown 保留标题和正文文本
    if suffix in {".md", ".markdown"}:
        return _parse_text_file(source_path, "markdown", "markdown")

    # HTML 移除脚本、样式和页面结构无关节点
    if suffix in {".html", ".htm"}:
        return _parse_html_file(source_path)

    # PDF 必须由 MinerU 和 Qwen-VL 客户端处理
    if suffix == ".pdf":
        if mineru_client is None or qwen_vl_client is None:
            raise ValueError("PDF 解析需要 MinerU 和 Qwen-VL 客户端")
        return _parse_pdf_file(source_path, mineru_client, qwen_vl_client)

    # 未知扩展名不得猜测文件内容
    raise ValueError(f"不支持的文件格式：{suffix}")


# 解析 TXT 或 Markdown 文本文件
def _parse_text_file(
    source_path: Path,
    content_format: str,
    parser_name: str,
) -> ParsedDocument:
    # 读取 UTF-8 文本内容
    content = source_path.read_text(encoding="utf-8")

    # Markdown 先移除格式标记，再统一清理空白
    if content_format == "markdown":
        content = _strip_markdown_markers(content)

    # 返回统一解析结果
    return ParsedDocument(
        source_path=source_path,
        content=clean_text(content),
        content_format=content_format,
        parser_name=parser_name,
    )


# 移除不影响正文语义的常见 Markdown 标记
def _strip_markdown_markers(content: str) -> str:
    # 移除标题、引用和无序列表前缀
    content = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", content)
    content = re.sub(r"(?m)^\s{0,3}>\s?", "", content)
    content = re.sub(r"(?m)^\s*[-*+]\s+", "", content)

    # 移除粗体、斜体和行内代码符号
    content = re.sub(r"(\*\*|__|`)", "", content)
    content = re.sub(r"(?<!\*)\*(?!\*)|(?<!_)_(?!_)", "", content)

    # 保留链接文字，去除链接地址
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)


# 解析 HTML 文件中的正文
def _parse_html_file(source_path: Path) -> ParsedDocument:
    # 读取 HTML 原文
    html_content = source_path.read_text(encoding="utf-8")

    # 运行 HTML 文本提取器
    extractor = _HtmlTextExtractor()
    extractor.feed(html_content)

    # 返回清洗后的 HTML 解析结果
    return ParsedDocument(
        source_path=source_path,
        content=clean_text(extractor.get_text()),
        content_format="html",
        parser_name="html",
    )


# 解析 PDF 并将远程结果转换为统一格式
def _parse_pdf_file(
    source_path: Path,
    mineru_client: object,
    qwen_vl_client: object,
) -> ParsedDocument:
    # 调用 MinerU 优先、Qwen-VL 补充的 PDF 流程
    parsed_pdf: ParsedPdf = parse_pdf(
        source_path,
        mineru_client=mineru_client,
        qwen_vl_client=qwen_vl_client,
    )

    # MinerU 返回的是 Markdown，按 Markdown 规则先去掉标记（与 .md 路径同一处理）。
    # 批次 22 实测必要性：MinerU 会把部分条文行提升为标题，输出
    # "## 第十七条 劳动合同应当具备以下条款:"，而条文边界正则是 `^\s*第X条`
    # —— 行首多了 "## " 就匹配不上，该条会被并进上一条（PDF 侧因此少 1 条）。
    # 去掉标记后 PDF 与库内 HTML 的切块结果对齐（实测 97 条 vs 98 条 → 一致）。
    content = clean_text(_strip_markdown_markers(parsed_pdf.content))

    # 返回统一解析结果
    return ParsedDocument(
        source_path=source_path,
        content=content,
        content_format="pdf",
        page_count=parsed_pdf.page_count,
        parser_name=parsed_pdf.parser_name,
    )


    # 统一文本清洗已在 cleaner.py 完成
