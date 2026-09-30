"""离线 RAG 的文本清洗、去水印和分块工具。"""

import logging
import re
from dataclasses import dataclass

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def clean_pdf_text(text: str) -> str:
    """清理 PDF 单页抽取结果：删除空字符、空行和行首尾空格。"""
    # PDF 中可能含 \x00；先删除，再逐行 strip，最后只保留非空行。
    return "\n".join(line.strip() for line in text.replace("\x00", "").splitlines() if line.strip())


def clean_text(text: str) -> str:
    """统一清洗全文：处理全角空格、连续空白和过多换行。"""
    # 删除空字符，并把中文全角空格替换成普通半角空格。
    text = text.replace("\x00", "").replace("\u3000", " ")
    # 同一行内多个空格或 Tab 合并为一个空格，减少无意义 token。
    text = re.sub(r"[ \t]+", " ", text)
    # 三个及以上换行压缩成两个，保留段落边界但避免大片空白。
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _watermark_key(line: str) -> str:
    """归一化一行文字，用于识别字体或空格不同但内容相同的水印。"""
    # 去掉所有空白并转为小写，使“内部 资料”和“内部资料”得到相同 key。
    return re.sub(r"\s+", "", line).strip().casefold()


def remove_repeated_watermarks(page_texts: list[str]) -> tuple[list[str], list[str]]:
    """删除多数页面重复出现的短文本水印，并返回清理后的页面和水印列表。

    这是面向 RAG 的文本去水印：同时覆盖 PDF 文字层和 OCR 结果，不改动原始
    PDF。表格行含 ``|``，为避免误删跨页表头，不参与水印判断。
    """
    # 配置关闭时原样返回；第二个返回值是“发现的水印列表”。
    if not settings.watermark_removal_enabled:
        return page_texts, []
    # 至少要求 2 页；默认配置要求水印至少出现在 3 页。
    minimum_pages = max(2, settings.watermark_min_pages)
    # 页数太少时无法可靠判断“跨页重复”，因此不做删除。
    if len(page_texts) < minimum_pages:
        return page_texts, []
    # page_occurrences 记录某一规范化文字出现在多少个不同页面。
    page_occurrences: dict[str, int] = {}
    # original_lines 保存原始写法，便于日志展示删除了什么。
    original_lines: dict[str, str] = {}
    # 逐页统计；同一句在同一页出现多次仍只算 1 页。
    for text in page_texts:
        page_keys: set[str] = set()
        for line in text.splitlines():
            # 去除行首尾空白后再判断。
            clean_line = line.strip()
            key = _watermark_key(clean_line)
            # 空行忽略；含 | 的表格行受保护；过长内容通常是正文，不当作水印。
            if not key or "|" in clean_line or len(key) > settings.watermark_max_chars:
                continue
            # set 保证同一页只计数一次。
            page_keys.add(key)
            # 只保存第一次看到的原始文字。
            original_lines.setdefault(key, clean_line)
        # 页面扫描结束后，再累计“出现页数”。
        for key in page_keys:
            page_occurrences[key] = page_occurrences.get(key, 0) + 1
    # required_pages 同时满足最少页数和页面覆盖比例，默认取 3 页与 60% 中较大者。
    required_pages = max(
        minimum_pages,
        # +0.9999 再取 int，相当于对正数向上取整。
        int(len(page_texts) * settings.watermark_page_ratio + 0.9999),
    )
    # 达到阈值的短文本被视为重复页眉、页脚或文字水印。
    watermark_keys = {
        key for key, count in page_occurrences.items() if count >= required_pages
    }
    # 没发现水印时不复制或改写页面文本。
    if not watermark_keys:
        return page_texts, []
    # 对每一页删除命中的水印行。
    cleaned_pages = []
    for text in page_texts:
        kept_lines = [
            line for line in text.splitlines()
            if _watermark_key(line.strip()) not in watermark_keys
        ]
        # 删除后再次清理可能留下的空行。
        cleaned_pages.append(clean_pdf_text("\n".join(kept_lines)))
    # 把规范化 key 转回人能读懂的原始文字。
    removed = [original_lines[key] for key in sorted(watermark_keys)]
    # 日志可以帮助管理员判断是否误删。
    logger.info("自动去除 %s 条跨页重复文字水印：%s", len(removed), removed)
    return cleaned_pages, removed


@dataclass(slots=True)
class TextChunk:
    """一个可独立向量化和检索的文本块。"""

    # 分块在当前文档中的顺序，从 0 开始；它会参与生成稳定的 Milvus 主键。
    index: int
    # 实际送给 BGE-M3 并保存在 Milvus 中的完整原文。
    text: str
    # 当前块前 120 个字符，用于文档摘要和管理页面快速预览。
    summary: str


def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[TextChunk]:
    """完成第 2 步：按段落/句子组织固定长度块并保留重叠上下文。"""
    # size 太小会让语义支离破碎；overlap 必须小于 size，否则滑动窗口无法前进。
    if size < 100 or overlap < 0 or overlap >= size:
        raise ValueError("分块参数不合法")
    # units 保存最小语义单元：先按空行切段，再按中英文句末标点切句。
    units: list[str] = []
    # clean_text 在分割前统一换行和空白，减少格式差异造成的空块。
    for paragraph in re.split(r"\n\s*\n", clean_text(text)):
        units.extend(
            value.strip()
            # 正向后视确保句末标点保留在原句中，而不是被正则删除。
            for value in re.split(r"(?<=[。！？；.!?;])\s*", paragraph.strip())
            if value.strip()
        )
    # chunks 是已经完成的块；current 是仍在尝试追加句子的当前块。
    chunks: list[str] = []
    current = ""
    for unit in units:
        # 单个句子本身超过上限时，无法按句子切，只能使用带重叠的字符滑窗。
        if len(unit) > size:
            # 先封存之前累计的正常短句，防止它们和超长句混在一起。
            if current:
                chunks.append(current.strip())
                current = ""
            # 每次移动 size-overlap 个字符，因此相邻窗口会共享 overlap 个字符。
            chunks.extend(unit[start : start + size] for start in range(0, len(unit), size - overlap))
            continue
        # candidate 表示“把当前句放入 current 后”的试装结果。
        candidate = f"{current}\n{unit}".strip()
        if current and len(candidate) > size:
            # 超出上限时封存旧块，并用旧块尾部重叠内容开启下一块。
            chunks.append(current.strip())
            current = f"{current[-overlap:] if overlap else ''}{unit}".strip()
        else:
            # 未超过上限则继续累积，让一个块尽量包含完整且连续的语义。
            current = candidate
    # 循环结束后 current 不会再触发“超长”，需要手动把最后一块加入结果。
    if current:
        chunks.append(current.strip())
    # enumerate 生成稳定顺序；摘要不额外调用模型，直接截取开头以控制入库成本。
    return [TextChunk(index, value, value[:120]) for index, value in enumerate(chunks)]

