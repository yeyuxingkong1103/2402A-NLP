# -*- coding: utf-8 -*-
"""
文本分块脚本（离线数据管线 · 第二步）

职责：
    读取 pdf_parse.py 的中间产物，用 LangChain 切分成检索单元（chunk），
    并保证**每个 chunk 都完整继承页码元数据**。

★ 页码元数据存活规则（本项目最关键的实现细节）★

    MinerU 输出的是「内容块」，每个块自带页码；而 LangChain 切分时
    一个块可能被切成多段、相邻小块也可能被合并。若不专门处理，
    页码会在切分/合并过程中丢失。因此：

      情形 1：单个内容块未超过 chunk_size
              -> 直接成块，页码 = 该块页码
      情形 2：单个内容块被切成多段
              -> 每段继承同一页码
      情形 3：多个相邻内容块合并成一个 chunk
              -> page_nums = 各块页码去重排序后的列表
                 page_no   = 首个块的页码（主页码，前端展示用）

    实现方式：把连续的正文块拼成一段长文本，同时记录「字符区间 -> 页码」
    的映射表；切分后按字符偏移反查每个 chunk 覆盖了哪些页。
    这样即使 chunk 跨页，也能精确得到完整页码列表。

    表格块与图片块作为独立内容块**单独成块，不参与文本切分**
    （表格被拦腰截断会导致语义碎裂，检索命中也无法被正确理解）。

用法：
    python -m scripts.chunk_split                # 处理全部解析产物
    python -m scripts.chunk_split --file xxx     # 只处理指定文档
    python -m scripts.chunk_split --show 3       # 展示前 3 个块便于抽查
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 允许以脚本方式直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# 中文场景的分隔符优先级：先按段落，再按句子，最后按标点
# 切分器会从前往后尝试：能在段落边界断开就优先在段落断，不行才退到句子、
# 标点，最后的空字符串表示"实在没有可断的地方就按字符数硬切"。
# 这样切出来的块尽量落在自然的语义边界上，而不是把一句话拦腰剪断。
_SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", "、", " ", ""]

# 参与文本切分的块类型；表格与图片不参与
# 正文与公式都是连续文本，可以拼接成一大段后统一切分。
_TEXT_TYPES = {"text", "equation"}
# 表格和图片必须独占一个块：表格被拦腰截断会导致语义碎裂，
# 检索命中了也读不懂，所以它们不参与文本切分。
_STANDALONE_TYPES = {"table", "image"}


# ===========================================================================
# 页码映射：字符区间 -> 页码
# ===========================================================================

class PageSpanMapper:
    """
    维护「字符区间 -> 页码 / 章节标题」映射。

    把连续的内容块拼接成一段长文本时，同步记录每块在长文本中的起止位置，
    切分后即可按偏移反查任意 chunk 覆盖的页码，实现跨页 chunk 的精确溯源。
    """

    def __init__(self) -> None:
        # 长文本按块分段存放，最后 join 起来就是完整正文
        self._parts: List[str] = []
        # 游标：下一个块将从长文本的第几个字符开始写
        self._cursor = 0
        # ★ 页码映射表 ★ 每加入一个块就记一条「起止字符位置 -> 页码」。
        # 切分之后靠它反查某个 chunk 覆盖了哪几页（见 pages_for）。
        self._page_spans: List[Tuple[int, int, int]] = []      # (start, end, page_no)
        # 同样记录「起止位置 -> 章节标题」，用于给 chunk 补上所属章节
        self._section_spans: List[Tuple[int, int, str]] = []   # (start, end, title)

    def add(self, text: str, page_no: int, section_title: str) -> None:
        """追加一个内容块"""
        # 空块直接不记，免得污染位置映射表。
        if not text:
            return
        # 块与块之间补一个空行作分隔，这段分隔符也要计入游标，否则偏移量会错。
        if self._parts:
            sep = "\n\n"
            self._parts.append(sep)
            self._cursor += len(sep)

        # 记下这个块在长文本里的起点
        start = self._cursor
        self._parts.append(text)
        self._cursor += len(text)

        # 写入映射表：这段字符区间属于哪一页。
        # 逐个块登记而不是只记整篇，正是"跨页 chunk 也能精确溯源"的关键。
        self._page_spans.append((start, self._cursor, int(page_no)))
        # 有章节信息才登记，避免塞进一堆空标题。
        if section_title:
            self._section_spans.append((start, self._cursor, section_title))

    @property
    def text(self) -> str:
        """拼接后的长文本"""
        # 拼接方式必须固定：所有位置偏移都是基于这个结果算出来的，
        # 拼接规则一改，映射表就对不上了。
        return "".join(self._parts)

    def pages_for(self, start: int, end: int) -> List[int]:
        """返回 [start, end) 区间覆盖到的全部页码（升序去重）"""
        # 遍历映射表，把这个区间**碰到过**的所有块的页码收集起来。
        # 用集合是为了去重：一个 chunk 可能横跨同一页里的好几个块，
        # 不去重的话同一页会出现多次。
        pages = {
            page for (s, e, page) in self._page_spans
            if not (e <= start or s >= end)   # 区间有交集
        }
        # 排序保证页码顺序稳定（前端展示、评测比对都需要确定性结果）；
        # 万一一个都没匹配上则返回 [1] 兜底，绝不允许页码为空。
        return sorted(pages) or [1]

    def section_for(self, start: int) -> str:
        """返回 start 位置所属的章节标题"""
        # 哪个区间把这个位置包住了，就返回哪个区间的标题。
        for (s, e, title) in self._section_spans:
            if s <= start < e:
                return title
        # 落在所有章节区间之外时返回空串（前端会显示为无章节）。
        return ""


# ===========================================================================
# 分块主体
# ===========================================================================

def _make_splitter():
    """构造 LangChain 递归字符切分器"""
    # 延迟导入：LangChain 加载较慢，只在真需要切分时才引进来。
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        # 每个块的目标长度（字符数）
        chunk_size=settings.chunk_size,
        # 相邻块之间的重叠字数：防止刚好被切断的那句话两边都读不完整
        chunk_overlap=settings.chunk_overlap,
        separators=_SEPARATORS,
        # 保留分隔符本身（句号、换行等），否则拼接处会丢标点、影响可读性
        keep_separator=True,
        # 用字符数计长：中文按字数算比按 token 算更直观、也更好控制
        length_function=len,
    )


def _split_with_pages(
    mapper: PageSpanMapper,
    splitter,
) -> List[Dict[str, Any]]:
    """
    对拼接后的长文本切分，并为每段反查页码。

    切分结果保持原文顺序，因此可依次在长文本中定位每段的位置，
    再按位置区间查出页码列表。
    """
    full_text = mapper.text
    # 全是空白（这一轮没攒到正文块）时直接返回，不必走切分。
    if not full_text.strip():
        return []

    # ★ 真正执行切分 ★ 把长文本切成一个个检索单元。
    pieces = splitter.split_text(full_text)
    results: List[Dict[str, Any]] = []
    # cursor 是"下一次查找的起点"，用来在长文本里按顺序定位每一个片段。
    cursor = 0

    for piece in pieces:
        # 切分器偶尔会产出纯空白片段，丢掉。
        if not piece.strip():
            continue

        # 依序定位该片段在原文中的位置。
        # 从 cursor 往后找而不是每次都从头找：因为文本里可能有重复内容，
        # 顺序查找才能保证定位到的是"真正的那一段"，也让偏移量单调递增。
        pos = full_text.find(piece, cursor)
        if pos < 0:
            # 理论上不会发生（切分保序）；兜底从当前位置继续
            pos = cursor
        end = pos + len(piece)

        # ★ 核心一步：用字符区间反查页码 ★
        # 这一行就是"跨页 chunk 也能拿到完整页码列表"的实现所在。
        pages = mapper.pages_for(pos, end)
        results.append({
            "content": piece,
            # 主页码：取覆盖到的第一页，前端显示"第 X 页"用它
            "page_no": pages[0],          # 主页码：首个覆盖页
            # 完整页码列表：跨页时是多个值，保证溯源信息不被抹掉
            "page_nums": pages,           # 完整页码列表（跨页时为多值）
            # 用片段起点所在位置反查所属章节
            "section_title": mapper.section_for(pos),
            "content_type": "text",
        })

        # 允许重叠：下一段的搜索起点只前进 1 个字符，避免跳过重叠区域
        cursor = max(cursor, pos + 1)

    return results


# 视觉描述在图注之后的拼接标记（前导换行符计入长度预算）
# 用一个固定标记把"图注"和"模型生成的描述"分开，除了可读性好之外，
# 还让重建旧索引时能靠它精确剥离描述（见 scripts/build_v1_index.py）。
_VISION_MARK = "\n[视觉解析] "


def compose_block_content(caption: str, vision_desc: str, *, max_chars: int) -> str:
    """
    拼接独立块（表格 / 图片）的 content，并做保护性截断。

    为什么需要保护性截断：
        直接对「图注 + 视觉描述」做硬切片，可能把描述拦腰截断，
        甚至在图注很长时把描述**整段切掉** —— 那样视觉解析就白做了，
        而且没有任何报错，属于静默失效。
        本函数保证：**图注优先完整保留，剩余预算分配给描述**。

    为什么是纯函数：
        截断是本项目最容易出静默错误的地方，抽成不依赖 I/O 的纯函数
        才能用大量边界用例覆盖（见 tests/test_chunk_compose.py）。

    参数：
        caption     : 原始图注（或表格文本）
        vision_desc : 视觉模型解析出的描述，可为空
        max_chars   : content 字符上限（来自 settings.chunk_max_chars）

    返回：长度不超过 max_chars 的 content 字符串。
    """
    caption = (caption or "").strip()
    vision_desc = (vision_desc or "").strip()

    # 没有视觉描述（V1 行为，或这张图解析失败）—— 那就只有图注。
    if not vision_desc:
        return caption[:max_chars]
    # 反过来，没有图注时把全部预算让给描述。
    if not caption:
        return vision_desc[:max_chars]

    # 算一下留给描述的字符预算：总上限 − 图注 − 那个拼接标记。
    budget = max_chars - len(caption) - len(_VISION_MARK)
    if budget <= 0:
        # 图注已占满预算，放弃描述，保证图注完整
        return caption[:max_chars]

    # 预算够用，图注与描述都完整保留。
    if len(vision_desc) <= budget:
        return caption + _VISION_MARK + vision_desc

    # 截断时预留一个字符给省略号，保证总长度恰好不超过上限
    # （不留这一字符的话，加上省略号反而会超出一个字）。
    truncated = vision_desc[: max(0, budget - 1)]
    return caption + _VISION_MARK + truncated + "…"


def load_vision_cache(parsed_dir: Path, stem: str) -> Dict[str, Dict[str, Any]]:
    """
    读取视觉解析产物 data/parsed/<stem>.vision.json。

    文件不存在时返回空字典，使分块流程在**没有视觉产物时也能正常运行**
    （退回 V1 行为：image chunk 的 content 只有图注）。
    这样 V2 的视觉解析就是可选的增量步骤，而非必需前置条件。
    """
    path = parsed_dir / f"{stem}.vision.json"
    # 没有这个文件是完全正常的情况（没跑过视觉解析），
    # 所以用 INFO 而不是 ERROR —— 它不是错误，只是这一步没做。
    if not path.exists():
        logger.info("未找到视觉解析产物，图片块将仅保留图注 | %s", path.name)
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        # 读坏了也照常往下走：退回"只有图注"，而不是让分块失败。
        logger.warning("视觉解析产物读取失败，按无描述处理 | %s | %s", path.name, exc)
        return {}
    # 结构不对（不是字典）时同样返回空，保证调用方拿到的类型是稳定的。
    return data if isinstance(data, dict) else {}


def _standalone_block(
    block: Dict[str, Any],
    vision_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    把表格块 / 图片块转换为独立 chunk 记录（不参与切分）。

    V2 变化：
        图片块若存在视觉解析描述，会以「图注 + [视觉解析] 描述」的形式
        并入 content，使图内语义可被检索与生成使用。
        拼接与截断统一交给 compose_block_content 处理，保证不超上限、
        且图注优先完整保留。
    """
    block_type = block.get("type", "text")
    text = (block.get("text") or "").strip()
    page_no = int(block.get("page_no", 1))

    if block_type == "image":
        # 用图片文件名（img_path 的 basename）去视觉缓存中查描述 ——
        # 两边约定的键就是这个文件名，见 vision_caption.py。
        image_name = Path(block.get("img_path") or "").name
        entry = (vision_cache or {}).get(image_name) or {}
        # 只有 status=ok 的描述才采用；failed 的当作没有，退回仅图注。
        vision_desc = entry.get("desc", "") if entry.get("status") == "ok" else ""
        # 连图注都没有时造一句占位说明，
        # 至少让这个块能被检索到、页码能被展示出来。
        if not text:
            text = f"[图片] 位于第 {page_no} 页的插图"

        # 把图注与视觉描述拼起来（内部保证不超长、且图注优先完整）。
        content = compose_block_content(
            text, vision_desc, max_chars=settings.chunk_max_chars
        )
        # 截断是静默的，必须留痕，否则无从排查描述为何不完整
        full_len = len(text) + len(_VISION_MARK) + len(vision_desc)
        if vision_desc and full_len > settings.chunk_max_chars:
            logger.warning(
                "图片块描述超长已截断 | 第 %s 页 | 拼接后长度=%d | 上限=%d",
                page_no, full_len, settings.chunk_max_chars,
            )
    else:
        # 表格块：直接用表格文本（没有就造一句占位说明），超长则硬截断。
        content = (text or f"[表格] 位于第 {page_no} 页的表格")[
            : settings.chunk_max_chars
        ]

    return {
        "content": content,
        # 独立块只属于一页，所以主页码就是它自己的页码
        "page_no": page_no,
        # 表格/图片独立成块，页码归属唯一，不会跨页
        "page_nums": [page_no],
        "section_title": block.get("section_title", ""),
        "content_type": block_type,
    }


def split_blocks(
    blocks: List[Dict[str, Any]],
    vision_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    把解析产出的内容块序列切分为最终 chunk 列表。

    表格、图片块会打断文本流，独立成块；正文块按顺序拼接后统一切分。

    参数：
        vision_cache : 视觉解析产物（图片文件名 -> 描述信息），
                       V2 用于把图内语义并入图片块；为空时退回 V1 行为。
    """
    splitter = _make_splitter()
    # mapper 负责"一边拼正文、一边记页码映射"
    mapper = PageSpanMapper()
    chunks: List[Dict[str, Any]] = []

    def flush_text() -> None:
        """把当前累积的正文切分并追加到结果"""
        nonlocal mapper
        # 确实攒到了正文才做切分
        if mapper.text.strip():
            chunks.extend(_split_with_pages(mapper, splitter))
        # 结算完就换一个新的 mapper，开始攒下一段正文 ——
        # 被表格/图片打断的两段正文不能跨越它们拼在一起，否则页码映射会错。
        mapper = PageSpanMapper()

    for block in blocks:
        block_type = block.get("type", "text")
        page_no = int(block.get("page_no", 1))
        section = block.get("section_title", "") or ""

        if block_type in _STANDALONE_TYPES:
            # 独立块：先结算前面的正文，再单独成块
            flush_text()
            chunks.append(_standalone_block(block, vision_cache))

        elif block_type in _TEXT_TYPES:
            # 正文/公式块先不切分，而是攒进 mapper（同时登记页码），
            # 等攒够一段连续正文后再统一按 chunk_size 切。
            mapper.add((block.get("text") or "").strip(), page_no, section)

        # 其余类型（如公式）已并入 text 处理，未知类型直接忽略

    # 循环结束后，把最后一段还没结算的正文切出来。
    flush_text()
    return chunks


def attach_ids(
    chunks: List[Dict[str, Any]],
    *,
    doc_id: str,
    file_name: str,
) -> List[Dict[str, Any]]:
    """为 chunk 补充 ID 与来源信息（chunk_id / doc_id / source_file）"""
    for index, chunk in enumerate(chunks):
        # 拼出全局唯一的块 ID（文档 ID + 四位序号）。
        # 它是 Milvus 向量记录、MySQL 元数据、在线检索结果三方的关联键 ——
        # 在线问答时就是拿着这个 ID 回 MySQL 查出文件名和页码的。
        chunk["chunk_id"] = f"{doc_id}_{index:04d}"
        chunk["doc_id"] = doc_id
        chunk["chunk_index"] = index
        chunk["source_file"] = file_name
        # 保证 page_nums 始终为非空列表
        if not chunk.get("page_nums"):
            chunk["page_nums"] = [int(chunk.get("page_no", 1))]
        chunk["page_no"] = int(chunk.get("page_no", 1))
    return chunks


# ===========================================================================
# 主流程
# ===========================================================================

def process_one(parsed_file: Path, *, force: bool = False) -> Dict[str, Any]:
    """处理单份解析产物，输出 chunk 列表"""
    data = json.loads(parsed_file.read_text(encoding="utf-8"))
    doc_id = data["doc_id"]
    file_name = data["file_name"]

    # 产物文件名：xxx.parsed.json -> xxx.chunks.json
    out_file = parsed_file.with_name(parsed_file.name.replace(
        ".parsed.json", ".chunks.json"))

    # 已经有产物就直接复用（增量处理）；带 --force 时强制重切。
    if out_file.exists() and not force:
        logger.info("已存在分块产物，跳过（--force 可强制重分块）：%s", out_file.name)
        return json.loads(out_file.read_text(encoding="utf-8"))

    blocks = data.get("blocks", [])

    # V2：加载视觉解析产物，把图内语义并入图片块（无产物时自动退回 V1 行为）
    stem = parsed_file.name[: -len(".parsed.json")]
    vision_cache = load_vision_cache(parsed_file.parent, stem)
    if vision_cache:
        # 只统计成功的条数；failed 的不算（它们会退回仅图注）。
        ok_count = sum(1 for v in vision_cache.values() if v.get("status") == "ok")
        logger.info("已加载视觉描述 %d 条 | %s", ok_count, file_name)

    # 先切块（这一步已经让每个块带上页码），再统一补 ID 与来源信息。
    chunks = split_blocks(blocks, vision_cache)
    chunks = attach_ids(chunks, doc_id=doc_id, file_name=file_name)

    result = {
        "doc_id": doc_id,
        "file_name": file_name,
        "file_path": data.get("file_path", ""),
        "file_hash": data.get("file_hash", ""),
        "page_count": data.get("page_count", 0),
        "parse_engine": data.get("parse_engine", ""),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    # 落盘：下一站 embedding_store.py 就读这个文件做向量化入库。
    out_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 单独统计跨页块数量。这是页码映射机制最值得关注的一类情况：
    # 跨页块越多，说明映射表越是在真正起作用（否则这些块的页码早就丢了）。
    cross_page = sum(1 for c in chunks if len(c["page_nums"]) > 1)
    logger.info(
        "分块完成：%s | %d 个块（其中跨页块 %d 个）| 产物 %s",
        file_name, len(chunks), cross_page, out_file.name,
    )
    return result


def main() -> int:
    # 命令行参数：--show 可以抽查前 N 个块的页码对不对，
    # 这是验证"页码映射有没有生效"最直接的手段。
    parser = argparse.ArgumentParser(description="LangChain 文本分块（保页码元数据）")
    parser.add_argument("--file", type=str, default=None,
                        help="只处理文件名包含该关键字的文档")
    parser.add_argument("--force", action="store_true", help="强制重新分块")
    parser.add_argument("--show", type=int, default=0,
                        help="展示前 N 个块，便于抽查页码是否正确")
    args = parser.parse_args()

    setup_logging()
    settings.ensure_directories()

    # 找出上一步（pdf_parse.py）产出的全部解析结果。
    parsed_files = sorted(settings.parsed_path.glob("*.parsed.json"))
    # --file 时按文件名关键字过滤，只重切指定的那一份。
    if args.file:
        parsed_files = [p for p in parsed_files if args.file in p.name]

    if not parsed_files:
        logger.error("未找到解析产物（*.parsed.json），请先运行 scripts.pdf_parse")
        return 1

    logger.info("开始分块，共 %d 份解析产物", len(parsed_files))

    total_chunks = 0
    # 逐份文档处理，单份失败只记录并继续，不打断其余文档。
    for parsed_file in parsed_files:
        try:
            result = process_one(parsed_file, force=args.force)
            total_chunks += result.get("chunk_count", 0)

            # --show N：把前 N 个块的类型、ID、页码打出来，
            # 这是人工核对"页码有没有跟丢"最快的方式。
            if args.show:
                logger.info("---- %s 前 %d 个块抽查 ----", result["file_name"], args.show)
                for chunk in result["chunks"][: args.show]:
                    # 预览里把换行压成空格，否则一条日志会被拆成好几行。
                    preview = chunk["content"][:60].replace("\n", " ")
                    logger.info(
                        "  [%s] chunk_id=%s page_no=%s page_nums=%s | %s...",
                        chunk["content_type"], chunk["chunk_id"],
                        chunk["page_no"], chunk["page_nums"], preview,
                    )
        except Exception as exc:
            logger.exception("分块失败：%s | %s", parsed_file.name, exc)

    logger.info("全部分块完成 | 共 %d 个 chunk", total_chunks)
    return 0


# 支持 python -m scripts.chunk_split 直接运行
if __name__ == "__main__":
    sys.exit(main())
