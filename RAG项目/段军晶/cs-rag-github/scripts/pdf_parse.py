# -*- coding: utf-8 -*-
"""
PDF 版面解析脚本（离线数据管线 · 第一步）

职责：
    调用 MinerU 解析知识库源 PDF，提取文本与**页码元数据**，
    输出统一的中间格式，供 chunk_split.py 消费。

★ 页码元数据在本脚本产生 ★
    MinerU 输出的每个内容块携带 page_idx（0 基）。
    本脚本统一转换为 page_no = page_idx + 1（1 基，符合用户阅读习惯）。
    这是全链路页码溯源的起点，后续所有环节都不得丢弃该字段。

★ 解析模式必须为 ocr ★
    本项目实测发现，MinerU 的 txt 模式（直接抽取 PDF 文字层）会**丢失行首字符**：
        "GB/T 11457 信息技术 软件工程术语"    ->  "/ 信息技术 软件工程术语"
        "a）需方依此确定软件质量量化评价需求…"  ->  ") 需方依此确定…"
    对标准知识库而言，标准编号恰是用户最常用的检索线索，丢失后按编号提问将
    完全无法命中。同一页改用 ocr 模式后字符全部完整还原。
    因此 MINERU_METHOD 固定为 ocr，代价是解析耗时增加（CPU 环境约 10-20 秒/页）。

    另注：不要用 --start/--end 做分页解析后拼接 —— 分页解析时 MinerU 输出的
    page_idx 会从 0 重新计数，导致页码错位。

兜底设计：
    MinerU 因模型或环境异常无法完成解析时，自动降级为轻量解析方案。
    降级后**页码元数据依然保留**（按页遍历 PDF），
    确保「页码溯源」这一硬性要求在任何情况下都不失效。

用法：
    python -m scripts.pdf_parse                 # 解析 source_docs 下全部 PDF
    python -m scripts.pdf_parse --file xxx.pdf  # 只解析指定文件
    python -m scripts.pdf_parse --force         # 忽略已有产物，强制重新解析
    python -m scripts.pdf_parse --no-fallback   # 禁止降级（MinerU 失败即报错）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 允许以脚本方式直接运行：python scripts/pdf_parse.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


# ===========================================================================
# 工具函数
# ===========================================================================

def compute_file_hash(path: Path) -> str:
    """计算文件 SHA256，用于去重与 doc_id 生成"""
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        # 每次读 1MB 分块累加计算，这样几十 MB 的大 PDF 也只需很小的内存。
        # 迭代到读不出内容（空字节）为止。
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def make_doc_id(file_hash: str) -> str:
    """由文件哈希生成稳定的文档 ID（内容不变则 ID 不变）"""
    # 取哈希前 16 位：既是内容指纹，长度又够短。
    # 关键在于"稳定"—— 同一份 PDF 无论解析多少次，doc_id 都完全一样，
    # 重复入库时才能按 doc_id 精确覆盖旧数据而不产生重复。
    return file_hash[:16]


def safe_stem(name: str) -> str:
    """把文件名转成安全的目录/文件名词干"""
    stem = Path(name).stem
    # 把汉字、字母、数字、下划线、连字符之外的所有字符都换成下划线。
    # 标准文件名里常带空格、括号、斜杠，直接拿来做目录名会出错。
    stem = re.sub(r"[^\w一-鿿\-]+", "_", stem)
    # 万一替换后什么都不剩（例如文件名全是符号），退回一个固定名字兜底。
    return stem.strip("_") or "document"


def read_json(path: Path) -> Any:
    """以 UTF-8 读取 JSON"""
    # 全程强制 UTF-8：Windows 默认是 GBK，不加这个会把中文读成乱码。
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    """以 UTF-8 写入 JSON（保留中文，不转义）"""
    # 目录不存在就自动建出来（parents=True 会连中间层级一起建）。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        # ensure_ascii=False 让中文按原样写进去，产物可以人工直接翻看；
        # indent=2 是为了可读 —— 答辩时能打开这个文件展示中间解析结果。
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ===========================================================================
# MinerU 解析
# ===========================================================================

def _find_content_list(mineru_out_dir: Path) -> Optional[Path]:
    """
    在 MinerU 输出目录中定位 content_list 文件。

    不写死路径结构：MinerU 不同版本的输出目录层级不同，
    统一用递归查找，优先取较新的 _content_list.json。
    """
    # 用递归查找而不是写死路径：MinerU 各版本输出的目录层级并不一致。
    # 排序键先看路径深度、再看文件名，取最浅层的那个 ——
    # 层级越浅说明越靠近输出根目录，越可能就是本次真正的产物。
    candidates = sorted(
        mineru_out_dir.rglob("*_content_list.json"),
        key=lambda p: (len(p.parts), p.name),
    )
    # 一个都没找到就返回 None，由调用方决定是否走兜底解析。
    return candidates[0] if candidates else None


# MinerU 命令行入口：用「当前解释器 + 入口脚本」调用，而非全局 mineru 命令。
# 原因：全局命令绑定的是安装时的解释器，可能与本项目虚拟环境的依赖版本不一致，
# 从而加载到错误的 numpy 等底层库（本项目已实际踩过该坑）。
_MINERU_ENTRY = Path(__file__).resolve().parent / "mineru_entry.py"


def run_mineru(pdf_path: Path, out_dir: Path) -> Optional[Path]:
    """
    调用 MinerU 解析 PDF。

    返回 content_list.json 的路径；失败返回 None（由调用方决定是否降级）。
    """
    # 连入口脚本都不在，说明工程不完整，直接放弃。
    if not _MINERU_ENTRY.exists():
        logger.error("MinerU 入口脚本缺失：%s", _MINERU_ENTRY)
        return None

    # 组装调用命令：-b 是后端、-m 是解析模式（必须是 ocr）、-l 是语言。
    # MinerU 是独立的命令行工具，所以这里必须走子进程调用。
    cmd = [
        sys.executable, str(_MINERU_ENTRY),
        "-p", str(pdf_path),
        "-o", str(out_dir),
        "-b", settings.mineru_backend,
        "-m", settings.mineru_method,
        "-l", settings.mineru_lang,
    ]
    logger.info("调用 MinerU 解析：%s", pdf_path.name)
    logger.debug("MinerU 命令：%s", " ".join(cmd))

    started = time.time()
    try:
        # 起子进程跑 MinerU。用子进程的好处是它崩了也不会把本脚本一起带走。
        # encoding/errors 保证中文日志和异常字符都能正常读出；
        # timeout 兜住"解析卡死"的情况（CPU 环境解析一页要 10-20 秒）。
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=settings.mineru_timeout,
        )
    except FileNotFoundError:
        logger.error("未找到 Python 解释器：%s", sys.executable)
        return None
    except subprocess.TimeoutExpired:
        logger.error("MinerU 解析超时（%s 秒）：%s", settings.mineru_timeout, pdf_path.name)
        return None

    elapsed = time.time() - started

    # 退出码非 0 说明 MinerU 自己报错了。返回 None，
    # 由上层决定是降级走兜底、还是直接判为失败。
    if proc.returncode != 0:
        # 截取尾部错误信息，避免日志被刷屏
        tail = (proc.stderr or proc.stdout or "")[-800:]
        logger.error("MinerU 解析失败（退出码 %s）：%s\n%s",
                     proc.returncode, pdf_path.name, tail)
        return None

    # 命令跑成功了但没产出文件，同样判为失败。
    content_list = _find_content_list(out_dir)
    if content_list is None:
        logger.error("MinerU 未产出 content_list.json：%s", pdf_path.name)
        return None

    logger.info("MinerU 解析完成：%s | 耗时 %.1f 秒 | 产物 %s",
                pdf_path.name, elapsed, content_list.name)
    return content_list


# ===========================================================================
# 兜底解析（MinerU 不可用时）
# ===========================================================================

def fallback_parse(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    轻量兜底解析：用 pdfplumber 逐页提取文本。

    ★ 页码元数据保留 ★
        按页遍历，天然知道每段文本属于第几页，page_no 直接可得。
        这是本兜底方案能守住「页码溯源」硬性要求的关键。

    缺点：丢失版面结构（标题层级、表格结构、图表），因此仅作为降级路径。
    """
    # 延迟导入：只有真的走兜底时才需要 pdfplumber。
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("兜底解析需要 pdfplumber，请先安装：pip install pdfplumber")

    logger.warning("启用兜底解析（pdfplumber）：%s", pdf_path.name)
    blocks: List[Dict[str, Any]] = []
    # 内容块的递增编号
    block_id = 0

    # ★ 按页遍历 ★ 这正是兜底方案还能保住页码的根本原因：
    # 某段文本属于第几页，在循环到那一页时就天然知道了，无需任何推断。
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_no = page_index + 1  # ★ 页码在此产生（1 基）
            # 抽出整页文本（抽不出来时用空串兜底）
            text = page.extract_text() or ""

            # 按空行切段，过滤过短片段
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            if not paragraphs:
                # 整页没有空行分隔时，按行聚合
                paragraphs = [ln.strip() for ln in text.splitlines() if ln.strip()]

            # 每个段落作为一个内容块，页码统一用当前页。
            for para in paragraphs:
                # 字段与 MinerU 链路刻意保持一致，这样下游 chunk_split.py
                # 完全不需要区分数据是从哪条链路来的。
                # 版面结构相关的字段（bbox / 图片路径 / 标题层级）填 None 或 0，
                # 因为兜底方案拿不到这些信息 —— 这正是它只能当降级方案的原因。
                blocks.append({
                    "block_id": block_id,
                    "type": "text",
                    "text": para,
                    "page_no": page_no,
                    "text_level": 0,
                    "bbox": None,
                    "img_path": None,
                    "table_body": None,
                    "caption": "",
                    "footnote": "",
                })
                block_id += 1

    logger.info("兜底解析完成：%s | %d 个内容块", pdf_path.name, len(blocks))
    return blocks


# ===========================================================================
# content_list.json -> 统一中间格式
# ===========================================================================

def _plain_text(value: Any) -> str:
    """把 MinerU 的字段（可能是 list 或 str）统一转成纯文本"""
    # MinerU 有些字段（比如表格标题）给的是列表，有些给的是字符串。
    # 这里统一规整成字符串，免得下游到处写类型判断。
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        # 多项内容用换行拼接，顺便把空元素过滤掉。
        return "\n".join(str(v) for v in value if v)
    return str(value)


def convert_content_list(content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    把 MinerU 的 content_list 转换成统一中间格式。

    ★ 页码转换 ★
        page_idx（0 基）-> page_no（1 基）：page_no = page_idx + 1

    同时跟踪标题层级，为后续每个内容块标注所属章节（提升答案可读性）。
    """
    blocks: List[Dict[str, Any]] = []
    # 当前所在章节的标题。一旦遇到标题块就更新它，
    # 后面的每个块都带上这个值，答案的"章节归属"就是这样来的。
    current_section = ""

    # 需要跳过的页眉页脚与页码行（不承载正文语义）
    skip_types = {"header", "footer", "page_number"}

    for idx, raw in enumerate(content_list):
        # 结构不正常的条目直接跳过，避免一条脏数据把整篇解析带崩。
        if not isinstance(raw, dict):
            continue

        block_type = str(raw.get("type", "text")).lower()
        if block_type in skip_types:
            continue

        # ---- 页码：0 基转 1 基 ----
        # MinerU 的 page_idx 从 0 开始计，而用户看到的是"第 1 页"，
        # 所以这里 +1 转成 1 基。★ 这一步就是全链路页码溯源的起点 ★
        page_idx = raw.get("page_idx")
        if page_idx is None:
            # 个别版本用的字段名不一样，兼容一下
            page_idx = raw.get("page_no_idx", 0)
        try:
            page_no = int(page_idx) + 1
        except (TypeError, ValueError):
            # 页码取不出来时兜底为第 1 页，保证这个字段永远有值、永不为空。
            page_no = 1

        # ---- 标题跟踪 ----
        # text_level >= 1 表示这是个标题块（层级越高，数值越小）。
        text_level = int(raw.get("text_level") or 0)
        body = _plain_text(raw.get("text"))

        # 遇到新标题就更新"当前章节"，后续块都会归到它下面。
        # 截到 120 字是为了防止个别超长标题把展示撑爆。
        if block_type == "text" and text_level >= 1 and body.strip():
            current_section = body.strip()[:120]

        # ---- 按类型构造块 ----
        if block_type == "table":
            # 表格块：把表题和表体拼成 content。表题一并保留是因为
            # 用户提问时常常会用表题里的措辞（比如直接说某张表的名称）。
            caption = _plain_text(raw.get("table_caption"))
            table_body = _plain_text(raw.get("table_body"))
            content = "\n".join(x for x in (caption, table_body) if x)
            blocks.append({
                "block_id": len(blocks),
                "type": "table",
                "text": content,
                "page_no": page_no,
                "text_level": 0,
                "bbox": raw.get("bbox"),
                "img_path": raw.get("img_path"),
                "table_body": table_body,
                "caption": caption,
                "footnote": _plain_text(raw.get("table_footnote")),
                "section_title": current_section,
            })

        elif block_type == "image":
            # 图片块：这一步只拿得到图注。图内的语义要等 vision_caption.py
            # 调视觉模型补上，再由 chunk_split.py 拼进这个块的 content。
            caption = _plain_text(raw.get("image_caption"))
            blocks.append({
                "block_id": len(blocks),
                "type": "image",
                # 没有图注时退而取 content 字段，尽量给这个块留点可检索的文本。
                "text": caption or raw.get("content") or "",
                "page_no": page_no,
                "text_level": 0,
                "bbox": raw.get("bbox"),
                "img_path": raw.get("img_path"),
                "table_body": None,
                "caption": caption,
                "footnote": _plain_text(raw.get("image_footnote")),
                "section_title": current_section,
            })

        else:
            # 正文、公式等一律按文本处理
            # 没有内容的空块没有收录价值，跳过。
            if not body.strip():
                continue
            blocks.append({
                "block_id": len(blocks),
                # 公式单独标类型，其余一律按正文处理
                "type": "equation" if block_type == "equation" else "text",
                "text": body,
                "page_no": page_no,
                "text_level": text_level,
                "bbox": raw.get("bbox"),
                "img_path": None,
                "table_body": None,
                "caption": "",
                "footnote": "",
                "section_title": current_section,
            })

    return blocks


# ===========================================================================
# 主流程
# ===========================================================================

def parse_one(pdf_path: Path, *, force: bool = False,
              allow_fallback: bool = True) -> Optional[Dict[str, Any]]:
    """
    解析单份 PDF，产出统一中间格式并落盘。

    返回解析结果摘要（含 doc_id、页数、块数、实际使用的解析引擎）。
    """
    # 先算内容指纹，再由指纹定出 doc_id —— 同一份 PDF 每次都得到同一个 ID。
    file_hash = compute_file_hash(pdf_path)
    doc_id = make_doc_id(file_hash)
    # 文件名里可能有空格括号，转成安全的词干当作各种产物的公共前缀。
    stem = safe_stem(pdf_path.name)

    # 统一中间产物：xxx.parsed.json，这就是交给 chunk_split.py 的文件。
    out_file = settings.parsed_path / f"{stem}.parsed.json"
    # 已有产物就跳过（增量解析，省掉最贵的 MinerU 那一步，动辄几十秒一页）；
    # 带 --force 时忽略它，强制重新解析。
    if out_file.exists() and not force:
        logger.info("已存在解析产物，跳过（--force 可强制重解析）：%s", out_file.name)
        return read_json(out_file)

    # MinerU 的原生产物（图片、content_list 等）都放在这个目录下。
    doc_out_dir = settings.parsed_path / stem
    # 记录本次实际走的是哪条链路，最终写进产物便于事后追溯。
    engine = "mineru"

    # ---- 主链路：MinerU ----
    # 强制重解析时先把上次的原生产物清掉，免得新旧结果混在一起。
    if doc_out_dir.exists() and force:
        shutil.rmtree(doc_out_dir, ignore_errors=True)
    doc_out_dir.mkdir(parents=True, exist_ok=True)

    content_list_path = run_mineru(pdf_path, doc_out_dir)
    blocks: List[Dict[str, Any]] = []

    if content_list_path is not None:
        try:
            # 把 MinerU 的原始输出翻译成我们统一的中间格式。
            raw_list = read_json(content_list_path)
            blocks = convert_content_list(raw_list)
        except Exception as exc:
            # 转换失败就当作"什么都没解析出来"，交给下面的兜底链路接手。
            logger.error("解析 content_list 失败：%s", exc)
            blocks = []

    # ---- 兜底链路 ----
    if not blocks:
        if not allow_fallback or not settings.mineru_fallback_enabled:
            # 两个开关任意一个关掉就不降级，直接失败返回。
            logger.error("MinerU 解析无结果，且兜底已禁用：%s", pdf_path.name)
            return None
        # 降级：改用 pdfplumber 逐页抽文本。
        # 版面结构（标题层级、表格结构、图表）会丢，但页码照样完整保留 ——
        # 也就是说"页码溯源"这条底线仍然守住了。
        engine = "fallback"
        blocks = fallback_parse(pdf_path)

    # 兜底也没抽出东西来，只能算失败。
    if not blocks:
        logger.error("解析后没有任何内容块：%s", pdf_path.name)
        return None

    # ---- 统计页数 ----
    # 优先直接读 PDF 本身拿到准确总页数（包含那些没有文字的空页）。
    page_count = 0
    try:
        from pypdf import PdfReader
        page_count = len(PdfReader(str(pdf_path)).pages)
    except Exception:
        # 读不了就退而求其次：取所有块里最大的页码当作页数（可能偏小）。
        page_count = max((b["page_no"] for b in blocks), default=0)

    # 最终产物：文档级元信息 + 全部内容块（每块都带 page_no）。
    result = {
        "doc_id": doc_id,
        "file_name": pdf_path.name,
        "file_path": str(pdf_path.resolve()),
        "file_hash": file_hash,
        "page_count": page_count,
        "block_count": len(blocks),
        "parse_engine": engine,
        "parsed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "blocks": blocks,
    }
    # 落盘，接下来交给 chunk_split.py 消费。
    write_json(out_file, result)

    # 日志里特意标出走的哪条链路：看到 引擎=fallback 就知道
    # 这次没跑成 MinerU，版面结构（标题层级、表格结构）是缺失的。
    logger.info("解析完成：%s | 引擎=%s | %d 页 | %d 个内容块 | 产物 %s",
                pdf_path.name, engine, page_count, len(blocks), out_file.name)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="PDF 版面解析（MinerU 主链路 + 轻量兜底）")
    parser.add_argument("--file", type=str, default=None,
                        help="只解析指定文件名（位于 SOURCE_DOCS_DIR 下）")
    parser.add_argument("--force", action="store_true",
                        help="忽略已有产物，强制重新解析")
    parser.add_argument("--no-fallback", action="store_true",
                        help="禁止降级：MinerU 失败即终止")
    args = parser.parse_args()

    setup_logging()
    settings.ensure_directories()

    # 待解析的源 PDF 都放在这个目录下。
    src_dir = settings.source_docs_path
    if not src_dir.exists():
        logger.error("源 PDF 目录不存在：%s", src_dir)
        return 1

    # --file 指定单个文件，否则扫目录下的全部 PDF。
    if args.file:
        targets = [src_dir / args.file]
    else:
        targets = sorted(src_dir.glob("*.pdf"))

    if not targets:
        logger.error("未在 %s 下找到任何 PDF 文件", src_dir)
        return 1

    logger.info("开始解析 %d 份 PDF：%s", len(targets), src_dir)

    success, failed = 0, 0
    # 逐份解析。单份失败只计数并继续 —— 一份坏 PDF 不该让整批任务停摆。
    for pdf_path in targets:
        if not pdf_path.exists():
            logger.error("文件不存在：%s", pdf_path)
            failed += 1
            continue
        try:
            # allow_fallback 取反：命令行传了 --no-fallback 就禁止降级。
            result = parse_one(pdf_path, force=args.force,
                               allow_fallback=not args.no_fallback)
            if result:
                success += 1
            else:
                failed += 1
        except Exception as exc:
            logger.exception("解析异常：%s | %s", pdf_path.name, exc)
            failed += 1

    logger.info("解析结束 | 成功 %d 份 | 失败 %d 份", success, failed)
    # 退出码便于自动化脚本判断成败：全部成功是 0，有失败是 1。
    return 0 if failed == 0 else 1


# 支持 python -m scripts.pdf_parse 直接运行
if __name__ == "__main__":
    sys.exit(main())
