# -*- coding: utf-8 -*-
"""
离线视觉解析脚本（V2 · 离线数据管线补充步骤）

职责：
    把 PDF 中图表的**图内语义**转成中文描述，供分块阶段并入知识块。

为什么需要这一步：
    MinerU 只能提取出图注（如「图 2 软件质量模型」），图内的结构信息
    在 V1 中完全缺失 —— 这是基线评测报告 4.5 节记录的核心弱点。
    基线报告中 R08 / G04 / S12 三个样本都指向第 6 页同一个知识块，
    而该块的图内信息（6 个质量特性）完全没有进入知识库。

用法：
    python -m scripts.vision_caption              # 增量解析（已解析的图跳过）
    python -m scripts.vision_caption --force      # 强制重新解析
    python -m scripts.vision_caption --dry-run    # 只列出待解析图片，不调用 API

产物：
    data/parsed/<文档stem>.vision.json
    以**图片文件名**为键，便于分块阶段按 img_path 查找。
    幂等：已成功解析的图片不会重复调用 API。

容错设计：
    单张图解析失败（超时 / 限流 / 返回为空）**不中断整体流程**，
    该图在缓存中标记为 failed，分块阶段据此退回「仅图注」（即 V1 行为）。
    即：一张图失败不会导致整库重建失败。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 图片定位
# ---------------------------------------------------------------------------

def _image_root(parsed_dir: Path, stem: str) -> Optional[Path]:
    """
    定位某份文档的图片根目录。

    MinerU 的目录结构为：
        data/parsed/<文档stem>/<原始文件名>/ocr/images/*.jpg

    而 .parsed.json 里 image 块的 img_path 是**相对的**，
    形如 "images/xxx.jpg"，需要补上前面的目录层级才能定位到真实文件。
    """
    doc_dir = parsed_dir / stem
    # 这份文档的解析目录都不存在，说明它压根没被解析过。
    if not doc_dir.is_dir():
        return None
    # MinerU 会在文档目录下再套一层以原始文件名为名的目录，
    # 取第一个子目录即可（正常情况下只会有一个）。
    subdirs = [d for d in doc_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    # 图片实际落在 <原始文件名>/ocr/images/ 之下，这里先定位到 ocr 这一层。
    return subdirs[0] / "ocr"


def _collect_images(parsed_dir: Path, stem: str) -> List[Dict[str, Any]]:
    """从 .parsed.json 中收集图片块，并把 img_path 解析为真实路径"""
    # 这里读的是 pdf_parse.py 的产物：里面记录了每个内容块的内容与页码。
    parsed_file = parsed_dir / f"{stem}.parsed.json"
    if not parsed_file.exists():
        return []

    try:
        data = json.loads(parsed_file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        # 单份文件读坏只跳过这一份，不影响其它文档继续处理。
        logger.warning("解析产物读取失败，跳过该文档 | %s | %s", parsed_file, exc)
        return []

    root = _image_root(parsed_dir, stem)
    if root is None:
        logger.warning("未找到图片目录，跳过 | %s", stem)
        return []

    images: List[Dict[str, Any]] = []
    # 只挑图片类型的块：文本块和表格块不需要走视觉模型。
    for block in data.get("blocks", []):
        if block.get("type") != "image":
            continue
        img_path = block.get("img_path")
        if not img_path:
            continue
        # img_path 在产物里是相对路径，补上根目录才能定位到真实文件。
        full = root / img_path
        if not full.exists():
            logger.warning("图片文件不存在，跳过 | %s", full)
            continue
        # 整理成后续要用的形态：
        #   image_name 图片文件名，在整个流程里当键用（缓存、分块阶段回查都靠它）
        #   caption    图注，会作为提示词的一部分告诉视觉模型这张图大概画的是什么
        #   page_no    页码，最终要并进知识块的页码里，保证溯源不丢
        #   section_title 所在章节标题，给视觉模型补充上下文
        images.append(
            {
                "image_path": full,
                "image_name": Path(img_path).name,
                "caption": (block.get("caption") or block.get("text") or "").strip(),
                "page_no": int(block.get("page_no", 0) or 0),
                "section_title": (block.get("section_title") or "").strip(),
            }
        )
    return images


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def _describe_all(
    images: List[Dict[str, Any]],
    cache: Dict[str, Any],
    *,
    force: bool,
) -> Dict[str, Any]:
    """逐张调用视觉模型。单张失败不中断整体流程。"""
    from backend.llm_client import get_vision_llm

    vision = get_vision_llm()
    # 视觉模型没配好就直接原样返回已有的缓存：这些图会保持"只有图注"的状态，
    # 也就是 V1 的行为 —— 而不是报错把整库构建崩掉。
    if not vision.available:
        logger.error(
            "视觉模型不可用：请检查 .env 中的 VISION_ENABLED 与 VISION_API_KEY"
        )
        return cache

    for item in images:
        name = item["image_name"]

        # 幂等：这张图上次已经解析成功过就跳过，不重复调 API（省钱也省时间）。
        # 加了 --force 就忽略这个判断，强制重解。
        if not force and cache.get(name, {}).get("status") == "ok":
            logger.info("已有描述，跳过 | %s", name)
            continue

        logger.info(
            "解析图片 | %s | 第 %s 页 | 图注=%s",
            name, item["page_no"], item["caption"][:30] or "(无)",
        )
        try:
            # 把图片连同图注、所属章节一起交给视觉模型：
            # 让它带着"这张图在讲什么"的上下文去描述，比只丢一张图准得多。
            desc = vision.describe_image(
                item["image_path"],
                caption=item["caption"],
                context=item["section_title"],
            )
        except Exception as exc:
            # 单张失败不得中断整库构建：退回「仅图注」即 V1 行为
            logger.warning("解析失败，该图退回仅图注 | %s | %s", name, exc)
            # 标成 failed 但不中断：分块阶段看到 failed 就只用图注，
            # 也就是退回 V1 的行为。一张图失败绝不该让整库重建失败。
            cache[name] = {
                "caption": item["caption"],
                "desc": "",
                "page_no": item["page_no"],
                "section_title": item["section_title"],
                "status": "failed",
            }
            continue

        # 模型返回空内容也算失败（status=failed），同样退回只用图注。
        desc = (desc or "").strip()
        status = "ok" if desc else "failed"
        # 缓存以图片文件名为键，分块阶段按 img_path 就能查到这里的结果。
        cache[name] = {
            "caption": item["caption"],
            "desc": desc,
            "page_no": item["page_no"],
            "section_title": item["section_title"],
            "status": status,
        }
        logger.info(
            "解析完成 | %s | 描述长度=%d | 状态=%s", name, len(desc), status
        )

    return cache


def process_one(
    parsed_dir: Path,
    stem: str,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """处理单份文档，返回成功解析的图片数"""
    images = _collect_images(parsed_dir, stem)
    # 这份文档里一张图都没有，直接返回。
    if not images:
        return 0

    # 产物文件：图片文件名 -> 描述。分块阶段会来读它。
    out_file = parsed_dir / f"{stem}.vision.json"

    # 干跑模式：只打印待办清单，绝不调用 API、也不写任何文件。
    # 用途是先看一眼"要花多少钱、要等多久"，再决定要不要真跑。
    if dry_run:
        print("\n【{}】待解析图片 {} 张".format(stem, len(images)))
        for item in images:
            print(
                "  第 {:>3} 页 | {} | 图注：{}".format(
                    item["page_no"],
                    item["image_name"][:16] + "…",
                    item["caption"][:44] or "(无图注)",
                )
            )
        return 0

    cache: Dict[str, Any] = {}
    # 读入上次的结果作为基础，本次只补那些还没解析过的图 —— 这就是"增量"。
    if out_file.exists() and not force:
        try:
            cache = json.loads(out_file.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            # 缓存文件坏了就当没有，重新解析，不影响整体流程。
            logger.warning("已有视觉产物读取失败，将重新解析 | %s | %s", out_file, exc)
            cache = {}

    cache = _describe_all(images, cache, force=force)

    # 把结果落盘。ensure_ascii=False 让中文描述按原文保存（方便人工翻看），
    # indent=2 让它带缩进可读 —— 答辩时可以直接打开这个文件展示图内语义。
    out_file.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 统计成功 / 失败张数，用于最后的汇总日志。
    ok = sum(1 for v in cache.values() if v.get("status") == "ok")
    failed = sum(1 for v in cache.values() if v.get("status") == "failed")
    logger.info(
        "视觉解析完成 | %s | 成功 %d / 失败 %d / 共 %d 张",
        stem, ok, failed, len(images),
    )
    return ok


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    # Windows 控制台默认是 GBK 编码，中文日志会变乱码，这里强制切成 UTF-8。
    # 个别环境不支持 reconfigure，失败就算了，不能因为它把整个脚本中断。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(
        description="离线视觉解析：把 PDF 图表的图内语义补进知识库"
    )
    parser.add_argument("--force", action="store_true", help="强制重新解析已有描述的图片")
    parser.add_argument("--dry-run", action="store_true", help="只列出待解析图片，不调用 API")
    args = parser.parse_args(argv)

    # 扫出 pdf_parse.py 的产物：一份文档对应一个 .parsed.json。
    parsed_dir = settings.parsed_path
    parse_files = sorted(parsed_dir.glob("*.parsed.json"))
    if not parse_files:
        logger.error("未找到解析产物，请先执行 scripts.pdf_parse | 目录=%s", parsed_dir)
        return 1

    logger.info("视觉解析开始 | 解析产物目录=%s | 文档数=%d", parsed_dir, len(parse_files))

    total = 0
    # 逐份文档处理，累加成功解析的图片张数。
    for f in parse_files:
        # 从 "xxx.parsed.json" 里剥出文档名 "xxx"，
        # 它同时也是该文档所有产物文件（含 .vision.json）的公共前缀。
        stem = f.name[: -len(".parsed.json")]
        total += process_one(parsed_dir, stem, force=args.force, dry_run=args.dry_run)

    if args.dry_run:
        print("\n（干跑模式：未调用视觉模型，未写入任何文件）")
    else:
        logger.info("全部完成 | 成功解析图片 %d 张", total)
    return 0


# 支持 python -m scripts.vision_caption 直接运行
if __name__ == "__main__":
    sys.exit(main())
