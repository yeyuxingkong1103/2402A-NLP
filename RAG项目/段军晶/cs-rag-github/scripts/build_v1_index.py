# -*- coding: utf-8 -*-
"""
重建 V1 原始索引（一次性评测工具）

用途：
    M3 基线是在 V2 之前的索引上测得的。V3 修正了评测页码匹配口径后，
    要让基线与 V1/V2/V3 在同一口径下可比，就必须能重新跑一次「V1 在旧索引上」
    的检索 —— 本脚本负责把旧索引还原出来。

还原方法：
    V2 给 6 个图片块的 content 追加了 "\n[视觉解析] 描述"。
    剥离该标记及其后内容即可精确还原旧 content；其余块内容未变。

⚠️ 本脚本只写 Milvus，**不触碰 MySQL**：
    page_no / page_nums 在新旧索引间完全一致（V2 只改 content），
    因此 MySQL 的元数据可原样复用。若误写 MySQL，会把知识库内容改回旧版。

用法：
    python -m scripts.build_v1_index              # 建临时集合并灌入
    python -m scripts.build_v1_index --drop       # 用完删除临时集合

安全护栏：
    --collection 采用**白名单**而非黑名单：只允许 `cs_kb_v1_` 前缀的临时集合名
    （本脚本的设计用途就是建 V1 索引临时集合）。以下情况一律报错并以非 0
    退出码退出，**不执行任何 drop 或 create**：
      · 空名 —— 因为 milvus_client 内部有 `name = name or settings.milvus_collection`
        的回落逻辑，空名会被悄悄解释成主集合
      · 等于项目主集合（settings.milvus_collection / cs_kb_chunks）
      · 不以 `cs_kb_v1_` 开头 —— 尤其要挡住本 Milvus 实例上其它项目的集合
        （djj3_2 等），误删会造成第三方数据不可恢复的丢失
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.db import milvus_client
from backend.embedder import get_embedder
from backend.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# 与 chunk_split._VISION_MARK 保持一致
# （该一致性由 tests/test_build_v1_index.py 强制，不是一句无人校验的注释）
VISION_MARK = "\n[视觉解析] "

# 允许操作的集合名前缀白名单。
# 本脚本的唯一用途是造「V1 原始索引」临时集合，故只放行这一族名字。
# 用白名单而非黑名单，是为了让「没预料到的名字」默认被拒绝而非默认放行。
ALLOWED_COLLECTION_PREFIX = "cs_kb_v1_"


def strip_vision(content: str) -> str:
    """
    剥离视觉描述，还原 V1 时期的 content。

    找不到标记时原样返回（该块在 V2 中未变化）。
    """
    idx = content.find(VISION_MARK)
    return content[:idx] if idx >= 0 else content


def collect_v1_chunks() -> List[Dict[str, Any]]:
    """读取当前 chunks，还原出 V1 时期的块列表"""
    chunks: List[Dict[str, Any]] = []
    files = sorted(settings.parsed_path.glob("*.chunks.json"))
    if not files:
        raise FileNotFoundError(f"未找到分块产物：{settings.parsed_path}")

    stripped = 0
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for c in data.get("chunks", []):
            old = strip_vision(c["content"])
            if old != c["content"]:
                stripped += 1
            chunks.append({**c, "content": old})

    logger.info(
        "还原 V1 chunks 完成 | 共 %d 块 | 其中 %d 块剥离了视觉描述",
        len(chunks), stripped,
    )
    return chunks


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="重建 V1 原始索引（评测用临时集合）")
    parser.add_argument("--collection", default="cs_kb_v1_orig",
                        help="临时集合名（默认 cs_kb_v1_orig）")
    parser.add_argument("--drop", action="store_true",
                        help="删除临时集合后退出")
    args = parser.parse_args(argv)

    # ★ 安全护栏：白名单校验集合名，拒绝一切非本脚本用途的目标 ★
    #
    # 为什么需要这道护栏：
    #   本脚本有两条路径都会**删除集合并重建** ——
    #     ① 构建分支的 create_collection(..., drop_existing=True)
    #     ② --drop 清理分支的 drop_collection(...)
    #   而 --collection 是用户可自由传入的。名字一旦指到不该碰的集合，
    #   后果都是**不可恢复的数据丢失**：
    #     · 主集合 cs_kb_chunks 被删 → 知识库 127 条向量连同 V2 的视觉描述
    #       与稀疏向量一起丢失，需重跑视觉 API 才能恢复
    #     · 本实例上其它项目的集合（djj3_2 等）被删 → 第三方数据丢失，
    #       且本脚本完全没有能力重建它
    #
    #   ★ 空名必须显式拒绝 ★
    #   milvus_client 内部普遍使用 `name = name or settings.milvus_collection`
    #   的回落写法（见 milvus_client.py:113 / :167），因此空名会被**悄悄
    #   解释成主集合**。若护栏只比较字面串，`--collection ""` 就能完整绕过
    #   护栏，而这恰恰是它要防的事。`--collection "$COL"` 在 COL 未设置时
    #   正是这种情形 —— 而本项目下一步就要用环境变量切换集合，该写法很自然。
    #
    #   因此这里用**白名单**而非黑名单：只放行本脚本设计用途内的
    #   cs_kb_v1_* 名字。业务上「不该碰的集合」是无穷多的，只有白名单才能
    #   让未预料到的名字默认被拒绝，而不是默认放行。
    #   校验位置在任何 drop / create **之前**，且早于任何 Milvus 交互
    #   （连 collection_exists 都不该先发生，否则 Milvus 不可达时用户看到的
    #   会是误导性的网络错误，而非真正的原因）。
    if not args.collection:
        logger.error(
            "拒绝操作：集合名为空。milvus_client 内部会把空名回落成主集合 %s，"
            "从而删掉知识库。请显式指定临时集合名（默认 cs_kb_v1_orig）",
            settings.milvus_collection,
        )
        return 1
    if args.collection == settings.milvus_collection:
        logger.error(
            "拒绝操作主集合 %s：本脚本会 drop 重建集合，误操作会销毁知识库。"
            "请使用临时集合名（默认 cs_kb_v1_orig）",
            args.collection,
        )
        return 1
    if not args.collection.startswith(ALLOWED_COLLECTION_PREFIX):
        logger.error(
            "拒绝操作集合 %s：本脚本只允许操作 %s* 前缀的临时集合"
            "（默认 cs_kb_v1_orig）。该 Milvus 实例上共存着其它项目的集合，"
            "误删会造成不可恢复的数据丢失。",
            args.collection,
            ALLOWED_COLLECTION_PREFIX,
        )
        return 1

    if args.drop:
        if milvus_client.collection_exists(args.collection):
            milvus_client.drop_collection(args.collection)
            logger.info("已删除临时集合：%s", args.collection)
        else:
            logger.info("临时集合不存在，无需删除：%s", args.collection)
        return 0

    chunks = collect_v1_chunks()

    # 空块列表必须报错，不能「成功」。
    # collect_v1_chunks 用 data.get("chunks", []) 容忍键缺失，只校验了文件存在；
    # 若分块产物的 schema 漂移导致 chunks 键整体缺失，流程会走成
    # 「0 个块 → 编码 0 条 → 长度校验 0==0 通过 → insert_vectors 在 total==0 时
    # 直接返回 0 → 日志打印『构建完成 | 向量 0 条』」。集合被置空而脚本报成功，
    # 下游评测会**静默少召回**，这是最难排查的一类故障。
    if not chunks:
        raise RuntimeError(
            f"未从分块产物中读到任何块，拒绝继续：{settings.parsed_path}。"
            "请检查 *.chunks.json 的结构是否仍含 chunks 键。"
        )

    texts = [c["content"] for c in chunks]
    logger.info("正在向量化 %d 个块（CPU 推理，请稍候）...", len(texts))
    vectors = get_embedder().encode(texts, show_progress=True)
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"向量数量与 chunk 数量不一致：{len(vectors)} != {len(chunks)}"
        )

    # ★ 建集合必须放在向量化**成功之后**（顺序即安全）★
    # create_collection(..., drop_existing=True) 会先删掉同名旧集合。
    # 若放在前面，则 embedding 失败（OOM、模型缺失）或上面的长度校验抛错时，
    # 旧集合早已被删除而新数据没写进去 —— 留下一个**被清空的集合**，
    # 而脚本以「失败」退出，用户容易误以为集合仍是完好的。
    # 先算后删，保证任何失败路径都不会破坏已有集合。
    #
    # 集合按 V1 时期建：不启用稀疏字段，以精确模拟旧索引。
    milvus_client.create_collection(args.collection, drop_existing=True)

    written = milvus_client.insert_vectors(
        chunk_ids=[c["chunk_id"] for c in chunks],
        doc_ids=[c["doc_id"] for c in chunks],
        page_nos=[c["page_no"] for c in chunks],
        contents=texts,
        embeddings=vectors,
        name=args.collection,
    )
    logger.info("V1 原始索引构建完成 | 集合=%s | 向量 %d 条", args.collection, written)
    logger.info(
        "接下来用环境变量指向该集合重跑基线：\n"
        "    MILVUS_COLLECTION=%s python -m scripts.ragas_eval --pipeline v1 --tag baseline",
        args.collection,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
