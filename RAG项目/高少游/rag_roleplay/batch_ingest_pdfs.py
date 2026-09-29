# -*- coding: utf-8 -*-
"""
知识库一键全量入库脚本

点击运行（无需任何参数），自动按顺序完成：
    1. 法律 PDF  -> 集合 rag_legal
    2. 心理 JSONL -> 集合 rag_psychology

特点：
    - 零参数：直接运行即可，不用手动切角色
    - 带进度条：JSONL 大批量入库时实时显示批次进度
    - 幂等去重：同一文件重复跑不会翻倍
    - 两段都跑：PDF 和 JSONL 一次全部写入，不互斥

用法：
    python batch_ingest_pdfs.py             # 一键全量
    python batch_ingest_pdfs.py --no-jsonl  # 只入 PDF
"""

import os  # 文件路径操作
import sys  # 系统接口
import time  # 计时
import argparse  # 命令行参数

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 插入当前目录

from logger import get_logger  # 日志模块

logger = get_logger(__name__)  # 本模块 logger


# ==================== 配置区（路径都写死在这里） ====================

# ---------- 各角色的默认 PDF 文件列表 ----------
ROLE_PDF_FILES = {
    "lawyer": [
        "G:\\知识库pdf文件\\法律\\中华人民共和国民法典_20200528.pdf",
        "G:\\知识库pdf文件\\法律\\中华人民共和国刑法_20201226.pdf",
        "G:\\知识库pdf文件\\法律\\中华人民共和国劳动合同法_20121228.pdf",
        "G:\\知识库pdf文件\\法律\\中华人民共和国民事诉讼法_20230901.pdf",
        "G:\\知识库pdf文件\\法律\\中华人民共和国治安管理处罚法_20250627.pdf",
    ],
    "psychologist": [
        # 心理学相关 PDF，有则填入
    ],
    "virtual_friend": [
        # 聊天素材类文档，有则填入
    ],
}

# ---------- 固定要入库的已处理 JSONL 数据集 ----------
# 角色 -> jsonl 路径；key 不存在或值为 None 表示该角色没有固定 jsonl
ROLE_JSONL_FILES = {
    "lawyer": None,
    "psychologist": r"G:\知识库pdf文件\心理\processed_psych.jsonl",
    "virtual_friend": r"G:\知识库pdf文件\虚拟朋友\cecily_persona.jsonl",
}


# ==================== 切分参数 ====================
CHUNK_SIZE = 500  # 每块目标字数
CHUNK_OVERLAP = 80  # 块间重叠


def process_single_pdf(pdf_path: str, collection_name: str) -> int:
    """
    处理单个 PDF：解析 -> 切分 -> 入库 -> 刷新 BM25
    返回：该文件入库的 chunk 数
    """
    from doc_parser import extract_text
    from text_splitter import split_text
    from db_milvus import ingest_documents
    from retriever import refresh_bm25

    filename = os.path.basename(pdf_path)
    logger.info(f"[开始] {filename} -> 集合 {collection_name}")
    start_time = time.time()

    if not os.path.exists(pdf_path):
        logger.error(f"文件不存在：{pdf_path}")
        return 0

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    logger.info(f"  解析 PDF 中（{len(file_bytes) / 1024 / 1024:.1f}MB）...")
    try:
        raw_text, is_md = extract_text(file_bytes, filename)
    except Exception as e:
        logger.error(f"  解析失败：{e}")
        return 0

    if not raw_text.strip():
        logger.warning(f"  PDF 文本为空（可能是扫描件，需要 OCR）")
        return 0

    logger.info(f"  解析完成：{len(raw_text)} 字符")
    logger.info(f"  切分中（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）...")
    docs = split_text(
        raw_text=raw_text,
        source=filename,
        strategy="paragraph",
        is_markdown=is_md,
    )

    if not docs:
        logger.warning(f"  切分为空，跳过入库")
        return 0

    logger.info(f"  切分完成：{len(docs)} 块")
    logger.info(f"  入库中（bge-m3 向量化 + 写入 Milvus 集合 {collection_name}）...")
    total = ingest_documents(docs, collection_name=collection_name)

    refresh_bm25(collection_name)

    elapsed = time.time() - start_time
    logger.info(f"[完成] {filename}：{len(docs)} 块，耗时 {elapsed:.1f}s，集合 {collection_name} 总计 {total} 条")
    return len(docs)


def ingest_pdfs(pdf_files: list, collection_name: str) -> int:
    """批量入 PDF，返回总 chunk 数"""
    print(f"\n{'=' * 60}")
    print(f"[PDF 部分] 角色 -> 集合 {collection_name}")
    print(f"{'=' * 60}")

    if not pdf_files:
        print("  该角色没有配置 PDF，跳过")
        return 0

    print(f"待处理文件：{len(pdf_files)} 个")
    for f in pdf_files:
        exists = "存在" if os.path.exists(f) else "不存在"
        size = os.path.getsize(f) / 1024 / 1024 if os.path.exists(f) else 0
        print(f"  [{exists}] {os.path.basename(f)}（{size:.1f}MB）")

    total_chunks = 0
    for i, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n[{i}/{len(pdf_files)}] 处理中...")
        chunks = process_single_pdf(pdf_path, collection_name)
        total_chunks += chunks
    return total_chunks


def ingest_jsonl_file(jsonl_path: str, collection_name: str) -> int:
    """读已处理好的 JSONL 文件并入库（process 函数内部走批处理、进度条在 ingest_jsonl 里）"""
    from db_milvus import ingest_jsonl
    from retriever import refresh_bm25

    if not os.path.exists(jsonl_path):
        print(f"[错误] 文件不存在：{jsonl_path}")
        return 0

    start = time.time()
    print(f"JSONL 入库中（{jsonl_path}）...")
    print(f"  首次读取会先加载整个 jsonl 到内存，随后按批向量化写入：")
    total = ingest_jsonl(jsonl_path, collection_name=collection_name)
    refresh_bm25(collection_name)  # 入完刷新 BM25
    elapsed = time.time() - start
    print(f"[完成] 集合 {collection_name} 当前语料 {total} 条，耗时 {elapsed:.1f}s")
    return total


def main():
    parser = argparse.ArgumentParser(description="知识库一键全量入库（PDF + JSONL 自动全跑）")
    parser.add_argument("--role", default=None,
                        help="只需入某个角色时指定（如 lawyer/psychologist），省略则自动跑所有配置的角色")
    parser.add_argument("--jsonl", default=None,
                        help="额外指定要入库的 JSONL 路径（叠加运行）")
    parser.add_argument("--no-jsonl", action="store_true",
                        help="跳过 JSONL 部分，只入 PDF")
    args = parser.parse_args()

    # ---- 确定本次要跑的角色 ----
    if args.role:
        roles = [args.role]  # 只跑指定角色
    else:
        # 一键全量：把所有配置了 PDF 或 jsonl 的角色都跑
        roles = [r for r in ROLE_PDF_FILES if ROLE_PDF_FILES.get(r) or ROLE_JSONL_FILES.get(r)]
        if not roles:
            roles = list(ROLE_PDF_FILES.keys())

    print("=" * 60)
    print("知识库一键全量入库")
    print(f"本次角色：{', '.join(roles)}")
    print("=" * 60)

    from config import get_collection_for_role

    total_start = time.time()
    grand_total = 0

    for role in roles:
        collection_name = get_collection_for_role(role)  # 每个角色独立集合
        print(f"\n{'#' * 60}")
        print(f"# 角色：{role} -> 集合：{collection_name}")
        print(f"{'#' * 60}")

        # ----- (1) PDF 入库 -----
        pdf_files = ROLE_PDF_FILES.get(role, [])
        grand_total += ingest_pdfs(pdf_files, collection_name)

        # ----- (2) JSONL 入库（除非显式 --no-jsonl） -----
        if args.no_jsonl:
            print("\n[JSONL] 已跳过（--no-jsonl）")
            continue

        jsonl_path = args.jsonl  # CLI 显式指定优先
        if not jsonl_path:
            jsonl_path = ROLE_JSONL_FILES.get(role)  # 否则用角色默认
        if jsonl_path:
            print(f"\n[JSONL 部分] 入库：{jsonl_path}")
            grand_total += ingest_jsonl_file(jsonl_path, collection_name)
        else:
            print("\n[JSONL 部分] 该角色没有配置 jsonl，跳过")

    # ---- 总结 ----
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("全部入库完成！")
    print("=" * 60)
    print(f"  角色     ：{', '.join(roles)}")
    print(f"  入库块数 ：{grand_total} 块")
    print(f"  总耗时   ：{total_elapsed:.1f} 秒（{total_elapsed / 60:.1f} 分钟）")
    print(f"\n现在可以启动服务测试检索：")
    print(f"  H:\\an\\envs\\langchain2\\python.exe main.py")
    print(f"  打开 http://localhost:8000")
    print("=" * 60)


if __name__ == "__main__":
    main()