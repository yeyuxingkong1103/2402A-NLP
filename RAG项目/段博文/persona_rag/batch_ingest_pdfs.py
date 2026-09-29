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

本模块在系统中的位置：
    这是一条「离线批处理」入口，不属于 FastAPI 服务：服务启动前先跑它把语料灌进 Milvus，
    之后 main.py 的检索接口才有数据可查。
    上游：人工在命令行执行；下游：doc_parser（解析）、text_splitter（切分）、
          db_milvus（向量化 + 写库）、retriever.refresh_bm25（刷新关键词索引）。

关键设计取舍：
    1. 角色与集合的对应关系不写死，统一走 config.get_collection_for_role，
       避免和在线服务用的集合名对不上。
    2. PDF 与 JSONL 两种数据源共用同一套「写入后就刷新 BM25」的约定，
       否则刚灌进去的数据只能被向量召回、搜不到关键词。
    3. 各步骤失败只记日志并返回 0，不中断整个批处理：批量任务里一个坏文件不该毁掉剩下的工作。
    4. 重依赖（解析器、向量库）都放在函数内部延迟导入：没装某个解析库时脚本仍能被导入查看。
"""

import os  # 文件路径操作
import sys  # 系统接口
import time  # 计时
import argparse  # 命令行参数

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 插入当前目录：保证直接运行脚本时能 import 到同目录的模块

from logger import get_logger  # 日志模块

logger = get_logger(__name__)  # 本模块 logger


# ==================== 配置区（路径都写死在这里） ====================

# ---------- 各角色的默认 PDF 文件列表 ----------
# 结构：角色 key -> PDF 绝对路径列表；没有素材的角色留空列表即可（下面会跳过并提示）
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
# JSONL 每行一条预处理好的样本，适合放已经清洗/结构化过的大批量数据，比 PDF 解析更省时间
ROLE_JSONL_FILES = {
    "lawyer": None,
    "psychologist": r"G:\知识库pdf文件\心理\processed_psych.jsonl",
    "virtual_friend": None,
}


# ==================== 切分参数 ====================
# 这两个值只用于打印提示信息（真正的切分参数由 text_splitter/config 决定），
# 保留在这里是为了让日志里能直观看到当前使用的切分粒度。
CHUNK_SIZE = 500  # 每块目标字数
CHUNK_OVERLAP = 80  # 块间重叠


def process_single_pdf(pdf_path: str, collection_name: str) -> int:
    """
    处理单个 PDF：解析 -> 切分 -> 入库 -> 刷新 BM25
    返回：该文件入库的 chunk 数

    参数：
        pdf_path         PDF 绝对路径（配置文件里写死的路径）。
        collection_name  目标 Milvus 集合名，由角色 key 经 config 映射得到。
    返回：成功时返回切分出的块数；文件不存在、解析失败、文本为空、切分为空时都返回 0。
    异常与降级：解析异常被就地捕获并记 error 日志，返回 0 让批处理继续跑下一个文件。
    注意：入库成功后立刻刷新该集合的 BM25（而不是全部跑完再刷），
          这样中途中断时已经入库的部分也是可检索的。
    """
    from doc_parser import extract_text  # 延迟导入：解析依赖较重，且只在真正处理文件时才需要
    from text_splitter import split_text
    from db_milvus import ingest_documents
    from retriever import refresh_bm25

    filename = os.path.basename(pdf_path)  # 只取文件名作为 source，方便之后按来源管理
    logger.info(f"[开始] {filename} -> 集合 {collection_name}")
    start_time = time.time()

    if not os.path.exists(pdf_path):
        logger.error(f"文件不存在：{pdf_path}")
        return 0

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()  # 一次性读入内存：解析库需要完整字节流，PDF 一般几十 MB 可接受

    logger.info(f"  解析 PDF 中（{len(file_bytes) / 1024 / 1024:.1f}MB）...")
    try:
        raw_text, is_md = extract_text(file_bytes, filename)  # is_md 表示是否按 Markdown 切
    except Exception as e:
        logger.error(f"  解析失败：{e}")
        return 0  # 单个文件失败不影响整批

    if not raw_text.strip():
        logger.warning(f"  PDF 文本为空（可能是扫描件，需要 OCR）")
        return 0  # 扫描版 PDF 抽不到文字，直接跳过而不是写空内容

    logger.info(f"  解析完成：{len(raw_text)} 字符")
    logger.info(f"  切分中（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）...")
    docs = split_text(  # 统一用 paragraph 策略：法律文本按条文/段落切最自然
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
    total = ingest_documents(docs, collection_name=collection_name)  # 向量化 + 批量写入（内部按批处理）

    refresh_bm25(collection_name)  # 写完就刷关键词索引，保证混合检索的两路都能命中新数据

    elapsed = time.time() - start_time
    logger.info(f"[完成] {filename}：{len(docs)} 块，耗时 {elapsed:.1f}s，集合 {collection_name} 总计 {total} 条")
    return len(docs)


def ingest_pdfs(pdf_files: list, collection_name: str) -> int:
    """批量入 PDF，返回总 chunk 数

    参数：
        pdf_files        某个角色的 PDF 路径列表（来自 ROLE_PDF_FILES）。
        collection_name  目标集合名。
    返回：本批所有 PDF 累计入库的块数；列表为空时返回 0。
    说明：先打印一遍文件的「存在/不存在 + 大小」清单，便于在长时间跑批前就发现路径写错，
          不用等到解析报错才知道。
    """
    print(f"\n{'=' * 60}")
    print(f"[PDF 部分] 角色 -> 集合 {collection_name}")
    print(f"{'=' * 60}")

    if not pdf_files:
        print("  该角色没有配置 PDF，跳过")
        return 0

    print(f"待处理文件：{len(pdf_files)} 个")
    for f in pdf_files:
        exists = "存在" if os.path.exists(f) else "不存在"
        size = os.path.getsize(f) / 1024 / 1024 if os.path.exists(f) else 0  # 先判断存在再取大小，避免抛异常
        print(f"  [{exists}] {os.path.basename(f)}（{size:.1f}MB）")

    total_chunks = 0
    for i, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n[{i}/{len(pdf_files)}] 处理中...")  # 用序号做进度提示：PDF 解析很慢，需要让用户知道进展
        chunks = process_single_pdf(pdf_path, collection_name)
        total_chunks += chunks
    return total_chunks


def ingest_jsonl_file(jsonl_path: str, collection_name: str) -> int:
    """读已处理好的 JSONL 文件并入库（process 函数内部走批处理、进度条在 ingest_jsonl 里）

    参数：
        jsonl_path       JSONL 文件路径，每行一条 JSON 样本。
        collection_name  目标集合名。
    返回：入库完成后该集合的语料总条数；文件不存在时返回 0。
    说明：入库实现放在 db_milvus.ingest_jsonl 里，本函数只负责计时、打印和刷新 BM25，
          好处是「逐条流式读取 + 分批向量化」的细节与在线服务共用同一份代码。
    注意：JSONL 可能很大，首次读取会先把整份数据读进内存，所以对内存有一定要求。
    """
    from db_milvus import ingest_jsonl
    from retriever import refresh_bm25

    if not os.path.exists(jsonl_path):
        print(f"[错误] 文件不存在：{jsonl_path}")
        return 0

    start = time.time()
    print(f"JSONL 入库中（{jsonl_path}）...")
    print(f"  首次读取会先加载整个 jsonl 到内存，随后按批向量化写入：")
    total = ingest_jsonl(jsonl_path, collection_name=collection_name)  # 内部按批向量化并写入 Milvus
    refresh_bm25(collection_name)  # 入完刷新 BM25：这一步耗时随语料量增长，属于预期成本
    elapsed = time.time() - start
    print(f"[完成] 集合 {collection_name} 当前语料 {total} 条，耗时 {elapsed:.1f}s")
    return total


def main():
    """命令行主流程：解析参数 -> 逐角色跑 PDF 与 JSONL -> 打印总结

    命令行参数：
        --role      只处理指定角色（如 lawyer）；省略则跑所有「配置了 PDF 或有默认 JSONL」的角色。
        --jsonl     额外指定一个 JSONL 路径并叠加入库；优先级高于角色默认的 JSONL。
        --no-jsonl  跳过 JSONL 部分，只入 PDF。
    返回：无返回值；所有角色处理完后打印总块数和总耗时。
    异常与降级：单个文件/单个步骤失败不中断整体，只是该部分块数记为 0。
    """
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
        # 这样「没素材」的角色会被自动排除，避免空跑浪费时间
        roles = [r for r in ROLE_PDF_FILES if ROLE_PDF_FILES.get(r) or ROLE_JSONL_FILES.get(r)]
        if not roles:
            roles = list(ROLE_PDF_FILES.keys())  # 兜底：都没配置时仍然把已知角色列出来

    print("=" * 60)
    print("知识库一键全量入库")
    print(f"本次角色：{', '.join(roles)}")
    print("=" * 60)

    from config import get_collection_for_role  # 延迟导入：放到用到处，避免影响 --help 的启动速度

    total_start = time.time()
    grand_total = 0

    for role in roles:
        collection_name = get_collection_for_role(role)  # 每个角色独立集合：检索时按角色路由，互不干扰
        print(f"\n{'#' * 60}")
        print(f"# 角色：{role} -> 集合：{collection_name}")
        print(f"{'#' * 60}")

        # ----- (1) PDF 入库 -----
        pdf_files = ROLE_PDF_FILES.get(role, [])  # 没配置就取空列表，交给 ingest_pdfs 打印「跳过」
        grand_total += ingest_pdfs(pdf_files, collection_name)

        # ----- (2) JSONL 入库（除非显式 --no-jsonl） -----
        if args.no_jsonl:
            print("\n[JSONL] 已跳过（--no-jsonl）")
            continue  # continue 而不是 break：PDF 那部分已经跑完了，直接换下一个角色

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