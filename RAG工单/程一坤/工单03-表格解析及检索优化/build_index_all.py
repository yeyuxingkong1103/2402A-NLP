# -*- coding: utf-8 -*-
"""
工单03：表格解析及检索优化 —— 双文档知识库构建
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
功能：在工单01/02基础上新增《招股说明书2.pdf》解析（文字+表格），
     并补充《招股说明书1.pdf》的表格数据，构建统一索引 zgs_all_v1。
运行：python build_index_all.py
"""
import sys
import os

# 当前脚本所在目录
HERE = os.path.dirname(os.path.abspath(__file__))
# 把上级目录的"00-公共模块"加入模块搜索路径
sys.path.insert(0, os.path.join(HERE, "..", "00-公共模块"))

# 两份招股说明书的 PDF 路径配置
from config import PDF_ZGS1, PDF_ZGS2
# 文字解析函数（工单01同款，含页眉清洗）
from pdf_parser import parse_pdf_text
# 表格解析函数：抽取 PDF 中的表格并线性化为文本块（工单03新增能力）
from table_parser import parse_pdf_tables
# 自研向量库
from vector_store import VectorStore

INDEX_NAME = "zgs_all_v1"  # 双文档统一索引名


def main():
    # 幂等检查：目标索引已存在则跳过构建（避免重复向量化耗时）
    store = VectorStore.load(INDEX_NAME)
    if store is not None:
        print(f"索引已存在: {INDEX_NAME}（{len(store.texts)}条），跳过构建")
        return

    # 复用工单01已向量化的招股说明书1文字分块，避免重复计算
    store = VectorStore.load("zgs1_v1")
    # 前置依赖校验：zgs1_v1 必须先由工单01 build_index.py 生成
    assert store is not None, "请先运行工单01 build_index.py"

    # 1. 招股说明书2 文字分块
    print(f"[1/3] 解析 {PDF_ZGS2} 文字 ...")
    # 逐页提取招股说明书2的文字（含页眉清洗，与工单01口径一致）
    pages2 = parse_pdf_text(PDF_ZGS2)
    # 函数内延迟导入 chunker（复用工单01的分块逻辑）
    from chunker import chunk_pages
    chunks2 = chunk_pages(pages2)
    print(f"      {len(pages2)} 页 → {len(chunks2)} 块")
    # 向量化并加入统一索引；source_name 标注来源文档，便于检索结果溯源
    store.add(chunks2, source_name="招股说明书2.pdf")

    # 2. 招股说明书2 表格
    print("[2/3] 提取招股说明书2表格 ...")
    # 抽取表格并转为文本块（表格数据无法被纯文字解析覆盖，必须单独提取）
    tables2 = parse_pdf_tables(PDF_ZGS2)
    print(f"      {len(tables2)} 个有效表格")
    store.add(tables2, source_name="招股说明书2.pdf(表格)")

    # 3. 招股说明书1 表格（军用收入占比等答案在表格中，工单02遗留问题的解法）
    print("[3/3] 提取招股说明书1表格 ...")
    # 工单02中"军用收入占比"类问题答错，根因是数字表格向量化质量差；
    # 工单03将表格单独线性化入库来补齐这块知识盲区
    tables1 = parse_pdf_tables(PDF_ZGS1)
    print(f"      {len(tables1)} 个有效表格")
    store.add(tables1, source_name="招股说明书1.pdf(表格)")

    # 把合并后的完整索引（招1文字+招2文字+招2表格+招1表格）持久化到磁盘
    store.save(INDEX_NAME)


if __name__ == "__main__":
    main()
