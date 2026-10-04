# -*- coding: utf-8 -*-
"""
索引构建脚本
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
功能：解析《招股说明书1.pdf》→ 分块 → 向量化 → 持久化到 index_store/
运行：python build_index.py
"""
import sys
import os
# 把上级目录的"00-公共模块"加入模块搜索路径，才能 import 公共模块（config/pdf_parser 等）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00-公共模块"))

# 导入 PDF 路径配置（config 中定义 PDF_ZGS1 指向《招股说明书1.pdf》的绝对路径）
from config import PDF_ZGS1
# 导入 PDF 文字解析函数：逐页提取文本，并做页眉页脚清洗
from pdf_parser import parse_pdf_text
# 导入分块函数：把逐页文本切成带页码元信息的语义分块
from chunker import chunk_pages
# 导入自研向量库：numpy 矩阵 + JSON 元数据持久化，底层调用 Ollama bge-m3 做向量化
from vector_store import VectorStore

INDEX_NAME = "zgs1_v1"  # 工单01 基础索引


def main():
    # 前置校验：PDF 文件不存在就直接报错退出，避免后续解析阶段报出难以定位的异常
    if not os.path.exists(PDF_ZGS1):
        print(f"[错误] 未找到 PDF: {PDF_ZGS1}")
        return

    # 1. 解析 PDF 文字
    print(f"[1/3] 解析 PDF: {PDF_ZGS1}")
    # 逐页提取文字；clean_headers 默认开启，清洗掉页眉公司名/页码行（工单01的降噪优化）
    pages = parse_pdf_text(PDF_ZGS1)
    print(f"      共 {len(pages)} 个有效页面")

    # 2. 文本分块
    print("[2/3] 文本分块 ...")
    # 按段落/长度把页面文本切成检索用的分块，每块携带来源页码便于答案溯源
    chunks = chunk_pages(pages)
    print(f"      共 {len(chunks)} 个分块")

    # 3. 向量化并持久化
    print("[3/3] 向量化并保存索引 ...")
    # 实例化向量库（内部连接 Ollama 的 bge-m3 embedding 服务）
    store = VectorStore()
    # 对全部分块做向量化并加入库中；source_name 记录来源文档名，供答案引用时展示
    store.add(chunks, source_name="招股说明书1.pdf")
    # 把向量矩阵（.npy）与分块元数据（.json）持久化到 index_store/，文件名即索引名
    store.save(INDEX_NAME)


if __name__ == "__main__":
    main()
