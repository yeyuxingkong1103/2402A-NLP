# -*- coding: utf-8 -*-
"""
工单07：ccf_competition 金融年报知识库构建
工单编号：人工智能NLP-RAG-功能测试及评估
功能：解析附件 ccf_competition/txt 下 9 份银行/保险/证券年报（已预提取文本），
     分块向量化构建统一索引 ccf_v1。
运行：python build_ccf_index.py
"""
import sys
import os
import glob
import re

# 当前脚本目录与公共模块路径，保证能 import 00-公共模块
HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "00-公共模块")
sys.path.insert(0, COMMON)

from config import CCF_TXT_DIR  # ccf_competition/txt 年报文本目录配置
from chunker import chunk_text  # 通用滑动窗口分块器（工单01/03）
from vector_store import VectorStore  # 自研 numpy 向量库

INDEX_NAME = "ccf_v1"  # 9份年报统一索引名（约2万块）


def file_display_name(path):
    """从文件名提取展示名：公司+年份"""
    base = os.path.basename(path)  # 只取文件名部分
    # 正则抓取文件名中的4位年份，抓不到则显示 "?"
    m = re.search(r"(\d{4})年", base)
    year = m.group(1) if m else "?"
    # 公司名在第二段（__公司__代码）
    # 文件名形如 "xxx__公司名__代码.txt"，按双下划线切分取第二段
    parts = base.split("__")
    company = parts[1] if len(parts) > 2 else base[:12]  # 格式不符则截前12字符兜底
    return f"{company}{year}年报"


def main():
    # 幂等保护：索引已存在直接跳过（2万块向量化的构建成本高）
    if VectorStore.load(INDEX_NAME) is not None:
        print("ccf_v1 索引已存在，跳过")
        return

    # 按文件名排序枚举全部 txt 年报
    files = sorted(glob.glob(os.path.join(CCF_TXT_DIR, "*.txt")))
    print(f"发现 {len(files)} 份年报文本")
    store = VectorStore()
    for f in files:
        # 先算展示名（公司+年份），入库时作为 source
        name = file_display_name(f)
        # 读整份年报文本；errors="ignore" 容忍个别非法字节避免读文件报错
        raw = open(f, "r", encoding="utf-8", errors="ignore").read()
        # 年报文本噪声较大：压缩空白
        # 把全角/半角空格、制表符连续串压成单个空格，降低对分块质量的影响
        raw = re.sub(r"[ \t\u3000]+", " ", raw)
        chunks = []  # 该文件的分块列表
        # 450字/块、60字重叠：块太大检索粒度粗，太小则上下文断裂；重叠防止句子被切断
        for c in chunk_text(raw, chunk_size=450, overlap=60):
            # txt 无页码信息，page 统一记0（检索结果展示时以 source 为主）
            chunks.append({"text": c, "page": 0})
        # 打印该文件的处理统计：原文字数 → 分块数
        print(f"  {name}: {len(raw)}字 → {len(chunks)}块")
        # 以公司+年份为来源名，分块向量化加入统一索引
        store.add(chunks, source_name=name)
    # 全部文件入库后持久化索引
    store.save(INDEX_NAME)


if __name__ == "__main__":
    main()
