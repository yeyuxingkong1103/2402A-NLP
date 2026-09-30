# -*- coding: utf-8 -*-
"""
离线数据管线脚本包

本包内为「知识库构建」阶段的离线处理脚本，与在线服务解耦：
    pdf_parse.py       —— MinerU 版面解析，提取文本与页码元数据
    chunk_split.py     —— LangChain 分块，保证页码元数据存活
    embedding_store.py —— BGE-M3 向量化，写入 Milvus 与 MySQL
    ragas_eval.py      —— RAGAS 评测

这些脚本既可单独以命令行方式运行，也可由后端接口按其函数粒度调用。
"""
