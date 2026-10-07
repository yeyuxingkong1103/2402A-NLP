# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-LightRAG优化任务
模块：LightRAG 配置
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LightRAGConfig:
    LLM_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    LLM_BASE_URL: str = "https://api.deepseek.com"
    LLM_MODEL: str = "deepseek-chat"
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DIM: int = 512
    WORKING_DIR: str = "./lightrag_data"
    CHUNK_TOKEN_SIZE: int = 800
    CHUNK_OVERLAP_TOKEN_SIZE: int = 100
    MAX_GLEANING: int = 1
    TOP_K: int = 10


if __name__ == "__main__":
    cfg = LightRAGConfig()
    print("LightRAG 配置：")
    print(f"  LLM: {cfg.LLM_BASE_URL} / {cfg.LLM_MODEL}")
    print(f"  LLM Key: {'已配置' if cfg.LLM_API_KEY else '未配置'}")
    print(f"  Embedding: {cfg.EMBEDDING_MODEL}")
    print(f"  Working Dir: {cfg.WORKING_DIR}")
