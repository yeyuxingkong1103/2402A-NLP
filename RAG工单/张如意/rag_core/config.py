# -*- coding: utf-8 -*-
"""
RAG 项目全局配置
工单编号：人工智能NLP-RAG（供 01~13 全部工单共用）

统一管理：路径、模型、超参、以及各工单的评测问题集。
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# 路径配置
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # 工单作业/
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"          # PDF 解析缓存
PARSED_DIR = DATA_DIR / "parsed"        # 解析后的结构化结果(json)
INDEX_DIR = DATA_DIR / "index"          # 向量库 / BM25 索引
IMAGE_DIR = DATA_DIR / "images"         # 从 PDF 抽出的图片
GRAPH_DIR = DATA_DIR / "graph"          # 知识图谱

for _d in (CACHE_DIR, PARSED_DIR, INDEX_DIR, IMAGE_DIR, GRAPH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 原始素材目录（工单附件）
SRC_DIR = Path(r"D:\工单\RAG 工单\RAG 工单")
SRC_ATTACH = SRC_DIR / "附件"

PDF_PROSPECTUS_1 = SRC_ATTACH / "招股说明书1.pdf"                 # 武汉兴图新科（军工）
PDF_PROSPECTUS_1_NW = SRC_ATTACH / "招股说明书1-无水印.pdf"        # 无水印版，图像解析用
PDF_PROSPECTUS_2 = SRC_ATTACH / "招股说明书2.pdf"                 # 武汉力源信息
PDF_SAMPLE_QUESTIONS = SRC_ATTACH / "sample_questions.pdf"        # 07/09 工单参考答案
CCF_DIR = SRC_ATTACH / "ccf_competition"                          # 07/08/12 工单金融年报
CCF_PDF_DIR = CCF_DIR / "pdf"
CCF_TXT_DIR = CCF_DIR / "txt"

# --------------------------------------------------------------------------
# 模型配置
# --------------------------------------------------------------------------
# --- 生成模型：DeepSeek API ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "deepseek-chat")

# --- 备用生成模型：本地 Ollama ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = "qwen3:4b"

# --- 向量模型：本地 bge-large-zh-v1.5 ---
EMBED_MODEL_NAME = os.environ.get(
    "RAG_EMBED_MODEL", "BAAI/bge-large-zh-v1.5"
)
EMBED_DIM = 1024
EMBED_BATCH_SIZE = 32
EMBED_MAX_LENGTH = 512

# 微调后模型（工单11 产出）
FINETUNED_EMBED_DIR = PROJECT_ROOT / "工单11-Embedding模型微调" / "models" / "bge-ft-finance"

# --------------------------------------------------------------------------
# 检索超参
# --------------------------------------------------------------------------
CHUNK_SIZE = 400            # 字符数
CHUNK_OVERLAP = 80
TOP_K_RECALL = 20           # 召回阶段
TOP_K_RERANK = 5            # 重排后送入 LLM
RRF_K = 60                  # 混合检索 RRF 融合常数
HYBRID_ALPHA = 0.6          # 向量权重，1-alpha 为 BM25 权重

# --------------------------------------------------------------------------
# 各工单评测问题集
# --------------------------------------------------------------------------
# 工单01/02/03/04 「兴图新科（招股说明书1）」问题集
QUESTIONS_XINGTU = [
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": 95,  "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},
    {"id": 33,  "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},
    {"id": 34,  "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"},
]

# 工单03/04 「力源信息（招股说明书2）」问题集
QUESTIONS_LIYUAN = [
    {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"},
    {"id": 2, "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？"},
    {"id": 3, "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？"},
    {"id": 4, "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？"},
]

# 工单04 图像类问题（需要多模态解析才能答对）
QUESTIONS_IMAGE = [
    {"id": 5, "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？"},
    {"id": 6, "question": "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？"},
]

# 全量问题（03/04 工单共用，含兴图 10 问）
QUESTIONS_ALL = QUESTIONS_LIYUAN + QUESTIONS_IMAGE + QUESTIONS_XINGTU

# 工单05 多轮对话脚本
MULTI_TURN_SCRIPT = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "他参与的哪个工程荣获了国家科技进步一等奖？",
    "这个公司的法定代表人是谁？",
    "那武汉力源信息技术股份有限公司呢？",
    "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
]


def get_all_questions() -> list[dict]:
    """返回工单01~06 共用的完整问题列表（去重）。"""
    seen, out = set(), []
    for q in QUESTIONS_ALL:
        if q["id"] not in seen:
            seen.add(q["id"])
            out.append(q)
    return out
