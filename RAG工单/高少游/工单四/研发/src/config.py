# -*- coding: utf-8 -*-
"""项目全局配置（图像内容解析与检索优化版）
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

说明：
本工单在《01-基于 PDF 文档的问答系统》《02-基于 PDF 文档的问答系统的优化》
《03-PDF 文档的表格解析及检索优化》基础上，新增对《招股说明书2.pdf》
（武汉力源信息技术股份有限公司）中**图像内容**的语义解析与检索：

    - 图像内容抽取：嵌入位图 + 矢量绘图簇（组织结构图等）→ 图形区域
    - 图像语义解析：多模态模型（Chinese-CLIP）跨模态编码 + 结构化语义还原
    - 图像检索：CLIP 跨模态图像索引，支持「以文搜图」定位答案所在图形
    - 图像答案合成：从图形语义中抽取答案（如组织结构层级、图表涨跌幅）

所有可调参数优先读取环境变量 / .env，其次使用默认值。
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- 目录与路径 -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent    # 项目根目录（研发/）
DATA_DIR = BASE_DIR / "data"                         # 数据目录（PDF、题目）
DB_DIR = BASE_DIR / "vector_db" / "optimized"        # 【优化后】文本 FAISS 向量库
BASE_DB_DIR = BASE_DIR / "vector_db" / "baseline"    # 【优化前】文本 FAISS 向量库
IMAGE_DB_DIR = BASE_DIR / "vector_db" / "image"      # 【本工单新增】图像 CLIP 向量库
FIGURE_DIR = BASE_DIR / "figures"                    # 【本工单新增】抽取出的图形素材
OUTPUT_DIR = BASE_DIR / "output"                     # 评估结果、日志输出目录
FIGURE_OUT_DIR = OUTPUT_DIR / "figures"              # 评估图表输出目录
SHOT_DIR = OUTPUT_DIR / "shots"                      # 界面截图输出目录

# 知识库源文档：01/02/03 工单的招股说明书1 + 本工单重点解析的招股说明书2
PDF_PATHS = [DATA_DIR / "招股说明书1.pdf", DATA_DIR / "招股说明书2.pdf"]
PDF_PATH = PDF_PATHS[0]                              # 兼容旧接口（默认首个文档）
QUESTIONS_PATH = DATA_DIR / "questions.json"         # 待评估问题清单

# ---- 多文档实体消歧 ----------------------------------------------------------
# 知识库同时收录两份招股说明书，二者都存在“发行股数 / 募集资金 / 关联方”等
# 同构表格与图形。若把公司名当噪声剔除，会出现跨文档串味（问力源却召回兴图新科）。
# 故建立「公司别名 → 源文件名」映射，用于文档级路由（检索过滤 + 重排加成）。
DOC_COMPANY = {
    "招股说明书1.pdf": ["武汉兴图新科电子股份有限公司", "兴图新科电子", "兴图新科", "兴图"],
    "招股说明书2.pdf": ["武汉力源信息技术股份有限公司", "力源信息技术", "力源信息", "力源"],
}
DOC_FILTER_ENABLE = os.getenv("DOC_FILTER_ENABLE", "1") == "1"   # 是否启用文档级路由
W_DOC = float(os.getenv("W_DOC", "0.18"))                       # 文档匹配加成权重

# ---- 模型配置（Ollama 本地模型） ---------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:1.5b")        # 本地大模型（生成/裁判）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:567m")  # 本地向量模型

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "768"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# ---- 【优化前】基线参数（与 01/02/03 工单一致，用于 before/after 对比）--------
# 基线不做表格结构化、也不做图像解析：表格与图形均被当作普通文本行参与切片
BASE_CHUNK_SIZE = 500
BASE_CHUNK_OVERLAP = 80
BASE_TOP_K = 6
BASE_BM25_TOP_K = 6
BASE_RRF_K = 60

# ---- 【优化后】分块参数 -------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))       # 结构感知切片大小
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120")) # 切片重叠
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "60"))# 过短片段合并阈值
USE_PARENT_CHILD = os.getenv("USE_PARENT_CHILD", "1") == "1"  # 父子块（小节级父块）

# ---- 【优化后】表格解析参数（03 工单能力，本工单沿用） -------------------------
USE_TABLE_PARSER = os.getenv("USE_TABLE_PARSER", "1") == "1"  # 是否启用表格结构化解析
TABLE_MIN_ROWS = int(os.getenv("TABLE_MIN_ROWS", "2"))        # 表格最少行数（过滤误检）
TABLE_MIN_COLS = int(os.getenv("TABLE_MIN_COLS", "2"))        # 表格最少列数
TABLE_KV_ENABLE = os.getenv("TABLE_KV_ENABLE", "1") == "1"    # 是否生成“键值对行”增强召回
TABLE_HEADER_ROWS = int(os.getenv("TABLE_HEADER_ROWS", "1"))  # 表头行数（用于多级表头合并）

# ---- 【本工单新增】图像内容解析参数 -------------------------------------------
USE_FIGURE_PARSER = os.getenv("USE_FIGURE_PARSER", "1") == "1"   # 是否启用图像语义解析
FIGURE_RENDER_ZOOM = float(os.getenv("FIGURE_RENDER_ZOOM", "2.5"))  # 图形区域渲染倍率
# 位图图形的最小有效面积占比（相对页面面积），用于剔除页眉页脚小图标 / 水印碎片
FIGURE_MIN_AREA_RATIO = float(os.getenv("FIGURE_MIN_AREA_RATIO", "0.03"))
# 位图图形的最小边长（像素），进一步过滤装饰性小图
FIGURE_MIN_PIXELS = int(os.getenv("FIGURE_MIN_PIXELS", "120"))
# 位图图形在页面上的最小尺寸（PDF 点），过滤公司 logo、地址小图等装饰图
FIGURE_MIN_PT_WIDTH = float(os.getenv("FIGURE_MIN_PT_WIDTH", "140"))
FIGURE_MIN_PT_HEIGHT = float(os.getenv("FIGURE_MIN_PT_HEIGHT", "70"))
# 矢量绘图簇的最小图形数（组织结构图 / 流程图由大量线条与矩形构成）
FIGURE_MIN_DRAWINGS = int(os.getenv("FIGURE_MIN_DRAWINGS", "8"))
# 矢量簇中矩形（节点框）数量的上界：超过则判定为「未被识别成表格」的网格边框
FIGURE_MAX_BOXES = int(os.getenv("FIGURE_MAX_BOXES", "60"))
# 图形区域内文字数量下限（无文字的装饰图形不入库）
FIGURE_MIN_TEXTS = int(os.getenv("FIGURE_MIN_TEXTS", "4"))
# 图形区域内文字数量上限（超过则判定为正文页，避免整页误判为图形）
FIGURE_MAX_TEXTS = int(os.getenv("FIGURE_MAX_TEXTS", "120"))
# 位图图形 OCR 前的预处理：招股说明书图形带浅灰水印文字，亮度高于该阈值的像素置白，
# 避免水印被 OCR 误识为图内标签（如“维教育”“刘敏”）。
OCR_WATERMARK_LUMA = int(os.getenv("OCR_WATERMARK_LUMA", "190"))

# ---- 【本工单新增】多模态模型（Chinese-CLIP）配置 ------------------------------
# 备注（工单要求）：PDF 中的图像语义解析使用多模态模型（CLIP 或多模态大模型）实现。
CLIP_ENABLE = os.getenv("CLIP_ENABLE", "1") == "1"                # 是否启用 CLIP 跨模态检索
CLIP_MODEL = os.getenv("CLIP_MODEL", "OFA-Sys/chinese-clip-vit-base-patch16")
CLIP_BATCH_SIZE = int(os.getenv("CLIP_BATCH_SIZE", "8"))
IMAGE_TOP_K = int(os.getenv("IMAGE_TOP_K", "5"))                  # 图像召回候选数
# 图形类型零样本分类的候选标签（CLIP 图文匹配）
FIGURE_TYPE_LABELS = [
    "组织结构图", "柱状图", "饼图", "折线图", "流程图", "产品结构图", "股权结构图",
]
# 图像问题判定阈值：CLIP 相似度高于该值即认为问题指向某图形
FIGURE_ROUTE_MIN_SIM = float(os.getenv("FIGURE_ROUTE_MIN_SIM", "0.28"))

# ---- 【优化后】检索参数 -------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "6"))                   # 最终返回片段数
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "30"))    # 向量召回候选数
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "30"))        # BM25 召回候选数
RANK_FUSION_K = int(os.getenv("RANK_FUSION_K", "60"))  # RRF 窗口
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "6"))     # 重排后保留数
ANSWER_POOL = int(os.getenv("ANSWER_POOL", "30"))      # 答案合成候选池大小（扩大召回）

# 重排权重（加权线性融合，权重之和约为 1）
W_VECTOR = float(os.getenv("W_VECTOR", "0.20"))        # 语义相似度
W_BM25 = float(os.getenv("W_BM25", "0.14"))            # 词法命中
W_KEYWORD = float(os.getenv("W_KEYWORD", "0.19"))      # 查询关键词覆盖率
W_NUMERIC = float(os.getenv("W_NUMERIC", "0.11"))      # 数值/年份匹配
W_ENTITY = float(os.getenv("W_ENTITY", "0.06"))        # 实体匹配
W_TABLE = float(os.getenv("W_TABLE", "0.08"))          # 表格块加成（03 工单）
W_FIGURE = float(os.getenv("W_FIGURE", "0.12"))        # 图形块加成（本工单新增）
W_CLIP = float(os.getenv("W_CLIP", "0.10"))            # CLIP 跨模态相似度（本工单新增）

# 查询扩展（Query 理解）
USE_QUERY_EXPANSION = os.getenv("USE_QUERY_EXPANSION", "1") == "1"
USE_LLM_QUERY_UNDERSTANDING = os.getenv("USE_LLM_QUERY_UNDERSTANDING", "0") == "1"

# ---- 【优化后】答案生成 -------------------------------------------------------
# extractive：抽取式答案合成（快、稳、可溯源，默认）
# llm：调用本地大模型生成（raw 模式规避思维链）
ANSWER_MODE = os.getenv("ANSWER_MODE", "extractive")
EXTRACTIVE_MAX_SENTENCES = int(os.getenv("EXTRACTIVE_MAX_SENTENCES", "4"))

VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "prospectus_qa_image")

# ---- 性能目标 -----------------------------------------------------------------
TARGET_RESPONSE_SECONDS = 3.0   # 需求：从提问到生成答案 <= 3s
TARGET_ACCURACY = 0.90          # 需求：问答准确率 >= 90%


def ensure_dirs() -> None:
    """确保关键目录存在（首次运行自动创建）。"""
    for d in (DATA_DIR, DB_DIR, BASE_DB_DIR, IMAGE_DB_DIR, FIGURE_DIR,
              OUTPUT_DIR, FIGURE_OUT_DIR, SHOT_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


ensure_dirs()