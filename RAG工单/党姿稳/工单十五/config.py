# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-15-优化技术图纸与文本的跨模态检索流程
# 模块职责：集中存放跨模态检索工单的全局配置（路径、模型、解析参数、跨模态检索超参、6 个验收问题）

import os                                                        # 路径拼接与目录创建
import sys                                                       # 修正标准输出编码

# ---------------------------------------------------------------------------
# 零、进程级环境变量（必须在 torch 导入前设置）
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")              # OpenMP 空闲线程休眠
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")         # 关闭分词多进程并行

for _name in ("stdout", "stderr"):                               # 标准输出切换 UTF-8
    _stream = getattr(sys, _name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ---------------------------------------------------------------------------
# 一、目录
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # 工单十五根目录
DATA_DIR = os.path.join(BASE_DIR, "data")                        # 缓存目录
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")                  # 报告目录
PATCH_DIR = os.path.join(BASE_DIR, "patches")                    # RAGFlow 改动说明目录
for _d in (DATA_DIR, EVAL_DIR, PATCH_DIR):                       # 自动建目录
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# 二、数据源（承接工单十四，改用跨模态测试文档）
# ---------------------------------------------------------------------------
ATTACH_DIR = r"C:\Users\dang5\Desktop\新工单\14-17附件\original_problems\original_problems"
DOC_DIR = os.path.join(ATTACH_DIR, "documents")                  # 1700 份专利 PDF
QUESTIONS_FILE = os.path.join(ATTACH_DIR, "questions.jsonl")     # 问题集
TARGET_PDF = os.path.join(DOC_DIR, "CN100347506C.pdf")           # 本工单目标文档（图 3 在第 11 页）

# ---------------------------------------------------------------------------
# 三、模型
# ---------------------------------------------------------------------------
EMBED_MODEL_PATH = r"C:\Users\dang5\Desktop\zongwenjianjia\zhuangao4\m3e-base"            # 文本嵌入（768 维）
RERANK_MODEL_PATH = r"C:\Users\dang5\Desktop\zongwenjianjia\zhuangao4\bge-reranker-base"  # 跨模态重排（Cross-Encoder）
VLM_MODEL_PATH = r"C:\Users\dang5\Desktop\zongwenjianjia\zhuangao4\Qwen2-VL-2B-Instruct"   # 视觉语言模型（图纸描述）
EMBED_THREADS = 16                                               # 编码线程数

# 视觉语言模型推理参数（针对本机 i5-1340P / 16GB 内存实测调优）
# 本机 CPU 不支持 AVX512-BF16，实测 bf16 矩阵乘比 fp32 慢约 41 倍（1024³ 矩阵：10097ms vs 247ms），
# 而 fp32 的 2B 模型约 8.8GB 会触发换页，因此视觉塔用 fp32、语言塔做动态 int8 量化。
VLM_QUANTIZE_LM = True                                           # 语言塔动态 int8 量化
VLM_MIN_PIXELS = 256 * 28 * 28                                   # 图像最小像素（控制视觉 token 数）
VLM_MAX_PIXELS = 640 * 28 * 28                                   # 图像最大像素（控制视觉 token 数）
VLM_MAX_NEW_TOKENS = 140                                         # 单张图纸描述的最大生成长度
VLM_THREADS = 12                                                 # torch 推理线程数

DEEPSEEK_API_KEY = "sk-REDACTED"         # DeepSeek 密钥
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"                # 接口地址
DEEPSEEK_MODEL = "deepseek-chat"                                 # 对话模型
LLM_TIMEOUT = 60                                                 # 单次超时（秒）

# ---------------------------------------------------------------------------
# 四、PDF 解析参数（沿用并遵循工单十四的解析修复方案）
# ---------------------------------------------------------------------------
RENDER_DPI = 200                                                 # 扫描页渲染 DPI
SCAN_PAGE_CHAR_THRESHOLD = 30                                    # 扫描页判定阈值（字符）
SCAN_RATIO_THRESHOLD = 0.70                                      # 扫描型判定占比
OCR_MIN_SCORE = 0.50                                             # OCR 置信度下限
FIGURE_MIN_AREA_RATIO = 0.05                                     # 图纸面积占比下限
CHUNK_SIZE = 500                                                 # 文本块长度
CHUNK_OVERLAP = 50                                               # 块间重叠

# ---------------------------------------------------------------------------
# 五、跨模态检索超参数
# ---------------------------------------------------------------------------
VECTOR_TOP_K = 30                                                # 文本向量召回条数
BM25_TOP_K = 30                                                  # 全文召回条数
FIGURE_TOP_K = 12                                                # 图像增强查询召回条数
FUSION_TOP_K = 30                                                # 融合后候选数
RERANK_TOP_K = 10                                                # 重排后候选数
FINAL_TOP_K = 6                                                  # 最终入上下文的块数
RRF_K = 60                                                       # RRF 平滑常数
CONTEXT_MAX_CHARS = 12000                                        # 上下文上限

# 多路召回融合权重（跨模态三路的相对重要性，可由外部配置调整）
ROUTE_WEIGHTS = {"text_vector": 0.35,                             # 原始问题文本向量路
                 "figure_vector": 0.35,                           # 图像增强查询向量路
                 "bm25": 0.15,                                    # 全文路
                 "figure_label": 0.15}                            # 图号/页码精确匹配路
RERANK_ENABLED = True                                            # 是否启用重排（外部可关）
RERANK_WEIGHT = 0.65                                             # 重排分在新排序中的权重
FUSION_WEIGHT = 0.35                                             # 融合分在新排序中的权重
FIGURE_LABEL_BOOST = 0.30                                        # 命中图号/页码时的额外加成
RERANK_MAX_LENGTH = 320                                          # 重排模型最大输入长度（控制 CPU 耗时）
RERANK_TEXT_CHARS = 300                                          # 参与重排的候选文本截断长度

# 视觉引用识别：从问题中抽取"图3""第11页图3""图示""图纸"等引用
VISUAL_REF_KEYWORDS = ["图", "图示", "图纸", "附图", "示意图", "图中", "见图"]
PAGE_REF_RE = r"第\s*([0-9]{1,3})\s*页"                           # "第11页"
FIG_REF_RE = r"图\s*([0-9]{1,2})"                                 # "图3"

# ---------------------------------------------------------------------------
# 六、6 个验收问题（工单十五：前 2 题纯文本，后 4 题图文强关联）
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    {"id": 1, "group": "text",
     "question": "根据专利文本，本发明主要涉及哪种物料的分配装置？",
     "options": ["A. 块状散料", "B. 液态金属", "C. 粉状燃料", "D. 气态还原剂"],
     "answer": "A"},
    {"id": 2, "group": "text",
     "question": "根据专利文本，本发明的分散装置包含以下哪个组件？",
     "options": ["A. 链条", "B. 螺旋叶片", "C. 振动筛", "D. 旋转刮板"],
     "answer": "A"},
    {"id": 3, "group": "figure",
     "question": "在文件中第11页图3中，编号13的部件相对于编号12的部件的位置关系是？",
     "options": ["A. 位于编号12的部件之内", "B. 位于编号12的部件之外",
                 "C. 位于编号12的部件上方", "D. 与编号12的部件并排"],
     "answer": "A"},
    {"id": 4, "group": "figure",
     "question": "在文件中第11页图3中，编号14的部件位于整个装置的哪个位置？",
     "options": ["A. 顶部", "B. 底部", "C. 左侧", "D. 右侧"],
     "answer": "A"},
    {"id": 5, "group": "figure",
     "question": "根据文件中第11页图3，散料从部件14进入后，下一步会经过哪个部件？",
     "options": ["A. 部件10", "B. 部件11", "C. 部件13", "D. 部件12"],
     "answer": "C"},
    {"id": 6, "group": "figure",
     "question": "在文件中第11页图3的装置中，如果需要调整链条的位置，需要操作哪个部件?",
     "options": ["A. 部件10", "B. 部件11", "C. 部件12", "D. 部件14"],
     "answer": "B"},
]

SNIPPET_PREVIEW = 120                                            # 报告片段预览长度
