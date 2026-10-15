# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-14-修复低质量工业PDF的解析与信息丢失
# 模块职责：集中存放全局配置（路径、模型、解析参数、检索超参数、6 个验收问题）

import os                                                       # 路径拼接与目录创建
import sys                                                      # 修正标准输出编码，避免中文日志在 Windows 控制台抛异常

# ---------------------------------------------------------------------------
# 零、进程级环境变量（必须在 torch 导入前设置）
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")             # OpenMP 空闲线程休眠，降低编码耗时抖动
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")        # 关闭分词多进程并行，避免与 torch 线程池争抢 CPU

for _name in ("stdout", "stderr"):                              # 统一把标准输出/错误流切到 UTF-8
    _stream = getattr(sys, _name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ---------------------------------------------------------------------------
# 一、目录
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # 工单十四根目录
DATA_DIR = os.path.join(BASE_DIR, "data")                       # 解析/向量/索引缓存目录
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")                 # 评估报告与图表输出目录
for _d in (DATA_DIR, EVAL_DIR):                                 # 首次运行时自动建目录
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# 二、数据源（IMDR 数据集中的目标专利文档）
# ---------------------------------------------------------------------------
ATTACH_DIR = r"C:\Users\dang5\Desktop\新工单\14-17附件\original_problems\original_problems"
DOC_DIR = os.path.join(ATTACH_DIR, "documents")                 # 1700 份专利 PDF
QUESTIONS_FILE = os.path.join(ATTACH_DIR, "questions.jsonl")    # 10096 条多模态问题
TARGET_PDF = os.path.join(DOC_DIR, "CN100342976C.pdf")          # 本工单要求构建知识库的目标文档

# ---------------------------------------------------------------------------
# 三、模型
# ---------------------------------------------------------------------------
EMBED_MODEL_PATH = r"C:\Users\dang5\Desktop\zongwenjianjia\zhuangao4\m3e-base"       # 本地 m3e-base 嵌入模型（768 维）
RERANK_MODEL_PATH = r"C:\Users\dang5\Desktop\zongwenjianjia\zhuangao4\bge-reranker-base"  # 本地 bge 重排模型
VLM_MODEL_PATH = r"C:\Users\dang5\Desktop\zongwenjianjia\zhuangao4\Qwen2-VL-2B-Instruct"  # 本地视觉语言模型（离线生成图纸描述）
EMBED_THREADS = 16                                              # 编码线程数（本机 16 核，实测取满核中位耗时最低）

DEEPSEEK_API_KEY = "sk-REDACTED"        # DeepSeek API 密钥
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"               # OpenAI 兼容接口
DEEPSEEK_MODEL = "deepseek-chat"                                # 对话模型
LLM_TIMEOUT = 60                                                # 单次请求超时（秒）

# ---------------------------------------------------------------------------
# 四、PDF 解析参数（"修复信息丢失"的核心开关）
# ---------------------------------------------------------------------------
RENDER_DPI = 200                                                # 图片型 PDF 渲染为位图用于 OCR 的分辨率
# 每页可提取字符数低于该阈值即判定为"扫描页"（需 OCR）；这是分流启发式，可配置
SCAN_PAGE_CHAR_THRESHOLD = 30
# 扫描页占比超过该比例则整份文件判定为"扫描型"
SCAN_RATIO_THRESHOLD = 0.70
OCR_MIN_SCORE = 0.50                                            # RapidOCR 置信度过滤阈值，低于此分的识别结果丢弃
FIGURE_MIN_AREA_RATIO = 0.05                                    # 页面图像面积占比超过此值才认为是"图纸页"

CHUNK_SIZE = 500                                                # 文本块目标长度（字符）
CHUNK_OVERLAP = 50                                              # 相邻块重叠长度（字符）

# ---------------------------------------------------------------------------
# 五、检索超参数
# ---------------------------------------------------------------------------
VECTOR_TOP_K = 30                                               # 向量召回条数
BM25_TOP_K = 30                                                 # 全文召回条数
FUSION_TOP_K = 30                                               # 融合后进入重排的条数
FINAL_TOP_K = 6                                                 # 最终送入 LLM 的上下文块数
RRF_K = 60                                                      # RRF 融合的平滑常数
VECTOR_WEIGHT = 0.6                                             # 加权融合时向量得分权重
BM25_WEIGHT = 0.4                                               # 加权融合时 BM25 得分权重
CONTEXT_MAX_CHARS = 12000                                       # 拼接上下文的最大字符数

# ---------------------------------------------------------------------------
# 六、6 个验收问题（来自工单十四，答案与 questions.jsonl 中的选项一致）
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    {"id": 1, "question": "根据文本信息，该静电除尘器的发明人是：",
     "options": ["A. P·吉特勒", "B. 未知", "C. 静电除尘器研究小组", "D. 该专利未提及发明人"],
     "answer": "A", "group": "text"},
    {"id": 2, "question": "根据文本信息，以下哪个描述符合该静电除尘器的特征？",
     "options": ["A. 具有平行于外壳主轴线的垂直方向的片状沉积电极。",
                 "B. 具有管状入口和出口，它们分别由3种不同圆锥形部分所构成",
                 "C. 管状入口具有单个圆锥形部分，达到外壳直径的80至95%，剩余部分采用台阶形式。",
                 "D. 主要用于液体的除尘"],
     "answer": "C", "group": "text"},
    {"id": 3, "question": "在文件中第7页的图片中，部件4相对于部件5在图片中的位置关系是？",
     "options": ["A. 部件4位于部件5的左侧", "B. 部件4位于部件5的右侧",
                 "C. 部件4位于部件5的上方", "D. 部件4位于部件5的下方"],
     "answer": "A", "group": "figure"},
    {"id": 4, "question": "在文件中第7页的图片中，尺寸X1，X2，X3分别代表什么部件的间隔距离？",
     "options": ["A. 配气带孔盘6，6'，6\"之间的间隔距离",
                 "B. 外壳2与圆锥形部分10之间的间隔距离",
                 "C. 管状入口1与管状出口9之间的间隔距离",
                 "D. 外壳中心轴线3与外壳2之间的间隔距离"],
     "answer": "A", "group": "figure"},
    {"id": 5, "question": "根据文件中第7页图示，气流方向(7)首先经过哪个部件？紧接着会经过哪个部件？",
     "options": ["A. 先经过部件4，再经过部件5", "B. 先经过部件10，再经过部件6\"",
                 "C. 先经过部件6\"，再经过部件6'", "D. 先经过部件6，再经过部件4"],
     "answer": "C", "group": "figure"},
    {"id": 6, "question": "根据文件中第7页图示，如果已知外壳直径D，那么h1和h2的尺寸可以用来计算什么？",
     "options": ["A. 计算气流速度", "B. 确定配气带孔盘6，6'，6\"的位置",
                 "C. 计算除尘效率", "D. 确定外壳2的位置"],
     "answer": "B", "group": "figure"},
]

SNIPPET_PREVIEW = 120                                           # 评估报告中每条检索片段的预览字符数
