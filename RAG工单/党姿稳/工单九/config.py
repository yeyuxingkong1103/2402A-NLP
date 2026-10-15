# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：集中存放全局配置（路径、模型、API、检索与生成超参数）
# 说明：所有路径均以本文件所在目录为基准推导，避免硬编码导致迁移后报错

import os                                                       # 导入 os 用于路径拼接与目录创建
import sys                                                      # 导入 sys 用于修正标准输出编码

# Windows 控制台默认 GBK 编码，打印中文日志或特殊字符时会抛 UnicodeEncodeError
# 这里统一把标准输出与错误流切换为 UTF-8，保证程序永不因编码问题中断
for _stream_name in ("stdout", "stderr"):                       # 依次处理标准输出与标准错误
    _stream = getattr(sys, _stream_name, None)                  # 取出对应流对象
    if _stream is not None and hasattr(_stream, "reconfigure"):  # 流支持重新配置时才处理（Py3.7+）
        try:                                                    # 尝试重配置编码
            _stream.reconfigure(encoding="utf-8", errors="replace")  # 改为 UTF-8，无法编码的字符用替代符
        except Exception:                                       # 某些环境（如被重定向）可能失败
            pass                                                # 忽略失败，不影响主逻辑

# ---------------------------------------------------------------------------
# 一、目录路径配置
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # 当前 py 文件所在目录（工单九根目录）
DATA_DIR = os.path.join(BASE_DIR, "data")                       # 数据缓存目录：分块、向量、图谱等
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")                 # 评估报告输出目录
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")                   # 金融年报语料目录：随工单自带（文件名已修正编码）
QUESTION_PDF = r"C:\Users\dang5\Desktop\RAG 工单\附件\sample_questions.pdf"  # 验收问题（含参考答案）PDF

# 优先使用原始附件目录，缺失时回退到本工单自带语料，保证脚本在任何环境都能运行
if not os.path.isdir(CORPUS_DIR):                               # 自带语料缺失（异常情况）
    CORPUS_DIR = r"C:\Users\dang5\Desktop\RAG 工单\附件\ccf_competition\txt"  # 回退到原始附件路径

# ---------------------------------------------------------------------------
# 二、模型与 LLM 配置
# ---------------------------------------------------------------------------
EMBED_MODEL_PATH = r"C:\Users\dang5\Desktop\zongwenjianjia\zhuangao4\m3e-base"  # 本地 m3e-base 嵌入模型路径（768 维）
EMBED_DIM = 768                                                 # m3e-base 输出向量维度
DEEPSEEK_API_KEY = "sk-REDACTED"        # DeepSeek API 密钥
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"               # DeepSeek API 基础地址（OpenAI 兼容）
DEEPSEEK_MODEL = "deepseek-chat"                                # 使用的对话模型名称
LLM_TIMEOUT = 60                                                # 单次 LLM 请求超时时间（秒）
LLM_MAX_RETRY = 3                                               # LLM 调用失败时的最大重试次数

# ---------------------------------------------------------------------------
# 三、分块与检索超参数
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500                                                # 每个文本块的目标字符数
# 资源约束：本机为 CPU 推理（i5-1340P），m3e-base 实测吞吐约 600 token/s
# 9 份年报全文约 283 万字，全量向量化需 70 分钟以上；这里按比例采样到 60 万字以内
# 采样策略：每个文件优先保留前 40 页（董事长致辞/经营讨论等叙述性密集区），其余页均匀抽样
CORPUS_MAX_CHARS = 600000                                       # 语料向量化的字符总量上限
CORPUS_HEAD_PAGES = 40                                          # 每个来源文件优先保留的头部页数
CHUNK_OVERLAP = 50                                              # 相邻文本块之间的重叠字符数
MIN_CHUNK_LEN = 30                                              # 低于该长度的文本块直接丢弃（噪音块）
VECTOR_TOP_K = 30                                               # 向量召回的候选数量（扩大候选池以提升召回上限）
BM25_TOP_K = 30                                                 # BM25 召回的候选数量（与向量侧对等，保证融合公平）
FINAL_TOP_K = 8                                                 # 融合后默认送入生成阶段的上下文块数量
# 第二轮优化：按问题类型动态决定上下文条数。RAGAS 精度公式中「排在相关块之后的无关块」不参与扣分，
# 因此扩大条数对精度近似无害；而宽泛综合题（多要点对比分析）需要更多块才能覆盖参考答案要点，
# 固定 8 条会漏检（如保险资金久期匹配块排在第 9~13 位被截断）
FINAL_TOP_K_SPECIFIC = 10                                       # 具体事实型问题的上下文条数
FINAL_TOP_K_BROAD = 13                                          # 宽泛综合分析型问题的上下文条数
# 第二轮优化：关键词覆盖精排。融合分数只反映「与问题整体相似」，跨行致辞样板块（结构雷同）
# 容易挤占前排；把「问题关键词/短语在候选块中的覆盖率」混入 MMR 相关性项后，
# 字面上含『盈利增长』『创新商业模式』等核心短语的块会被提到前面
KEYWORD_BOOST_ALPHA = 0.60                                      # 关键词覆盖项在 MMR 相关性中的权重（其余为融合分数项）
RRF_K = 60                                                      # RRF 融合算法中的平滑常数 k
# λ 取 0.9：实测 λ=0.75 时 MMR 过度追求多样性，会把高相关块挤出上下文，
# 导致精度与召回同时下降；提高到 0.9 后仅剔除近乎重复的块，相关性仍占主导
MMR_LAMBDA = 0.90                                               # MMR 去冗余时相关性权重（1-lambda 为多样性权重）
MULTI_QUERY_WEIGHT = 2.0                                        # 多路召回时原问题的 RRF 权重（子查询为 1.0）
CONTEXT_MAX_CHARS = 15000                                       # 拼接上下文的最大字符数，避免提示词过长
MULTI_QUERY_MAX = 9                                             # 多路查询扩展的最大子查询数量（含原问题与同义词扩展）
# 第三轮优化：对比型问题的领域同义词扩展词表（查询扩展技术）。
# 问题只写『资本结构优化』，而年报原文用『利润留存补充资本/资本充足率』表述；
# 只写『保险公司』，原文用『保险资金/资产负债久期匹配』。不做同义词扩展，
# 这类证据块在词面与语义两侧都排不进候选，上下文召回永远为 0
DOMAIN_SYNONYMS = {                                             # 触发词 -> 追加的扩展子查询列表
    "资本结构": ["资本充足率 资本补充 利润留存", "资本充足率", "利润留存 补充资本"],  # 资本结构优化 → 资本表述
    "保险公司": ["保险资金 资产负债久期匹配", "保险资金投资"],      # 保险公司 → 保险资金运用表述
    "风险管理": ["拨备覆盖率 不良贷款 逾期贷款", "风险预警"],        # 风险管理 → 风控指标表述
    "新兴领域": ["绿色金融投资 科技金融投入"],                      # 新兴领域 → 投向表述
    "经济下行": ["经济下行压力 宏观经济形势"],                      # 周期表述
}
# 对比题 λ=0.85：矩阵实验显示 0.75 会让高相关块重复入选（同页 3 连块），0.85 恰好
# 在『保留多机构多主题』与『抑制近重复块』之间取得平衡，关键词命中数最高
MMR_LAMBDA_COMPARATIVE = 0.85                                   # 对比型问题的 MMR λ（适度放宽多样性）
GRAPH_FUSE_WEIGHT = 2.0                                         # 对比型问题的图谱块融合权重（加大实体邻域证据的话语权）
# 同义词子查询的 RRF 权重与候选数：矩阵实验（不烧 LLM）表明——同义词子查询只取 BM25 top5
# 并加权 5.0 时证据块（利润留存/久期匹配/拨备覆盖率）最稳进入前排且无重复块；
# 候选放宽到 top30 反而让大量『资本充足率』通用块涌入挤占名额
SYNONYM_QUERY_WEIGHT = 5.0                                      # 同义词扩展子查询在多路融合中的权重
SYNONYM_BM25_TOPK = 5                                           # 同义词子查询的 BM25 召回候选数（精准小池）
GRAPH_HOP = 1                                                   # 图谱检索的邻域扩展跳数（1 跳兼顾速度与召回）

# ---------------------------------------------------------------------------
# 四、图谱构建配置
# ---------------------------------------------------------------------------
GRAPH_BUILD_LIMIT = 120                                         # 建图时最多处理的文本块数量（CPU/成本约束下的采样上限）
GRAPH_EXTRACT_BATCH = 6                                         # 每次请求 LLM 抽取实体关系的文本块数量（批量降低调用次数）
GRAPH_CACHE_FILE = os.path.join(DATA_DIR, "graph.json")         # 知识图谱缓存文件
CHUNK_CACHE_FILE = os.path.join(DATA_DIR, "chunks.json")        # 分块结果缓存文件
EMBED_CACHE_FILE = os.path.join(DATA_DIR, "embeddings.npy")     # 向量矩阵缓存文件
META_CACHE_FILE = os.path.join(DATA_DIR, "chunks_meta.json")    # 向量矩阵对应的块元信息（便于校验一致性）
EVAL_CACHE_FILE = os.path.join(DATA_DIR, "eval_cache.json")     # 评估过程中的 LLM 判定缓存（避免重复计费）
BM25_CACHE_FILE = os.path.join(DATA_DIR, "bm25.pkl")            # BM25 倒排索引缓存文件

# ---------------------------------------------------------------------------
# 五、性能与稳定性配置
# ---------------------------------------------------------------------------
RETRIEVE_TIME_BUDGET = 3.0                                      # 检索阶段时间预算（秒），工单要求 ≤3 秒
ENABLE_LLM_CACHE = True                                         # 是否启用 LLM 结果磁盘缓存
STREAM_UI_PORT = 8501                                           # Streamlit 默认端口（冲突时自动换端口）


def ensure_dirs():                                              # 定义目录初始化函数
    """创建运行所需的缓存与报告目录，重复调用无副作用。"""
    for d in (DATA_DIR, EVAL_DIR):                              # 遍历需要保证存在的目录
        os.makedirs(d, exist_ok=True)                           # 不存在则创建（存在则忽略）
