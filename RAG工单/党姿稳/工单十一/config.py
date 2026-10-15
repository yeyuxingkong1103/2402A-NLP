# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
# 模块职责：集中存放全局配置（路径、嵌入模型、LLM、数据集、训练与评估超参数）
# 说明：所有路径均以本文件所在目录为基准推导，避免硬编码导致迁移后报错

import os                                                       # 导入 os 用于路径拼接与目录创建
import sys                                                      # 导入 sys 用于修正标准输出编码

# Windows 控制台默认 GBK 编码，打印中文日志会抛 UnicodeEncodeError，这里统一改为 UTF-8
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # 当前 py 文件所在目录（工单十一根目录）
DATA_DIR = os.path.join(BASE_DIR, "data")                       # 数据目录：分块、问答对、三元组与缓存
MODEL_DIR = os.path.join(BASE_DIR, "model")                     # 模型目录：微调后的嵌入模型保存到此处
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")                   # 语料目录：9 份金融年报 txt
FINETUNED_MODEL_DIR = os.path.join(MODEL_DIR, "m3e-base-finance")  # 微调后模型保存路径（产出物之一）

# ---------------------------------------------------------------------------
# 二、嵌入模型配置
# ---------------------------------------------------------------------------
EMBED_MODEL_PATH = r"C:\Users\dang5\Desktop\zongwenjianjia\zhuangao4\m3e-base"  # 本地 m3e-base 基础模型（768 维）
EMBED_DIM = 768                                                 # m3e-base 输出向量维度
EMBED_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："      # m3e 系列查询侧官方推荐前缀
EMBED_THREADS = 8                                               # 编码线程数（实测超过 8 后调度开销反超收益）

# ---------------------------------------------------------------------------
# 三、大模型配置（用于生成领域问答对数据集）
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = "sk-REDACTED"        # DeepSeek API 密钥
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"               # DeepSeek API 基础地址（OpenAI 兼容）
DEEPSEEK_MODEL = "deepseek-chat"                                # 使用的对话模型名称
LLM_TIMEOUT = 60                                                # 单次 LLM 请求超时时间（秒）
LLM_MAX_RETRY = 3                                               # LLM 调用失败时的最大重试次数
LLM_WORKERS = 8                                                 # 问答对生成的并发线程数（提升数据集生成速度）
LLM_CACHE_FILE = os.path.join(DATA_DIR, "llm_cache.json")       # LLM 结果磁盘缓存（避免重复计费）

# ---------------------------------------------------------------------------
# 四、语料分块配置（与工单九保持一致，保证微调与线上检索域一致）
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500                                                # 每个文本块的目标字符数
CHUNK_OVERLAP = 50                                              # 相邻文本块之间的重叠字符数
MIN_CHUNK_LEN = 30                                              # 低于该长度的文本块直接丢弃（噪音块）
# 资源约束：本机为 CPU 编码，9 份年报全文约 695 万字，全量向量化需 1 小时以上
# 采样策略与工单九一致：每个文件优先保留前 40 页（董事长致辞/经营讨论等叙述密集区），其余页均匀抽样
CORPUS_MAX_CHARS = 600000                                       # 语料字符总量上限
CORPUS_HEAD_PAGES = 40                                          # 每个来源文件优先保留的头部页数
CHUNK_CACHE_FILE = os.path.join(DATA_DIR, "chunks.json")        # 分块结果缓存
BASE_EMBED_CACHE = os.path.join(DATA_DIR, "embeddings_base.npy")  # 基础模型向量缓存（微调前评估复用）

# ---------------------------------------------------------------------------
# 五、数据集生成配置（工单步骤 1、2：数据集生成与问答对生成）
# ---------------------------------------------------------------------------
QA_CHUNK_SAMPLE = 160                                           # 参与生成问答对的语料块数量（采样控制成本）
QA_PER_CHUNK = 2                                                # 每个语料块要求大模型生成的问答对数量
QA_MIN_ANSWER_LEN = 10                                          # 答案过短的问答对直接丢弃（过滤无效样本）
EVAL_CHUNK_RATIO = 0.2                                          # 留出评估集占比（按块划分，避免同块泄漏）
RANDOM_SEED = 42                                                # 全局随机种子，保证数据集可复现
QA_FILE = os.path.join(DATA_DIR, "qa_pairs.json")               # 问答对数据集落盘路径（产出物之一）

# ---------------------------------------------------------------------------
# 六、三元组构造配置（工单步骤 3：数据集与模型加载）
# ---------------------------------------------------------------------------
NEG_PER_ANCHOR = 1                                              # 每个锚点挖掘的难负例数量
HARD_NEG_POOL = 50                                              # 难负例候选池大小（取检索排名 2~51 的块作为难负例来源）
TRIPLET_FILE = os.path.join(DATA_DIR, "triplets.json")          # 三元组数据集落盘路径（产出物之一）

# ---------------------------------------------------------------------------
# 七、训练配置（工单步骤 4、5：定义损失函数与定义训练参数）
# ---------------------------------------------------------------------------
LOSS_TYPE = "TripletLoss"                                       # 主损失函数：三元组损失（适用于锚点/正例/负例）
DISTANCE_METRIC = "cosine"                                      # 距离度量：余弦距离（与检索阶段一致）
TRIPLET_MARGIN = 0.5                                            # 三元组损失的间隔（正负例距离差的目标下界）
USE_MATRYOSHKA = True                                           # 是否叠加套娃损失（工单要求的损失函数之一）
MATRYOSHKA_DIMS = [768, 384, 192]                               # 套娃损失的截断维度层级
EPOCHS = 3                                                      # 训练轮数（CPU 约束下取小值）
BATCH_SIZE = 16                                                 # 批大小（CPU 内存与速度的折中）
LEARNING_RATE = 2e-5                                            # 学习率（微调嵌入模型的常用量级）
WARMUP_RATIO = 0.1                                              # 学习率预热比例
MAX_SEQ_LENGTH = 256                                            # 最大序列长度（年报块平均约 250 字，截断到 256 足够）
TRAIN_LOG_FILE = os.path.join(DATA_DIR, "train_log.json")       # 训练过程记录（训练损失与耗时）

# ---------------------------------------------------------------------------
# 八、评估配置（工单步骤 6、7、8：创建评估器与微调前后评估）
# ---------------------------------------------------------------------------
EVAL_KS = [1, 5, 10]                                            # 检索评估的 K 值：Recall@1 / Recall@5 / Recall@10
NDCG_K = 10                                                     # nDCG 的截断位置
EVAL_BATCH = 32                                                 # 评估时编码的批大小
EVAL_RESULT_FILE = os.path.join(DATA_DIR, "eval_result.json")   # 微调前后指标落盘（产出物之一）
REPORT_FILE = os.path.join(BASE_DIR, "微调报告.md")              # 微调报告（产出物之一）


def ensure_dirs():                                              # 定义目录初始化函数
    """创建运行所需的全部目录，重复调用无副作用。"""
    for d in (DATA_DIR, MODEL_DIR, FINETUNED_MODEL_DIR):        # 遍历需要保证存在的目录
        os.makedirs(d, exist_ok=True)                           # 不存在则创建（存在则忽略）
