# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
"""全局配置：路径、数据集、切分与训练超参

所有可调项集中在这里，训练脚本不写死任何常量。
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ---------- 模型 ----------
# 工单点名要用 BAAI/bge-base-en-v1.5，模型文件已下载到本机
BASE_MODEL = Path(os.getenv("BASE_MODEL", r"D:\Pycharm\yzq\models\bge-base-en-v1.5"))
FINETUNED_MODEL = BASE_DIR / "output" / "bge-base-en-v1.5-fiqa"

# 缓存目录：语料向量、数据集等中间产物。
# 语料有 57638 篇，编码一次要几分钟，必须缓存，否则每跑一次评估都重来。
CACHE_DIR = BASE_DIR / "cache"

# ---------- 数据集 ----------
# FiQA-2018：金融领域问答检索数据集（BEIR 基准之一）。
# 选它的理由：自带 query / corpus / 相关性标注（qrels），能算 nDCG、Recall 这类
# 标准检索指标，微调前后的提升有客观依据，而不是靠"看起来更准了"。
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
CORPUS_REPO = "BeIR/fiqa"
QRELS_REPO = "BeIR/fiqa-qrels"

# ---------- 大模型（生成问答对用）----------
# 与工单 01-10 用的是同一个接口，环境变量也沿用，不另起一套配置
LLM_API_BASE = os.getenv("DEEPSEEK_BASE_URL")
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-flash")

# ---------- 数据切分 ----------
# qrels 覆盖 648 个 query。按 query 切分（不是按标注行切），
# 保证同一个 query 的正例不会同时出现在训练集和评估集里 —— 那是数据泄漏。
SPLIT_SEED = 42
TRAIN_RATIO = 0.7

# ---------- 难负例挖掘 ----------
# 用微调前的模型检索，取排名靠前但不相关的文档当难负例。
# 负例越难，模型学到的边界越细；随机负例太容易，对金融这种术语密集的领域帮助有限。
HARD_NEG_TOP_K = 50        # 每个 query 检索多少个候选里挑
HARD_NEG_PER_QUERY = 4     # 每个 query 挑几个难负例

# ⚠️ 但**训练时默认不用**这些难负例。原因是 FiQA 的标注极其稀疏：
# 1,706 条标注 / 57,638 篇文档 = 覆盖率 0.003%。检索 top-50 里"不在标注中"的文档，
# 绝大多数只是**没被标注**，并不是不相关。拿它们当负例，等于教模型把正确答案推开 ——
# 实测这么做会让 nDCG@10 从 0.4350 掉到 0.3686；在去重修好、冻结底部层之后
# 重测仍是 0.3486（见 优化/过程问题记录.md 问题 7）。
# 改成只用 batch 内负例（标准 MNRL）：其他 query 的正例与本 query 几乎不可能相关，
# 不存在假负例问题。
USE_HARD_NEGATIVES = False

# ---------- 评估 ----------
# nDCG@10 / Recall@10 / MRR@10 是 BEIR 的标准口径，方便与公开结果对照
METRIC_KS = (1, 5, 10)
ENCODE_BATCH = 64
ENCODE_MAX_LEN = 512       # bge 系列训练时就是 512，超出的截断

# BGE 官方用法：短查询检索长文档（s2p）时，**查询侧**要加这句指令，文档侧不加。
# FiQA 正是 s2p 场景。微调与评估必须用同一套口径，否则两次结果没有可比性 ——
# 给基线加指令、给微调后的模型不加，等于人为抬高微调收益。
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# ---------- 训练 ----------
# ⚠️ TRAIN_MAX_LEN 是踩了 OOM 之后定下来的，别随手调大。
# MNR 一个 batch 要跑三次前向（anchor / positive / negative）且都要留激活做反向，
# batch=32 + max_seq_len=512 实测要 12GB 以上，6GB 的卡直接爆。
# 现在：训练序列长砍到 256（语料中位约 121 token，512 是浪费），实测占用约 3GB。
#
# batch 24 是**不用难负例**（两次前向）时的值；若把 USE_HARD_NEGATIVES 打开
# 变成三次前向，要相应降到 16，否则 6GB 卡会 OOM。
TRAIN_BATCH = 24
TRAIN_MAX_LEN = 256
EPOCHS = 2                 # 最终取值；见下面「早停」一段
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

# 冻结底部 6 层编码器（含 embedding），只训顶部 6 层（约 43M / 109M 参数）。
# 这是本工单**决定成败的一个参数**：训练样本只有约 5.6k，全参数微调会把
# 预训练学到的通用语义整体带偏 —— 实测 nDCG@10 从 0.4350 掉到 0.4317，
# 而且**任何**加大力度的调整（加难负例、调温度、多训几轮）都只会更差。
# 冻住底部只做顶部适配后，同一个评估集上变成 0.4412。
# 详见 优化/过程问题记录.md 问题 9。
FREEZE_LAYERS = 6

# 早停。⚠️ 这里的 PATIENCE 保留只是为了兜底，**本工单的正式训练并不启用它**。
# 原因：只用 batch 内负例时，模型把向量空间摊开会让 batch 内那些"别的问题的
# 正确答案"更难分开，dev loss 因此从第一次评估起就单调上涨，与检索质量脱钩。
# 拿它早停会回滚到第 25 步 —— 等于交一个几乎没训的模型。正式训练用
# `--keep-last --patience 1000` 关掉它，固定跑满 EPOCHS 轮。
# 详见 优化/过程问题记录.md 问题 10。
DEV_RATIO = 0.15
EVAL_EVERY = 25
PATIENCE = 5
