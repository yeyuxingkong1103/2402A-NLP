# -*- coding: utf-8 -*-
"""
工单11 步骤2：Embedding 模型微调与前后评估
工单编号：人工智能NLP-RAG 项目-Embedding模型微调任务
功能：
  数据集与模型加载 → 微调前评估 → 定义损失函数(MultipleNegativesRankingLoss，
  即对比损失，正例对拉近、批内负例推远) → 定义训练参数 → 训练 → 微调后评估
  → 保存微调模型，输出前后指标对比（验收要求：微调后检索效果更好，有数据支撑）。
基座模型：BAAI/bge-small-zh-v1.5（小模型，CPU 可训练）
运行：python finetune.py
"""
import sys   # 标准库：修改模块搜索路径
import os    # 标准库：路径拼接与环境变量操作
import json  # 标准库：解析 jsonl 数据集
import random  # 标准库：随机种子与样本打乱

# 关键：清除代理环境变量，否则 HF 模型下载会被本机代理劫持污染
# （踩坑：本机代理会把 huggingface.co 的请求劫持到错误地址，导致模型下载失败）
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_k, None)  # 逐个删除代理变量（不存在时不报错）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像

HERE = os.path.dirname(os.path.abspath(__file__))  # 本脚本所在目录（项目根）
sys.path.insert(0, HERE)  # 加入搜索路径，保证能 import 本目录下的模块

# sentence-transformers：Embedding 微调核心库；
# InputExample 包装训练样本，losses 提供对比损失，evaluation 提供检索评估器
from sentence_transformers import (
    SentenceTransformer, InputExample, losses, models, evaluation)
# DataLoader：按 batch 抽样训练数据（对比损失依赖 batch 内负例组织方式）
from torch.utils.data import DataLoader

BASE_MODEL = "BAAI/bge-small-zh-v1.5"  # 基座模型：中文小模型（384维），CPU 可训练
OUT_MODEL = os.path.join(HERE, "models", "bge-small-zh-v1.5-ft")  # 微调模型输出目录
DATA_DIR = os.path.join(HERE, "dataset")  # 数据集目录（train.jsonl / eval.jsonl）
EPOCHS = 3      # 训练轮数：3 轮在小数据集上足以收敛且不易过拟合
BATCH_SIZE = 16 # batch 大小：对比学习下 batch 越大负例越多、效果越好，但受 CPU 内存限制
LR = 2e-5       # 学习率：微调常用的小学习率，避免破坏预训练知识


def load_pairs(path):
    """读取 jsonl 格式的（query, 正例片段）训练/评估对"""
    pairs = []  # 收集所有样本对
    with open(path, encoding="utf-8") as f:
        for line in f:  # jsonl 每行一个 JSON 对象
            d = json.loads(line)  # 解析当前行
            pairs.append((d["query"], d["pos"]))  # 取 query 与正例片段组成二元组
    return pairs


def build_ir_evaluator(eval_pairs, all_corpus, name):
    """构建 InformationRetrievalEvaluator：query → 在候选语料中召回相关片段"""
    queries, relevant = {}, {}  # 查询集 与 每个查询对应的相关文档集合
    corpus = dict(all_corpus)   # 复制共享语料（评估器要求独占 dict）
    for i, (q, pos) in enumerate(eval_pairs):  # 遍历评估对，注册查询与正例
        cid = f"c{i}"           # 该正例在语料中的唯一 ID
        corpus[cid] = pos       # 正例片段加入候选语料
        queries[f"q{i}"] = q    # 注册查询
        relevant[f"q{i}"] = {cid}  # 标注该查询的唯一相关文档
        # 加同批其他正例作为难负例候选
    return evaluation.InformationRetrievalEvaluator(
        queries, corpus, relevant,          # 查询集、候选语料、相关标注
        name=name, show_progress_bar=False, # 评估器名称（决定指标键名前缀）；关进度条减少输出
        accuracy_at_k=[1, 3], precision_recall_at_k=[3])  # 算 Accuracy@1/3 与 P/R@3


def resolve_local_model():
    """优先用本地已下载模型目录（curl直接获取，绕开HF网络问题）"""
    local = os.path.join(HERE, "models", "base")  # 本地基座模型目录（hf-mirror 下载）
    if os.path.exists(os.path.join(local, "modules.json")):
        # modules.json 是 sentence-transformers 模型的标志文件，存在即视为完整模型
        return local
    return BASE_MODEL  # 本地没有则退回 HF 名称（联网下载）


def main():
    random.seed(42)  # 固定随机种子，保证打乱/训练结果可复现
    train_pairs = load_pairs(os.path.join(DATA_DIR, "train.jsonl"))  # 加载训练对
    eval_pairs = load_pairs(os.path.join(DATA_DIR, "eval.jsonl"))    # 加载评估对
    print(f"训练对 {len(train_pairs)}，评估对 {len(eval_pairs)}")

    # ── 模型加载 ─────────────────────────────────────────
    local = resolve_local_model()  # 解析实际使用的模型路径
    print(f"加载模型: {local}")
    model = SentenceTransformer(local)  # 加载 Embedding 模型（本地目录或 HF 名称）

    # ── 微调前评估 ───────────────────────────────────────
    # 语料：全部训练+评估的正例片段，键名 bg*（作为检索候选池）
    corpus = {f"bg{i}": pos for i, (_, pos) in enumerate(train_pairs + eval_pairs)}
    evaluator = build_ir_evaluator(eval_pairs, corpus, "ft_eval")  # 构建评估器
    print("\n===== 微调前评估 =====")
    before = evaluator(model)  # 微调前指标（dict：键名格式 {任务名}_{指标}@k）
    print(before)

    # ── 数据集与损失函数（对比损失：正例拉近、批内负例推远） ──
    # 每个训练对包装成 InputExample； MNRL 默认取 (anchor, positive) 两列
    train_examples = [InputExample(texts=[q, p]) for q, p in train_pairs]
    random.shuffle(train_examples)  # 打乱样本顺序（DataLoader 内还会再 shuffle，双保险）
    # DataLoader 按 batch=16 抽样：batch 内其他样本的正例互为负例（in-batch negatives）
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    # MultipleNegativesRankingLoss：对比学习损失，拉近正例对、推远批内其他样本对
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # ── 训练参数 ─────────────────────────────────────────
    # 预热步数 = 总步数的 10%（学习率从 0 线性升到 LR，防训练初期震荡）
    warmup_steps = max(1, int(len(train_dataloader) * EPOCHS * 0.1))
    print(f"\n===== 开始微调: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}, "
          f"warmup={warmup_steps} =====")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],  # (数据加载器, 损失函数) 对
        evaluator=evaluator,               # 训练中定期评估
        epochs=EPOCHS,                     # 训练 3 轮
        evaluation_steps=max(5, len(train_dataloader)),  # 每隔 N 步评估一次（至少跑满一轮）
        warmup_steps=warmup_steps,         # 学习率预热步数
        optimizer_params={"lr": LR},       # 优化器学习率 2e-5
        show_progress_bar=True,            # 显示训练进度条
        use_amp=False,  # CPU 不用 AMP
    )

    # ── 微调后评估 ───────────────────────────────────────
    print("\n===== 微调后评估 =====")
    after = evaluator(model)  # 同一评估器再跑一遍，得到微调后指标
    print(after)

    model.save(OUT_MODEL)  # 保存微调后模型到 models/bge-small-zh-v1.5-ft
    print(f"\n微调模型已保存: {OUT_MODEL}")

    # ── 前后对比报告 ─────────────────────────────────────
    def parse(score_dict, key):
        # 安全取值：键不存在时返回 0.0，避免 KeyError
        return score_dict.get(key, 0.0)

    lines = [
        "# 工单11：Embedding 微调前后评估对比",
        "",
        # 报告头部：记录模型、数据量与全部训练超参，保证可复现
        f"基座: {BASE_MODEL} | 训练对: {len(train_pairs)} | 评估对: {len(eval_pairs)} | "
        f"epochs={EPOCHS} lr={LR} batch={BATCH_SIZE} | 损失: MultipleNegativesRankingLoss(对比损失)",
        "",
        "| 指标 | 微调前 | 微调后 | 变化 |",
        "|---|---|---|---|",
    ]
    keys = [k for k in before.keys()]  # 以微调前的指标键为基准遍历
    for k in keys:
        b, a = before[k], after[k]  # 前后指标值
        lines.append(f"| {k} | {b:.4f} | {a:.4f} | {a-b:+.4f} |")  # 变化量带符号显示
    lines += ["", "> 注：键名格式为 {任务名}_{指标}@k，如 Cosmos_Similarity_P@1 / Recall@3。",
              "> 验收标准：微调后在评估集上的检索指标应整体优于微调前。"]
    # 写出 Markdown 对比报告
    with open(os.path.join(HERE, "微调前后评估对比.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n对比报告: 微调前后评估对比.md")


if __name__ == "__main__":
    main()  # 直接运行时执行主流程
