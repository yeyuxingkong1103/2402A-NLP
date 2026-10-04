# -*- coding: utf-8 -*-
"""
工单11 步骤3：微调前后模型评估对比（验收指标）
工单编号：人工智能NLP-RAG 项目-Embedding模型微调任务
功能：分别加载基座模型与微调后模型，在评估集上计算 Recall@1/3、MRR，
     输出《微调前后评估对比.md》（验收要求：微调后检索效果更好，有数据支撑）。
运行：python eval_compare.py
"""
import os  # 标准库：路径拼接
import json  # 标准库：读取 jsonl 评估集
import numpy as np  # 数值计算：向量编码结果的矩阵运算与排序

HERE = os.path.dirname(os.path.abspath(__file__))  # 本脚本所在目录
from sentence_transformers import SentenceTransformer  # Embedding 模型加载与编码

BASE = os.path.join(HERE, "models", "base")  # 基座模型本地目录（hf-mirror 下载）
FT = os.path.join(HERE, "models", "bge-small-zh-v1.5-ft")  # 微调后模型目录（finetune.py 产出）


def load_pairs(path):
    """读取 jsonl：每行一个 JSON，返回 dict 列表"""
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def evaluate(model, eval_pairs, distractors, name):
    """query 在 (正例+干扰片段) 语料中做余弦检索，计算 Recall@1/3 与 MRR"""
    # 语料：每个评估对的正例 + 共享干扰块
    corpus, relevant = {}, {}  # corpus：候选文档池；relevant：查询→相关文档标注
    for i, item in enumerate(eval_pairs):  # 注册每个评估对的正例
        corpus[f"pos{i}"] = item["pos"]          # 正例片段入语料池
        relevant[f"q{i}"] = {"pos{i}"}           # 查询 q{i} 的唯一相关文档为 pos{i}
    for j, d in enumerate(distractors):  # 注册干扰块（同领域难负例）
        corpus[f"neg{j}"] = d

    cids = list(corpus.keys())  # 语料文档 ID 列表（与下标对应，便于后续取回）
    # 编码语料：normalize_embeddings=True 归一化后点积即余弦相似度；batch=32 控制内存
    cvecs = model.encode([corpus[c] for c in cids], normalize_embeddings=True,
                         show_progress_bar=False, batch_size=32)
    # 编码全部评估 query（同样归一化）
    qvecs = model.encode([p["query"] for p in eval_pairs], normalize_embeddings=True,
                         show_progress_bar=False, batch_size=32)
    # 相似度矩阵：shape=(query数, 文档数)，归一化向量的点积即余弦相似度
    sim = qvecs @ np.array(cvecs).T

    r1 = r3 = 0.0  # Recall@1 与 Recall@3 的累计命中计数
    mrr = 0.0      # MRR（平均倒数排名）的累计值
    for i, item in enumerate(eval_pairs):
        order = np.argsort(-sim[i])  # 按相似度降序排列文档下标（取负后升序=原值降序）
        ranked = [cids[k] for k in order]  # 排序后的文档 ID 序列
        target = f"pos{i}"  # 该查询的目标正例 ID
        rank = ranked.index(target) + 1  # 正例在检索结果中的排名（1 起始）
        r1 += rank == 1   # 排名第 1 则 Recall@1 命中
        r3 += rank <= 3   # 排名前 3 则 Recall@3 命中
        mrr += 1.0 / rank  # MRR 累加倒数排名
    n = len(eval_pairs)  # 评估查询总数
    # 返回平均指标（踩坑：sentence-transformers 内置 evaluator 返回 numpy.float64
    # 且结构非标准 dict 无 .keys()，故这里直接手算指标，结果就是普通 float）
    return {"Recall@1": r1 / n, "Recall@3": r3 / n, "MRR": mrr / n}


def main():
    eval_pairs = load_pairs(os.path.join(HERE, "dataset", "eval.jsonl"))   # 评估集
    train_pairs = load_pairs(os.path.join(HERE, "dataset", "train.jsonl")) # 训练集（取干扰块用）
    # 干扰块：训练集正例（同领域难负例，让评估更严格）
    distractors = [p["pos"] for p in train_pairs[:60]]
    print(f"评估对 {len(eval_pairs)}，语料规模 {len(eval_pairs)+len(distractors)}")

    rows = []  # 收集各模型评估结果
    # 依次评估：基座模型 → 微调后模型（路径不同，其余流程一致）
    for name, path in [("微调前(基座 bge-small-zh-v1.5)", BASE),
                       ("微调后(bge-small-zh-v1.5-ft)", FT)]:
        model = SentenceTransformer(path)  # 加载当前模型
        m = evaluate(model, eval_pairs, distractors, name)  # 在同一评估集上算指标
        rows.append((name, m))  # 记录（模型名, 指标dict）
        print(name, m)

    L = ["# 工单11：Embedding 微调前后评估对比",  # Markdown 报告行缓冲
         "",
         # 报告头部：模型、数据来源、损失函数与超参
         "基座: BAAI/bge-small-zh-v1.5 | 数据: LLM生成的招股书问答对 "
         f"（训练{len(train_pairs)}对/评估{len(eval_pairs)}对） | 损失: MultipleNegativesRankingLoss(对比损失) | "
         "epochs=3, lr=2e-5, batch=16",
         "",
         "评估方式：query 在（评估正例+60个同领域干扰块）语料中做余弦检索。",
         "",
         "| 指标 | 微调前 | 微调后 | 变化 |",
         "|---|---|---|---|"]
    # 以第一个模型（基座）的指标键为序遍历，逐行生成对比表格
    for k in rows[0][1]:
        b, a = rows[0][1][k], rows[1][1][k]  # 基座指标 / 微调后指标
        L.append(f"| {k} | {b:.4f} | {a:.4f} | {a-b:+.4f} |")  # 变化量带符号
    # 验收结论：以 MRR 为综合判据，微调后更高则达标
    verdict = "✅ 微调后检索效果优于微调前，满足验收标准（有数据指标支撑）" \
        if rows[1][1]["MRR"] > rows[0][1]["MRR"] else "⚠️ 微调后未提升，需调整训练参数/数据量"
    L += ["", f"**验收结论：{verdict}**"]
    out = os.path.join(HERE, "微调前后评估对比.md")  # 报告输出路径
    open(out, "w", encoding="utf-8").write("\n".join(L))  # 写出 Markdown
    print("\n报告:", out)


if __name__ == "__main__":
    main()  # 直接运行时执行评估对比流程
