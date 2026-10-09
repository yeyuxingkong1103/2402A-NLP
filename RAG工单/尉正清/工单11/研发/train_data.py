# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
"""训练侧的两块：数据管道 + 损失函数

- 数据管道：读三元组 → 去重 → 切 dev → DataLoader
- 损失函数：MultipleNegativesRankingLoss（MNR）及其取句向量的辅助函数

单独拆出来是因为 `finetune.py` 超过 300 行上限，而这两块正好是自洽的一层：
它们只依赖 config 和 data，不碰训练循环、早停、保存那些流程控制。

**两个可调开关放在这里而不是 config 里**：难负例与温度到底取哪个值要靠实验定，
正式训练用命令行覆盖（`--use-hard` / `--temperature`），见 `configure()`。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader

import data as D
from config import USE_HARD_NEGATIVES

OUT = D.DATA_DIR
TEMPERATURE = 0.02          # MNR 的标准缩放系数，等价于 ST 里 MNRL 的 scale=50


def configure(use_hard=None, temperature=None):
    """覆盖下面两个开关。传 None 表示保持不动。

    为什么不做成函数参数一路传下去：它们要在 `batch_loss`、`mnr_loss`、
    `_triples_to_dataset`、`build_loaders` 四个地方用到，一路透传只会让
    每个函数的签名都多两个跟它自己逻辑无关的参数。
    """
    global USE_HARD_NEGATIVES, TEMPERATURE
    if use_hard is not None:
        USE_HARD_NEGATIVES = use_hard
    if temperature is not None:
        TEMPERATURE = temperature


# ---------------- 数据管道 ----------------
def collate(batch):
    """按 batch 里实际存在的列拼。不用难负例时没有 negative 列，硬取会 KeyError。"""
    keys = [k for k in ("anchor", "positive", "negative") if k in batch[0]]
    return {k: [b[k] for b in batch] for k in keys}


def _dedupe(rows):
    """按 (query_id, positive_id) 去重。

    build_dataset.py 按「正例 × 难负例」做笛卡尔积，一个 (anchor, positive)
    会对应 HARD_NEG_PER_QUERY 行负例各异的记录。训练时不用难负例，
    这些行就是**内容完全相同的样本**，留着等于把每份数据重复看 4 遍：
    4720 行里只有 1180 组唯一样本，名义上 4 个 epoch 实际是 16 遍 ——
    实测这正是「训练越久、检索指标越低」的原因（见 优化/过程问题记录.md 问题 8）。

    顺带还有个更隐蔽的副作用：dev 集不过 shuffle，同一组重复样本挤在同一个
    batch 里，MNR 会把**内容一模一样**的两条文档当成互为负例，dev loss 因此
    失去意义 —— 它一路上涨并非过拟合，而是被这种自相矛盾的约束顶起来的
    （去重之后它仍然上涨，另有原因，见问题 10）。
    """
    seen, uniq = set(), []
    for r in rows:
        key = (r["query_id"], r["positive_id"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


def _triples_to_dataset(corpus, rows):
    """把三元组转成 datasets.Dataset。

    不用难负例时不建 negative 列 —— 多一列就多几百 KB 的文本要过 tokenizer，
    纯属浪费。
    """
    text = {c["_id"]: c["text"] for c in corpus}
    data = {
        "anchor":   [r["anchor"] for r in rows],
        "positive": [text[r["positive_id"]] for r in rows],
    }
    if USE_HARD_NEGATIVES:
        data["negative"] = [text[r["negative_id"]] for r in rows]
    return Dataset.from_dict(data)


def build_loaders(corpus, batch, dev_qids, use_gen=False):
    """返回 (train_loader, dev_loader, n_train, n_dev)。

    dev 按 query 划分：dev_qids 里的 query 对应的三元组全部进 dev，
    训练集里一条都不留。

    use_gen=True 时把 gen_qa.py 生成的问答对并进训练集。它们的 id 形如 gen123，
    不会落在 dev_qids 里，所以只进训练。生成时已排除评估集的标准答案文档，
    不存在污染。
    """
    rows = D._read_jsonl(OUT / "train_triples.jsonl")
    # 只在**不用**难负例时去重：用难负例时那 4 行负例各异，是 4 条不同的训练
    # 样本，去重会把每个正例的难负例砍到只剩 1 个。
    if not USE_HARD_NEGATIVES:
        rows = _dedupe(rows)
    if use_gen:
        gen_path = OUT / "gen_pairs.jsonl"
        if not gen_path.exists():
            raise SystemExit(f"[错误] {gen_path} 不存在，先跑 python gen_qa.py")
        gen = D._read_jsonl(gen_path)
        rows = rows + [{"query_id": f"gen{i}", "anchor": D.with_instruction(g["query"]),
                        "positive_id": g["positive_id"]} for i, g in enumerate(gen)]
        print(f"  （并入 {len(gen):,} 条生成的问答对）")

    dev_set = set(dev_qids)
    tr = [r for r in rows if r["query_id"] not in dev_set]
    dv = [r for r in rows if r["query_id"] in dev_set]
    tr_ds, dv_ds = _triples_to_dataset(corpus, tr), _triples_to_dataset(corpus, dv)
    return (DataLoader(tr_ds, batch_size=batch, shuffle=True, collate_fn=collate),
            DataLoader(dv_ds, batch_size=batch, shuffle=False, collate_fn=collate),
            len(tr), len(dv))


# ---------------- 损失函数 ----------------
def embed(model, texts):
    """取句向量并归一化。

    用 preprocess 而不是 tokenize：后者在 sentence-transformers 5.x 已废弃。
    返回的 BatchEncoding 里除了张量还有个字符串字段 modality，只能搬张量。
    """
    feats = model.preprocess(texts)
    feats = {k: (v.to(model.device) if torch.is_tensor(v) else v)
             for k, v in feats.items()}
    out = model(feats)["sentence_embedding"]
    return F.normalize(out, p=2, dim=1)


def batch_loss(model, batch):
    """按配置算一个 batch 的 loss。

    use_hard=False 时**不计算** negative 的前向 —— 算了也不用，白白多占一份
    计算图和显存（实测这一步会直接把 6GB 的卡撑爆）。
    """
    a = embed(model, batch["anchor"])
    p = embed(model, batch["positive"])
    if USE_HARD_NEGATIVES:
        return mnr_loss(a, p, embed(model, batch["negative"]))
    return mnr_loss(a, p)


def mnr_loss(anchor, positive, negative=None, use_hard=None):
    """MultipleNegativesRankingLoss。

    候选集默认只取本 batch 的所有 positive（共 B 个）：
    第 i 行的正确答案是第 i 个 positive，其余 B-1 个是 batch 内负例 ——
    它们来自别的问题，跟本 query 几乎不可能相关，不存在假负例问题。

    use_hard=True 时把外挂的难负例也拼进候选集（共 2B 个）。
    在 FiQA 上这样做**是有害的**（标注太稀疏，难负例里混着大量假负例），
    原因见 config.USE_HARD_NEGATIVES 与 优化/过程问题记录.md 问题 7。
    """
    if use_hard is None:
        use_hard = USE_HARD_NEGATIVES      # 调用时才解析，--use-hard 才能覆盖
    if use_hard and negative is not None:
        candidates = torch.cat([positive, negative], dim=0)   # 2B × d
    else:
        candidates = positive                                 # B × d
    scores = anchor @ candidates.t() / TEMPERATURE
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(scores, labels)
