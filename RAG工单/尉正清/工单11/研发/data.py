# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
"""FiQA 数据集的加载、本地缓存与切分

只做 CPU 侧的字符串处理，不碰模型，方便单独跑和单测。
"""
import json
import os
import random

# datasets 会读 HF_ENDPOINT 决定去哪下载。国内直连 huggingface.co 会超时，
# 必须在 import datasets 之前把镜像地址设进环境变量。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from config import (CACHE_DIR, CORPUS_REPO, QRELS_REPO, SPLIT_SEED,
                    TRAIN_RATIO)

DATA_DIR = CACHE_DIR / "fiqa"


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_fiqa(force=False):
    """返回 (corpus, queries, qrels)。

    corpus  : [{"_id": str, "text": str}, ...]      57638 篇
    queries : [{"_id": str, "text": str}, ...]       6648 条
    qrels   : [{"query-id", "corpus-id", "score"}]   1706 条标注，覆盖 648 个 query

    首次跑会从 HF 镜像下载并落一份 JSONL 到本地 —— 之后离线也能跑，
    交付物里也就能带上一份可复现的数据快照。
    """
    c_path = DATA_DIR / "corpus.jsonl"
    q_path = DATA_DIR / "queries.jsonl"
    r_path = DATA_DIR / "qrels.jsonl"

    if not force and all(p.exists() for p in (c_path, q_path, r_path)):
        return _read_jsonl(c_path), _read_jsonl(q_path), _read_jsonl(r_path)

    from datasets import load_dataset

    corpus = load_dataset(CORPUS_REPO, "corpus", split="corpus")
    queries = load_dataset(CORPUS_REPO, "queries", split="queries")
    qrels = load_dataset(QRELS_REPO, split="test")

    corpus = [{"_id": str(r["_id"]), "text": r["text"]} for r in corpus]
    queries = [{"_id": str(r["_id"]), "text": r["text"]} for r in queries]
    # 只保留 score > 0 的标注：BEIR 的 qrels 里 score=0 表示明确判定为不相关，
    # 本工单的评估只关心正例，负例由检索结果自己产生。
    qrels = [{"query-id": str(r["query-id"]), "corpus-id": str(r["corpus-id"])}
             for r in qrels if r["score"] > 0]

    write_jsonl(c_path, corpus)
    write_jsonl(q_path, queries)
    write_jsonl(r_path, qrels)
    return corpus, queries, qrels


def with_instruction(text, instruction=None):
    """给查询加 BGE 指令前缀。文档侧不加。"""
    if instruction is None:
        from config import QUERY_INSTRUCTION as instruction
    return f"{instruction}{text}" if instruction else text


def relevance_map(qrels):
    """{query_id: {corpus_id, ...}} —— 每个 query 的相关文档集合。"""
    rel = {}
    for row in qrels:
        rel.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return rel


def split_queries(rel, seed=SPLIT_SEED, train_ratio=TRAIN_RATIO):
    """按 **query** 切分训练集与评估集。

    必须按 query 切，不能按标注行切：同一个 query 的多个正例若被分到两边，
    评估集里就有了训练时见过的答案，指标会虚高。这是这类任务最容易踩的坑。
    """
    ids = sorted(rel)
    rng = random.Random(seed)
    rng.shuffle(ids)
    cut = int(len(ids) * train_ratio)
    return sorted(ids[:cut]), sorted(ids[cut:])


def split_train_dev(train_qids, dev_ratio=0.1, seed=SPLIT_SEED):
    """从训练 query 里再切一份 dev 出来。

    取自**训练 query**（不是评估 query），评估集始终不参与任何训练决策。

    ⚠️ 它原本是为「用 dev loss 判断是否过拟合、及时早停」而切的，但那条路
    后来被证伪了：只用 batch 内负例时 dev loss 从第一次评估起就单调上涨，
    与检索质量脱钩（见 优化/过程问题记录.md 问题 10）。本工单的正式训练
    因此关掉了早停。保留 dev 是为了**监控训练是否发散**，不是为了选模型。
    """
    ids = sorted(train_qids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    cut = int(len(ids) * dev_ratio)
    return sorted(ids[cut:]), sorted(ids[:cut])


def build_triples(rel, queries, hard_negs):
    """拼成 (anchor, positive, negative) 三元组。

    hard_negs 由 build_dataset.mine_hard_negatives() 产出，形如
    {query_id: [corpus_id, ...]}，是检索排名靠前但不相关的文档。

    ⚠️ 这里对「正例 × 难负例」做笛卡尔积，所以一个 (anchor, positive) 会有
    HARD_NEG_PER_QUERY 行。**不用难负例训练时必须先去重**（train_data._dedupe），
    否则每份数据会被重复看 4 遍 —— 详见 优化/过程问题记录.md 问题 8。
    """
    qtext = {q["_id"]: q["text"] for q in queries}
    triples = []
    for qid in sorted(rel):
        raw = qtext.get(qid)
        if not raw:
            continue
        # 训练时 anchor 也带指令前缀，与推理口径一致（见 config.QUERY_INSTRUCTION）
        anchor = with_instruction(raw)
        for pos in sorted(rel[qid]):
            for neg in hard_negs.get(qid, []):
                triples.append({"query_id": qid, "anchor": anchor,
                                "positive_id": pos, "negative_id": neg})
    return triples


