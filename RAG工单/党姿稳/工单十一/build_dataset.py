# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
# 模块职责：工单步骤 3「数据集与模型加载」——加载问答对数据集与基础嵌入模型，挖掘难负例构造三元组
# 使用方式：python build_dataset.py
# 产出：data/triplets.json（(锚点, 正例, 负例) 三元组数据集，供 Triplet Loss 训练使用）

import os                                                       # 导入 os 用于缓存文件判断
import json                                                     # 导入 json 用于数据集读写
import numpy as np                                              # 导入 numpy 用于相似度计算
import corpus                                                   # 导入语料与分块模块
import embedder                                                 # 导入嵌入模型加载与编码模块
import config                                                   # 导入本工单配置模块


def load_qa_pairs():                                            # 定义问答对数据集加载函数
    """读取 data/qa_pairs.json，返回训练与评估两个划分。"""
    if not os.path.isfile(config.QA_FILE):                      # 数据集文件不存在
        raise RuntimeError("问答对数据集不存在，请先运行 gen_dataset.py")  # 抛出明确错误提示
    with open(config.QA_FILE, "r", encoding="utf-8") as f:      # 以 UTF-8 打开数据集
        data = json.load(f)                                     # 解析为字典
    print("[信息] 加载问答对数据集：训练 %d 条，评估 %d 条"        # 输出加载统计
          % (len(data["train"]), len(data["eval"])))            # 填入两侧数量
    return data["train"], data["eval"]                          # 返回训练集与评估集


def load_resources():                                           # 定义资源加载函数
    """加载语料块、基础模型的语料向量缓存与基础嵌入模型，供难负例挖掘使用。"""
    chunks = corpus.build_or_load_chunks()                      # 加载（或构建）语料分块
    model = embedder.load_model()                               # 加载基础嵌入模型（工单要求的「模型加载」）
    vecs = embedder.encode_chunks(model, chunks,               # 加载或构建语料向量缓存
                                  config.BASE_EMBED_CACHE)      # 缓存路径取基础模型向量文件
    return chunks, model, vecs                                 # 返回语料块、模型与语料向量


def _is_valid_negative(neg, pos, neg_vec, pos_vec):             # 定义难负例有效性判定函数
    """排除与正例同源同页或高度相似的块，避免把近似重复当成负例。"""
    if neg["id"] == pos["id"]:                                  # 与正例是同一块
        return False                                            # 无效
    same_page = (neg["source"] == pos["source"]             # 与正例同源
                 and abs(neg["page"] - pos["page"]) <= 1)       # 且页码相邻（分块重叠导致）
    if same_page:                                               # 属于同页邻块
        return False                                            # 无效，视为同一事实的不同切法
    return float(np.dot(neg_vec, pos_vec)) < 0.95               # 与正例过于相似的也剔除


def mine_triplets(train_pairs, chunks, model, chunk_vecs):      # 定义难负例挖掘函数
    """用基础模型检索，把「排序靠前但与正例无关」的块作为难负例，构造三元组。"""
    id2idx = {c["id"]: i for i, c in enumerate(chunks)}         # 建立块 ID 到向量行号的映射
    questions = [p["question"] for p in train_pairs]            # 收集全部锚点问题
    q_vecs = embedder.encode(model, questions,                 # 编码问题（查询侧加 m3e 官方前缀）
                             batch_size=64, is_query=True, show=True)  # 显示进度条
    triplets = []                                               # 初始化三元组列表
    skipped = 0                                                 # 统计被跳过的样本数
    for p, qv in zip(train_pairs, q_vecs):                      # 遍历每个问答对及其查询向量
        pos_idx = id2idx.get(p["chunk_id"])                     # 定位正例块在矩阵中的行号
        if pos_idx is None:                                     # 正例块不在语料中（异常情况）
            skipped += 1                                        # 计数跳过
            continue                                            # 跳过该样本
        scores = chunk_vecs.dot(qv)                             # 计算该查询与全部语料块的余弦相似度
        order = np.argsort(-scores)                             # 按相似度降序得到块行号序列
        pos_vec = chunk_vecs[pos_idx]                           # 取出正例向量用于相似度过滤
        negs = []                                               # 本锚点选中的负例列表
        for idx in order[:config.HARD_NEG_POOL]:                # 在排名前 HARD_NEG_POOL 的块中挑选
            if idx == pos_idx:                                  # 跳过正例本身
                continue                                        # 继续下一个候选
            cand = chunks[int(idx)]                             # 取出候选块
            if _is_valid_negative(cand, chunks[pos_idx],        # 校验其是否为有效负例
                                  chunk_vecs[int(idx)], pos_vec):  # 传入两者向量
                negs.append(cand["text"])                       # 记录负例正文
            if len(negs) >= config.NEG_PER_ANCHOR:              # 达到所需负例数量
                break                                           # 停止挑选
        if len(negs) < config.NEG_PER_ANCHOR:                   # 有效负例不足
            skipped += 1                                        # 计数跳过
            continue                                            # 放弃该样本
        triplets.append({"anchor": p["question"],               # 锚点：用户问题
                         "positive": chunks[pos_idx]["text"],   # 正例：答案所在语料块
                         "negative": negs[0],                   # 负例：检索排名靠前的干扰块
                         "pos_id": p["chunk_id"],               # 正例块 ID，便于溯源
                         "neg_rank": 1})                        # 负例在检索结果中的名次（越靠前越难）
    print("[信息] 三元组构造完成：%d 条，跳过 %d 条"              # 输出统计
          % (len(triplets), skipped))                           # 填入数量
    return triplets                                             # 返回三元组列表


def save_triplets(triplets):                                    # 定义三元组落盘函数
    """把三元组写入 data/triplets.json。"""
    config.ensure_dirs()                                        # 确保数据目录存在
    with open(config.TRIPLET_FILE, "w", encoding="utf-8") as f:  # 以 UTF-8 打开输出文件
        json.dump(triplets, f, ensure_ascii=False, indent=1)    # 保留中文并缩进写入
    print("[信息] 三元组已落盘 %s" % config.TRIPLET_FILE)         # 输出落盘路径


def build(force=False):                                         # 定义三元组构建主函数
    """完整的步骤 3 流程：加载数据集与模型 → 挖掘难负例 → 落盘三元组。"""
    if os.path.isfile(config.TRIPLET_FILE) and not force:       # 缓存存在且未强制刷新
        with open(config.TRIPLET_FILE, "r", encoding="utf-8") as f:  # 以 UTF-8 打开缓存
            saved = json.load(f)                                # 读取已保存的三元组
        print("[信息] 命中三元组缓存：%d 条" % len(saved))         # 输出缓存命中提示
        return saved                                            # 直接返回缓存结果
    train_pairs, _ = load_qa_pairs()                            # 加载问答对训练集
    chunks, model, chunk_vecs = load_resources()                # 加载语料、模型与语料向量
    triplets = mine_triplets(train_pairs, chunks, model, chunk_vecs)  # 挖掘难负例
    save_triplets(triplets)                                     # 落盘
    return triplets                                             # 返回三元组


if __name__ == "__main__":                                      # 仅在直接运行时执行
    import sys                                                  # 导入 sys 解析命令行参数
    build(force="--force" in sys.argv)                          # 支持 --force 强制重新构建
