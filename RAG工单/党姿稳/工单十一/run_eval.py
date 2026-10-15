# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
# 模块职责：工单步骤 7「微调前评估模型」与步骤 8「微调后评估模型」，并输出对比结论
# 使用方式：python run_eval.py
# 产出：data/eval_result.json（微调前后指标）、控制台对比表

import os                                                       # 导入 os 用于路径与文件判断
import json                                                     # 导入 json 用于结果读写
import numpy as np                                              # 导入 numpy 用于向量处理
import corpus                                                   # 导入语料与分块模块
import embedder                                                 # 导入嵌入模型加载与编码模块
import evaluator                                                # 导入检索效果评估器
import config                                                   # 导入本工单配置模块


def load_eval_pairs():                                          # 定义评估集加载函数
    """读取问答对数据集中的评估划分。"""
    if not os.path.isfile(config.QA_FILE):                      # 数据集文件不存在
        raise RuntimeError("问答对数据集不存在，请先运行 gen_dataset.py")  # 抛出明确错误
    with open(config.QA_FILE, "r", encoding="utf-8") as f:      # 以 UTF-8 打开数据集
        data = json.load(f)                                     # 解析为字典
    pairs = data["eval"]                                        # 取出评估集
    print("[信息] 评估集 %d 条（覆盖 %d 个语料块）"                # 输出评估集规模
          % (len(pairs), len({p["chunk_id"] for p in pairs})))  # 填入数量与块数
    return pairs                                                # 返回评估集


def eval_before(chunks, pairs):                                 # 定义微调前评估函数（工单步骤 7）
    """用基础模型评估检索效果，直接复用已缓存的语料向量，避免重复编码。"""
    model = embedder.load_model(config.EMBED_MODEL_PATH)        # 加载基础嵌入模型
    vecs = embedder.encode_chunks(model, chunks, config.BASE_EMBED_CACHE)  # 读取基础模型语料向量缓存
    ev = evaluator.RetrievalEvaluator(chunks, vecs)             # 创建评估器
    return ev.evaluate(model, pairs, tag="微调前（m3e-base）")    # 执行评估并返回结果


def eval_after(chunks, pairs):                                  # 定义微调后评估函数（工单步骤 8）
    """用微调后的模型评估检索效果，语料向量单独缓存，与微调前使用同一候选池与同一评估集。"""
    if not os.path.isdir(config.FINETUNED_MODEL_DIR):           # 微调模型目录不存在
        raise RuntimeError("微调模型不存在，请先运行 train_model.py")  # 抛出明确错误
    cache = os.path.join(config.DATA_DIR, "embeddings_finetuned.npy")  # 微调模型的语料向量缓存路径
    model = embedder.load_model(config.FINETUNED_MODEL_DIR)     # 加载微调后的模型
    vecs = embedder.encode_chunks(model, chunks, cache)         # 编码（或读取缓存）语料向量
    ev = evaluator.RetrievalEvaluator(chunks, vecs)             # 创建评估器（候选池与微调前一致）
    return ev.evaluate(model, pairs, tag="微调后（m3e-base-finance）")  # 执行评估并返回结果


def case_study(chunks, pairs, limit=3):                         # 定义案例对照函数
    """对若干评估问题，输出微调前后检索到的前 3 条来源，便于人工核验改善情况。"""
    from sentence_transformers import util                      # 延迟导入距离工具
    model_b = embedder.load_model(config.EMBED_MODEL_PATH)      # 基础模型
    model_a = embedder.load_model(config.FINETUNED_MODEL_DIR)   # 微调模型
    vecs_b = np.load(config.BASE_EMBED_CACHE)                   # 基础模型语料向量
    vecs_a = np.load(os.path.join(config.DATA_DIR, "embeddings_finetuned.npy"))  # 微调模型语料向量
    id2idx = {c["id"]: i for i, c in enumerate(chunks)}         # 块 ID 到行号映射
    samples = pairs[:limit]                                     # 取前若干条评估问题
    qs = [p["question"] for p in samples]                       # 收集问题文本
    qb = embedder.encode(model_b, qs, is_query=True)            # 基础模型编码查询
    qa = embedder.encode(model_a, qs, is_query=True)            # 微调模型编码查询
    out = []                                                    # 初始化案例列表
    for i, p in enumerate(samples):                             # 遍历每个样例
        target = id2idx.get(p["chunk_id"])                      # 正确块行号
        row = {"question": p["question"],                       # 问题
               "answer_chunk": "%s 第%d页" % (p["source"], p["page"]),  # 正确来源
               "before": _top3(vecs_b, qb[i], chunks, target),  # 微调前前 3 条
               "after": _top3(vecs_a, qa[i], chunks, target)}   # 微调后前 3 条
        out.append(row)                                         # 收集该案例
    return out                                                  # 返回案例列表


def _top3(vecs, qv, chunks, target):                            # 定义前 3 条来源提取函数
    """取检索前 3 条的来源标注，命中正确块时用【命中】标记。"""
    order = np.argsort(-vecs.dot(qv))[:3]                       # 按相似度降序取前 3
    out = []                                                    # 初始化结果列表
    for r, idx in enumerate(order):                             # 遍历前 3 条
        c = chunks[int(idx)]                                    # 取出对应语料块
        mark = "【命中】" if int(idx) == target else ""           # 命中正确块时加标记
        out.append("#%d %s 第%d页%s" % (r + 1, c["source"],     # 组装展示文本
                                        c["page"], mark))       # 含名次、来源与命中标记
    return out                                                  # 返回前 3 条说明


def run():                                                      # 定义评估主函数
    """执行微调前评估 → 微调后评估 → 对比 → 落盘。"""
    chunks = corpus.build_or_load_chunks()                      # 加载语料分块（候选池）
    pairs = load_eval_pairs()                                   # 加载评估集
    before = eval_before(chunks, pairs)                         # 步骤 7：微调前评估
    after = eval_after(chunks, pairs)                           # 步骤 8：微调后评估
    cmp_result = evaluator.compare(before, after)               # 对比两次评估
    cases = case_study(chunks, pairs)                           # 生成案例对照
    result = {"before": before,                                 # 微调前结果
              "after": after,                                   # 微调后结果
              "compare": cmp_result,                            # 对比结论
              "cases": cases,                                   # 案例对照
              "corpus_chunks": len(chunks),                     # 候选池规模
              "eval_size": len(pairs)}                          # 评估集规模
    with open(config.EVAL_RESULT_FILE, "w", encoding="utf-8") as f:  # 以 UTF-8 打开结果文件
        json.dump(result, f, ensure_ascii=False, indent=1)      # 写入评估结果
    _print_table(cmp_result)                                    # 打印对比表
    print("[信息] 评估结果已落盘 %s" % config.EVAL_RESULT_FILE)   # 输出落盘路径
    return result                                               # 返回评估结果


def _print_table(cmp_result):                                   # 定义对比表打印函数
    """把指标对比结果打印为对齐的文本表格。"""
    print("\n================ 微调前后检索效果对比 ================")  # 打印表头
    print("%-12s %10s %10s %10s %s" % ("指标", "微调前", "微调后", "变化", "结论"))  # 打印列名
    for r in cmp_result["rows"]:                                # 遍历每一行
        print("%-12s %10s %10s %+10s %s" % (r["metric"], r["before"],  # 依次打印指标名与前值
                                            r["after"], r["delta"],  # 后值与变化量
                                            "改善" if r["better"] else "未改善"))  # 打印结论
    print("-----------------------------------------------------")  # 打印分隔线
    print("改善指标：%d/%d" % (cmp_result["improved"], cmp_result["total"]))  # 打印汇总


if __name__ == "__main__":                                      # 仅在直接运行时执行
    run()                                                      # 启动微调前后评估
