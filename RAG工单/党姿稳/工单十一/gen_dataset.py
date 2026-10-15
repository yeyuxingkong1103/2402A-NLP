# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
# 模块职责：工单步骤 1「数据集生成」与步骤 2「问答对生成」——用大模型从语料块批量生成领域问答对
# 使用方式：python gen_dataset.py
# 产出：data/qa_pairs.json（按块划分的训练集与评估集，避免同块泄漏）

import random                                                   # 导入 random 用于可复现采样与打乱
from concurrent.futures import ThreadPoolExecutor, as_completed  # 导入线程池用于并发调用大模型
import corpus                                                   # 导入语料与分块模块
import llm_client                                               # 导入大模型客户端
import config                                                   # 导入本工单配置模块

_SYSTEM = """你是金融年报领域的数据标注专家。你的任务是从给定的年报片段中提取事实，\
并据此编写高质量的问答对，用于训练金融领域的检索模型。必须遵守：
1. 问题和答案的所有信息必须严格来自给定片段，不得引入片段之外的知识，不得编造数字。
2. 问题要像真实用户的提问：用自然语言、指向明确，避免「这段文字说了什么」这类空泛问法。
3. 答案要简洁完整，涉及数字、比例、金额时必须原样引用片段中的数值。
4. 每题聚焦一个事实点；不要出判断题、不要出「是否」类问题。
5. 若片段中是表格或残缺文本、无法据此提出有意义的问题，返回空数组。"""

_USER = """请阅读下面的年报片段，生成 %d 个问答对。

【年报片段】（来源：%s 第%d页）
%s

只输出 JSON 数组，不要输出任何解释文字，格式严格如下：
[{"question": "问题", "answer": "答案"}, ...]"""


def _gen_one(idx, chunk):                                       # 定义单个语料块的问答生成函数
    """针对一个语料块调用大模型生成问答对，返回带元信息的问答列表。"""
    user = _USER % (config.QA_PER_CHUNK,                        # 期望生成的问答对数量
                    chunk["source"],                            # 来源短名
                    chunk["page"],                              # 来源页码
                    chunk["text"][:1200])                       # 片段正文（截断控制提示词长度）
    data = llm_client.chat_json(_SYSTEM, user,                 # 调用大模型并解析 JSON
                                temperature=0.3,                # 适度温度提升问题表达的多样性
                                max_tokens=1200)                # 输出长度上限
    if not isinstance(data, list):                              # 返回值不是数组（解析失败或无内容）
        return []                                               # 返回空列表
    items = []                                                  # 初始化结果列表
    for rec in data:                                            # 遍历模型返回的每条记录
        if not isinstance(rec, dict):                           # 跳过非字典项
            continue                                            # 继续下一条
        q = str(rec.get("question", "")).strip()                # 取出问题文本
        a = str(rec.get("answer", "")).strip()                  # 取出答案文本
        if len(q) < 6 or len(a) < config.QA_MIN_ANSWER_LEN:      # 过滤过短的无效样本
            continue                                            # 继续下一条
        items.append({"question": q,                            # 问题
                      "answer": a,                              # 答案
                      "chunk_id": chunk["id"],                  # 答案所在的语料块 ID（正例）
                      "source": chunk["source"],                # 来源短名，便于人工核验
                      "page": chunk["page"]})                   # 来源页码
    return items                                                # 返回该块生成的问答对


def generate(force=False):                                      # 定义问答对生成主函数
    """并发调用大模型生成问答对并落盘；已有缓存时直接复用。"""
    import os                                                   # 延迟导入 os 用于缓存判断
    config.ensure_dirs()                                        # 确保数据目录存在
    if os.path.isfile(config.QA_FILE) and not force:            # 缓存已存在且未强制刷新
        import json                                             # 延迟导入 json
        with open(config.QA_FILE, "r", encoding="utf-8") as f:  # 以 UTF-8 打开缓存
            saved = json.load(f)                                # 读取已保存的数据集
        print("[信息] 命中问答对缓存：训练 %d 条，评估 %d 条"     # 输出缓存命中提示
              % (len(saved["train"]), len(saved["eval"])))      # 填入统计数字
        return saved                                            # 直接返回缓存结果
    chunks = corpus.build_or_load_chunks()                      # 加载（或构建）语料分块
    rng = random.Random(config.RANDOM_SEED)                     # 用固定种子初始化随机数生成器
    pool = chunks[:]                                            # 复制一份候选池
    rng.shuffle(pool)                                           # 打乱候选顺序，保证采样无偏
    sampled = pool[:config.QA_CHUNK_SAMPLE]                     # 抽取指定数量的语料块
    print("[信息] 采样 %d 个语料块，开始并发生成问答对（%d 线程）"  # 输出开始提示
          % (len(sampled), config.LLM_WORKERS))                 # 填入采样数与线程数
    pairs = []                                                  # 初始化问答对总表
    with ThreadPoolExecutor(max_workers=config.LLM_WORKERS) as ex:  # 创建线程池
        futures = {ex.submit(_gen_one, i, c): c for i, c in enumerate(sampled)}  # 提交全部任务
        for n, fut in enumerate(as_completed(futures), 1):      # 按完成顺序收集结果
            try:                                                # 尝试取回结果
                pairs.extend(fut.result())                      # 合并该块生成的问答对
            except Exception as e:                              # 单块失败时容错
                print("[警告] 第 %d 个语料块生成失败：%r" % (n, e))  # 输出告警
            if n % 20 == 0:                                     # 每完成 20 个块打印一次进度
                print("[信息] 进度 %d/%d，已生成 %d 条问答对"      # 输出进度信息
                      % (n, len(sampled), len(pairs)))          # 填入进度数字
    pairs = _dedup(pairs)                                       # 去重，避免重复问题污染训练
    print("[信息] 问答对生成完成：%d 条" % len(pairs))            # 输出统计
    return _split(pairs, rng)                                   # 按块划分训练集与评估集并落盘


def _dedup(pairs):                                              # 定义去重函数
    """按问题文本去重，保留首次出现的样本；同一问题只保留一条。"""
    seen, out = set(), []                                       # 已见问题集合与输出列表
    for p in pairs:                                             # 遍历全部问答对
        key = p["question"].strip()                             # 以问题文本作为去重键
        if key in seen:                                         # 已出现过
            continue                                            # 跳过重复项
        seen.add(key)                                           # 记录该问题
        out.append(p)                                           # 加入输出列表
    return out                                                  # 返回去重后的列表


def _split(pairs, rng):                                         # 定义数据集划分函数
    """按语料块划分训练集与评估集：同一块只出现在一侧，杜绝答案泄漏。"""
    chunk_ids = sorted({p["chunk_id"] for p in pairs})          # 收集涉及的全部语料块 ID
    rng.shuffle(chunk_ids)                                      # 打乱块顺序，避免来源集中
    n_eval = max(1, int(len(chunk_ids) * config.EVAL_CHUNK_RATIO))  # 计算评估集占用的块数
    eval_ids = set(chunk_ids[:n_eval])                          # 前若干块划给评估集
    train = [p for p in pairs if p["chunk_id"] not in eval_ids]  # 其余块构成训练集
    evals = [p for p in pairs if p["chunk_id"] in eval_ids]     # 评估集问答对
    saved = {"train": train, "eval": evals}                     # 组装待落盘结构
    import json                                                 # 延迟导入 json
    with open(config.QA_FILE, "w", encoding="utf-8") as f:      # 以 UTF-8 打开数据集文件
        json.dump(saved, f, ensure_ascii=False, indent=1)       # 保留中文并缩进写入
    print("[信息] 数据集已落盘 %s" % config.QA_FILE)             # 输出落盘路径
    print("[信息] 训练集 %d 条（%d 个块），评估集 %d 条（%d 个块）"  # 输出划分统计
          % (len(train), len(chunk_ids) - n_eval, len(evals), n_eval))  # 填入各侧数量
    return saved                                                # 返回划分结果


if __name__ == "__main__":                                      # 仅在直接运行时执行
    import sys                                                  # 导入 sys 解析命令行参数
    generate(force="--force" in sys.argv)                       # 支持 --force 强制重新生成
