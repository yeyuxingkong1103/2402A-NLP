# 工单编号：人工智能NLP-RAG-混合检索任务
# 文件说明：扩样召回率评估（114 题）。只跑 `LLMContextRecall` 单项——
#           该指标只需要 question/reference/contexts，**不需要生成答案**，因此省掉全部生成开销；
#           同时输出**确定性召回**（expect 是否出现在 top-8）作为对照口径，两者含义不同、不可合并。
# 用法：D:/myconda1/python.exe tests/eval_recall_expanded.py
import collections, json, sys, time                                                   # 标准库
from pathlib import Path                                                              # 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))                       # 允许 import app 包
from app import config as C, eval_ragas as E, rag_core as R                           # 复用配置、判官与检索

TOP_K = 8                                                                             # 与线上默认一致
ROUNDS = 2                                                                            # 跑两轮，看判官噪声对均值的扰动


def load_questions() -> list:                                                         # 读 114 题
    """合并 tests/questions100.json 的两个子集，并统一 question 字段。"""
    data = json.loads((C.ROOT / "tests" / "questions100.json").read_text(encoding="utf-8"))  # 读文件
    rows = list(data["subset_A_natural"]) + list(data["subset_B_probe"])               # 子集A + 子集B
    for q in rows:                                                                    # 统一字段
        q.setdefault("type", "自然题(手工)")                                            # 手工题无 type
        q["question"] = q.get("question_zh") or q["question"]                          # 题库里有中/英两版
    return rows                                                                       # 返回


def retrieve_all(rows: list) -> list:                                                 # 逐题检索
    """只做检索、不生成答案；返回带 contexts 的样本描述。"""
    R.bm25_load(); R.embed(["预热"]); R.rerank_model()                                 # 预热，避免首题承担加载开销
    out = []                                                                          # 结果
    for i, q in enumerate(rows, 1):                                                   # 逐题
        cands, _ = R.retrieve(q["question"], "hybrid", TOP_K, rerankers=C.DEFAULT_RERANKERS)  # 默认配置检索
        ctx = [c.get("text") or "" for c in cands]                                     # 上下文
        det = all(e.replace(",", "") in " ".join(ctx).replace(",", "")                 # 确定性召回：关键字段是否出现
                  for e in q.get("expect", []))                                        # （去掉千分位再比）
        out.append({"id": f"x{i:03d}", "subset": "手工自然" if i <= 23 else "扩样探针",   # 子集（A 在前 23 条）
                    "type": q["type"], "question": q["question"],                      # 题型与问句
                    "reference": q["ground_truth"], "contexts": ctx, "det": det})       # 参考、上下文、确定性召回
    return out                                                                        # 返回


def main() -> int:                                                                    # 返回退出码
    """检索一次 → 跑 ROUNDS 轮 context_recall → 按子集与题型打印。"""
    rows = load_questions()                                                           # 读题
    print(f"题数 {len(rows)}", flush=True)                                             # 打印
    t0 = time.time(); samples = retrieve_all(rows)                                    # 检索
    print(f"检索完成，用时 {time.time() - t0:.0f}s", flush=True)                        # 打印
    from ragas import EvaluationDataset, SingleTurnSample, evaluate                    # ragas 0.4 API
    from ragas.metrics import LLMContextRecall                                         # 只测这一个指标
    from ragas.run_config import RunConfig                                             # 并发控制
    ds = EvaluationDataset([SingleTurnSample(user_input=s["question"],                # 数据集
                                             response="（仅测检索，不生成答案）",         # 该指标不需要答案
                                             reference=s["reference"],                # 标准答案
                                             retrieved_contexts=s["contexts"] or ["（无检索结果）"])
                           for s in samples])                                          # 覆盖全部题
    llm, rounds = E._judge(), []                                                       # 判官与逐轮结果
    for r in range(ROUNDS):                                                            # 逐轮
        t = time.time()                                                                # 计时
        res = evaluate(ds, metrics=[LLMContextRecall()], llm=llm, embeddings=E._embeddings(),  # 单指标评估
                       run_config=RunConfig(max_workers=C.EVAL_MAX_WORKERS,             # 默认 16 并发会打满限流
                                            timeout=180, max_retries=3),                # 超时与重试
                       show_progress=False, raise_exceptions=False)                     # 单条失败不中断整体
        col = res.to_pandas()["context_recall"]                                        # 逐题得分
        vals = [None if x != x else float(x) for x in col]                             # nan 显式转 None
        rounds.append(vals)                                                            # 收集
        ok = [x for x in vals if x is not None]                                        # 有效值
        print(f"第{r + 1}轮 context_recall = {sum(ok) / len(ok):.4f}  "                 # 均值
              f"有效 {len(ok)}/{len(vals)}  用时 {time.time() - t:.0f}s", flush=True)   # 有效率与耗时
    print("-" * 72)                                                                   # 分隔
    print(f"{'类型':<12}{'n':<5}{'确定性召回':<12}{'RAGAS 两轮均值':<16}")                # 表头
    g = collections.defaultdict(lambda: {"n": 0, "det": 0, "rag": []})                 # 分组统计
    for i, s in enumerate(samples):                                                    # 逐题归组
        k = s["type"]; g[k]["n"] += 1; g[k]["det"] += int(s["det"])                    # 计数
        for rd in rounds:                                                             # 两轮
            if rd[i] is not None:                                                     # 有效才计入
                g[k]["rag"].append(rd[i])                                             # 收集
    for k, v in sorted(g.items(), key=lambda x: -x[1]["n"]):                           # 按题数降序
        rag = sum(v["rag"]) / len(v["rag"]) if v["rag"] else float("nan")              # 均值
        print(f"{k:<12}{v['n']:<5}{v['det'] / v['n']:<12.4f}{rag:<16.4f}")              # 打印：确定性 vs RAGAS
    print("-" * 72)                                                                   # 分隔
    print("注意：确定性召回与 RAGAS 召回口径不同，不可合并成一个数；")                     # 口径提醒
    print("      两者差异大即说明 reference 构造有问题（本例：专有名词型 0.78 vs 0.25）。")  # 本例结论
    return 0                                                                          # 退出


if __name__ == "__main__":                                                            # 允许直接运行
    sys.exit(main())                                                                  # 退出码
