"""bge-reranker-v2-m3 精排。

存在的理由（FR-3.6 / AC-4）：RRF 融合是**位置**融合，只保证两路都召回的靠前，
不判断"这条到底相不相关"。精排用的是逐条交叉编码，把问句与法条原文一起过一遍
模型，比双塔相似度准得多。AC-4 要求精排 top3 命中率 ≥ 85%，靠的就是这一步。

模型是构造参数而非模块级单例：单测要能塞替身，否则每条测试都要等 2.2GB 加载。

实测（2026-09-23）：模型目录无 modules.json，CrossEncoder 裸加载；
输出已过 sigmoid ∈ [0,1]；CPU 上 30 条候选 3.43s，故生产路径必须用 cuda。

实测（2026-09-27，RTX 4060 Laptop）：cuda 上 30 条候选 0.27~0.28s（首次调用
0.55s 含预热），显存约 2.2GB（fp32）；AC-12 的 2s 检索预算下精排不是瓶颈。
"""
from __future__ import annotations

# 环境规避（pandas 先于 sklearn）与编码侧共用同一份实现，
# 为什么必须规避、什么时候能删，全写在 app.compat 里
from app.compat import ensure_import_order

RERANK_MODEL_PATH = r"D:\Model\reranker"

# 技术方案 5.2 的起始值：输入 top 20~50 取 30、输出 top 3~5 取 5。
# 输入取 30 而非 50 是实测倒推——CPU 上 30 条 3.43s，50 条会到 5s 级
RERANK_INPUT_TOPK = 30
RERANK_OUTPUT_TOPK = 5


def load_reranker(model_path: str = RERANK_MODEL_PATH, device: str = "cuda"):
    """加载精排模型。device 默认 cuda——CPU 跑不赢 AC-12 的 2s 检索延迟。"""
    # 必须先于 sentence_transformers 调用：它内部"先 torch 后 sklearn"会踩中
    # 本机环境缺陷，精确条件与删除条件见 app.compat.ensure_import_order
    ensure_import_order()
    from sentence_transformers import CrossEncoder
    # max_length 512 够用：父块文本实测最长 400 字
    return CrossEncoder(model_path, max_length=512, device=device)


def pairs_for(query: str, blocks: list[dict]) -> list[tuple[str, str]]:
    """把父块拼成精排要的 (问句, 原文) 对。"""
    return [(query, block["text"]) for block in blocks]


def top_blocks(query: str, blocks: list[dict], scorer,
               input_topk: int = RERANK_INPUT_TOPK,
               output_topk: int = RERANK_OUTPUT_TOPK) -> list[dict]:
    """对父块打分并返回 top N，分数写进 rerank_score。

    scorer 的签名是 (list[(query, text)]) -> list[float]，即 CrossEncoder.predict。
    收成参数是为了让排序逻辑可离线单测——加载 2.2GB 模型做单测不可接受。

    不原地修改传入的 blocks：上游还要用原顺序做精确块合并。
    """
    if not blocks:
        return []
    candidates = blocks[:input_topk]
    scores = scorer(pairs_for(query, candidates))
    # 分数必须与候选一一对应：zip 遇短序列会静默截断，块被悄悄丢掉且无人报错，
    # 上线后表现为"部分候选莫名消失"，比当场抛错难查得多
    if len(scores) != len(candidates):
        raise ValueError(
            f"精排分数条数不符：期望 {len(candidates)}，实际 {len(scores)}")
    scored = [dict(block, rerank_score=float(score))
              for block, score in zip(candidates, scores)]
    scored.sort(key=lambda b: b["rerank_score"], reverse=True)
    return scored[:output_topk]
