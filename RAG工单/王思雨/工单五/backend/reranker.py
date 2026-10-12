# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：重排层。用 bge-reranker-v2-m3 对混合检索候选做交叉编码精排，取 top_n 交给大模型。
#           重排失败时必须回退到混合检索的原始排序，并如实上报 degraded，不能静默降级。
from typing import List, Dict, Tuple                                            # 类型标注

from . import config as C                                                       # 复用集中配置

_RERANKER = None                                                                # 重排模型单例
_LOAD_ERROR = ""                                                                # 记录加载失败原因，避免每次请求重复重试加载


def rerank_model():                                                             # 懒加载重排模型
    """首次调用加载并常驻；加载失败返回 None，由调用方回退到原始排序。"""
    global _RERANKER, _LOAD_ERROR                                               # 声明全局状态
    if _RERANKER is not None:                                                   # 已加载
        return _RERANKER                                                        # 直接用
    if _LOAD_ERROR:                                                             # 之前已确认加载不了
        return None                                                             # 不重复尝试，避免每次请求都卡住
    try:                                                                        # 加载可能因路径/显存失败
        from FlagEmbedding import FlagReranker                                  # 延迟导入
        _RERANKER = FlagReranker(C.RERANK_MODEL_PATH, use_fp16=(C.DEVICE == "cuda"))  # fp16 仅在 GPU 上开
        return _RERANKER                                                        # 返回单例
    except Exception as exc:                                                    # 加载失败
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"                            # 记住原因
        print(f"[重排降级] 重排模型加载失败：{_LOAD_ERROR}")                       # 明确报因
        return None                                                             # 返回 None


def is_ready() -> bool:                                                         # 健康检查用
    """报告重排模型是否可用（加载过且成功）。"""
    return _RERANKER is not None                                                # 单例存在即就绪


def rerank(query: str, candidates: List[Dict], top_n: int = None) -> Tuple[List[Dict], List[str]]:  # 重排入口
    """对候选精排并截断；返回 (排序后的候选, 降级原因列表)。

    慢的根因是"成对打分"的填充开销：整表块可达数千字，与查询配对时会按最长序列补 padding。
    故对打分用的文本做 RERANK_MAX_CHARS 截断——只截打分输入，不改实际交给大模型的原文。
    """
    top_n = top_n or C.RERANK_TOP_N                                             # 保留条数
    degraded: List[str] = []                                                    # 降级记录
    if not candidates:                                                          # 没有候选
        return [], degraded                                                     # 直接返回
    model = rerank_model()                                                      # 取模型
    if model is None:                                                           # 模型不可用
        degraded.append("rerank")                                               # 记降级
        return candidates[:top_n], degraded                                     # 回退到混合检索原始排序
    pairs = [[query, str(c.get("text", ""))[:C.RERANK_MAX_CHARS]]               # 查询-片段配对（截断打分输入）
             for c in candidates]                                               # 覆盖全部候选
    try:                                                                        # 打分可能因显存/输入异常失败
        scores = model.compute_score(pairs, normalize=True)                     # 归一化到 0~1，便于跨查询比较
    except Exception as exc:                                                    # 打分失败
        print(f"[重排降级] 打分失败：{type(exc).__name__}: {exc}")                # 明确报因
        degraded.append("rerank")                                               # 记降级
        return candidates[:top_n], degraded                                     # 回退原始排序
    if isinstance(scores, (int, float)):                                        # 单条候选时返回标量
        scores = [float(scores)]                                                # 统一成列表
    ranked = sorted(zip(candidates, scores),                                    # 候选与分数配对
                    key=lambda t: float(t[1]), reverse=True)                    # 按重排分降序
    out: List[Dict] = []                                                        # 结果
    for item, score in ranked[:top_n]:                                          # 取前 top_n
        rec = dict(item)                                                        # 复制，避免污染原对象
        rec["rerank_score"] = float(score)                                      # 附上重排得分
        out.append(rec)                                                         # 收集
    return out, degraded                                                        # 返回
