# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：混合检索。向量召回（Milvus）+ 关键词召回（BM25）双路并跑，用加权 RRF 融合成一份候选，
#           再交给重排模型。任一路不可用都要能降级并如实上报 degraded 标记，不能静默失效。
from typing import List, Dict, Tuple                                            # 类型标注

from . import config as C                                                       # 复用集中配置
from . import bm25_store, milvus_store                                          # 两个检索通道
from .text_clean import is_noisy                                                # 水印污染判定


def _rrf(rank: int) -> float:                                                   # 单路排名折算成分数
    """RRF：score = 1/(K + rank)。不同通道的原始分数量纲不可比，用排名折算才公平。"""
    return 1.0 / (C.RRF_K + rank + 1)                                           # rank 从 0 起，故 +1


def _norm_rrf(rank: int, total: int) -> float:                                  # 通道内归一化的排名分
    """把该通道的 RRF 线性映射到 0~1：第 1 名=1，最后一名=0。

    只有一条结果时无法比较，按 1.0 处理（它是该通道唯一给出的信息）。
    """
    if total <= 1:                                                              # 该通道只有一条或没有
        return 1.0                                                              # 直接给满分
    lo, hi = _rrf(total - 1), _rrf(0)                                           # 该通道的取值范围 [最低, 最高]（rank 越小分越高）
    return (_rrf(rank) - lo) / (hi - lo) if hi > lo else 1.0                    # 名次越靠前越接近 1


def _fusion_key(item: Dict) -> str:                                             # 融合去重键
    """优先用块号；没有块号时退回"文档名+页码+文本前 64 字"，避免同一来源被算两次。"""
    cid = item.get("chunk_id")                                                  # 块号
    if cid:                                                                     # 有块号
        return str(cid)                                                         # 用它
    return f"{item.get('doc', '')}#{item.get('page_no', 0)}#{str(item.get('text', ''))[:64]}"  # 兜底键


class HybridRetriever:                                                          # 混合检索器
    """持有两个通道的句柄；检索时并跑、融合、降权、排序，输出统一结构的候选列表。"""

    def __init__(self, milvus=None, bm25_index=None, chunks: List[Dict] = None,  # 构造
                 embed_fn=None):                                                # 预留自定义编码函数（测试用）
        self.milvus = milvus                                                    # Milvus 客户端（可为 None）
        self.bm25_index = bm25_index                                            # BM25 索引（可为 None）
        self.bm25_chunks = chunks or []                                         # BM25 侧的块列表（与索引同序）
        self.embed_fn = embed_fn or milvus_store.embed                          # 编码函数

    def retrieve(self, query: str, top_k: int = None,                             # 主检索入口
                 vec_k: int = None, bm25_k: int = None) -> Tuple[List[Dict], List[str]]:  # 返回候选与降级标记
        """双路召回 → 加权 RRF 融合 → 去重 → 水印降权 → 截断；返回 (候选列表, 降级原因列表)。"""
        top_k = top_k or C.FUSION_TOP_K                                         # 融合后保留条数
        vec_k = vec_k or C.VEC_TOP_K                                            # 向量通道条数
        bm25_k = bm25_k or C.BM25_TOP_K                                         # 关键词通道条数
        degraded: List[str] = []                                                # 降级记录
        vec_hits = milvus_store.vector_search(self.milvus, query, vec_k)        # 向量通道（内部已兜底）
        if not vec_hits:                                                        # 向量通道空手而归
            degraded.append("milvus")                                           # 记降级，前端可提示"仅关键词检索"
        bm_hits = bm25_store.search(self.bm25_index, self.bm25_chunks, query, bm25_k)  # 关键词通道
        if not bm_hits:                                                         # 关键词通道空手而归
            degraded.append("bm25")                                             # 记降级
        # 先按排名算 RRF，再"各自归一化到 0~1"，最后乘权重相加。
        # 坑（实测）：直接把权重乘到原始 RRF 上是错的——两路各取 20 条时，
        #   RRF 取值范围是 [1/(K+20), 1/(K+1)] ≈ [0.0125, 0.0164]，
        #   乘 0.7 得 [0.0088, 0.0115]，乘 0.3 得 [0.0038, 0.0049]，两段区间完全不相交！
        #   后果是"BM25 排第 1 的块"永远打不过"向量排第 20 的块"，关键词通道被架空。
        #   实测表现：BM25 第 1 名的目标块在融合后直接掉出前 20，最终答不出题。
        #   归一化后，权重才真正表达"通道重要度"，且不会因名次偏移互相压制。
        n_vec, n_bm = len(vec_hits), len(bm_hits)                               # 两路各自的条数
        fused: Dict[str, Dict] = {}                                             # 融合结果（键为去重键）
        for rank, item in enumerate(vec_hits):                                  # 向量通道
            key = _fusion_key(item)                                             # 去重键
            rec = fused.setdefault(key, dict(item))                             # 首次出现则建档
            rec["vec_score"] = float(item.get("vec_score", 0.0))                # 记录向量得分
            rec["_vr"] = _norm_rrf(rank, n_vec)                                 # 该通道内归一化后的排名分
        for rank, item in enumerate(bm_hits):                                   # 关键词通道
            key = _fusion_key(item)                                             # 去重键
            rec = fused.setdefault(key, dict(item))                             # 建档
            rec["bm25_score"] = float(item.get("bm25_score", 0.0))              # 记录 BM25 得分
            rec["_br"] = _norm_rrf(rank, n_bm)                                  # 该通道内归一化后的排名分
            for field in ("text", "doc", "page_no", "locator", "block_type"):   # 补齐向量侧没带的字段
                if not rec.get(field) and item.get(field):                      # 缺了才补
                    rec[field] = item[field]                                    # 补齐
        for rec in fused.values():                                              # 汇总加权
            rec["fusion"] = (C.VEC_WEIGHT * rec.pop("_vr", 0.0)                 # 向量通道贡献
                             + C.BM25_WEIGHT * rec.pop("_br", 0.0))             # 关键词通道贡献
            rec["vec_rank_score"] = round(float(rec.get("vec_score", 0.0)), 4)  # 保留原始分，便于排障
            rec["bm25_rank_score"] = round(float(rec.get("bm25_score", 0.0)), 4)  # 同上
        items = list(fused.values())                                            # 转成列表
        for it in items:                                                        # 逐条后处理
            it.setdefault("vec_score", 0.0)                                     # 补齐缺失字段，便于统一排序
            it.setdefault("bm25_score", 0.0)                                    # 同上
            if is_noisy(str(it.get("text", ""))):                               # 水印污染块：数字与表头的对应关系不可信
                it["fusion"] = float(it.get("fusion", 0.0)) * 0.5               # 降权而非删除，保留召回但压后
                it["noisy"] = True                                              # 打标便于排查
        items.sort(key=lambda x: float(x.get("fusion", 0.0)), reverse=True)     # 按融合分降序
        return items[:top_k], degraded                                          # 截断后返回
