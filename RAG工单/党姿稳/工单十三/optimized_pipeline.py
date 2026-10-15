# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-RAG 性能瓶颈识别与优化
# 模块职责：优化后的 RAG 流水线（离线本地查询理解、预归一化矩阵检索、查询级结果缓存）

import config                                                   # 导入本工单配置模块
import pipeline                                                 # 导入阶段共享实现
import resources                                                # 导入资源加载模块
import bm25 as bm25_mod                                         # 导入 BM25 模块（复用其分词器）
from logger import LOGGER                                       # 导入统一日志对象

_MIN_ENTITY_LEN = 2                                             # 参与词典匹配的实体名最小长度
_STOPWORDS = {"什么", "哪些", "多少", "怎么", "如何", "是否", "分别", "其中", "报告期内"}  # 低信息量疑问词


def match_entities(query, entity_names):                        # 定义词典实体匹配函数
    """用图谱实体名做最长匹配：实体名在查询中直接出现即视为命中，完全离线无需 LLM。"""
    hits = []                                                   # 命中实体列表
    for name in entity_names:                                   # 按长度降序遍历实体名（最长优先）
        if len(name) < _MIN_ENTITY_LEN:                         # 过短的名字噪音大，跳过
            continue                                            # 继续下一个
        if name in query:                                       # 实体名出现在查询中
            if not any(name in h for h in hits):                # 已被更长的命中包含则不重复记录
                hits.append(name)                               # 记录命中
    return hits                                                 # 返回命中实体


def extract_keywords_local(query, entity_names):                # 定义本地关键词抽取函数
    """离线抽取查询关键词：图谱实体（最长匹配）+ jieba 分词结果，全程无网络调用。"""
    entities = match_entities(query, entity_names)              # 先抽图谱实体
    tokens = [t for t in bm25_mod.tokenize(query)               # 再用 jieba 切分查询
              if len(t) >= _MIN_ENTITY_LEN and t not in _STOPWORDS]  # 过滤单字与疑问词
    words, seen = [], set()                                     # 去重后的关键词列表
    for w in entities + tokens:                                 # 实体优先，其次是与问题相关的词
        if w not in seen:                                       # 未出现过才保留
            seen.add(w)                                         # 记录已保留
            words.append(w)                                     # 追加关键词
    return {"entities": entities, "keywords": words[:config.ENTITY_TOP_K * 2],  # 返回结构化结果
            "all_keywords": words}                              # 保留完整关键词列表


class OptimizedPipeline(object):                                # 定义优化后流水线类
    """优化后的 RAG 流水线：查询理解离线化、向量矩阵预归一化、结果缓存化。"""

    def __init__(self, chunks, model, vecs, bm25, entity_names, warm=True):  # 构造方法
        self.chunks = chunks                                    # 语料块列表
        self.model = model                                      # 嵌入模型
        self.unit_vecs = resources.normalize_matrix(vecs)       # 启动时一次性归一化，查询时直接点积
        self.bm25 = bm25                                        # BM25 索引
        self.entity_names = entity_names                        # 图谱实体词典
        self._kw_cache = {}                                     # 查询理解结果缓存（同一查询直接命中）
        self._vec_cache = {}                                    # 查询向量缓存
        if warm:                                                # 需要预热时执行冷启动预热
            resources.warm_up(model, entity_names)              # 预热模型与 jieba 词典

    def _understand(self, query):                               # 定义查询理解方法
        """带缓存的查询理解：首次计算关键词与查询向量，后续同查询直接复用。"""
        if query not in self._kw_cache:                         # 缓存未命中才计算
            self._kw_cache[query] = extract_keywords_local(query, self.entity_names)  # 离线抽取关键词
            self._vec_cache[query] = self.model.encode([query], show_progress_bar=False,  # 编码查询向量
                                                       batch_size=1)[0]  # 取首条
        return self._kw_cache[query], self._vec_cache[query]    # 返回关键词与查询向量

    def run(self, query, tracer):                               # 定义单次查询执行方法
        """按五个阶段执行一次完整 RAG 流程，并把每阶段耗时写入追踪器。"""
        tracer.begin(config.STAGE_QUERY)                        # 阶段一开始
        kw, qvec = self._understand(query)                      # 离线查询理解（无 LLM 调用）
        search_text = pipeline.expand_query(query, kw["all_keywords"])  # 用本地抽出的关键词扩写查询
        tracer.end(config.STAGE_QUERY, keyword_count=len(kw["keywords"]),  # 阶段一结束
                   entity_count=len(kw["entities"]))            # 记录实体数量指标
        tracer.begin(config.STAGE_RETRIEVE)                     # 阶段二开始
        vec_hits = pipeline.vector_search_fast(qvec, self.unit_vecs, config.VECTOR_TOP_K)  # 向量召回
        bm_hits = pipeline.bm25_search(self.bm25, search_text, config.BM25_TOP_K)  # BM25 召回
        fused = pipeline.rrf_fuse([[r for r, _ in vec_hits], [r for r, _ in bm_hits]])  # RRF 融合
        rows = pipeline.mmr_dedup(fused, self.unit_vecs, config.FINAL_TOP_K)  # MMR 去冗余
        tracer.end(config.STAGE_RETRIEVE, vector_hits=len(vec_hits),  # 阶段二结束
                   bm25_hits=len(bm_hits), fused_candidates=len(fused))  # 记录候选数量指标
        return pipeline.run_tail(query, self.chunks, rows, tracer)  # 执行阶段三~五


def build(chunks=None, model=None, warm=True):                  # 定义流水线构建函数
    """加载共享资源并组装优化后流水线。"""
    chunks = chunks if chunks is not None else resources.load_chunks()  # 加载语料块
    model = model if model is not None else resources.load_model()  # 加载嵌入模型
    vecs = resources.load_vectors()                             # 加载向量矩阵
    idx = resources.load_bm25()                                 # 加载 BM25 索引
    names = resources.load_entity_names()                       # 加载实体词典
    LOGGER.info("优化后流水线构建完成（向量已预归一化，模型与分词器已预热）")  # 输出构建完成日志
    return OptimizedPipeline(chunks, model, vecs, idx, names, warm=warm)  # 返回流水线实例


if __name__ == "__main__":                                      # 仅在直接运行时执行
    import json                                                 # 导入 json 用于打印摘要
    from logger import StageTracer                              # 导入阶段追踪器
    _pipe = build()                                             # 构建流水线
    _tracer = StageTracer(persist=False)                        # 创建不落盘的追踪器
    _resp = _pipe.run(config.QUERIES[13]["query"], _tracer)     # 跑一条样例查询
    print(json.dumps(_tracer.summary(), ensure_ascii=False, indent=1))  # 打印阶段耗时摘要
    print(_resp["answer"][:200])                                # 打印答案前缀
