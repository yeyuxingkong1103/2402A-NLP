# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-RAG 性能瓶颈识别与优化
# 模块职责：优化前的 RAG 流水线（查询理解依赖在线 LLM 调用，向量检索每次重算归一化）

import json                                                     # 导入 json 用于解析 LLM 返回
import requests                                                 # 导入 requests 用于调用 LLM 接口
import config                                                   # 导入本工单配置模块
import pipeline                                                 # 导入阶段共享实现
import resources                                                # 导入资源加载模块
from logger import LOGGER                                       # 导入统一日志对象

_KW_SYSTEM = "你是查询理解专家，负责从用户问题中抽取用于检索的关键词与实体。"  # 关键词抽取系统提示
_KW_USER = """请从下面的问题中抽取检索关键词：

问题：%s

要求：抽取具体实体名（公司名、人名、产品名、财务指标、金额、时间等）与核心概念词，
每行一个，不要编号、不要解释。"""                                            # 关键词抽取用户提示


def extract_keywords_llm(query):                                # 定义在线关键词抽取函数
    """调用 DeepSeek 抽取检索关键词；这是优化前流水线中嵌在检索链路里的 LLM 调用。"""
    url = config.DEEPSEEK_BASE_URL.rstrip("/") + "/chat/completions"  # 拼装接口地址
    payload = {"model": config.DEEPSEEK_MODEL,                  # 模型名称
               "messages": [{"role": "system", "content": _KW_SYSTEM},  # 系统提示
                            {"role": "user", "content": _KW_USER % query}],  # 用户提示
               "temperature": 0.0,                              # 确定性输出
               "max_tokens": 200}                               # 关键词很短，限制输出长度
    headers = {"Authorization": "Bearer " + config.DEEPSEEK_API_KEY,  # 鉴权头
               "Content-Type": "application/json"}              # 请求体类型
    resp = requests.post(url, json=payload, headers=headers, timeout=config.LLM_TIMEOUT)  # 发送请求
    resp.raise_for_status()                                     # 非 2xx 抛出异常
    text = resp.json()["choices"][0]["message"]["content"]      # 取返回正文
    words = [w.strip(" -•\t") for w in text.splitlines()]        # 按行拆分并清理列表符号
    return [w for w in words if w][:config.ENTITY_TOP_K]        # 过滤空行并截断到上限


class BaselinePipeline(object):                                 # 定义优化前流水线类
    """优化前的 RAG 流水线：查询理解走在线 LLM，向量检索每次重算全量范数。"""

    def __init__(self, chunks, model, vecs, bm25, entity_names):  # 构造方法
        self.chunks = chunks                                    # 语料块列表
        self.model = model                                      # 嵌入模型
        self.vecs = vecs                                        # 原始（未归一化）向量矩阵
        self.unit_vecs = resources.normalize_matrix(vecs)       # 供 MMR 使用的一份归一化矩阵
        self.bm25 = bm25                                        # BM25 索引
        self.entity_names = entity_names                        # 实体词典（本版未使用）

    def run(self, query, tracer):                               # 定义单次查询执行方法
        """按五个阶段执行一次完整 RAG 流程，并把每阶段耗时写入追踪器。"""
        tracer.begin(config.STAGE_QUERY)                        # 阶段一开始
        keywords = extract_keywords_llm(query)                  # 在线 LLM 抽取关键词（主要瓶颈）
        qvec = self.model.encode([query], show_progress_bar=False,  # 编码查询向量
                                 batch_size=1)[0]               # 取首条（仅一条）
        tracer.end(config.STAGE_QUERY, keyword_count=len(keywords))  # 阶段一结束
        tracer.begin(config.STAGE_RETRIEVE)                     # 阶段二开始
        vec_hits = pipeline.vector_search_naive(qvec, self.vecs, config.VECTOR_TOP_K)  # 未优化的向量检索
        search_text = pipeline.expand_query(query, keywords)    # 用 LLM 抽出的关键词扩写查询
        bm_hits = pipeline.bm25_search(self.bm25, search_text, config.BM25_TOP_K)  # BM25 检索
        fused = pipeline.rrf_fuse([[r for r, _ in vec_hits], [r for r, _ in bm_hits]])  # RRF 融合
        rows = pipeline.mmr_dedup(fused, self.unit_vecs, config.FINAL_TOP_K)  # MMR 去冗余
        tracer.end(config.STAGE_RETRIEVE, vector_hits=len(vec_hits),  # 阶段二结束
                   bm25_hits=len(bm_hits), fused_candidates=len(fused))  # 记录候选数量指标
        return pipeline.run_tail(query, self.chunks, rows, tracer)  # 执行阶段三~五


def build(chunks=None, model=None):                             # 定义流水线构建函数
    """加载共享资源并组装优化前流水线，返回 (流水线, 语料, 向量, BM25, 实体词典)。"""
    chunks = chunks if chunks is not None else resources.load_chunks()  # 加载语料块
    model = model if model is not None else resources.load_model()  # 加载嵌入模型
    vecs = resources.load_vectors()                             # 加载向量矩阵
    idx = resources.load_bm25()                                 # 加载 BM25 索引
    names = resources.load_entity_names()                       # 加载实体词典
    LOGGER.info("优化前流水线构建完成")                            # 输出构建完成日志
    return BaselinePipeline(chunks, model, vecs, idx, names), chunks, vecs, idx, names  # 返回流水线与资源


if __name__ == "__main__":                                      # 仅在直接运行时执行
    from logger import StageTracer                              # 导入阶段追踪器
    _pipe, _chunks, _, _, _ = build()                           # 构建流水线
    _tracer = StageTracer(persist=False)                        # 创建不落盘的追踪器
    _resp = _pipe.run(config.QUERIES[13]["query"], _tracer)     # 跑一条样例查询
    print(json.dumps(_tracer.summary(), ensure_ascii=False, indent=1))  # 打印阶段耗时摘要
    print(_resp["answer"][:200])                                # 打印答案前缀
