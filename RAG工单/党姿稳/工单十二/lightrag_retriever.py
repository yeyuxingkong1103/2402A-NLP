# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：LightRAG 查询响应流程——双层关键词抽取、局部/全局双路图谱检索并与 BM25 融合

import config                                                   # 导入本工单配置模块
import fusion                                                   # 导入 RRF 与 MMR 融合算法
import lightrag_index                                           # 导入实体索引（实体链接与图谱扩散）
import llm                                                      # 导入 LLM 封装（带磁盘缓存）
from logger import LOGGER                                       # 导入统一日志对象

# 双层关键词抽取提示词：LightRAG 的核心设计——把查询拆成「低层具体实体」与「高层概念主题」
_KW_SYSTEM = "你是查询理解专家，负责把用户问题拆解为用于知识图谱检索的两类关键词。"
_KW_USER = """请从下面的问题中提取两类关键词：

1. 低层关键词（low_level_keywords）：问题中出现的具体实体，如公司名、人名、产品名、技术标准名、财务指标名、金额、时间等。
2. 高层关键词（high_level_keywords）：问题涉及的宽泛概念、主题或领域，如行业、业务领域、战略方向、上下游环节等（不是具体实体名）。

问题：%s

要求：
1. 关键词必须来自问题本身或其同义表述，不要凭空补充问题之外的信息。
2. 每类 1~6 个关键词；若某类确实没有合适的关键词，返回空数组。
3. 关键词尽量使用名词短语，不要整句。

只输出如下 JSON：
{"low_level_keywords": ["..."], "high_level_keywords": ["..."]}"""


def extract_keywords(question):                                 # 定义双层关键词抽取函数
    """调用 LLM 把问题拆为低层（具体实体）与高层（概念主题）关键词；失败时回退为问题本身。"""
    obj = llm.chat_json(_KW_SYSTEM, _KW_USER % question, max_tokens=400)  # 调用 LLM 并解析 JSON
    low, high = [], []                                          # 初始化两类关键词
    if isinstance(obj, dict):                                   # 解析成功时读取字段
        low = [str(k).strip() for k in (obj.get("low_level_keywords") or []) if str(k).strip()]  # 低层关键词
        high = [str(k).strip() for k in (obj.get("high_level_keywords") or []) if str(k).strip()]  # 高层关键词
    if not low:                                                 # 低层关键词为空时回退（保证检索不空转）
        low = [question]                                        # 用整个问题作为兜底关键词
    return {"low_level_keywords": low, "high_level_keywords": high}  # 返回两类关键词


class LightRagIndex(object):                                    # 定义 LightRAG 索引类
    """LightRAG 检索索引：知识图谱 + 实体向量 + 扁平 RAG 索引（提供 BM25 与块正文）。"""

    def __init__(self, graph, base_index, ent_index):           # 构造方法
        self.graph = graph                                      # 知识图谱
        self.base = base_index                                  # 扁平 RAG 索引（复用向量与 BM25）
        self.ents = ent_index                                   # 实体索引（实体链接与图谱扩散）
        self.row2chunk = {c["id"]: i for i, c in enumerate(self.base.chunks)}  # 块编号 -> 行号

    def _rows_of(self, chunk_ids):                              # 定义块编号转行号函数
        """把实体关联的块编号转换为语料行号，过滤越界编号并保持顺序去重。"""
        out, seen = [], set()                                   # 结果与去重集合
        for cid in chunk_ids:                                   # 遍历块编号
            row = self.row2chunk.get(cid)                       # 查表取行号
            if row is not None and row not in seen:             # 有效且未出现过
                seen.add(row)                                   # 记录
                out.append(row)                                 # 追加行号
        return out                                              # 返回行号列表

    def retrieve(self, question, top_k=None, with_trace=False):  # 定义检索主方法
        """执行双层检索：局部路 + 全局路 + BM25 路 三路 RRF 融合，再经 MMR 去冗余。"""
        top_k = top_k or config.FINAL_TOP_K                     # 最终返回的上下文块数量
        kw = extract_keywords(question)                         # 第一步：抽取双层关键词
        low_hits = self.ents.match(kw["low_level_keywords"])    # 低层关键词链接到具体实体
        high_hits = self.ents.match(kw["high_level_keywords"])  # 高层关键词链接到概念相关实体
        local_rows = self._rows_of(self.ents.chunks_of([n for n, _ in low_hits]))  # 局部路：实体关联块
        expanded = self.ents.expand([n for n, _ in high_hits])  # 全局路：沿图谱关系扩散邻域
        global_rows = self._rows_of(self.ents.chunks_of([n for n, _ in expanded]))  # 全局路：邻域关联块
        bm25_hits = self.base.bm25.search(question, top_k=config.BM25_TOP_K)  # 第三路：BM25 全文检索
        bm25_rows = [i for i, _ in bm25_hits]                   # 仅取行号
        fused, scores = fusion.rrf_fuse([local_rows, global_rows, bm25_rows],  # 三路 RRF 融合
                                        weights=config.LIGHTRAG_LIST_WEIGHTS)  # 使用图谱侧加权
        selected = fusion.mmr_dedup(fused, scores, self.base.vecs, top_k)  # MMR 去冗余并截断
        fusion.retrieve_summary("LightRAG", len(fused), len(selected))  # 输出候选数量变化
        via = {}                                                # 行号 -> 召回路径标记
        for rows, tag in ((local_rows, "局部"), (global_rows, "全局"), (bm25_rows, "BM25")):  # 标记三条路径
            for r in rows:                                      # 遍历该路径的行号
                via.setdefault(r, []).append(tag)               # 追加路径标记（可能多路命中）
        results = []                                            # 初始化结果列表
        for rank, i in enumerate(selected):                     # 遍历最终选中的块
            c = self.base.chunks[i]                             # 取出语料块
            results.append({"rank": rank,                        # 名次（从 0 开始）
                            "idx": i,                            # 语料行号
                            "id": c["id"],                       # 块编号
                            "source": c["source"],               # 来源说明书
                            "page": c["page"],                   # 来源页码
                            "text": c["text"],                   # 块正文
                            "score": round(scores.get(i, 0.0), 6),  # RRF 融合得分
                            "via": "+".join(via.get(i, []))})    # 召回路径标记
        trace = {"low_level_keywords": kw["low_level_keywords"],  # 低层关键词（用于报告展示）
                 "high_level_keywords": kw["high_level_keywords"],  # 高层关键词
                 "matched_entities": [n for n, _ in low_hits][:10],  # 低层命中的实体
                 "expanded_entities": [n for n, _ in expanded][:10],  # 高层扩散到的实体
                 "local_candidates": len(local_rows),           # 局部路候选数
                 "global_candidates": len(global_rows),         # 全局路候选数
                 "bm25_candidates": len(bm25_rows)}             # BM25 路候选数（字典在此闭合）
        LOGGER.info("LightRAG 检索完成：低层词 %d / 高层词 %d / 实体 %d / 返回 %d 块",
                    len(kw["low_level_keywords"]), len(kw["high_level_keywords"]),
                    len(low_hits) + len(expanded), len(results))  # 输出检索摘要
        return (results, trace) if with_trace else results      # 按需返回追溯信息


def build_index(model, graph, base_index, force=False):         # 定义索引构建函数
    """构建 LightRAG 索引：在扁平 RAG 索引之上叠加实体向量索引。"""
    ent = lightrag_index.build_or_load_entity_index(model, graph, force=force)  # 构建实体索引
    return LightRagIndex(graph, base_index, ent)                # 组装 LightRAG 索引
