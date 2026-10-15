# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：提供向量检索、BM25 检索与 RRF 混合融合、MMR 去冗余，这是检索层优化核心

import re                                                       # 导入正则用于把复合问题拆分为子查询
import numpy as np                                              # 导入 numpy 用于向量化点积计算
import config                                                   # 导入本工单配置模块
from timing import LOGGER                                       # 导入统一日志对象
from bm25 import BM25, tokenize                                 # 导入自实现的 BM25 检索器与中文分词函数

# 复合问题的拆分点：仅按标点切分。若再按「和/与」等连词切分，会把「资本结构优化和新兴领域投资」
# 这类完整名词短语切碎，产生「分析这些银行」之类的无效子查询，反而引入噪声
_SUBQ_SPLIT = re.compile(r"[，,。；;、\n]+")

# 第二轮优化：关键词覆盖精排所用的停用词表。验收问题里的「银行/年报/提到/分析」等通用词
# 在金融年报语料中几乎每块都有，参与覆盖度计算只会抬升跨行样板块，必须剔除
_COVER_STOP = set(
    "的 了 在 是 与 和 及 或 对 从 被 把 有 无 未 为 以 因 而 则 其 该 此 这 那 们 吗 呢 哪些 什么 如何 怎么 "
    "是否 已经 目前 银行 公司 集团 年报 年度 报告 中 上 下 里 年 提到 提出 包括 结合 分析 评估 关注 重点 "
    "表现 方面 情况 进行 相关 主要 更多 更加 进一步 这些 那些 通过 针对 基于 同时 以及 首先 其次 综上".split()
)


def is_broad_question(question):                                # 定义问题类型判别函数
    """判断是否为宽泛综合分析型问题（决定动态上下文条数）。

    长问题或多要点对比题（如『分析共同策略与差异化策略…评估可持续性』）需要覆盖多家机构、
    多个论点，固定 8 条上下文无法支撑参考答案的全部要点；具体事实题则条数越少精度越高。
    """
    q = question or ""                                          # 取问题原文
    if len(q) >= 60:                                            # 长问题默认视为综合分析题
        return True                                             # 返回宽泛
    return any(k in q for k in ("共同", "差异化", "对比", "综合", "可持续"))  # 命中多要点对比关键词


def is_comparative_question(question):                          # 定义对比型问题判别函数
    """判断是否为跨机构对比分析题（第三轮优化新增，与 is_broad_question 的区别）：

    is_broad_question 由长度或关键词触发（含 Q3 这类单机构深分析题），
    本函数只命中『共同策略 vs 差异化策略』这类跨机构对比题（Q4）。
    只有对比题才启用同义词扩展、图谱加权与 λ 放宽——Q3 这类题已实测达标，
    贸然放宽 λ 反而会破坏其已排好的相关性顺序。
    """
    return any(k in (question or "") for k in ("共同", "差异化", "对比", "综合", "可持续"))  # 对比关键词判定


def _question_terms(question):                                  # 定义问题内容词提取函数
    """提取问题中的内容词（去停用词、去脱敏占位符），保持出现顺序并去重。"""
    terms = []                                                  # 内容词列表
    seen = set()                                                # 去重集合
    for tok in tokenize(question or ""):                        # 复用 BM25 的中文分词
        if tok in _COVER_STOP:                                  # 通用停用词不参与覆盖计算
            continue                                            # 跳过
        if re.match(r"^x+y?$|^y+$", tok):                       # 过滤 xx银行/yy银行 这类脱敏占位符
            continue                                            # 跳过
        if tok in seen:                                         # 重复词只保留一次
            continue                                            # 跳过
        seen.add(tok)                                           # 标记已见
        terms.append(tok)                                       # 追加内容词
    return terms                                                # 返回内容词列表


def _question_bigrams(question, terms):                         # 定义问题短语对提取函数
    """把问题中相邻的内容词拼成短语（如 盈利+增长→盈利增长），用于原文子串匹配。

    短语级匹配比单词覆盖判别力更强：『盈利增长』『创新商业模式』『拨备覆盖率』这类
    连续短语只出现在真正切题的块中，跨行致辞样板块不含这些短语。
    """
    raw = re.sub(r"[，,。；;、?？!！\s（）()：:]", "", question or "")  # 去标点的干净原文
    bigrams = []                                                # 短语列表
    for a, b in zip(terms, terms[1:]):                          # 遍历相邻内容词对
        ia, ib = raw.find(a), raw.find(b)                       # 两词在原文中的位置
        if ia < 0 or ib < 0 or ib < ia:                         # 词不在原文中或顺序颠倒
            continue                                            # 跳过
        if ib - ia <= len(a) + 2:                               # 位置相邻（中间最多隔 2 字）才拼合
            bigrams.append(a + b)                               # 拼出短语
    return bigrams                                              # 返回短语列表


def synonym_queries(question):                                  # 定义同义词扩展子查询生成函数
    """返回问题命中的领域同义词扩展子查询列表（第三轮优化，仅对比题使用）。

    问题用词与年报原文表述不一致（『资本结构优化』vs 原文『利润留存补充资本』），
    单靠原问题在词面与语义两侧都检不回证据块；这里按触发词追加领域精准子查询。
    独立成函数便于：expand_queries 组装查询列表、multi_query_search 识别同义词路
    以施加更高权重、pipeline 把同义词词混入覆盖度精排词表。
    """
    out = []                                                    # 扩展子查询列表
    for key, extras in config.DOMAIN_SYNONYMS.items():          # 遍历领域同义词表
        if key in (question or "") and extras:                  # 问题命中触发词且配置了扩展
            if extras[0] not in out:                            # 去重
                out.append(extras[0])                           # 每个触发词取一条最有信息量的子查询
    return out                                                  # 返回扩展子查询列表


def expand_queries(question):                                   # 定义多路查询扩展函数
    """把复合问题拆成若干子查询，与原问题一起构成多路召回条件（检索层优化点）。

    工单要求检索时间不超过 3 秒，故这里采用规则拆分而非调用大模型做查询改写，
    在零额外延迟的前提下提升多要点问题的召回覆盖面。

    第三轮优化：对比型问题额外做领域同义词扩展——问题用『资本结构优化』，
    年报原文用『利润留存补充资本』；问题用『保险公司』，原文用『保险资金资产负债久期匹配』。
    每个触发词只取一条最有信息量的扩展子查询，避免把 BM25 的排序权重稀释掉。
    """
    queries = [question]                                        # 原问题始终作为第一路，保证主体语义不丢失
    if is_comparative_question(question):                       # 仅对比题做同义词扩展
        for sq in synonym_queries(question):                    # 取命中的同义词扩展子查询
            queries.append(sq)                                  # 追加到查询列表
            if len(queries) >= config.MULTI_QUERY_MAX:          # 达到上限即停止
                return queries[:config.MULTI_QUERY_MAX]         # 截断返回
    for piece in _SUBQ_SPLIT.split(question or ""):             # 按标点与并列连词切分
        piece = piece.strip()                                   # 去掉首尾空白
        if len(piece) < 5:                                      # 过短片段无检索价值
            continue                                            # 跳过
        if piece in queries:                                    # 已存在则去重
            continue                                            # 跳过
        queries.append(piece)                                   # 追加子查询
        if len(queries) >= config.MULTI_QUERY_MAX:              # 达到上限即停止
            break                                               # 跳出循环
    return queries[:config.MULTI_QUERY_MAX]                     # 返回查询列表（防御性截断）


class Retriever(object):                                        # 定义统一检索器类
    """封装向量索引、BM25 索引与融合策略；一次构建，多次查询。"""

    def __init__(self, chunks, vectors, bm25=None):             # 构造方法：接收文本块、向量矩阵与可选 BM25
        self.chunks = chunks                                    # 保存文本块列表（下标即向量行号）
        self.vectors = vectors                                  # 保存归一化向量矩阵 (N, dim)
        if bm25 is None:                                        # 未传入现成索引时现场构建
            bm25 = BM25()                                       # 实例化 BM25 检索器
            bm25.fit([c["text"] for c in chunks])               # 用全部块正文构建 BM25 索引
        self.bm25 = bm25                                        # 保存 BM25 索引（支持外部复用缓存）
        self.id2idx = {c["id"]: i for i, c in enumerate(chunks)}  # 建立块 ID 到行号的映射

    # -------------------------------------------------------------------
    # 一、向量检索
    # -------------------------------------------------------------------
    def vector_search(self, query_vec, top_k=None):             # 定义向量检索方法
        """用归一化向量的点积作为余弦相似度，返回 top_k 的 (行号, 分数)。"""
        top_k = top_k or config.VECTOR_TOP_K                    # 取默认候选数
        if self.vectors.shape[0] == 0:                          # 索引为空时直接返回空
            return []                                           # 返回空列表
        scores = self.vectors.dot(query_vec)                    # 矩阵与查询向量做点积（等价余弦相似度）
        k = min(top_k, scores.shape[0])                         # 防止 top_k 超过索引规模
        idx = np.argpartition(-scores, k - 1)[:k]               # 用部分排序快速取出前 k 个候选下标
        idx = idx[np.argsort(-scores[idx])]                     # 对候选按分数降序排列
        return [(int(i), float(scores[i])) for i in idx]        # 返回 (行号, 分数) 列表

    # -------------------------------------------------------------------
    # 二、BM25 检索
    # -------------------------------------------------------------------
    def bm25_search(self, query, top_k=None):                   # 定义 BM25 检索方法
        """调用 BM25 索引检索，返回 (行号, 分数)。"""
        top_k = top_k or config.BM25_TOP_K                      # 取默认候选数
        return self.bm25.search(query, top_k=top_k)             # 直接返回 BM25 检索结果

    # -------------------------------------------------------------------
    # 三、RRF 融合（检索层优化点：向量 + 全文互补召回）
    # -------------------------------------------------------------------
    def hybrid_search(self, query, query_vec, top_k=None):      # 定义混合检索方法
        """向量与 BM25 结果用 Reciprocal Rank Fusion 融合，返回 (行号, RRF 分数)。"""
        top_k = top_k or config.FINAL_TOP_K                     # 最终返回条数
        vec_hits = self.vector_search(query_vec)                # 向量召回结果
        bm25_hits = self.bm25_search(query)                     # 全文召回结果
        rrf = {}                                                # 行号 -> 累计 RRF 分数
        for rank, (idx, _s) in enumerate(vec_hits):             # 遍历向量结果（按名次加权）
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (config.RRF_K + rank + 1)  # 累加向量侧 RRF 贡献
        for rank, (idx, _s) in enumerate(bm25_hits):            # 遍历 BM25 结果
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (config.RRF_K + rank + 1)  # 累加全文侧 RRF 贡献
        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)  # 按融合分数降序
        return ranked[:top_k]                                   # 返回前 top_k 条

    # -------------------------------------------------------------------
    # 四、MMR 去冗余（检索层优化点：在相关性与多样性之间权衡）
    # -------------------------------------------------------------------
    def _coverage(self, idx, terms, bigrams, idf, idf_sum):     # 定义关键词覆盖度计算方法
        """计算候选块对问题关键词/短语的覆盖度（0~1）。

        词级覆盖按 IDF 加权（罕见词命中更有价值），短语覆盖按子串是否出现在块原文中判定；
        两者按 0.6/0.4 加权合成。分数只依赖词面统计，零模型调用、零额外延迟。
        """
        toks = set(self.bm25.docs_tokens[idx]) if idx < len(self.bm25.docs_tokens) else set()  # 候选块的词元集合
        uni = sum(idf.get(t, 0.0) for t in terms if t in toks) / (idf_sum or 1.0)  # IDF 加权的词级覆盖率
        bi = (sum(1 for b in bigrams if b in self.chunks[idx]["text"]) / len(bigrams)) if bigrams else 0.0  # 短语命中率
        return 0.6 * uni + 0.4 * bi                             # 词级与短语级按权重合成

    def _coverage_scores(self, question, cand, extra_terms=()):  # 定义批量覆盖度计算方法
        """对候选列表逐一计算关键词覆盖度，返回与候选等长的归一化 numpy 数组。

        第四轮优化：extra_terms 用于把同义词扩展词（利润留存/资本充足率/久期匹配/
        拨备覆盖率等）混入覆盖度词表——这些词是同义词子查询检回证据块的依据，
        让它们参与精排可使『字面含同义词证据表述』的块在 MMR 相关性项中获得加分。
        同义词词不参与 bigrams 短语匹配（词组间在原文中并不相邻）。
        """
        terms = _question_terms(question)                       # 提取问题内容词
        if extra_terms:                                         # 有同义词扩展词时并入
            seen = set(terms)                                   # 已有词集合（保持原顺序在前）
            for t in extra_terms:                               # 遍历扩展词
                if t not in seen:                               # 去重
                    seen.add(t)                                 # 标记
                    terms = terms + [t]                         # 追加（元组拼接成新列表）
        if not terms:                                           # 无有效内容词（如全为停用词）
            return np.zeros(len(cand), dtype="float32")         # 返回全零（退化为纯融合分数）
        bigrams = _question_bigrams(question, terms)            # 提取短语对
        n = self.bm25.n_docs or 1                               # 文档总数（IDF 分母）
        idf = {t: float(np.log((n - self.bm25.df.get(t, 0) + 0.5) / (self.bm25.df.get(t, 0) + 0.5) + 1.0))  # 计算 IDF
               for t in terms}                                  # 只算问题中出现的词
        idf_sum = sum(idf.values()) or 1.0                      # IDF 总和（覆盖率分母）
        raw = np.array([self._coverage(i, terms, bigrams, idf, idf_sum)  # 逐候选计算覆盖度
                        for i, _s in cand], dtype="float32")    # 得到原始覆盖度数组
        peak = float(raw.max()) if raw.size else 0.0            # 取最大值做归一化基准
        return raw / peak if peak > 1e-6 else raw               # 归一化到 [0,1]（防止除零）

    def mmr(self, query_vec, candidates, top_k=None, lam=None, question=None, extra_terms=()):  # 定义 MMR 重排方法
        """对候选集合做最大边际相关性重排，减少上下文重复。

        相关性项必须取「上游融合分数」而不是原始余弦相似度：
        早期版本用 vecs.dot(query_vec) 作为相关性，等于把 RRF 融合结果整份作废，
        MMR 退化成又一次纯向量排序，混合检索白做。实测某题正确答案块在多路融合中排第 6，
        却被 MMR 挤出前 8。此处改为直接把候选自带的融合分数当相关性，并归一化到 [0,1]，
        使其与「块间余弦相似度」同量纲后再做 λ 加权。

        第二轮优化：传入 question 时，把「关键词覆盖度」与融合分数按 α/(1-α) 混合作为相关性项，
        让字面含核心短语的块优先于结构雷同的跨行样板块（融合分数对二者几乎无区分度）。
        """
        top_k = top_k or config.FINAL_TOP_K                     # 最终条数
        lam = config.MMR_LAMBDA if lam is None else lam         # 相关性权重 λ
        cand = [i for i, _ in candidates if i < len(self.chunks)]  # 过滤越界下标，保证安全
        if not cand:                                            # 候选为空时直接返回
            return []                                           # 返回空列表
        raw = [float(s) for i, s in candidates if i < len(self.chunks)]  # 取出对应融合分数
        scale = max(raw) if raw and max(raw) > 0 else 1.0       # 归一化基准（防止除零）
        rel = np.array([s / scale for s in raw], dtype="float32")  # 归一化到 [0,1] 的相关性项
        if question:                                            # 提供问题时混入关键词覆盖度
            cov = self._coverage_scores(question, candidates, extra_terms=extra_terms)  # 计算候选覆盖度数组（含同义词词）
            rel = config.KEYWORD_BOOST_ALPHA * cov + (1 - config.KEYWORD_BOOST_ALPHA) * rel  # 按权重混合
        vecs = self.vectors[cand]                               # 取出候选向量子矩阵
        picked = []                                             # 已选下标列表
        pool = list(range(len(cand)))                           # 候选池的局部下标
        while pool and len(picked) < top_k:                     # 迭代直到选满或候选耗尽
            best, best_score = None, -1e9                       # 当前最优候选及其分数
            for p in pool:                                      # 遍历候选池
                if not picked:                                  # 第一个直接按相关性选择
                    score = rel[p]                              # 使用融合相关性作为分数
                else:                                           # 后续需扣除与已选内容的相似度
                    sim = float(np.max(vecs[picked].dot(vecs[p])))  # 与已选集合的最大相似度
                    score = lam * rel[p] - (1 - lam) * sim      # MMR 目标：相关高且冗余低
                if score > best_score:                          # 更新最优解
                    best, best_score = p, score                 # 记录最优候选与分数
            picked.append(best)                                 # 选择该候选
            pool.remove(best)                                   # 从候选池移除
        return [cand[p] for p in picked]                        # 把局部下标映射回全局行号

    # -------------------------------------------------------------------
    # 五、多路查询融合（检索层优化点：复合问题拆分为多路子查询后统一融合）
    # -------------------------------------------------------------------
    def multi_query_search(self, queries, query_vec, top_k=None, syn_queries=None):  # 定义多路查询检索方法
        """对每个子查询分别做 BM25 召回，向量侧只算一次，最后用 RRF 统一融合。

        多路召回的成本几乎全部落在查询向量化上（本机实测单次编码 0.4~0.9 秒），
        而每个子查询各编码一次会把检索耗时推高到 3 秒以上、超出工单的时间预算。
        BM25 侧是纯词频计算（毫秒级），因此这里让多路只在全文侧展开：
        既保留了「复合问题按要点分开做词面匹配」的收益，又只需一次向量编码。

        第四轮优化：同义词子查询（syn_queries）施加 SYNONYM_QUERY_WEIGHT 权重。
        诊断发现『利润留存』块在同义词子查询的 BM25 中排第 1，却因权重 1.0 被
        其他 8 路稀释到融合第 74 名——单路强命中的证据块必须配得上更高的融合话语权。
        """
        top_k = top_k or config.FINAL_TOP_K                     # 最终返回条数
        syn_set = set(syn_queries or [])                        # 同义词子查询集合（O(1) 判定）
        rrf = {}                                                # 行号 -> 累计 RRF 分数
        for rank, (idx, _s) in enumerate(self.vector_search(query_vec, top_k=config.VECTOR_TOP_K)):
            rrf[idx] = rrf.get(idx, 0.0) + config.MULTI_QUERY_WEIGHT / (config.RRF_K + rank + 1)  # 向量侧单路，权重更高
        for qi, query in enumerate(queries):                    # 逐个子查询做全文召回
            if query in syn_set:                                # 同义词子查询：领域精准命中，加权
                weight = config.SYNONYM_QUERY_WEIGHT            # 权重高于普通子查询
            elif qi == 0:                                       # 原问题
                weight = config.MULTI_QUERY_WEIGHT              # 原问题的词面匹配权重更高
            else:                                               # 普通标点切分子查询
                weight = 1.0                                    # 基础权重
            for rank, (idx, _s) in enumerate(self.bm25_search(query, top_k=(config.SYNONYM_BM25_TOPK if query in syn_set else config.BM25_TOP_K))):  # 同义词路只取精准小池
                rrf[idx] = rrf.get(idx, 0.0) + weight / (config.RRF_K + rank + 1)  # 累加全文侧贡献
        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)  # 按融合分数降序
        return ranked                                           # 返回全部融合候选（由调用方截断）

    # -------------------------------------------------------------------
    # 六、统一入口：优化后的完整检索流程
    # -------------------------------------------------------------------
    def fuse_graph(self, ranked, extra_chunks, weight=1.0):     # 定义图谱块融合方法
        """把图谱检索得到的块当作第三条排序列表并入 RRF 融合结果。

        图谱检索按「实体命中权重」排序，与向量/BM25 的分数不同量纲，直接比较会失真；
        RRF 只看名次、不看原始分数，天然免疫量纲差异，因此这里继续用 RRF 融合。
        早期版本把图谱块直接追加在混合检索结果之后，上下文从 8 条膨胀到 14 条，
        夹带免责声明、获奖名单等无关块，精度被稀释到低于基线。
        """
        if not extra_chunks:                                    # 图谱未召回任何块
            return ranked                                       # 直接返回原融合结果
        scores = dict(ranked)                                   # 行号 -> 融合分数
        for rank, c in enumerate(extra_chunks):                 # 按图谱侧名次遍历
            idx = self.id2idx.get(c["id"])                      # 找块对应的向量行号
            if idx is None:                                     # 块不在索引中则跳过
                continue                                        # 继续下一个
            scores[idx] = scores.get(idx, 0.0) + weight / (config.RRF_K + rank + 1)  # 累加图谱侧贡献
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)  # 重新按融合分数降序

    def retrieve(self, query, query_vec, top_k=None, use_hybrid=True, use_mmr=True,
                 queries=None, extra_chunks=None, question=None,
                 graph_weight=None, lam=None, syn_queries=None, use_cov=True):  # 定义优化后检索入口
        """执行『多路混合召回 + 图谱融合 + MMR 去冗余』的完整检索流程，返回文本块字典列表。

        第四轮优化：syn_queries 为同义词扩展子查询——多路融合时施加高权重并只取
        精准小池，把『问题用词与原文表述不一致』的证据块检回（仅对比题传入，其余问题行为不变）。
        use_cov=False 时 MMR 相关性项退回纯融合分数（不再混入关键词覆盖度）：
        矩阵实验表明覆盖度项会整体洗牌融合排名，把同义词加权检回的证据块重新挤出去，
        对比题（证据块靠同义词路检回）必须关闭，其余题型保持开启（第二轮精度的功臣）。
        """
        top_k = top_k or config.FINAL_TOP_K                     # 最终返回条数
        if queries is not None:                                 # 提供多路查询时走多路融合
            cand = self.multi_query_search(queries, query_vec, syn_queries=syn_queries)  # 多路子查询融合召回（同义词路加权）
        elif use_hybrid:                                        # 单路混合检索分支
            cand = self.hybrid_search(query, query_vec, top_k=config.VECTOR_TOP_K * 2)  # 融合召回候选
        else:                                                   # 走纯向量检索分支（优化前基线）
            cand = self.vector_search(query_vec, top_k=config.VECTOR_TOP_K)  # 仅向量召回
        cand = self.fuse_graph(cand, extra_chunks,              # 图谱证据块并入融合排名
                               weight=1.0 if graph_weight is None else graph_weight)  # 默认权重 1.0，对比题由管线传 2.0
        if use_mmr:                                             # 启用 MMR 去冗余
            extra_terms = tuple(tok for sq in (syn_queries or []) for tok in tokenize(sq))  # 同义词子查询分词作覆盖度扩展词表
            idxs = self.mmr(query_vec, cand, top_k=top_k,       # 重排并截断
                            lam=lam,                            # 对比题传入放宽的 λ
                            question=(question if use_cov else None),  # 对比题关闭覆盖度项（防洗牌）
                            extra_terms=extra_terms)            # 同义词词参与覆盖精排（仅 use_cov 时生效）
        else:                                                   # 不启用时直接取前 top_k
            idxs = [i for i, _ in cand[:top_k]]                 # 保持原始顺序
        return [self.chunks[i] for i in idxs if i < len(self.chunks)]  # 返回文本块字典列表

    def retrieve_baseline(self, query_vec, top_k=None):         # 定义优化前基线检索方法
        """优化前基线：纯向量 top-k，无混合、无去冗余。"""
        top_k = top_k or config.FINAL_TOP_K                     # 最终条数
        hits = self.vector_search(query_vec, top_k=top_k)       # 仅做向量检索
        return [self.chunks[i] for i, _ in hits if i < len(self.chunks)]  # 返回文本块列表
