# -*- coding: utf-8 -*-
"""
混合检索模块（工单06，供工单03/05/07复用）
工单编号：人工智能NLP-RAG-混合检索任务
功能：
  1. BM25 全文检索（jieba分词 + 倒排索引），支持布尔查询(AND/OR/NOT)、
     短语匹配("...")、模糊匹配（编辑距离≤1的词项近似）；
  2. 多检索策略：向量检索 / 全文检索 / 混合检索（权重可配）；
  3. 三种重排器：LLM重排器、TF-IDF重排器、用户反馈自适应重排器。
"""
import os        # 索引目录与文件读写
import re        # 查询解析中的短语/操作符正则
import json      # BM25 索引与反馈记录的 JSON 持久化
import difflib   # 模糊匹配的编辑距离相似度计算

import jieba                  # 中文分词
from rank_bm25 import BM25Okapi  # BM25 打分实现（需 pip install rank_bm25）

from config import INDEX_DIR  # 索引持久化目录（与向量库共用）


def tokenize(text):
    """jieba 搜索引擎模式分词"""
    # cut_for_search 在精确模式基础上再切长词，召回更多词项，更适合检索场景
    return list(jieba.cut_for_search(str(text)))


class BM25Index:
    """BM25 全文检索索引（基于倒排思想，rank_bm25 实现）"""

    def __init__(self):
        self.texts = []       # 原始分块文本，与 metadatas 下标对齐
        self.metadatas = []   # 元数据 [{"page":.., "source":..}, ...]
        self._bm25 = None     # BM25Okapi 模型对象，build 后才可用
        self._tokenized = []  # 分词后的语料缓存

    # ── 构建 / 加载 ────────────────────────────────────────
    def build(self, texts, metadatas):
        self.texts = list(texts)
        self.metadatas = list(metadatas)
        print(f"[BM25] 分词 {len(self.texts)} 条 ...")
        # 对全部文本分词，得到 BM25 需要的"词袋"语料（list[list[str]]）
        self._tokenized = [tokenize(t) for t in self.texts]
        # BM25Okapi 在构造时统计 df（文档频率）和平均文档长度，build 一次即可
        self._bm25 = BM25Okapi(self._tokenized)
        return self  # 返回 self 支持链式调用

    def save(self, name):
        # 只持久化文本与元数据；BM25 模型由 load 后重新 build 得出（分词很快，不必序列化模型）
        os.makedirs(INDEX_DIR, exist_ok=True)
        data = {"texts": self.texts, "metadatas": self.metadatas}
        # 后缀 .bm25.json 与向量索引 {name}.json 区分开，避免互相覆盖
        with open(os.path.join(INDEX_DIR, f"{name}.bm25.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[BM25] 已保存 {name}")

    @classmethod
    def load(cls, name, auto_build=True):
        path = os.path.join(INDEX_DIR, f"{name}.bm25.json")
        if not os.path.exists(path):
            return None  # 索引文件不存在返回 None，由调用方决定是否先 build
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        idx = cls()
        # 表达式写法：auto_build 为 True 时构建 BM25；None 时跳过
        idx.build(data["texts"], data["metadatas"]) if auto_build else None
        if not auto_build:
            # 不立即构建的场景（如只读文本列表）：先把数据挂上，用前再 build
            idx.texts, idx.metadatas = data["texts"], data["metadatas"]
        return idx

    # ── 查询解析：布尔 / 短语 / 模糊 ────────────────────────
    @staticmethod
    def parse_query(query):
        """解析查询：提取短语("...")、布尔操作符(AND/OR/NOT)、普通词项"""
        # 两组捕获组兼容中英文引号：取每组第一个非空匹配结果
        phrases = re.findall(r'"([^"]+)"|"(.+?)"', query)
        phrases = [a or b for a, b in phrases]
        # 从原查询中删掉短语部分，剩余部分再做分词与布尔解析
        q = re.sub(r'"[^"]*"', " ", query)
        must, should, must_not = [], [], []  # AND 词 / OR 词 / NOT 排除词
        tokens = [t for t in tokenize(q) if t.strip()]  # 分词并滤掉纯空白 token
        i = 0
        while i < len(tokens):
            t = tokens[i].upper()  # 操作符不区分大小写（and/And 也识别）
            if t == "AND" and i + 1 < len(tokens):
                # "X AND Y"：把 Y 加入必须包含列表，i 跳 2 跳过已消费的操作符和词
                must.append(tokens[i + 1]); i += 2
            elif t == "OR" and i + 1 < len(tokens):
                should.append(tokens[i + 1]); i += 2
            elif t == "NOT" and i + 1 < len(tokens):
                must_not.append(tokens[i + 1]); i += 2
            else:
                # 普通词项默认进 should；操作符自身出现在边界位置则丢弃
                if t not in ("AND", "OR", "NOT"):
                    should.append(tokens[i])
                i += 1
        return {"phrases": phrases, "must": must, "should": should, "must_not": must_not}

    @staticmethod
    def _fuzzy_in(token, text):
        """模糊匹配：词项或其编辑距离≤1的近似词出现在文本中"""
        if token in text:
            return True  # 精确命中直接返回，省去模糊计算
        # 从文本中抽相似词做编辑距离近似（限制长度防止全表扫描过慢）
        # 正则：以 token 首字符开头、总长不超过 len(token)+2 的词——
        # 编辑距离≤1 的近似词首字符大概率相同，先粗筛再精算
        for w in set(re.findall(re.escape(token[0]) + r"\w{0,%d}" % (len(token) + 1), text)):
            # SequenceMatcher 比率 ≥0.85 近似视为"编辑距离≤1"级别的相似
            if difflib.SequenceMatcher(None, token, w).ratio() >= 0.85:
                return True
        return False

    def search(self, query, top_k=10, fuzzy=True):
        """全文检索：BM25 打分 + 布尔/短语过滤"""
        parsed = self.parse_query(query)  # 先解析出短语/布尔/普通词
        # BM25 打分只用 should+must 的词（NOT 词不该参与相关性打分）；
        # 全被过滤光时回退用原始 query，保证至少有打分依据
        base_query = " ".join(parsed["should"] + parsed["must"]) or query
        scores = self._bm25.get_scores(tokenize(base_query))  # 返回每条语料的 BM25 分
        # 先多召回 6 倍候选（top_k*6），过滤后可能剩不到 top_k，多备些候选
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: top_k * 6]

        results = []
        for i in order:
            text = self.texts[i]
            if scores[i] <= 0:
                continue  # BM25 分 ≤0 说明与查询词无交集，直接跳过
            # 布尔约束
            # AND：任一 must 词缺失即淘汰（must 全部必须出现）
            if any(w not in text for w in parsed["must"]):
                continue
            # NOT：出现任一排除词即淘汰
            if any(w in text for w in parsed["must_not"]):
                continue
            if parsed["should"] and fuzzy:
                # 模糊模式：should 词至少有一个（近似）命中才保留
                if not any(self._fuzzy_in(w, text) for w in parsed["should"]):
                    continue
            elif parsed["should"] and not any(w in text for w in parsed["should"]):
                continue
            # 短语约束：所有 "..." 短语必须以连续子串形式出现（顺序也不能乱）
            if parsed["phrases"] and not all(p in text for p in parsed["phrases"]):
                continue
            results.append({"text": text, "page": self.metadatas[i].get("page"),
                            "source": self.metadatas[i].get("source"),
                            "score": float(scores[i])})  # 转 float 便于 JSON 序列化
            if len(results) >= top_k:
                break  # 凑够 top_k 提前退出，省去遍历剩余候选
        return results


# ── 混合检索 ────────────────────────────────────────────────
def _norm(scores):
    """min-max 归一化到 0~1"""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    # 分母加 1e-8：所有分数相同时 hi==lo，防除零；此时全部归一为 0
    return [(s - lo) / (hi - lo + 1e-8) for s in scores]


def hybrid_search(vector_store, bm25_index, query,
                  mode="hybrid", vec_w=0.7, ft_w=0.3, top_k=5, fetch_k=20):
    """统一检索入口
    mode: vector=仅向量 | fulltext=仅BM25全文 | hybrid=加权混合
    融合算法：各路分数 min-max 归一化后加权求和（可配权重）
    """
    if mode == "vector":
        # 纯向量模式：语义匹配强，但精确词（编号/专有名词）召回弱
        return vector_store.search(query, top_k=top_k)
    if mode == "fulltext":
        # 纯全文模式：精确词匹配强，但同义改写召回弱
        return bm25_index.search(query, top_k=top_k)

    # 混合模式：两路各多召 fetch_k 条再融合（fetch_k 应明显大于 top_k）
    v_hits = vector_store.search(query, top_k=fetch_k)
    f_hits = bm25_index.search(query, top_k=fetch_k)

    fused = {}
    # 两路分数分布不同（余弦 vs BM25），必须先各自归一化到 0~1 才能加权比较
    for h, s in zip(v_hits, _norm([h["score"] for h in v_hits])):
        key = h["text"]
        fused[key] = {**h, "vs": s, "fs": 0.0}  # vs=向量归一分，fs=全文归一分
    for h, s in zip(f_hits, _norm([h["score"] for h in f_hits])):
        key = h["text"]
        if key in fused:
            fused[key]["fs"] = s  # 两路同时命中：补上全文分
        else:
            # 仅全文命中：向量分记 0（该块没进入向量召回）
            fused[key] = {**h, "vs": 0.0, "fs": s}

    for h in fused.values():
        # 加权求和：默认 vec_w=0.7 偏重语义；文本以键值覆盖原 score
        h["score"] = vec_w * h["vs"] + ft_w * h["fs"]
    # 按融合分降序取 top_k
    ranked = sorted(fused.values(), key=lambda x: -x["score"])[:top_k]
    return ranked


# ── 三种重排器 ──────────────────────────────────────────────
# 重排 Prompt：只输出编号 JSON 数组，降低解析失败概率
LLM_RERANK_PROMPT = """你是检索结果重排器。请根据与问题的相关性，对下列片段重新排序。
只输出排序后的编号列表（JSON数组，如 [3,1,2]），不要解释。

【问题】{query}

【候选片段】
{candidates}

【排序结果】"""


def llm_rerank(query, hits, client, top_m=8):
    """基于LLM的重排器：让LLM对候选片段按相关性排序"""
    cands = hits[:top_m]  # 只重排前 top_m 条（LLM 上下文有限且推理慢）
    # 每条候选截断到 150 字符够判断相关性，控制 prompt 长度
    cand_text = "\n".join(f"[{i+1}] {h['text'][:150]}" for i, h in enumerate(cands))
    try:
        # temperature=0.0：排序结果要确定性，不可随机
        raw = client.generate(LLM_RERANK_PROMPT.format(query=query, candidates=cand_text),
                              temperature=0.0, num_predict=80)
        import json as _json  # 局部导入避免与模块级 json 重名混淆
        # 只匹配"数字+逗号+空白"组成的数组，滤掉 LLM 可能附带的解释文字
        m = re.search(r"\[[\d,\s]+\]", raw)
        order = _json.loads(m.group(0)) if m else []
        # 按编号（1 起）取出对应候选；越界编号（1<=i<=len）直接丢弃
        reranked = [cands[i - 1] for i in order if 1 <= i <= len(cands)]
        # LLM 可能漏掉部分编号：把未出现在结果中的候选按原顺序补回尾部
        reranked += [h for h in cands if h not in reranked]
        return reranked + hits[top_m:]  # top_m 之后的候选不参与重排，原样拼接
    except Exception:
        # LLM 输出不合法/调用失败：降级返回原始顺序，保证检索不中断
        return hits


def tfidf_rerank(query, hits, top_m=10):
    """基于TF-IDF的重排器：候选片段集上构建TF-IDF向量，与query余弦相似度重排"""
    cands = hits[:top_m]
    # 把每个候选和 query 都分词，query 放最后作为待比较向量
    docs = [tokenize(h["text"]) for h in cands] + [tokenize(query)]
    # 构建 IDF
    import math  # 局部导入 log/sqrt
    N = len(docs)  # 文档总数（含 query 自身）
    df = {}  # 词 -> 出现该词的文档数
    for d in docs:
        # set(d)：同一文档内重复词只计一次（df 按文档计不按词频计）
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    # 标准 IDF 公式变体：log((N+1)/(df+1)) + 1，加 1 平滑防负值/除零
    idf = {w: math.log((N + 1) / (c + 1)) + 1 for w, c in df.items()}

    def tfidf_vec(tokens):
        """把词列表转成稀疏 TF-IDF 向量（dict: 词->权重）"""
        tf = {}  # 词 -> 词频计数
        for w in tokens:
            tf[w] = tf.get(w, 0) + 1
        # TF = 词数/总词数（归一化的词频），乘 IDF 得 TF-IDF；
        # 语料中没见过的词 idf 取 0（df.get 默认值）
        return {w: (c / len(tokens)) * idf.get(w, 0) for w, c in tf.items()}

    vecs = [tfidf_vec(d) for d in docs]
    qv = vecs[-1]  # 最后一个是 query 的向量

    def cos(a, b):
        """稀疏 dict 向量的余弦相似度"""
        # 并集遍历：任一向量缺失的词按 0 计
        dot = sum(a.get(w, 0) * b.get(w, 0) for w in set(a) | set(b))
        na = math.sqrt(sum(v * v for v in a.values()))  # 向量 a 的模
        nb = math.sqrt(sum(v * v for v in b.values()))  # 向量 b 的模
        # 加 1e-8 防零向量除零
        return dot / (na * nb + 1e-8)

    # 按与 query 的余弦相似度降序排候选下标，再重排成新列表
    scored = sorted(range(len(cands)), key=lambda i: -cos(vecs[i], qv))
    reranked = [cands[i] for i in scored]
    return reranked + hits[top_m:]  # top_m 之后原样拼接


FEEDBACK_FILE = os.path.join(INDEX_DIR, "feedback.json")  # 反馈记录持久化文件


class FeedbackReranker:
    """基于用户反馈的自适应重排器：
    用户采纳过的来源页（反馈记录）在后续检索中获得加权，
    反馈随使用累积，实现自适应个性化排序。"""

    def __init__(self, boost=0.15):
        self.boost = boost  # 每次历史采纳带来的加分幅度
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, encoding="utf-8") as f:
                self.fb = json.load(f)  # 加载历史反馈，跨次运行累积
        else:
            self.fb = {}  # {"来源名|页码": 次数}

    def rerank(self, hits):
        def key(h):
            # 构造 "来源名|页码" 反馈键，与 record 写入的格式一致
            k = f"{h.get('source')}|{h.get('page')}"
            # 返回负的"分+反馈加权"：sorted 升序排列即等效于分数降序；
            # 采纳次数越多（fb 值越大），排得越靠前
            return -h["score"] - self.boost * self.fb.get(k, 0)
        return sorted(hits, key=key)

    def record(self, hits):
        """把本轮被采纳的来源记入反馈"""
        for h in hits[:3]:  # 只记前 3 条（用户实际查看的头部结果）
            k = f"{h.get('source')}|{h.get('page')}"
            self.fb[k] = self.fb.get(k, 0) + 1  # 该来源采纳次数 +1
        os.makedirs(INDEX_DIR, exist_ok=True)
        # ensure_ascii=False：键含中文来源名，直接存 UTF-8 可读
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(self.fb, f, ensure_ascii=False)


# 重排器注册表：名称 -> 函数，供 apply_rerank 按名分发
RERANKERS = {"llm": llm_rerank, "tfidf": tfidf_rerank}


def apply_rerank(rerank_name, query, hits, client=None, top_k=5):
    """统一重排入口: rerank_name ∈ {none, llm, tfidf, feedback}"""
    if rerank_name == "none" or not hits:
        return hits  # 不重排或无候选，原样返回
    if rerank_name == "feedback":
        # 反馈重排器是类：临时实例化用当前反馈文件重排，截 top_k
        return FeedbackReranker().rerank(hits)[:top_k]
    fn = RERANKERS.get(rerank_name)
    if fn is None:
        return hits  # 未知重排器名：降级不重排（容错）
    # 函数签名不同：tfidf 不需要 client，llm 需要——按名构造参数元组
    args = (query, hits) if rerank_name == "tfidf" else (query, hits, client)
    return fn(*args)[:top_k]  # 解包调用后截取 top_k
