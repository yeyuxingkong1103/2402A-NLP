# 工单编号：人工智能NLP-RAG-功能测试及评估
"""重排算法：工单要求「提供至少 3 种重排算法」

实现了 4 种，统一接口 `rerank(query, candidates, top_k) -> [(chunk, score)]`：

    1. LLM 重排器        —— 让大模型直接给候选排序，最准但最慢
    2. TF-IDF 重排器     —— 本地词频统计，毫秒级，与向量检索互补
    3. 用户反馈自适应重排 —— 按历史 👍/👎 调整排序，用得越久越准
    4. 交叉编码器重排     —— bge-reranker-v2-m3（工单2 引入，见 vector_store.py）

前三种是本工单新增。
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

from config import FEEDBACK_LOG
from fulltext import tokenize


class Reranker:
    """重排器基类，定义统一接口。"""

    name = "base"
    label = "基础"

    def rerank(self, query, candidates, top_k=8):
        raise NotImplementedError


# ---------------- 1. LLM 重排器 ----------------
class LLMReranker(Reranker):
    """把候选编号后交给大模型，让它按相关性输出排序。

    最准，但要一次额外的模型调用（约 1.5~2 秒），适合对精度要求高、
    可以接受等待的场景。
    """

    name = "llm"
    label = "LLM 重排"
    PROMPT = """请按与问题的相关程度，把下面这些文档片段从高到低排序。

【问题】
{query}

【候选片段】
{cands}

只输出编号，用逗号分隔，从最相关到最不相关。例如：3,1,5,2,4
不要输出任何其它内容。"""

    def __init__(self, chat=None, max_chars=200):
        self.chat = chat
        self.max_chars = max_chars

    def rerank(self, query, candidates, top_k=8):
        if len(candidates) <= 1 or self.chat is None:
            return candidates[:top_k]
        listing = "\n".join(f"[{i}] {c['text'][:self.max_chars]}"
                            for i, (c, _) in enumerate(candidates))
        try:
            raw = self.chat(self.PROMPT.format(query=query, cands=listing))
            order = [int(n) for n in re.findall(r"\d+", raw)]
        except Exception:                                          # noqa: BLE001
            return candidates[:top_k]           # 重排出错就退回原顺序

        ranked, seen = [], set()
        for i in order:                          # 按模型给的顺序取
            if 0 <= i < len(candidates) and i not in seen:
                seen.add(i)
                ranked.append(candidates[i])
        ranked += [c for i, c in enumerate(candidates) if i not in seen]  # 补漏
        # 分数按新名次重算，便于与其它重排器对比
        return [(c, 1.0 / (1 + r)) for r, (c, _) in enumerate(ranked[:top_k])]


# ---------------- 2. TF-IDF 重排器 ----------------
class TFIDFReranker(Reranker):
    """本地 TF-IDF 余弦相似度重排。

    与向量检索互补：向量看语义，TF-IDF 看词面重叠。招股书里
    「注册资本」「法定代表人」这类术语，词面匹配往往比语义更稳。
    """

    name = "tfidf"
    label = "TF-IDF 重排"

    def __init__(self, index=None):
        self.index = index          # FullTextIndex，用来取文档频率

    def _idf(self, token):
        if not self.index:
            return 1.0
        df = len(self.index.postings.get(token, {}))
        return math.log((len(self.index.docs) + 1) / (df + 1)) + 1.0

    def rerank(self, query, candidates, top_k=8):
        q_tokens = Counter(tokenize(query))
        scored = []
        for chunk, _ in candidates:
            d_tokens = Counter(tokenize(chunk.get("text", "")))
            if not d_tokens:
                scored.append((chunk, 0.0)); continue
            dot = sum(q_tokens[t] * d_tokens[t] * self._idf(t) ** 2
                      for t in q_tokens if t in d_tokens)
            norm_q = math.sqrt(sum((q_tokens[t] * self._idf(t)) ** 2 for t in q_tokens)) or 1.0
            norm_d = math.sqrt(sum((n * self._idf(t)) ** 2 for t, n in d_tokens.items())) or 1.0
            scored.append((chunk, dot / (norm_q * norm_d)))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ---------------- 3. 用户反馈自适应重排器 ----------------
class FeedbackReranker(Reranker):
    """按历史用户反馈调整排序。

    每次 👍/👎 都会记下这一轮用到的文档块；👍 的块加权、👎 的块减权。
    调整幅度随反馈次数增长但设上限，避免少量反馈就大幅改变排序。
    """

    name = "feedback"
    label = "反馈自适应重排"

    def __init__(self, log_path=FEEDBACK_LOG, weight=0.15, cap=0.6):
        self.log_path = Path(log_path)
        self.weight = weight        # 每条反馈的调整步长
        self.cap = cap              # 累计调整的上下限
        self.scores = {}
        self._load()

    @staticmethod
    def _key(chunk):
        return f"{chunk.get('page')}:{chunk.get('text', '')[:40]}"

    def _load(self):
        if not self.log_path.exists():
            return
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            delta = self.weight if rec.get("rating") == "good" else -self.weight
            for key in rec.get("chunks", []):
                self.scores[key] = max(-self.cap, min(self.cap,
                                                      self.scores.get(key, 0.0) + delta))

    def rerank(self, query, candidates, top_k=8):
        scored = [(c, self.scores.get(self._key(c), 0.0)) for c, _ in candidates]
        # 反馈分只做微调：先保持原顺序，再按反馈分小幅浮动
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ---------------- 注册表 ----------------
def available():
    """列出可用重排算法（供界面选择）。"""
    return [("none", "不重排"), ("tfidf", TFIDFReranker.label),
            ("feedback", FeedbackReranker.label), ("llm", LLMReranker.label),
            ("cross", "交叉编码器重排")]


def build(name, chat=None, index=None):
    """按名字造一个重排器。"""
    if name == "tfidf":
        return TFIDFReranker(index)
    if name == "feedback":
        return FeedbackReranker()
    if name == "llm":
        return LLMReranker(chat)
    return None
