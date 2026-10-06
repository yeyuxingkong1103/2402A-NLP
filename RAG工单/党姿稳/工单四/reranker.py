# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
重排模块：对混合检索召回的候选块，用 LLM 做相关性重排（Rerank）。
混合召回解决"捞得到"，重排解决"排得准"——这是本次优化的关键一环。
"""
from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


class LLMReranker:
    """基于大模型的重排器：给定查询与候选块，输出最相关块的序号"""

    def __init__(self):
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    def rerank(self, query, candidates, top_k=5):
        """
        candidates: [(chunk_dict, score), ...]
        返回重排后的 [(chunk_dict, score), ...]，长度 <= top_k
        """
        if len(candidates) <= top_k:
            return candidates

        # 构造候选清单（截断，控制 token）
        listing = []
        for i, (c, _s) in enumerate(candidates):
            txt = (c["text"] if isinstance(c, dict) else c)[:220].replace("\n", " ")
            listing.append(f"[{i}] {txt}")
        prompt = f"""我在为一篇招股说明书做检索问答。下面是用户问题和检索到的候选文档片段（编号从0开始）。

【用户问题】
{query}

【候选片段】
{chr(10).join(listing)}

请从这些片段中，选出最可能包含该问题答案的 {top_k} 个片段，按相关性从高到低排序。
只输出片段编号，用英文逗号分隔，不要输出任何其他内容。例如：3,0,7,1,5"""

        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
            )
            nums = self._parse(resp.choices[0].message.content, len(candidates))
        except Exception as e:
            print(f"[重排] LLM 重排失败，回退原顺序: {e}")
            return candidates[:top_k]

        if not nums:
            return candidates[:top_k]
        # 去重并补齐，保证返回 top_k 个
        seen, ordered = set(), []
        for n in nums + list(range(len(candidates))):
            if n not in seen:
                seen.add(n)
                ordered.append(candidates[n])
            if len(ordered) == top_k:
                break
        return ordered

    @staticmethod
    def _parse(text, n_candidates):
        """解析 LLM 返回的编号串"""
        import re
        nums = [int(x) for x in re.findall(r'\d+', text or "")]
        return [n for n in nums if 0 <= n < n_candidates]
