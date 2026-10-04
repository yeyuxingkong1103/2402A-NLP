# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-LightRAG优化
LightRAG：图结构 + 双层检索机制。
  低层次（局部）检索：具体实体/属性 → 向量库相似匹配；
  高层次（全局）检索：概念/主题关键词 → 知识图谱沿关系扩展。
参考：https://github.com/HKUDS/LightRAG
"""
import config
from graph_builder import KnowledgeGraph
from llm import call_llm
from rag import RAG, tokenize


class LightRAG:
    def __init__(self, docs):
        self.docs = docs
        self.graph = KnowledgeGraph()
        self.graph.build(docs)
        self.vector = RAG(docs)   # 局部检索复用语料向量

    def _local_keywords(self, query):
        """局部关键词：具体实体（公司名、人名、金额等）。"""
        return [t for t in tokenize(query) if len(t) >= 2]

    def _global_keywords(self, query):
        """全局关键词：概念性词（行业/领域/标准/结构等）。"""
        hints = ["行业", "领域", "标准", "项目", "工程", "结构", "比重",
                 "上游", "下游", "军用", "销售部", "关联方"]
        return [k for k in hints if k in query]

    def _local_retrieve(self, query, top_k=5):
        return self.vector.retrieve(query, top_k)

    def _global_retrieve(self, query, top_k=5):
        expanded = set()
        for kw in self._global_keywords(query):
            for ent in self.graph.entities:
                if kw in ent:
                    expanded |= self.graph.neighbors(ent, depth=1)
        docs, seen = [], set()
        for ent in expanded:
            meta = self.graph.entities.get(ent)
            if meta and meta["doc_id"] not in seen:
                seen.add(meta["doc_id"])
                for d in self.docs:
                    if d["id"] == meta["doc_id"]:
                        docs.append(d)
        return docs[:top_k]

    def query(self, question, top_k=5):
        local = self._local_retrieve(question, top_k)
        global_ = self._global_retrieve(question, top_k)
        uniq, seen = [], set()
        for c in local + global_:
            if c["text"] not in seen:
                seen.add(c["text"])
                uniq.append(c)
        ctx = "\n".join(c["text"] for c in uniq[:top_k])
        ans = call_llm(f"根据以下上下文回答问题：\n{ctx}\n\n问题：{question}")
        return {
            "contexts": [c["text"] for c in uniq[:top_k]],
            "answer": ans or "（离线模式）检索到的相关内容见上下文。",
            "local": [c["text"] for c in local],
            "global": [c["text"] for c in global_],
        }
