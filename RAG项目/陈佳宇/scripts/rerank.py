from sentence_transformers import CrossEncoder

class BGERerank:
    def __init__(self):
        self.model = CrossEncoder("BAAI/bge-reranker-base")

    def rerank(self, query, docs, top_k=2):
        pairs = [[query, item["text"]] for item in docs]
        scores = self.model.predict(pairs)
        for idx, item in enumerate(docs):
            item["rerank_score"] = float(scores[idx])
        docs_sorted = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        return docs_sorted[:top_k]

if __name__ == "__main__":
    rk = BGERerank()
    test_docs = [
        {"text":"Alice周末会去美术馆看画展","score":0.66},
        {"text":"Alice喜欢吃草莓蛋糕","score":0.62}
    ]
    res = rk.rerank("Alice周末去哪里", test_docs)
    print(res)
