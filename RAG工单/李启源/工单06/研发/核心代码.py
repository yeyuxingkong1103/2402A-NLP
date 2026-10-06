"""工单06：混合检索的可运行核心代码。"""
import re

def run(query, documents, mode="hybrid"):
    words = set(re.findall(r"[一-鿿]|[a-zA-Z0-9_]+", query.lower()))
    results = []
    for name, text in documents.items():
        keyword = sum(text.lower().count(word) for word in words)
        vector_like = len(words & set(re.findall(r"[一-鿿]|[a-zA-Z0-9_]+", text.lower()))) / max(1, len(words))
        score = keyword if mode == "keyword" else vector_like if mode == "vector" else keyword * 0.6 + vector_like * 0.4
        results.append((score, name, text[:200]))
    return sorted(results, reverse=True)[:3]

if __name__ == "__main__":
    docs = {"说明书.txt": "公司注册资本为1000万元，主营电子信息业务"}
    print(run("注册资本", docs, "hybrid"))
