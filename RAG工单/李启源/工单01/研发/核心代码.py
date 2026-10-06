"""工单01：基于PDF文档的问答系统的可运行核心代码。"""
import re

def run(query, documents):
    words = set(re.findall(r"[一-鿿]|[a-zA-Z0-9_]+", query.lower()))
    hits = []
    for name, text in documents.items():
        score = sum(text.lower().count(word) for word in words)
        if score:
            hits.append((score, name, text[:300]))
    hits.sort(reverse=True)
    return hits[:3] or [(0, "提示", "没有找到相关内容")]

if __name__ == "__main__":
    documents = {"演示文档.txt": "公司注册资本为1000万元，主营电子信息业务"}
    print(run("注册资本", documents))
