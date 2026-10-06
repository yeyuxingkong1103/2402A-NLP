"""工单02：PDF问答系统优化的可运行核心代码。"""
import re

def run(query, documents):
    chunks = []
    for name, text in documents.items():
        for start in range(0, len(text), 160):
            chunks.append((name, text[start:start + 180]))
    words = set(re.findall(r"[一-鿿]|[a-zA-Z0-9_]+", query.lower()))
    scored = [(sum(part.lower().count(word) for word in words), name, part) for name, part in chunks]
    return sorted(scored, reverse=True)[:3]

if __name__ == "__main__":
    documents = {"演示文档.txt": "公司注册资本为1000万元，主营电子信息业务"}
    print(run("注册资本", documents))
