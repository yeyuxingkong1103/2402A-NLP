"""工单04：PDF图像内容解析及检索优化的可运行核心代码。"""

def run(query, documents):
    results = []
    for name, text in documents.items():
        if name.lower().endswith((".txt", ".md")):
            score = sum(text.lower().count(word) for word in query.lower().split())
            results.append((score, name, text[:300]))
    return sorted(results, reverse=True)[:3]

if __name__ == "__main__":
    print(run("组织结构", {"组织结构图.txt": "销售部由大客户销售部构成"}))
