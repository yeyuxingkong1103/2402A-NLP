"""工单05：Query理解及多轮对话优化的可运行核心代码。"""
import re

def run(query, documents, history):
    company = re.search(r"[一-鿿]{2,}公司", query)
    if company:
        history["topic"] = company.group()
    if re.search(r"他|她|这个公司|该公司", query) and history.get("topic"):
        query = re.sub(r"他|她|这个公司|该公司", history["topic"], query)
    words = set(query.lower().split())
    return query, [text[:200] for text in documents.values() if any(word in text for word in words)]

if __name__ == "__main__":
    history = {}
    docs = {"知识库.txt": "武汉兴图新科电子股份有限公司参与制定技术标准"}
    print(run("武汉兴图新科电子股份有限公司的情况", docs, history))
    print(run("这个公司的技术标准是什么", docs, history))
