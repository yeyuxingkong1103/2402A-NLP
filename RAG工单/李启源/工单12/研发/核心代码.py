"""工单12：LightRAG优化的可运行核心代码。"""
import re

def run(documents):
    graph = {}
    for text in documents:
        entities = re.findall(r"[一-鿿]{2,}(?:公司|中心|部门|行业)", text)
        for left, right in zip(entities, entities[1:]):
            graph.setdefault(left, set()).add(right)
            graph.setdefault(right, set()).add(left)
    return {key: sorted(value) for key, value in graph.items()}

if __name__ == "__main__":
    print(run(["武汉兴图新科电子股份有限公司与电子信息行业研发中心相关" ]))
