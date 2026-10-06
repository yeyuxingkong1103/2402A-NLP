"""工单11：Embeddings模型微调的可运行核心代码。"""
import re

def run(pairs):
    vocabulary = {}
    for query, positive in pairs:
        for word in re.findall(r"[一-鿿]|[a-zA-Z0-9_]+", query + positive):
            vocabulary[word] = vocabulary.get(word, 0) + 1
    return vocabulary

if __name__ == "__main__":
    print(run([("金融问答", "金融检索"), ("注册资本", "资本金额")]))
