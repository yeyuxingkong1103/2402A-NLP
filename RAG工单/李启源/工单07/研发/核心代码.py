"""工单07：功能测试及RAG评估的可运行核心代码。"""

def run(cases, retrieve):
    report = []
    for question, expected in cases:
        answer = retrieve(question)
        matched = sum(term in answer for term in expected)
        report.append({"question": question, "recall": matched / max(1, len(expected)), "answer": answer})
    return report

if __name__ == "__main__":
    cases = [("注册资本是多少", ("注册资本",)), ("主营业务是什么", ("电子信息",))]
    result = run(cases, lambda _: "公司注册资本为1000万元，主营电子信息业务")
    print(result)
