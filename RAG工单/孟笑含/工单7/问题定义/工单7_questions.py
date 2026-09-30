# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估
模块：10 个问题的 RAG 检索 + 评估
功能：从 sample_questions.pdf 提取的 10 个问题，跑 RAG 检索并评估
"""

import json
import pickle

# 10 个问题（4 个来自 sample_questions.pdf + 6 个补充）
QUESTIONS = [
    {"id": 1, "question": "平安银行在2019年的董事长致辞中，提到其盈利增长的关键因素有哪些？",
     "source": "sample_questions.pdf", "doc": "2019年_平安银行"},
    {"id": 2, "question": "招商银行在其2019年年报中提到的创新商业模式有哪些？",
     "source": "sample_questions.pdf", "doc": "2019年_招商银行"},
    {"id": 3, "question": "平安银行的风险管理体系如何体现其应对宏观经济周期波动的能力？",
     "source": "sample_questions.pdf", "doc": "2019年_平安银行"},
    {"id": 4, "question": "分析这些银行和保险公司在面对经济周期波动时的共同策略与差异化策略。",
     "source": "sample_questions.pdf", "doc": "多个年报"},
    {"id": 5, "question": "平安银行2019年营业收入是多少？",
     "source": "补充", "doc": "2019年_平安银行"},
    {"id": 6, "question": "中国平安2019年归属于母公司股东的净利润是多少？",
     "source": "补充", "doc": "2019年_中国平安"},
    {"id": 7, "question": "招商银行2019年不良贷款率是多少？",
     "source": "补充", "doc": "2019年_招商银行"},
    {"id": 8, "question": "中信证券2020年营业收入是多少？",
     "source": "补充", "doc": "2020年_中信证券"},
    {"id": 9, "question": "中国人寿2020年保费收入是多少？",
     "source": "补充", "doc": "2020年_中国人寿"},
    {"id": 10, "question": "国泰君安2021年营业收入是多少？",
     "source": "补充", "doc": "2021年_国泰君安"},
]

if __name__ == "__main__":
    print("=" * 70)
    print("工单7：10 个问题列表")
    print("=" * 70)
    for q in QUESTIONS:
        print(f"【ID:{q['id']}】[{q['source']}] {q['question']}")
    print()
    with open("工单7问题列表.json", "w", encoding="utf-8") as f:
        json.dump(QUESTIONS, f, ensure_ascii=False, indent=2)
    print("✅ 已保存到 工单7问题列表.json")
