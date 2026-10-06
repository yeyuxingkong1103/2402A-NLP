# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：calibrate —— 拒答阈值标定
# 说明：统计“库内问题”与“库外问题”的最高余弦置信度分布，选一个能分开两者的阈值。
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import engine

IN_DOMAIN = [
    "报告期内来自军用领域的收入分别是多少？",
    "公司参与制定了哪个技术标准？",
    "军用领域收入占主营业务收入的比重分别是多少？",
    "电子信息行业的上游涉及哪些企业？",
    "公司在哪个领域已经成为重要供应商？",
    "电子信息行业的下游主要包括哪些行业？",
    "哪个工程荣获了国家科技进步一等奖？",
    "公司的注册资本是多少？",
    "公司的法定代表人是谁？",
    "募集资金中多少用于补充流动资金？",
]

OUT_DOMAIN = [
    "今天北京天气怎么样？",
    "帮我写一首关于春天的七言绝句。",
    "Python 里怎么读取 CSV 文件？",
    "推荐几家好吃的火锅店。",
    "牛顿第二定律的公式是什么？",
    "2024 年世界杯冠军是谁？",
    "怎么给手机换电池？",
    "红烧肉的做法是什么？",
]

def conf(q):
    _h, c = engine.retrieve(q)
    return c

ins = [(q, conf(q)) for q in IN_DOMAIN]
outs = [(q, conf(q)) for q in OUT_DOMAIN]

print("=== 库内问题（应 >= 阈值）===")
for q, c in sorted(ins, key=lambda x: x[1]):
    print("  %.3f  %s" % (c, q))
print("=== 库外问题（应 < 阈值）===")
for q, c in sorted(outs, key=lambda x: -x[1]):
    print("  %.3f  %s" % (c, q))

in_min = min(c for _q, c in ins)
out_max = max(c for _q, c in outs)
print("\n库内最低 = %.3f | 库外最高 = %.3f" % (in_min, out_max))
if in_min > out_max:
    print("建议阈值取中点 = %.3f （此区间可分）" % ((in_min + out_max) / 2))
else:
    print("两组有重叠，建议阈值 = %.3f（偏向保证库内不误拒）" % (out_max,))
# 重叠统计
ov = [q for q, c in outs if c >= in_min]
print("库外中高于库内最低的题数:", len(ov))
