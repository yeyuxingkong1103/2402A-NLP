# 工单编号：人工智能NLP-RAG-Query 理解优化任务
"""多轮对话标准答案：工单演示的那 5 轮

既校验**改写是否正确**（指代有没有解对、省略有没有补全），
也校验**回答是否正确**（事实点是否命中）。

改写校验用 expect 描述：
    {"company": 改写后应出现的主体, "keyword": 改写后应保留的问法关键词}
    或 "原样" —— 该轮本来就自足，不该被改写。
"""

DIALOGUE = [
    {
        "turn": 1,
        "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "expect": "原样",                       # 已点明主体，不该改写
        "facts": [["6,464.51", "6464.51"], ["14,414.16", "14414.16"],
                  ["18,780.67", "18780.67"], ["4,627.15", "4627.15"]],
        "source_pages": [343],
    },
    {
        "turn": 2,
        "question": "他参与的哪个工程荣获了国家科技进步一等奖？",
        "expect": {"company": "武汉兴图新科", "keyword": "工程"},   # 「他」→ 兴图新科
        "facts": [["情报"], ["指挥"], ["科技进步一等奖"]],
        "source_pages": [181],
    },
    {
        "turn": 3,
        "question": "这个公司的法定代表人是谁？",
        "expect": {"company": "武汉兴图新科", "keyword": "法定代表人"},
        "facts": [["程家明"]],
        "source_pages": [22, 52],
    },
    {
        "turn": 4,
        "question": "那武汉力源信息技术股份有限公司呢？",
        # 省略式追问：要沿用上一轮的问法（法定代表人），只换主体
        "expect": {"company": "武汉力源", "keyword": "法定代表人"},
        "facts": [["赵马克"]],
        "source_pages": [23, 25],
    },
    {
        "turn": 5,
        "question": "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
        "expect": "原样",
        "facts": [["大客户销售部"],
                  ["珠海"], ["深圳"], ["成都"]],
        "source_pages": [39],
    },
]


def check_rewrite(expect, resolved):
    """改写是否正确。返回 (是否通过, 说明)。"""
    if expect == "原样":
        return True, "未改写（本轮自足）"
    ok_company = expect["company"] in (resolved or "")
    ok_keyword = expect["keyword"] in (resolved or "")
    return ok_company and ok_keyword, (
        f"主体{'✓' if ok_company else '✗'} 问法{'✓' if ok_keyword else '✗'}")


def _normalize(text):
    """归一化：去掉 Markdown 强调符、千分位逗号、空白和全角标点。"""
    for ch in ('*', ' ', chr(10), chr(9), ',', '，', '、', '：', ':',
               '（', '）', '(', ')', '《', '》', '|', '　'):
        text = text.replace(ch, '')
    return text
