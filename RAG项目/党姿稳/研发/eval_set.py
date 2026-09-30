"""
eval_set.py — 评测指标定义与内置评测集

15 道题，三个领域各 5 道，每题带标准答案（ground_truth），
供 evaluate.py 计算 RAGAS 四项指标。
"""

from __future__ import annotations

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

EVAL_SET: list[dict] = [
    {"domain": "legal", "question": "试用期最长可以约定多久？",
     "ground_truth": "劳动合同期限三个月以上不满一年的试用期不得超过一个月；一年以上不满三年的不得超过二个月；三年以上固定期限和无固定期限的不得超过六个月。"},
    {"domain": "legal", "question": "经济补偿金怎么计算？",
     "ground_truth": "按劳动者在本单位工作的年限，每满一年支付一个月工资；六个月以上不满一年按一年计算，不满六个月支付半个月工资。"},
    {"domain": "legal", "question": "周末加班工资怎么算？",
     "ground_truth": "休息日安排工作又不能安排补休的，支付不低于工资的百分之二百的工资报酬。"},
    {"domain": "legal", "question": "未签订书面劳动合同有什么后果？",
     "ground_truth": "用人单位自用工之日起超过一个月不满一年未订立书面劳动合同的，应当向劳动者每月支付二倍的工资。"},
    {"domain": "legal", "question": "在上下班途中发生交通事故算工伤吗？",
     "ground_truth": "在上下班途中受到非本人主要责任的交通事故伤害的，应当认定为工伤。"},

    {"domain": "medical", "question": "感冒需要用抗生素吗？",
     "ground_truth": "普通感冒多由病毒引起，抗生素对病毒无效，不应常规使用；病程通常五到七天可自愈。"},
    {"domain": "medical", "question": "成年人每天建议睡多久？",
     "ground_truth": "成年人推荐睡眠时长为七到九小时，存在个体差异。"},
    {"domain": "medical", "question": "空腹血糖多少算糖尿病？",
     "ground_truth": "空腹血糖大于或等于7.0 mmol/L 达到糖尿病诊断标准，正常范围为3.9至6.1 mmol/L。"},
    {"domain": "medical", "question": "孩子发烧到多少度需要立即就医？",
     "ground_truth": "体温超过41摄氏度属于超高热需立即就医；此外出现抽搐、精神萎靡、尿量明显减少时也应立即就医。"},
    {"domain": "medical", "question": "高血压的诊断标准是什么？",
     "ground_truth": "在未使用降压药物的情况下，非同日三次测量收缩压大于等于140mmHg或舒张压大于等于90mmHg。"},

    {"domain": "english", "question": "现在完成时和一般过去时有什么区别？",
     "ground_truth": "一般过去时陈述过去特定时间发生的事实，常与具体时间状语连用；现在完成时强调对现在的影响或持续状态，常与already、yet、since、for连用。"},
    {"domain": "english", "question": "被动语态怎么构成？",
     "ground_truth": "被动语态基本结构为 be 加过去分词，执行者需要强调时用 by 引出。"},
    {"domain": "english", "question": "although 和 but 可以同时使用吗？",
     "ground_truth": "不可以。although 与 but 不同现，中文的虽然但是结构在英语中只保留其一。"},
    {"domain": "english", "question": "look forward to 后面接什么形式？",
     "ground_truth": "look forward to 中的 to 是介词，后面接名词或动名词，例如 look forward to your reply 或 look forward to hearing from you。"},
    {"domain": "english", "question": "与现在事实相反的虚拟语气怎么用？",
     "ground_truth": "与现在事实相反时，if 从句用过去式（be 动词用 were），主句用 would/could/might 加动词原形。"},
]
