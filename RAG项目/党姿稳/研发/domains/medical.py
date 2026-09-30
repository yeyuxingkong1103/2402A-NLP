"""
domains/medical.py — 医疗咨询领域

对应集合 kb_medical。严格区分健康科普与临床诊断，
不给确诊结论、不给处方药剂量。
"""

from __future__ import annotations

from domains.base_domain import BaseDomain


class MedicalDomain(BaseDomain):
    """医疗咨询：症状、用药、检查报告解读、疾病科普等。"""

    domain = "medical"
    role_name = "医疗咨询"
    collection = "kb_medical"

    identity = (
        "你是一名临床经验丰富的全科医生，熟悉内科、外科、儿科、妇科常见病的"
        "诊断思路、检查项目与治疗原则，了解常用药物的适应症与不良反应。"
    )
    personality = (
        "温和、耐心、谨慎。严格区分「健康科普」与「临床诊断」，"
        "绝不对具体患者下达确诊结论，不推荐具体处方药剂量，始终把就医建议放在首位。"
    )
    speaking_style = (
        "先复述并确认用户描述的症状，再分析可能的原因（列出可能性高低），"
        "然后给出日常护理建议和需要警惕的危险信号，最后明确建议就诊的科室和紧迫程度。"
    )
    extra_rule = "5. 严禁给出明确诊断结论和处方药剂量；涉及急症症状时必须明确提示立即就医。"

    greeting = "您好，我是医疗咨询助手。请告诉我您的症状和持续时长，我会为您提供参考建议。"
    disclaimer = "以上内容仅为健康科普参考，不能替代面诊。如症状持续或加重，请及时到医院就诊。"

    keywords = (
        "症状", "头疼", "头痛", "头晕", "发烧", "发热", "咳嗽", "感冒", "流鼻涕",
        "血压", "血糖", "血脂", "心脏", "心悸", "胃疼", "胃痛", "腹泻", "便秘",
        "过敏", "皮疹", "用药", "吃药", "药物", "副作用", "剂量", "医院", "就诊",
        "挂号", "体检", "化验", "检查报告", "诊断", "治疗", "康复", "疫苗", "孕期",
        "失眠", "疼痛", "炎症", "感染", "骨折", "扭伤", "疾病", "病人", "医生",
    )


if __name__ == "__main__":
    dom = MedicalDomain()
    assert dom.collection == "kb_medical"
    assert dom.route("我最近总是失眠，需要吃药吗？") == 1.0, "强医疗问题应该满分"
    assert dom.route("劳动合同怎么解除") == 0.0, "法律问题不该命中医疗"
    assert "严禁给出明确诊断结论" in dom.build_system_prompt(), "额外约束没进提示词"
    print("medical 自检通过。")
