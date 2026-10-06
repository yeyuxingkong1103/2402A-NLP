# -*- coding: utf-8 -*-
"""
测试套件 - 10 个 CCF 风格测试问题 + 参考答案 + 相关块标注
工单编号: 人工智能 NLP-RAG-功能测试及评估
"""
import json
import os
from typing import List, Dict

# 10 个测试问题 (覆盖 CCF 竞赛风格)
TEST_SUITE: List[Dict] = [
    {
        "id": 1,
        "question": "武汉兴图新科电子股份有限公司的注册资本是多少?",
        "type": "fact",
        "difficulty": "easy",
        "reference": "武汉兴图新科电子股份有限公司的注册资本为7360万元。",
        "ref_keywords": ["7360", "注册资本", "万元"],
        "relevant_chunk_keywords": ["注册资本", "7360"],
    },
    {
        "id": 2,
        "question": "报告期内,武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少?",
        "type": "financial",
        "difficulty": "easy",
        "reference": "报告期内军用领域收入分别为:6464.51万元、14414.16万元、18780.67万元、4627.14万元。",
        "ref_keywords": ["6464.51", "14414.16", "18780.67", "4627.14", "军用", "收入"],
        "relevant_chunk_keywords": ["军用", "收入", "万元"],
    },
    {
        "id": 3,
        "question": "武汉兴图新科电子股份有限公司的法定代表人是谁?",
        "type": "fact",
        "difficulty": "easy",
        "reference": "武汉兴图新科电子股份有限公司的法定代表人为XXX。",
        "ref_keywords": ["法定代表人"],
        "relevant_chunk_keywords": ["法定代表人"],
    },
    {
        "id": 4,
        "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目?",
        "type": "investment",
        "difficulty": "medium",
        "reference": "募集资金拟投资:高精度ADC/DAC芯片研发、高速接口芯片研发、射频前端芯片研发、补充流动资金。",
        "ref_keywords": ["ADC", "接口", "射频", "流动资金", "募集"],
        "relevant_chunk_keywords": ["募集", "投资", "项目", "芯片"],
    },
    {
        "id": 5,
        "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁,持股比例和本公司关系是什么?",
        "type": "relation",
        "difficulty": "medium",
        "reference": "控股股东武汉力源科技有限公司持股35%,实际控制人赵彤持股10%(董事长)、赵刚持股5%(董事兼总经理)。",
        "ref_keywords": ["武汉力源科技", "35", "控股股东", "赵彤", "赵刚", "持股"],
        "relevant_chunk_keywords": ["关联方", "持股", "控制", "控股股东"],
    },
    {
        "id": 6,
        "question": "武汉力源信息技术股份有限公司组织结构图中,销售部有几个部门构成,其中大客户销售部有几个销售处构成?",
        "type": "image",
        "difficulty": "medium",
        "reference": "销售部有4个部门(大客户销售部、渠道销售部、产品销售部、电商销售部);大客户销售部有4个销售处(华东、华南、华北、西南)。",
        "ref_keywords": ["销售部", "大客户销售部", "销售处"],
        "relevant_chunk_keywords": ["销售部", "销售处", "组织"],
    },
    {
        "id": 7,
        "question": "2008年中国IC市场应用结构与增长图中,增长率最快的是哪个行业?负增长的是哪个行业?",
        "type": "image",
        "difficulty": "medium",
        "reference": "增长率最快的行业是汽车电子和嵌入式系统;负增长的行业是消费电子和通讯设备。",
        "ref_keywords": ["增长", "最快", "负增长", "汽车电子", "消费电子"],
        "relevant_chunk_keywords": ["增长", "IC", "增长率", "行业"],
    },
    {
        "id": 8,
        "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖?",
        "type": "fact",
        "difficulty": "easy",
        "reference": "武汉兴图新科电子股份有限公司参与的某工程荣获了国家科技进步一等奖。",
        "ref_keywords": ["工程", "科技进步", "一等奖"],
        "relevant_chunk_keywords": ["科技进步", "一等奖"],
    },
    {
        "id": 9,
        "question": "报告期内,武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少?",
        "type": "financial",
        "difficulty": "medium",
        "reference": "军用领域收入占主营业务收入的比重分别为:82.10%、97.31%、94.84%、94.34%。",
        "ref_keywords": ["82.10", "97.31", "94.84", "94.34", "比重"],
        "relevant_chunk_keywords": ["比重", "占比", "军用"],
    },
    {
        "id": 10,
        "question": "武汉力源信息技术股份有限公司本次发行募集资金的多少用于补充流动资金?",
        "type": "investment",
        "difficulty": "medium",
        "reference": "本次募集资金的17.5%用于补充流动资金,金额为7000万元。",
        "ref_keywords": ["补充流动资金", "7000", "17.5"],
        "relevant_chunk_keywords": ["流动资金", "募集", "补充"],
    },
]


def get_test_suite() -> List[Dict]:
    """获取测试套件"""
    return TEST_SUITE


def get_question_by_id(qid: int) -> Dict:
    for q in TEST_SUITE:
        if q["id"] == qid:
            return q
    return {}


def save_test_suite(path: str):
    """保存测试套件到 JSON"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(TEST_SUITE, f, ensure_ascii=False, indent=2)
    print(f"测试套件已保存: {path}")


if __name__ == "__main__":
    print(f"测试套件: {len(TEST_SUITE)} 个问题")
    for q in TEST_SUITE:
        print(f"  Q{q['id']} [{q['type']}] [{q['difficulty']}] {q['question'][:40]}...")
