# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-LightRAG优化
src/lightrag_v12/rule_based_llm.py —— 基于规则的实体/关系抽取

当 DeepSeek API 余额不足时，使用 jieba + 正则从金融文本中抽取实体和关系，
保证知识图谱构建流程可离线完成。
输出格式与 LightRAG 期望的 LLM 输出一致：
  entity<|#|>实体名<|#|>实体类型<|#|>描述
  relation<|#|>主体<|#|>关系<|#|>客体<|#|>描述
"""
import re

# 工单十二：金融领域实体类型
ENTITY_PATTERNS = [
    # 公司/企业
    (r'[\u4e00-\u9fa5]{2,30}(?:股份有限公司|有限责任公司|有限公司|集团|公司|银行|证券|保险|信托)',
     'Organization'),
    # 人名（中文 2-4 字）
    (r'[\u4e00-\u9fa5]{2,4}(?:先生|女士|董事长|总经理|总裁|CEO|法定代表人)', 'Person'),
    # 金额
    (r'\d+(?:\.\d+)?\s*(?:万元|亿元|元|万|亿)', 'MonetaryValue'),
    # 百分比
    (r'\d+(?:\.\d+)?\s*%', 'Percentage'),
    # 日期
    (r'\d{4}\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?', 'Date'),
    # 股票/股份数量
    (r'\d+(?:\.\d+)?\s*(?:万股|股|亿股)', 'ShareAmount'),
    # 地区
    (r'[\u4e00-\u9fa5]{2,8}(?:省|市|区|县|镇|村)', 'Location'),
    # 行业
    (r'(?:集成电路|半导体|电子信息|军工|军用|民用|IC|元器件|分销|研发|生产|销售|信息技术)', 'Industry'),
    # 项目/产品
    (r'[\u4e00-\u9fa5]{2,15}(?:项目|工程|系统|产品|技术|标准|专利)', 'Project'),
]

# 工单十二：关系模式（主语 + 关系词 + 宾语）
RELATION_PATTERNS = [
    (r'([\u4e00-\u9fa5]{2,20}(?:公司|集团|银行))[，,。；：:\s]*(?:注册资本|注册资金)[为是：:\s]*([\d.]+\s*[万亿]?元)',
     '注册资本'),
    (r'([\u4e00-\u9fa5]{2,20}(?:公司|集团))[，,。；：:\s]*(?:法定代表人|董事长|总经理)[为是：:\s]*([\u4e00-\u9fa5]{2,4})',
     '法定代表人'),
    (r'([\u4e00-\u9fa5]{2,20}(?:公司|集团))[，,。；：:\s]*(?:控股股东|实际控制人)[为是：:\s]*([\u4e00-\u9fa5]{2,20}(?:公司|集团|有限公司))',
     '控股股东'),
    (r'([\u4e00-\u9fa5]{2,20}(?:公司|集团))[，,。；：:\s]*(?:持股比例|持股)[为是：:\s]*([\d.]+\s*%)',
     '持股比例'),
    (r'([\u4e00-\u9fa5]{2,20}(?:公司|集团))[，,。；：:\s]*(?:营业收入|营收|收入)[为是：:\s]*([\d.]+\s*[万亿]?元)',
     '营业收入'),
    (r'([\u4e00-\u9fa5]{2,20}(?:公司|集团))[，,。；：:\s]*(?:来自|从事|涉及)[，,。；：:\s]*([\u4e00-\u9fa5]{2,15}(?:领域|行业|业务|产品))',
     '业务领域'),
    (r'([\u4e00-\u9fa5]{2,20}(?:公司|集团))[，,。；：:\s]*(?:参与|制定)[，,。；：:\s]*([\u4e00-\u9fa5]{2,20}(?:标准|规范))',
     '参与制定'),
    (r'([\u4e00-\u9fa5]{2,20}(?:公司|集团))[，,。；：:\s]*(?:发行股数|发行股份)[为是：:\s不低]*([\d.]+\s*[万亿]?股)',
     '发行股数'),
    (r'([\u4e00-\u9fa5]{2,20}(?:公司|集团))[，,。；：:\s]*(?:募集资金)[，,。；：:\s]*(?:用于|投资)[，,。；：:\s]*([\u4e00-\u9fa5]{2,30})',
     '募集资金用途'),
    (r'([\u4e00-\u9fa5]{2,20}(?:公司|集团))[，,。；：:\s]*(?:荣获|获得)[，,。；：:\s]*([\u4e00-\u9fa5]{2,30}(?:奖|一等奖))',
     '获奖'),
    (r'([\u4e00-\u9fa5]{2,20}(?:公司|集团))[，,。；：:\s]*(?:成为|是)[，,。；：:\s]*([\u4e00-\u9fa5]{2,20}(?:供应商|客户))',
     '行业地位'),
    (r'([\u4e00-\u9fa5]{2,20}(?:公司|集团))[，,。；：:\s]*(?:位于|坐落于|地址在)[，,。；：:\s]*([\u4e00-\u9fa5]{2,15}(?:省|市|区))',
     '所在地'),
]


def extract_entities(text):
    """工单十二：从文本中抽取实体，返回 LightRAG 格式字符串"""
    entities = []
    seen = set()
    for pattern, etype in ENTITY_PATTERNS:
        for m in re.finditer(pattern, text):
            name = m.group(0).strip()
            if name not in seen and len(name) >= 2:
                seen.add(name)
                entities.append(f"entity<|#|>{name}<|#|>{etype}<|#|>与{name}相关的金融实体")
    # 限制每 chunk 最多 40 实体（与 LightRAG 配置一致）
    return "\n".join(entities[:40])


def extract_relations(text):
    """工单十二：从文本中抽取关系，返回 LightRAG 格式字符串"""
    relations = []
    seen = set()
    for pattern, rel in RELATION_PATTERNS:
        for m in re.finditer(pattern, text):
            subj = m.group(1).strip()
            obj = m.group(2).strip()
            key = f"{subj}|{rel}|{obj}"
            if key not in seen:
                seen.add(key)
                relations.append(f"relation<|#|>{subj}<|#|>{rel}<|#|>{obj}<|#|>{subj}的{rel}是{obj}")
    return "\n".join(relations[:100])


def rule_based_entity_extraction(text, **kwargs):
    """工单十二：替代 LLM 的实体/关系抽取函数

    返回格式与 LightRAG 的 entity_extraction LLM 调用一致：
    entity<|#|>名<|#|>类型<|#|>描述
    relation<|#|>主<|#|>关系<|#|>客<|#|>描述
    """
    entities = extract_entities(text)
    relations = extract_relations(text)
    result = []
    if entities:
        result.append(entities)
    if relations:
        result.append(relations)
    return "\n".join(result)


def rule_based_keyword_extraction(text, **kwargs):
    """工单十二：替代 LLM 的关键词抽取"""
    # 用 jieba 分词取高频词作为关键词
    try:
        import jieba
        words = jieba.lcut(text)
        stop = set("的了是在和与及或不也都就而但这那我你他她它有被把让对从向到于为由以因等")
        keywords = [w for w in words if len(w) >= 2 and w not in stop]
        # 去重保序
        seen = set()
        result = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                result.append(w)
        return ",".join(result[:20])
    except ImportError:
        return text[:200]


if __name__ == "__main__":
    test = "武汉力源信息技术股份有限公司注册资本3000万元，法定代表人为王某某。公司控股股东为武汉力源创业投资有限公司，持股比例为35%。"
    print("=== 实体抽取 ===")
    print(extract_entities(test))
    print()
    print("=== 关系抽取 ===")
    print(extract_relations(test))
