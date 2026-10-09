# -*- coding: utf-8 -*-
"""
数据集生成器 - 问答对 + 三元组 + 相似度对
工单编号: 人工智能 NLP-RAG 项目-Embedding 模型微调任务
"""
import json, os, random, logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


# ============ 金融领域预设数据 (开箱即用) ============
PRESET_QA_PAIRS = [
    # === 公司基本信息 ===
    {"query": "武汉兴图新科电子股份有限公司的注册资本是多少?", "positive": "武汉兴图新科电子股份有限公司的注册资本为7360万元。", "type": "fact"},
    {"query": "武汉力源信息技术股份有限公司的注册资本是多少?", "positive": "武汉力源信息技术股份有限公司的注册资本为8000万元。", "type": "fact"},
    {"query": "武汉兴图新科的法定代表人是谁?", "positive": "武汉兴图新科电子股份有限公司的法定代表人为XXX。", "type": "fact"},

    # === 控股股东 / 关联方 ===
    {"query": "武汉力源的控股股东是谁,持股比例是多少?", "positive": "武汉力源科技有限公司是武汉力源信息技术股份有限公司的控股股东,持股35%。", "type": "relation"},
    {"query": "与武汉力源存在控制关系的关联方有哪些?", "positive": "控股股东武汉力源科技有限公司持股35%,实际控制人赵彤、赵刚。", "type": "relation"},

    # === 募集资金 ===
    {"query": "武汉力源本次发行募集资金投资哪些项目?", "positive": "募集资金拟投资高精度ADC/DAC芯片、高速接口芯片、射频前端芯片研发及补充流动资金。", "type": "investment"},
    {"query": "募集资金多少用于补充流动资金?", "positive": "募集资金的17.5%用于补充流动资金,金额为7000万元。", "type": "investment"},
    {"query": "武汉力源本次发行股数是多少?", "positive": "武汉力源本次发行股数为2000万股,占发行后总股本的25.00%。", "type": "investment"},

    # === 军用收入 ===
    {"query": "报告期内武汉兴图新科来自军用领域的收入分别是多少?", "positive": "报告期内军用领域收入分别为6464.51万元、14414.16万元、18780.67万元、4627.14万元。", "type": "financial"},
    {"query": "军用领域收入占主营业务收入的比重是多少?", "positive": "军用领域收入占主营业务收入的比重分别为82.10%、97.31%、94.84%、94.34%。", "type": "financial"},

    # === 标准 / 奖项 ===
    {"query": "武汉兴图新科参与制定了什么技术标准?", "positive": "武汉兴图新科电子股份有限公司参与制定了AVS编解码技术标准。", "type": "standard"},
    {"query": "武汉兴图新科参与的哪个工程荣获了国家科技进步一等奖?", "positive": "武汉兴图新科参与的某工程荣获了国家科技进步一等奖。", "type": "standard"},

    # === 组织结构 ===
    {"query": "武汉力源组织结构图中销售部有几个下属?", "positive": "销售部有4个部门构成:大客户销售部、渠道销售部、产品销售部、电商销售部。", "type": "structure"},
    {"query": "大客户销售部有几个销售处?", "positive": "大客户销售部有4个销售处:华东、华南、华北、西南销售处。", "type": "structure"},

    # === 行业增长 ===
    {"query": "2008年IC市场增长最快的行业是哪些?", "positive": "增长率最快的行业是汽车电子和嵌入式系统。", "type": "industry"},
    {"query": "2008年IC市场负增长的行业是哪些?", "positive": "负增长的行业是消费电子和通讯设备。", "type": "industry"},
]

# 同义词扩展 (用于生成相似度对)
FINANCIAL_SYNONYMS = {
    "注册资本": ["股本", "实缴资本", "登记资本"],
    "控股股东": ["控制方", "实际控制人", "控股方", "最大股东"],
    "持股比例": ["股权比例", "持股比例", "股份比例", "股权占比"],
    "募集资金": ["募资", "融资", "发行募集资金"],
    "法定代表人": ["法人代表", "公司负责人"],
    "军用领域": ["军工领域", "军事领域", "军方"],
    "收入占比": ["收入比重", "收入占主营业务收入比例", "收入比例"],
    "销售部": ["销售部门", "营销部"],
    "销售处": ["办事处", "分公司", "地区销售"],
    "标准": ["技术标准", "行业标准", "国家标准"],
}


class DatasetBuilder:
    """数据集构建器"""

    def __init__(self, base_pairs: List[Dict] = None):
        self.qa_pairs = base_pairs or PRESET_QA_PAIRS

    def generate_positive_pairs(self) -> List[Tuple[str, str]]:
        """生成正例对 (query, positive)"""
        pairs = []
        for item in self.qa_pairs:
            pairs.append((item["query"], item["positive"]))
        logger.info(f"[正例对] 生成 {len(pairs)} 对")
        return pairs

    def generate_triplets(self, neg_per_pos: int = 4) -> List[Tuple[str, str, str]]:
        """
        生成三元组 (anchor, positive, negative)

        Negative 采样:
          1. 同类型不同问题的答案 (hard negative)
          2. 随机其他类型答案 (easy negative)
        """
        triplets = []
        all_positives = [item["positive"] for item in self.qa_pairs]

        for item in self.qa_pairs:
            anchor = item["query"]
            positive = item["positive"]
            item_type = item.get("type", "")

            # Hard negative: 同类型但不同答案
            same_type_positives = [p for p in all_positives
                                    if p != positive and any(
                                        t in p for t in _type_to_keywords(item_type))]
            other_positives = [p for p in all_positives if p != positive]

            # 优先 hard negative
            hard_negs = same_type_positives[:neg_per_pos]
            easy_negs = [p for p in other_positives if p not in hard_negs][:neg_per_pos - len(hard_negs)]

            for neg in hard_negs + easy_negs:
                triplets.append((anchor, positive, neg))

        logger.info(f"[三元组] 生成 {len(triplets)} 组")
        return triplets

    def generate_similarity_pairs(self) -> List[Tuple[str, str, float]]:
        """
        生成带相似度分数的句子对

        score: 1.0 (完全同义词) / 0.8 (高度相关) / 0.3 (弱相关) / 0.0 (不相关)
        """
        pairs = []

        # 1. 同义词替换
        for item in self.qa_pairs:
            query = item["query"]
            positive = item["positive"]

            # 尝试替换同义词生成变体
            for orig, syns in FINANCIAL_SYNONYMS.items():
                if orig in query:
                    for syn in syns:
                        variant = query.replace(orig, syn)
                        pairs.append((variant, query, 0.9))
                        pairs.append((variant, positive, 0.85))

        # 2. 不同类型的 query-positive (弱相关)
        types = list(set(item.get("type", "") for item in self.qa_pairs))
        if len(types) >= 2:
            for i, item1 in enumerate(self.qa_pairs):
                for item2 in self.qa_pairs:
                    if item1.get("type") != item2.get("type"):
                        # 不同类型, 相似度低
                        pairs.append((item1["query"], item2["positive"], 0.1))

        logger.info(f"[相似度对] 生成 {len(pairs)} 对")
        return pairs

    def generate_all(self) -> Dict:
        """生成全部数据集"""
        return {
            "positive_pairs": self.generate_positive_pairs(),
            "triplets": self.generate_triplets(),
            "similarity_pairs": self.generate_similarity_pairs(),
            "metadata": {
                "num_qa_pairs": len(self.qa_pairs),
                "num_triplets": len(self.generate_triplets()),
                "num_similarity": len(self.generate_similarity_pairs()),
                "synonyms": FINANCIAL_SYNONYMS,
            },
        }

    def save(self, path: str, data: Dict = None):
        data = data or self.generate_all()
        # tuple 转 list (JSON 序列化)
        for key in ["positive_pairs", "triplets", "similarity_pairs"]:
            data[key] = [list(item) for item in data[key]]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[数据集] 已保存: {path}")


def _type_to_keywords(type_str: str) -> List[str]:
    mapping = {
        "fact": ["注册资本", "法定代表人", "是谁"],
        "relation": ["控股股东", "持股", "关联方"],
        "investment": ["募集", "发行", "资金"],
        "financial": ["收入", "比重", "占比"],
        "standard": ["标准", "工程", "一等奖"],
        "structure": ["销售部", "部门", "处"],
        "industry": ["行业", "增长", "负增长"],
    }
    return mapping.get(type_str, [])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    builder = DatasetBuilder()
    data = builder.generate_all()
    print(f"\n数据集统计:")
    for k, v in data.items():
        if k != "metadata":
            print(f"  {k}: {len(v)}")
    print(f"  qa_pairs: {data['metadata']['num_qa_pairs']}")
