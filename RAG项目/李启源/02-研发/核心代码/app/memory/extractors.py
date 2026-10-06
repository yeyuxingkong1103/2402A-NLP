"""
信息提取器 - 从对话中提取实体、意图、偏好
"""

import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class InformationExtractor:
    """
    信息提取器

    职责：
    1. 提取实体（订单号、产品、时间、金额等）
    2. 分类意图
    3. 检测用户偏好
    4. 识别重要信息
    """

    # 意图分类（简化版，实际应该用NLP模型）
    INTENT_KEYWORDS = {
        "order_inquiry": ["订单", "查询订单", "订单状态", "订单号"],
        "refund_inquiry": ["退货", "退款", "退换", "退回", "申请退"],
        "shipping_inquiry": ["物流", "快递", "发货", "配送", "送达", "到货"],
        "policy_question": ["政策", "规定", "条款", "能不能", "可以吗"],
        "product_question": ["产品", "商品", "功能", "参数", "规格", "型号"],
        "complaint": ["投诉", "不满意", "太差", "骗人", "垃圾"],
        "price_inquiry": ["价格", "多少钱", "优惠", "折扣", "促销"],
        "technical_issue": ["坏了", "故障", "问题", "不能用", "无法", "报错"],
        "greeting": ["你好", "您好", "hi", "hello"],
        "thanks": ["谢谢", "感谢", "多谢"],
    }

    # 实体识别正则（简化版）
    # 注意：不能用 \w / [\w\-]。Python 3 的 \w 默认是 Unicode 的，汉字也算 \w，
    # 所以 r"(?:订单号)[\s:：-]*([\w\-]{10,30})" 在「订单号是ORD-...」上会把「是」
    # 一起吞进去，抽出脏订单号。下面一律写成显式的 ASCII 字符类。
    ENTITY_PATTERNS = {
        "order_id": (
            r"(?:订单号|订单编号|订单|order\s*(?:id|no)?|ORD)"
            r"[\s:：是为#\-]*"
            r"((?:ORD[-_]?)?[A-Za-z0-9][A-Za-z0-9\-_]{7,29})"
        ),
        "phone": r"1[3-9]\d{9}",
        "email": r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
        "money": r"[¥$￥]\s*(\d+(?:\.\d{1,2})?)",
        "date": r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)",
    }

    def __init__(self):
        pass

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        提取文本中的实体

        Args:
            text: 输入文本

        Returns:
            实体字典 {"order_id": [...], "phone": [...], ...}
        """
        entities = {}

        try:
            for entity_type, pattern in self.ENTITY_PATTERNS.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # 如果匹配结果是元组，取第一个元素
                    entities[entity_type] = [
                        m if isinstance(m, str) else m[0]
                        for m in matches
                    ]

            # 产品名称提取（简化版：提取名词）
            # 这里可以接入NER模型来做更准确的提取
            product_keywords = [
                "耳机", "手机", "电脑", "手表", "充电器", "数据线",
                "音箱", "键盘", "鼠标", "显示器", "相机"
            ]
            products = []
            for keyword in product_keywords:
                if keyword in text:
                    products.append(keyword)

            if products:
                entities["product"] = products

        except Exception as e:
            logger.warning(f"实体提取失败: {e}")

        return entities

    def classify_intent(self, text: str) -> str:
        """
        分类用户意图

        Args:
            text: 用户查询文本

        Returns:
            意图类型（如果无法识别返回 "general"）
        """
        try:
            text_lower = text.lower()

            # 简单的关键词匹配
            for intent, keywords in self.INTENT_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        return intent

            # 默认意图
            return "general"

        except Exception as e:
            logger.warning(f"意图分类失败: {e}")
            return "general"

    def detect_sentiment(self, text: str) -> str:
        """
        检测情感倾向

        Args:
            text: 输入文本

        Returns:
            情感类型（positive, neutral, negative）
        """
        # 简化版：基于关键词
        positive_words = ["满意", "好", "棒", "赞", "不错", "喜欢", "感谢"]
        negative_words = ["不满意", "差", "烂", "垃圾", "骗", "投诉", "退", "坏"]

        text_lower = text.lower()

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if negative_count > positive_count:
            return "negative"
        elif positive_count > negative_count:
            return "positive"
        else:
            return "neutral"

    def is_important_message(
        self,
        message: Dict,
        entities: Dict[str, List[str]],
        intent: str
    ) -> bool:
        """
        判断消息是否包含重要信息

        Args:
            message: 消息对象
            entities: 提取的实体
            intent: 意图类型

        Returns:
            是否重要
        """
        # 包含订单号
        if entities.get("order_id"):
            return True

        # 包含金额
        if entities.get("money"):
            return True

        # 投诉意图
        if intent == "complaint":
            return True

        # 负面情感
        sentiment = self.detect_sentiment(message.get("content", ""))
        if sentiment == "negative":
            return True

        return False

    def extract_user_preference(
        self,
        messages: List[Dict]
    ) -> List[Dict]:
        """
        从对话历史中提取用户偏好

        Args:
            messages: 消息列表

        Returns:
            偏好列表 [{"type": "product", "text": "..."}, ...]
        """
        preferences = []

        try:
            # 统计用户提到的产品
            product_mentions = {}
            for msg in messages:
                if msg.get("role") != "user":
                    continue

                entities = self.extract_entities(msg.get("content", ""))
                products = entities.get("product", [])

                for product in products:
                    product_mentions[product] = product_mentions.get(product, 0) + 1

            # 提取高频产品作为偏好
            for product, count in product_mentions.items():
                if count >= 2:  # 至少提到2次
                    preferences.append({
                        "type": "product",
                        "text": f"用户对 {product} 感兴趣",
                        "confidence": min(0.5 + count * 0.1, 0.95)
                    })

            # TODO: 可以添加更多偏好提取逻辑
            # - 语言风格偏好（简洁 vs 详细）
            # - 服务偏好（快速解决 vs 详细解释）
            # - 时间偏好（咨询时段）

        except Exception as e:
            logger.warning(f"提取用户偏好失败: {e}")

        return preferences

    def extract_key_intents(
        self,
        messages: List[Dict]
    ) -> List[str]:
        """
        提取对话的主要意图

        Args:
            messages: 消息列表

        Returns:
            意图列表（去重）
        """
        intents = set()

        try:
            for msg in messages:
                if msg.get("role") == "user":
                    intent = self.classify_intent(msg.get("content", ""))
                    if intent != "general":
                        intents.add(intent)

        except Exception as e:
            logger.warning(f"提取关键意图失败: {e}")

        return list(intents)
