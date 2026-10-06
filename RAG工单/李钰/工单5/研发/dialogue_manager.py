# -*- coding: utf-8 -*-
"""
会话管理器 - 多轮对话的状态维护与指代消解
工单编号: 人工智能 NLP-RAG-Query 理解优化任务

核心功能:
    1. 会话状态: 当前公司、话题、历史轮次
    2. 指代消解: "他"/"这个公司" → 完整公司名
    3. 省略补全: "那XX呢?" → 继承上一轮意图
    4. 公司切换检测
"""
import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 公司别名映射 (统一实体识别)
COMPANY_ALIASES = {
    "武汉兴图新科电子股份有限公司": ["武汉兴图新科", "兴图新科", "兴图", "发行人"],
    "武汉力源信息技术股份有限公司": ["武汉力源", "力源信息", "力源"],
}
ALL_COMPANIES = list(COMPANY_ALIASES.keys())


class SessionContext:
    """会话上下文"""
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.current_company: Optional[str] = None
        self.company_history: List[str] = []
        self.last_question: str = ""
        self.last_enriched: str = ""
        self.last_answer: str = ""
        self.last_topic: str = ""
        self.turn_count: int = 0
        self.history: List[Dict] = []  # [{role, content, enriched}]

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "current_company": self.current_company,
            "company_history": self.company_history,
            "last_question": self.last_question,
            "last_topic": self.last_topic,
            "turn_count": self.turn_count,
        }

    def record(self, question: str, enriched: str, answer: str, topic: str):
        self.turn_count += 1
        self.last_question = question
        self.last_enriched = enriched
        self.last_answer = answer
        self.last_topic = topic
        if self.current_company:
            if self.current_company not in self.company_history:
                self.company_history.append(self.current_company)
        self.history.append({
            "role": "user", "content": question, "enriched": enriched
        })
        self.history.append({"role": "assistant", "content": answer})


class DialogueManager:
    """多轮对话管理器"""

    def __init__(self):
        self._sessions: Dict[str, SessionContext] = {}

    def get_session(self, session_id: str = "default") -> SessionContext:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionContext(session_id)
        return self._sessions[session_id]

    def identify_company(self, query: str) -> Optional[str]:
        """从问题中识别公司名 (含别名)"""
        for full, aliases in COMPANY_ALIASES.items():
            if full in query:
                return full
            for alias in aliases:
                if alias in query:
                    return full
        return None

    def resolve_reference(self, query: str, ctx: SessionContext) -> str:
        """
        指代消解: 将 "他"/"这个公司" 等指代补全为完整问题

        Returns:
            消解后的完整问题
        """
        resolved = query

        # 1. 人称代词 → 上一轮公司
        pronouns = ["他", "她", "它", "他们", "她们", "它们"]
        if ctx.current_company:
            for p in pronouns:
                # 只替换开头的 "他" 或句首
                if resolved.startswith(p) or re.match(rf"^{p}[，,。.!！?？]", resolved):
                    resolved = ctx.current_company + " " + resolved[len(p):]
                    logger.info(f"[指代消解] '{query}' → '{resolved}' ({p}→{ctx.current_company})")
                    break

        # 2. "这个公司"/"该公司"/"这家公司" → 当前公司
        if ctx.current_company:
            ref_patterns = ["这个公司", "该公司", "这家公司", "这个企业", "该企业"]
            for pat in ref_patterns:
                if pat in resolved:
                    resolved = resolved.replace(pat, ctx.current_company)
                    logger.info(f"[指代消解] '{query}' → '{resolved}' ({pat}→{ctx.current_company})")
                    break

        return resolved

    def detect_company_switch(self, query: str, ctx: SessionContext) -> Optional[str]:
        """
        检测公司切换模式: "那XX呢?" / "XX公司呢?"

        Returns:
            切换后的公司名 (None 表示未切换)
        """
        # 模式1: "那XX呢?" / "那XX公司呢?"
        m = re.match(r"那(.+?)(?:公司|有限公司)?[?？]?$", query.strip())
        if m and ctx.current_company:
            candidate = m.group(1).strip()
            # 补全为完整公司名
            new_company = self._match_company(candidate)
            if new_company and new_company != ctx.current_company:
                logger.info(f"[公司切换] {ctx.current_company} → {new_company} (来自: '{query}')")
                return new_company

        # 模式2: "那XX的...呢?" / "那XX公司的...?"
        m2 = re.match(r"那(.+?)(?:公司|有限公司)?的(.+?)[?？]?$", query.strip())
        if m2 and ctx.current_company:
            candidate = m2.group(1).strip()
            new_company = self._match_company(candidate)
            if new_company and new_company != ctx.current_company:
                logger.info(f"[公司切换] {ctx.current_company} → {new_company} (来自: '{query}')")
                return new_company

        return None

    def _match_company(self, text: str) -> Optional[str]:
        """模糊匹配公司名"""
        text = text.replace(" ", "")
        for full, aliases in COMPANY_ALIASES.items():
            if text in full or full in text:
                return full
            for a in aliases:
                if text in a or a in text or a in text:
                    return full
        return None

    def infer_topic_from_last(self, ctx: SessionContext) -> str:
        """从上一轮问题推断话题 (用于省略补全)"""
        last = ctx.last_enriched or ctx.last_question
        if not last:
            return ""
        # 提取核心属性关键词
        topic_keywords = {
            "收入": ["收入", "军用", "主营"],
            "占比": ["比重", "占比", "比例"],
            "标准": ["技术标准", "参与制定"],
            "注册资本": ["注册资本", "股本"],
            "法定代表人": ["法定代表人", "法人代表"],
            "募集资金": ["募集资金", "投资项目"],
            "关联方": ["关联方", "持股"],
            "工程奖项": ["工程", "科技进步", "一等奖"],
            "组织结构": ["组织结构", "销售部", "销售处"],
        }
        for topic, kws in topic_keywords.items():
            if any(k in last for k in kws):
                return topic
        # 兜底: 提取"的XX"模式
        m = re.search(r"的(.+?)[?？]", last)
        if m:
            return m.group(1)
        return ""

    def fill_ellipsis(self, query: str, ctx: SessionContext,
                      switched_company: Optional[str] = None) -> str:
        """
        省略补全: 多轮对话中用户省略上一轮已提到的实体

        场景:
          1. "那武汉力源呢?" → 继承上一轮话题
          2. "法定代表人是谁?" → 继承上一轮公司
          3. "他参与的..." → 指代消解 (已在 resolve_reference 处理)
        """
        filled = query
        target_company = switched_company or ctx.current_company

        # 场景1: "那XX呢?" → 需要补全话题
        m = re.match(r"那(.+?)(?:公司|有限公司)?[?？]?$", query.strip())
        if m and ctx.last_topic and target_company:
            topic = ctx.last_topic
            # 将话题转为问题片段
            topic_phrases = {
                "收入": "来自军用领域的收入分别是多少",
                "占比": "来自军用领域的收入占主营业务收入的比重分别是多少",
                "标准": "参与制定了哪个技术标准",
                "注册资本": "注册资本是多少",
                "法定代表人": "法定代表人是谁",
                "募集资金": "本次募集资金拟投资哪些项目",
                "关联方": "与公司存在控制关系的关联方是谁,持股比例和本公司关系是什么",
                "工程奖项": "参与的哪个工程荣获了国家科技进步一等奖",
                "组织结构": "组织结构图中销售部有几个部门构成",
            }
            phrase = topic_phrases.get(ctx.last_topic, ctx.last_topic)
            filled = f"{target_company}的{phrase}?"
            logger.info(f"[省略补全] '{query}' → '{filled}' (继承话题: {ctx.last_topic})")
            return filled

        # 场景2: 无公司名但有属性问题 → 继承当前公司
        if target_company and not self.identify_company(query):
            # 检查是否为属性问题 (含"是多少"/"是谁"/"哪些"等)
            if re.search(r"是多少|是谁|哪些|有哪些|多少|分别是|参与|来自|占|构成|关系", query):
                filled = f"{target_company}{query}"
                logger.info(f"[省略补全] '{query}' → '{filled}' (继承公司: {target_company})")

        return filled

    def understand_query(self, query: str, session_id: str = "default") -> Dict:
        """
        Query 理解主入口 (多轮版)

        Returns:
            {
                "original": str,
                "enriched": str,          # 完整补全后的查询
                "company": str,           # 当前聚焦公司
                "company_switched": bool, # 是否发生公司切换
                "is_ellipsis": bool,      # 是否省略补全
                "turn_count": int,
            }
        """
        ctx = self.get_session(session_id)
        original = query

        # 1. 公司切换检测 (优先级最高)
        switched_company = self.detect_company_switch(query, ctx)

        # 2. 首次对话: 识别公司
        detected_company = self.identify_company(query)
        if ctx.current_company is None:
            ctx.current_company = detected_company

        # 3. 指代消解
        query_step1 = self.resolve_reference(query, ctx)

        # 4. 省略补全
        if switched_company:
            # 公司切换: 继承上一轮话题
            ctx.current_company = switched_company
            enriched = self.fill_ellipsis(query_step1, ctx, switched_company)
        elif not self.identify_company(query_step1) and ctx.current_company:
            # 无公司名 + 有上下文 → 省略补全
            enriched = self.fill_ellipsis(query_step1, ctx)
        else:
            enriched = query_step1

        # 5. 更新当前公司 (可能补全出了新公司)
        final_company = self.identify_company(enriched) or ctx.current_company
        if final_company != ctx.current_company:
            ctx.current_company = final_company

        # 6. 更新话题
        topic = self.infer_topic_from_last(ctx)
        # 简单判断当前查询的话题
        for k in ["收入", "注册资本", "法定代表人", "募集资金", "关联方",
                  "标准", "工程", "组织结构", "占比"]:
            if k in enriched:
                topic = k
                break
        ctx.last_topic = topic

        return {
            "original": original,
            "enriched": enriched,
            "company": ctx.current_company,
            "company_switched": switched_company is not None,
            "turn_count": ctx.turn_count + 1,
            "topic": topic,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dm = DialogueManager()
    sid = "test"

    # 模拟工单5的多轮对话
    dialogues = [
        "报告期内,武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少?",
        "他参与的哪个工程荣获了国家科技进步一等奖?",
        "这个公司的法定代表人是谁?",
        "那武汉力源信息技术股份有限公司呢?",
        "武汉力源信息技术股份有限公司组织结构图中,哪个销售部的销售处最多?有哪些销售处?",
    ]

    for q in dialogues:
        r = dm.understand_query(q, sid)
        print(f"\n[轮次 {r['turn_count']}] {q[:50]}...")
        print(f"  补全: {r['enriched'][:60]}...")
        print(f"  公司: {r['company']} | 话题: {r['topic']} | 切换: {r['company_switched']}")
