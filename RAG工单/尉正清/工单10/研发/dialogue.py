# 工单编号：人工智能NLP-RAG-金融问答系统部署
"""多轮对话：指代消解 + 省略补全

工单5 的核心。连续提问时用户大量使用指代和省略：

    Q1 报告期内，武汉兴图新科…来自军用领域的收入分别是多少？    ← 明确主体
    Q2 他参与的哪个工程荣获了国家科技进步一等奖？              ← "他" 指上一轮的公司
    Q3 这个公司的法定代表人是谁？                              ← "这个公司" 同上
    Q4 那武汉力源信息技术股份有限公司呢？                       ← 省略了"问什么"

单轮检索系统直接拿 Q2~Q4 这种问句去检索必然失败 —— 问题里既没有主体，
也看不出要问什么。必须先用对话历史把它改写成**自足的独立问题**，再走正常检索。

改写采用规则实现而不是再调一次大模型：工单要求响应不超过 3 秒，而多一次
大模型调用就要多 1.5~2 秒，加上生成会直接超标。
"""
import re

# 公司名样式。只认「已加载文档对应的公司」，避免把「国信证券股份有限公司」
# 这类中介机构当成提问主体。
_COMPANY = re.compile(r"[一-龥]{2,12}(?:股份)?有限公司")

# 指代上一轮主体的说法
_PRONOUNS = ("他们", "他", "她", "它", "该公司", "这个公司", "这家公司", "上述公司")

# 省略式追问：「那 X 呢？」「X 如何？」
_ELLIPSIS = re.compile(r"^那?(?P<ent>.+?)(?:呢|如何|怎么样)[？?]?\s*$")

# 判断「呢」前面那截是不是一个真正的问题（而不是光一个主体）。
# 「…的注册资本是多少呢？」是真问题，「武汉力源信息技术股份有限公司呢？」才是省略追问。
_QUESTION_WORDS = ("谁", "什么", "多少", "哪", "怎么", "为什么", "何时", "是否", "几")


class Conversation:
    """一次会话的上下文。多轮问答共用同一个实例。"""

    def __init__(self, companies=None):
        self.companies = list(companies or [])   # 已加载文档的公司名
        self.turns = []                          # [{"question","resolved","answer","company"}]
        self.current_company = ""                # 当前话题主体

    # ---------- 内部工具 ----------
    def _companies_in(self, text):
        """找出文本里出现的、属于已加载文档的公司名。"""
        found = _COMPANY.findall(text or "")
        return [c for c in found if c in self.companies] if self.companies else found

    def _remember(self, resolved):
        """从改写后的问题里更新当前话题主体。"""
        hits = self._companies_in(resolved)
        if hits:
            self.current_company = hits[0]
        return self.current_company

    # ---------- 核心：改写 ----------
    def resolve(self, question):
        """把带指代/省略的问题改写成自足问题。

        返回 {"resolved": 改写结果, "strategy": 用的哪种改写, "company": 主体}
        """
        q = (question or "").strip()
        if not q:
            return {"resolved": "", "strategy": "空问题", "company": self.current_company}

        # ① 省略式追问优先于「已点明主体」判断。
        #   「武汉力源信息技术股份有限公司呢？」既出现了公司名、又是省略追问，
        #   若先按「已点明主体」原样返回，就把「问什么」丢了 —— 实测会退化成
        #   拿「…呢？」这种没有信息量的句子去检索。
        #   护栏：「呢」前面若含疑问词，说明本身就是完整问题，不做省略补全。
        match = _ELLIPSIS.match(q)
        if match and self.turns and self.current_company:
            entity = match.group("ent").strip("，, 的")
            if not any(word in entity for word in _QUESTION_WORDS):
                for word in _PRONOUNS:                 # 「那他呢？」
                    entity = entity.replace(word, self.current_company)
                prev = self.turns[-1]["resolved"]
                if self.current_company in prev:
                    resolved = prev.replace(self.current_company, entity)
                    self.current_company = entity
                    return {"resolved": resolved,
                            "strategy": f"省略补全（沿用上一轮问法，主体换成「{entity}」）",
                            "company": entity}

        # ② 问题里已经点明了公司 —— 直接用它，同时更新话题主体
        explicit = self._companies_in(q)
        if explicit:
            company = self._remember(q)
            return {"resolved": q, "strategy": "原样（问题已点明主体）", "company": company}

        # ③ 代词指代：「他 / 这个公司」—— 换成当前话题主体
        if self.current_company:
            for word in _PRONOUNS:
                if word in q:
                    resolved = q.replace(word, self.current_company, 1)
                    return {"resolved": resolved,
                            "strategy": f"指代消解（「{word}」→「{self.current_company}」）",
                            "company": self.current_company}
            # ④ 既没公司名也没代词 —— 视为延续上一轮的话题
            return {"resolved": f"{self.current_company}{q}",
                    "strategy": f"话题延续（补上主体「{self.current_company}」）",
                    "company": self.current_company}

        return {"resolved": q, "strategy": "原样（无上下文可依）", "company": ""}

    # ---------- 记录一轮 ----------
    def add_turn(self, question, resolved, answer):
        self.turns.append({"question": question, "resolved": resolved,
                           "answer": answer, "company": self.current_company})

    def reset(self):
        self.turns.clear()
        self.current_company = ""
