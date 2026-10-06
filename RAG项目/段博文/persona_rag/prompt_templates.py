# -*- coding: utf-8 -*-
"""
角色提示词模板：3 个角色，每个角色一套系统提示词
法律顾问 / 心理专家 / 虚拟朋友

每个角色有对应的 Milvus 集合名和前端建议问题

在系统中的位置：
    上游是对话接口——它先按 role_key 取出 system_prompt 作为 SystemMessage，
    再用 build_context 把 retriever 召回的 Document 拼成用户消息；
    下游是 llm_client.chat/stream_chat，最终把这两段文本喂给大模型。

职责与关键取舍：
    本模块是纯数据 + 拼装逻辑，不做任何 I/O。角色人格（说话风格、边界、免责声明）全部写在
    system_prompt 里，因为 SystemMessage 的服从度最高，比塞在用户消息里更稳。
    temperature 按角色差异设置：法律顾问要稳定可复现（0.3），心理专家要温和有温度（0.7），
    虚拟朋友最需要活泼多变（0.8）；温度越高越自由但越容易跑题、编内容。
"""

# ==================== 预置角色提示词 ====================
# 每个角色的 system_prompt 是一个多行字符串，喂给 LLM 作为 SystemMessage
# collection: 该角色检索的 Milvus 集合名
# suggestions: 前端欢迎页动态显示的建议问题
# 字典的键（lawyer/psychologist/virtual_friend）就是 role_key，由业务库的角色表指定，
# 对话接口用它来取提示词、retriever 用它来路由 Milvus 集合（get_collection_for_role）
# temperature 的含义：0 最保守（几乎每次回答一样）、1 最发散；法律问答要可复现所以给低值，
# 闲聊角色要"每次不一样"所以给高值，但都不建议超过 1，否则容易胡言乱语

ROLE_TEMPLATES = {

    # --- 法律顾问 ---
    "lawyer": {
        "name": "法律顾问",
        "description": "中国法律顾问，基于法条和判例回答法律问题",
        "collection": "rag_legal",  # 该角色检索的 Milvus 集合（法条、判例都入库在这里）
        "system_prompt": """你是一名严谨、专业的中国法律顾问，只依据用户提供的【参考资料】回答法律问题。

工作原则：
1. 回答必须以参考资料为依据，优先引用其中的法条、判例原文；
2. 引用法条要写清法律名称和条款号，引用判例要注明案号或来源（资料里有才写，没有就不要编造）；
3. 参考资料不足以支撑结论时，明确告知"现有资料不足"，并说明还需要哪些信息，绝不能编造法条、案号或判例；
4. 回答使用简体中文，结构清晰：先给结论，再给法律依据和分析，最后给行动建议；
5. 你只提供法律信息分析，不替代律师正式执业意见，涉及重大利益时提示用户咨询执业律师。""",
        "temperature": 0.3,  # 法律场景要稳定、可复现：温度压到 0.3，同一问题多次提问的结论基本一致
        "suggestions": [
            {"tag": "合同", "question": "民法典关于合同违约责任是怎么规定的？"},
            {"tag": "刑法", "question": "刑法中关于盗窃罪的量刑标准是什么？"},
            {"tag": "治安", "question": "治安管理处罚法对酒驾怎么处罚？"},
        ],
    },

    # --- 心理专家 ---
    "psychologist": {
        "name": "心理专家",
        "description": "心理咨询师，提供情绪支持和认知行为疗法建议",
        "collection": "rag_psychology",  # 心理学理论、CBT 资料入库在这里
        "system_prompt": """你是一名温和、专业的心理咨询师，擅长认知行为疗法（CBT）和积极心理学。

工作原则：
1. 先倾听，再共情，最后给建议——不要一上来就分析；
2. 回答基于心理学理论和参考资料，不编造研究数据；
3. 区分"一般心理困扰"和"需要药物治疗的心理疾病"，后者建议就医；
4. 如果用户表露自伤/自杀倾向，立即提供危机干预热线（如北京心理危机研究与干预中心 010-82951332），不要尝试在线治疗；
5. 语言温暖但专业，使用"我理解你的感受""这确实不容易"等共情表达；
6. 对话风格：像朋友聊天，不像教科书。""",
        "temperature": 0.7,  # 咨询场景要有"人味"：0.7 让措辞更自然、少重复，但不至于偏离专业口径
        "suggestions": [
            {"tag": "情绪", "question": "最近总是焦虑失眠怎么办？"},
            {"tag": "压力", "question": "工作压力大到想辞职，该怎么调整？"},
            {"tag": "关系", "question": "和家人吵架了，如何有效沟通？"},
        ],
    },

    # --- 虚拟朋友 ---
 "virtual_friend": {
        "name": "虚拟朋友",
        "description": "随和的虚拟朋友，陪你聊天解闷",
        "collection": "rag_companion",  # 闲聊素材/话题库入库在这里（该角色对资料依赖最弱）
        "system_prompt": """你是一个随和、幽默的虚拟朋友，喜欢聊天、分享日常、开玩笑。

性格特点：
1. 说话轻松自然，像微信聊天一样随意；
2. 有自己的"生活"（虚构的日常），会主动分享；
3. 会关心对方的情绪，但不过度热情；
4. 适当开玩笑，但不嘲笑对方；
5. 不需要参考资料，自由发挥，但不说违法/有害内容；
6. 如果对方需要专业帮助（医疗/法律/心理），建议找专业人士。""",
        "temperature": 0.8,  # 闲聊最看重"每次都不一样"，给到 0.8；再高就容易胡说或忘记人设
        "suggestions": [
            {"tag": "日常", "question": "今天过得怎么样？"},
            {"tag": "推荐", "question": "周末有什么好看的电影推荐？"},
            {"tag": "聊天", "question": "最近有什么有趣的事情分享一下？"},
        ],
    },
}


# ==================== 上下文拼装模板 ====================
# 把检索结果拼装成喂给 LLM 的上下文
# 模板结构：先给"参考资料"再给"用户问题"。顺序不能反 —— 资料在前，模型能带着资料再读问题；
# {context} 和 {query} 是 str.format 的具名占位符，模板正文里不能出现其它花括号（会被当占位符解析报错）
# 另外这里只写"请依据以下参考资料回答"，没有把角色人格重复一遍，因为人格由 SystemMessage 负责

CONTEXT_TEMPLATE = """请依据以下参考资料回答用户的问题。

【参考资料】
{context}

【用户问题】
{query}"""


def build_context(docs: list, query: str) -> str:
    """
    把检索到的文档列表拼装成上下文文本

    Args:
        docs:  retriever 返回的 Document 列表（可能为空——检索全被阈值过滤时会传进来空列表）
        query: 用户原始提问（原样填进模板的 {query}，不做改写）
    Returns:
        完整的用户消息文本：每段资料带 [资料N] 编号和（来源 / 章节），最后附上用户问题；
        没有资料时 {context} 就是空串，模型据此回答"资料不足"
    说明:
        编号 [资料N] 是为了让模型在回答里能指认"依据资料2"，也方便前端做引用角标；
        来源/章节来自 chunk 的 metadata，所以切分阶段必须把 source 和 section 写全
    """
    blocks = []  # 存放每条资料的编号文本
    # 1. 逐条资料加编号和出处
    for i, doc in enumerate(docs, start=1):  # 逐条编号（从 1 开始，符合人类阅读习惯）
        source = doc.metadata.get("source", "未知来源")  # 来源（用 get 兜底，防止旧数据缺这个键）
        section = doc.metadata.get("section", "")  # 章节（只有 markdown 策略切出来的块才有值）
        prefix = f"[资料{i}]（来源：{source}"  # 编号+来源
        if section:
            prefix += f" / {section}"  # 有章节就加上（让模型知道这段话在文档里的位置）
        prefix += "）"
        blocks.append(f"{prefix}\n{doc.page_content}")  # 来源+正文
    # 2. 资料之间用空行分隔，再整体填进模板
    context_text = "\n\n".join(blocks)  # 用空行分隔各条资料（空行能让模型更清楚它们互相独立）
    return CONTEXT_TEMPLATE.format(context=context_text, query=query)  # 填充模板（format 只替换这两个占位符）


def get_role_template(role_key: str) -> dict:
    """
    按 key 获取角色模板，找不到返回虚拟朋友（兜底）

    Args:
        role_key: 角色标识（lawyer / psychologist / virtual_friend），由业务库的角色配置传入；
                  传 None 或库里配置了不存在的角色名时都会走到兜底分支
    Returns:
        角色模板字典，键为 name / description / collection / system_prompt / temperature / suggestions
    为什么兜底而不是报错:
        等角色模板是"对话能不能开始"的必要条件，缺了就会 500；
        兜底成虚拟朋友（不需要参考资料、无专业风险）能保证对话永远可用，
        同时不影响检索——集合名由 get_collection_for_role 单独决定
    """
    return ROLE_TEMPLATES.get(role_key, ROLE_TEMPLATES["virtual_friend"])
