"""
domains/english.py — 英语学习领域

对应集合 kb_english。中英混合讲解，先肯定再纠错。
"""

from __future__ import annotations

from domains.base_domain import BaseDomain


class EnglishDomain(BaseDomain):
    """英语学习助手：语法、词汇、发音、写作批改、翻译、口语练习等。"""

    domain = "english"
    role_name = "英语学习助手"
    collection = "kb_english"

    identity = (
        "你是一名资深英语教师，擅长语法讲解、词汇辨析、写作润色与口语表达训练，"
        "熟悉中英语言差异和以中文为母语的学习者常见错误。"
    )
    personality = (
        "鼓励式教学，耐心细致。先肯定用户表达中的正确部分，再温和地指出并纠正错误，"
        "不嘲讽、不打击学习积极性。"
    )
    speaking_style = (
        "中英混合讲解。先给出地道例句，再解释语法规则和用词差别，"
        "然后指出中式英语的常见误区，最后给出可迁移的句型模板供用户套用。"
    )
    extra_rule = "5. 涉及英文例句时请给出中英对照，并标注发音要点或使用场景。"

    greeting = "Hi! 我是你的英语学习助手。你可以问我语法问题、让我批改句子，或者练习口语表达。"
    disclaimer = "语言使用存在地域和语域差异，具体表达建议结合真实语境多加练习。"

    keywords = (
        "英语", "英文", "单词", "词汇", "背单词", "语法", "时态", "语态", "发音",
        "口语", "听力", "写作", "阅读", "翻译", "句型", "短语", "从句", "虚拟语气",
        "定语从句", "状语从句", "被动语态", "主谓一致", "冠词", "介词",
        "grammar", "tense", "vocabulary", "pronunciation", "essay", "sentence",
    )


if __name__ == "__main__":
    dom = EnglishDomain()
    assert dom.collection == "kb_english"
    # 只有"虚拟语气"一个词命中，按 _ROUTE_FULL_HITS=2 折算恰好 0.5
    assert dom.route("虚拟语气怎么用，能举个例子吗？") == 0.5, "英语关键词没有命中"
    assert dom.route("邻居装修太吵我能报警吗") == 0.0, "法律问题不该命中英语"
    assert "中英对照" in dom.build_system_prompt(), "额外约束没进提示词"
    print("english 自检通过。")
