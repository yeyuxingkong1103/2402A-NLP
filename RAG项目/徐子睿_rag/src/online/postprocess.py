"""src/online/postprocess.py —— 答案后处理：清洗、引用生成、拒答兜底。

在链路中的位置：
    src/online/llm.py 的输出 → 【本文件】 → src/online/chain.py 返回给接口层

三个函数对应三件必须做的事，顺序不能颠倒（见 chain.py）：
    clean_answer     先把模型输出的噪声去掉
    references       再从"本轮实际检索到的片段"生成引用（不信模型自己写的）
    enforce_refusal  最后检查有没有依据，没有就换成固定拒答文案

核心主张："答案带引用 + 无依据不编造"就落在本文件里。
引用由程序按检索结果生成、拒答由程序按有没有命中的片段决定 ——
都不依赖模型自觉，而是由代码保证。
"""
from __future__ import annotations

import re
from typing import Any


def clean_answer(text: str) -> str:
    """清洗模型输出的噪声。

    参数：
        text: 模型原始输出
    返回：
        清洗后的文本（首尾空白已去除）。

    三步，对应三类实际的模型输出问题：
        1. 去掉 <think>…</think> —— 推理型模型的思考过程不该展示给用户
        2. 去掉 Markdown 代码块围栏 —— 模型习惯把答案包在 ``` 里，
           但这里是纯文本界面，围栏会原样显示成乱码
        3. 3 个以上连续换行压成 2 个 —— 模型常输出大片空行

    用 flags=re.S 让 . 能匹配换行：
        思考过程经常是多行的，不开 DOTALL 就删不干净，会留下一堆残句。
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    text = text.replace("```markdown", "").replace("```", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def references(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从检索片段生成引用列表（带去重）。

    参数：
        hits: 排好序的检索结果
    返回：
        [{"source": 来源, "page": 页码, "content": 正文前 160 字}, ...]

    去重键用 (source, page) 而不是整条内容：
        同一页可能被切成多个 chunk 都被召回，
        对用户来说"这一页"只需要出现一次引用 —— 按页去重更符合阅读直觉。

    content 截断到 160 字：
        引用是用来"让用户核对依据"的，前 160 字足够看出这段在讲什么；
        给全文会让响应体膨胀数倍，而用户真要看全文会去知识库页翻。

    先出现的先保留（不排序）：
        hits 是已按相关性排序的，引用顺序跟着相关性走，
        最新的依据排在最前，与答案的行文顺序也基本一致。
    """
    result, seen = [], set()
    for hit in hits:
        entity = hit.get("entity") or hit
        source = entity.get("doc_source") or entity.get("source") or "unknown"
        page = entity.get("page", -1)
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        result.append({"source": source, "page": page, "content": (entity.get("content") or entity.get("text") or "")[:160]})
    return result


def enforce_refusal(answer: str, hits: list[dict[str, Any]]) -> str:
    """没有检索到任何依据时，强制替换成固定拒答文案。

    参数：
        answer: 已清洗的模型答案
        hits: 本轮检索结果
    返回：
        有依据时原样返回答案；无依据时返回固定拒答文案。

    为什么要"强制替换"而不是"信任模型自己拒答"：
        没有依据时模型仍可能凭参数记忆编出一个像模像样的答案 ——
        这正是 RAG 要防的核心风险。既然本轮确实一条片段都没检索到，
        那么任何答案都必然没有知识库依据，直接换掉才是可靠的。

    这里判断的是"有没有 hits"（检索层面的事实），
    而不是"模型说没说不知道"（模型层面的自述）——
    前者是客观的，后者可能被模型漏报。

    注意拒答文案与 backend/roleplay.py、backend/server.py 里的措辞一致：
        都是"知识库中未找到相关依据"这个统一口径，
        前端和评测脚本都靠这句话来判定一次回答是否为拒答。
    """
    if hits:
        return answer
    return "知识库中未找到相关依据，无法回答；建议补充资料或咨询专业人士。"
