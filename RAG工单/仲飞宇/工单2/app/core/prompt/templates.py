"""提示词模板：系统人设 + 检索上下文 + 短期记忆 + 用户问题。

链路位置：pipeline.answer / pipeline.answer_stream 在「检索完 → 取完短期记忆」之后调用
build_messages，产出的 messages 直接交给 LLMClient。流式与非流式共用这一份拼装，两条路径
看到的系统消息因此一字不差。本模块不碰 store、不发请求，只做字符串拼接与字符预算计算。

这里也是「只依据知识库资料作答」那段通用约束的唯一出处（分有资料 / 零召回两种措辞），
角色自己的免责口径在 role_presets.py 的各条 system_prompt 里。
"""
from __future__ import annotations

from ..logging_config import get_logger

log = get_logger("prompt")

# OpenAI 消息格式
Messages = list[dict]


def _format_contexts(contexts: list[dict]) -> str:
    # 「[资料N]（来源：标题）」这个前缀与下面作答要求里「每一条结论都应能对应到上面某条
    # 资料」是绑定的，改前缀会让那句要求失去所指。标题缺失时写「未知来源」而不是留空，
    # 否则会留下一个空的「（来源：）」让模型去猜。
    parts = []
    for i, c in enumerate(contexts, 1):
        title = c.get("title") or "未知来源"
        parts.append(f"[资料{i}]（来源：{title}）\n{c['text']}")
    return "\n\n".join(parts)


def _group_turns(history: list[dict]) -> list[list[dict]]:
    """把扁平的消息列表按「一轮」分组：一条 user + 紧随其后的 assistant。

    从第一条 user 开始——万一历史以 assistant 开头（不该发生，但存储损坏时可能），
    那条「无头」助手消息会让模型莫名其妙。
    """
    # 只放行 user/assistant：记忆里若混进 system 之类的角色，会以「历史」的身份挤进
    # 消息列表，多家 OpenAI 兼容端点对多条 system 的处理并不一致（有的只认第一条、
    # 有的直接报错），不如在这里统一滤掉。
    msgs = [h for h in history if h.get("role") in ("user", "assistant")]
    start = next((i for i, m in enumerate(msgs) if m["role"] == "user"), len(msgs))
    msgs = msgs[start:]

    turns: list[list[dict]] = []
    cur: list[dict] = []
    for m in msgs:
        if m["role"] == "user" and cur:
            turns.append(cur)
            cur = []
        cur.append(m)
    if cur:
        turns.append(cur)
    return turns


def trim_history(history: list[dict], max_chars: int | None) -> list[dict]:
    """按字符预算保留最近的若干轮，超出就丢弃更早的。

    为什么按字符而不是按轮数：真正决定「装不装得下」的是**回答长度**。实测单条
    回答 1670 字时，4096 的上下文只装得下 3 轮；回答 28 字时 40 轮也没问题。
    按轮数裁剪会在长回答时照样超限，短回答时又白白浪费。

    这里做的裁剪是**显式的**：不裁的话 Ollama 会自己静默丢掉最老的几轮，
    应用既看不见也控制不了。裁在应用侧至少是可预测、可记录的。

    以「轮」为最小单位，不拆散一问一答——留下有问无答或答非所问的半轮，
    比整轮丢掉更糟。
    """
    turns = _group_turns(history)
    if max_chars is None:
        return [m for t in turns for m in t]

    kept: list[list[dict]] = []
    total = 0
    for t in reversed(turns):
        n = sum(len(m.get("content") or "") for m in t)
        # kept 非空才判预算：最新一轮再长也要留下，否则整段历史会全空
        if kept and total + n > max_chars:
            break
        kept.append(t)
        total += n
    kept.reverse()

    # 留痕：被丢弃是这套机制存在的理由（Ollama 会静默丢），应用侧再静默就说不过去了
    dropped = len(turns) - len(kept)
    if dropped:
        log.info(
            "历史裁剪：丢弃最早 %d 轮，保留 %d 轮（预算 %d 字符，实际 %d）",
            dropped, len(kept), max_chars, total,
        )

    return [m for t in kept for m in t]


def build_messages(
    persona: dict,
    contexts: list[dict],
    history: list[dict],
    question: str,
    max_history_chars: int | None = None,
    max_prompt_chars: int | None = None,
) -> Messages:
    """组装一条完整对话的消息列表。

    persona: {"name": ..., "system_prompt": ...}
    contexts: 检索到的 chunk 列表 [{text, title, source, score}]
    history: 短期记忆 [{role, content}]（按时间顺序）
    max_history_chars: 历史部分的字符预算；None 表示不裁剪
    max_prompt_chars: 整个提示词的字符预算（系统消息 + 历史 + 提问）；None 表示不限制

    两个预算同时给时取更紧的那个。整体预算是必要的：max_history_chars 管不到系统
    消息，而系统消息里塞着 TOP_K 条检索资料，随 chunk_size 增长。只有历史预算时，
    资料一大，历史就按原额度照留，prompt 整体超窗，Ollama 反而会静默丢最老轮次——
    正是这套裁剪要消灭的情况。
    """
    # persona 的 name 只作调用侧的语义标注，不拼进提示词：人设文本自己已经交代了身份
    # （「你是一位资深的心内科医生」），再注入一句「你的名字是 XX」对答案没有帮助，
    # 反而白占 PROMPT_MAX_CHARS 的预算。
    system = persona["system_prompt"]

    if contexts:
        system += "\n\n===== 知识库资料 =====\n" + _format_contexts(contexts)
        # 光靠人设里那句「不要编造资料中不存在的内容」不够：模型被问到资料里只有
        # 一句结论的问题时，会把它知道的专业常识（机制、发生率、其他情形…）
        # 一并写进来。它不认为这是编造——那些内容专业上是对的——所以这条禁令对它
        # 不生效。实测 ACEI 一题：资料只有「常见副作用为干咳」，回答却补出了缓激肽
        # 机制、体位性低血压、高钾血症、肾功能影响、血管性水肿，faithfulness 判 0。
        # 必须点破「补充你不知道的资料外内容也算违规」，并说明详略要与资料相称，
        # 否则模型会按「helpful = 完整」的本能继续扩写。
        #
        # 措辞是「专业内容」而不是「医学内容」：这份模板服务全部角色，
        # 其中没有一个是纯医疗角色，写死成医学等于对律师那位没有约束
        # （实测律师题就补出了资料里没有的「双倍工资最长 11 个月」）。
        #
        # 2026-09-21 试过**再加一条**「数字、数量、期限必须逐个能在资料里找到，不许把
        # 「一拳头」换算成克」——起因是实测见过一次营养师答案把「一拳头主食」写成
        # 「50-80 克」。A/B（新旧各 3 次采样，同一问题同一角色）结果：那条故障本身没能
        # 再复现（新旧都 0/3），而**稳定复现**的律师题（资料只写「超过一个月不满一年
        # 应付二倍工资」，回答补「最长不超过 11 个月」）加条款前后都是 2/2 照样补。
        # 即：没有证据支持这条有用，而系统消息每多一个字都在吃 PROMPT_MAX_CHARS 的
        # 历史预算，所以回滚、不留。真治这类"资料没写、模型按常识补"，得换手段
        # （用 RAGAS 量化 faithfulness / 调温度 / 对答案里的数字做事后校验），
        # 不是继续往提示词里加条款。
        system += (
            "\n\n===== 作答要求 =====\n"
            "上面的【知识库资料】是本次回答唯一可用的事实来源。严格遵守：\n"
            "1. 只陈述资料里明确写出的事实。不要补充资料之外的专业内容——包括成因/机制、"
            "发生率、具体数值、其他情形、注意事项等，即使这些内容你本来就知道、"
            "且专业上正确；也不要用「可能与……有关」「通常认为」这类推测句式把它"
            "夹带进来——加个「可能」仍然是在补充资料外的内容；\n"
            "2. 回答的详略要与资料的信息量相称：资料只给出一句结论时，就把这一句讲清楚，"
            "不要为显得完整而扩写成一篇科普；\n"
            "3. 资料不足以回答的部分，直接说「资料中没有提到」，不要用推测或常识补全；\n"
            "4. 每一条结论都应能对应到上面某条资料。"
        )
    else:
        # 一条资料都没召回（集合没建/名字配错、score_threshold 把召回全滤掉、BM25 那一路
        # 失效、这个角色根本还没入库……）。以前这种时候 system 只剩人设，而人设里那句
        # 「不要编造资料中不存在的内容」也就失去了所指——/chat 返回 200 + sources: []，
        # 模型按常识把专业问答写得很像样，表面上一切正常，正是本文件注释自陈的
        # faithfulness=0 形态（零资料下自由发挥）。上面的空召回告警只能让**服务端**看见，
        # 用户看到的仍是一段像模像样的答案。
        #
        # 但不能一刀切成"拒答"：角色扮演、寒暄、追问设定本来就该照常回，否则没入库的角色
        # 一开口就是"知识库中没有相关内容"。所以只禁"具体事实"，把"没有资料"变成模型
        # 说得出、用户看得见的一句话。
        system += (
            "\n\n===== 作答要求 =====\n"
            "本次**没有检索到任何知识库资料**（资料为空）。严格遵守：\n"
            "1. 与角色设定、闲聊、寒暄相关的回应照常给出，保持人设语气；\n"
            "2. 不要凭空给出具体的专业事实、数值、标准或建议，也不要引用任何资料编号"
            "——这些必须以资料为依据，而本次没有资料；请直接说明「知识库中没有检索到"
            "相关内容」，并提示用户可以补充文档或换个问法。"
        )

    budget = max_history_chars
    if max_prompt_chars is not None:
        # 系统消息和本轮提问是「固定开销」，先扣掉，剩下的才轮到历史
        room = max_prompt_chars - len(system) - len(question)
        if budget is None or room < budget:
            if room <= 0:
                # 预算被"固定开销"吃光了。此时历史怎么裁都救不回来，prompt 必然超窗，
                # 而 Ollama 超窗是**静默丢最老轮次**（实测 finish_reason 仍是 stop，
                # 没报错），日志里只剩一条「保留 N 轮」与实际不符。
                # 以前这里只说「系统消息 N 字 >= 上限」，把排查引向 chunk_size——
                # 但真正吃光预算的常常是**本轮提问本身**（question 长度此前没上界）。
                log.warning(
                    "prompt 预算被吃光，历史无论怎么裁都会超窗，Ollama 将静默丢弃最老轮次："
                    "系统消息 %d 字 + 本轮提问 %d 字 >= 上限 %d 字。"
                    "请调小 chunk_size（接口上限 1000）/ 缩短提问 / 调大 PROMPT_MAX_CHARS",
                    len(system), len(question), max_prompt_chars,
                )
            else:
                log.info(
                    "整体预算收紧历史：%s -> %d 字符（prompt 上限 %d，系统消息 %d 字）",
                    "不限" if budget is None else budget, room, max_prompt_chars, len(system),
                )
            budget = room

    # 固定形状：[system] + 历史（已裁剪）+ [本轮提问]。提问永远是最后一条，且不计入
    # 历史的字符预算（上面扣 room 时已经先减掉了它的长度）。历史里的 content 原样透传，
    # 写进记忆的已经是 postprocess 清洗过的文本，这里不再二次处理。
    messages: Messages = [{"role": "system", "content": system}]
    for h in trim_history(history, budget):
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": question})
    return messages
