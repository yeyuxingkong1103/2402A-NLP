"""追问的查询改写。**每轮都做**（代价与理由见下）。

    rewrite_query(history, question, config) -> str | None

---

## 它解决的具体问题

用户在多轮对话里追问「那怎么预防」，而上一轮问的是「血压多少算高血压？」。

链路是 **先检索、再生成**：检索用本轮原文，而「那」不是可检索的实体词。
本模块把它还原成「高血压 怎么预防」，让检索有东西可查。

## ⚠️ 为什么是"每轮都做"，而不是"只在检索失败时做"

初版只在检索落空时改写（`is_empty` / `below_threshold`）。**那个门禁被实测推翻了：**

    问：血压多少算高血压？  → 正常
    问：那怎么预防          → 回答的是**流感**预防

实测「那怎么预防」的检索数字：

    语义路   top 余弦 0.6078（**过了阈值**）
    关键词路  覆盖率 0.33（它**正确**地拒绝了这些片段）

**关键词路拦住了，语义路把住了关**：没有主语的短问句被 BGE-M3 编码成模糊向量，
落点离「流感预防」的文本很近。于是 `is_empty=False`、`below_threshold=False` ——
**失效长得像成功**，门禁天然看不见。

同一句话的两个检索结果（实测）：

    原查询「那怎么预防」     top 余弦 0.6078   覆盖率 0.33
    改写「高血压 怎么预防」   top 余弦 0.7302   覆盖率 0.67

于是改成每轮都试，由 `dialogue._better` 择优。

**代价：每轮 +1 次模型调用（约 1~2 秒）。**
缓解是提示词的规则 3（"问题已自足就原样返回"）—— 那时本函数返回 `None`，
调用方**不会再检索一次**，只花那一次模型调用。

## ⚠️ 它推翻了一条既有裁决

`specs/010` 的 research **R12** 明确写着「MUST NOT 做查询改写」，
理由是"每轮多一次模型调用，收益不可验证"。

**那个前提变了**：用户实测撞上了这个限制。修订的理由与代价记在
`docs/superpowers/specs/2026-10-08-query-rewrite-design.md`，
R12 的原文保留并附了修订注记（不删原文 —— 删掉它会让"为什么曾经不做"
这个信息丢失，而下一个看到改写成本的人还会重新推演一遍）。

---

## ⚠️ 本模块 MUST NOT 做的事

**改写只换检索词，不换答案的依据。** 生成阶段仍然只见本轮检索到的原文片段 ——
宪法原则 II（无据不答与强制溯源引用）不因改写而松动一丝。

**改写失败 MUST 照旧拒答。** MUST NOT 降级成"拿历史硬答" —— 那与
"检索为空时不拒答"是同一件事，只是换了个触发点。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from backend.generate.client import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMError, call_model

from . import ChatError
from .models import Message

logger = logging.getLogger(__name__)

__all__ = ["REWRITE_PROMPT_PATH", "load_prompt", "render", "rewrite_query"]

# backend/chat/rewrite.py → parents[0]=chat, [1]=backend, [2]=仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REWRITE_PROMPT_PATH = PROJECT_ROOT / "data" / "prompts" / "chat_rewrite.md"

# 两个占位符。**必须都存在于提示词文件中**，且各自独占一行（注入点）——
# 与 `backend/generate/prompt.py` 的 `_has_injection_point` 同一取向：
# 记号出现在说明文字里不算数，只有当它单独占一行时才是一个真正的注入点。
PLACEHOLDERS: tuple[str, ...] = ("{history}", "{question}")

# 送进提示词的历史条数上限。
#
# ⚠️ 改写只需要"最近在聊什么"，不需要完整上下文。给太多会让提示词变长、
#    延迟增加，而收益递减 —— 「它」的指代对象几乎总在最近一两轮里。
#    取 6 条 = 最近 3 轮问答。
HISTORY_TURNS = 6

# 改写结果的长度上限（字符）。
#
# ⚠️ 这是**防御性**的，不是业务规则：提示词已经要求"短"，但模型偶尔会
#    附上一句解释（"改写后：xxx"）。超出这个长度的一律判为不可用 ——
#    宁可不改写（照旧拒答），也不要拿一段话去检索。
MAX_QUERY_CHARS = 60


def load_prompt(path: str | None = None) -> str:
    """读取改写提示词。文件缺失或缺少注入点即抛 `ChatError`。

    ⚠️ **MUST NOT 回退到内置默认提示词** —— 一份悄悄生效的默认提示词意味着
    线上行为与仓库里的文件不一致，且没有任何迹象。
    与 `backend/generate/prompt.py:load_prompt` 同一取向。
    """

    target = Path(path) if path else REWRITE_PROMPT_PATH

    if not target.is_file():
        raise ChatError(
            "改写提示词文件不存在：%s\n"
            "  它是查询改写的唯一输入，缺失时 MUST NOT 回退到内置默认值。" % target
        )

    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ChatError("改写提示词无法读取：%s —— %s" % (target, type(exc).__name__)) from exc

    if not text.strip():
        raise ChatError("改写提示词为空：%s" % target)

    missing = [t for t in PLACEHOLDERS if not any(l.strip() == t for l in text.splitlines())]
    if missing:
        raise ChatError(
            "改写提示词缺少注入点：%s（文件 %s）\n"
            "  注入点 = 某一行的内容**恰好**是这个记号。只出现在说明文字里不算数 ——"
            "那种情况下渲染不会报错，但对应的内容永远不会进入提示词。"
            % ("、".join(missing), target)
        )

    return text


def render(history: list[Message], question: str, path: str | None = None) -> str:
    """把历史与本轮问题填进提示词。

    ⚠️ **用 `str.replace` 而不是 `str.format`** —— 提示词里有大量字面大括号
    （`{history}` 这类示例、JSON 片段），`format()` 会把它们当占位符并抛
    `KeyError`。与 `backend/generate/prompt.py:render` 同一理由。

    ⚠️ **一趟替换**，不是连续两次 `replace`：连续替换会让"先替换进去的内容"
    被后一次再扫描一遍 —— 用户问题里只要出现 `{history}` 这几个字符，
    整段历史就会被插进问题中间。一趟替换对每个位置只看一次。
    """

    lines = []
    for m in history[-HISTORY_TURNS:]:
        # ⚠️ 只输出角色与内容。`timestamp` / `message_id` 对改写毫无意义，
        #    送进去只是白占 token。
        who = "用户" if m.is_user else "助手"
        lines.append("%s：%s" % (who, m.content))

    # 空历史时给一句明确的"(无)"而不是留空 ——
    # 留空会让模型以为"这里本该有东西但丢了"，而真相是首轮没有历史。
    history_text = "\n".join(lines) if lines else "（无 —— 这是这段对话的第一轮）"

    values = {"history": history_text, "question": question}
    out = load_prompt(path)
    for key, value in values.items():
        out = out.replace("{%s}" % key, value)
    return out


# 比较"改写词与原问题是否实质相同"时忽略的字符。
#
# ⚠️ 这一条是实测补的：问「血压多少算高血压？」，模型返回「血压多少算高血压」——
#    它只是**去掉了问号**。按字面比较就是"不同"，于是白跑一次编码 + 一次检索。
#
#    而那一次检索的结果与原结果几乎必然相同（同样的词，同样的向量），
#    只是白花 100~300 ms。**正常提问会因此每次都多一次无用的检索。**
_PUNCT_RE = re.compile(r"[\s，。？！、；：,.?!;:\"'“”‘’()（）]+")


def _normalized(text: str) -> str:
    """去掉空白与标点，用于判断"两个查询是否实质相同"。"""
    return _PUNCT_RE.sub("", text)


def _clean(raw: str) -> str | None:
    """把模型输出洗成一个可用的查询。不可用返回 `None`。

    洗四件事，各挡一类具体的坏输出：

        去首尾空白          模型常带换行
        只取第一行          有时会附一行解释
        剥掉包裹的引号      `"高血压 怎么预防"` 会让检索把引号也当成词
        超长即弃            见 `MAX_QUERY_CHARS`

    ⚠️ **无可替换内容时返回 `None`，MUST NOT 返回空串** ——
    空串会被当成"改写成功但查询为空"，继续走下去就是一次注定落空的检索。
    """

    if not raw:
        return None

    text = raw.strip()
    if not text:
        return None

    # 只取第一行：提示词要求一行，但模型偶尔会在后面补一句说明。
    text = text.splitlines()[0].strip()

    # 剥掉成对的引号 / 反引号（中英文都剥）。
    while len(text) >= 2 and text[0] in "\"'`“”「『" and text[-1] in "\"'`”“」』":
        text = text[1:-1].strip()

    if not text or len(text) > MAX_QUERY_CHARS:
        return None
    return text


async def rewrite_query(
    history: list[Message], question: str, config
) -> str | None:
    """改写本轮问题。**任何失败都返回 `None`，MUST NOT 抛异常。**

    ---

    ## 为什么不用 `stream_model`

    `stream_model` 的模块文档写着「重试只在第一个增量吐出去**之前**有效 ——
    首个增量一旦离开本函数，调用方就可能已经把它发给了用户」。

    而改写是**自己攒完再用**的：没有任何人看到中间过程。那条约束的理由在这里
    不成立，用它等于白付代价（中途失败不重试）。

    `call_model` 的文档正好相反：「增量被本函数攒着、没有第二个人看到，
    所以中途失败也可以**整体重试**」—— 那正是这里要的。

    ## 为什么失败不抛异常

    调用方（`dialogue.py`）在改写失败时要走的是**原来那条拒答路径** ——
    一个必然被处理、且处置唯一的失败，抛异常只是把同一件事写成两层。
    与 `backend/api/capture.py` 的取向一致：区分"吞掉"与"捕获"的标准是
    **事后能不能查出来**，这里每一条返回 `None` 的路径都留了日志。
    """

    try:
        prompt = render(history, question)
    except ChatError as exc:
        # 提示词文件缺失/写坏 —— 是**编程错误**，不是运行时故障。
        # 记 ERROR 后返回 None（照旧拒答），不把整个会话打挂。
        logger.error("chat_rewrite_prompt_invalid error=%s", exc.message.replace("\n", " "))
        return None

    try:
        raw = await call_model(
            prompt,
            api_key=config.agicto_api_key,
            base_url=config.llm_base_url or DEFAULT_BASE_URL,
            model=config.llm_model or DEFAULT_MODEL,
            timeout=config.request_timeout_s or 10.0,
        )
    except LLMError as exc:
        logger.warning(
            "chat_rewrite_failed attempts=%s reason=%s",
            exc.attempts,
            exc.message.replace("\n", " "),
        )
        return None
    except Exception as exc:  # noqa: BLE001 —— 见上方"为什么不抛"
        logger.error("chat_rewrite_error error=%s", type(exc).__name__)
        return None

    cleaned = _clean(raw)
    if cleaned is None:
        # ⚠️ **不回显模型输出** —— 它可能包含用户问题原文（含隐私）。
        #    只记长度，与 FR-025 的日志约束一致。
        logger.warning("chat_rewrite_unusable raw_chars=%d", len(raw or ""))
        return None

    # 改写与原问题**实质相同**时，视作"不需要改写"（提示词规则 3 的正常结果）。
    #
    # ⚠️ 判据是"去掉空白与标点后相同"，不是字面相同 —— 实测：问
    #    「血压多少算高血压？」，模型返回「血压多少算高血压」，只差一个问号。
    #    按字面比就是"改写了"，于是白跑一次编码 + 一次检索（见 `_PUNCT_RE`）。
    #
    # 返回它本身没有意义 —— 它既不比原文更好，也不是一个新查询。
    if _normalized(cleaned) == _normalized(question):
        logger.info("chat_rewrite_noop question_len=%d", len(question))
        return None

    # ⚠️ 只记长度，MUST NOT 记改写内容 —— 它是用户问题的变形，同样含隐私。
    logger.info(
        "chat_rewrite_ok question_len=%d rewritten_len=%d", len(question), len(cleaned)
    )
    return cleaned
