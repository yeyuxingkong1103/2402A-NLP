"""提示词的加载与渲染。

    load_prompt()   —— 读提示词文件（**启动期调用一次**）
    render(...)     —— 填变量 + 格式化检索片段（请求期）
    format_passages(...) —— 把 `RetrievedPassage` 排成提示词要求的文本

---

## 为什么提示词从文件读，不写成 Python 字符串

提示词会被**非开发人员**审阅与修改 —— 医疗内容需要临床人员把关措辞。写成
Python 字符串的后果是可预期的：改一个词要走代码评审、要懂 diff、还要担心缩进。

放 `.md` 文件里则任何编辑器都能改，且改动在 diff 里是一段可读的中文。

**代价是「文件可能缺失或写坏」**，所以下面每个加载点都要显式失败，而不是回退到
一份内置的默认提示词 —— 一份悄悄生效的默认提示词意味着线上模型的行为与仓库里
的提示词不一致，且没有任何迹象。

## 为什么必须缓存

提示词约 5 KB，每次请求读一次盘在现代机器上可以忽略。缓存它的理由不是性能，
是**一致性**：一次问答中途文件被改，读到半新半旧的内容比读到旧内容更糟。
启动期读一次，进程生命周期内不变 —— 与 `backend/api/config.py` 的取向相同。
"""

from __future__ import annotations

import re
from functools import lru_cache

from . import ANSWER_PROMPT_FILE, DEFAULT_MODE, MODES, PromptError

__all__ = ["load_prompt", "render", "format_passages", "check_prompt"]

# 三个占位符。**必须全部存在于提示词文件中** —— 少一个意味着提示词里有一段
# 永远不会被替换的样板文字，而那通常意味着作者改了结构却忘了改这里。
PLACEHOLDERS: tuple[str, ...] = ("{mode}", "{question}", "{passages}")

# 匹配这三个记号的**精确**模式（不是 `\{[a-z_]+\}`）。
#
# 提示词文件里还有大量本该原样保留的字面大括号 —— `{n}`、`{file_name}`、
# `{page_start}` 等，它们出现在「输入格式」一节里作为示例。用宽泛模式会把它们
# 一起吃掉，留下一片空白。
_PLACEHOLDER_RE = re.compile(r"\{(mode|question|passages)\}")


@lru_cache(maxsize=1)
def load_prompt(path: str | None = None) -> str:
    """读取提示词全文。**结果按路径缓存。**

    ⚠️ 不要用 `str.format()` 渲染它，见 `render()` 的说明。
    """

    target = type(ANSWER_PROMPT_FILE)(path) if path else ANSWER_PROMPT_FILE

    if not target.is_file():
        raise PromptError(
            "提示词文件不存在：%s\n"
            "  它是答案生成的唯一输入，缺失时 MUST NOT 回退到内置默认值 ——\n"
            "  那样线上模型的行为会与仓库里的提示词不一致，且没有任何迹象。" % target
        )

    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptError("提示词文件无法读取：%s —— %s" % (target, exc)) from exc

    if not text.strip():
        raise PromptError("提示词文件为空：%s" % target)

    missing = [token for token in PLACEHOLDERS if token not in text]
    if missing:
        raise PromptError(
            "提示词缺少占位符：%s（文件 %s）\n"
            "  缺占位符意味着有一段样板文字永远不会被替换。" % ("、".join(missing), target)
        )

    return text


def _has_injection_point(text: str, token: str) -> bool:
    """`token` 是否**独占一行** —— 即存在一个真正的注入点。

    ⚠️ **这条检查是补上一个真实缺陷的。**

    最初的版本只判断 `token in text`，看起来够了，实际被一句话骗过：提示词开头
    有一行变量说明「**渲染变量**：`{mode}`、`{question}`、`{passages}`」——
    它让三个记号都"存在于文件中"，于是检查通过。但正文里**根本没有 `{passages}`
    的注入点**，真实检索片段从未进入提示词。

    失效方式是典型的静默型：`render()` 不报错、替换也"没有残留"（因为压根没有
    那个记号），模型拿到的提示词看上去完整无缺 —— 只是它手里没有任何原文。

    要求"独占一行"之后，"出现在说明文字里的记号"不再算数，只有真正的注入位置
    才算。这也顺带成了一条格式约定：**每个变量都写在一个独立的三引号块里**。
    """

    return any(line.strip() == token for line in text.splitlines())


@lru_cache(maxsize=1)
def check_prompt() -> dict:
    """提示词自检。**启动期调用**，结果用作一条可见的启动输出。

    返回 `{"path": …, "chars": …, "lines": …}`。

    检查四件事，任一不符即抛 `PromptError`：

    1. 文件存在且非空；
    2. 三个变量都**存在**；
    3. 三个变量各自有一个**独占一行的注入点**（见 `_has_injection_point`）；
    4. **不含那两句逐字话术**（constitution 原则 IV / V）。

    第 3 条是这份检查里唯一不能省的一条 —— 它挡的是一种**没有任何迹象**的失效：
    提示词看起来完整、替换也不报错，但模型手里其实没有原文。详见
    `_has_injection_point` 的说明。

    第 4 条的理由：`docs/05` §4.4 要求「全仓检索这两句逐字文本，只应命中
    `assembly_service.build`」。提示词是**最容易被顺手写进这两句**的地方
    （"让模型也复述一遍免责声明吧"），而一旦写进去，「只应命中一处」这条可测试的
    约束就失效了 —— 且不会有任何测试失败，因为提示词是数据，没有测试覆盖它。
    """

    text = load_prompt()
    _assert_no_verbatim_boilerplate(text)

    no_anchor = [token for token in PLACEHOLDERS if not _has_injection_point(text, token)]
    if no_anchor:
        raise PromptError(
            "提示词缺少注入点：%s（文件 %s）\n"
            "  注入点 = 某一行的内容**恰好**是这个记号。只出现在说明文字里不算数——\n"
            "  那种情况下渲染不会报错，但对应的内容永远不会进入提示词。"
            % ("、".join(no_anchor), ANSWER_PROMPT_FILE)
        )

    return {
        "path": str(ANSWER_PROMPT_FILE),
        "chars": len(text),
        "lines": text.count("\n") + 1,
    }


def _assert_no_verbatim_boilerplate(text: str) -> None:
    """拒绝提示词中出现固定话术的片段。

    传入的是**关键词片段**而非整句：整句匹配挡不住「只写半句」的情况，而那同样
    会让用户看到两遍相近的话。

    ⚠️ 这些片段本身**不是**逐字话术（逐字话术在 `backend/api/__init__.py`），
    所以本模块的存在不会让「全仓只应命中一处」的检索多出命中。
    """

    banned = ("请立即就医或拨打急救电话", "不能替代执业医师的当面诊断")
    for fragment in banned:
        if fragment in text:
            raise PromptError(
                "提示词里出现了固定话术：%r\n"
                "  逐字话术由装配层（assembly_service.build）唯一拼接，"
                "提示词 MUST NOT 复制它 ——\n"
                "  否则它会在回答里出现两遍，且「全仓只应命中一处」的约束失效。"
                % fragment
            )


def render(
    question: str, passages: list, mode: str = DEFAULT_MODE
) -> str:
    """渲染提示词。返回可直接发给大模型的完整文本。

    ⚠️ **用 `str.replace` 而不是 `str.format`。**

    提示词文件里包含大量**字面大括号** —— 输入格式说明里的 `{n}`、`{file_name}`、
    `{page_start}` 等。`format()` 会把它们当成占位符并抛 `KeyError`，而
    「把提示词里的示例改成双大括号转义」会让非开发人员看不懂也改不动。
    只替换这三个明确的记号，其余原样保留。
    """

    if mode not in MODES:
        raise PromptError(
            "未知的回答模式：%r（可选：%s）" % (mode, " / ".join(MODES))
        )

    if not question or not question.strip():
        raise PromptError("问题为空。空问题 MUST NOT 走到生成阶段（§4.3 的入参约束）。")

    if not passages:
        # `docs/05` §4.3：passages 为空 MUST 抛错，不允许"无上下文生成"。
        # 这条断言放在这里而不是注释里 —— 它比注释可靠，且失败点离调用方更近。
        raise PromptError(
            "检索片段为空。生成 MUST NOT 在没有上下文时进行（docs/05 §4.3）。\n"
            "  调用方应先判断 `RetrievalResult.is_empty` 并走拒答路径。"
        )

    values = {
        "mode": mode,
        "question": question.strip(),
        "passages": format_passages(passages),
    }

    # ⚠️ **一趟替换**，不是连续三次 `replace`。
    #
    # 连续替换时，「先替换进去的内容」会被「后一次替换」再次扫描：用户问题里
    # 只要出现 `{passages}` 这几个字符（罕见，但这是一段用户可以任意输入的文本），
    # 就会把整段检索片段插进问题中间。一趟替换对每个位置只看一次，从结构上
    # 排除了这种嵌套。
    return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], load_prompt())


def format_passages(passages: list) -> str:
    """把检索片段排成提示词要求的结构。

    ⚠️ **这里决定了模型能看到什么，因此改动它的影响面比看起来大。**

    只输出**库里真实存在的字段**（`backend/retrieve/models.py` 的
    `RetrievedPassage`）：来源文件、页码、章节、块类型、正文。

    **不要在这里补「发布机构」「年份」「URL」** —— 库中没有这些字段，补上去等于
    让模型看到空值或占位符，而提示词第四节明令禁止它输出片段之外的来源属性。
    两者叠加的结果是模型要么无视这段结构，要么为了"填满"而编造 —— 后者直接违反
    constitution 原则 II。要加这些字段，先补数据管线（S2/S4/S6），不是在这里拼。

    `{section}` 为空时写「（无）」而不是留空：留空会让那一行看起来是数据缺失，
    模型对「缺失」的处理是推测，对「明确写着无」的处理是跳过。
    """

    blocks: list[str] = []
    for index, passage in enumerate(passages, start=1):
        section = (getattr(passage, "section", None) or "").strip() or "（无）"
        blocks.append(
            "[片段 {n} | 来源文件：{file_name} | 页码：{page_start}-{page_end} "
            "| 章节：{section} | 块类型：{block_type}]\n正文：{text}".format(
                n=index,
                file_name=passage.file_name,
                page_start=passage.page_start,
                page_end=passage.page_end,
                section=section,
                block_type=passage.block_type,
                text=passage.text,
            )
        )

    return "\n\n".join(blocks)
