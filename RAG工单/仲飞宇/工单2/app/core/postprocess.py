"""生成结果后处理：正则清洗、敏感词替换、格式归一化。"""
from __future__ import annotations

import re

# 敏感词/违禁表述替换（示例，可按需扩充）
SENSITIVE_MAP = {
    "　": " ",
}

# 推理模型（qwen3 / deepseek-r1 等）的思维链，整段丢弃。
# qwen3 的模板确认是标准的 `<think>...</think>`（见 ollama /api/show）。
# 这里只清理「闭合完整」的块，故意不做「只有开始标签就删到结尾」的兜底：
# 各家推理标记不统一，一旦结束标记没被识别，兜底会把思维链之后的正文一起删掉，
# 那比残留一段推理文本严重得多。宁可留着，也不能吃掉答案。
#
# ⚠️ 可达性（2026-09-17 实测，别再重新推导）：
# 当前 LLM_PROVIDER=ollama 时，这个正则**不会被触发**。Ollama 把推理放在独立
# 字段里——原生 /api/chat 是 message.thinking，/v1 兼容端点是 message.reasoning
# （流式则是 delta.reasoning）——而 llm.py 只取 content，推理在进入本函数之前
# 就被丢掉了。实测三条路径的 content 均不含任何 think 标签。
#
# 它只在「把推理内联进 content」的服务商上才生效（部分 openai_compat 中转如此）。
# 那种形态真实模型走 ollama 复现不出来，覆盖它的测试见 tests/test_think_stripping.py。
_THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>\s*", re.S | re.I)

# 去除模型偶尔产出的“参考来源：[序号]”幻觉式尾巴（真实来源由 sources 字段给出）。
# 必须限定为「独占一行的、以参考来源开头的行」：早先的写法 \n*\s*(...).*$ 配合 re.S，
# 会连正文里正常的行内引用（如「限盐（参考来源：资料1）。」）一起匹配上，
# 并把该位置之后的整篇正文全部吃掉。
_REF_HEADER = re.compile(r"^(?:参考来源|参考资料|参考文献)[ \t]*[:：][^\n]*$")
# 表头下面常见的纯引用条目行（`[资料1] 中国高血压防治指南`）。
# 只删**紧跟在表头之后**的那些：表头一去掉，这些行看起来跟正文一模一样
# （带方括号的"事实"），还会被写进短期记忆。不做全局匹配，免得误伤正文里
# 形如 `[1] 第一步…` 的正常编号列表。
_REF_ENTRY = re.compile(r"^\[(?:资料|参考|来源)?\s*\d+\]")


def _strip_reference_block(lines: list[str]) -> list[str]:
    """删掉「参考来源：」行以及紧随其后的引用条目行。"""
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _REF_HEADER.match(lines[i].strip()):
            i += 1
            while i < len(lines) and _REF_ENTRY.match(lines[i].strip()):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return out


def replace_sensitive(text: str) -> str:
    for k, v in SENSITIVE_MAP.items():
        text = text.replace(k, v)
    return text


def postprocess(text: str) -> str:
    """对 LLM 输出做通用清洗：去思维链、去幻觉引用、空白归一。"""
    if not text:
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _THINK_BLOCK.sub("", text)
    text = "\n".join(_strip_reference_block(text.split("\n")))
    # 折叠连续空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 行内多余空白
    text = re.sub(r"[ \t]+", " ", text)
    text = replace_sensitive(text)
    return text.strip()
