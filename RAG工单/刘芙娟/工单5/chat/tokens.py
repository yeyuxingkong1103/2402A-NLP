"""token 估算。**这是估算，不是精确计数。**

    estimate_tokens(text) -> int

---

## ⚠️ 为什么是估算

项目现有的 tokenizer（`transformers` 加载的 BGE-M3）服务于**嵌入模型**，
与对话模型的计费口径无关 —— 它切出的 token 数不能拿来预估 deepseek 的消耗。

而调用 deepseek 自己的 tokenizer 需要额外的网络依赖，与"启动期加载一次、
请求期不做出网"的既有取向直接冲突。

所以这里用**粗估**：中文约 1 字 = 1 token，其他约 4 字符 = 1 token。

## 这个估算的误差有多大、会不会出问题

中文侧比较准（1 字 ≈ 1 token 是主流中文 BPE 的经验值）；
英文侧偏保守。真正的影响是：**预算可能被用超，或提前裁掉几条消息**。

超预算时由模型返回 400，被 `LLMError` 捕获后走既有的失败处置 —— 不会静默出错。
而裁多几条的代价是模型看到的上下文略少，不影响正确性。

⚠️ **MUST NOT 在本模块或任何文档中声称计数与模型的实际计费口径一致。**
（FR-011 明确要求如实标注。）
"""

from __future__ import annotations

__all__ = ["estimate_tokens", "is_cjk"]

# CJK 统一汉字区间。与 `backend/retrieve/lexical.py:_is_content` 用的是同一段 ——
# 那边判"这个 token 有没有实际内容"，这边判"这个字符算不算一个 token"。
_CJK_START = "一"
_CJK_END = "鿿"

# 非 CJK 字符每多少个算一个 token。
#
# 取 4 而不是 3：英文的技术文本（药物名、缩写）token 密度比自然英文低，
# 取 4 让估算**偏保守**（宁可多算，也不要少算到超预算）。
_NON_CJK_PER_TOKEN = 4


def is_cjk(ch: str) -> bool:
    """该字符是否落在 CJK 统一汉字区间。

    只覆盖基本区（U+4E00–U+9FFF），不含扩展区（生僻字、异体字）与日韩假名。
    理由：医疗文本里的生僻汉字极少，而扩展区的码点范围有六段且不连续，
    为它们增加六个判断分支换不来可测量的准确度提升。

    ⚠️ 全角标点（，。；）**不算** CJK —— 它们走"每 4 个字符 1 token"的分支。
    虽然有偏差，但标点在中文文本里占比很低，且方向是**保守**（多算）。
    """

    return _CJK_START <= ch <= _CJK_END


def estimate_tokens(text: str) -> int:
    """估算 `text` 的 token 数。空串返回 0。

    ⚠️ **返回值是估算值。** 调用方用它做预算裁剪（`context.build_context_window`），
    MUST NOT 用它做任何需要精确性的判断（如计费、配额）。
    """

    if not text:
        return 0

    cjk = sum(1 for ch in text if is_cjk(ch))
    other = len(text) - cjk

    # 向上取整：`other` 不为 0 时至少算 1 个 token。
    # 用 `(other + 3) // 4` 而不是 `ceil(other / 4)` —— 后者要经过浮点，
    # 而大文本下浮点的舍入方向不值得推敲。
    return cjk + (other + _NON_CJK_PER_TOKEN - 1) // _NON_CJK_PER_TOKEN
