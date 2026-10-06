# -*- coding: utf-8 -*-
"""引用展示：命中句高亮 + 参考资料卡片（HTML）。"""
from html import escape  # HTML 转义：片段原文可能含 <>&，防注入破坏页面

def _bigrams(text: str) -> set:  # 字级二元组集合：中文没有空格分词，用 bigram 近似"词重叠"
    """中文用字级二元组近似词重叠，避免为引用展示再引一次分词。"""
    clean = "".join(ch for ch in text if ch.strip())  # 先去掉空白字符，不让空格参与匹配
    return {clean[i:i + 2] for i in range(len(clean) - 1)}  # 滑动窗口取相邻两字成集合，后面用集合交集算重叠度

def _split_sentences(text: str) -> list:  # 按中文标点断句
    """按中文标点断句。"""
    out, cur = [], ""  # out 存成句，cur 攒当前句
    for ch in text:  # 逐字扫描
        cur += ch  # 累积字符
        if ch in "。！？；\n":  # 遇到句末标点或换行就切一刀
            out.append(cur)  # 成句入列
            cur = ""  # 清空缓冲，开始下一句
    if cur:  # 末尾没标点的残句
        out.append(cur)  # 也收进来
    return [s for s in out if s.strip()]  # 滤掉纯空白句

def highlight(text: str, query: str) -> str:  # 引用高亮：在原文里标出与问题最相关的那句话
    """把与 query 重叠最多的那句话高亮，返回转义后的 HTML。"""
    grams = _bigrams(query)  # 查询的 bigram 集合
    best, best_hit = "", 0  # 记录重叠数最多的句子及其重叠数
    for sent in _split_sentences(text):  # 逐句比对
        hit = len(grams & _bigrams(sent))  # 集合交集大小 = 该句与查询的字面重叠度
        if hit > best_hit:  # 刷新最优句
            best, best_hit = sent, hit  # 记下来
    safe = escape(text)  # 先整体转义再插标签：顺序不能反，否则 <mark> 也会被转义掉
    if best.strip() and best_hit:  # 有有效命中才高亮
        safe = safe.replace(escape(best), f"<mark>{escape(best)}</mark>", 1)  # 只替换第一处，包 <mark> 黄底高亮
    return safe  # 返回可安全渲染的 HTML

def label(item: dict, rank: int) -> str:  # 生成引用卡片的折叠标题
    """expander 标题：来源文档 + 页码 + 相关度。"""
    page = f"第 {item.get('page')} 页" if item.get("page") else "页码未知"  # 页码缺失时友好兜底
    return (  # 相邻 f-string 隐式拼接
        f"📚 参考资料 [{rank}] {item.get('source', '未知来源')} · {page}"  # 来源+序号+页码：答案可溯源，用户能回查原文
        f" · 相关度 {item.get('score', 0):.2f}"  # 附相关度分数，置信度透明展示
    )  # 拼出完整标题

def card(item: dict, query: str) -> str:  # 展开后的引用卡片：元信息行 + 高亮原文
    """展开后的原文片段（命中句已高亮）。"""
    page = f"第 {item.get('page')} 页" if item.get("page") else "页码未知"  # 同样的页码兜底
    return (  # 相邻 f-string 隐式拼接成一段 HTML
        f"<div class='cite-meta'>📄 <b>{escape(str(item.get('source', '未知来源')))}</b> · "  # 元信息行：来源文档名加粗，防注入转义
        f"{page} · 相关度 {item.get('score', 0):.2f}"  # 页码与相关度
        f"</div><div class='cite-text'>{highlight(item.get('text', ''), query)}</div>"  # 正文区：嵌入高亮后的原文片段
    )  # 拼出完整卡片 HTML

def trace(query: str, ranked: list, limit: float, question_type: str = "") -> str:  # 生成检索追踪文本：复盘"为什么答/为什么不答"
    """检索详情文本（UI 可展开查看意图、阈值、每条候选分数，答辩讲解用）。"""
    from src import reranker  # 函数内导入：避免与展示层产生不必要的模块耦合

    status = "已启用" if reranker.is_ready() else "未启用（回退向量分数）"  # 展示重排模型当前状态
    head = (f"检索问句：{query}\n问题类型：{question_type}\n阈值：{limit:.2f}"  # 头部信息：实际生效的问句/意图/阈值
            f"\n重排模型：{status}")  # 补上重排模型状态
    if not ranked:  # 无候选（没召回或全被过滤）
        return head + "\n（无候选）"  # 明示空结果
    rows = [f"[{i}] {c['score']:.3f} | {c['source']} 第{c['page']}页"  # 每行：排名、分数、来源、页码
            for i, c in enumerate(ranked, 1)]  # 序号从 1 开始
    return head + "\n" + "\n".join(rows)  # 拼成完整追踪文本
