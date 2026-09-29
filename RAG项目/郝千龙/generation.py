# -*- coding: utf-8 -*-  # 声明本文件使用 UTF-8 编码，避免中文注释乱码
"""多角色通用：提示词拼装 + RAG 生成（含流式）。"""  # 模块文档字符串：说明本文件职责——拼提示词 + RAG 检索增强生成，并支持流式输出
from __future__ import annotations  # 启用"延迟注解求值"，让 list[dict] 等泛型注解在低版本 Python 也能直接写

from collections.abc import Iterator  # 从抽象基类导入 Iterator，用于 chat_stream 返回类型注解（比 typing.Iterator 更规范）

from openai import OpenAI  # 导入官方 OpenAI SDK 客户端类，兼容所有 OpenAI 协议端点（DeepSeek/豆包/vLLM 等）

from config import DEFAULT_ROLE_CODE, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, SHORT_MEMORY_TURNS  # 导入全局配置：API Key、端点、模型名、短记忆轮数、默认角色
from rag import SentencePair, TranslationRetriever  # 导入 RAG 模块的数据类 SentencePair 与检索器 TranslationRetriever
from roles import get_role  # 导入角色工厂函数，按 role_code 取角色人设（system_prompt 等）


def format_retrieved(hits: list[tuple[SentencePair, float]]) -> str:  # 定义格式化函数：入参是 (句子对, 得分) 列表，出参是字符串
    """把检索结果格式化为带序号与得分的文本（进提示词 / 展示）。"""  # 文档字符串：用途说明——既可塞进提示词，也可直接展示给用户
    if not hits:  # 防御性判断：没有任何检索结果时
        return "（本次没有检索到相关资料）"  # 返回兜底提示文案，避免空字符串进提示词导致模型乱编
    lines = []  # 用列表收集每行文本，便于最后 join 拼接（比字符串 += 更高效）
    for i, (pair, score) in enumerate(hits, 1):  # enumerate 从 1 开始编号，解包出 (句子对, 相似度得分)
        lines.append(f"{i}. [{score:.2f}] {pair.as_text()}")  # 拼出"序号. [得分] 中英文对照"格式，得分保留两位小数
    return "\n".join(lines)  # 用换行符连接所有行，得到一段可直接塞进提示词的多行文本


def fallback_answer(query: str, hits: list[tuple[SentencePair, float]]) -> str:  # 兜底回答函数：入参用户问题 + 检索结果，出参是字符串
    """兜底回答：未配置 LLM_API_KEY 时，只讲解检索结果（纯检索模式）。"""  # 文档字符串：说明在没有大模型可用时如何"只用检索结果"回答
    if not hits:  # 如果连一条相关资料都没检索到
        return "知识库里没有找到很接近的内容。请换一种说法，或补充例句后再问。"  # 引导用户换说法或扩充语料，体现检索系统的反馈闭环
    return (  # 有资料时返回拼装好的"检索讲解式"答案（隐式字符串拼接）
        "我先用知识库里最接近的资料来回答：\n\n"  # 答案开场白：明确告知答案来源是知识库，而非模型生成
        f"{format_retrieved(hits)}\n\n"  # 复用 format_retrieved 把检索结果格式化后插入，避免重复造轮子
        "学习建议：对比中英文差异，跟读 1～2 句，再改写成你自己的话。\n"  # 给出可执行的学习建议，体现"教师"角色的引导性
        "（未配置 LLM_API_KEY 时为检索讲解模式。）"  # 显式标注当前为检索模式，方便答辩/排障时识别
    )


def build_messages(  # 定义提示词拼装函数：构造 OpenAI 协议的 messages 列表
    query: str,  # 用户本轮提出的问题
    hits: list[tuple[SentencePair, float]],  # RAG 检索到的 (句子对, 得分) 列表
    history: list[dict],  # 之前的对话历史，每条形如 {"role":..., "content":...}
    role_code: str = DEFAULT_ROLE_CODE,  # 角色 code，默认取配置的默认角色，决定 system_prompt 与检索路径
) -> list[dict]:  # 返回值：可直接传给 OpenAI chat 接口的 messages 列表
    """拼装多轮对话提示词：system 角色设定 + 短期记忆 + 检索资料 + 用户问题。"""  # 文档字符串：点明拼装顺序——这是 RAG grounding 的关键
    role = get_role(role_code)  # 通过角色工厂取出该角色的人设字典（含 system_prompt 等）
    user_block = f"【检索到的资料】\n{format_retrieved(hits)}\n\n【用户问题】\n{query}"  # 把检索资料与用户问题合并成一条 user 消息，并用【】标签清晰分块
    messages = [{"role": "system", "content": role["system_prompt"]}]  # 第 1 块：system 消息放角色人设，约束模型行为与幻觉
    trimmed = history[-SHORT_MEMORY_TURNS * 2 :]  # 截取最近 N 轮（每轮 user+assistant 两条所以 *2），控制上下文长度与成本
    messages.extend(trimmed)  # 第 2 块：把截取后的历史拼到 system 之后，保持对话连贯
    messages.append({"role": "user", "content": user_block})  # 第 3 块：把"检索资料 + 用户问题"作为本轮 user 消息追加，让模型"开卷考试"
    return messages  # 返回拼好的 messages，顺序为 system → 历史 → 检索资料+问题，符合 RAG 提示词最佳实践


def _client() -> OpenAI:  # 私有工厂函数：返回一个 OpenAI SDK 客户端实例，下划线前缀表示内部使用
    """OpenAI 客户端（兼容任意 OpenAI 协议端点：DeepSeek/豆包/本地 vLLM 等）。"""  # 文档字符串：强调兼容性——只要遵守 OpenAI 协议即可接入
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL or None)  # 用配置中的 Key 与端点构造客户端；base_url 为空时回退 None 走官方默认


def chat(  # 非流式主入口函数：一次性返回完整答案 + 命中的检索结果
    query: str,  # 用户本轮问题
    retriever: TranslationRetriever,  # 已建好的检索器实例（向量+BM25 等混合）
    history: list[dict] | None = None,  # 对话历史，None 时按空列表处理
    top_k: int | None = None,  # 每路检索返回条数，None 用检索器默认
    role_code: str = DEFAULT_ROLE_CODE,  # 角色码，决定 system_prompt 与检索语料过滤
) -> tuple[str, list[tuple[SentencePair, float]]]:  # 返回 (模型文本答案, 检索命中列表)，方便 UI 展示引用
    """非流式生成：改写 → 混合检索 → 重排 → LangChain/OpenAI 生成。"""  # 文档字符串：勾勒整条 RAG 处理链路，便于答辩讲流程
    from query_rewrite import rewrite_query  # 延迟导入 query 改写模块，避免顶层循环依赖、加快冷启动
    from rerank import rerank  # 延迟导入重排模块；同样为减少不必要的导入开销

    history = history or []  # None 兜底成空列表，后续切片才不会报错
    rewritten = rewrite_query(query)               # ① Query 改写：把口语化/带错别字的问题规范化，提升召回率
    hits = retriever.hybrid_search(rewritten, top_k=top_k, role_code=role_code)  # ② 混合召回（向量+BM25），按 role_code 过滤向量路的语料域
    if hits:  # 有召回结果时才做重排，避免空列表报错
        docs = [p.as_text() for p, _ in hits]  # 把每个句子对拍平成纯文本，供 rerank 模型打分
        order = rerank(query, docs)                # ③ 重排精排：用 cross-encoder 之类的模型对召回结果二次排序，提高精度
        hits = [hits[i] for i in order if i < len(hits)]  # 按重排顺序重建 hits 列表，并防止越界索引
    if not LLM_API_KEY:  # 没有配置 LLM Key 时，不能调用大模型
        return fallback_answer(query, hits), hits  # ④ 无 KEY：返回纯检索讲解式答案，体现"检索可独立工作"
    from langchain_chain import langchain_generate  # 延迟导入 LangChain 链路；只在实际要用到时才导入，减小初始依赖

    # ⑤ 优先走 LangChain 链路（历史拼成文本注入提示词）
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-SHORT_MEMORY_TURNS * 2 :])  # 把最近 N 轮历史拼成纯文本，便于塞进单一 prompt
    lc = langchain_generate(query, format_retrieved(hits), history_text, role_code)  # 调 LangChain 链路生成，传入问题、检索资料、历史文本、角色
    if lc:  # LangChain 返回非空文本，视为成功
        return lc.strip(), hits  # 直接返回 LangChain 的答案（已 strip 去首尾空白）
    # ⑥ LangChain 失败：原生 OpenAI SDK 兜底
    messages = build_messages(query, hits, history, role_code=role_code)  # 用 build_messages 重新拼装标准 messages，作为兜底链路的输入
    resp = _client().chat.completions.create(  # 调用 OpenAI 协议的 chat completions 接口
        model=LLM_MODEL,  # 使用配置中指定的模型名（如 deepseek-chat、qwen-plus 等）
        messages=messages,  # 传入拼好的多轮 messages
        temperature=0.4,  # 温度 0.4 偏低，兼顾稳定性与表达自然度，适合教学场景
    )
    text = (resp.choices[0].message.content or "").strip()  # 取第一个候选的文本，or "" 防止 None，再 strip 去首尾空白
    return text, hits  # 返回模型答案与检索命中，调用方可据 hits 展示引用源


def chat_stream(  # 流式主入口函数：生成器函数，逐 chunk 吐出文本，前端可边收边显示
    query: str,  # 用户本轮问题
    retriever: TranslationRetriever,  # 已建好的检索器实例
    history: list[dict] | None = None,  # 对话历史，None 时按空列表处理
    top_k: int | None = None,  # 每路检索返回条数
    role_code: str = DEFAULT_ROLE_CODE,  # 角色码
) -> Iterator[str]:  # 返回类型注解：字符串迭代器，调用方按需迭代获取片段
    """流式生成：检索阶段同 chat，生成阶段逐 chunk yield（低首字延迟）。"""  # 文档字符串：说明流式优势——首字延迟低、用户体验好
    history = history or []  # None 兜底成空列表
    from query_rewrite import rewrite_query  # 延迟导入改写模块
    from rerank import rerank  # 延迟导入重排模块

    rewritten = rewrite_query(query)  # ① Query 改写，与 chat 保持一致，保证两条链路召回一致
    hits = retriever.hybrid_search(rewritten, top_k=top_k, role_code=role_code)  # ② 混合召回
    if hits:  # 有召回才重排
        docs = [p.as_text() for p, _ in hits]  # 拍平成纯文本供重排模型打分
        order = rerank(query, docs)  # ③ 重排
        hits = [hits[i] for i in order if i < len(hits)]  # 按重排顺序重建 hits 并防越界
    if not LLM_API_KEY:  # 没有 Key 时无法走大模型流式
        yield fallback_answer(query, hits)  # yield 出兜底讲解答案，作为单条流式片段返回
        return  # 直接结束生成器，不再走后续大模型流式逻辑
    messages = build_messages(query, hits, history, role_code=role_code)  # 复用 build_messages 拼提示词，保证流式与非流式输入一致
    stream = _client().chat.completions.create(  # 调用 OpenAI 流式接口，返回的是一个可迭代 stream
        model=LLM_MODEL,  # 模型名
        messages=messages,  # 拼好的 messages
        temperature=0.4,  # 与非流式保持一致温度，确保行为可对照
        stream=True,  # 开启流式：服务端逐 token 返回，而不是等整段答案
    )
    for chunk in stream:  # 逐 chunk 迭代流式响应
        delta = chunk.choices[0].delta.content or ""  # 取当前 chunk 的增量文本，可能为 None 故用 or "" 兜底
        if delta:  # 只在有内容时 yield，过滤掉空 chunk（如首尾的 role/done 帧）
            yield delta  # 逐块产出，上层（FastAPI StreamingResponse）转发给前端实现打字机效果


# =====================================================================
# 知识点说明（RAG：生成环节 / Transformer 大模型）
# ---------------------------------------------------------------------
# 1. 大模型即 Transformer 解码器（Decoder-only，如 GPT/DeepSeek/Qwen）：
#    自回归地逐 token 生成；stream=True 时服务端逐 chunk 返回，即本文件
#    chat_stream 的流式输出（首字延迟低、体验好，也是性能优化手段）。
# 2. RAG 生成范式：把检索到的资料放进提示词（build_messages：
#    system 角色设定 + 历史短期记忆 + 【检索到的资料】+ 用户问题），
#    让模型"开卷考试"——答案锚定（grounding）到资料，减少幻觉。
# 3. 提示词工程：system_prompt 定义角色人设与"必须依据检索资料、
#    缺资料要明说"的规则（见 roles.py），是约束幻觉的第一道防线；
#    temperature=0.4 兼顾稳定与自然。
# 4. 兜底策略：未配置 LLM_API_KEY 时 fallback_answer 仅讲解检索结果，
#    体现"检索可独立工作、生成可插拔"的分层设计。
# =====================================================================
