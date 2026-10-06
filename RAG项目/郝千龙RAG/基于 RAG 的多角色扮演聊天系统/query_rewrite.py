# -*- coding: utf-8 -*-  # 声明源文件编码为 UTF-8，保证文件内中文字符串与注释正常解析
"""【查询改写 · query_rewrite.py】检索前预处理：有大模型时把口语化问题扩写成更适合检索的关键词，无 LLM 时原样返回。"""  # 模块文档字符串：中文名 + 文件名 + 一句话作用
from __future__ import annotations  # 启用 PEP 563 延迟注解求值，兼容低版本 Python 的类型注解写法

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL  # 从全局配置读取：API Key、服务地址、模型名
from logger import log  # 导入项目统一日志器，用于记录改写结果与失败告警


def rewrite_query(query: str) -> str:
    """把口语化问题改写成适合检索的短查询；未配置 KEY 或失败时原样返回。"""  # 文档字符串：说明降级策略
    text = query.strip()  # 去除首尾空白，避免空格干扰后续判断与请求体
    if not text or not LLM_API_KEY:  # 空查询或未配置 API Key 时无法/无需调用 LLM
        return text  # 降级：不调大模型，直接用原查询，保证主流程可用
    try:
        from openai import OpenAI  # 延迟导入：仅在有 Key 时才加载 openai 依赖，避免无用导入报错

        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL or None)  # 构造客户端；base_url 为空则用 None 走官方默认地址（兼容自建/兼容服务）
        resp = client.chat.completions.create(  # 调用 Chat Completions 接口生成改写
            model=LLM_MODEL,  # 指定使用的模型名（由配置决定）
            temperature=0,  # 温度 0：改写结果稳定可复现，避免相同输入产生不同查询影响召回稳定性
            messages=[  # 构造对话消息列表
                {
                    "role": "system",  # 系统指令角色：约束模型行为
                    "content": "把用户问题改写成更适合检索中英例句库的短查询，保留关键词，不要解释。只输出改写后的查询。",  # 提示模型只输出改写后的查询、保留关键词、不要解释
                },
                {"role": "user", "content": text},  # 用户消息：把待改写的原始查询发给模型
            ],
        )
        rewritten = (resp.choices[0].message.content or text).strip()  # 取首个回复内容；为空时回退到原查询并去空白
        log.info("query rewrite: %s -> %s", text, rewritten)  # 记录改写前后对比，便于观察效果与排查
        return rewritten or text  # 改写结果非空则返回，否则再次回退原查询，双重保险防止返回空字符串
    except Exception as exc:  # 捕获网络异常、鉴权失败、限流等错误
        log.warning("query rewrite failed: %s", exc)  # 记录告警，便于定位 LLM 调用问题
        return text  # 降级原样返回，保证检索主流程不中断


# =====================================================================
# 知识点说明（RAG：Query 优化）
# ---------------------------------------------------------------------
# 1. Query 改写/扩写：用户口语化提问常与知识库措辞不一致（词汇鸿沟），
#    用大模型把问题改写成更贴近知识库的检索词，可显著提升召回率。
#    相关技术还有：Multi-Query（一次生成多个变体查询）、HyDE
#    （先让 LLM 生成假想答案再拿去检索）、Step-Back Prompting
#    （先抽象上一步再检索）。
# 2. 权衡：改写本身要调一次 LLM，增加延迟与成本；temperature=0 保证
#    改写稳定；LLM_API_KEY 未配置时原样返回，保证可降级。
# 3. 在 RAG 中的位置：检索前的"查询理解"环节，与检索后的重排序
#    （rerank.py）一前一后，是 RAG 提效最明显的两个轻量优化点。
# =====================================================================
