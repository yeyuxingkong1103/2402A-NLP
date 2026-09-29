# -*- coding: utf-8 -*-
"""RAG 主链（在线阶段）：急症拦截 → 意图分流 → 查询改写 → 混合检索 → 重排 → 四种模式应答。即"问题向量化→语义检索Top-K→拼提示词→大模型生成答案"的完整实现。"""
import logging  # 日志：记录各环节耗时与异常，性能分析的第一步

from dataclasses import dataclass, field  # dataclass 装饰器快速定义数据类；field 给可变字段（列表）安全的默认值

from langchain_core.messages import HumanMessage, SystemMessage  # 消息封装：SystemMessage=系统指令（人设），HumanMessage=用户输入
from langchain_openai import ChatOpenAI  # 仅用 LangChain 做 LLM 对接与消息封装，RAG 主链自研编排

from src import citation, prompts, query_tools, reranker, safety  # 引用高亮/提示词/查询工具/重排/安全层——每步独立封装，改一处不影响全局
from src.config import settings  # 统一配置中心：模型名、阈值、top_k 等都从 .env 读
from src.intent import detect_intent, is_multi_route, lab_report_query  # 意图识别：决定走哪条应答路径（知识/个人/症状/化验单等）
from src.retrieval import multi_search  # 混合检索入口：BM25 + Milvus 向量双路召回
from src.role import RoleManager  # 角色管理：全科/西医/心理的人设提示词与检索增强词

logger = logging.getLogger(__name__)  # 本模块日志器，输出到控制台 + logs/app.log（按天滚动）
EMERGENCY, CLARIFY, NO_CONTENT, ANSWER, CAPABILITY = "emergency", "clarify", "no_content", "answer", "capability"  # 五种应答模式：急症/追问澄清/无内容兜底/正常回答/能力自述

_CAPABILITY_HINTS = ("你能做什么", "你是谁", "你可以干什么", "你会什么", "你的功能", "你能干什么", "你可以做什么")  # 能力类问题关键词：这类问题检索相似度必然极低，命中即绕过 RAG 直接答
_CAPABILITY_REPLY = ("我是医小助，基于知识库的 AI 健康顾问。可以：\n"  # 预设文案：固定内容零延迟、零幻觉风险
                     "📗 回答高血压防治/食养问题　📷 识别药盒、化验单图片　"
                     "💡 健康科普（不替代医生诊断）\n\n直接提问，或在左侧上传图片。")


@dataclass  # 装饰器自动生成 __init__/__repr__，省去手写样板代码
class RAGResponse:
    """检索 + 决策结果，UI 据此渲染不同形态。"""

    mode: str  # 应答模式：emergency/clarify/no_content/answer/capability 五选一
    query: str = ""  # 改写后的检索词（可能与用户原话不同）
    message: str = ""  # 系统话术（急症提示/澄清问句/兜底说明），非 LLM 生成
    intent: str = ""  # 识别出的意图类别，供 UI 打标签（如"化验单解读模式"）
    options: list = field(default_factory=list)  # 澄清模式的候选按钮选项
    citations: list = field(default_factory=list)  # 引用片段（含来源+页码+相关度），答案可溯源
    prompt: str = ""  # 组装好的完整提示词，answer 模式直接送 LLM
    trace: str = ""  # 检索追踪信息（命中文档/分数），调试用


class RAGChain:
    """完整的 RAG 对话链。"""

    def __init__(self, role_manager: RoleManager):
        self.role_manager = role_manager  # 注入角色管理器，生成时取人设提示词
        self.llm = ChatOpenAI(  # 在线 API 方式调 DeepSeek：不用管部署和算力，上手快；涉密数据才考虑本地部署（Ollama/vLLM）
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.3,  # 低温：事实型问答要稳不要"创意"，减少模型发挥防幻觉
            timeout=30,        # 单次 API 超时 30s：防止网络抖动把请求卡到 90s+
            max_retries=0,    # 禁止自动重试：超时就是超时，快速失败走兜底，别让用户干等
        )

    def prepare(self, role_id, query, embedder, bm25, short_memory=None,
                long_memory=None, user_id="default", clarified=False, history=None):
        """检索决策全链路。异常统一捕获返回 NO_CONTENT，绝不抛到 UI 层。"""
        try:
            return self._prepare_inner(  # 真正的链路在 _prepare_inner，本层只负责兜底
                role_id, query, embedder, bm25, short_memory,
                long_memory, user_id, clarified, history)
        except Exception as exc:
            logger.exception("检索过程出错")  # 完整堆栈进日志，便于事后定位
            return RAGResponse(  # 任何异常都降级为兜底回答，用户永远看到友好提示而非报错堆栈
                mode=NO_CONTENT, query=query,
                message=f"⚠️ 检索过程出错：{exc}。建议重新提问。")

    def _prepare_inner(self, role_id, query, embedder, bm25, short_memory=None,
                       long_memory=None, user_id="default", clarified=False, history=None):
        if safety.emergency_hits(query):  # 安全层最高优先级：急症关键词（含"胸通"等错别字容错，宁误报不漏报）
            return RAGResponse(mode=EMERGENCY, message=safety.EMERGENCY_REPLY)  # 绕过大模型直接返回固定文案：急救提示不需要生成，只需要准确和快

        if any(h in query for h in _CAPABILITY_HINTS):  # 能力类问题：知识库里没有"系统介绍"，检索必低分
            return RAGResponse(mode=CAPABILITY, message=_CAPABILITY_REPLY, intent="knowledge")  # 不走 RAG 直接返回预设文案，避免误触发兜底 LLM

        role = self.role_manager.get(role_id) or {}  # 取当前角色配置（人设/检索增强词/禁用术语）
        if history is None:
            history = short_memory.get_history(user_id) if short_memory else []  # 短期记忆：Redis List 存最近 20 轮对话，key-value 存取 O(1)，多轮对话的 cache

        # 输入断句/缺宾语时先追问，避免硬答
        if query_tools.needs_clarify(query) and not clarified:  # clarified=True 表示用户已选过澄清选项，不再二次追问
            return RAGResponse(
                mode=CLARIFY, query=query,
                message="您这句话好像还没说完，我先确认一下您想说的是哪种情况？",
                options=query_tools.incomplete_options(query),  # 生成可点击的补全选项，降低用户输入成本
            )

        # 多轮改写：把省略句补成可独立检索的问句
        search_query = query_tools.rewrite_query(self.llm, query, history,
                                                 force_join=clarified)  # Query 改写：结合对话历史把"那这个呢"这类省略句补全为完整问句再检索
        is_lab, search_query = lab_report_query(query, search_query)  # 化验单意图：检测 OCR 关键词并提取异常项拼成检索词（如"尿蛋白阳性 临床意义"）

        question_type = "lab_report" if is_lab else detect_intent(search_query)  # 意图分流：知识科普/个人病情/情绪/症状，各有专属提示词
        top_k = settings.list_top_k if is_multi_route(search_query) else settings.answer_top_k  # 清单型问题（"不能吃什么"）拆多路召回取更多条，保证找全

        candidates = multi_search(role, search_query, bm25, embedder)  # 混合检索：BM25 字面精确匹配（药名/术语）+ 向量语义匹配（概念相关但未必共享关键词），两路互补提高召回率与准确率
        # 情绪/症状类口语问句缺疾病主体时补锚点再打分，避免误判为无关问题
        rank_query = query_tools.anchor_query(search_query)  # 如"心里堵得慌"补"高血压"锚点，让重排打分有疾病上下文
        ranked = reranker.rerank(rank_query, candidates, top_k)  # 重排序：bge-reranker-v2-m3 交叉编码器，输入「查询+候选文档对」逐对精排打分——先粗排召回、后精排取 Top-K

        top = ranked[0]["score"] if ranked else 0.0  # 最高相关度分数，作为"知识库有没有货"的判据
        # 症状类问题（头晕/头痛/耳鸣等）知识库无鉴别诊断内容，降阈值避免误判为无关
        is_symptom = any(w in search_query for w in ("头晕", "头痛", "心慌", "乏力", "耳鸣"))
        limit = 0.21 if is_symptom else reranker.threshold()  # 相似度阈值过滤：低于阈值宁可走兜底也不硬答，防幻觉的第一道防线
        trace = citation.trace(rank_query, ranked, limit, question_type)  # 记录检索追踪（命中条数/分数），日志可复盘

        # 低相关度不再反问，直接 LLM 兜底（化验单除外，已在上面处理）
        if top < limit and not is_lab:  # 检索不到是 RAG 最常见的失败点，本项目策略：阈值过滤 + LLM 通用知识兜底并显式标注
            return RAGResponse(mode=NO_CONTENT, query=query, trace=trace,
                               intent=question_type,
                               message="⚠️ 知识库中暂无相关指南内容，以下是基于通用知识的回答。")  # 诚实标注"非知识库内容"，防误导

        recalls = long_memory.recall(user_id, search_query) if long_memory else []  # 长期记忆：Milvus 向量化存的用户病史/用药，语义检索召回相关记忆拼入上下文
        # P0/P1: 只把 score≥0.3 的片段传给 LLM 和 UI，低分项不进 prompt 避免乱引用
        # 化验单无 KB 命中时不引用低分片段，靠 OCR 内容 + LLM 医学知识解读
        cited = [] if (is_lab and top < limit) else ([r for r in ranked if r["score"] >= limit] or ranked[:1])  # 阈值过滤引用；全低于阈值时保底留 1 条（有命中场景）
        kb_text = "\n".join(f"【参考资料{i}】{c['text']}" for i, c in enumerate(cited, 1))  # 把片段拼成带编号的参考文本（上下文构建第一步：准备参考内容）
        prompt = prompts.answer_prompt(  # 上下文构建：按提示词模板组装，模板至少替换两个变量——参考内容、用户问题，外加系统指令
            role_prompt=self.role_manager.build_system_prompt(role_id),  # 系统指令：角色人设 + "仅根据参考资料回答，没有就说没有"的防幻觉约束
            kb_text=kb_text,  # 变量1：参考内容
            history=prompts.format_history(history),  # 短期对话历史（多轮上下文连贯）
            summary=prompts.format_summary(recalls),  # 长期记忆摘要（用户既往病史）
            query=query,  # 变量2：用户原始问题（不是改写后的检索词，保证回答贴合用户本意）
            intent=question_type,  # 意图决定模板分支（知识类禁止反问/个人类先结论后追问）
        )
        return RAGResponse(mode=ANSWER, query=search_query, prompt=prompt,
                           citations=cited, trace=trace, intent=question_type)

    def stream_answer(self, role_id: str, prompt: str):
        """流式生成回答。异常转为一条错误文本 yield 出来，不抛到 UI。"""
        try:
            messages = [
                SystemMessage(content=self.role_manager.build_system_prompt(role_id)),  # 系统指令单独一条：人设+防幻觉规则
                HumanMessage(content=prompt),  # 用户消息：组装好的"参考资料+历史+问题"
            ]
            for chunk in self.llm.stream(messages, timeout=30):  # 流式输出：逐 token 返回，边生成边显示，降低首字延迟（TTFT）
                if not chunk.content:  # 跳过空 chunk（心跳/元数据帧）
                    continue
                yield chunk.content  # 生成器逐段吐给 UI
        except Exception as exc:
            logger.exception("流式生成失败")
            yield f"\n\n⚠️ 生成失败：{exc}"  # 异常也 yield 成文本，UI 流式循环不用额外 try

    def general_tips(self, role_id: str, query: str, intent: str = "") -> str:
        """无命中兜底：LLM 基于 OCR/通用知识回答，空响应给最低保障。"""
        try:
            content = prompts.no_content_prompt(self.role_manager.get_role_name(role_id), query, intent)  # 兜底专用模板：按意图分 4 路（OCR 图片/情绪/症状/通用）
            result = self.llm.invoke([HumanMessage(content=content)], timeout=15).content  # 兜底非流式一次性返回，超时收窄到 15s（兜底要快）
            return result.strip() if result and result.strip() else "知识库暂无相关内容，建议咨询医生或药师。"  # LLM 空响应的最低保障：绝不让用户看到空白
        except Exception as exc:
            logger.exception("兜底建议生成失败")
            return f"⚠️ 生成失败：{exc}"
