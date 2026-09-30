# -*- coding: utf-8 -*-
"""
查询改写模块

职责：
    把用户的自然语言问题改写成**利于字面检索**的关键词序列，
    用于补足稠密与稀疏检索在「术语鸿沟」上的短板。

为什么需要（实测依据，见 spec 2.5）：
    样本 R09 的答案块根本不在 V2 的候选池内（稠密/稀疏 top-20 均无），
    因此**重排无法修复它**（重排只能在池内排序），**修正评测口径也无法修复它**
    （问题出在压根没召回）。改变送进检索的文本是唯一杠杆。

为什么默认关闭（见 ADR-025）：
    实测改写耗时 3.2~4.4 秒（deepseek-v4-flash 是推理模型，需先输出推理内容），
    叠加后会使端到端超出 N3 的 8 秒目标。因此作为**对照实验分支**保留，
    由 QUERY_REWRITE_ENABLED 控制，默认 false。

降级约定（见 ADR-024）：
    rewrite() **永不抛异常** —— 超时、LLM 故障、返回为空都返回空列表，
    调用方退回「只用原问题检索」。改写是加分项，绝不允许它中断问答。
"""

from __future__ import annotations

import queue
import re
import threading
from typing import Any, List, Optional, Tuple

from backend.config import settings
from backend.llm_client import get_chat_llm
from backend.logging_config import get_logger

logger = get_logger(__name__)


# 提示词：要求输出「关键词序列」而非完整句子 —— 字面检索对关键词更敏感
# {question} 是唯一的占位符，调用处用 .format(question=...) 填入用户原问题。
# 这里反复强调"40 字以内""不要解释、序号、开场白"，是因为模型的输出会被
# 直接送去检索：多出来的客套话会变成检索噪音，把真正的关键词稀释掉。
REWRITE_PROMPT = """你是检索查询改写助手。用户要在一份国家标准文档中查找答案。

请把用户问题改写为**更适合字面检索**的查询，要求：
1. 拆解出问题中的关键术语，并补上标准中可能出现的同义表述与相关术语
2. 用空格分隔关键词，不要写成完整句子
3. 只输出改写后的查询本身，不要任何解释、序号或开场白
4. 控制在 40 字以内

用户问题：{question}"""

# 这个正则要解决的问题：模型很爱把改写结果写成 "1. xxx" "2. yyy" 的列表，
# 而序号本身没有任何检索价值，留着只会干扰字面匹配，所以要在解析时剥掉。
# 匹配行首的序号与项目符号，如 "1. " "2) " "- " "• "
#
# ★ 序号后必须跟空白或行尾才剥离 ★
#   否则会把标准文本里的编号误剥：'3.5 版本 质量特性' 中的 "3." 被当成序号，
#   结果被削成 '5 版本 质量特性'（GB/T 文本里「3.5 术语」「4.1 软件质量模型」
#   这类编号极常见，而改写查询中出现它们并非不可能）。
#   注意 "4.1 软件质量模型" 同样不受影响：'\d+' 匹配到 "4" 后，
#   '\.' 吃掉 "."，紧随其后的 "1" 既不空白也非行尾，整体不匹配。
_LEADING_MARK = re.compile(r"^\s*(?:\d+[\.\)、](?=\s|$)|[-*•]\s+)")


def _parse_queries(raw: str, *, max_n: int) -> List[str]:
    """
    把模型输出解析为查询列表。

    模型可能返回多行（每行一条），也可能只返回一行。此处按行拆分、
    去除行首序号、去空、去重（忽略大小写），并截断到 max_n 条。
    """
    # 模型返回空串或纯空白时直接给空列表，调用方会随之退化成只用原问题检索。
    if not raw or not raw.strip():
        return []

    # seen 记录已经收下的查询（统一转小写），用于跨行去重：
    # 模型偶尔会把同一条查询换个大小写重复输出，而重复的查询等于白跑一次检索。
    seen = set()
    result: List[str] = []
    # 按行拆分：约定的输出格式就是一行一条查询。
    for line in raw.splitlines():
        # 先剥掉行首序号（"1. " 之类），再去掉首尾空白，得到干净的查询文本。
        text = _LEADING_MARK.sub("", line).strip()
        # 空行（模型常用它来分隔段落）直接跳过。
        if not text:
            continue
        # 用小写形式当去重键，这样 "Quality Model" 与 "quality model" 算同一条。
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        # 攒够 max_n 条就不再看后面的内容，避免把无关的大段文本也带进检索。
        if len(result) >= max_n:
            break
    return result


class QueryRewriter:
    """查询改写器（带超时与降级）"""

    def __init__(
        self,
        *,
        max_queries: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        # query_rewrite_max_queries / query_rewrite_timeout 均已在
        # backend/config.py（十四、查询改写）中定义，并可由 .env 覆盖，
        # 此处省略参数时即取配置值。getattr 的兜底默认值仅为防御性写法，
        # 与其他配置项保持一致。
        # max_queries：一次提问最多生成几条改写查询。
        # 每多一条就得多跑一次检索，直接抬高端到端耗时，所以默认只开 1 条。
        self.max_queries = max_queries or getattr(
            settings, "query_rewrite_max_queries", 1
        )
        # timeout：改写这一步的等待上限（秒）。超了就放弃改写、只用原问题 ——
        # 不能让一次慢的 LLM 调用把整个问答拖死。
        self.timeout = (
            timeout
            if timeout is not None
            else getattr(settings, "query_rewrite_timeout", 20)
        )

    def rewrite(self, question: str) -> List[str]:
        """
        生成改写查询。**任何失败都返回空列表，不抛异常。**

        返回：
            [改写查询, ...]，可能为空（表示降级为只用原问题）。
            返回值中**不包含原问题** —— 原问题由调用方自行加入。
        """
        # 空问题没什么可改写的，直接降级。
        if not question or not question.strip():
            return []

        # 真正调 LLM 的那一步。单独包成内部函数，是为了能把它丢进子线程执行，
        # 从而给它套上超时控制。
        def _call() -> str:
            return get_chat_llm().chat(
                [{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
                # 改写要的是稳定、可复现的关键词，不是文采：温度压到 0.1，
                # 让同一个问题每次尽量改写出相同结果，否则检索候选会随机漂移，
                # 对照实验和评测也就没法复现了。
                temperature=0.1,
                max_tokens=settings.llm_max_tokens,
            )

        # ------------------------------------------------------------------
        # 超时实现：daemon 线程 + 队列
        #
        # 为什么不用 ThreadPoolExecutor（含「显式 shutdown(wait=False)」的写法）：
        #   CPython 3.9+ 的 ThreadPoolExecutor 工作线程是**非 daemon** 线程，
        #   concurrent.futures.thread 在 atexit 注册的 _python_exit 会 join
        #   所有遗留线程。于是 shutdown(wait=False) 只是让**调用方**在超时点
        #   拿回控制权，**并不减少进程墙钟时间**：
        #     - 一次性脚本（任务本身 3s、timeout=1）实测：
        #         ThreadPoolExecutor 版：rewrite 1.03s 返回，进程退出 3.03s
        #           （退出时被 join，多等 2.0s；总墙钟 = 任务本身的耗时）；
        #         本实现（daemon）：rewrite 1.35s 返回，进程退出 1.35s（零额外等待）。
        #       即：显式 shutdown(wait=False) 只改善「调用方返回时刻」，
        #       一次性进程的总墙钟一秒都没省下。
        #     - 真实 LLM 场景下被弃线程跑到 109.17s 才结束（本模块修复轮实测）。
        #   长驻服务不受影响，但 Task 9 的对照实验脚本是**一次性进程**，
        #   一旦触发改写超时就会在结束处假死到 LLM 调用自然结束（最坏 100s+），
        #   污染对照实验的墙钟统计，并可能被误判为死锁。
        #   daemon 线程**不会**被 _python_exit join，进程可立即退出。
        #
        # 取舍（保留原判断）：线程无法强杀。超时后该 daemon 线程仍会把这次
        #   LLM 调用跑完（白耗一次 API、占一个连接），只是不再阻塞任何退出路径。
        #   代价可接受：改写是加分项，宁可浪费一次调用也不能拖住问答。
        # ------------------------------------------------------------------
        # 子线程与主线程之间的传话筒，只容得下一条（maxsize=1）。
        # 元素是 ("ok", 改写文本) 或 ("err", 异常对象)：
        # 用元组第一项区分"跑成功了"还是"报错了"，主线程据此决定要不要降级。
        result_q: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=1)

        # 子线程里执行的任务：把 LLM 结果（或异常对象）塞进队列后立刻结束。
        # 这里故意捕获 BaseException 而不只是 Exception —— 子线程里抛出的异常
        # 若没人接住，线程会悄无声息地死掉、队列永远收不到消息，
        # 主线程只能一路干等到超时。宁可把异常打包回传给主线程处理。
        def _worker() -> None:
            try:
                result_q.put(("ok", _call()))
            except BaseException as exc:  # noqa: BLE001 - 需覆盖一切，含 LLMError
                result_q.put(("err", exc))

        try:
            # daemon=True 是重点：进程退出时不会被 atexit 的 _python_exit join。
            #
            # ★ 线程的创建与启动必须留在 try 之内 ★
            #   线程/FD 耗尽时 start() 会抛 RuntimeError("can't start new thread")。
            #   旧写法（ThreadPoolExecutor）的线程是在 try 内的 pool.submit() 里
            #   启动的，该异常会被下面的 except Exception 接住并降级；start() 一旦
            #   挪到 try 之外，这层覆盖就丢了 —— 「永不抛异常」契约会在资源耗尽时
            #   被击穿（长驻服务里超时泄漏的 daemon 线程会累积，使其并非纯理论）。
            threading.Thread(target=_worker, daemon=True).start()

            # 在这里等改写结果，最多等 timeout 秒。
            # 等不到会抛 queue.Empty，由下面的分支接住并降级。
            kind, value = result_q.get(timeout=self.timeout)

            # 子线程报错了：把异常在主线程重抛一次，
            # 好让它落进下面统一的 except 里，走同一条降级路径。
            if kind == "err":
                # 工作线程的异常经队列回传后在此重抛，以复用下方同一处降级逻辑。
                # 只有 Exception 才重抛：BaseException 中不属于 Exception 的部分
                # （KeyboardInterrupt / SystemExit）在主线程重抛会击穿
                # 「永不抛异常」契约，故就地降级。
                if not isinstance(value, Exception):
                    logger.warning("查询改写被中断，降级为只用原问题：%r", value)
                    return []
                raise value

            raw: str = value
        # 超时：改写没能在时限内返回，放弃改写。
        # 注意那个被放弃的子线程其实还在跑，但它是 daemon 线程，不会拖住进程退出。
        except queue.Empty:
            logger.warning("查询改写超时（%ss），降级为只用原问题", self.timeout)
            return []
        # LLM 报错（限流、网络、鉴权失败……）：同样放弃改写。
        # 两个 except 返回的都是空列表，调用方拿到空列表时的表现完全一致 ——
        # 这就是本模块「永不抛异常」契约的落地方式。
        except Exception as exc:
            logger.warning("查询改写失败，降级为只用原问题：%s", exc)
            return []

        # 解析同样纳入保护：契约是「rewrite() 永不抛异常」，不能只靠
        # 「chat() 返回 str 且 max_queries 为真值 int」这类外部类型约定维持。
        # 把模型的原始文本拆成一条条可以直接拿去检索的查询。
        try:
            queries = _parse_queries(raw, max_n=self.max_queries)
        except Exception as exc:
            logger.warning("查询改写结果解析失败，降级为只用原问题：%s", exc)
            return []

        # 模型说了话却没给出可用的查询（例如只回了一句"好的"）。
        # 记一条日志便于排查，然后照常降级。
        if not queries:
            logger.warning("查询改写返回为空，降级为只用原问题")
        return queries


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

# 模块级单例容器：整个进程共用同一个改写器实例
_rewriter: Optional[QueryRewriter] = None


def get_rewriter() -> QueryRewriter:
    """获取查询改写器单例"""
    # 全局只建一个实例：构造时要读配置，建多个会让配置口径不一致；
    # 而且在线链路每次问答都会用到它，反复构造纯属浪费。
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter
