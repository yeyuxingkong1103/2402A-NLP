"""问答编排：检索 → 生成 → 校验 → 重生成 → 降级。

存在的理由：这段分支逻辑是整个 ③a 里唯一"必须按顺序发生、且失败路径比
成功路径还多"的地方。五种终态各有各的语义，混淆任何一种都会让下游误判：

    ok             有依据、引用全过
    need_more_info 信息不足，模型在追问（不是失败）
    out_of_scope   拒答或非法律问题（不是失败）
    abstain        校验/解析两次都没过 → 按"未找到相关依据"降级
    error          LLM 不可用 → **绝不能与 abstain 混为一谈**，
                   否则 ④期会把服务故障答成"查无此条"

免责声明的出口收到 `_side_disclaimer` 一处：成功路径由 boundaries 注入，
但拒答/降级/故障三条出口走不到那里，各写各的话迟早漏掉一条。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from langchain_core.exceptions import OutputParserException

from app.generation import boundaries
from app.generation.chain import build_chain
from app.generation.cite_verify import verify_answer
from app.generation.corpus import LawCorpus
from app.generation.llm_router import get_llm
from app.generation.profiles import SIDE_INTERNAL, SIDE_PUBLIC, build_payload
from app.generation.schema import Answer, Citation, answer_parser
from app.recommend.price_intent import is_price_intent
from app.retrieval.pipeline import retrieve

# 重生成时附在载荷末尾的提示。必须把**具体失败原因**带给模型，
# 只说"再试一次"它会照原样再犯
RETRY_HINT = "【注意】上一次的回答未通过引用校验，原因如下："

ABSTAIN_REPLY = "未找到相关依据。"

# 附加区块故障的信号出口。区块挂掉时答案照给（区块是装饰），但**不许无声**：
# 不写日志的话，连续故障只表现为「区块总是不出现」，与「本来就没有区块」不可区分
logger = logging.getLogger(__name__)

# 模型漏进 content 的特殊标记，形如 <|OPENAI|> / <|TOOL_CALL|>（Task 7 真连实测：
# DeepSeek 在 json 模式下把它们顶掉一个空格塞进 JSON 里，json.loads 当场失败）。
# 不洗掉就会走"解析失败→重生成→降级"，用户拿到"未找到相关依据"而真实原因
# 在任何日志里都看不见。只删这种形状的标记：它们零信息量，且正常法律文本
# 不会出现 `<|...|>`，故不会误伤正文
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|<>]+\|>")


def _side_disclaimer(side: str) -> str:
    """按侧给免责；未知侧当场抛错，绝不放行成空串。

    拒答、降级、故障三条出口都发生在 get_llm 与 apply_boundaries 之前/之外，
    指望不上它们兜底（拒答尤其：它在 get_llm 之前就返回了）。公众侧漏免责
    是合规问题，最危险的失败方向是 fail-open，所以这里与 profiles.system_prompt、
    llm_router.get_llm、boundaries.apply_boundaries 同向——宁可炸掉。
    判定与 boundaries 同源（引用同一组常量），test_answer 里有一条断言把
    两份规则钉在一起：任一边改了而另一边没改，那条断言必红。
    """
    if side == SIDE_PUBLIC:
        return boundaries.PUBLIC_DISCLAIMER
    if side == SIDE_INTERNAL:
        return ""
    raise ValueError(f"未知的侧别：{side!r}")


@dataclass
class QAResult:
    """编排层的返回。status 的取值比模型契约多两个：abstain 与 error。

    retrieval 一起带出来，是为了让评测脚本不必为了算 Recall 再检索一遍——
    重跑一次检索要多花一次精排与编码，100 题上就是白烧几分钟。
    """
    status: str
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    disclaimer: str = ""
    sources: list[dict] = field(default_factory=list)
    attempts: int = 0
    failures: list[str] = field(default_factory=list)
    retrieval: "RetrievalResult | None" = None
    # ③b-1：公众侧附加区块（案由/领域/律师/费用）。放末尾且带默认值，既有的
    # 构造调用一个都不用改；**律师侧恒为 None**，公众侧只有 out_of_scope 里
    # 不问价的那些出口（拒答与非法律提问）为 None —— 判据收在 _extras_for
    # 一处，别在这里复述
    extras: dict | None = None


class Answerer:
    """把检索、生成、校验、边界串起来。

    依赖全部可注入：生产用真连接与真模型，测试用替身。
    检索用函数注入（而不是整条 pipeline）是为了让测试不必造 MySQL/Milvus。
    """

    def __init__(self, *, conn=None, client=None, encoder=None, reranker=None,
                 llm_factory=get_llm, retrieve_fn=None, corpus=None, extras_fn=None):
        self._corpus = corpus if corpus is not None else LawCorpus(conn, client)
        self._llm_factory = llm_factory
        # ③b-1 的注入点：默认 None = 关闭。③a 的调用方（含 CLI）不传就完全
        # 拿不到附加区块，既有的行为与断言一字不变
        self._extras_fn = extras_fn
        self._retrieve_fn = retrieve_fn or (lambda question: retrieve(
            question, conn=conn, client=client, encoder=encoder, reranker=reranker))

    def _generate(self, chain, payload: str) -> Answer:
        """跑一次链并解析；异常抛给调用方分类处理。

        链由调用方构造（只构造一次）：构造里含 get_format_instructions() 与
        模板编译，放进重试循环等于每次尝试都白做一遍。
        """
        raw = chain.invoke({"payload": payload})
        # 链尾已由 StrOutputParser 取出文本，这里做结构化解析。
        # 不用链上的 PydanticOutputParser 是因为解析失败要能与校验失败
        # 走同一条重生成通路，放在编排里更直白
        return answer_parser().parse(_SPECIAL_TOKEN_RE.sub("", raw))

    def _verify(self, ans: Answer) -> list[str]:
        return verify_answer(ans, article_of=self._corpus.article,
                             para_index_of=self._corpus.para_index)

    def _extras_for(self, question: str, side: str, status: str) -> dict | None:
        """附加区块的唯一组装出口：按侧别、回答状态与问价意图决定要不要组装（③b-1）。

        判据是「**公众侧 +（状态非 out_of_scope 或 问句命中问价意图）**」，
        三部分是三次裁决叠出来的，每一次都写清理由：

        - 侧别：律师侧永不经过这条路径（数据不出域，红线）。
        - 状态（第一次裁决）：枚举里除 out_of_scope 外的四个状态 ——
          need_more_info / abstain / error，以及 ok —— 都是**用户问法条问不出、
          但可能正在问价**的情形；问「律师费大概多少」时法条本来就答不出，
          那正是费用区块最有用的时刻，把它一起掐掉等于用户什么都没拿到。
        - 问价意图（第二次裁决，本批）：上一版的「状态非 out_of_scope」有个
          **实测推翻的前提** —— 它假定 out_of_scope 意味着「与法律无关」，而模型
          把「律师费大概多少」判成了 out_of_scope（原话「这属于律师服务收费的
          市场行情问题，不是由上面这些法条来规定的内容」），于是裁决举的那个例子
          改后仍然什么都不给（2/2 复现）。判决：**问价的话不看状态，照给**。
          为什么是词表而不是再放宽状态，见 price_intent 模块 docstring。
          「今天天气怎么样」不含问价词 → 仍不给，这是这条判据必须保住的方向。
          代价**已知并接受**：放行的问答多一次费用检索（便宜）与一次费用生成调用
          （DeepSeek，不便宜）；误命中的代价同此，不会给出错数（费用数字仍要过
          fees 的数值回查）。

        收成一个出口的理由与顶部 _side_disclaimer 同源：本类有八条返回路径，
        判据（侧别 / 状态 / extras_fn 是否为 None / 异常吞掉）各出口各写一份的话，
        迟早有一处漏写，而那种分叉在测试里只表现为「某个出口恰好没被覆盖」。
        异常吞掉也在这里：区块是装饰、答案不是（这条路径接真 fee_corpus 检索与
        DeepSeek，最容易运行期挂），挂了只丢区块并留一条 warning —— 无声地丢会让
        「连续故障」与「本来就没有区块」不可区分。
        """
        # 短路次序即成本次序：律师侧不构造 DeepSeek 客户端、out_of_scope 且不问价
        # 时一个依赖都不碰。`status == ... and not ...` 就是判据的「或 问价意图」
        # 那半边：命中问价时 status 不再参与判断。词表检查排在 extras_fn 之后，
        # 让「③a 调用方不传 extras_fn」这条路径连一次匹配都不跑（纯省，非护栏）
        if (self._extras_fn is None or side != SIDE_PUBLIC
                or (status == "out_of_scope" and not is_price_intent(question))):
            return None
        try:
            return self._extras_fn(question)
        except Exception as exc:  # noqa: BLE001 —— 区块故障不改变答案
            logger.warning("附加区块组装失败，本答已降级为无区块："
                           "side=%s status=%s 异常=%r", side, status, exc)
            return None

    def answer(self, question: str, side: str = "internal") -> QAResult:
        """完整跑一条问答。"""
        # 拒答在检索之前：一次检索与一次生成都省下（设计文档 4.4）。
        # 下面 extras= 这一行**从第四批起是真正的护栏**（上一版它是纯形状一致性）：
        # 判据多了「问价意图」这半边后，拒答出口也能给区块了 ——「帮我写份起诉状，
        # 律师费大概多少」命中词表就照给，删掉这一行会让那条用例当场变红。留在这里
        # 是为了让八条返回路径都过同一个判据：漏写某个出口的表现只是那条路径没有
        # 区块，不留任何其他痕迹（上一批正是在这条出口上靠变异才发现的）
        if boundaries.is_refusal(question):
            return QAResult(status="out_of_scope",
                            answer=boundaries.REFUSAL_REPLY,
                            disclaimer=_side_disclaimer(side),
                            extras=self._extras_for(question, side, "out_of_scope"))
        try:
            llm = self._llm_factory(side)
        except Exception as exc:  # 密钥缺失等配置问题：是 error，不是"没依据"
            return QAResult(status="error", answer=f"生成服务不可用：{exc}",
                            disclaimer=_side_disclaimer(side),
                            extras=self._extras_for(question, side, "error"))
        try:
            result = self._retrieve_fn(question)
        except Exception as exc:
            return QAResult(status="error", answer=f"检索服务不可用：{exc}",
                            disclaimer=_side_disclaimer(side),
                            extras=self._extras_for(question, side, "error"))
        if not result.blocks:
            # 无召回不调模型：既省一次调用，也断了它自由发挥的念想
            return QAResult(status="abstain", answer=ABSTAIN_REPLY, attempts=0,
                            disclaimer=_side_disclaimer(side), retrieval=result,
                            extras=self._extras_for(question, side, "abstain"))
        try:
            # 拼载荷与建链都是"准备请求"，放在重试循环之外：载荷每轮都一样，
            # 而链的构造含 get_format_instructions() 与模板编译，重试时白做
            payload = build_payload(question, result.blocks)
            chain = build_chain(llm, side)
        except Exception as exc:
            # 这一步失败是**代码/数据故障**（模板坏了、召回块缺字段），
            # 不是模型输出问题：必须 error，绝不能落进下面的"解析失败"通路
            # 去重试一次再降级——那会让脚本坏掉时每个问题都答"未找到相关依据"
            return QAResult(status="error", answer=f"生成服务不可用：{exc}",
                            sources=result.blocks, disclaimer=_side_disclaimer(side),
                            retrieval=result,
                            extras=self._extras_for(question, side, "error"))
        failures: list[str] = []
        for attempt in range(1, 3):
            try:
                ans = self._generate(chain, payload)
                failures = self._verify(ans)
            except OutputParserException as exc:
                # 只有**模型输出解析失败**进重生成通路：一次 JSON 不合规就废掉
                # 整条答案太脆。这里刻意不收 ValueError——OutputParserException
                # 本身就是 ValueError 的子类（收割它不需要父类），而父类会把
                # 校验依赖抛的 ValueError（如 corpus 里 int() 撞上畸形 Milvus 值）
                # 一并吞进来标成"解析失败"→重试→降级，于是"校验库挂了"被答成
                # "未找到相关依据"，正是设计禁止的故障/无依据混淆
                failures = [f"输出解析失败：{exc}"]
                ans = None
            except Exception as exc:
                # 模型连不上/超时/校验依赖的库挂了：这是**服务故障**，
                # 不能降级成"未找到相关依据"——那会把故障答成"查无此条"。
                # failures 一并带出：首轮失败的线索是排查故障时唯一的现场
                return QAResult(status="error", answer=f"生成服务不可用：{exc}",
                                sources=result.blocks, attempts=attempt,
                                failures=failures,
                                disclaimer=_side_disclaimer(side), retrieval=result,
                                extras=self._extras_for(question, side, "error"))
            if not failures and ans is not None:
                bounded = boundaries.apply_boundaries(ans, side)
                # AC-6「输出中无未校验引用」：非 ok 的答案（追问/拒答）不过四关
                # （verify_answer 对它们一律放行，因为那本就不该给结论），
                # 所以引用必须在这里清空——不然模型编造的条号会原样落到用户
                # 手里（真库实测：need_more_info + 第9999条 曾完整带出）。
                # 清空放在出口而不是 verify_answer：那边的"放行"是重生成通路
                # 的语义，改了会让非 ok 答案白白重试两轮再降级
                citations = bounded.citations if bounded.status == "ok" else []
                # ③b-1：附加区块按 _extras_for 的判据组装（原先只判 `status == "ok"`，
                # 用户 2026-09-29 裁决解耦）。放到这里而不是更早：模型自报的
                # need_more_info / out_of_scope 与 ok 走的是同一条出口，状态只有到
                # 这一步才定下来（判据与理由见 _extras_for）
                extras = self._extras_for(question, side, bounded.status)
                return QAResult(status=bounded.status, answer=bounded.answer,
                                citations=citations,
                                disclaimer=bounded.disclaimer,
                                sources=result.blocks, attempts=attempt,
                                retrieval=result, extras=extras)
            if attempt == 1:
                # 把失败原因回灌给模型，只重试一次
                payload = payload + f"\n\n{RETRY_HINT}\n" + "\n".join(failures)
            else:
                return QAResult(status="abstain", answer=ABSTAIN_REPLY,
                                sources=result.blocks, attempts=attempt,
                                failures=failures,
                                disclaimer=_side_disclaimer(side), retrieval=result,
                                extras=self._extras_for(question, side, "abstain"))
        # 循环必然在 attempt==2 时返回；这行只为让所有分支都有返回值
        raise AssertionError("重生成循环走到了不可达分支")
