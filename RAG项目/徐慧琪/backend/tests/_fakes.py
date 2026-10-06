# 编排层测试共用的假体与共享常量。
#
# 为什么拆成独立模块：test_answer.py 的非空行数已顶到「单文件 ≤ 300」的硬闸门，
# 再加用例就只能靠压注释；拆出来还能让将来别的模块复用同一批假体，避免同一个
# 假 LLM 在多处各写一遍、改一处漏一处。本模块名不带 test_ 前缀，pytest 不收集它。
import json

from langchain_core.messages import AIMessage

from app.generation.answer import Answerer
from app.retrieval.pipeline import RetrievalResult

BLOCK_584 = {
    "chunk_id": "abc", "article_no": 584, "article_no_cn": "五百八十四",
    "paragraph_no": None, "item_no": None, "path": "第三编 合同",
    "chunk_type": "father", "parent_id": None,
    "text": "第五百八十四条 当事人一方不履行合同义务或者履行合同义务不符合约定，造成对方损失的，"
            "损失赔偿额应当相当于因违约所造成的损失。",
    "source": "exact", "rank_index": None, "rerank_score": None,
    "status": "现行有效", "law_id": "minfadian",
}

GOOD_QUOTE = BLOCK_584["text"][7:30]

# 改写过、非逐字的引用：所有"校验失败"用例共用这一份，避免同一串字面量抄四遍
BAD_QUOTE = "不履行合同义务造成损失的应当赔偿全部损失"


class _RunnableStub:
    """替身的基类：把 __call__ 接到 invoke 上。

    LCEL 的 `|` 只认 Runnable / callable / dict，只有 invoke 的普通对象会被拒（实测
    `Expected a Runnable, callable or dict`），而那会被编排层当成**服务故障**转成 error
    ——假 LLM 的所有断言就全测到别的东西上去了。
    """

    def __call__(self, messages):
        return self.invoke(messages)


class FakeLLM(_RunnableStub):
    """假 LLM：按预设脚本依次返回内容，并记录每次收到的 prompt。

    必须返回真正的 AIMessage：链尾是 StrOutputParser，它只认 str 与
    BaseMessage，喂一个"长得像消息"的普通对象会抛 ValueError，
    而那会被编排层当成解析失败、悄悄转成 abstain——测试就测不到该测的东西了。
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        if not self.replies:
            raise AssertionError("假 LLM 的脚本用完了，说明重生成次数超出预期")
        return AIMessage(content=self.replies.pop(0))


class Boom(_RunnableStub):
    """一调就炸的 LLM：模拟模型不可用（连接失败、超时）。"""

    def invoke(self, messages):
        raise RuntimeError("ollama 连接失败")


class Flaky(_RunnableStub):
    """首轮照脚本回，第二轮起抛故障：测"重试途中服务挂掉"。"""

    def __init__(self, first):
        self.first, self.calls = first, 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("ollama 连接中断")
        return AIMessage(content=self.first)


def _json_answer(quote=GOOD_QUOTE, status="ok", answer="应当赔偿。", citations=True,
                 article="第五百八十四条"):
    # citations=False 给追问/非法律问题用：它们本来就不该给结论、也就没有引用，
    # 校验关要照样放行——配上引用反而会让"放行"这条断言测了个寂寞。
    # article 可换：非 ok 状态的用例要造一条编造的条号
    cites = [{"law": "中华人民共和国民法典", "article": article,
              "paragraph": None, "item": None, "quote": quote}] if citations else []
    return json.dumps({"status": status, "answer": answer, "citations": cites,
                       "disclaimer": ""}, ensure_ascii=False)


def _answerer(llm, *, blocks=(BLOCK_584,), exact_nos=(584,), corpus=None,
              extras_fn=None):
    """构造 Answerer，把检索与语料查询都换成替身。

    `extras_fn` 原样透传（③b-1）：不传时 Answerer 拿到 None = 附加区块关闭，
    调用方的行为与 ③a 完全一致。
    """
    result = RetrievalResult(question="q", blocks=list(blocks), chunks=[],
                             exact_nos=list(exact_nos), recalled_blocks=list(blocks))

    def fake_retrieve(question):
        return result

    corpus = corpus if corpus is not None else type("C", (), {
        "article": lambda self, no: {"text": BLOCK_584["text"], "status": "现行有效",
                                     "law_id": "minfadian"},
        "para_index": lambda self, no: {"paragraphs": set(), "items": set()},
    })()
    return Answerer(retrieve_fn=fake_retrieve, corpus=corpus,
                    llm_factory=lambda side: llm, extras_fn=extras_fn)
