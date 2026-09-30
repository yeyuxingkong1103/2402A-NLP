"""附加区块的真依赖装配：费用检索 + 生成 + 留痕，外加给 Answerer 的 extras_fn。

存在的理由（HTTP 接口设计 §三 定调第 3 条）：这段装配原先只活在 tools/ask.py 的
main() 旁边，而 HTTP 服务的公众侧问答要的是**同一套**东西 —— 两处各装一遍正是
「CLI 能跑、服务不能」的来源。故搬进生产包；它落在 app/recommend/ 而不是
core/factory.py，是因为这一整段（提示词、模型输出解析、检索回退、留痕建表）都是
费用域的判断，与 attach / fees / fee_log 同处一层，工厂那边只留「建连接、加载模型」
这类与域无关的装配。
"""
from __future__ import annotations

import json

# 费用抽取的提示词。三条约束都是**实测**出来的，不是修辞：
# ①回查对 unit 只做「片段里出现过就放行」的存在性检查（fees._unit_has_source），
#   而 unit 承载量级，混进「件」这类计数单位就会渲染成「3000 ~ 30000元每件」、
#   量纲当场变形。回查拦不住它，只能靠提示词把 unit 限定到货币/费率；
# ②这类计数单位不是丢掉而是改由 charge_basis 承接（用户 2026-09-29 裁决）：
#   计价基础丢了，用户会把「1000元/有效工作小时」读成一次性收费（真跑实证 N10）；
# ③回查是必要不充分条件（fees.py 的模块 docstring）：模型把片段里的数字用错
#   档位它挡不住，所以这里明写「逐字取数、不得推算」。
FEE_PROMPT = """你是律师服务收费口径抽取器。只依据给定的【收费口径片段】输出 JSON。

硬性要求：
1. 数字只能**逐字**取自片段，严禁推算、换算、补全或凭常识给数。
2. 输出的 JSON 形如 {"low": 下界, "high": 上界, "unit": 单位, "charge_basis": 计价基础}。
   low / high 取片段里与案由最相关的一组收费区间的下界与上界：
   片段写「8%~12%」就输出 8 和 12，写「20000--100000元」就输出 20000 和 100000。
3. unit 只能是货币或费率单位（元 / 万元 / %）；片段里没有单位就给 null。
4. charge_basis 是计价基础，回答「这笔钱按什么算」：片段写「1000元—8000元/
   有效工作小时」就输出「有效工作小时」，写「每件收费 50 元」就输出「件」，
   **逐字照抄片段里的那个词**；片段没写计价基础（如纯按标的额比例收费）就给 null。
5. 片段里找不到可用区间时，low 与 high 都给 null。
6. 只输出 JSON，不要任何解释或额外文字。
"""


def _parse_json(raw: str) -> dict:
    """从模型输出里取出 JSON 对象；取不出就抛（异常由 attach 记成「不可用」）。

    先整体解析，失败再退到最外层花括号：Task 7 真连实测 DeepSeek 会把
    <|OPENAI|> 这类标记顶进正文，此时整体解析必失败，而花括号内仍是好 JSON。
    刻意不在这里校验字段名与取值：low/high/unit 的合法性与回查都归
    fees.estimate，这里多校一次就有了两份口径，改一处漏一处。
    """
    text = raw.strip()
    try:
        data = json.loads(text)
    except ValueError:
        data = json.loads(text[text.find("{"):text.rfind("}") + 1])
    if not isinstance(data, dict):
        # 数组/标量会让下游的 .get() 直接 AttributeError，报错面目模糊；
        # 这里当场说清「返回的不是对象」
        raise ValueError(f"模型返回的不是 JSON 对象：{text[:80]!r}")
    return data


def generate_fee(llm, snippet: str, cause: str) -> dict:
    """从命中片段抽区间（fees.estimate 要的 generate_fn 契约）。

    返回值原样交给 estimate：low/high 是字符串数字也无妨（_as_number 会折算），
    unit 给什么由提示词约束、由回查判定。这里只补两个模型报不出、而 AC-21
    留痕要用的字段：model 与 request_id。
    """
    payload = f"【收费口径片段】\n{snippet}\n\n【案由】{cause or '未识别'}"
    reply = llm.invoke([("system", FEE_PROMPT), ("human", payload)])
    data = _parse_json(reply.content)
    data["model"] = getattr(llm, "model_name", None)
    # OpenAI 兼容接口把响应 id 放在元数据里；取不到就留空（留痕列可空），
    # 不编一个 —— request_id 是给人回查用的，编了就失去意义
    data["request_id"] = (reply.response_metadata or {}).get("id")
    return data


def build_extras_fn(conn, client, encoder, reranker=None):
    """装配附加区块的真依赖，返回 Answerer 的 extras_fn。

    LLM 客户端在这里构造一次（不是每次问答现建）：与 chain.build_chain 的
    「链只构造一次」同口径，省掉每问一次的客户端初始化。
    `reranker` 是可注入的精排模型（工厂已为法条侧加载过同一个对象）：技术方案
    6.4 要求费用检索也走「混合检索 + 精排」，不接的话 top1 由 RRF 位置融合决定，
    正是终审点名的「取错片段」上游。复用而不重新加载 —— 2.2GB 的模型加载两次
    是纯粹的浪费，且两个对象在同一块显存里并存没有意义。默认 None 时退回纯
    RRF（离线装配与非公众侧的退化路径，不是正常路径）。
    """
    # 依赖在函数内导入：单测导入本模块时不触发模型加载，也让
    # monkeypatch("app.generation.llm_router.get_llm") 这类断点仍然有效
    from app.generation.llm_router import get_llm
    from app.generation.profiles import SIDE_PUBLIC
    from app.recommend import attach, fee_log
    from app.recommend.fee_search import search

    llm = get_llm(SIDE_PUBLIC)
    # 留痕表先建（幂等）：attach 的 guarded_log 会把留痕失败**吞掉**，表不存在
    # 时不会有任何红灯，只会让 AC-21 的证据链从第一行起就空 —— 那比报错更糟
    fee_log.ensure_table(conn)
    # 检索查询串用「案由 or 原问句」：案由认不出（cause 为 None，attach 会把它换成
    # 空串）按设计 §七 是**正常路径**且「费用照常走检索」，而空串发到编码器只会抛
    # 「sparse 转换后为空」——把它降级成「暂无口径依据」等于把「没查」说成「没有」，
    # 正是 ③a 的红线。question 就在闭包作用域里，回退不必改 attach / estimate 的接口
    scorer = reranker.predict if reranker is not None else None
    return lambda question: attach.build(
        question,
        search_fn=lambda cause: search(client, encoder, cause or question,
                                       scorer=scorer),
        generate_fn=lambda snippet, cause: generate_fee(llm, snippet, cause),
        log_fn=fee_log.make_logger(conn))
