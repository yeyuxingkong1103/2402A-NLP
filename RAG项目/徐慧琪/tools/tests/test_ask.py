# CLI 的参数解析与结果渲染是纯逻辑，可离线断言；
# 真正跑问答由本任务的 Step 5 人工执行。
# ③b-1 的装配用例（build_extras_fn）随实现搬到了 backend/tests/test_extras.py：
# 被测代码在生产包 app/recommend/ 里，用例跟着走，这里只留 CLI 自己的东西。
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ask import SIDES, build_parser, format_extras, format_result


def test_sides_are_the_two_channels():
    assert SIDES == ("internal", "public")


def test_parser_defaults_to_internal_side():
    args = build_parser().parse_args(["第五百八十四条"])
    assert args.side == "internal"
    assert args.question == "第五百八十四条"


def test_parser_accepts_public_side():
    assert build_parser().parse_args(["-s", "public", "押金不退"]).side == "public"


def test_parser_rejects_unknown_side():
    import pytest
    with pytest.raises(SystemExit):
        build_parser().parse_args(["-s", "third", "问题"])


def _result(**kwargs):
    from app.generation.answer import QAResult
    base = {"status": "ok", "answer": "应当赔偿。", "citations": [], "disclaimer": "",
            "sources": [], "attempts": 1, "failures": []}
    base.update(kwargs)
    return QAResult(**base)


def test_format_shows_status_attempts_and_answer():
    text = format_result(_result())
    assert "ok" in text
    assert "应当赔偿。" in text
    assert "1" in text


def test_format_lists_sources_with_article_numbers():
    text = format_result(_result(sources=[
        {"article_no": 584, "path": "第三编 合同", "text": "第五百八十四条 ……",
         "source": "exact", "rerank_score": None}]))
    assert "584" in text
    assert "第三编 合同" in text
    # 上面两条只查"子串在不在"，把条号与路径对调（`第{path}条（{article_no}）`）
    # 照样两条都命中——忠实变异实测这条坏法仍全绿。故这里钉住两个字段的**位置**：
    # 条号必须紧跟「第」，路径必须在括号里
    assert "第584条（第三编 合同）" in text


def test_format_shows_failure_reasons_on_abstain():
    # 降级时必须把原因打出来，否则调试只能靠猜
    text = format_result(_result(status="abstain", answer="未找到相关依据。",
                                 failures=["引用失败：quote 不是原第584条的原文摘录"]))
    assert "quote" in text


def test_format_shows_disclaimer_for_public():
    text = format_result(_result(disclaimer="本回答仅供参考，不构成正式法律意见。"))
    assert "仅供参考" in text


# 引用条目：quote 取真实的第584条原文片段，本组只测渲染、不测校验
def _citation(article="584", paragraph=None, item=None):
    from app.generation.schema import Citation
    return Citation(law="中华人民共和国民法典", article=article,
                    paragraph=paragraph, item=item,
                    quote="当事人一方不履行合同义务或者履行合同义务不符合约定")


def test_format_citation_keeps_chinese_article_without_double_wrapping():
    # schema.py 的契约允许「第五百八十四条」这种写法；渲染器若再包一层
    # 「第…条」，就会渲染成「第第五百八十四条条」（Task 13 冒烟 M1 的原始标本）
    text = format_result(_result(citations=[_citation(article="第五百八十四条")]))
    assert "第五百八十四条：" in text
    assert "第第五百八十四条条" not in text
    # 漏写尾字的半截写法（「第584」）同样不许被包成「第第584条」
    half = format_result(_result(citations=[_citation(article="第584")]))
    assert "第584条：" in half
    assert "第第584条" not in half


def test_format_citation_wraps_bare_number_and_omits_empty_parts():
    # 裸数字写法要补上「第…条」；款/项为空时不得留下「款 项」空占位
    text = format_result(_result(citations=[_citation(article="584", paragraph=1)]))
    assert "第584条第1款：" in text
    bare = format_result(_result(citations=[_citation(article="584", item="一")]))
    assert "第584条第一项：" in bare
    assert "款" not in bare, "未引用款时不该出现「款」字空占位"


# ---- ③b-1：附加区块渲染 ----
# 区块形状由 test_attach.py 钉住；这里只钉「每种 fee 状态各印成什么」。
BLOCK = {"cause": "房屋租赁合同纠纷", "field": "房产与建设工程",
         "lawyers": [{"name": "张文", "org": "明理律师事务所",
                      "field": "房产与建设工程",
                      "contact_hint": "通过平台留言获取", "demo": True,
                      "note": "示例数据"}],
         "fee": {"status": "ok", "low": 1, "high": 10, "unit": None,
                 "source_doc": "诉讼费用交纳办法", "source_no": "第十三条"},
         "disclaimer": "参考区间，不构成报价或委托"}


def test_format_extras_renders_lawyer_and_fee():
    # 逐字比对整行而不是查子串：只查「1」「10」的话，把 low/high 印反、或把
    # 依据的文档名与条号对调都照样命中（Task 4 的同款教训）
    text = format_extras(BLOCK)
    assert "  - 张文（示例数据）｜明理律师事务所｜房产与建设工程｜通过平台留言获取" in text
    assert "【费用区间】1 ~ 10（依据：诉讼费用交纳办法 第十三条）" in text
    assert "参考区间，不构成报价或委托" in text


def test_format_extras_renders_unit_when_given():
    """语料写「1万元」而结果只印「1 ~ 10」= 静默错 4 个数量级（Task 4 复审的红线）。"""
    text = format_extras(dict(BLOCK, fee=dict(BLOCK["fee"], unit="万元")))
    assert "【费用区间】1 ~ 10万元（依据：诉讼费用交纳办法 第十三条）" in text


def test_format_extras_omits_unit_without_printing_none():
    # unit 缺省不印；尤其不能把 None 印出来（f-string 的静默默认）
    text = format_extras(BLOCK)
    assert "【费用区间】1 ~ 10（依据：诉讼费用交纳办法 第十三条）" in text
    assert "None" not in text


def test_format_extras_prints_charge_basis_on_the_fee_line():
    """真跑实证 N10：语料「四、计时收费标准：1000元—8000元/有效工作小时。」被渲染成
    「1000 ~ 8000元」，用户读成一次性收费。用户 2026-09-29 裁决：计价基础附在费用行上
    （不读依据原文也能看懂计价方式）。整行逐字比对，防「印到别的行」或「漏了斜杠」。"""
    fee = dict(BLOCK["fee"], low=1000, high=8000, unit="元",
               charge_basis="有效工作小时")
    text = format_extras(dict(BLOCK, fee=fee))
    assert ("【费用区间】1000 ~ 8000元/有效工作小时"
            "（依据：诉讼费用交纳办法 第十三条）") in text


def test_format_extras_normalizes_a_slashed_charge_basis():
    """模型若把片段里的「/有效工作小时」连斜杠一起抄回来，不得印成「//」——提示词要的
    是词本身，但模型未必照做，渲染层去一次前缀比事后改留痕便宜。"""
    fee = dict(BLOCK["fee"], charge_basis="/小时")
    assert "【费用区间】1 ~ 10/小时（依据" in format_extras(dict(BLOCK, fee=fee))


def test_format_extras_omits_charge_basis_without_printing_none():
    """缺省（None / 空串 / 缺键）不印，且老形状的 fee 字典（本批之前产出的、
    没有 charge_basis 键）照样渲染 —— 渲染层用 .get 读它正是为了这个。"""
    assert "【费用区间】1 ~ 10（依据" in format_extras(BLOCK)
    for empty in ({"charge_basis": ""}, {"charge_basis": None}, {"charge_basis": "/"}):
        text = format_extras(dict(BLOCK, fee={**BLOCK["fee"], **empty}))
        assert "【费用区间】1 ~ 10（依据" in text and "None" not in text


def test_format_extras_prints_cause_label_only_when_known():
    # 案由标签是本子项目三块交付之一，两个方向都要钉：有案由必须印（若只查
    # 「没印 None」，删掉渲染里那一行照样绿）；认不出（tag 为 None）时整行省略
    # —— 省掉这行是设计（见 ask.py 注释），印「None」才是故障
    assert "【案由】房屋租赁合同纠纷" in format_extras(BLOCK)
    blank = format_extras(dict(BLOCK, cause=None))
    assert "【案由】" not in blank and "None" not in blank


def test_format_extras_prints_lawyer_heading_only_when_cards_exist():
    # 标题与卡片同进同退：有卡片不带标题会让「示例数据」的三张卡看不出是演示数据
    assert "【示例律师】（演示数据）" in format_extras(BLOCK)
    assert "【示例律师】" not in format_extras(dict(BLOCK, lawyers=[]))


def test_format_extras_tolerates_card_without_note():
    # DEMO_MODE 关掉后卡片是 5 键（无 note）：card['note'] 会当场 KeyError，
    # 只有 card.get("note") 兼容两种形状
    card = {k: v for k, v in BLOCK["lawyers"][0].items() if k != "note"}
    text = format_extras(dict(BLOCK, lawyers=[card]))
    assert "张文｜明理律师事务所" in text and "（None）" not in text


def test_format_extras_omits_basis_bracket_when_both_parts_missing():
    # fees.py 的 snippet.get 不保证依据字段非空：照原样印会得到「（依据：None None）」
    text = format_extras(dict(BLOCK, fee=dict(BLOCK["fee"], source_doc=None, source_no=None)))
    assert "【费用区间】1 ~ 10" in text
    assert "依据" not in text and "None" not in text


def test_format_extras_keeps_a_lone_basis_part():
    # 只有一个时不整段丢掉：单靠文档名也比什么都不给强（且不得印出空位的 None）
    text = format_extras(dict(BLOCK, fee=dict(BLOCK["fee"], source_no=None)))
    assert "【费用区间】1 ~ 10（依据：诉讼费用交纳办法）" in text


# ---- 依据原文（用户 2026-09-29 裁决：渲染要印命中片段原文）----
# 存在的理由：只印「数字 + 文件名」时用户无从核对，真链路测得的两处误导
# （「1000 ~ 8000元」丢了「/有效工作小时」、风险代理的 5~15% 回款提成被读成
# 律师费比例）都是原文一摆出来就能看穿的。下面三条按「印什么 / 不印什么 /
# 多行怎么印」分开钉，夹具单列而不加进 BLOCK：BLOCK 保持「没有 basis 键」的
# 退化形状，正好也是「basis 为空」那条用例的标本。
def test_format_extras_prints_basis_verbatim_after_fee_line():
    # 逐字整段比对（含两格缩进与标头）：只查片段里的某个词在不在，把片段截掉
    # 九成、或把原文插到费用行**之前**，都照样绿
    snippet = "四、计时收费标准：1000元—8000元/有效工作小时。"
    text = format_extras(dict(BLOCK, fee=dict(BLOCK["fee"], basis=snippet)))
    assert ("【费用区间】1 ~ 10（依据：诉讼费用交纳办法 第十三条）"
            f"\n【依据原文】\n  {snippet}") in text


def test_format_extras_keeps_multiline_basis_readable():
    # 片段含换行（语料按语义单元分段，最长 821 字）：逐行缩进印，最后一行也必须
    # 在 —— 截到首行、或只缩进第一行，是最容易「看起来正常」的两种坏法
    snippet = ("二、 计时收费\n1． 普通律师 800~1000 元/小时\n"
               "3． 资深律师/合伙人 3000~8000 元/小时")
    text = format_extras(dict(BLOCK, fee=dict(BLOCK["fee"], basis=snippet)))
    assert ("【依据原文】\n  二、 计时收费\n"
            "  1． 普通律师 800~1000 元/小时\n"
            "  3． 资深律师/合伙人 3000~8000 元/小时") in text


def test_format_extras_omits_basis_block_when_empty():
    # basis 可能为空（设计允许：estimate 的 ok 支取 snippet["text"]，不保证非空）：
    # 空串 / None / 缺键三种都不印标头（印一个空标头等于说「有依据但看不见」）
    for empty in ({"basis": ""}, {"basis": None}, {}):
        text = format_extras(dict(BLOCK, fee={**BLOCK["fee"], **empty}))
        assert "【依据原文】" not in text
    assert "【依据原文】" not in format_extras(BLOCK)


def test_format_extras_renders_no_corpus_wording():
    text = format_extras(dict(BLOCK, fee={"status": "no_corpus"}))
    assert "【费用区间】暂无费用口径依据" in text
    # 免责行在 format_extras 里是无条件打印的，非 ok 三支也照样要印（AC-21：
    # 展示费用区间时 100% 带提示，而用户读到的是渲染层）。只在 ok 用例里断言时，
    # 把免责改为「仅 status == ok 时打印」三条非 ok 状态全绿、免责全部消失
    assert "【提示】参考区间，不构成报价或委托" in text


def test_format_extras_renders_rejected_wording():
    text = format_extras(dict(BLOCK, fee={"status": "rejected", "reason": "无法指回片段"}))
    assert "不猜数" in text and "暂无费用口径依据" not in text
    assert "【提示】参考区间，不构成报价或委托" in text  # 同上：判拒不等于可以不给提示


def test_format_extras_says_the_efficacy_rejection_in_its_own_words():
    """效力判拒（设计 §七「不采用该片段，报告里说明」）不能与回查判拒共用一句话：
    一律印「生成结果无法指回收费口径」会把「口径过期、我们没用它」错说成「模型编了
    数」——用户据此对收费文件的时效得出相反结论。分支按 estimate 写的 reason 走，
    原文仍不印（内部措辞，同 unavailable 不印异常的安全口径）。"""
    stale = format_extras(dict(BLOCK, fee={
        "status": "rejected", "reason": "命中片段效力非现行有效，已按不采用处理"}))
    assert "效力非现行有效" in stale and "无法指回收费口径" not in stale
    # 反方向：回查判拒仍是原来那句（分流不得过度，否则两种判拒的区分就没了）
    other = format_extras(dict(BLOCK, fee={
        "status": "rejected", "reason": "生成区间无法指回命中片段，已按不猜数处理"}))
    assert "无法指回收费口径" in other and "效力" not in other


def test_format_extras_renders_unavailable_distinctly():
    """故障与「没有依据」必须给不同文案 —— 用户看到的意思不一样。"""
    block = dict(BLOCK, fee={"status": "unavailable", "reason": "费用信息暂时不可用：milvus down"})
    text = format_extras(block)
    assert "暂时不可用" in text
    assert "暂无费用口径依据" not in text
    # reason 里的原始异常是给开发者排查的：印给公众既误导又泄露内部实现
    assert "milvus down" not in text
    assert "【提示】参考区间，不构成报价或委托" in text  # 故障支同样不许吞掉免责


def test_format_extras_empty_for_none_block():
    # 律师侧、以及公众侧那类「与法律无关」的出口（out_of_scope）拿到的就是 None：
    # 渲染必须静默退化成空串（解耦后 need_more_info/abstain/error 都有区块了，
    # 但 None 仍是一条合法输入）
    assert format_extras(None) == ""


def test_format_result_appends_extras_only_when_present():
    # QAResult 带着区块时 CLI 要真印出来；extras 默认 None → 既有输出一字不变
    text = format_result(_result(extras=BLOCK))
    assert "【专业领域】房产与建设工程" in text and "【费用区间】1 ~ 10" in text
    assert "【" not in format_result(_result()), "没有区块时不得多出任何一行"
