"""费用区间：数值回查 + 生成编排（编排部分见本文件后半）。

技术方案 6.4 的两条硬规矩在这里落地：
  ①区间**不得脱离命中片段编造** —— 由 verify_range 保证「数字有字面出处」；
  ②命中不到**不猜数** —— 由编排层的 no_corpus 分支保证。
外加设计 §七 的一行（原实现漏掉）：命中片段效力非现行 → **不采用该片段**，
不采用的理由不给用户编数（判 rejected，与回查不过同一个出口）。
回查是**必要不充分**条件：它挡的是「模型报了个片段里没有的数」，
挡不住「模型把片段里的数用错了地方」——后者靠提示词约束与人工冒烟。
"""
from __future__ import annotations

import re

# 只认阿拉伯数字。中文数字（「百分之八」）不抽 —— 语料与模型输出都约定用
# 阿拉伯数字写法，宁可拒掉也不做易错的等价换算。
# 千分位（1,000 / 1，000）必须在这里**当成一个数**吃掉：若只匹配 `\d+`，它的子串
# 「1」「000」会各自成为合法出处 —— 片段里唯一的金额 1000 就能放行
# verify_range(1, 1)，正是安全阀要挡的误放行方向。逗号只在**夹在数字之间且后随
# 恰好三位**时才算数的一部分，故不会把「每件50元，共3件」粘成一个数（「，」在中文
# 行文里是分句符，只有这个严格形态才当分隔符；data/ 实测 0 例千分位，此处是为
# fee_corpus 预置口径）
_NUMBER = re.compile(r"\d{1,3}(?:[,，]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    """抽出文本里的阿拉伯数字，**并归一化成与上报值同一套比对字面量**。

    归一化必须做在这一层，而不是只做在上报值一侧：两边用同一把尺子比，片段写
    「2.50」「50.0」时模型逐字照抄才能过。只归一化一侧会让**接受方向静默退化成
    永远拒绝** —— 安全阀「有出处必须过」的那半边就没了。
    """
    if not text:
        return set()
    # 千分位分隔符要先剥掉：float("1,000") 会抛 ValueError，不能直接喂给 _literal
    return {_literal(float(m.group(0).replace(",", "").replace("，", "")))
            for m in _NUMBER.finditer(text)}


def _literal(value: float) -> str:
    """把数值写成比对用的字面量：8.0 与 8 视为同一个数。

    上报值与片段两侧都走这个函数，比对才是对称的（单侧归一化是已修过的坏法）。
    """
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def verify_range(low: float | None, high: float | None, snippet: str) -> bool:
    """区间的两个端点是否都能在片段里找到字面出处。

    缺失端点、倒置区间、空片段一律判否 —— 宁可输出「暂无费用口径依据」，
    也不输出一个无法指回片段的数字（技术方案 6.4 硬规矩①）。
    倒置检查放在分词之前：它是与片段无关的形状判定（`10 > 1` 本身就是无效区间），
    越早拒越不必分词，也免得下游读到一个 low/high 反了的「区间」。
    """
    if low is None or high is None:
        return False
    if low > high:
        return False
    pool = numbers_in(snippet)
    # 下面这两行是**提前返回，不是护栏**：snippet 为空时 `in set()` 本来就为 False，
    # 删掉它们结论不变（实测 15 条用例照样全绿）——留在这里只为把「空片段一律判否」
    # 这个意图写明白，免得下一位读者以为它在防守什么。真正拦住空片段的是末行的 `and`
    if not pool:
        return False
    return _literal(low) in pool and _literal(high) in pool


# 可采用的片段效力。与 tools/ingest_fee.STATUS_VALID 是同一个字面量，但**有意
# 不互相 import**：tools 依赖 backend，backend 的 recommend 层只做纯判定（本文件
# 只 import re 就是这条线的证据），反向依赖会把 pymilvus/编码器拖进这一层。法条侧
# 同样是各层自持一份（cite_verify 的 `article["status"] != "现行有效"`、milvus 的
# build_filter_expr 默认值）—— 判据是「非现行一律不采用」，多一处字面量不改变语义
STATUS_VALID = "现行有效"

# AC-21 定稿原话，渲染方直接取用，不得改写
FEE_DISCLAIMER = "参考区间，不构成报价或委托"


def _as_number(value):
    """把生成器上报的端点折算成数值；折不动的返回 None，交由回查判拒。

    为什么折算必须做在编排层：回查的契约只收「数值 | None」（传字符串时
    `low > high` 会抛 TypeError），而生成器的上游是模型 —— JSON 里
    "low": "1" 这种字符串数字很常见。不折算就有两个反向恶果：混合类型
    （1 与 "10"）把**降级**抛成异常，被调用方记成「服务不可用」（设计 §七
    要区分的正是故障与降级）；纯字符串端点（"1" 与 "10"）则走字典序比较并
    逐字命中，静默放行成 ok，让字符串灌进结果与留痕。
    布尔值单独排除：JSON 的 true/false 会折算成 1/0，让「是/否」冒充金额。
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        # 超范围的大整数（如 10**400）在回查折算字面量时 float() 会抛
        # OverflowError —— 让它逃出去又是「降级冒充故障」，正是折算要避免的方向
        try:
            float(value)
        except OverflowError:
            return None
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# unit 的邻近窗口宽度（字符）：单位必须跟在**上报数字**之后、同一行、这个间隔内。
# 取值依据见 _unit_near_number 的 docstring（语料实测倒推）
_UNIT_WINDOW = 6


def _unit_near_number(unit: str, value: float, snippet: str) -> bool:
    """unit 是否出现在**这个数**的同一行邻近窗口内（而不是片段里任何地方）。

    为什么必须收这个口（终审复核实测）：片段「标的额 100 万元以上的，每件收费
    50 元」+ 上报 (100, 100, unit="元") —— 回查认「100」（出自「100 万元」），
    单位这边认「50 元」里那个独立的「元」，两关各自都说得通，合起来却是拿**另一个
    数的单位**给 100 盖章，渲染成「100 ~ 100元」，与语料差 10000 倍。纯存在性
    检查看不出单位属于谁，所以判定要缩到「这个数的邻域里」。

    窗口取 6 个字符是实测倒推：语料 300 个数字词元里 180 个与单位的间隔 ≤4；
    间隔 6 的那几个都是「单位只写在区间末端」的形态（「800~1000 元/小时」的 800，
    另一个端点 1000 的间隔是 1）；而本条要挡的那句间隔是 14。只允许数字**之后**：
    中文行文里单位写在数字之前的形态（「每万元收 100」）在语料里 0 例；跨行不算
    （与 _magnitude_attached 同口径：换行是排版切出来的，不是紧邻）。
    判定按「存在一处邻近」而不是「处处邻近」——与 _unit_has_source 同口径。
    """
    literal = _literal(value)
    for match in _NUMBER.finditer(snippet):
        token = _literal(float(match.group(0).replace(",", "").replace("，", "")))
        if token != literal:
            continue
        # 窗口切片里跑的是同一个 _unit_has_source：切片头部的字符恰好是那个数字，
        # 单位若在切片起点（「50元」的元）则无前置字符、直接算独立出现；若在
        # 「万元」里（前一个字符是「万」）则被 lookbehind 挡住 —— 与全片段口径同源
        tail = re.split(r"\r?\n", snippet[match.end():], maxsplit=1)[0]
        if _unit_has_source(unit, tail[:_UNIT_WINDOW + len(unit)]):
            return True
    return False


def _unit_has_source(unit: str, snippet: str) -> bool:
    """单位是否作为**独立的单位词**出现在片段里（存在性检查，不比对数值）。

    为什么不能用 `unit in snippet`：「元」是「万元」的后半截，子串匹配会判它
    有出处 —— 模型报「元」而片段写的是「万元」时照样放行，恰好让 unit 字段在
    它存在的主要理由（量级不得静默错档）上失效。故要求**至少有一处**出现，
    其前一个字符不是汉字（量级词万/千/百/亿都是汉字，而独立单位紧跟数字或
    出现在行首）：`1万元` 里的「万元」过（前是 1）、`50元` 里的「元」过
    （前是 0）、`万元` 里的「元」拒（前是 万）。判定按「存在一处独立出现」，
    不要求处处独立 —— `50元；超过1万元` 里「元」的坏出现不该牵连那处好出现。
    代价（有意接受）：片段用行文提到单位（「费用以元计」）而数字在别处给出时，
    该出现判无出处 → 拒。方向是 fail-closed，且数字本身仍受回查约束。
    """
    return re.search(r"(?<![一-鿿])" + re.escape(unit), snippet) is not None


def _charge_basis_has_source(charge_basis: str, snippet: str) -> bool:
    """计价基础是否在片段里出现过 —— **宽松存在性**，与 unit 的严格 lookbehind 有意不同。

    为什么可以和 unit 松：两者承载的东西不同。`unit` 承载**量级**（「元」是「万元」
    的后半截，子串匹配会让「报元、片段写万元」静默错 4 个数量级），所以它必须要求
    独立出现。计价基础只承载**口径**（这笔钱按什么算），不含任何量级信息，模型报
    「小时」而片段写「有效工作小时」、报「件」而片段写「每件」，都是**合法出处** ——
    用严格匹配反而会把正确上报拒掉（复核实测：`_unit_has_source("小时", "…元/有效
    工作小时。")` 返回 False）。故这里用最宽的存在性判据，只拦「片段里根本没提过
    这个词」的纯编造。

    代价（有意接受）：模型报一个与片段同义但不同字面的说法（「小时」vs「工时」）会被
    判拒 —— 拒的方向 fail-closed，用户看到的是「暂无依据」而不是一个读不出计价方式
    的数字。
    """
    return charge_basis in snippet


# 量级词：紧跟数字出现时会改变金额档位的中文数量级单位（「1 万元」≠「1 元」）。
# 缺单位时靠它判「这个数在片段里是带量级的」—— 见 estimate 的缺省闸门
_MAGNITUDE = "万千百亿"

# 数字之后（允许同行空白）紧跟量级词。语料里数字与「万」之间既有无空格的写法，
# 也有被 PDF 文本层塞进空格的写法，两种都是紧邻；换行不算（见 _magnitude_attached
# 的 docstring）。空白类写成 \t 与 \u3000 的转义形式（半角制表符、全角空格；
# 半角空格在中间）：全角空格与半角空格在源码里几乎看不出差别，字面写法就是
# 一处「读起来对、实为另一个字符」的隐患
_MAGNITUDE_AFTER = re.compile(rf"[ \t\u3000]*[{_MAGNITUDE}]")


def _magnitude_attached(value: float, snippet: str) -> bool:
    """这个数在片段里是否**紧跟量级词**（万/千/百/亿）：任一处出现即算。

    查的是数字**词元**（_NUMBER 的匹配）而不是字面串：片段「1,000 万元」归一化
    后的字面量是「1000」，直接搜「1000」加量级词搜不到 —— 分词与归一化必须和
    numbers_in 用同一把尺子（两边不同源的教训见 numbers_in 的 docstring）。
    数字与量级词之间的空白要跳过：真实语料里「数+万」共 95 处，其中带空格 76 处、
    无空格 19 处（2026-09-29 复算 data/raw/fee；第一批的代码注释与报告写的
    「79 处 + 另 19 处」是错的，此处以复算为准），不跳过就等于闸门在语料的主体
    形态（76/95 是带空格的那种）上失效。
    只跳**同一行**的空白：跨行的「1\\n万」是排版切出来的，不是紧邻，判它有量级
    同样是静默错档。判定取「存在一处紧邻」而不是「处处紧邻」——与 _unit_has_source
    同口径，任一处成立就说明这个数可能就是那一处的那个数，fail-closed 才拦得住。
    """
    literal = _literal(value)
    for match in _NUMBER.finditer(snippet):
        token = _literal(float(match.group(0).replace(",", "").replace("，", "")))
        if token == literal and _MAGNITUDE_AFTER.match(snippet[match.end():]):
            return True
    return False


def _rejected(reason: str) -> dict:
    """判拒的统一出口（效力闸门与区间/单位闸门共用）。

    为什么要收成一处：rejected 现在有两个入口，各写一份 8 键字典就会有两份
    形状，漏一个 unit 键就是渲染方在**判决这一支**上的 KeyError（形状恒定的
    理由见 attach._unavailable 的同款注释）。basis/source_doc/source_no 一律
    None：「依据」指被采用的片段，判拒的区间没有依据。
    """
    return {"status": "rejected", "low": None, "high": None, "unit": None,
            "charge_basis": None, "basis": None, "source_doc": None,
            "source_no": None, "reason": reason}


def estimate(cause: str, *, search_fn, generate_fn, log_fn) -> dict:
    """在 fee_corpus 上取一段收费口径，据它产出区间。

    三个依赖全部注入：search_fn(cause) -> list[dict]、generate_fn(snippet, cause)
    -> {"low", "high", "unit"?, "charge_basis"?}、log_fn(**fields) 留痕（AC-21 证据）。
    hit 的字段契约：text 必给；source_doc / source_no 可缺（只进「依据」）；
    status 必给且必须是现行 —— 缺失同样判拒，见下面的效力闸门。
    `unit` 是模型的**可选**上报（照抄片段里的单位，如「万元」）；不给就只印
    数字，给了就必须能在片段里找到出处，否则与编造数字一样判拒。
    `charge_basis` 同为可选上报（计价基础，如「小时」「件」），判据同样「有值
    必须有出处、缺省不判拒」，但存在性是宽松匹配（见 _charge_basis_has_source）。
    **缺省不是无条件的**：命中数字紧邻量级词（万/千/百/亿）而 unit 缺失时也判拒
    —— 片段写「1 万元」而结果印「1」是静默错 4 个数量级，正是 unit 字段存在的理由。

    **故障一律向上传播**：检索或模型不可用时不返回任何状态，由调用方
    区分「服务不可用」与「语料里确实没有」——两者给用户的含义完全不同。
    故本函数内**没有任何 try/except**，这是有意的：吞掉异常就会把故障
    降级成 no_corpus，而那条红线是 ③a 的既定语义。
    """
    hits = search_fn(cause)
    if not hits:
        # 未命中是设计行为而非降级，故连模型都不问（不猜数的第一步是不去猜）；
        # 这一支也不写留痕：没有片段可追溯，留痕表里没有它的位置
        return {"status": "no_corpus", "low": None, "high": None, "unit": None,
                "charge_basis": None, "basis": None, "source_doc": None,
                "source_no": None, "reason": "语料未收录该案由的收费口径"}
    snippet = hits[0]
    # 设计 §七「命中片段的 status 非现行 / 已过期 → 不采用该片段」在编排层的兜底：
    # 检索侧已带 status 过滤（主防线，tools/ingest_fee.search 的 expr），但 search_fn
    # 是可注入的，编排层不假定检索方一定过滤了。判据与法条侧同形（cite_verify：
    # `article["status"] != "现行有效"`）：**缺失与空串同样判拒** —— 证明不了现行
    # 就不等于现行，放行它的方向正是「静默引用过期价位」。判在生成之前：一段不会
    # 被采用的语料不值得再问一次模型（与 no_corpus 不问模型同口径）
    if snippet.get("status") != STATUS_VALID:
        log_fn(cause=cause, snippet_id=snippet.get("source_no"), range_low=None,
               range_high=None, model=None, request_id=None, status="rejected")
        return _rejected("命中片段效力非现行有效，已按不采用处理")
    produced = generate_fn(snippet["text"], cause)
    low, high = _as_number(produced.get("low")), _as_number(produced.get("high"))
    raw_unit = produced.get("unit")
    unit = (raw_unit.strip() or None) if isinstance(raw_unit, str) else None
    # 单位与数字受**同一道闸门**：编造单位与编造数字同样危险 —— 片段写「1万元」
    # 而结果只印「1」是静默错 4 个数量级。这里是**存在性**检查而非数值回查，
    # 且要求两件事：独立出现（「元」不算「万元」的后半截，见 _unit_has_source）、
    # 且出现在**上报数字自己的**邻近窗口里（见 _unit_near_number：光在片段里出现
    # 过还不够，那可能属于另一个数）。端点缺一个时不做邻近判定 —— 那种情形本来就
    # 由 range_bad 判拒，reason 也该说区间，不该替它背「单位无法指回」的锅
    has_bounds = low is not None and high is not None
    unit_near = has_bounds and unit is not None and (
        _unit_near_number(unit, low, snippet["text"])
        or _unit_near_number(unit, high, snippet["text"]))
    unit_bad = (unit is not None and has_bounds and not unit_near) or (
        raw_unit is not None and not isinstance(raw_unit, str))
    # 计价基础（用户 2026-09-29 裁决，满足 FR-9.4「含计费方式说明」）：也是**可选**
    # 上报，与 unit 同走「报了就必须有出处、缺省不判拒」。真跑实证 N10：语料的
    # 「1000元—8000元/有效工作小时」被渲染成「1000 ~ 8000元」，用户会读成一次性
    # 收费 —— 丢掉的就是这个字段。存在性检查是**宽松**的（见 _charge_basis_has_source：
    # 它不承载量级，用 unit 的严格 lookbehind 会把合法上报一起拒掉）
    raw_charge = produced.get("charge_basis")
    charge_basis = (raw_charge.strip() or None) if isinstance(raw_charge, str) else None
    charge_bad = (charge_basis is not None
                  and not _charge_basis_has_source(charge_basis, snippet["text"])) or (
        raw_charge is not None and not isinstance(raw_charge, str))
    # 缺省闸门（fail-closed 的那半边）：unit 不给**不等于**可以省掉量级。命中数字
    # 在片段里紧邻量级词时，片段只证明了「1 万」这个数，没证明「1」；渲染成
    # 「1 ~ 50」与语料差 10000 倍，且没有任何字段能事后看出错档（unit 为 None 与
    # 合法缺省在结果里长得一样）。判在回查之后（`not range_bad` 短路）：只有通过了
    # 回查的数才有资格谈量级，且 10**400 那类折不动字面量的值已在 _as_number 被挡
    range_bad = not verify_range(low, high, snippet["text"])
    magnitude_bad = (unit is None and not range_bad and (
        _magnitude_attached(low, snippet["text"])
        or _magnitude_attached(high, snippet["text"])))
    common = {"cause": cause, "snippet_id": snippet.get("source_no"),
              "model": produced.get("model"), "request_id": produced.get("request_id")}
    if unit_bad or charge_bad or magnitude_bad or range_bad:
        # 判拒的来由各写各的 reason：留痕表里既没有 unit 列也没有效力列，事后只
        # 能靠这行文字分辨堵在哪一关（效力 / 区间 / 单位出处 / 计价基础出处 /
        # 量级缺省）。出口与形状仍是同一个 rejected（见 _rejected）
        reason = ("生成单位无法指回命中片段，已按不猜数处理" if unit_bad else
                  "生成计价基础无法指回命中片段，已按不猜数处理" if charge_bad else
                  "命中数字紧邻量级词而模型未上报单位，已按不猜数处理" if magnitude_bad
                  else "生成区间无法指回命中片段，已按不猜数处理")
        # 被拒的区间也要留痕：AC-21 要的是可追溯，包括「拒绝过什么」。
        # 片段本身仍可由 snippet_id 追溯（结果里不带依据字段，理由见 _rejected）
        log_fn(**common, range_low=low, range_high=high, status="rejected")
        return _rejected(reason)
    log_fn(**common, range_low=low, range_high=high, status="ok")
    return {"status": "ok", "low": low, "high": high, "unit": unit,
            "charge_basis": charge_basis, "basis": snippet["text"],
            "source_doc": snippet.get("source_doc"),
            "source_no": snippet.get("source_no"), "reason": None}
